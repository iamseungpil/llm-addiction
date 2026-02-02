#!/usr/bin/env python3
"""
Phase 1: SAE Feature Extraction

Extract SAE feature activations from each game's decision point.
Supports both Base SAE and Boosted SAE (Base + Residual from Phase 0).

Input:
    - Experiment data JSON (3,200 games)
    - (Optional) Phase 0: residual_sae_layer_{L}.pt

Output:
    - phase1_features/layer_{L}_features.npz for each layer
        - features: [n_games, n_sae_features]
        - outcomes: [n_games]
        - game_ids: [n_games]
        - hidden_states: [n_games, d_model] (optional)

Usage:
    # Base SAE only
    python phase1_feature_extraction.py --gpu 0

    # Boosted SAE (requires Phase 0)
    python phase1_feature_extraction.py --gpu 0 --use-boost

    # Specific layers
    python phase1_feature_extraction.py --gpu 0 --use-boost --layers 25,30,35
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent))

from config import load_config, get_phase_output_dir
from utils import (
    GemmaBaseModel, GemmaSAE,
    load_experiment_data, reconstruct_decision_prompt,
    clear_gpu_memory, logger
)


class BoostedSAE:
    """
    Boosted SAE: Base SAE + Residual SAE for improved reconstruction.

    Final features = Base features + corrections from Residual SAE
    """

    def __init__(
        self,
        base_sae: GemmaSAE,
        residual_sae_dir: Path,
        device: str = 'cuda:0'
    ):
        self.base_sae = base_sae
        self.residual_sae_dir = residual_sae_dir
        self.device = device
        self.residual_saes = {}  # layer -> ResidualSAE model

    def load_residual_sae(self, layer: int):
        """Load Residual SAE for a specific layer."""
        if layer in self.residual_saes:
            return True

        model_path = self.residual_sae_dir / f"residual_sae_layer_{layer}.pt"
        if not model_path.exists():
            logger.warning(f"Residual SAE not found for layer {layer}: {model_path}")
            return False

        # Import ResidualSAE class
        from phase0_sae_boost import ResidualSAE

        checkpoint = torch.load(model_path, map_location=self.device)
        model = ResidualSAE(
            d_model=checkpoint['d_model'],
            d_sae=checkpoint['d_sae']
        ).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        self.residual_saes[layer] = model
        logger.info(f"  Loaded Residual SAE for layer {layer}")
        return True

    def encode(self, hidden_state: torch.Tensor, layer: int) -> Optional[torch.Tensor]:
        """
        Encode hidden state using Boosted SAE.

        The Boosted approach:
        1. Get base features from Base SAE
        2. Compute residual correction from Residual SAE
        3. Return base features (residual improves reconstruction, not features)

        Note: For feature extraction, we use base features.
        The Residual SAE improves reconstruction fidelity but doesn't add new features.
        """
        # Get base features
        base_features = self.base_sae.encode(hidden_state, layer)
        if base_features is None:
            return None

        return base_features

    def encode_with_boost_info(
        self,
        hidden_state: torch.Tensor,
        layer: int
    ) -> Optional[Dict]:
        """
        Encode with additional boost information.

        Returns base features plus reconstruction quality metrics.
        """
        # Load residual SAE if needed
        has_residual = self.load_residual_sae(layer)

        # Get base features and reconstruction
        base_features = self.base_sae.encode(hidden_state, layer)
        if base_features is None:
            return None

        base_recon = self.base_sae.decode(base_features, layer)
        if base_recon is None:
            return {'features': base_features, 'boosted': False}

        # Compute base reconstruction error
        base_error = torch.norm(hidden_state - base_recon).item()

        result = {
            'features': base_features,
            'base_recon_error': base_error,
            'boosted': False
        }

        # If residual SAE available, compute boosted reconstruction
        if has_residual and layer in self.residual_saes:
            with torch.no_grad():
                residual_pred = self.residual_saes[layer](hidden_state)['reconstruction']
                boosted_recon = base_recon + residual_pred
                boosted_error = torch.norm(hidden_state - boosted_recon).item()

            result['boosted'] = True
            result['boosted_recon_error'] = boosted_error
            result['error_reduction'] = (base_error - boosted_error) / base_error * 100

        return result


class Phase1FeatureExtraction:
    """Extract SAE features from game decision points."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        gpu_id: int = 0,
        target_layers: Optional[List[int]] = None,
        use_boost: bool = False
    ):
        # Load config
        self.config = load_config(config_path)

        # Setup device
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        self.device = 'cuda:0'

        # Target layers
        if target_layers:
            self.target_layers = target_layers
        else:
            self.target_layers = self.config.get('target_layers', list(range(20, 42)))

        # Boost mode
        self.use_boost = use_boost

        # Output directory
        self.output_dir = get_phase_output_dir(self.config, 1)

        # Phase 0 directory (for boost)
        self.phase0_dir = get_phase_output_dir(self.config, 0)

        # Models (loaded lazily)
        self.model = None
        self.sae = None
        self.boosted_sae = None

    def load_models(self):
        """Load Gemma model and SAE (with optional boost)."""
        logger.info("Loading models...")

        # Load Gemma Base model
        self.model = GemmaBaseModel(device=self.device)
        self.model.load()

        # Load SAE handler
        sae_config = self.config.get('sae', {})
        base_sae = GemmaSAE(
            device=self.device,
            width=sae_config.get('width', '16k')
        )

        if self.use_boost:
            # Check Phase 0 outputs exist
            if not self.phase0_dir.exists():
                logger.warning(f"Phase 0 directory not found: {self.phase0_dir}")
                logger.warning("Falling back to Base SAE only")
                self.sae = base_sae
                self.use_boost = False
            else:
                logger.info("Using Boosted SAE (Base + Residual)")
                self.boosted_sae = BoostedSAE(
                    base_sae=base_sae,
                    residual_sae_dir=self.phase0_dir,
                    device=self.device
                )
                # Pre-load residual SAEs
                for layer in self.target_layers:
                    self.boosted_sae.load_residual_sae(layer)
        else:
            logger.info("Using Base SAE only")
            self.sae = base_sae

    def extract_features_for_game(
        self,
        game: Dict,
        layers: List[int]
    ) -> Dict[int, Dict]:
        """
        Extract features for a single game.

        Returns:
            Dict mapping layer to {features, hidden_state, boost_info}
        """
        # Reconstruct decision prompt
        prompt = reconstruct_decision_prompt(game)

        # Get hidden states from model
        hidden_states = self.model.get_hidden_states(prompt, layers, position='last')

        # Encode through SAE
        results = {}
        for layer in layers:
            if layer not in hidden_states:
                continue

            h = hidden_states[layer]  # [1, d_model]

            if self.use_boost and self.boosted_sae:
                # Use Boosted SAE
                boost_result = self.boosted_sae.encode_with_boost_info(h, layer)
                if boost_result is not None:
                    results[layer] = {
                        'features': boost_result['features'].squeeze(0).cpu().numpy(),
                        'hidden_state': h.squeeze(0).cpu().numpy(),
                        'boosted': boost_result.get('boosted', False),
                        'base_recon_error': boost_result.get('base_recon_error'),
                        'boosted_recon_error': boost_result.get('boosted_recon_error')
                    }
            else:
                # Use Base SAE
                features = self.sae.encode(h, layer)
                if features is not None:
                    results[layer] = {
                        'features': features.squeeze(0).cpu().numpy(),
                        'hidden_state': h.squeeze(0).cpu().numpy(),
                        'boosted': False
                    }

        return results

    def run(self, save_hidden_states: bool = True):
        """
        Run Phase 1 feature extraction.

        Args:
            save_hidden_states: Whether to save raw hidden states
        """
        logger.info("=" * 70)
        logger.info("PHASE 1: SAE FEATURE EXTRACTION")
        logger.info("=" * 70)
        logger.info(f"Target layers: {self.target_layers}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Use Boost: {self.use_boost}")

        # Load models
        self.load_models()

        # Load experiment data
        data_path = self.config.get('data', {}).get('experiment_data')
        exp_data = load_experiment_data(data_path)
        games = exp_data['results']

        logger.info(f"Processing {len(games)} games...")

        # Initialize storage per layer
        layer_data = {
            layer: {
                'features': [],
                'hidden_states': [] if save_hidden_states else None,
                'outcomes': [],
                'game_ids': [],
                'boost_info': []
            }
            for layer in self.target_layers
        }

        # Process each game
        checkpoint_freq = self.config.get('phase1', {}).get('checkpoint_frequency', 100)

        for idx, game in enumerate(tqdm(games, desc="Extracting features")):
            try:
                # Extract features
                game_features = self.extract_features_for_game(game, self.target_layers)

                # Store results
                for layer, data in game_features.items():
                    layer_data[layer]['features'].append(data['features'])
                    layer_data[layer]['outcomes'].append(game['outcome'])
                    layer_data[layer]['game_ids'].append(idx)
                    layer_data[layer]['boost_info'].append({
                        'boosted': data.get('boosted', False),
                        'base_error': data.get('base_recon_error'),
                        'boosted_error': data.get('boosted_recon_error')
                    })

                    if save_hidden_states:
                        layer_data[layer]['hidden_states'].append(data['hidden_state'])

                # Clear GPU memory periodically
                if (idx + 1) % self.config.get('clear_cache_frequency', 50) == 0:
                    clear_gpu_memory()

                # Save checkpoint
                if (idx + 1) % checkpoint_freq == 0:
                    self._save_checkpoint(layer_data, idx + 1)

            except Exception as e:
                logger.error(f"Error processing game {idx}: {e}")
                continue

        # Save final results
        self._save_results(layer_data, save_hidden_states)

        logger.info("Phase 1 complete!")

    def _save_checkpoint(self, layer_data: Dict, n_processed: int):
        """Save checkpoint."""
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        for layer, data in layer_data.items():
            if not data['features']:
                continue

            checkpoint_file = checkpoint_dir / f"layer_{layer}_checkpoint_{n_processed}.npz"
            np.savez_compressed(
                checkpoint_file,
                features=np.array(data['features']),
                outcomes=np.array(data['outcomes']),
                game_ids=np.array(data['game_ids'])
            )

        logger.info(f"  Checkpoint saved at {n_processed} games")

    def _save_results(self, layer_data: Dict, save_hidden_states: bool):
        """Save final results for all layers."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Compute boost statistics
        boost_stats = {}

        for layer, data in layer_data.items():
            if not data['features']:
                logger.warning(f"No features extracted for layer {layer}")
                continue

            output_file = self.output_dir / f"layer_{layer}_features.npz"

            save_dict = {
                'features': np.array(data['features']),
                'outcomes': np.array(data['outcomes']),
                'game_ids': np.array(data['game_ids']),
                'layer': layer,
                'timestamp': timestamp,
                'boosted': self.use_boost
            }

            if save_hidden_states and data['hidden_states']:
                save_dict['hidden_states'] = np.array(data['hidden_states'])

            np.savez_compressed(output_file, **save_dict)

            n_games = len(data['features'])
            n_bankrupt = sum(1 for o in data['outcomes'] if o == 'bankruptcy')

            # Compute boost stats
            if self.use_boost and data['boost_info']:
                boosted_count = sum(1 for b in data['boost_info'] if b.get('boosted'))
                base_errors = [b['base_error'] for b in data['boost_info'] if b.get('base_error')]
                boosted_errors = [b['boosted_error'] for b in data['boost_info'] if b.get('boosted_error')]

                if base_errors and boosted_errors:
                    avg_base = np.mean(base_errors)
                    avg_boosted = np.mean(boosted_errors)
                    reduction = (avg_base - avg_boosted) / avg_base * 100

                    boost_stats[layer] = {
                        'boosted_count': boosted_count,
                        'avg_base_error': float(avg_base),
                        'avg_boosted_error': float(avg_boosted),
                        'error_reduction_pct': float(reduction)
                    }

                    logger.info(f"  Layer {layer}: {n_games} games ({n_bankrupt} bankrupt), "
                               f"boost reduction: {reduction:.1f}%")
                else:
                    logger.info(f"  Layer {layer}: {n_games} games ({n_bankrupt} bankrupt)")
            else:
                logger.info(f"  Layer {layer}: {n_games} games ({n_bankrupt} bankrupt)")

        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'n_layers': len(self.target_layers),
            'target_layers': self.target_layers,
            'n_games': len(layer_data[self.target_layers[0]]['features']) if layer_data[self.target_layers[0]]['features'] else 0,
            'model': self.config.get('model', {}).get('model_id'),
            'sae_width': self.config.get('sae', {}).get('width'),
            'boosted': self.use_boost,
            'boost_stats': boost_stats
        }

        metadata_file = self.output_dir / "extraction_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Phase 1: SAE Feature Extraction')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    parser.add_argument('--layers', type=str, default=None,
                       help='Comma-separated layer numbers (e.g., "25,30,35")')
    parser.add_argument('--no-hidden', action='store_true',
                       help='Do not save raw hidden states')
    parser.add_argument('--use-boost', action='store_true',
                       help='Use Boosted SAE (requires Phase 0)')

    args = parser.parse_args()

    # Parse layers
    target_layers = None
    if args.layers:
        target_layers = [int(l) for l in args.layers.split(',')]

    # Run extraction
    phase1 = Phase1FeatureExtraction(
        config_path=args.config,
        gpu_id=args.gpu,
        target_layers=target_layers,
        use_boost=args.use_boost
    )
    phase1.run(save_hidden_states=not args.no_hidden)


if __name__ == '__main__':
    main()
