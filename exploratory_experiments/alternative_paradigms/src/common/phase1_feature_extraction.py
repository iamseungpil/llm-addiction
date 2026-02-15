#!/usr/bin/env python3
"""
Phase 1: SAE Feature Extraction for Alternative Paradigms

Extracts SAE features from gambling experiments (Blackjack).
Follows the same structure as paper_experiments/llama_sae_analysis/phase1_feature_extraction.py

Usage:
    python src/common/phase1_feature_extraction.py --paradigm blackjack --model llama --gpu 0
    python src/common/phase1_feature_extraction.py --paradigm blackjack --model gemma --gpu 0
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from typing import List, Dict, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common import ModelLoader, setup_logger, clear_gpu_memory, set_random_seed

logger = setup_logger(__name__)


class SAELoader:
    """Load SAE models from HuggingFace"""

    def __init__(self, model_type: str):
        """
        Initialize SAE loader.

        Args:
            model_type: 'llama' or 'gemma'
        """
        self.model_type = model_type

        if model_type == 'llama':
            self.sae_repo = "fnlp/Llama3_1-8B-Base-LXR-8x"
            self.layers = list(range(25, 32))  # Layers 25-31
            self.hidden_dim = 4096
            self.n_features = 131072
        elif model_type == 'gemma':
            self.sae_repo = "google/gemma-scope-2b-pt-res"  # Use 2B for lower VRAM
            self.layers = list(range(1, 27))  # Layers 1-26
            self.hidden_dim = 2048
            self.n_features = 16384  # 16K features (base width)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        logger.info(f"SAE Loader initialized for {model_type.upper()}")
        logger.info(f"  Repository: {self.sae_repo}")
        logger.info(f"  Layers: {self.layers}")
        logger.info(f"  Features per layer: {self.n_features}")

    def load_sae(self, layer: int, device: str = 'cuda:0'):
        """
        Load SAE for specific layer.

        Args:
            layer: Layer number
            device: Device to load on

        Returns:
            SAE model
        """
        from sae_lens import SAE

        if layer not in self.layers:
            raise ValueError(f"Layer {layer} not available for {self.model_type}")

        if self.model_type == 'llama':
            release = f"llama_31_8b_scan_8192_L{layer}"
            sae_id = f"{self.sae_repo}/main/{release}"
        else:  # gemma
            sae_id = f"{self.sae_repo}/layer_{layer}/width_16k/average_l0_71"

        logger.info(f"  Loading SAE for layer {layer}...")
        sae = SAE.from_pretrained(sae_id, device=device)[0]
        sae.eval()

        return sae


class AlternativeParadigmFeatureExtractor:
    """Extract SAE features from alternative paradigm experiments"""

    DEFAULT_DATA_DIR = '/scratch/x3415a02/data/llm-addiction'

    def __init__(
        self,
        paradigm: str,
        model_name: str,
        gpu_id: int = 0,
        data_dir: str = None
    ):
        """
        Initialize feature extractor.

        Args:
            paradigm: 'blackjack'
            model_name: 'llama' or 'gemma'
            gpu_id: GPU ID
            data_dir: Data directory
        """
        self.paradigm = paradigm
        self.model_name = model_name
        self.gpu_id = gpu_id
        self.device = f'cuda:{gpu_id}'

        base_dir = Path(data_dir) if data_dir else Path(self.DEFAULT_DATA_DIR)
        self.data_dir = base_dir / paradigm
        self.output_dir = self.data_dir / 'sae_features'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model_loader = ModelLoader(model_name, gpu_id)
        self.sae_loader = SAELoader(model_name)

    def load_experiment_data(self) -> List[Dict]:
        """
        Load experiment JSON files.

        Returns:
            List of game dictionaries
        """
        logger.info(f"Loading experiment data from {self.data_dir}")

        # Find JSON files
        json_files = list(self.data_dir.glob(f"*_{self.model_name}_*.json"))
        if not json_files:
            raise FileNotFoundError(f"No experiment files found in {self.data_dir}")

        # Load all games
        all_games = []
        for json_file in sorted(json_files):
            logger.info(f"  Loading {json_file.name}")
            with open(json_file) as f:
                data = json.load(f)
                games = data.get('games', [])
                all_games.extend(games)

        logger.info(f"  Loaded {len(all_games)} games")
        return all_games

    def extract_prompts_and_outcomes(self, games: List[Dict]) -> Tuple[List[str], List[str], List[int]]:
        """
        Extract prompts and outcomes from games.

        Args:
            games: List of game dictionaries

        Returns:
            (prompts, outcomes, game_ids)
            outcomes: 'bankrupt' or 'voluntary_stop'
        """
        prompts = []
        outcomes = []
        game_ids = []

        for game in games:
            game_id = game['game_id']
            game_outcome = game['outcome']

            # Extract trials/rounds
            trials = game.get('rounds', [])

            # Get final trial (last decision point before outcome)
            if trials:
                final_trial = trials[-1]
                prompt = final_trial.get('full_prompt')

                if prompt:
                    prompts.append(prompt)
                    outcomes.append(game_outcome)
                    game_ids.append(game_id)

        logger.info(f"  Extracted {len(prompts)} decision points")
        logger.info(f"    Bankrupt: {outcomes.count('bankrupt')}")
        logger.info(f"    Voluntary stop: {outcomes.count('voluntary_stop')}")

        return prompts, outcomes, game_ids

    def extract_hidden_states(self, prompts: List[str]) -> torch.Tensor:
        """
        Extract hidden states from prompts.

        Args:
            prompts: List of prompts

        Returns:
            Hidden states tensor (n_samples, hidden_dim)
        """
        logger.info(f"Extracting hidden states from {len(prompts)} prompts...")

        hidden_states = []

        for prompt in tqdm(prompts, desc="  Extracting hidden states"):
            with torch.no_grad():
                # Tokenize
                inputs = self.model_loader.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048
                ).to(self.device)

                # Forward pass with hidden states
                outputs = self.model_loader.model(
                    **inputs,
                    output_hidden_states=True
                )

                # Get last token hidden state from last layer
                last_layer_hidden = outputs.hidden_states[-1]  # (batch_size, seq_len, hidden_dim)
                last_token_hidden = last_layer_hidden[:, -1, :]  # (batch_size, hidden_dim)

                hidden_states.append(last_token_hidden.cpu())

        # Stack into single tensor
        hidden_states = torch.cat(hidden_states, dim=0)  # (n_samples, hidden_dim)

        logger.info(f"  Hidden states shape: {hidden_states.shape}")

        return hidden_states

    def encode_with_sae(self, hidden_states: torch.Tensor, layer: int) -> np.ndarray:
        """
        Encode hidden states with SAE.

        Args:
            hidden_states: Hidden states (n_samples, hidden_dim)
            layer: SAE layer

        Returns:
            SAE features (n_samples, n_features)
        """
        logger.info(f"Encoding with SAE layer {layer}...")

        # Load SAE
        sae = self.sae_loader.load_sae(layer, device=self.device)

        # Encode
        features = []
        batch_size = 32

        for i in tqdm(range(0, len(hidden_states), batch_size), desc="  Encoding"):
            batch = hidden_states[i:i+batch_size].to(self.device)

            with torch.no_grad():
                # SAE encode
                sae_features = sae.encode(batch)  # (batch_size, n_features)
                features.append(sae_features.cpu().numpy())

        features = np.concatenate(features, axis=0)  # (n_samples, n_features)

        logger.info(f"  Features shape: {features.shape}")
        logger.info(f"  Mean activation: {features.mean():.4f}")
        logger.info(f"  Sparsity: {(features == 0).sum() / features.size:.2%}")

        # Clean up SAE
        del sae
        clear_gpu_memory()

        return features

    def save_features(self, layer: int, features: np.ndarray, outcomes: List[str], game_ids: List[int]):
        """
        Save features to NPZ file (same format as paper experiments).

        Args:
            layer: Layer number
            features: SAE features
            outcomes: Outcomes
            game_ids: Game IDs
        """
        output_file = self.output_dir / f'layer_{layer}_features.npz'

        np.savez(
            output_file,
            features=features,
            outcomes=np.array(outcomes),
            game_ids=np.array(game_ids),
            layer=layer,
            model_type=self.model_name,
            paradigm=self.paradigm,
            n_samples=len(outcomes),
            n_features=features.shape[1],
            timestamp=datetime.now().isoformat()
        )

        logger.info(f"  Saved to {output_file}")

    def run(self):
        """Run full Phase 1 extraction pipeline"""
        logger.info(f"\n{'='*60}")
        logger.info(f"PHASE 1: SAE FEATURE EXTRACTION")
        logger.info(f"Paradigm: {self.paradigm.upper()}")
        logger.info(f"Model: {self.model_name.upper()}")
        logger.info(f"{'='*60}\n")

        # Load model
        logger.info("Step 1: Loading model...")
        self.model_loader.load()

        # Load experiment data
        logger.info("\nStep 2: Loading experiment data...")
        games = self.load_experiment_data()

        # Extract prompts and outcomes
        logger.info("\nStep 3: Extracting prompts and outcomes...")
        prompts, outcomes, game_ids = self.extract_prompts_and_outcomes(games)

        if len(prompts) == 0:
            raise ValueError("No prompts extracted! Check experiment data format.")

        # Extract hidden states (once)
        logger.info("\nStep 4: Extracting hidden states...")
        hidden_states = self.extract_hidden_states(prompts)

        # Encode with SAE for each layer
        logger.info(f"\nStep 5: Encoding with SAE ({len(self.sae_loader.layers)} layers)...")
        for layer in self.sae_loader.layers:
            logger.info(f"\n  === Layer {layer} ===")
            features = self.encode_with_sae(hidden_states, layer)
            self.save_features(layer, features, outcomes, game_ids)

        # Clean up
        self.model_loader.unload()

        logger.info(f"\n{'='*60}")
        logger.info("PHASE 1 COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Layers processed: {len(self.sae_loader.layers)}")
        logger.info(f"Samples per layer: {len(outcomes)}")


def main():
    parser = argparse.ArgumentParser(description='Phase 1: SAE Feature Extraction')
    parser.add_argument('--paradigm', type=str, required=True, choices=['blackjack'],
                        help='Paradigm to extract features from')
    parser.add_argument('--model', type=str, required=True, choices=['llama', 'gemma'],
                        help='Model type')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Data directory (default: /scratch/x3415a02/data/llm-addiction)')

    args = parser.parse_args()

    # Set seed for reproducibility
    set_random_seed(42)

    # Run extraction
    extractor = AlternativeParadigmFeatureExtractor(
        paradigm=args.paradigm,
        model_name=args.model,
        gpu_id=args.gpu,
        data_dir=args.data_dir
    )

    extractor.run()


if __name__ == '__main__':
    main()
