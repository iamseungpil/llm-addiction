#!/usr/bin/env python3
"""
Phase 3: Steering Vector Extraction

Compute CAA-style steering vectors: mean(bankrupt) - mean(safe)

Input:
    - Experiment data JSON

Output:
    - phase3_steering/steering_vectors.npz

Usage:
    python phase3_steering_vector.py --gpu 0
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
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from config import load_config, get_phase_output_dir
from utils import (
    GemmaBaseModel,
    load_experiment_data, reconstruct_decision_prompt, group_by_outcome,
    clear_gpu_memory, logger
)


class Phase3SteeringVector:
    """Extract steering vectors using Contrastive Activation Addition."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        gpu_id: int = 0,
        target_layers: Optional[List[int]] = None
    ):
        self.config = load_config(config_path)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        self.device = 'cuda:0'

        if target_layers:
            self.target_layers = target_layers
        else:
            self.target_layers = self.config.get('target_layers', list(range(20, 42)))

        self.output_dir = get_phase_output_dir(self.config, 3)

        self.max_samples = self.config.get('phase3', {}).get('max_samples_per_group', 500)

        self.model = None

    def load_model(self):
        """Load Gemma model."""
        self.model = GemmaBaseModel(device=self.device)
        self.model.load()

    def match_games_by_condition(
        self,
        bankrupt_games: List[Dict],
        safe_games: List[Dict]
    ) -> List[tuple]:
        """
        Match bankrupt games to safe games by condition (bet_type, prompt_combo).

        Returns:
            List of (bankrupt_game, safe_game) pairs
        """
        # Group safe games by condition
        safe_by_condition = defaultdict(list)
        for game in safe_games:
            key = (game['bet_type'], game['prompt_combo'])
            safe_by_condition[key].append(game)

        # Match bankrupt games
        pairs = []
        used_safe = set()

        for b_game in bankrupt_games:
            key = (b_game['bet_type'], b_game['prompt_combo'])
            candidates = safe_by_condition.get(key, [])

            # Find unused safe game
            for s_game in candidates:
                s_id = id(s_game)
                if s_id not in used_safe:
                    pairs.append((b_game, s_game))
                    used_safe.add(s_id)
                    break

        return pairs

    def extract_hidden_states(
        self,
        games: List[Dict],
        desc: str = "Extracting"
    ) -> Dict[int, np.ndarray]:
        """
        Extract hidden states for a list of games.

        Returns:
            Dict mapping layer to array of hidden states [n_games, d_model]
        """
        layer_states = {layer: [] for layer in self.target_layers}

        for game in tqdm(games, desc=desc):
            try:
                prompt = reconstruct_decision_prompt(game)
                hidden = self.model.get_hidden_states(prompt, self.target_layers, position='last')

                for layer in self.target_layers:
                    if layer in hidden:
                        layer_states[layer].append(hidden[layer].squeeze(0).numpy())

            except Exception as e:
                logger.warning(f"Error extracting hidden state: {e}")
                continue

            # Periodic memory cleanup
            if len(layer_states[self.target_layers[0]]) % 50 == 0:
                clear_gpu_memory()

        # Convert to arrays
        return {
            layer: np.array(states) if states else None
            for layer, states in layer_states.items()
        }

    def compute_steering_vectors(
        self,
        bankrupt_states: Dict[int, np.ndarray],
        safe_states: Dict[int, np.ndarray]
    ) -> Dict[int, Dict]:
        """
        Compute steering vectors for each layer.

        steering_vector[layer] = mean(bankrupt) - mean(safe)
        """
        steering_vectors = {}

        for layer in self.target_layers:
            b_states = bankrupt_states.get(layer)
            s_states = safe_states.get(layer)

            if b_states is None or s_states is None:
                continue

            if len(b_states) < 2 or len(s_states) < 2:
                continue

            b_mean = np.mean(b_states, axis=0)
            s_mean = np.mean(s_states, axis=0)

            steering_vec = b_mean - s_mean
            magnitude = np.linalg.norm(steering_vec)

            steering_vectors[layer] = {
                'vector': steering_vec,
                'bankrupt_mean': b_mean,
                'safe_mean': s_mean,
                'magnitude': magnitude,
                'n_bankrupt': len(b_states),
                'n_safe': len(s_states)
            }

            logger.info(f"  Layer {layer}: magnitude={magnitude:.4f} (n={len(b_states)})")

        return steering_vectors

    def run(self):
        """Run Phase 3 steering vector extraction."""
        logger.info("=" * 70)
        logger.info("PHASE 3: STEERING VECTOR EXTRACTION")
        logger.info("=" * 70)
        logger.info(f"Target layers: {self.target_layers}")
        logger.info(f"Max samples per group: {self.max_samples}")

        # Load model
        self.load_model()

        # Load experiment data
        data_path = self.config.get('data', {}).get('experiment_data')
        exp_data = load_experiment_data(data_path)
        games = exp_data['results']

        # Group by outcome
        grouped = group_by_outcome(games)
        bankrupt_games = grouped['bankruptcy']
        safe_games = grouped['voluntary_stop']

        logger.info(f"Total: {len(bankrupt_games)} bankrupt, {len(safe_games)} safe")

        # Match games by condition
        pairs = self.match_games_by_condition(bankrupt_games, safe_games)
        logger.info(f"Matched pairs: {len(pairs)}")

        # Limit samples
        if len(pairs) > self.max_samples:
            np.random.seed(self.config.get('random_seed', 42))
            indices = np.random.choice(len(pairs), self.max_samples, replace=False)
            pairs = [pairs[i] for i in indices]
            logger.info(f"Limited to {len(pairs)} pairs")

        # Separate paired games
        matched_bankrupt = [p[0] for p in pairs]
        matched_safe = [p[1] for p in pairs]

        # Extract hidden states
        logger.info("\nExtracting bankrupt hidden states...")
        bankrupt_states = self.extract_hidden_states(matched_bankrupt, "Bankrupt")

        logger.info("\nExtracting safe hidden states...")
        safe_states = self.extract_hidden_states(matched_safe, "Safe")

        # Compute steering vectors
        logger.info("\nComputing steering vectors...")
        steering_vectors = self.compute_steering_vectors(bankrupt_states, safe_states)

        # Save results
        self._save_results(steering_vectors, bankrupt_states, safe_states)

        logger.info("\nPhase 3 complete!")

    def _save_results(
        self,
        steering_vectors: Dict[int, Dict],
        bankrupt_states: Dict[int, np.ndarray],
        safe_states: Dict[int, np.ndarray]
    ):
        """Save steering vectors and hidden states."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Prepare save dict
        save_dict = {
            'timestamp': timestamp,
            'model': self.config.get('model', {}).get('model_id'),
            'n_layers': len(steering_vectors)
        }

        for layer, data in steering_vectors.items():
            save_dict[f'layer_{layer}_vector'] = data['vector']
            save_dict[f'layer_{layer}_bankrupt_mean'] = data['bankrupt_mean']
            save_dict[f'layer_{layer}_safe_mean'] = data['safe_mean']
            save_dict[f'layer_{layer}_magnitude'] = data['magnitude']
            save_dict[f'layer_{layer}_n_bankrupt'] = data['n_bankrupt']
            save_dict[f'layer_{layer}_n_safe'] = data['n_safe']

        # Save steering vectors
        vectors_file = self.output_dir / f"steering_vectors_{timestamp}.npz"
        np.savez_compressed(vectors_file, **save_dict)
        logger.info(f"Saved: {vectors_file}")

        # Also save as latest
        latest_file = self.output_dir / "steering_vectors.npz"
        np.savez_compressed(latest_file, **save_dict)
        logger.info(f"Saved: {latest_file}")

        # Save hidden states (for potential use in other phases)
        hidden_dir = self.output_dir / "hidden_states"
        hidden_dir.mkdir(exist_ok=True)

        for layer in self.target_layers:
            if bankrupt_states.get(layer) is not None:
                np.save(
                    hidden_dir / f"layer_{layer}_bankrupt.npy",
                    bankrupt_states[layer]
                )
            if safe_states.get(layer) is not None:
                np.save(
                    hidden_dir / f"layer_{layer}_safe.npy",
                    safe_states[layer]
                )

        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'layers': list(steering_vectors.keys()),
            'magnitudes': {str(l): float(d['magnitude']) for l, d in steering_vectors.items()},
            'samples': {str(l): {'bankrupt': d['n_bankrupt'], 'safe': d['n_safe']}
                       for l, d in steering_vectors.items()}
        }

        metadata_file = self.output_dir / "steering_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Phase 3: Steering Vector Extraction')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    parser.add_argument('--layers', type=str, default=None,
                       help='Comma-separated layer numbers')

    args = parser.parse_args()

    target_layers = None
    if args.layers:
        target_layers = [int(l) for l in args.layers.split(',')]

    phase3 = Phase3SteeringVector(
        config_path=args.config,
        gpu_id=args.gpu,
        target_layers=target_layers
    )
    phase3.run()


if __name__ == '__main__':
    main()
