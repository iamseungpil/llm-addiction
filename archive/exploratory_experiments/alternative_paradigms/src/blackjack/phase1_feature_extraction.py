#!/usr/bin/env python3
"""
Phase 1: SAE Feature Extraction for Blackjack Experiment

Extracts hidden states and SAE features from blackjack round prompts.

Usage:
    python src/blackjack/phase1_feature_extraction.py --model llama --gpu 0 --input results.json
    python src/blackjack/phase1_feature_extraction.py --model gemma --gpu 0 --input results.json --layers 20,25,30,35,40
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common import setup_logger, clear_gpu_memory, set_random_seed
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

logger = setup_logger(__name__)


class BlackjackFeatureExtractor:
    """Extract SAE features from Blackjack experiment rounds."""

    def __init__(self, model_name: str, gpu_id: int = 0):
        """
        Initialize feature extractor.

        Args:
            model_name: 'llama' or 'gemma'
            gpu_id: GPU device ID
        """
        self.model_name = model_name
        self.device = torch.device(f'cuda:{gpu_id}' if gpu_id >= 0 and torch.cuda.is_available() else 'cpu')

        logger.info(f"Using device: {self.device}")

        # Model configurations
        if model_name == 'llama':
            self.model_id = "meta-llama/Llama-3.1-8B"
            self.sae_release = "fnlp/Llama3_1-8B-Base-LXR-8x"
            self.default_layers = [25, 26, 27, 28, 29, 30, 31]
            self.d_sae = 32768
        elif model_name == 'gemma':
            self.model_id = "google/gemma-2-9b"
            self.sae_release = "gemma-scope-9b-pt-res-canonical"
            self.default_layers = list(range(20, 42))
            self.d_sae = 131072
        else:
            raise ValueError(f"Unknown model: {model_name}")

        self.model = None
        self.tokenizer = None
        self.saes = {}

    def load_model(self):
        """Load language model."""
        logger.info(f"Loading model: {self.model_id}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map={'': self.device},
            low_cpu_mem_usage=True
        )
        self.model.eval()

        logger.info(f"Model loaded successfully")

    def load_sae(self, layer: int):
        """Load SAE for a specific layer."""
        if layer in self.saes:
            return

        logger.info(f"Loading SAE for layer {layer}")

        if self.model_name == 'gemma':
            sae_id = f"layer_{layer}/width_131k/canonical"
        elif self.model_name == 'llama':
            sae_id = f"layer_{layer}"
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        sae, _, _ = SAE.from_pretrained(
            release=self.sae_release,
            sae_id=sae_id,
            device=str(self.device)
        )

        self.saes[layer] = sae
        logger.info(f"SAE loaded for layer {layer}")

    def extract_features_for_round(self, prompt: str, layers: List[int]) -> Dict[int, np.ndarray]:
        """
        Extract SAE features for a single round prompt.

        Args:
            prompt: Round prompt text
            layers: List of layer numbers

        Returns:
            Dictionary mapping layer -> SAE features [d_sae]
        """
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=2048
        ).to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )

        hidden_states = outputs.hidden_states

        # Extract features for each layer
        layer_features = {}

        for layer in layers:
            layer_hidden = hidden_states[layer + 1]
            last_token_hidden = layer_hidden[0, -1, :]

            if layer not in self.saes:
                self.load_sae(layer)

            sae = self.saes[layer]
            sae_features = sae.encode(last_token_hidden)

            layer_features[layer] = sae_features.cpu().numpy()

        return layer_features

    def process_json_file(
        self,
        json_file: Path,
        output_dir: Path,
        target_layers: List[int],
        checkpoint_freq: int = 100
    ):
        """
        Process Blackjack JSON file and extract features.

        Args:
            json_file: Path to Blackjack results JSON
            output_dir: Output directory for NPZ files
            target_layers: Layers to extract features from
            checkpoint_freq: Save checkpoint every N rounds
        """
        logger.info(f"Loading JSON: {json_file}")

        with open(json_file, 'r') as f:
            data = json.load(f)

        games = data['games']
        logger.info(f"Found {len(games)} games")

        # Count total rounds
        total_rounds = sum(len(game['rounds']) for game in games)
        logger.info(f"Total rounds across all games: {total_rounds}")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process each layer separately
        for layer in target_layers:
            logger.info(f"\n{'='*70}")
            logger.info(f"Processing Layer {layer}")
            logger.info(f"{'='*70}")

            # Check for existing final file
            final_file = output_dir / f'layer_{layer}_features.npz'
            if final_file.exists():
                logger.info(f"Layer {layer} already completed, skipping")
                continue

            # Pre-allocate arrays
            all_features = np.zeros((total_rounds, self.d_sae), dtype=np.float32)
            all_bets = np.zeros(total_rounds, dtype=np.int32)
            all_outcomes = []
            all_payouts = np.zeros(total_rounds, dtype=np.int32)
            all_game_ids = np.zeros(total_rounds, dtype=np.int32)
            all_rounds = np.zeros(total_rounds, dtype=np.int32)
            all_bet_types = []
            all_components = []

            # Process rounds
            round_idx = 0

            for game in tqdm(games, desc=f"Layer {layer}"):
                game_id = game['game_id']
                bet_type = game['bet_type']
                components = game['components']

                for round_data in game['rounds']:
                    try:
                        # Get prompt
                        prompt = round_data.get('full_prompt')
                        if not prompt:
                            logger.warning(f"Empty prompt for game {game_id}, round {round_data['round']}, skipping")
                            continue

                        # Extract features
                        features = self.extract_features_for_round(prompt, [layer])

                        # Store
                        all_features[round_idx] = features[layer]
                        all_bets[round_idx] = round_data['bet']
                        all_outcomes.append(round_data['outcome'])
                        all_payouts[round_idx] = round_data['payout']
                        all_game_ids[round_idx] = game_id
                        all_rounds[round_idx] = round_data['round']
                        all_bet_types.append(bet_type)
                        all_components.append(components)

                        round_idx += 1

                    except Exception as e:
                        logger.error(f"Error processing game {game_id}, round {round_data['round']}: {e}")
                        continue

                    # Checkpoint
                    if round_idx % checkpoint_freq == 0:
                        logger.info(f"  Checkpoint: {round_idx}/{total_rounds} rounds processed")

            # Trim to actual size
            all_features = all_features[:round_idx]
            all_bets = all_bets[:round_idx]
            all_payouts = all_payouts[:round_idx]
            all_game_ids = all_game_ids[:round_idx]
            all_rounds = all_rounds[:round_idx]

            # Save final NPZ
            np.savez_compressed(
                final_file,
                features=all_features,
                bets=all_bets,
                outcomes=np.array(all_outcomes, dtype=object),
                payouts=all_payouts,
                game_ids=all_game_ids,
                rounds=all_rounds,
                bet_types=np.array(all_bet_types, dtype=object),
                components=np.array(all_components, dtype=object),
                layer=layer,
                model_type=self.model_name,
                timestamp=datetime.now().isoformat()
            )

            logger.info(f"Layer {layer} completed: {round_idx} rounds saved to {final_file}")

            # Clear SAE from memory
            if layer in self.saes:
                del self.saes[layer]
                torch.cuda.empty_cache()

        logger.info("\n" + "="*70)
        logger.info("Feature extraction completed!")
        logger.info("="*70)


def main():
    parser = argparse.ArgumentParser(description="Blackjack Phase 1: Feature Extraction")
    parser.add_argument('--model', type=str, required=True, choices=['llama', 'gemma'],
                        help='Model to use')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to Blackjack results JSON file')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: same dir as input with _features suffix)')
    parser.add_argument('--layers', type=str, default=None,
                        help='Comma-separated layer numbers (e.g., "25,26,27,28,29,30,31")')

    args = parser.parse_args()

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        input_path = Path(args.input)
        output_dir = input_path.parent / f"{input_path.stem}_features"

    # Parse layers
    extractor = BlackjackFeatureExtractor(args.model, args.gpu)

    if args.layers:
        target_layers = [int(x.strip()) for x in args.layers.split(',')]
    else:
        target_layers = extractor.default_layers

    logger.info(f"Target layers: {target_layers}")

    # Load model
    extractor.load_model()

    # Process JSON
    extractor.process_json_file(
        json_file=Path(args.input),
        output_dir=output_dir,
        target_layers=target_layers
    )

    logger.info(f"All features saved to: {output_dir}")


if __name__ == '__main__':
    main()
