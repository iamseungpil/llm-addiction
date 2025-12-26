#!/usr/bin/env python3
"""
Prepare Clean/Corrupt Prompt Pairs for Sparse Feature Circuits

This module prepares contrastive prompt pairs from gambling experiment data:
- Clean prompts: Final decision prompts from bankruptcy games
- Corrupt prompts: Final decision prompts from voluntary stop games

These pairs are used for attribution patching to discover causal features.

Usage:
    python prepare_prompt_pairs.py --model llama --output prompt_pairs_llama.json
    python prepare_prompt_pairs.py --model gemma --output prompt_pairs_gemma.json
"""

import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import sys

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    setup_logging,
    load_experiment_data,
    group_by_outcome,
    PromptBuilder,
    get_default_config_path
)
import yaml


@dataclass
class PromptPair:
    """A clean/corrupt prompt pair for attribution patching."""
    clean_prompt: str           # Bankruptcy context
    corrupt_prompt: str         # Safe (voluntary stop) context
    clean_outcome: str          # 'bankruptcy'
    corrupt_outcome: str        # 'voluntary_stop'
    clean_metadata: Dict        # bet_type, prompt_combo, etc.
    corrupt_metadata: Dict
    pair_id: int


class PromptPairGenerator:
    """
    Generate contrastive prompt pairs for circuit discovery.

    Strategy: Match bankruptcy and safe games by bet_type and prompt_combo
    to control for confounds. This ensures the only systematic difference
    is the behavioral outcome.
    """

    def __init__(
        self,
        bankruptcy_games: List[Dict],
        safe_games: List[Dict],
        logger=None
    ):
        self.bankruptcy_games = bankruptcy_games
        self.safe_games = safe_games
        self.logger = logger

    def _log(self, msg: str):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def generate_matched_pairs(
        self,
        n_pairs: Optional[int] = None,
        match_conditions: bool = True
    ) -> List[PromptPair]:
        """
        Generate matched prompt pairs.

        Args:
            n_pairs: Number of pairs to generate (default: all bankruptcy games)
            match_conditions: If True, match by bet_type and prompt_combo

        Returns:
            List of PromptPair objects
        """
        pairs = []

        if match_conditions:
            # Group safe games by condition for efficient matching
            safe_by_condition = {}
            for game in self.safe_games:
                key = (game['bet_type'], game['prompt_combo'])
                if key not in safe_by_condition:
                    safe_by_condition[key] = []
                safe_by_condition[key].append(game)

            # Match each bankruptcy game with a random safe game from same condition
            for i, bankrupt_game in enumerate(self.bankruptcy_games):
                key = (bankrupt_game['bet_type'], bankrupt_game['prompt_combo'])

                if key not in safe_by_condition or not safe_by_condition[key]:
                    self._log(f"Warning: No matching safe game for condition {key}")
                    # Fall back to random safe game
                    safe_game = random.choice(self.safe_games)
                else:
                    safe_game = random.choice(safe_by_condition[key])

                # Reconstruct prompts
                clean_prompt = PromptBuilder.reconstruct_decision_prompt(bankrupt_game)
                corrupt_prompt = PromptBuilder.reconstruct_decision_prompt(safe_game)

                pair = PromptPair(
                    clean_prompt=clean_prompt,
                    corrupt_prompt=corrupt_prompt,
                    clean_outcome='bankruptcy',
                    corrupt_outcome='voluntary_stop',
                    clean_metadata={
                        'bet_type': bankrupt_game['bet_type'],
                        'prompt_combo': bankrupt_game['prompt_combo'],
                        'final_balance': bankrupt_game['final_balance'],
                        'total_rounds': bankrupt_game['total_rounds']
                    },
                    corrupt_metadata={
                        'bet_type': safe_game['bet_type'],
                        'prompt_combo': safe_game['prompt_combo'],
                        'final_balance': safe_game['final_balance'],
                        'total_rounds': safe_game['total_rounds']
                    },
                    pair_id=i
                )
                pairs.append(pair)

                if n_pairs and len(pairs) >= n_pairs:
                    break
        else:
            # Random pairing
            n = n_pairs or len(self.bankruptcy_games)
            for i in range(min(n, len(self.bankruptcy_games))):
                bankrupt_game = self.bankruptcy_games[i]
                safe_game = random.choice(self.safe_games)

                clean_prompt = PromptBuilder.reconstruct_decision_prompt(bankrupt_game)
                corrupt_prompt = PromptBuilder.reconstruct_decision_prompt(safe_game)

                pair = PromptPair(
                    clean_prompt=clean_prompt,
                    corrupt_prompt=corrupt_prompt,
                    clean_outcome='bankruptcy',
                    corrupt_outcome='voluntary_stop',
                    clean_metadata={
                        'bet_type': bankrupt_game['bet_type'],
                        'prompt_combo': bankrupt_game['prompt_combo'],
                        'final_balance': bankrupt_game['final_balance'],
                        'total_rounds': bankrupt_game['total_rounds']
                    },
                    corrupt_metadata={
                        'bet_type': safe_game['bet_type'],
                        'prompt_combo': safe_game['prompt_combo'],
                        'final_balance': safe_game['final_balance'],
                        'total_rounds': safe_game['total_rounds']
                    },
                    pair_id=i
                )
                pairs.append(pair)

        self._log(f"Generated {len(pairs)} prompt pairs")
        return pairs

    def generate_aggregated_pairs(
        self,
        n_clean: int = 100,
        n_corrupt: int = 100
    ) -> Tuple[List[str], List[str]]:
        """
        Generate separate lists of clean and corrupt prompts for aggregated analysis.

        This is useful when you want to compute mean activations across groups
        rather than paired comparisons.

        Args:
            n_clean: Number of bankruptcy prompts
            n_corrupt: Number of safe prompts

        Returns:
            Tuple of (clean_prompts, corrupt_prompts)
        """
        # Sample bankruptcy games
        clean_games = random.sample(
            self.bankruptcy_games,
            min(n_clean, len(self.bankruptcy_games))
        )
        clean_prompts = [
            PromptBuilder.reconstruct_decision_prompt(g)
            for g in clean_games
        ]

        # Sample safe games
        corrupt_games = random.sample(
            self.safe_games,
            min(n_corrupt, len(self.safe_games))
        )
        corrupt_prompts = [
            PromptBuilder.reconstruct_decision_prompt(g)
            for g in corrupt_games
        ]

        self._log(f"Generated {len(clean_prompts)} clean, {len(corrupt_prompts)} corrupt prompts")
        return clean_prompts, corrupt_prompts


def save_prompt_pairs(pairs: List[PromptPair], output_path: Path) -> None:
    """Save prompt pairs to JSON file."""
    data = {
        'n_pairs': len(pairs),
        'pairs': [asdict(p) for p in pairs]
    }
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_prompt_pairs(input_path: Path) -> List[PromptPair]:
    """Load prompt pairs from JSON file."""
    with open(input_path, 'r') as f:
        data = json.load(f)

    pairs = []
    for p in data['pairs']:
        pairs.append(PromptPair(**p))
    return pairs


def main():
    parser = argparse.ArgumentParser(description='Prepare prompt pairs for circuit discovery')
    parser.add_argument('--model', type=str, required=True, choices=['llama', 'gemma'])
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (default: auto-detect)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output filename (default: prompt_pairs_{model}.json)')
    parser.add_argument('--n-pairs', type=int, default=None,
                       help='Number of pairs to generate (default: all)')
    parser.add_argument('--match-conditions', action='store_true', default=True,
                       help='Match pairs by bet_type and prompt_combo')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Load config
    config_path = args.config or str(get_default_config_path())
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set seed
    random.seed(args.seed)

    # Setup logging using config paths
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging('prepare_pairs', output_dir / 'logs')

    logger.info("=" * 60)
    logger.info(f"Preparing prompt pairs for {args.model.upper()}")
    logger.info("=" * 60)

    # Load experiment data using config paths
    if args.model == 'llama':
        data_path = config['llama_data_path']
    else:
        data_path = config['gemma_data_path']

    exp_data = load_experiment_data(data_path, logger)
    results = exp_data['results']

    # Group by outcome
    grouped = group_by_outcome(results)
    bankruptcy_games = grouped['bankruptcy']
    safe_games = grouped['voluntary_stop']

    logger.info(f"Bankruptcy games: {len(bankruptcy_games)}")
    logger.info(f"Safe games: {len(safe_games)}")

    # Generate pairs
    generator = PromptPairGenerator(bankruptcy_games, safe_games, logger)
    pairs = generator.generate_matched_pairs(
        n_pairs=args.n_pairs,
        match_conditions=args.match_conditions
    )

    # Save
    output_filename = args.output or f'prompt_pairs_{args.model}.json'
    output_path = output_dir / output_filename
    save_prompt_pairs(pairs, output_path)

    logger.info(f"Saved {len(pairs)} pairs to {output_path}")

    # Print sample
    logger.info("\nSample pair:")
    logger.info(f"Clean (bankruptcy) prompt preview: {pairs[0].clean_prompt[:200]}...")
    logger.info(f"Corrupt (safe) prompt preview: {pairs[0].corrupt_prompt[:200]}...")

    logger.info("\nPrompt pair preparation complete!")


if __name__ == '__main__':
    main()
