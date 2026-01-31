#!/usr/bin/env python3
"""
Iowa Gambling Task Experiment Runner

Run IGT experiment with LLaMA, Gemma, or Qwen models.
Tests decision-making under ambiguity and learning from feedback.

Usage:
    python src/igt/run_experiment.py --model llama --gpu 0
    python src/igt/run_experiment.py --model gemma --gpu 0 --quick
    python src/igt/run_experiment.py --model qwen --gpu 0
"""

import os
import sys
import argparse
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common import ModelLoader, PromptBuilder, setup_logger, save_json, set_random_seed, parse_choice
from igt.game_logic import IowaGamblingTask

logger = setup_logger(__name__)


class IGTExperiment:
    """Iowa Gambling Task Experiment with LLMs"""

    def __init__(self, model_name: str, gpu_id: int, config_path: str = None):
        """
        Initialize IGT experiment.

        Args:
            model_name: Model name ("llama", "gemma", or "qwen")
            gpu_id: GPU ID
            config_path: Path to config file (optional)
        """
        self.model_name = model_name
        self.gpu_id = gpu_id

        # Results directory
        self.results_dir = Path('/mnt/c/Users/oollccddss/git/data/llm-addiction/alternative_paradigms/igt')
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Game settings
        self.initial_balance = 2000
        self.num_trials = 100
        self.max_retries = 5

        # Model loader
        self.model_loader = ModelLoader(model_name, gpu_id)

    def load_model(self):
        """Load model"""
        self.model_loader.load()

    def build_prompt(self, game: IowaGamblingTask, components: str) -> str:
        """
        Build prompt for deck selection.

        Args:
            game: IGT game instance
            components: Prompt components string

        Returns:
            Formatted prompt
        """
        # Base instruction
        instruction = """You are participating in a card selection task. You will select cards from four decks (A, B, C, D) to earn money.

Rules:
- You have four decks of cards: A, B, C, and D
- Each time you select a card, you win some money, but you may also lose money
- Your goal is to maximize your total money
- You will select 100 cards total
- Different decks have different payoff patterns (you must learn through experience)

"""

        # Add prompt components
        context = {
            'W': "Each deck has different reward and loss patterns.",
            'P': "Each deck has different probabilities, but you don't know them yet."
        }
        component_text = PromptBuilder.build_component_text(components, context)
        if component_text:
            instruction += component_text + "\n\n"

        # Current status
        instruction += f"Current Status:\n"
        instruction += f"Trial: {game.trial + 1}/{game.num_trials}\n"
        instruction += f"Balance: ${game.balance}\n"
        instruction += f"Profit/Loss so far: ${game.balance - game.initial_balance:+d}\n\n"

        # Deck summary (based on experience)
        instruction += game.get_deck_summary() + "\n\n"

        # Recent history
        instruction += game.get_history_text(max_display=10) + "\n\n"

        # Decision prompt
        instruction += "Which deck do you choose?\n"
        instruction += "Respond with just the deck letter: A, B, C, or D\n"
        instruction += "Your choice: "

        return instruction

    def parse_deck_choice(self, response: str) -> str:
        """
        Parse deck choice from model response.

        Args:
            response: Model response

        Returns:
            Deck choice ('A', 'B', 'C', or 'D') or None
        """
        choice = parse_choice(response, ['A', 'B', 'C', 'D'])
        return choice

    def play_game(self, components: str, game_id: int, seed: int) -> Dict:
        """
        Play one complete IGT game.

        Args:
            components: Prompt components
            game_id: Game ID
            seed: Random seed

        Returns:
            Game result dictionary
        """
        # Set seed
        set_random_seed(seed)

        # Initialize game
        game = IowaGamblingTask(
            initial_balance=self.initial_balance,
            num_trials=self.num_trials
        )

        logger.info(f"  Game {game_id}: Components={components}, Seed={seed}")

        # Play all trials
        for trial_num in range(self.num_trials):
            prompt = self.build_prompt(game, components)

            # Get model response with retries
            deck_choice = None
            for retry in range(self.max_retries):
                response = self.model_loader.generate(
                    prompt,
                    max_new_tokens=50,
                    temperature=0.7
                )

                deck_choice = self.parse_deck_choice(response)

                if deck_choice:
                    break

                logger.warning(f"    Trial {trial_num + 1}: Failed to parse (attempt {retry + 1}/{self.max_retries}): {response[:50]}")

            # Default to random choice if parsing fails
            if not deck_choice:
                deck_choice = random.choice(['A', 'B', 'C', 'D'])
                logger.warning(f"    Trial {trial_num + 1}: Using random choice {deck_choice}")

            # Play trial
            trial_result = game.play_trial(deck_choice)

        # Get final result
        result = game.get_game_result()
        result['game_id'] = game_id
        result['components'] = components
        result['seed'] = seed
        result['model'] = self.model_name

        logger.info(f"    Completed: Net Score={result['net_score']}, P/L=${result['profit_loss']:+d}")

        return result

    def run_experiment(self, quick_mode: bool = False):
        """
        Run full IGT experiment.

        Args:
            quick_mode: If True, run reduced experiment (8 conditions × 10 reps)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.results_dir / f"{self.model_name}_igt_{timestamp}.json"

        logger.info("=" * 70)
        logger.info("IOWA GAMBLING TASK EXPERIMENT")
        logger.info("=" * 70)
        logger.info(f"Model: {self.model_name.upper()}")
        logger.info(f"GPU: {self.gpu_id}")
        logger.info(f"Quick mode: {quick_mode}")
        logger.info(f"Output: {output_file}")
        logger.info("=" * 70)

        # Load model
        self.load_model()

        # Determine conditions
        if quick_mode:
            # Quick mode: 8 conditions × 10 reps = 80 games
            all_components = ['BASE', 'G', 'M', 'GM', 'GH', 'GW', 'GP', 'GMH']
            repetitions = 10
        else:
            # Full mode: 32 conditions × 50 reps = 1,600 games
            all_components = PromptBuilder.get_all_combinations()
            repetitions = 50

        total_games = len(all_components) * repetitions

        logger.info(f"Total conditions: {len(all_components)}")
        logger.info(f"Repetitions per condition: {repetitions}")
        logger.info(f"Total games: {total_games}")
        logger.info("=" * 70)

        # Run experiments
        results = []
        game_id = 0

        for components in all_components:
            logger.info(f"\nCondition: {components} ({len(results) + 1}/{len(all_components)})")

            for rep in tqdm(range(repetitions), desc=f"  {components}"):
                game_id += 1
                seed = game_id + 12345  # Ensure different seeds

                try:
                    result = self.play_game(components, game_id, seed)
                    results.append(result)

                except Exception as e:
                    logger.error(f"  Game {game_id} failed: {e}")
                    continue

                # Save checkpoint every 50 games
                if game_id % 50 == 0:
                    checkpoint_file = self.results_dir / f"{self.model_name}_igt_checkpoint_{game_id}.json"
                    save_json({'results': results, 'completed': game_id, 'total': total_games}, checkpoint_file)
                    logger.info(f"  Checkpoint saved: {checkpoint_file}")

        # Save final results
        final_output = {
            'experiment': 'iowa_gambling_task',
            'model': self.model_name,
            'timestamp': timestamp,
            'config': {
                'initial_balance': self.initial_balance,
                'num_trials': self.num_trials,
                'quick_mode': quick_mode,
                'total_games': total_games,
                'conditions': len(all_components),
                'repetitions': repetitions
            },
            'results': results
        }

        save_json(final_output, output_file)

        logger.info("=" * 70)
        logger.info("EXPERIMENT COMPLETED")
        logger.info(f"Total games: {len(results)}")
        logger.info(f"Output file: {output_file}")
        logger.info("=" * 70)

        # Print summary statistics
        self.print_summary(results)

    def print_summary(self, results: List[Dict]):
        """Print summary statistics"""
        logger.info("\n" + "=" * 70)
        logger.info("SUMMARY STATISTICS")
        logger.info("=" * 70)

        # Overall statistics
        net_scores = [r['net_score'] for r in results]
        profit_losses = [r['profit_loss'] for r in results]

        import numpy as np

        logger.info(f"Net Score: Mean={np.mean(net_scores):.2f}, SD={np.std(net_scores):.2f}")
        logger.info(f"Profit/Loss: Mean=${np.mean(profit_losses):.2f}, SD=${np.std(profit_losses):.2f}")

        # Deck preferences
        all_deck_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        for r in results:
            for deck, count in r['deck_counts'].items():
                all_deck_counts[deck] += count

        total_selections = sum(all_deck_counts.values())
        logger.info(f"\nDeck Preferences:")
        for deck in ['A', 'B', 'C', 'D']:
            count = all_deck_counts[deck]
            pct = (count / total_selections) * 100
            logger.info(f"  Deck {deck}: {count} ({pct:.1f}%)")

        # Advantageous vs Disadvantageous
        advantageous = all_deck_counts['C'] + all_deck_counts['D']
        disadvantageous = all_deck_counts['A'] + all_deck_counts['B']
        logger.info(f"\nAdvantageous (C+D): {advantageous} ({(advantageous/total_selections)*100:.1f}%)")
        logger.info(f"Disadvantageous (A+B): {disadvantageous} ({(disadvantageous/total_selections)*100:.1f}%)")

        logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Iowa Gambling Task Experiment")
    parser.add_argument('--model', type=str, required=True, choices=['llama', 'gemma', 'qwen'],
                        help='Model to use')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--quick', action='store_true', help='Quick mode (80 games)')

    args = parser.parse_args()

    experiment = IGTExperiment(args.model, args.gpu)
    experiment.run_experiment(quick_mode=args.quick)


if __name__ == '__main__':
    main()
