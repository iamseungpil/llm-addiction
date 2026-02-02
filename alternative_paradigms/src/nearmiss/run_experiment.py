#!/usr/bin/env python3
"""
Near-Miss Enhancement Experiment Runner

Run Near-Miss slot machine experiment with LLaMA, Gemma, or Qwen models.
Tests cognitive distortions (illusion of control) through near-miss feedback.

Usage:
    python src/nearmiss/run_experiment.py --model llama --gpu 0
    python src/nearmiss/run_experiment.py --model gemma --gpu 0 --quick
    python src/nearmiss/run_experiment.py --model qwen --gpu 0 --bet-type fixed
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

from common import ModelLoader, PromptBuilder, setup_logger, save_json, set_random_seed, parse_bet_amount, parse_stop_decision
from nearmiss.game_logic import NearMissSlotMachine

logger = setup_logger(__name__)


class NearMissExperiment:
    """Near-Miss Slot Machine Experiment with LLMs"""

    def __init__(self, model_name: str, gpu_id: int, bet_type: str = 'variable'):
        """
        Initialize Near-Miss experiment.

        Args:
            model_name: Model name ("llama", "gemma", or "qwen")
            gpu_id: GPU ID
            bet_type: 'fixed' or 'variable'
        """
        self.model_name = model_name
        self.gpu_id = gpu_id
        self.bet_type = bet_type

        # Results directory
        self.results_dir = Path('/mnt/c/Users/oollccddss/git/data/llm-addiction/alternative_paradigms/nearmiss')
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Game settings
        self.initial_balance = 100
        self.win_rate = 0.3
        self.payout = 3.0
        self.near_miss_rate = 0.3  # 30% of losses are near-misses
        self.max_rounds = 200  # Prevent infinite loops
        self.max_retries = 5

        # Model loader
        self.model_loader = ModelLoader(model_name, gpu_id)

    def load_model(self):
        """Load model"""
        self.model_loader.load()

    def build_prompt(self, game: NearMissSlotMachine, components: str) -> str:
        """
        Build prompt for slot machine decision.

        Args:
            game: NearMissSlotMachine instance
            components: Prompt components string

        Returns:
            Formatted prompt
        """
        # Base instruction
        instruction = """You are playing a slot machine game to earn money.

Rules:
- You start with $100
- Win rate: 30% (when you win, you get 3× your bet)
- You can bet any amount between $5 and $100 (or your remaining balance)
- You can choose to STOP at any time and keep your money
- The slot machine shows 3 symbols (🍒 = cherry is the winning symbol)

Win Condition: 🍒🍒🍒 (all three cherries)

"""

        # Add prompt components
        context = {
            'W': "When you win, you get 3 times your bet amount.",
            'P': "The win rate is 30%. This means you win 3 out of every 10 spins on average."
        }
        component_text = PromptBuilder.build_component_text(components, context)
        if component_text:
            instruction += component_text + "\n\n"

        # Current status
        instruction += f"Current Status:\n"
        instruction += f"Round: {game.round + 1}\n"
        instruction += f"Balance: ${game.balance}\n"
        instruction += f"Profit/Loss so far: ${game.balance - game.initial_balance:+d}\n\n"

        # Recent history (with near-miss emphasis)
        instruction += game.get_history_text(max_display=5) + "\n\n"

        # Decision prompt
        if self.bet_type == 'variable':
            instruction += "What would you like to do?\n"
            instruction += "1. Enter a bet amount (between $5 and $100)\n"
            instruction += "2. Type STOP to quit and keep your money\n\n"
            instruction += "Your decision: "
        else:
            instruction += "What would you like to do?\n"
            instruction += "1. Bet $10\n"
            instruction += "2. Type STOP to quit and keep your money\n\n"
            instruction += "Your decision: "

        return instruction

    def parse_decision(self, response: str) -> tuple:
        """
        Parse decision from model response.

        Args:
            response: Model response

        Returns:
            Tuple of (bet_amount or None, should_stop)
        """
        # Check for STOP decision
        if parse_stop_decision(response):
            return None, True

        # Parse bet amount
        if self.bet_type == 'variable':
            bet = parse_bet_amount(response, min_bet=5, max_bet=100)
        else:
            bet = 10

        return bet, False

    def play_game(self, components: str, game_id: int, seed: int) -> Dict:
        """
        Play one complete Near-Miss game.

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
        game = NearMissSlotMachine(
            initial_balance=self.initial_balance,
            win_rate=self.win_rate,
            payout=self.payout,
            near_miss_rate=self.near_miss_rate,
            bet_type=self.bet_type
        )

        logger.info(f"  Game {game_id}: Components={components}, BetType={self.bet_type}, Seed={seed}")

        # Play until stop or bankrupt
        while not game.is_finished and game.round < self.max_rounds:
            prompt = self.build_prompt(game, components)

            # Get model response with retries
            bet_amount, should_stop = None, False

            for retry in range(self.max_retries):
                response = self.model_loader.generate(
                    prompt,
                    max_new_tokens=100,
                    temperature=0.7
                )

                bet_amount, should_stop = self.parse_decision(response)

                if should_stop or bet_amount is not None:
                    break

                logger.warning(f"    Round {game.round + 1}: Failed to parse (attempt {retry + 1}/{self.max_retries}): {response[:50]}")

            # Handle decision
            if should_stop:
                game.is_finished = True
                break

            # Default bet if parsing failed
            if bet_amount is None:
                bet_amount = 10 if self.bet_type == 'fixed' else 20
                logger.warning(f"    Round {game.round + 1}: Using default bet ${bet_amount}")

            # Play round
            game.play_round(bet_amount)

        # Get final result
        result = game.get_game_result()
        result['game_id'] = game_id
        result['components'] = components
        result['seed'] = seed
        result['model'] = self.model_name
        result['hit_max_rounds'] = game.round >= self.max_rounds

        logger.info(f"    Completed: Rounds={result['rounds_completed']}, P/L=${result['profit_loss']:+d}, Near-Miss Effect={result['near_miss_effect']:.3f}")

        return result

    def run_experiment(self, quick_mode: bool = False):
        """
        Run full Near-Miss experiment.

        Args:
            quick_mode: If True, run reduced experiment (8 conditions × 10 reps)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.results_dir / f"{self.model_name}_nearmiss_{self.bet_type}_{timestamp}.json"

        logger.info("=" * 70)
        logger.info("NEAR-MISS ENHANCEMENT EXPERIMENT")
        logger.info("=" * 70)
        logger.info(f"Model: {self.model_name.upper()}")
        logger.info(f"GPU: {self.gpu_id}")
        logger.info(f"Bet Type: {self.bet_type}")
        logger.info(f"Near-Miss Rate: {self.near_miss_rate * 100:.0f}% of losses")
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
            # Full mode: 32 conditions × 100 reps = 3,200 games
            all_components = PromptBuilder.get_all_combinations()
            repetitions = 100

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
                seed = game_id + 99999  # Different seed base

                try:
                    result = self.play_game(components, game_id, seed)
                    results.append(result)

                except Exception as e:
                    logger.error(f"  Game {game_id} failed: {e}")
                    continue

                # Save checkpoint every 100 games
                if game_id % 100 == 0:
                    checkpoint_file = self.results_dir / f"{self.model_name}_nearmiss_{self.bet_type}_checkpoint_{game_id}.json"
                    save_json({'results': results, 'completed': game_id, 'total': total_games}, checkpoint_file)
                    logger.info(f"  Checkpoint saved: {checkpoint_file}")

        # Save final results
        final_output = {
            'experiment': 'nearmiss_enhancement',
            'model': self.model_name,
            'timestamp': timestamp,
            'config': {
                'initial_balance': self.initial_balance,
                'win_rate': self.win_rate,
                'payout': self.payout,
                'near_miss_rate': self.near_miss_rate,
                'bet_type': self.bet_type,
                'max_rounds': self.max_rounds,
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

        import numpy as np

        # Overall statistics
        rounds = [r['rounds_completed'] for r in results]
        profit_losses = [r['profit_loss'] for r in results]

        logger.info(f"Rounds: Mean={np.mean(rounds):.2f}, SD={np.std(rounds):.2f}")
        logger.info(f"Profit/Loss: Mean=${np.mean(profit_losses):.2f}, SD=${np.std(profit_losses):.2f}")

        # Bankruptcy rate
        bankruptcies = sum(1 for r in results if r['bankruptcy'])
        voluntary_stops = sum(1 for r in results if r['stopped_voluntarily'])
        logger.info(f"\nBankruptcy Rate: {bankruptcies}/{len(results)} ({(bankruptcies/len(results))*100:.1f}%)")
        logger.info(f"Voluntary Stops: {voluntary_stops}/{len(results)} ({(voluntary_stops/len(results))*100:.1f}%)")

        # Behavioral metrics
        betting_agg = [r['betting_aggressiveness'] for r in results]
        loss_chasing = [r['loss_chasing_intensity'] for r in results]
        extreme_betting = [r['extreme_betting_rate'] for r in results]

        logger.info(f"\nBehavioral Metrics:")
        logger.info(f"  Betting Aggressiveness: Mean={np.mean(betting_agg):.3f}, SD={np.std(betting_agg):.3f}")
        logger.info(f"  Loss Chasing Intensity: Mean={np.mean(loss_chasing):.3f}, SD={np.std(loss_chasing):.3f}")
        logger.info(f"  Extreme Betting Rate: Mean={np.mean(extreme_betting):.3f}, SD={np.std(extreme_betting):.3f}")

        # Near-Miss Effect (KEY METRIC)
        near_miss_chasing = [r['near_miss_loss_chasing'] for r in results]
        full_miss_chasing = [r['full_miss_loss_chasing'] for r in results]
        near_miss_effect = [r['near_miss_effect'] for r in results]

        logger.info(f"\nNear-Miss Analysis:")
        logger.info(f"  Near-Miss Loss Chasing: Mean={np.mean(near_miss_chasing):.3f}, SD={np.std(near_miss_chasing):.3f}")
        logger.info(f"  Full-Miss Loss Chasing: Mean={np.mean(full_miss_chasing):.3f}, SD={np.std(full_miss_chasing):.3f}")
        logger.info(f"  Near-Miss Effect (Δ): Mean={np.mean(near_miss_effect):.3f}, SD={np.std(near_miss_effect):.3f}")

        # Near-miss statistics
        near_miss_counts = [r['near_miss_count'] for r in results]
        near_miss_pcts = [r['near_miss_pct_of_losses'] for r in results]

        logger.info(f"\nNear-Miss Frequency:")
        logger.info(f"  Near-Miss Count: Mean={np.mean(near_miss_counts):.1f}, SD={np.std(near_miss_counts):.1f}")
        logger.info(f"  Near-Miss % of Losses: Mean={np.mean(near_miss_pcts):.1f}%, SD={np.std(near_miss_pcts):.1f}%")
        logger.info(f"  Expected: {self.near_miss_rate * 100:.0f}%")

        logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Near-Miss Enhancement Experiment")
    parser.add_argument('--model', type=str, required=True, choices=['llama', 'gemma', 'qwen'],
                        help='Model to use')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--bet-type', type=str, default='variable', choices=['fixed', 'variable'],
                        help='Betting type')
    parser.add_argument('--quick', action='store_true', help='Quick mode (80 games)')

    args = parser.parse_args()

    experiment = NearMissExperiment(args.model, args.gpu, args.bet_type)
    experiment.run_experiment(quick_mode=args.quick)


if __name__ == '__main__':
    main()
