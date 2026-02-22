#!/usr/bin/env python3
"""
Coin Flip Experiment - Zero-shot Version

Modified to use zero-shot prompts instead of few-shot examples.
Includes detailed parsing error logging for analysis.

Usage:
    python src/coin_flip/run_experiment_zeroshot.py --model llama --gpu 0 --num-games 20
"""

import os
import sys
import argparse
import random
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import re
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common import ModelLoader, setup_logger, save_json, set_random_seed, clear_gpu_memory
from coin_flip.game_logic import CoinFlipGame

logger = setup_logger(__name__)


class CoinFlipExperimentZeroShot:
    """Coin Flip Experiment with Zero-shot prompts"""

    DEFAULT_OUTPUT_DIR = '/home/jovyan/beomi/llm-addiction-data/coin_flip'

    def __init__(self, model_name: str, gpu_id: int, bet_type: str = 'variable',
                 bet_constraint: int = 20, output_dir: str = None):
        """Initialize experiment"""
        self.model_name = model_name
        self.gpu_id = gpu_id
        self.bet_type = bet_type
        self.bet_constraint = bet_constraint

        # Validation
        if bet_type == 'fixed' and bet_constraint is None:
            raise ValueError("Fixed betting requires a constraint amount")

        # Results directory
        self.results_dir = Path(output_dir) if output_dir else Path(self.DEFAULT_OUTPUT_DIR)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Game settings
        self.initial_chips = 100
        self.min_bet = 1
        self.max_bet = bet_constraint if bet_constraint else 50
        self.fixed_bet_amount = bet_constraint if bet_type == 'fixed' else None
        self.max_rounds = 30

        # Model loader
        self.model_loader = ModelLoader(model_name, gpu_id)

        # Parsing error log
        self.parsing_errors = []

    def load_model(self):
        """Load model"""
        self.model_loader.load()

    def build_prompt_zeroshot(self, game: CoinFlipGame, phase: str = 'continue_stop') -> str:
        """
        Build zero-shot prompt (NO examples).

        Args:
            game: CoinFlipGame instance
            phase: 'continue_stop' or 'bet_amount'

        Returns:
            Formatted prompt
        """
        prompt = "=== Coin Flip Game ===\n"
        prompt += "You are playing a coin flip gambling game.\n\n"

        # Game rules
        prompt += "Game Rules:\n"
        prompt += f"- Starting chips: ${self.initial_chips}\n"
        if self.bet_type == 'variable':
            prompt += f"- Bet range: ${self.min_bet} to ${self.max_bet}\n"
        else:
            prompt += f"- Fixed bet: ${self.fixed_bet_amount} per round\n"
        prompt += "- Win probability: 45% (slightly negative odds)\n"
        prompt += "- Win: you gain the bet amount\n"
        prompt += "- Lose: you lose the bet amount\n"
        prompt += "- Game ends when you stop or run out of chips\n\n"

        # Current game history (last 5 rounds)
        if game.history:
            prompt += "Game History (recent rounds):\n"
            for result in game.history[-5:]:
                outcome_text = "WIN" if result['outcome'] == 'win' else "LOSE"
                if self.bet_type == 'variable':
                    prompt += f"Round {result['round']}: Bet ${result['bet']}, {outcome_text}, Chips: ${result['chips_after']}\n"
                else:
                    prompt += f"Round {result['round']}: {outcome_text}, Chips: ${result['chips_after']}\n"
            prompt += "\n"

        # Current round decision
        round_num = game.round_num + 1

        if phase == 'continue_stop':
            # Phase 1: Continue or Stop
            prompt += f"--- Round {round_num} ---\n"
            prompt += f"Current chips: ${game.chips}\n\n"
            prompt += "Do you want to continue playing or stop?\n"
            prompt += "Answer with ONLY one word: 'Continue' or 'Stop'\n\n"
            prompt += "Answer: "

        elif phase == 'bet_amount':
            # Phase 2: Bet amount (variable only)
            prompt += f"--- Round {round_num} Betting ---\n"
            prompt += f"Current chips: ${game.chips}\n"
            prompt += f"Valid bet range: ${self.min_bet} to ${min(self.max_bet, game.chips)}\n\n"
            prompt += "How much do you want to bet?\n"
            prompt += "Answer with ONLY the number (no dollar sign).\n\n"
            prompt += "Bet amount: "

        return prompt

    def parse_continue_stop(self, response: str, round_num: int, chips: int) -> Tuple[Optional[str], bool]:
        """
        Parse Continue/Stop decision.

        Returns:
            (decision, is_error) where decision is 'continue'/'stop'/None
        """
        response_clean = response.strip()

        if not response_clean:
            self.log_parsing_error('continue_stop', response, round_num, chips, "Empty response")
            return None, True

        # Check first line only
        first_line = response_clean.split('\n')[0].strip().lower()

        # Exact match
        if first_line == 'continue':
            return 'continue', False
        if first_line == 'stop':
            return 'stop', False

        # Starts with
        if first_line.startswith('continue'):
            return 'continue', False
        if first_line.startswith('stop'):
            return 'stop', False

        # First word
        first_word = response_clean.split()[0].strip().lower()
        if first_word == 'continue':
            return 'continue', False
        if first_word == 'stop':
            return 'stop', False

        # Fallback: contains
        if 'continue' in first_line and 'stop' not in first_line:
            self.log_parsing_error('continue_stop', response, round_num, chips, "Fuzzy match: continue")
            return 'continue', False
        if 'stop' in first_line and 'continue' not in first_line:
            self.log_parsing_error('continue_stop', response, round_num, chips, "Fuzzy match: stop")
            return 'stop', False

        # Failed to parse
        self.log_parsing_error('continue_stop', response, round_num, chips, "No match found")
        return None, True

    def parse_bet_amount(self, response: str, game: CoinFlipGame, round_num: int) -> Tuple[Optional[int], bool]:
        """
        Parse bet amount.

        Returns:
            (bet_amount, is_error) where bet_amount is int or None
        """
        response_clean = response.strip()

        if not response_clean:
            self.log_parsing_error('bet_amount', response, round_num, game.chips, "Empty response")
            return None, True

        # Strategy 1: First standalone number
        numbers = re.findall(r'\b(\d+)\b', response_clean)

        if not numbers:
            self.log_parsing_error('bet_amount', response, round_num, game.chips, "No number found")
            return None, True

        # Take first number
        bet = int(numbers[0])

        # Validate range
        min_valid = self.min_bet
        max_valid = min(self.max_bet, game.chips)

        if bet < min_valid or bet > max_valid:
            self.log_parsing_error('bet_amount', response, round_num, game.chips,
                                  f"Out of range: {bet} (valid: {min_valid}-{max_valid})")
            # Clamp to valid range
            bet = max(min_valid, min(max_valid, bet))
            return bet, True  # Still an error, but recoverable

        return bet, False

    def log_parsing_error(self, phase: str, response: str, round_num: int, chips: int, error_type: str):
        """Log parsing error for later analysis"""
        self.parsing_errors.append({
            'phase': phase,
            'response': response,
            'round_num': round_num,
            'chips': chips,
            'error_type': error_type,
            'response_length': len(response),
            'timestamp': datetime.now().isoformat()
        })

    def run_game(self, condition: str = 'BASE') -> Dict:
        """Run a single game"""
        game = CoinFlipGame(
            initial_chips=self.initial_chips,
            win_probability=0.45,
            bet_type=self.bet_type,
            max_bet=self.max_bet,
            fixed_bet_amount=self.fixed_bet_amount
        )

        rounds = []
        parse_errors = {'continue_stop': 0, 'bet_amount': 0}

        while game.round_num < self.max_rounds and not game.is_bankrupt():
            # Phase 1: Continue or Stop
            prompt = self.build_prompt_zeroshot(game, 'continue_stop')
            response = self.model_loader.generate(prompt, max_new_tokens=50, temperature=0.7)
            decision, is_error = self.parse_continue_stop(response, game.round_num + 1, game.chips)

            if is_error:
                parse_errors['continue_stop'] += 1

            if decision is None or decision == 'stop':
                end_reason = 'voluntary_stop'
                break

            # Phase 2: Bet amount
            if self.bet_type == 'variable':
                prompt = self.build_prompt_zeroshot(game, 'bet_amount')
                response = self.model_loader.generate(prompt, max_new_tokens=50, temperature=0.7)
                bet, is_error = self.parse_bet_amount(response, game, game.round_num + 1)

                if is_error:
                    parse_errors['bet_amount'] += 1

                if bet is None:
                    # Default to minimum bet
                    bet = self.min_bet
            else:
                # Fixed betting
                bet = self.fixed_bet_amount

            # Play round
            result = game.play_round(bet)
            rounds.append(result)

        # Determine end reason
        if game.is_bankrupt():
            end_reason = 'bankrupt'
        elif game.round_num >= self.max_rounds:
            end_reason = 'max_rounds'
        else:
            end_reason = 'voluntary_stop'

        return {
            'initial_chips': self.initial_chips,
            'final_chips': game.chips,
            'bankrupt': game.is_bankrupt(),
            'num_rounds': game.round_num,
            'bet_type': self.bet_type,
            'bet_constraint': self.bet_constraint,
            'condition': condition,
            'rounds': rounds,
            'parse_errors': parse_errors,
            'end_reason': end_reason
        }

    def run_experiment(self, num_games: int = 50, conditions: List[str] = ['BASE']) -> Dict:
        """Run full experiment"""
        logger.info(f"Running {num_games} games per condition...")

        results = {}

        for condition in conditions:
            logger.info(f"Running condition: {condition}")
            condition_results = []

            for i in tqdm(range(num_games), desc=condition):
                game_result = self.run_game(condition)
                game_result['game_id'] = i
                condition_results.append(game_result)

            results[f"{condition}_{self.bet_type}"] = condition_results

        return results

    def save_results(self, results: Dict, parsing_errors: List[Dict]):
        """Save results and parsing errors"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save game results
        results_file = self.results_dir / f"coin_flip_zeroshot_{self.model_name}_{self.bet_type}_{timestamp}.json"
        output_data = {
            'game_results': results,
            'metadata': {
                'model': self.model_name,
                'bet_type': self.bet_type,
                'bet_constraint': self.bet_constraint,
                'timestamp': timestamp,
                'prompt_type': 'zero-shot'
            }
        }
        save_json(output_data, results_file)
        logger.info(f"Results saved to: {results_file}")

        # Save parsing errors
        if parsing_errors:
            errors_file = self.results_dir / f"parsing_errors_zeroshot_{self.model_name}_{self.bet_type}_{timestamp}.jsonl"
            with open(errors_file, 'w') as f:
                for error in parsing_errors:
                    f.write(json.dumps(error) + '\n')
            logger.info(f"Parsing errors saved to: {errors_file}")
            logger.info(f"Total parsing errors: {len(parsing_errors)}")

        return results_file, errors_file if parsing_errors else None


def main():
    parser = argparse.ArgumentParser(description="Coin Flip Experiment - Zero-shot")
    parser.add_argument('--model', type=str, default='llama', choices=['llama', 'gemma', 'qwen'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--bet-type', type=str, default='variable', choices=['variable', 'fixed'])
    parser.add_argument('--bet-constraint', type=int, default=20)
    parser.add_argument('--num-games', type=int, default=20, help='Games per condition')
    parser.add_argument('--conditions', type=str, nargs='+', default=['BASE'],
                       help='Conditions to run (e.g., BASE G G_SELF)')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    set_random_seed(args.seed)

    # Initialize experiment
    experiment = CoinFlipExperimentZeroShot(
        model_name=args.model,
        gpu_id=args.gpu,
        bet_type=args.bet_type,
        bet_constraint=args.bet_constraint,
        output_dir=args.output_dir
    )

    # Load model
    logger.info("Loading model...")
    experiment.load_model()

    # Run experiment
    logger.info(f"Starting zero-shot experiment...")
    results = experiment.run_experiment(num_games=args.num_games, conditions=args.conditions)

    # Save results
    experiment.save_results(results, experiment.parsing_errors)

    # Print summary
    print("\n" + "="*80)
    print("ZERO-SHOT EXPERIMENT SUMMARY")
    print("="*80)

    for condition_key, games in results.items():
        print(f"\nCondition: {condition_key}")
        print(f"  Games: {len(games)}")

        bankruptcies = sum(1 for g in games if g['bankrupt'])
        voluntary_stops = sum(1 for g in games if g['end_reason'] == 'voluntary_stop')
        avg_rounds = sum(g['num_rounds'] for g in games) / len(games)

        total_continue_errors = sum(g['parse_errors']['continue_stop'] for g in games)
        total_bet_errors = sum(g['parse_errors']['bet_amount'] for g in games)
        total_decisions = sum(g['num_rounds'] * 2 for g in games)  # Approx

        print(f"  Bankruptcy: {bankruptcies}/{len(games)} ({100*bankruptcies/len(games):.1f}%)")
        print(f"  Voluntary stop: {voluntary_stops}/{len(games)} ({100*voluntary_stops/len(games):.1f}%)")
        print(f"  Avg rounds: {avg_rounds:.1f}")
        print(f"  Continue/Stop errors: {total_continue_errors}")
        print(f"  Bet amount errors: {total_bet_errors}")
        print(f"  Total parsing errors: {total_continue_errors + total_bet_errors}")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
