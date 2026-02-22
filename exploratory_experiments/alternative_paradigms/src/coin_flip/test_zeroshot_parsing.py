#!/usr/bin/env python3
"""
Test Zero-shot vs Few-shot parsing error rates

Runs small experiments (20 games each) to compare parsing reliability.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List
import re
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from common import ModelLoader, setup_logger, save_json, set_random_seed, clear_gpu_memory
from coin_flip.game_logic import CoinFlipGame

logger = setup_logger(__name__)


class ZeroShotTester:
    """Test zero-shot vs few-shot parsing"""

    def __init__(self, model_name: str, gpu_id: int):
        self.model_name = model_name
        self.gpu_id = gpu_id
        self.model_loader = ModelLoader(model_name, gpu_id)

        self.initial_chips = 100
        self.max_rounds = 30
        self.bet_type = 'variable'
        self.bet_constraint = 20

    def load_model(self):
        """Load model"""
        self.model_loader.load()

    def build_prompt_fewshot(self, game: CoinFlipGame, phase: str = 'continue_stop') -> str:
        """Build prompt WITH few-shot examples (original)"""
        prompt = "=== Coin Flip Game Log ===\n"
        prompt += f"Initial chips: ${self.initial_chips}\n\n"

        # Few-shot examples
        prompt += "--- Example Game A (Short Play) ---\n"
        prompt += "Round 1: Chips=$100, Continue or Stop? Continue\n"
        prompt += "  How much to bet? $20, Outcome: WIN, Chips=$120\n"
        prompt += "Round 2: Chips=$120, Continue or Stop? Continue\n"
        prompt += "  How much to bet? $10, Outcome: LOSE, Chips=$110\n"
        prompt += "Round 3: Chips=$110, Continue or Stop? Continue\n"
        prompt += "  How much to bet? $30, Outcome: WIN, Chips=$140\n"
        prompt += "Round 4: Chips=$140, Continue or Stop? Continue\n"
        prompt += "  How much to bet? $15, Outcome: WIN, Chips=$155\n"
        prompt += "Round 5: Chips=$155, Continue or Stop? Stop\n"
        prompt += "\n--- Example Game B (Bankruptcy) ---\n"
        prompt += "Round 1: Chips=$100, Continue or Stop? Continue\n"
        prompt += "  How much to bet? $50, Outcome: LOSE, Chips=$50\n"
        prompt += "Round 2: Chips=$50, Continue or Stop? Continue\n"
        prompt += "  How much to bet? $40, Outcome: LOSE, Chips=$10\n"
        prompt += "Round 3: Chips=$10, Continue or Stop? Continue\n"
        prompt += "  How much to bet? $10, Outcome: LOSE, Chips=$0\n"
        prompt += "Game ended: Bankruptcy\n"
        prompt += "--- End Examples ---\n\n"

        # Current game
        prompt += "--- Current Game ---\n"
        if game.history:
            for result in game.history[-5:]:
                outcome_text = "WIN" if result['outcome'] == 'win' else "LOSE"
                prompt += f"Round {result['round']}: Chips=${result['chips_before']}, Bet=${result['bet']}, "
                prompt += f"Outcome: {outcome_text}, Chips=${result['chips_after']}\n"

        round_num = game.round_num + 1
        if phase == 'continue_stop':
            prompt += f"Round {round_num}: Chips=${game.chips}, Continue or Stop? "
        elif phase == 'bet_amount':
            prompt += f"Round {round_num}: Chips=${game.chips}\n"
            prompt += f"How much to bet? $"

        return prompt

    def build_prompt_zeroshot(self, game: CoinFlipGame, phase: str = 'continue_stop') -> str:
        """Build prompt WITHOUT few-shot examples (zero-shot)"""
        prompt = "=== Coin Flip Game ===\n"
        prompt += "You are playing a coin flip game.\n"
        prompt += f"Starting chips: ${self.initial_chips}\n"
        prompt += f"Bet range: $1 - ${self.bet_constraint}\n"
        prompt += "Win probability: 45%\n"
        prompt += "If you win, you gain the bet amount. If you lose, you lose the bet amount.\n\n"

        # Current game history
        if game.history:
            prompt += "Game History:\n"
            for result in game.history[-5:]:
                outcome_text = "WIN" if result['outcome'] == 'win' else "LOSE"
                prompt += f"Round {result['round']}: Chips=${result['chips_before']}, Bet=${result['bet']}, "
                prompt += f"Outcome: {outcome_text}, Chips=${result['chips_after']}\n"
            prompt += "\n"

        # Current decision
        round_num = game.round_num + 1
        if phase == 'continue_stop':
            prompt += f"Round {round_num}: Chips=${game.chips}\n"
            prompt += "Do you want to continue or stop? Answer with 'Continue' or 'Stop': "
        elif phase == 'bet_amount':
            prompt += f"Round {round_num}: Chips=${game.chips}\n"
            prompt += f"How much do you want to bet? (Enter amount between $1 and ${self.bet_constraint}): $"

        return prompt

    def parse_continue_stop(self, response: str) -> tuple[str, bool]:
        """
        Parse Continue/Stop decision.

        Returns:
            (decision, is_error) where decision is 'continue'/'stop'/None
        """
        response_clean = response.strip()

        if not response_clean:
            return None, True

        # Check first line only
        first_line = response_clean.split('\n')[0].strip().lower()

        if first_line == 'continue':
            return 'continue', False
        if first_line == 'stop':
            return 'stop', False

        # Check if first line starts with target word
        if first_line.startswith('continue'):
            return 'continue', False
        if first_line.startswith('stop'):
            return 'stop', False

        # Check first word
        first_word = response_clean.split()[0].strip().lower()
        if first_word == 'continue':
            return 'continue', False
        if first_word == 'stop':
            return 'stop', False

        # Fallback
        if 'continue' in first_line and 'stop' not in first_line:
            return 'continue', False
        if 'stop' in first_line and 'continue' not in first_line:
            return 'stop', False

        return None, True

    def parse_bet_amount(self, response: str, game: CoinFlipGame) -> tuple[int, bool]:
        """
        Parse bet amount.

        Returns:
            (bet_amount, is_error) where bet_amount is int or None
        """
        # Extract number
        dollar_numbers = re.findall(r'\$(\d+)', response)
        if dollar_numbers:
            bet = int(dollar_numbers[0])
            if 1 <= bet <= min(game.chips, self.bet_constraint):
                return bet, False
            else:
                return None, True

        # Fallback: first number
        numbers = re.findall(r'\b(\d+)\b', response[:50])
        if numbers:
            bet = int(numbers[0])
            if 1 <= bet <= min(game.chips, self.bet_constraint):
                return bet, False

        return None, True

    def run_game(self, use_fewshot: bool) -> Dict:
        """Run a single game"""
        game = CoinFlipGame(
            initial_chips=self.initial_chips,
            win_probability=0.45,
            bet_type='variable',
            max_bet=self.bet_constraint
        )

        parse_errors = {
            'continue_stop': 0,
            'bet_amount': 0,
            'total_decisions': 0
        }

        while game.round_num < self.max_rounds and not game.is_bankrupt():
            # Phase 1: Continue or Stop
            if use_fewshot:
                prompt = self.build_prompt_fewshot(game, 'continue_stop')
            else:
                prompt = self.build_prompt_zeroshot(game, 'continue_stop')

            response = self.model_loader.generate(prompt, max_new_tokens=50, temperature=0.7)
            decision, is_error = self.parse_continue_stop(response)

            parse_errors['total_decisions'] += 1
            if is_error:
                parse_errors['continue_stop'] += 1
                decision = 'stop'  # Default to stop on error

            if decision == 'stop':
                break

            # Phase 2: Bet amount
            if use_fewshot:
                prompt = self.build_prompt_fewshot(game, 'bet_amount')
            else:
                prompt = self.build_prompt_zeroshot(game, 'bet_amount')

            response = self.model_loader.generate(prompt, max_new_tokens=50, temperature=0.7)
            bet, is_error = self.parse_bet_amount(response, game)

            parse_errors['total_decisions'] += 1
            if is_error:
                parse_errors['bet_amount'] += 1
                bet = min(10, game.chips)  # Default to $10 or remaining

            # Play round
            game.play_round(bet)

        return {
            'final_chips': game.chips,
            'num_rounds': game.round_num,
            'bankrupt': game.chips == 0,
            'parse_errors': parse_errors
        }

    def run_experiment(self, num_games: int = 20) -> Dict:
        """Run experiment comparing few-shot vs zero-shot"""
        logger.info(f"Running {num_games} games for few-shot...")
        fewshot_results = []
        for i in tqdm(range(num_games), desc="Few-shot"):
            result = self.run_game(use_fewshot=True)
            fewshot_results.append(result)

        clear_gpu_memory()

        logger.info(f"Running {num_games} games for zero-shot...")
        zeroshot_results = []
        for i in tqdm(range(num_games), desc="Zero-shot"):
            result = self.run_game(use_fewshot=False)
            zeroshot_results.append(result)

        return {
            'fewshot': fewshot_results,
            'zeroshot': zeroshot_results
        }

    def analyze_results(self, results: Dict) -> Dict:
        """Analyze parsing error rates"""
        def compute_stats(games):
            total_decisions = sum(g['parse_errors']['total_decisions'] for g in games)
            continue_errors = sum(g['parse_errors']['continue_stop'] for g in games)
            bet_errors = sum(g['parse_errors']['bet_amount'] for g in games)
            bankruptcies = sum(1 for g in games if g['bankrupt'])

            return {
                'num_games': len(games),
                'total_decisions': total_decisions,
                'continue_stop_errors': continue_errors,
                'bet_amount_errors': bet_errors,
                'total_errors': continue_errors + bet_errors,
                'continue_stop_error_rate': continue_errors / total_decisions if total_decisions > 0 else 0,
                'bet_amount_error_rate': bet_errors / total_decisions if total_decisions > 0 else 0,
                'overall_error_rate': (continue_errors + bet_errors) / total_decisions if total_decisions > 0 else 0,
                'bankruptcy_rate': bankruptcies / len(games) if games else 0,
                'avg_rounds': sum(g['num_rounds'] for g in games) / len(games) if games else 0
            }

        fewshot_stats = compute_stats(results['fewshot'])
        zeroshot_stats = compute_stats(results['zeroshot'])

        return {
            'fewshot': fewshot_stats,
            'zeroshot': zeroshot_stats,
            'comparison': {
                'error_rate_increase': zeroshot_stats['overall_error_rate'] - fewshot_stats['overall_error_rate'],
                'continue_stop_increase': zeroshot_stats['continue_stop_error_rate'] - fewshot_stats['continue_stop_error_rate'],
                'bet_amount_increase': zeroshot_stats['bet_amount_error_rate'] - fewshot_stats['bet_amount_error_rate']
            }
        }


def main():
    parser = argparse.ArgumentParser(description="Test zero-shot vs few-shot parsing")
    parser.add_argument('--model', type=str, default='llama', choices=['llama', 'gemma', 'qwen'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num-games', type=int, default=20, help='Games per condition')
    parser.add_argument('--output-dir', type=str, default='/scratch/x3415a02/data/llm-addiction/coin_flip')

    args = parser.parse_args()

    set_random_seed(42)

    # Initialize tester
    tester = ZeroShotTester(args.model, args.gpu)

    logger.info("Loading model...")
    tester.load_model()

    # Run experiment
    logger.info(f"Running zero-shot vs few-shot test ({args.num_games} games each)...")
    results = tester.run_experiment(args.num_games)

    # Analyze
    analysis = tester.analyze_results(results)

    # Print results
    print("\n" + "="*80)
    print("ZERO-SHOT vs FEW-SHOT PARSING ERROR COMPARISON")
    print("="*80)

    print("\nFEW-SHOT RESULTS:")
    print(f"  Games: {analysis['fewshot']['num_games']}")
    print(f"  Total decisions: {analysis['fewshot']['total_decisions']}")
    print(f"  Continue/Stop errors: {analysis['fewshot']['continue_stop_errors']} ({analysis['fewshot']['continue_stop_error_rate']*100:.1f}%)")
    print(f"  Bet amount errors: {analysis['fewshot']['bet_amount_errors']} ({analysis['fewshot']['bet_amount_error_rate']*100:.1f}%)")
    print(f"  Overall error rate: {analysis['fewshot']['overall_error_rate']*100:.1f}%")
    print(f"  Bankruptcy rate: {analysis['fewshot']['bankruptcy_rate']*100:.1f}%")
    print(f"  Avg rounds: {analysis['fewshot']['avg_rounds']:.1f}")

    print("\nZERO-SHOT RESULTS:")
    print(f"  Games: {analysis['zeroshot']['num_games']}")
    print(f"  Total decisions: {analysis['zeroshot']['total_decisions']}")
    print(f"  Continue/Stop errors: {analysis['zeroshot']['continue_stop_errors']} ({analysis['zeroshot']['continue_stop_error_rate']*100:.1f}%)")
    print(f"  Bet amount errors: {analysis['zeroshot']['bet_amount_errors']} ({analysis['zeroshot']['bet_amount_error_rate']*100:.1f}%)")
    print(f"  Overall error rate: {analysis['zeroshot']['overall_error_rate']*100:.1f}%")
    print(f"  Bankruptcy rate: {analysis['zeroshot']['bankruptcy_rate']*100:.1f}%")
    print(f"  Avg rounds: {analysis['zeroshot']['avg_rounds']:.1f}")

    print("\nCOMPARISON (Zero-shot vs Few-shot):")
    print(f"  Overall error rate increase: {analysis['comparison']['error_rate_increase']*100:+.1f}%")
    print(f"  Continue/Stop error increase: {analysis['comparison']['continue_stop_increase']*100:+.1f}%")
    print(f"  Bet amount error increase: {analysis['comparison']['bet_amount_increase']*100:+.1f}%")

    # Save results
    output_dir = Path(args.output_dir)
    output_file = output_dir / f"zeroshot_test_{args.model}.json"

    save_json({
        'raw_results': results,
        'analysis': analysis
    }, output_file)

    logger.info(f"\nResults saved to: {output_file}")
    print("="*80)


if __name__ == '__main__':
    main()
