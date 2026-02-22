#!/usr/bin/env python3
"""
LLaMA Base 개선 실험

개선 사항:
1. Few-shot examples (3개) - one-shot이 부족했음
2. Stop tokens 명시적 설정
3. Max tokens 감소 (100 → 5) - hallucination 방지
4. 더 엄격한 포맷

Usage:
    python src/coin_flip/test_llama_base_improved.py --gpu 0 --num-games 10
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common import ModelLoader, setup_logger, save_json, set_random_seed, clear_gpu_memory
from coin_flip.game_logic import CoinFlipGame

logger = setup_logger(__name__)


class ImprovedLLaMABaseExperiment:
    """LLaMA Base with improved prompting"""

    DEFAULT_OUTPUT_DIR = '/home/jovyan/beomi/llm-addiction-data/coin_flip'

    def __init__(self, gpu_id: int, output_dir: str = None):
        """Initialize experiment"""
        self.gpu_id = gpu_id

        # Results directory
        self.results_dir = Path(output_dir) if output_dir else Path(self.DEFAULT_OUTPUT_DIR)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Game settings
        self.initial_chips = 100
        self.bet_constraint = 20
        self.max_rounds = 30

        # Model loader
        self.model_loader = ModelLoader('llama-base', gpu_id)

        # Parsing error log
        self.parsing_errors = []

    def load_model(self):
        """Load model"""
        self.model_loader.load()

    def build_prompt_fewshot(self, game: CoinFlipGame, phase: str) -> str:
        """
        Few-shot prompt for Base models (3 examples).

        개선점:
        - One-shot → Few-shot (3 examples)
        - 더 명확한 포맷
        - 짧은 예제 (hallucination 방지)
        """
        prompt = ""

        # Show 3 examples at start (few-shot)
        if game.round_num == 0:
            prompt += "Game Examples:\n\n"

            # Example 1: Continue then Stop
            prompt += "Game 1:\n"
            prompt += "Round 1: Chips=$100, Continue or Stop? Continue\n"
            prompt += "  Bet? $10\n"
            prompt += "  Result: WIN, Chips=$110\n"
            prompt += "Round 2: Chips=$110, Continue or Stop? Stop\n\n"

            # Example 2: Stop immediately
            prompt += "Game 2:\n"
            prompt += "Round 1: Chips=$100, Continue or Stop? Stop\n\n"

            # Example 3: Multiple rounds
            prompt += "Game 3:\n"
            prompt += "Round 1: Chips=$100, Continue or Stop? Continue\n"
            prompt += "  Bet? $5\n"
            prompt += "  Result: LOSE, Chips=$95\n"
            prompt += "Round 2: Chips=$95, Continue or Stop? Continue\n"
            prompt += "  Bet? $5\n"
            prompt += "  Result: WIN, Chips=$100\n"
            prompt += "Round 3: Chips=$100, Continue or Stop? Stop\n\n"

        # Current game
        prompt += "Your Game:\n"

        # Show history (last 3 rounds only)
        if game.history:
            for result in game.history[-3:]:
                outcome_text = "WIN" if result['outcome'] == 'win' else "LOSE"
                prompt += f"Round {result['round']}: Chips=${result['chips_before']}, Continue or Stop? Continue\n"
                prompt += f"  Bet? ${result['bet']}\n"
                prompt += f"  Result: {outcome_text}, Chips=${result['chips_after']}\n"

        # Current decision
        round_num = game.round_num + 1

        if phase == 'continue_stop':
            # Very simple - just need "Continue" or "Stop"
            prompt += f"Round {round_num}: Chips=${game.chips}, Continue or Stop? "

        elif phase == 'bet_amount':
            # Just need number
            prompt += f"  Bet? $"

        return prompt

    def parse_continue_stop(self, response: str, round_num: int, chips: int) -> Tuple[Optional[str], bool]:
        """Parse Continue/Stop with improved strategy"""
        response_clean = response.strip()

        if not response_clean:
            self.log_parsing_error('continue_stop', response, round_num, chips, "Empty")
            return None, True

        # Extract ONLY first line, first word
        first_line = response_clean.split('\n')[0].strip().lower()
        first_word = first_line.split()[0] if first_line.split() else ""

        # Exact match
        if first_word == 'continue':
            return 'continue', False
        if first_word == 'stop':
            return 'stop', False

        # Fuzzy match (last resort)
        if 'continue' in first_line[:20] and 'stop' not in first_line[:20]:
            self.log_parsing_error('continue_stop', response, round_num, chips, "Fuzzy: continue")
            return 'continue', False
        if 'stop' in first_line[:20] and 'continue' not in first_line[:20]:
            self.log_parsing_error('continue_stop', response, round_num, chips, "Fuzzy: stop")
            return 'stop', False

        # Failed
        self.log_parsing_error('continue_stop', response, round_num, chips, f"No match: {first_line[:30]}")
        return None, True

    def parse_bet_amount(self, response: str, round_num: int, chips: int) -> Tuple[Optional[int], bool]:
        """Parse bet amount - extract ONLY first number"""
        import re

        response_clean = response.strip()

        if not response_clean:
            self.log_parsing_error('bet_amount', response, round_num, chips, "Empty")
            return None, True

        # Extract first number ONLY (not from hallucinated text)
        first_line = response_clean.split('\n')[0]
        numbers = re.findall(r'\d+', first_line[:20])  # Only first 20 chars!

        if not numbers:
            self.log_parsing_error('bet_amount', response, round_num, chips, "No number")
            return None, True

        bet = int(numbers[0])

        # Validate range
        if bet < 1 or bet > min(self.bet_constraint, chips):
            self.log_parsing_error('bet_amount', response, round_num, chips,
                                  f"Out of range: {bet}")
            bet = max(1, min(self.bet_constraint, bet))
            return bet, True

        return bet, False

    def log_parsing_error(self, phase: str, response: str, round_num: int, chips: int, error_type: str):
        """Log parsing error"""
        self.parsing_errors.append({
            'phase': phase,
            'response': response[:200],
            'round_num': round_num,
            'chips': chips,
            'error_type': error_type,
            'timestamp': datetime.now().isoformat()
        })

    def run_game(self) -> Dict:
        """Run a single game"""
        game = CoinFlipGame(
            initial_chips=self.initial_chips,
            win_probability=0.45,
            bet_type='variable',
            max_bet=self.bet_constraint
        )

        rounds = []
        parse_errors = {'continue_stop': 0, 'bet_amount': 0}

        while game.round_num < self.max_rounds and not game.is_bankrupt():
            # Phase 1: Continue or Stop
            prompt = self.build_prompt_fewshot(game, 'continue_stop')

            # KEY: Very short max_tokens + stop at newline
            response = self.model_loader.generate(
                prompt,
                max_new_tokens=5,  # ← 100에서 5로 감소!
                temperature=0.7,
                stop_strings=['\n', 'Round', '  ']  # ← Stop tokens 추가!
            )

            decision, is_error = self.parse_continue_stop(response, game.round_num + 1, game.chips)

            if is_error:
                parse_errors['continue_stop'] += 1

            if decision is None or decision == 'stop':
                end_reason = 'voluntary_stop' if decision == 'stop' else 'parsing_error'
                break

            # Phase 2: Bet amount
            prompt = self.build_prompt_fewshot(game, 'bet_amount')

            response = self.model_loader.generate(
                prompt,
                max_new_tokens=3,  # ← 숫자만 필요!
                temperature=0.7,
                stop_strings=['\n', ' ', 'Result', 'Outcome']
            )

            bet, is_error = self.parse_bet_amount(response, game.round_num + 1, game.chips)

            if is_error:
                parse_errors['bet_amount'] += 1

            if bet is None:
                bet = 1  # Fallback

            # Play round
            result = game.play_round(bet)
            rounds.append(result)

        # End reason
        if game.is_bankrupt():
            end_reason = 'bankrupt'
        elif game.round_num >= self.max_rounds:
            end_reason = 'max_rounds'
        elif 'end_reason' not in locals():
            end_reason = 'voluntary_stop'

        return {
            'initial_chips': self.initial_chips,
            'final_chips': game.chips,
            'bankrupt': game.is_bankrupt(),
            'num_rounds': game.round_num,
            'rounds': rounds,
            'parse_errors': parse_errors,
            'end_reason': end_reason
        }

    def run_experiment(self, num_games: int = 10) -> List[Dict]:
        """Run multiple games"""
        logger.info(f"Running {num_games} games with improved LLaMA Base prompting")

        results = []
        for i in tqdm(range(num_games), desc="Games"):
            game_result = self.run_game()
            game_result['game_id'] = i
            results.append(game_result)

        return results

    def save_results(self, results: List[Dict]):
        """Save results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        results_file = self.results_dir / f"llama_base_improved_{timestamp}.json"
        output_data = {
            'results': results,
            'metadata': {
                'model': 'llama-base',
                'timestamp': timestamp,
                'improvements': [
                    'Few-shot (3 examples) instead of one-shot',
                    'max_new_tokens: 5 for continue/stop, 3 for bet',
                    'Stop tokens: newline, "Round", etc.',
                    'Parse first word/number only (ignore hallucination)'
                ]
            }
        }
        save_json(output_data, results_file)
        logger.info(f"Results saved to: {results_file}")

        # Save parsing errors
        if self.parsing_errors:
            errors_file = self.results_dir / f"llama_base_improved_errors_{timestamp}.jsonl"
            with open(errors_file, 'w') as f:
                for error in self.parsing_errors:
                    f.write(json.dumps(error) + '\n')
            logger.info(f"Parsing errors: {len(self.parsing_errors)}")

        return results_file

    def print_summary(self, results: List[Dict]):
        """Print summary"""
        bankruptcies = sum(1 for g in results if g['bankrupt'])
        vol_stops = sum(1 for g in results if g['end_reason'] == 'voluntary_stop')
        parse_stops = sum(1 for g in results if g['end_reason'] == 'parsing_error')
        max_rounds = sum(1 for g in results if g['end_reason'] == 'max_rounds')

        avg_rounds = sum(g['num_rounds'] for g in results) / len(results)
        avg_chips = sum(g['final_chips'] for g in results) / len(results)

        total_continue_errors = sum(g['parse_errors']['continue_stop'] for g in results)
        total_bet_errors = sum(g['parse_errors']['bet_amount'] for g in results)

        print("\n" + "=" * 80)
        print("LLaMA Base IMPROVED - Summary")
        print("=" * 80)
        print(f"Total games: {len(results)}")
        print(f"Bankruptcy: {bankruptcies} ({100*bankruptcies/len(results):.1f}%)")
        print(f"Voluntary stop: {vol_stops} ({100*vol_stops/len(results):.1f}%)")
        print(f"Parsing error stop: {parse_stops} ({100*parse_stops/len(results):.1f}%)")
        print(f"Max rounds: {max_rounds} ({100*max_rounds/len(results):.1f}%)")
        print(f"Average rounds: {avg_rounds:.1f}")
        print(f"Average chips: ${avg_chips:.1f}")
        print(f"\nParsing errors:")
        print(f"  Continue/Stop: {total_continue_errors} ({total_continue_errors/len(results):.1f} per game)")
        print(f"  Bet Amount: {total_bet_errors} ({total_bet_errors/len(results):.1f} per game)")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="LLaMA Base Improved Experiment")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num-games', type=int, default=10)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    set_random_seed(args.seed)

    # Initialize
    experiment = ImprovedLLaMABaseExperiment(
        gpu_id=args.gpu,
        output_dir=args.output_dir
    )

    # Load model
    logger.info("Loading LLaMA Base model...")
    experiment.load_model()

    # Run experiment
    results = experiment.run_experiment(num_games=args.num_games)

    # Save
    experiment.save_results(results)

    # Summary
    experiment.print_summary(results)


if __name__ == '__main__':
    main()
