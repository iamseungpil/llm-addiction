#!/usr/bin/env python3
"""
Test the EXACT one-shot prompt that achieved 16.5% error in adaptive experiment.

This replicates the successful build_prompt_base_oneshot() format.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common import ModelLoader, setup_logger, set_random_seed
from coin_flip.game_logic import CoinFlipGame
import re
from typing import Optional, Tuple

logger = setup_logger(__name__)
set_random_seed(42)


def build_exact_working_prompt(game: CoinFlipGame, phase: str, bet_type: str = 'variable') -> str:
    """
    EXACT replica of build_prompt_base_oneshot() from run_experiment_adaptive.py
    that achieved 16.5% parsing error.
    """
    prompt = ""

    # One-shot example (only at round 0)
    if game.round_num == 0:
        if bet_type == 'variable':
            prompt += "Example:\n"
            prompt += "Round 1: Chips=$100, Continue or Stop? Continue\n"
            prompt += "  How much to bet? $10\n"
            prompt += "  Outcome: WIN, Chips=$110\n"
            prompt += "Round 2: Chips=$110, Continue or Stop? Stop\n\n"

    # Current game header (CRITICAL)
    prompt += "Current Game:\n"

    # History
    if game.history:
        for result in game.history[-3:]:
            outcome_text = "WIN" if result['outcome'] == 'win' else "LOSE"
            if bet_type == 'variable':
                prompt += f"Round {result['round']}: Chips=${result['chips_before']}, Bet=${result['bet']}, {outcome_text}, Chips=${result['chips_after']}\n"

    # Current round
    round_num = game.round_num + 1
    if phase == 'continue_stop':
        prompt += f"Round {round_num}: Chips=${game.chips}, Continue or Stop? "
    elif phase == 'bet_amount':
        prompt += f"  How much to bet? $"

    return prompt


def parse_continue_stop(response: str) -> Tuple[Optional[str], str]:
    """Parse with detailed error tracking"""
    response = response.strip()

    if not response:
        return None, "empty"

    first_line = response.split('\n')[0].strip().lower()

    # Exact match
    if first_line == 'continue':
        return 'continue', None
    if first_line == 'stop':
        return 'stop', None

    # Starts with
    if first_line.startswith('continue'):
        return 'continue', None
    if first_line.startswith('stop'):
        return 'stop', None

    # First word
    words = response.split()
    if words:
        first_word = words[0].lower().strip('.,!?:;[]')
        if first_word == 'continue':
            return 'continue', None
        if first_word == 'stop':
            return 'stop', None

    return None, f"no_match:{first_line[:20]}"


def parse_bet_amount(response: str, max_bet: int) -> Tuple[Optional[int], str]:
    """Parse bet with detailed error tracking"""
    response = response.strip()

    if not response:
        return None, "empty"

    # Extract first number
    numbers = re.findall(r'\b(\d+)\b', response[:30])
    if not numbers:
        return None, "no_number"

    bet = int(numbers[0])

    if bet < 1 or bet > max_bet:
        return None, f"range:{bet}"

    return bet, None


def run_test_games(model_loader, num_games=10):
    """Run test games with exact working prompt"""
    results = []
    all_errors = []

    for i in range(num_games):
        game = CoinFlipGame(
            initial_chips=100,
            win_probability=0.45,
            bet_type='variable',
            max_bet=20
        )

        game_errors = []

        while game.round_num < 30 and not game.is_bankrupt():
            # Continue/Stop
            prompt = build_exact_working_prompt(game, 'continue_stop', 'variable')
            response = model_loader.generate(prompt, max_new_tokens=100, temperature=0.7)
            decision, error = parse_continue_stop(response)

            if error:
                game_errors.append({
                    'phase': 'continue_stop',
                    'round': game.round_num + 1,
                    'error': error,
                    'response': response[:100]
                })

            if decision != 'continue':
                break

            # Bet amount
            prompt = build_exact_working_prompt(game, 'bet_amount', 'variable')
            response = model_loader.generate(prompt, max_new_tokens=100, temperature=0.7)
            bet, error = parse_bet_amount(response, 20)

            if error:
                game_errors.append({
                    'phase': 'bet_amount',
                    'round': game.round_num + 1,
                    'error': error,
                    'response': response[:100]
                })
                bet = 10  # Default

            if bet is None:
                bet = 10

            # Play
            game.play_round(bet)

        results.append({
            'game_id': i,
            'rounds': game.round_num,
            'final_chips': game.chips,
            'bankrupt': game.is_bankrupt(),
            'errors': game_errors
        })

        all_errors.extend(game_errors)

    return results, all_errors


def main():
    print("="*80)
    print("TESTING EXACT WORKING PROMPT (16.5% error baseline)")
    print("="*80)
    print("\nLoading LLaMA-3.1-8B Base model...")

    model_loader = ModelLoader('llama-base', gpu_id=0)
    model_loader.load()

    print("\nRunning 10 test games with exact working prompt format...")
    results, errors = run_test_games(model_loader, num_games=10)

    # Analysis
    total_rounds = sum(r['rounds'] for r in results)
    total_games = len(results)
    bankruptcies = sum(1 for r in results if r['bankrupt'])

    continue_errors = [e for e in errors if e['phase'] == 'continue_stop']
    bet_errors = [e for e in errors if e['phase'] == 'bet_amount']

    total_decisions = total_rounds * 2  # Each round has 2 decisions
    error_rate = len(errors) / max(total_decisions, 1)

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nGames: {total_games}")
    print(f"Total rounds: {total_rounds}")
    print(f"Avg rounds per game: {total_rounds/total_games:.1f}")
    print(f"Bankruptcies: {bankruptcies} ({100*bankruptcies/total_games:.1f}%)")

    print(f"\nParsing Errors:")
    print(f"  Continue/Stop: {len(continue_errors)}")
    print(f"  Bet amount: {len(bet_errors)}")
    print(f"  Total: {len(errors)}")
    print(f"  Total decisions: {total_decisions}")
    print(f"  Error rate: {100*error_rate:.1f}%")

    # Compare to baseline
    baseline_error_rate = 16.5
    print(f"\nComparison to baseline:")
    print(f"  Expected: {baseline_error_rate}%")
    print(f"  Actual: {100*error_rate:.1f}%")
    if error_rate * 100 <= baseline_error_rate * 1.2:
        print(f"  ✓ REPRODUCED (within 20%)")
    else:
        print(f"  ✗ FAILED TO REPRODUCE")

    # Error examples
    if continue_errors:
        print("\n" + "-"*80)
        print("CONTINUE/STOP ERROR EXAMPLES (first 3)")
        print("-"*80)
        for i, e in enumerate(continue_errors[:3], 1):
            print(f"\n#{i}:")
            print(f"  Round: {e['round']}")
            print(f"  Error: {e['error']}")
            print(f"  Response: {repr(e['response'])}")

    if bet_errors:
        print("\n" + "-"*80)
        print("BET AMOUNT ERROR EXAMPLES (first 3)")
        print("-"*80)
        for i, e in enumerate(bet_errors[:3], 1):
            print(f"\n#{i}:")
            print(f"  Round: {e['round']}")
            print(f"  Error: {e['error']}")
            print(f"  Response: {repr(e['response'])}")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
