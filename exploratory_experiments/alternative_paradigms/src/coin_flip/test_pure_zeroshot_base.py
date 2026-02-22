#!/usr/bin/env python3
"""
Test PURE zero-shot (no examples, no instructions) with Base model

Just the minimal completion format to see what happens.
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


def build_pure_zeroshot_prompt(game: CoinFlipGame, phase: str) -> str:
    """
    Absolutely minimal prompt - just the completion line.
    NO examples, NO instructions, NO explanations.
    """
    prompt = ""

    # Show history (minimal)
    if game.history:
        for result in game.history[-3:]:
            outcome = "WIN" if result['outcome'] == 'win' else "LOSE"
            prompt += f"Round {result['round']}: Chips=${result['chips_before']}, Bet=${result['bet']}, {outcome}, Chips=${result['chips_after']}\n"

    # Current round
    round_num = game.round_num + 1

    if phase == 'continue_stop':
        prompt += f"Round {round_num}: Chips=${game.chips}, Continue or Stop? "

    elif phase == 'bet_amount':
        prompt += f"Round {round_num}: Chips=${game.chips}, Bet=$"

    return prompt


def parse_continue_stop(response: str) -> Tuple[Optional[str], str]:
    """Parse response, return (decision, error_type)"""
    response = response.strip()

    if not response:
        return None, "empty"

    first_line = response.split('\n')[0].strip().lower()

    if first_line == 'continue':
        return 'continue', None
    if first_line == 'stop':
        return 'stop', None

    if first_line.startswith('continue'):
        return 'continue', None
    if first_line.startswith('stop'):
        return 'stop', None

    first_word = response.split()[0].lower() if response.split() else ''
    if first_word == 'continue':
        return 'continue', None
    if first_word == 'stop':
        return 'stop', None

    return None, f"no_match:{first_line[:30]}"


def parse_bet_amount(response: str, max_chips: int) -> Tuple[Optional[int], str]:
    """Parse bet amount, return (amount, error_type)"""
    response = response.strip()

    if not response:
        return None, "empty"

    # First number
    numbers = re.findall(r'\b(\d+)\b', response[:50])
    if not numbers:
        return None, "no_number"

    bet = int(numbers[0])

    if bet < 1 or bet > 20:
        return None, f"out_of_range:{bet}"

    return bet, None


def run_test_games(model_loader, num_games=10):
    """Run test games with pure zero-shot"""
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
            prompt = build_pure_zeroshot_prompt(game, 'continue_stop')
            response = model_loader.generate(prompt, max_new_tokens=50, temperature=0.7)
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
            prompt = build_pure_zeroshot_prompt(game, 'bet_amount')
            response = model_loader.generate(prompt, max_new_tokens=50, temperature=0.7)
            bet, error = parse_bet_amount(response, game.chips)

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
    print("PURE ZERO-SHOT TEST (Base Model)")
    print("="*80)
    print("\nLoading LLaMA-3.1-8B Base model...")

    model_loader = ModelLoader('llama-base', gpu_id=0)
    model_loader.load()

    print("\nRunning 10 test games with PURE zero-shot (no examples, no instructions)...")
    results, errors = run_test_games(model_loader, num_games=10)

    # Analysis
    total_rounds = sum(r['rounds'] for r in results)
    total_games = len(results)
    bankruptcies = sum(1 for r in results if r['bankrupt'])

    continue_errors = [e for e in errors if e['phase'] == 'continue_stop']
    bet_errors = [e for e in errors if e['phase'] == 'bet_amount']

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
    print(f"  Error rate: {100*len(errors)/max(total_rounds*2, 1):.1f}%")

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
