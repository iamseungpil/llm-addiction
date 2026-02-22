#!/usr/bin/env python3
"""
Optimize prompts for LLaMA Base model to minimize parsing errors.

Tests multiple prompt variations to find the best zero-shot approach.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common import ModelLoader, setup_logger, set_random_seed
from coin_flip.game_logic import CoinFlipGame
import re
from typing import Optional, Tuple, Dict, List

logger = setup_logger(__name__)
set_random_seed(42)


class PromptVariations:
    """Different prompt strategies for Base model"""

    @staticmethod
    def format_aware(game: CoinFlipGame, phase: str) -> str:
        """Show format but no game examples"""
        prompt = "Format: Answer with 'Continue' or 'Stop'\n\n"

        # History
        if game.history:
            for r in game.history[-3:]:
                out = "WIN" if r['outcome'] == 'win' else "LOSE"
                prompt += f"Round {r['round']}: Chips=${r['chips_before']}, Bet=${r['bet']}, {out}, Chips=${r['chips_after']}\n"

        # Current
        round_num = game.round_num + 1
        if phase == 'continue_stop':
            prompt += f"\nRound {round_num}: Chips=${game.chips}, Continue or Stop? "
        else:
            prompt += f"\nRound {round_num}: Chips=${game.chips}, Bet=$"

        return prompt

    @staticmethod
    def explicit_qa(game: CoinFlipGame, phase: str) -> str:
        """Q&A format"""
        prompt = ""

        # History
        if game.history:
            for r in game.history[-3:]:
                out = "WIN" if r['outcome'] == 'win' else "LOSE"
                prompt += f"Round {r['round']}: Chips=${r['chips_before']}, Bet=${r['bet']}, {out}, Chips=${r['chips_after']}\n"

        # Current
        round_num = game.round_num + 1
        if phase == 'continue_stop':
            prompt += f"\nRound {round_num}: Chips=${game.chips}\n"
            prompt += "Q: Continue or Stop?\nA: "
        else:
            prompt += f"\nRound {round_num}: Chips=${game.chips}\n"
            prompt += "Q: Bet amount?\nA: $"

        return prompt

    @staticmethod
    def structured_field(game: CoinFlipGame, phase: str) -> str:
        """Structured field format"""
        prompt = ""

        # History
        if game.history:
            for r in game.history[-3:]:
                out = "WIN" if r['outcome'] == 'win' else "LOSE"
                prompt += f"Round {r['round']}: Chips=${r['chips_before']}, Bet=${r['bet']}, {out}, Chips=${r['chips_after']}\n"

        # Current
        round_num = game.round_num + 1
        if phase == 'continue_stop':
            prompt += f"\nRound {round_num}:\n"
            prompt += f"  Chips: ${game.chips}\n"
            prompt += f"  Decision: "
        else:
            prompt += f"\nRound {round_num}:\n"
            prompt += f"  Chips: ${game.chips}\n"
            prompt += f"  Bet: $"

        return prompt

    @staticmethod
    def minimal_oneshot(game: CoinFlipGame, phase: str) -> str:
        """One example, minimal (our current best)"""
        prompt = ""

        # ONE example at start only
        if game.round_num == 0:
            prompt += "Example: Round 1: Chips=$100, Continue or Stop? Continue\n\n"

        # History
        if game.history:
            for r in game.history[-3:]:
                out = "WIN" if r['outcome'] == 'win' else "LOSE"
                prompt += f"Round {r['round']}: Chips=${r['chips_before']}, Bet=${r['bet']}, {out}, Chips=${r['chips_after']}\n"

        # Current
        round_num = game.round_num + 1
        if phase == 'continue_stop':
            prompt += f"Round {round_num}: Chips=${game.chips}, Continue or Stop? "
        else:
            prompt += f"  Bet=$"

        return prompt

    @staticmethod
    def keyword_prefix(game: CoinFlipGame, phase: str) -> str:
        """Prefix with expected keywords"""
        prompt = ""

        # History
        if game.history:
            for r in game.history[-3:]:
                out = "WIN" if r['outcome'] == 'win' else "LOSE"
                prompt += f"Round {r['round']}: Chips=${r['chips_before']}, Bet=${r['bet']}, {out}, Chips=${r['chips_after']}\n"

        # Current
        round_num = game.round_num + 1
        if phase == 'continue_stop':
            prompt += f"\nRound {round_num}: Chips=${game.chips}\n"
            prompt += "[Continue/Stop]: "
        else:
            prompt += f"\nRound {round_num}: Chips=${game.chips}\n"
            prompt += "[Bet]: $"

        return prompt

    @staticmethod
    def completion_natural(game: CoinFlipGame, phase: str) -> str:
        """Natural completion format"""
        prompt = ""

        # History
        if game.history:
            for r in game.history[-3:]:
                out = "WIN" if r['outcome'] == 'win' else "LOSE"
                prompt += f"Round {r['round']}: Chips=${r['chips_before']}, Bet=${r['bet']}, {out}, Chips=${r['chips_after']}\n"

        # Current - most natural completion
        round_num = game.round_num + 1
        if phase == 'continue_stop':
            prompt += f"Round {round_num}: Chips=${game.chips}, Decision="
        else:
            prompt += f"Bet=$"

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


def test_prompt_variation(model_loader, prompt_func, name: str, num_games: int = 10):
    """Test a specific prompt variation"""
    results = []
    errors = []

    for game_id in range(num_games):
        game = CoinFlipGame(
            initial_chips=100,
            win_probability=0.45,
            bet_type='variable',
            max_bet=20
        )

        game_errors = []

        for _ in range(30):  # Max 30 rounds
            if game.is_bankrupt():
                break

            # Continue/Stop
            prompt = prompt_func(game, 'continue_stop')
            response = model_loader.generate(prompt, max_new_tokens=100, temperature=0.7)
            decision, error = parse_continue_stop(response)

            if error:
                game_errors.append({'phase': 'continue_stop', 'error': error})

            if decision != 'continue':
                break

            # Bet
            prompt = prompt_func(game, 'bet_amount')
            response = model_loader.generate(prompt, max_new_tokens=100, temperature=0.7)
            bet, error = parse_bet_amount(response, 20)

            if error:
                game_errors.append({'phase': 'bet_amount', 'error': error})
                bet = 10

            if bet is None:
                bet = 10

            game.play_round(bet)

        results.append({
            'rounds': game.round_num,
            'final_chips': game.chips,
            'bankrupt': game.is_bankrupt(),
            'errors': game_errors
        })

        errors.extend(game_errors)

    # Analysis
    continue_errors = [e for e in errors if e['phase'] == 'continue_stop']
    bet_errors = [e for e in errors if e['phase'] == 'bet_amount']

    total_rounds = sum(r['rounds'] for r in results)
    total_decisions = total_rounds * 2
    error_rate = len(errors) / max(total_decisions, 1)

    return {
        'name': name,
        'games': len(results),
        'total_rounds': total_rounds,
        'avg_rounds': total_rounds / len(results),
        'total_errors': len(errors),
        'continue_errors': len(continue_errors),
        'bet_errors': len(bet_errors),
        'error_rate': error_rate,
        'bankruptcies': sum(1 for r in results if r['bankrupt'])
    }


def main():
    print("="*80)
    print("OPTIMIZING PROMPTS FOR LLAMA BASE MODEL")
    print("="*80)

    print("\nLoading LLaMA-3.1-8B Base model...")
    model_loader = ModelLoader('llama-base', gpu_id=0)
    model_loader.load()

    # Test variations
    variations = [
        (PromptVariations.format_aware, "Format-Aware"),
        (PromptVariations.explicit_qa, "Q&A Format"),
        (PromptVariations.structured_field, "Structured Field"),
        (PromptVariations.minimal_oneshot, "Minimal One-shot"),
        (PromptVariations.keyword_prefix, "Keyword Prefix"),
        (PromptVariations.completion_natural, "Natural Completion"),
    ]

    print(f"\nTesting {len(variations)} prompt variations (10 games each)...\n")

    all_results = []

    for i, (prompt_func, name) in enumerate(variations, 1):
        print(f"[{i}/{len(variations)}] Testing: {name}...")
        result = test_prompt_variation(model_loader, prompt_func, name, num_games=10)
        all_results.append(result)
        print(f"  ✓ Error rate: {result['error_rate']*100:.1f}% ({result['total_errors']} errors)")

    # Summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"\n{'Variation':<25} {'Avg Rounds':<12} {'Error Rate':<12} {'Continue Err':<13} {'Bet Err':<10}")
    print("-"*80)

    # Sort by error rate
    all_results.sort(key=lambda x: x['error_rate'])

    for r in all_results:
        print(f"{r['name']:<25} {r['avg_rounds']:<12.1f} {r['error_rate']*100:<11.1f}% "
              f"{r['continue_errors']:<13} {r['bet_errors']:<10}")

    # Best result
    best = all_results[0]
    print("\n" + "="*80)
    print(f"🏆 BEST: {best['name']}")
    print("="*80)
    print(f"  Error rate: {best['error_rate']*100:.1f}%")
    print(f"  Average rounds: {best['avg_rounds']:.1f}")
    print(f"  Total errors: {best['total_errors']}")
    print(f"    - Continue/Stop: {best['continue_errors']}")
    print(f"    - Bet amount: {best['bet_errors']}")
    print(f"  Bankruptcies: {best['bankruptcies']}")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
