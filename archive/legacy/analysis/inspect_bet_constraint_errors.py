#!/usr/bin/env python3
"""
Inspect specific errors in bet constraint experiment
"""

import json
from pathlib import Path

RESULTS_DIR = Path('/data/llm_addiction/investment_choice_bet_constraint/results')

def inspect_errors():
    print("="*80)
    print("INSPECTING ERRORS IN BET CONSTRAINT EXPERIMENT")
    print("="*80)

    files_with_errors = [
        'claude_haiku_10_fixed_20251121_113053.json',
        'claude_haiku_30_fixed_20251121_173220.json',
        'gemini_flash_50_fixed_20251122_082823.json',
        'gpt4o_mini_10_variable_20251121_140601.json'
    ]

    for filename in files_with_errors:
        filepath = RESULTS_DIR / filename
        if not filepath.exists():
            continue

        print(f"\n{'='*80}")
        print(f"File: {filename}")
        print("="*80)

        with open(filepath) as f:
            data = json.load(f)

        results = data.get('results', [])
        error_count = 0

        for i, game in enumerate(results):
            # Check for various error indicators
            has_error = False
            error_details = []

            # Check for 'error' field
            if 'error' in game:
                has_error = True
                error_details.append(f"Error field: {game['error']}")

            # Check decisions for errors
            decisions = game.get('decisions', [])
            for j, decision in enumerate(decisions):
                if 'error' in decision:
                    has_error = True
                    error_details.append(f"Decision {j} error: {decision['error']}")

                # Check for parsing failures
                if decision.get('choice') is None and decision.get('response'):
                    has_error = True
                    error_details.append(f"Decision {j}: Failed to parse choice from response")

            # Check for incomplete games
            if len(decisions) == 0:
                has_error = True
                error_details.append("Game has no decisions")

            # Check for abnormal termination
            if game.get('status') == 'error':
                has_error = True
                error_details.append(f"Game status: error - {game.get('error_message', 'No message')}")

            if has_error:
                error_count += 1
                print(f"\n  Game {i} (ID: {game.get('game_id', 'N/A')}):")
                print(f"    Prompt: {game.get('prompt_condition', 'N/A')}")
                print(f"    Decisions: {len(decisions)}")
                print(f"    Errors:")
                for detail in error_details:
                    print(f"      - {detail}")

                # Show first decision if exists
                if decisions:
                    first_dec = decisions[0]
                    print(f"    First decision response: \"{first_dec.get('response', 'N/A')[:200]}...\"")

        print(f"\n  Total games with errors: {error_count}")

    print("\n" + "="*80)

if __name__ == '__main__':
    inspect_errors()
