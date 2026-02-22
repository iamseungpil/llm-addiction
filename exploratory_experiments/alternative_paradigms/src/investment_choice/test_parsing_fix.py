#!/usr/bin/env python3
"""
Test the fixed parsing logic for Variable betting.

Tests:
1. Choice and bet_amount are parsed correctly from "Option X, $Y" format
2. Fallback to 10% of constraint when amount not specified
3. Examples in round 0 teach the model the correct format
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import ModelLoader, setup_logger, set_random_seed
from investment_choice.run_experiment import InvestmentChoiceExperiment

logger = setup_logger(__name__)

def analyze_parsing(results):
    """Analyze parsing results to detect the bug."""
    stats = {
        'total_decisions': 0,
        'choice_eq_bet': 0,  # Bug indicator: choice == bet_amount
        'choice_ne_bet': 0,  # Good: choice != bet_amount
        'fallback_used': 0,
        'bet_distribution': Counter(),
        'choice_distribution': Counter(),
    }

    for game in results:
        for decision in game.get('decisions', []):
            choice = decision.get('choice')
            bet = decision.get('bet_amount')

            if choice and bet:
                stats['total_decisions'] += 1
                stats['choice_distribution'][choice] += 1
                stats['bet_distribution'][bet] += 1

                if choice == bet:
                    stats['choice_eq_bet'] += 1
                else:
                    stats['choice_ne_bet'] += 1

                # Check if fallback was used (10% of constraint)
                # For c30: 10% = 3, for c50: 10% = 5, for c70: 10% = 7
                constraint = int(game.get('bet_constraint', 30))
                fallback_bet = max(1, int(constraint * 0.1))
                if bet == fallback_bet:
                    stats['fallback_used'] += 1

    return stats


def print_parsing_analysis(stats, constraint):
    """Print parsing analysis results."""
    print(f"\n{'='*60}")
    print(f"PARSING ANALYSIS (constraint={constraint})")
    print(f"{'='*60}")

    total = stats['total_decisions']
    if total == 0:
        print("  No decisions found!")
        return

    print(f"  Total decisions: {total}")
    print(f"\n  Parsing Bug Check:")
    print(f"    choice == bet:     {stats['choice_eq_bet']:4d} ({stats['choice_eq_bet']/total*100:5.1f}%) ← BUG if high")
    print(f"    choice != bet:     {stats['choice_ne_bet']:4d} ({stats['choice_ne_bet']/total*100:5.1f}%) ← GOOD")
    print(f"    Fallback used:     {stats['fallback_used']:4d} ({stats['fallback_used']/total*100:5.1f}%)")

    print(f"\n  Choice distribution:")
    for choice in sorted(stats['choice_distribution'].keys()):
        count = stats['choice_distribution'][choice]
        print(f"    Option {choice}: {count:4d} ({count/total*100:5.1f}%)")

    print(f"\n  Bet amount distribution (top 10):")
    for bet, count in stats['bet_distribution'].most_common(10):
        print(f"    ${bet:3d}: {count:4d} ({count/total*100:5.1f}%)")

    # Bug detection
    bug_rate = stats['choice_eq_bet'] / total * 100
    if bug_rate > 50:
        print(f"\n  ⚠️  PARSING BUG DETECTED! (choice==bet in {bug_rate:.1f}% of cases)")
    else:
        print(f"\n  ✅ Parsing looks good (choice==bet in only {bug_rate:.1f}% of cases)")


def run_quick_test(gpu=0, constraint=30, n_games=10):
    """Run a quick test with the fixed parsing."""
    print("="*60)
    print(f"QUICK TEST: Variable Betting Parsing Fix")
    print(f"  GPU: {gpu}")
    print(f"  Constraint: c{constraint}")
    print(f"  Games per condition: {n_games}")
    print("="*60)

    # Create experiment (model loading is done internally)
    print("\nInitializing experiment...")

    experiment = InvestmentChoiceExperiment(
        model_name='llama',
        gpu_id=gpu,
        bet_type='variable',
        bet_constraint=str(constraint)
    )

    # Load model
    print("Loading LLaMA model...")
    experiment.load_model()

    # Manually run Variable betting games only (not using run_experiment)
    print(f"\nRunning {n_games} games per condition (BASE, G, M, GM)...")

    results = []
    game_id = 0
    conditions = ['BASE', 'G', 'M', 'GM']

    for condition in conditions:
        print(f"  Condition: {condition}")
        for rep in range(n_games):
            game_id += 1
            seed = game_id + 88888
            set_random_seed(seed)

            try:
                result = experiment.play_game(condition, game_id, seed)
                results.append(result)
            except Exception as e:
                print(f"    Game {game_id} failed: {e}")
                continue

    # Analyze parsing
    stats = analyze_parsing(results)
    print_parsing_analysis(stats, constraint)

    # Show sample responses
    print(f"\n{'='*60}")
    print("SAMPLE RESPONSES (first 5 decisions)")
    print(f"{'='*60}")

    for i, game in enumerate(results[:2]):  # First 2 games
        print(f"\nGame {i+1} (condition={game['prompt_condition']}):")
        for j, decision in enumerate(game.get('decisions', [])[:3]):  # First 3 decisions
            choice = decision.get('choice')
            bet = decision.get('bet_amount')
            response = decision.get('response', '')[:80]
            print(f"  Round {j+1}: choice={choice}, bet=${bet}")
            print(f"    Response: \"{response}...\"")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path('/scratch/x3415a02/data/llm-addiction/investment_choice')
    output_file = output_dir / f"parsing_fix_test_c{constraint}_{timestamp}.json"

    output_data = {
        'test': 'parsing_fix_validation',
        'timestamp': timestamp,
        'config': {
            'constraint': constraint,
            'n_games': n_games,
            'bet_type': 'variable',
        },
        'parsing_stats': {
            'total_decisions': stats['total_decisions'],
            'choice_eq_bet': stats['choice_eq_bet'],
            'choice_ne_bet': stats['choice_ne_bet'],
            'fallback_used': stats['fallback_used'],
            'bug_rate': stats['choice_eq_bet'] / max(stats['total_decisions'], 1) * 100,
        },
        'results': results,
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved: {output_file}")
    print(f"{'='*60}")

    return output_data


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--constraint', type=int, default=30)
    parser.add_argument('--n-games', type=int, default=20, help='Games per condition (total = n_games × 4)')
    args = parser.parse_args()

    run_quick_test(args.gpu, args.constraint, args.n_games)
