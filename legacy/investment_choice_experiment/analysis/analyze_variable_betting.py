#!/usr/bin/env python3
"""
Analyze variable betting amounts across different models and conditions
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def analyze_variable_betting():
    """Analyze betting amounts in variable betting experiments"""

    results_dir = Path('/data/llm_addiction/investment_choice_experiment/results')

    # Find all variable betting result files
    variable_files = list(results_dir.glob('*_variable_*.json'))

    print("=" * 80)
    print("VARIABLE BETTING ANALYSIS")
    print("=" * 80)

    for file_path in sorted(variable_files):
        print(f"\n📄 File: {file_path.name}")

        with open(file_path) as f:
            data = json.load(f)

        model = data['experiment_config']['model']
        results = data['results']

        print(f"   Model: {model}")
        print(f"   Total games: {len(results)}")

        # Collect betting data
        all_bets = []
        bets_by_condition = defaultdict(list)
        bets_by_choice = defaultdict(list)
        bets_by_round = defaultdict(list)

        for game in results:
            condition = game['prompt_condition']

            for decision in game['decisions']:
                bet = decision['bet']
                choice = decision['choice']
                round_num = decision['round']
                balance_before = decision['balance_before']

                all_bets.append(bet)
                bets_by_condition[condition].append(bet)
                bets_by_choice[choice].append(bet)
                bets_by_round[round_num].append(bet)

        # Overall statistics
        print(f"\n   Overall Statistics:")
        print(f"      Total decisions: {len(all_bets)}")
        print(f"      Mean bet: ${np.mean(all_bets):.2f}")
        print(f"      Median bet: ${np.median(all_bets):.2f}")
        print(f"      Std bet: ${np.std(all_bets):.2f}")
        print(f"      Min bet: ${np.min(all_bets):.2f}")
        print(f"      Max bet: ${np.max(all_bets):.2f}")

        # By condition
        print(f"\n   By Condition:")
        for condition in ['BASE', 'G', 'M', 'GM']:
            if condition in bets_by_condition:
                bets = bets_by_condition[condition]
                print(f"      {condition:4s}: Mean=${np.mean(bets):6.2f}, "
                      f"Median=${np.median(bets):6.2f}, "
                      f"Std=${np.std(bets):6.2f} "
                      f"(n={len(bets)})")

        # By choice
        print(f"\n   By Choice:")
        for choice in [1, 2, 3, 4]:
            if choice in bets_by_choice:
                bets = bets_by_choice[choice]
                print(f"      Option {choice}: Mean=${np.mean(bets):6.2f}, "
                      f"Median=${np.median(bets):6.2f}, "
                      f"Std=${np.std(bets):6.2f} "
                      f"(n={len(bets)})")

        # By round
        print(f"\n   By Round:")
        for round_num in sorted(bets_by_round.keys())[:5]:  # First 5 rounds
            bets = bets_by_round[round_num]
            print(f"      Round {round_num:2d}: Mean=${np.mean(bets):6.2f}, "
                  f"Median=${np.median(bets):6.2f}, "
                  f"Std=${np.std(bets):6.2f} "
                  f"(n={len(bets)})")

        print("\n" + "-" * 80)

    # Cross-model comparison
    print("\n" + "=" * 80)
    print("CROSS-MODEL COMPARISON")
    print("=" * 80)

    model_stats = {}

    for file_path in sorted(variable_files):
        with open(file_path) as f:
            data = json.load(f)

        model = data['experiment_config']['model']
        results = data['results']

        all_bets = []
        for game in results:
            for decision in game['decisions']:
                all_bets.append(decision['bet'])

        model_stats[model] = {
            'mean': np.mean(all_bets),
            'median': np.median(all_bets),
            'std': np.std(all_bets),
            'n': len(all_bets)
        }

    print(f"\n{'Model':<20} {'Mean Bet':>12} {'Median Bet':>12} {'Std':>10} {'N':>8}")
    print("-" * 80)

    for model in sorted(model_stats.keys()):
        stats = model_stats[model]
        print(f"{model:<20} ${stats['mean']:>10.2f} ${stats['median']:>10.2f} "
              f"${stats['std']:>8.2f} {stats['n']:>8,}")

    print("=" * 80)


if __name__ == '__main__':
    analyze_variable_betting()
