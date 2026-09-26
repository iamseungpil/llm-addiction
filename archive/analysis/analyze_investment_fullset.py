#!/usr/bin/env python3
"""
Analyze Investment Choice Fullset Experiment Results
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def load_results(filepath):
    """Load experiment results from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def analyze_game_outcomes(results):
    """Analyze game outcomes and behavioral metrics"""
    stats = {
        'total_games': len(results),
        'bankruptcies': 0,
        'goals_achieved': 0,
        'avg_rounds': [],
        'avg_final_balance': [],
        'avg_max_balance': [],
        'by_condition': defaultdict(lambda: {
            'games': 0,
            'bankruptcies': 0,
            'goals_achieved': 0,
            'rounds': [],
            'final_balance': [],
            'max_balance': [],
            'total_invested': [],
            'total_profit': [],
            'choice_distribution': defaultdict(int)
        })
    }

    for game in results:
        condition = f"{game['bet_type']}"
        rounds_played = game.get('rounds_completed', 0)
        final_balance = game.get('final_balance', 0)

        # Get history data
        history = game.get('history', [])

        # Calculate max balance
        max_balance = final_balance
        for round_data in history:
            balance_after = round_data.get('balance_after', 0)
            if balance_after > max_balance:
                max_balance = balance_after

        total_invested = game.get('total_invested', 0)
        total_won = game.get('total_won', 0)

        # Choice distribution
        choice_counts = game.get('choice_counts', {})

        # Overall stats
        stats['avg_rounds'].append(rounds_played)
        stats['avg_final_balance'].append(final_balance)
        stats['avg_max_balance'].append(max_balance)

        if game.get('bankruptcy', False):
            stats['bankruptcies'] += 1

        if game.get('goal_achieved', False):
            stats['goals_achieved'] += 1

        # By condition stats
        cond_stats = stats['by_condition'][condition]
        cond_stats['games'] += 1
        cond_stats['rounds'].append(rounds_played)
        cond_stats['final_balance'].append(final_balance)
        cond_stats['max_balance'].append(max_balance)
        cond_stats['total_invested'].append(total_invested)
        cond_stats['total_profit'].append(total_won - total_invested)

        # Update choice distribution
        for choice, count in choice_counts.items():
            cond_stats['choice_distribution'][choice] += count

        if game.get('bankruptcy', False):
            cond_stats['bankruptcies'] += 1
        if game.get('goal_achieved', False):
            cond_stats['goals_achieved'] += 1

    return stats

def print_analysis(model_name, stats):
    """Print formatted analysis results"""
    print(f"\n{'='*60}")
    print(f"Investment Choice Analysis: {model_name}")
    print(f"{'='*60}\n")

    # Overall statistics
    print(f"Overall Statistics:")
    print(f"  Total Games: {stats['total_games']}")
    print(f"  Bankruptcies: {stats['bankruptcies']} ({stats['bankruptcies']/stats['total_games']*100:.1f}%)")
    print(f"  Goals Achieved: {stats['goals_achieved']} ({stats['goals_achieved']/stats['total_games']*100:.1f}%)")
    print(f"  Avg Rounds: {np.mean(stats['avg_rounds']):.1f} ± {np.std(stats['avg_rounds']):.1f}")
    print(f"  Avg Final Balance: ${np.mean(stats['avg_final_balance']):.1f} ± ${np.std(stats['avg_final_balance']):.1f}")
    print(f"  Avg Max Balance: ${np.mean(stats['avg_max_balance']):.1f} ± ${np.std(stats['avg_max_balance']):.1f}")

    # By condition
    print(f"\n{'='*60}")
    print(f"Statistics by Condition:")
    print(f"{'='*60}\n")

    for condition, cond_stats in sorted(stats['by_condition'].items()):
        n = cond_stats['games']
        print(f"{condition.upper()} Betting (n={n}):")
        print(f"  Bankruptcy Rate: {cond_stats['bankruptcies']}/{n} ({cond_stats['bankruptcies']/n*100:.1f}%)")
        print(f"  Goal Achievement: {cond_stats['goals_achieved']}/{n} ({cond_stats['goals_achieved']/n*100:.1f}%)")
        print(f"  Avg Rounds: {np.mean(cond_stats['rounds']):.1f} ± {np.std(cond_stats['rounds']):.1f}")
        print(f"  Avg Final Balance: ${np.mean(cond_stats['final_balance']):.1f} ± ${np.std(cond_stats['final_balance']):.1f}")
        print(f"  Avg Max Balance: ${np.mean(cond_stats['max_balance']):.1f} ± ${np.std(cond_stats['max_balance']):.1f}")
        print(f"  Avg Total Invested: ${np.mean(cond_stats['total_invested']):.1f} ± ${np.std(cond_stats['total_invested']):.1f}")
        print(f"  Avg Total Profit: ${np.mean(cond_stats['total_profit']):.1f} ± ${np.std(cond_stats['total_profit']):.1f}")

        # Choice distribution
        total_choices = sum(cond_stats['choice_distribution'].values())
        if total_choices > 0:
            print(f"  Choice Distribution:")
            for choice in ['1', '2', '3', '4']:
                count = cond_stats['choice_distribution'].get(choice, 0)
                pct = count / total_choices * 100 if total_choices > 0 else 0
                choice_name = {
                    '1': 'Safe/Stop',
                    '2': 'Moderate Risk',
                    '3': 'High Risk',
                    '4': 'Very High Risk'
                }.get(choice, choice)
                print(f"    Option {choice} ({choice_name}): {count} ({pct:.1f}%)")
        print()

def main():
    data_dir = Path("/home/jovyan/beomi/llm-addiction-data/investment_choice")

    # LLaMA results
    llama_file = data_dir / "llama_investment_unlimited_20260213_110822.json"
    print(f"Loading LLaMA results from: {llama_file}")
    llama_data = load_results(llama_file)

    print(f"\nLLaMA Experiment Config:")
    print(json.dumps(llama_data['config'], indent=2))

    llama_stats = analyze_game_outcomes(llama_data['results'])
    print_analysis("LLaMA-3.1-8B", llama_stats)

    # Gemma results
    gemma_file = data_dir / "gemma_investment_unlimited_20260213_133237.json"
    print(f"\n{'='*60}")
    print(f"Loading Gemma results from: {gemma_file}")
    gemma_data = load_results(gemma_file)

    if len(gemma_data['results']) == 0:
        print("\n⚠️  WARNING: Gemma results are EMPTY!")
        print("The experiment may have failed or not saved results properly.")
        print("Check the error logs for details.")
    else:
        gemma_stats = analyze_game_outcomes(gemma_data['results'])
        print_analysis("Gemma-2-9B", gemma_stats)

    print(f"\n{'='*60}")
    print("Analysis Complete")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
