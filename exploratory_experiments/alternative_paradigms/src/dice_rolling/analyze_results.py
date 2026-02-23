#!/usr/bin/env python3
"""
Dice Rolling Experiment Analysis
Analyzes gambling behavior across 4 conditions:
- BASE-Fixed (no goal, fixed bet)
- BASE-Variable (no goal, variable bet)
- GM-Fixed (with goal, fixed bet)
- GM-Variable (with goal, variable bet)
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys

def load_experiment(filepath):
    """Load experiment JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def analyze_condition(games):
    """Analyze behavioral metrics for a condition"""
    metrics = {
        'n_games': len(games),
        'bankruptcy_rate': 0,
        'avg_rounds': [],
        'avg_final_chips': [],
        'avg_bets': [],
        'max_bets': [],
        'goal_escalations_count': [],
        'win_rate': [],
        'total_wagered': [],
        'net_profit': []
    }

    for game in games:
        # Bankruptcy
        if game['bankrupt']:
            metrics['bankruptcy_rate'] += 1

        # Rounds played
        metrics['avg_rounds'].append(game['num_rounds'])

        # Final chips
        metrics['avg_final_chips'].append(game['final_chips'])

        # Bet analysis
        bets = [r['bet'] for r in game['rounds']]
        metrics['avg_bets'].append(np.mean(bets))
        metrics['max_bets'].append(max(bets))

        # Goal escalations
        metrics['goal_escalations_count'].append(len(game.get('goal_escalations', [])))

        # Win rate
        wins = sum(1 for r in game['rounds'] if r['outcome'] == 'win')
        metrics['win_rate'].append(wins / len(game['rounds']))

        # Total wagered
        total_wagered = sum(r['bet'] for r in game['rounds'])
        metrics['total_wagered'].append(total_wagered)

        # Net profit
        net_profit = game['final_chips'] - game['initial_chips']
        metrics['net_profit'].append(net_profit)

    # Compute summary statistics
    summary = {
        'n_games': metrics['n_games'],
        'bankruptcy_rate': (metrics['bankruptcy_rate'] / metrics['n_games']) * 100,
        'avg_rounds': np.mean(metrics['avg_rounds']),
        'std_rounds': np.std(metrics['avg_rounds']),
        'avg_final_chips': np.mean(metrics['avg_final_chips']),
        'std_final_chips': np.std(metrics['avg_final_chips']),
        'avg_bet': np.mean(metrics['avg_bets']),
        'std_bet': np.std(metrics['avg_bets']),
        'avg_max_bet': np.mean(metrics['max_bets']),
        'total_goal_escalations': sum(metrics['goal_escalations_count']),
        'avg_goal_escalations': np.mean(metrics['goal_escalations_count']),
        'avg_win_rate': np.mean(metrics['win_rate']) * 100,
        'avg_total_wagered': np.mean(metrics['total_wagered']),
        'avg_net_profit': np.mean(metrics['net_profit'])
    }

    return summary

def print_comparison_table(results):
    """Print formatted comparison table"""
    print("\n" + "="*80)
    print("DICE ROLLING EXPERIMENT - BEHAVIORAL ANALYSIS")
    print("="*80)

    conditions = ['BASE-Fixed', 'BASE-Variable', 'GM-Fixed', 'GM-Variable']

    # Header
    print(f"\n{'Metric':<30}", end='')
    for cond in conditions:
        print(f"{cond:>15}", end='')
    print()
    print("-" * 90)

    # Metrics to display
    metrics = [
        ('N Games', 'n_games', '.0f'),
        ('Bankruptcy Rate (%)', 'bankruptcy_rate', '.1f'),
        ('Avg Rounds', 'avg_rounds', '.1f'),
        ('Avg Final Chips ($)', 'avg_final_chips', '.1f'),
        ('Avg Bet ($)', 'avg_bet', '.1f'),
        ('Avg Max Bet ($)', 'avg_max_bet', '.1f'),
        ('Total Goal Escalations', 'total_goal_escalations', '.0f'),
        ('Avg Goal Escalations', 'avg_goal_escalations', '.2f'),
        ('Win Rate (%)', 'avg_win_rate', '.1f'),
        ('Avg Total Wagered ($)', 'avg_total_wagered', '.1f'),
        ('Avg Net Profit ($)', 'avg_net_profit', '.1f')
    ]

    for label, key, fmt in metrics:
        print(f"{label:<30}", end='')
        for cond in conditions:
            if cond in results:
                value = results[cond][key]
                print(f"{value:>15{fmt}}", end='')
            else:
                print(f"{'N/A':>15}", end='')
        print()

    print("="*80)

def main():
    # Data directory
    data_dir = Path('/scratch/x3415a02/data/llm-addiction/dice_rolling')

    # Load all experiment files
    files = {
        'BASE-Fixed': 'dice_gemma_fixed_10_20260223_225546.json',
        'GM-Fixed': 'dice_gemma_variable_50_20260223_212250.json',
        'BASE-Variable': 'dice_gemma_variable_50_20260223_232336.json',
        'GM-Variable': 'dice_gemma_variable_50_20260223_235239.json'
    }

    results = {}

    for condition, filename in files.items():
        filepath = data_dir / filename
        if not filepath.exists():
            print(f"Warning: {filename} not found, skipping...")
            continue

        print(f"Loading {condition}: {filename}")
        data = load_experiment(filepath)

        # Get games from the first component (should be 'BASE' or 'GM')
        component_key = list(data['results'].keys())[0]
        games = data['results'][component_key]

        # Analyze
        results[condition] = analyze_condition(games)

        # Print metadata
        print(f"  Model: {data['metadata']['model']}")
        print(f"  Bet type: {data['metadata']['bet_type']}")
        print(f"  Components: {', '.join(data['metadata']['components'])}")
        print(f"  Games: {len(games)}")
        print()

    # Print comparison table
    print_comparison_table(results)

    # Key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)

    # Autonomy effect (Variable vs Fixed betting)
    if 'BASE-Fixed' in results and 'BASE-Variable' in results:
        base_diff = results['BASE-Variable']['bankruptcy_rate'] - results['BASE-Fixed']['bankruptcy_rate']
        print(f"\n1. AUTONOMY EFFECT (BASE condition):")
        print(f"   Variable betting: {results['BASE-Variable']['bankruptcy_rate']:.1f}% bankruptcy")
        print(f"   Fixed betting: {results['BASE-Fixed']['bankruptcy_rate']:.1f}% bankruptcy")
        print(f"   Difference: {base_diff:+.1f}% (Variable - Fixed)")

    # Goal effect
    if 'BASE-Variable' in results and 'GM-Variable' in results:
        goal_diff = results['GM-Variable']['bankruptcy_rate'] - results['BASE-Variable']['bankruptcy_rate']
        print(f"\n2. GOAL MANIPULATION EFFECT (Variable betting):")
        print(f"   With goal (GM): {results['GM-Variable']['bankruptcy_rate']:.1f}% bankruptcy")
        print(f"   Without goal (BASE): {results['BASE-Variable']['bankruptcy_rate']:.1f}% bankruptcy")
        print(f"   Difference: {goal_diff:+.1f}% (GM - BASE)")

    # Goal escalations
    if 'GM-Variable' in results:
        print(f"\n3. GOAL ESCALATION (GM-Variable):")
        print(f"   Total escalations: {results['GM-Variable']['total_goal_escalations']:.0f}")
        print(f"   Avg per game: {results['GM-Variable']['avg_goal_escalations']:.2f}")
        print(f"   Escalation rate: {results['GM-Variable']['avg_goal_escalations'] / results['GM-Variable']['avg_rounds'] * 100:.1f}% per round")

    # Betting behavior
    if 'BASE-Variable' in results and 'GM-Variable' in results:
        print(f"\n4. BETTING AGGRESSIVENESS (Variable conditions):")
        print(f"   BASE avg bet: ${results['BASE-Variable']['avg_bet']:.1f}")
        print(f"   GM avg bet: ${results['GM-Variable']['avg_bet']:.1f}")
        print(f"   BASE max bet: ${results['BASE-Variable']['avg_max_bet']:.1f}")
        print(f"   GM max bet: ${results['GM-Variable']['avg_max_bet']:.1f}")

    print("\n" + "="*80)

if __name__ == '__main__':
    main()
