#!/usr/bin/env python3
"""
Create bet constraint effect figure for Investment Choice experiment.
Shows bankruptcy rate by bet constraint for Fixed vs Variable betting.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.labelweight'] = 'bold'


def load_data():
    """Load Investment Choice data - DEDUPLICATED"""
    results_dir = '/data/llm_addiction/investment_choice_extended_cot/results'

    seen_combos = {}
    for f in sorted(os.listdir(results_dir), reverse=True):
        if f.endswith('.json'):
            with open(os.path.join(results_dir, f)) as fp:
                data = json.load(fp)
                config = data['experiment_config']
                key = (config['model'], config['bet_type'], config['bet_constraint'])
                if key not in seen_combos:
                    seen_combos[key] = data

    all_games = []
    for key, data in seen_combos.items():
        config = data['experiment_config']
        for game in data['results']:
            game['model'] = config['model']
            game['bet_type'] = config['bet_type']
            game['bet_constraint'] = int(config['bet_constraint'])
            all_games.append(game)

    print(f"Loaded {len(all_games)} games from {len(seen_combos)} unique combinations")
    return all_games


def calculate_stats_by_constraint(all_games):
    """Calculate bankruptcy rates by bet constraint and betting type"""
    stats = defaultdict(lambda: {'fixed': {'total': 0, 'bankrupt': 0},
                                  'variable': {'total': 0, 'bankrupt': 0}})

    for game in all_games:
        constraint = game['bet_constraint']
        bet_type = game['bet_type']
        is_bankrupt = game['final_balance'] <= 0

        stats[constraint][bet_type]['total'] += 1
        if is_bankrupt:
            stats[constraint][bet_type]['bankrupt'] += 1

    return stats


def create_figure(stats):
    """Create bet constraint effect figure"""
    fig, ax = plt.subplots(figsize=(10, 6))

    constraints = sorted(stats.keys())
    x = np.arange(len(constraints))
    width = 0.35

    fixed_rates = []
    variable_rates = []

    for c in constraints:
        fixed_rate = 100 * stats[c]['fixed']['bankrupt'] / stats[c]['fixed']['total'] if stats[c]['fixed']['total'] > 0 else 0
        variable_rate = 100 * stats[c]['variable']['bankrupt'] / stats[c]['variable']['total'] if stats[c]['variable']['total'] > 0 else 0
        fixed_rates.append(fixed_rate)
        variable_rates.append(variable_rate)

    bars1 = ax.bar(x - width/2, fixed_rates, width, label='Fixed', color='#27ae60', edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, variable_rates, width, label='Variable', color='#e74c3c', edgecolor='black', linewidth=1)

    ax.set_ylabel('Bankruptcy Rate (%)')
    ax.set_xlabel('Bet Constraint ($)')
    ax.set_title('Bankruptcy Rate by Bet Constraint\n(Investment Choice Experiment, 6,400 games)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'${c}' for c in constraints])
    ax.set_ylim(0, 100)
    ax.legend(loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, rate in zip(bars1, fixed_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    for bar, rate in zip(bars2, variable_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    return fig, constraints, fixed_rates, variable_rates


def main():
    print("=" * 60)
    print("BET CONSTRAINT EFFECT ANALYSIS")
    print("=" * 60)

    print("\nLoading data...")
    all_games = load_data()

    print("\nCalculating statistics by bet constraint...")
    stats = calculate_stats_by_constraint(all_games)

    print("\nCreating figure...")
    fig, constraints, fixed_rates, variable_rates = create_figure(stats)

    # Save
    output_dir = '/home/ubuntu/llm_addiction/rebuttal_analysis/figures'
    fig.savefig(f'{output_dir}/bet_constraint_effect.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(f'{output_dir}/bet_constraint_effect.pdf', bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\nFigure saved to {output_dir}/bet_constraint_effect.png")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'Constraint':<12} {'Fixed':<15} {'Variable':<15} {'Difference':<15}")
    print("-" * 60)
    for c, f, v in zip(constraints, fixed_rates, variable_rates):
        print(f"${c:<11} {f:>12.1f}% {v:>13.1f}% {v-f:>+13.1f}%")


if __name__ == '__main__':
    main()
