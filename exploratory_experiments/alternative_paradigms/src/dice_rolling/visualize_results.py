#!/usr/bin/env python3
"""
Visualization for Dice Rolling Experiment
"""

import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_experiment(filepath):
    """Load experiment JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_comparison(data_dir, output_dir):
    """Create comparison plots"""

    files = {
        'BASE-Fixed': 'dice_gemma_fixed_10_20260223_225546.json',
        'BASE-Variable': 'dice_gemma_variable_50_20260223_232336.json',
        'GM-Variable': 'dice_gemma_variable_50_20260223_235239.json'
    }

    # Load data
    all_data = {}
    for condition, filename in files.items():
        filepath = data_dir / filename
        if filepath.exists():
            data = load_experiment(filepath)
            component_key = list(data['results'].keys())[0]
            all_data[condition] = data['results'][component_key]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Dice Rolling Experiment - Behavioral Comparison', fontsize=16, fontweight='bold')

    # 1. Bankruptcy rates
    ax = axes[0, 0]
    conditions = list(all_data.keys())
    bankruptcy_rates = [
        sum(1 for g in all_data[c] if g['bankrupt']) / len(all_data[c]) * 100
        for c in conditions
    ]
    colors = ['#3498db', '#e74c3c', '#9b59b6']
    bars = ax.bar(conditions, bankruptcy_rates, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Bankruptcy Rate (%)', fontsize=12)
    ax.set_title('Bankruptcy Rate by Condition', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, rate in zip(bars, bankruptcy_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 2. Avg bet by condition
    ax = axes[0, 1]
    avg_bets = []
    for condition in conditions:
        all_bets = [r['bet'] for g in all_data[condition] for r in g['rounds']]
        avg_bets.append(np.mean(all_bets))

    bars = ax.bar(conditions, avg_bets, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Average Bet ($)', fontsize=12)
    ax.set_title('Average Bet Size by Condition', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    for bar, bet in zip(bars, avg_bets):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'${bet:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 3. Loss chasing rates
    ax = axes[1, 0]
    loss_chase_rates = []
    for condition in conditions:
        loss_chase_events = 0
        total_loss_events = 0

        for game in all_data[condition]:
            rounds = game['rounds']
            for i in range(len(rounds) - 1):
                if rounds[i]['outcome'] == 'lose':
                    total_loss_events += 1
                    if rounds[i + 1]['bet'] > rounds[i]['bet']:
                        loss_chase_events += 1

        rate = (loss_chase_events / total_loss_events * 100) if total_loss_events > 0 else 0
        loss_chase_rates.append(rate)

    bars = ax.bar(conditions, loss_chase_rates, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Loss Chasing Rate (%)', fontsize=12)
    ax.set_title('Loss Chasing After Losses', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 50)
    ax.grid(axis='y', alpha=0.3)

    for bar, rate in zip(bars, loss_chase_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 4. Goal escalation vs bankruptcy (GM-Variable only)
    ax = axes[1, 1]
    gm_games = all_data['GM-Variable']

    escalation_bankruptcy = {}
    for game in gm_games:
        n_escalations = len(game.get('goal_escalations', []))
        if n_escalations not in escalation_bankruptcy:
            escalation_bankruptcy[n_escalations] = {'total': 0, 'bankrupt': 0}
        escalation_bankruptcy[n_escalations]['total'] += 1
        if game['bankrupt']:
            escalation_bankruptcy[n_escalations]['bankrupt'] += 1

    escalations = sorted(escalation_bankruptcy.keys())
    bankruptcy_by_esc = [
        (escalation_bankruptcy[n]['bankrupt'] / escalation_bankruptcy[n]['total'] * 100)
        for n in escalations
    ]
    game_counts = [escalation_bankruptcy[n]['total'] for n in escalations]

    bars = ax.bar(escalations, bankruptcy_by_esc, color='#9b59b6', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Number of Goal Escalations', fontsize=12)
    ax.set_ylabel('Bankruptcy Rate (%)', fontsize=12)
    ax.set_title('Bankruptcy by Goal Escalations (GM-Variable)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (bar, rate, count) in enumerate(zip(bars, bankruptcy_by_esc, game_counts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{rate:.0f}%\n(n={count})', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    output_path = output_dir / 'dice_rolling_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    # Create bet distribution plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Bet Distribution Comparison (Variable Betting)', fontsize=14, fontweight='bold')

    for idx, condition in enumerate(['BASE-Variable', 'GM-Variable']):
        ax = axes[idx]
        all_bets = [r['bet'] for g in all_data[condition] for r in g['rounds']]

        ax.hist(all_bets, bins=20, color=colors[idx + 1], alpha=0.7, edgecolor='black')
        ax.set_xlabel('Bet Size ($)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{condition} (median=${np.median(all_bets):.0f})', fontsize=12)
        ax.grid(axis='y', alpha=0.3)

        # Add vertical line for median
        ax.axvline(np.median(all_bets), color='red', linestyle='--', linewidth=2, label=f'Median: ${np.median(all_bets):.0f}')
        ax.legend()

    plt.tight_layout()
    output_path = output_dir / 'dice_rolling_bet_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def main():
    data_dir = Path('/scratch/x3415a02/data/llm-addiction/dice_rolling')
    output_dir = data_dir

    print("\nGenerating visualizations...")
    plot_comparison(data_dir, output_dir)

    print("\nVisualization complete!")
    print(f"Output directory: {output_dir}")

if __name__ == '__main__':
    main()
