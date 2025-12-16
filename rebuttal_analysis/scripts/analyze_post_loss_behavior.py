#!/usr/bin/env python3
"""
Analysis of post-loss behavior (behavioral gambler's fallacy)
Generates Figure 6: Post-Loss Risk Escalation
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Directories
RESULTS_DIR = Path('/data/llm_addiction/investment_choice_bet_constraint/results')
OUTPUT_DIR = Path('/home/ubuntu/llm_addiction/rebuttal_analysis')
FIGURES_DIR = OUTPUT_DIR / 'figures' / 'main'

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Model names
MODEL_NAMES = {
    'gpt4o_mini': 'GPT-4o-mini',
    'gpt41_mini': 'GPT-4.1-mini',
    'gemini_flash': 'Gemini-2.5-Flash',
    'claude_haiku': 'Claude-3.5-Haiku'
}

def analyze_post_loss_streaks():
    """Analyze decision patterns following loss streaks"""

    results = {
        'overall': defaultdict(lambda: {'option4': 0, 'total': 0}),
        'by_model': defaultdict(lambda: defaultdict(lambda: {'option4': 0, 'total': 0})),
        'by_streak': defaultdict(lambda: {'option4': 0, 'total': 0})
    }

    total_games = 0
    games_with_3plus_streak = 0

    for result_file in RESULTS_DIR.glob('*.json'):
        with open(result_file) as f:
            data = json.load(f)

        model = data.get('experiment_config', {}).get('model', 'unknown')
        games = data.get('results', [])

        for game in games:
            total_games += 1
            decisions = game.get('decisions', [])

            if len(decisions) < 4:  # Need at least 3 losses + 1 decision
                continue

            # Track consecutive losses
            loss_streak = 0
            had_3plus_streak = False

            for i, decision in enumerate(decisions):
                win = decision.get('win', False)

                if not win:
                    loss_streak += 1
                else:
                    loss_streak = 0

                # Check if we have 3+ loss streak and next decision exists
                if loss_streak >= 3 and i + 1 < len(decisions):
                    had_3plus_streak = True
                    next_decision = decisions[i + 1]
                    next_choice = next_decision.get('choice')

                    if next_choice:
                        # Record decision after streak
                        results['overall'][loss_streak]['total'] += 1
                        results['by_model'][model][loss_streak]['total'] += 1
                        results['by_streak'][loss_streak]['total'] += 1

                        if next_choice == 4:
                            results['overall'][loss_streak]['option4'] += 1
                            results['by_model'][model][loss_streak]['option4'] += 1
                            results['by_streak'][loss_streak]['option4'] += 1

            if had_3plus_streak:
                games_with_3plus_streak += 1

    return results, total_games, games_with_3plus_streak

def create_figure6_post_loss(results, total_games, games_with_3plus_streak):
    """Create Figure 6: Post-Loss Risk Escalation"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: Option 4 selection by streak length
    ax = axes[0]

    streak_lengths = sorted([k for k in results['by_streak'].keys() if k >= 3])
    option4_rates = []
    totals = []

    for streak in streak_lengths:
        data = results['by_streak'][streak]
        total = data['total']
        option4 = data['option4']
        rate = (option4 / total * 100) if total > 0 else 0

        option4_rates.append(rate)
        totals.append(total)

    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(streak_lengths)))
    bars = ax.bar(range(len(streak_lengths)), option4_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels
    for i, (bar, rate, n) in enumerate(zip(bars, option4_rates, totals)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 2,
                f'{rate:.1f}%\n(n={n})',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.axhline(y=50, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='Baseline (50%)')
    ax.set_xlabel('Consecutive Loss Streak Length', fontsize=14)
    ax.set_ylabel('Option 4 Selection Rate (%)', fontsize=14)
    ax.set_title('(A) Post-Loss Risk Escalation by Streak Length', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(range(len(streak_lengths)))
    ax.set_xticklabels([f'{s}+ losses' for s in streak_lengths])
    ax.set_ylim(0, 100)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    # Panel B: By model
    ax = axes[1]

    models = sorted(results['by_model'].keys())
    x = np.arange(len(models))
    width = 0.25

    streak_3 = []
    streak_4 = []
    streak_5plus = []

    for model in models:
        # 3 losses
        data3 = results['by_model'][model].get(3, {'option4': 0, 'total': 0})
        rate3 = (data3['option4'] / data3['total'] * 100) if data3['total'] > 0 else 0
        streak_3.append(rate3)

        # 4 losses
        data4 = results['by_model'][model].get(4, {'option4': 0, 'total': 0})
        rate4 = (data4['option4'] / data4['total'] * 100) if data4['total'] > 0 else 0
        streak_4.append(rate4)

        # 5+ losses (aggregate)
        option4_5plus = sum(results['by_model'][model].get(s, {'option4': 0})['option4'] for s in range(5, 11))
        total_5plus = sum(results['by_model'][model].get(s, {'total': 0})['total'] for s in range(5, 11))
        rate_5plus = (option4_5plus / total_5plus * 100) if total_5plus > 0 else 0
        streak_5plus.append(rate_5plus)

    ax.bar(x - width, streak_3, width, label='3 Losses', color='#e74c3c', alpha=0.7)
    ax.bar(x, streak_4, width, label='4 Losses', color='#c0392b', alpha=0.7)
    ax.bar(x + width, streak_5plus, width, label='5+ Losses', color='#922b21', alpha=0.7)

    ax.axhline(y=50, color='gray', linestyle='--', linewidth=2, alpha=0.5)
    ax.set_xlabel('Model', fontsize=14)
    ax.set_ylabel('Option 4 Selection Rate (%)', fontsize=14)
    ax.set_title('(B) Post-Loss Risk Escalation by Model', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_NAMES.get(m, m) for m in models], rotation=15, ha='right')
    ax.set_ylim(0, 100)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save
    output_png = FIGURES_DIR / 'fig6_post_loss_behavior.png'
    output_pdf = FIGURES_DIR / 'fig6_post_loss_behavior.pdf'
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_pdf, bbox_inches='tight')
    plt.close()

    print(f"✓ Figure 6 saved: {output_png}")
    print(f"✓ Figure 6 saved: {output_pdf}")

    # Print summary statistics
    print("\n" + "="*80)
    print("POST-LOSS BEHAVIOR SUMMARY")
    print("="*80)
    print(f"Total games analyzed: {total_games:,}")
    print(f"Games with 3+ loss streaks: {games_with_3plus_streak:,} ({games_with_3plus_streak/total_games*100:.1f}%)")

    # Overall 3+ streak statistics
    total_3plus = sum(results['overall'][s]['total'] for s in range(3, 11))
    option4_3plus = sum(results['overall'][s]['option4'] for s in range(3, 11))
    rate_3plus = (option4_3plus / total_3plus * 100) if total_3plus > 0 else 0

    print(f"\nPost-loss decisions after 3+ streaks: {total_3plus:,}")
    print(f"Option 4 selections: {option4_3plus:,} ({rate_3plus:.1f}%)")
    print(f"Behavioral gambler's fallacy rate: {rate_3plus:.1f}%")

def main():
    print("="*80)
    print("POST-LOSS BEHAVIOR ANALYSIS")
    print("="*80)

    print("\n1. Analyzing post-loss streaks...")
    results, total_games, games_with_3plus_streak = analyze_post_loss_streaks()

    print("\n2. Creating Figure 6 (Post-Loss Risk Escalation)...")
    create_figure6_post_loss(results, total_games, games_with_3plus_streak)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
