#!/usr/bin/env python3
"""
Create 4-panel figure for Finding 1: Betting Aggressiveness
Investment Choice Extended CoT Experiment Analysis
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Style settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11

# Colors matching Image #3 style
COLORS = {
    'Option 1': '#2ecc71',  # Green (safe)
    'Option 2': '#3498db',  # Blue (moderate)
    'Option 3': '#f39c12',  # Orange (risky)
    'Option 4': '#e74c3c',  # Red (extreme)
}

BAR_COLORS = {
    'fixed': '#3498db',
    'variable': '#e74c3c',
    'BASE': '#3498db',
    'G': '#2ecc71',
    'M': '#f39c12',
    'GM': '#e74c3c',
}

def load_data():
    """Load all experiment results"""
    results_dir = '/data/llm_addiction/investment_choice_extended_cot/results'
    all_games = []

    for f in sorted(os.listdir(results_dir)):
        if f.endswith('.json'):
            with open(os.path.join(results_dir, f)) as fp:
                data = json.load(fp)
                config = data['experiment_config']
                for game in data['results']:
                    game['model'] = config['model']
                    game['constraint'] = int(config['bet_constraint'])
                    game['bet_type'] = config['bet_type']
                    all_games.append(game)

    return all_games

def calculate_stats(all_games):
    """Calculate bankruptcy and option distribution stats"""
    stats = {
        'bankruptcy_by_bet_type': {},
        'bankruptcy_by_prompt': {},
        'option_dist_by_bet_type': {},
        'option_dist_by_prompt': {},
    }

    # Bankruptcy by bet type
    for bt in ['fixed', 'variable']:
        games = [g for g in all_games if g['bet_type'] == bt]
        bankrupt = sum(1 for g in games if g['final_balance'] <= 0)
        stats['bankruptcy_by_bet_type'][bt] = 100 * bankrupt / len(games) if games else 0

    # Bankruptcy by prompt
    for prompt in ['BASE', 'G', 'M', 'GM']:
        games = [g for g in all_games if g['prompt_condition'] == prompt]
        bankrupt = sum(1 for g in games if g['final_balance'] <= 0)
        stats['bankruptcy_by_prompt'][prompt] = 100 * bankrupt / len(games) if games else 0

    # Option distribution by bet type
    for bt in ['fixed', 'variable']:
        games = [g for g in all_games if g['bet_type'] == bt]
        choices = defaultdict(int)
        total = 0
        for g in games:
            for d in g.get('decisions', []):
                choices[d['choice']] += 1
                total += 1
        stats['option_dist_by_bet_type'][bt] = {
            i: 100 * choices[i] / total if total > 0 else 0 for i in [1, 2, 3, 4]
        }

    # Option distribution by prompt
    for prompt in ['BASE', 'G', 'M', 'GM']:
        games = [g for g in all_games if g['prompt_condition'] == prompt]
        choices = defaultdict(int)
        total = 0
        for g in games:
            for d in g.get('decisions', []):
                choices[d['choice']] += 1
                total += 1
        stats['option_dist_by_prompt'][prompt] = {
            i: 100 * choices[i] / total if total > 0 else 0 for i in [1, 2, 3, 4]
        }

    return stats

def create_figure(stats):
    """Create 4-panel figure"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Panel (a): Bankruptcy Rate by Betting Type
    ax = axes[0, 0]
    bet_types = ['Fixed', 'Variable']
    bankruptcy_rates = [stats['bankruptcy_by_bet_type']['fixed'],
                        stats['bankruptcy_by_bet_type']['variable']]
    colors = [BAR_COLORS['fixed'], BAR_COLORS['variable']]

    bars = ax.bar(bet_types, bankruptcy_rates, color=colors, width=0.6, edgecolor='white', linewidth=1.5)
    ax.set_ylabel('Bankruptcy Rate (%)')
    ax.set_title('(a) Bankruptcy Rate by Betting Type')
    ax.set_ylim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar, rate in zip(bars, bankruptcy_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

    # Panel (b): Bankruptcy Rate by Prompt Condition
    ax = axes[0, 1]
    prompts = ['BASE', 'G', 'M', 'GM']
    bankruptcy_rates = [stats['bankruptcy_by_prompt'][p] for p in prompts]
    colors = [BAR_COLORS[p] for p in prompts]

    bars = ax.bar(prompts, bankruptcy_rates, color=colors, width=0.6, edgecolor='white', linewidth=1.5)
    ax.set_ylabel('Bankruptcy Rate (%)')
    ax.set_title('(b) Bankruptcy Rate by Prompt Condition')
    ax.set_ylim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar, rate in zip(bars, bankruptcy_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

    # Panel (c): Option Distribution by Betting Type (stacked bar)
    ax = axes[1, 0]
    bet_types = ['Fixed', 'Variable']
    x = np.arange(len(bet_types))
    width = 0.6

    bottom = np.zeros(len(bet_types))
    for opt in [1, 2, 3, 4]:
        values = [stats['option_dist_by_bet_type']['fixed'][opt],
                  stats['option_dist_by_bet_type']['variable'][opt]]
        bars = ax.bar(x, values, width, bottom=bottom, label=f'Option {opt}',
                      color=COLORS[f'Option {opt}'], edgecolor='white', linewidth=0.5)

        # Add percentage labels for significant portions
        for i, (val, b) in enumerate(zip(values, bottom)):
            if val > 8:  # Only label if > 8%
                ax.text(x[i], b + val/2, f'{val:.0f}%', ha='center', va='center',
                        fontsize=10, fontweight='bold', color='white')
        bottom += values

    ax.set_ylabel('Distribution (%)')
    ax.set_title('(c) Option Distribution by Betting Type')
    ax.set_xticks(x)
    ax.set_xticklabels(bet_types)
    ax.set_ylim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', fontsize=9)

    # Panel (d): Option Distribution by Prompt Condition (stacked bar)
    ax = axes[1, 1]
    prompts = ['BASE', 'G', 'M', 'GM']
    x = np.arange(len(prompts))
    width = 0.6

    bottom = np.zeros(len(prompts))
    for opt in [1, 2, 3, 4]:
        values = [stats['option_dist_by_prompt'][p][opt] for p in prompts]
        bars = ax.bar(x, values, width, bottom=bottom, label=f'Option {opt}',
                      color=COLORS[f'Option {opt}'], edgecolor='white', linewidth=0.5)

        for i, (val, b) in enumerate(zip(values, bottom)):
            if val > 8:
                ax.text(x[i], b + val/2, f'{val:.0f}%', ha='center', va='center',
                        fontsize=10, fontweight='bold', color='white')
        bottom += values

    ax.set_ylabel('Distribution (%)')
    ax.set_title('(d) Option Distribution by Prompt Condition')
    ax.set_xticks(x)
    ax.set_xticklabels(prompts)
    ax.set_ylim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', fontsize=9)

    plt.suptitle('Investment Choice Extended CoT Experiment Analysis\n(6,000 games across 4 models)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig

def main():
    print("Loading data...")
    all_games = load_data()
    print(f"Loaded {len(all_games)} games")

    print("Calculating statistics...")
    stats = calculate_stats(all_games)

    print("\nStats summary:")
    print(f"Bankruptcy by bet type: {stats['bankruptcy_by_bet_type']}")
    print(f"Bankruptcy by prompt: {stats['bankruptcy_by_prompt']}")

    print("\nCreating figure...")
    fig = create_figure(stats)

    # Save figure
    output_dir = '/home/ubuntu/llm_addiction/rebuttal_analysis/figures'
    os.makedirs(output_dir, exist_ok=True)

    fig.savefig(f'{output_dir}/finding1_investment_choice_extended.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(f'{output_dir}/finding1_investment_choice_extended.pdf',
                bbox_inches='tight', facecolor='white')

    print(f"\nFigure saved to {output_dir}/finding1_investment_choice_extended.png")
    plt.close()

if __name__ == '__main__':
    main()
