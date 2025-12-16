#!/usr/bin/env python3
"""
Create 3-panel figure for Prompt Effects Analysis
- Panel (a): Bankruptcy Rate by Prompt Condition
- Panel (b): Option Distribution by Prompt Condition (Option 4 in red, others grayscale)
- Panel (c): Goal Escalation by Prompt Condition
"""

import json
import os
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from typing import Optional

# Style settings - larger fonts for publication
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 16

# Colors for prompt conditions - G and GM highlighted
PROMPT_COLORS = {
    'BASE': '#b0b0b0',  # Gray
    'G': '#e67e22',     # Dark orange (highlight)
    'M': '#a8d4f0',     # Light sky blue
    'GM': '#e74c3c',    # Red (same as Option 4)
}

# Colors for Option distribution - Option 4 red, others grayscale
OPTION_COLORS = {
    'Option 1': '#a8d4f0',  # Light sky blue
    'Option 2': '#a0a0a0',  # Medium gray
    'Option 3': '#707070',  # Dark gray
    'Option 4': '#e74c3c',  # Red (highlight)
}


def load_data():
    """Load experiment results - DEDUPLICATED (latest file per combination only)"""
    results_dir = '/data/llm_addiction/investment_choice_extended_cot/results'

    # Group files by (model, bet_type, bet_constraint) and keep only latest
    seen_combos = {}
    for f in sorted(os.listdir(results_dir), reverse=True):  # Latest first
        if f.endswith('.json'):
            with open(os.path.join(results_dir, f)) as fp:
                data = json.load(fp)
                config = data['experiment_config']
                key = (config['model'], config['bet_type'], config['bet_constraint'])
                if key not in seen_combos:  # Keep first (latest) only
                    seen_combos[key] = (f, data)

    # Extract games from deduplicated files
    all_games = []
    for key, (filename, data) in seen_combos.items():
        config = data['experiment_config']
        for game in data['results']:
            game['model'] = config['model']
            game['constraint'] = int(config['bet_constraint'])
            game['bet_type'] = config['bet_type']
            all_games.append(game)

    print(f"Loaded {len(all_games)} games from {len(seen_combos)} unique combinations")
    return all_games


def extract_goal_amount(text: str) -> Optional[int]:
    """Extract goal/target amount from model response"""
    text_lower = text.lower()

    patterns = [
        r'(?:target|goal).*?\$(\d+)',
        r'(?:my target is|target is|goal is)\s*\$?(\d+)',
        r'(?:aim for|reach|achieve)\s*\$(\d+)',
        r'(?:target amount|goal amount).*?\$(\d+)',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        if matches:
            return int(matches[0])
    return None


def analyze_goal_escalation(game):
    """Check if a game has goal escalation (increasing targets)"""
    decisions = game.get('decisions', [])
    goals = []

    for decision in decisions:
        response = decision.get('response', '')
        goal = extract_goal_amount(response)
        if goal is not None:
            goals.append(goal)

    if len(goals) < 2:
        return False

    # Check if any goal is higher than a previous goal
    for i in range(1, len(goals)):
        if goals[i] > goals[i-1]:
            return True
    return False


def calculate_stats(all_games):
    """Calculate all statistics"""
    stats = {
        'bankruptcy_by_prompt': {},
        'option_dist_by_prompt': {},
        'goal_escalation_by_prompt': {},
    }

    prompts = ['BASE', 'G', 'M', 'GM']

    for prompt in prompts:
        games = [g for g in all_games if g['prompt_condition'] == prompt]

        # Bankruptcy rate
        bankrupt = sum(1 for g in games if g['final_balance'] <= 0)
        stats['bankruptcy_by_prompt'][prompt] = 100 * bankrupt / len(games) if games else 0

        # Option distribution
        choices = defaultdict(int)
        total = 0
        for g in games:
            for d in g.get('decisions', []):
                choices[d['choice']] += 1
                total += 1
        stats['option_dist_by_prompt'][prompt] = {
            i: 100 * choices[i] / total if total > 0 else 0 for i in [1, 2, 3, 4]
        }

        # Goal escalation rate
        escalation_count = sum(1 for g in games if analyze_goal_escalation(g))
        stats['goal_escalation_by_prompt'][prompt] = 100 * escalation_count / len(games) if games else 0

    return stats


def create_figure(stats):
    """Create 3-panel horizontal figure"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    prompts = ['BASE', 'G', 'M', 'GM']
    x = np.arange(len(prompts))
    width = 0.6

    # Panel (a): Bankruptcy Rate by Prompt Condition
    ax = axes[0]
    bankruptcy_rates = [stats['bankruptcy_by_prompt'][p] for p in prompts]
    colors = [PROMPT_COLORS[p] for p in prompts]

    bars = ax.bar(x, bankruptcy_rates, color=colors, width=width, edgecolor='black', linewidth=1)
    ax.set_ylabel('Bankruptcy Rate (%)')
    ax.set_title('(a) Bankruptcy Rate', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(prompts)
    ax.set_ylim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    for bar, rate in zip(bars, bankruptcy_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=16)

    # Panel (b): Option Distribution by Prompt Condition (stacked bar)
    ax = axes[1]

    bottom = np.zeros(len(prompts))
    for opt in [1, 2, 3, 4]:
        values = [stats['option_dist_by_prompt'][p][opt] for p in prompts]
        color = OPTION_COLORS[f'Option {opt}']
        bars = ax.bar(x, values, width, bottom=bottom, label=f'Option {opt}',
                      color=color, edgecolor='black', linewidth=1)

        # Add percentage labels for significant portions
        for i, (val, b) in enumerate(zip(values, bottom)):
            if val > 10:  # Only label if > 10%
                text_color = 'white' if opt == 4 else ('white' if opt == 3 else 'black')
                ax.text(x[i], b + val/2, f'{val:.0f}%', ha='center', va='center',
                        fontsize=14, fontweight='bold', color=text_color)
        bottom += values

    ax.set_ylabel('Distribution (%)')
    ax.set_title('(b) Option Distribution', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(prompts)
    ax.set_ylim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    legend = ax.legend(loc='center left', fontsize=14, framealpha=0.95, edgecolor='black')
    for text in legend.get_texts():
        text.set_fontweight('normal')

    # Panel (c): Goal Escalation by Prompt Condition
    ax = axes[2]
    escalation_rates = [stats['goal_escalation_by_prompt'][p] for p in prompts]
    colors = [PROMPT_COLORS[p] for p in prompts]

    bars = ax.bar(x, escalation_rates, color=colors, width=width, edgecolor='black', linewidth=1)
    ax.set_ylabel('Goal Escalation Rate (%)')
    ax.set_title('(c) Goal Escalation', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(prompts)
    ax.set_ylim(0, 70)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    for bar, rate in zip(bars, escalation_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=16)

    plt.suptitle('Prompt Effects on Gambling Behavior',
                 fontsize=24, fontweight='bold', y=0.96)
    plt.tight_layout()

    return fig


def main():
    print("Loading data...")
    all_games = load_data()
    print(f"Loaded {len(all_games)} games")

    print("\nCalculating statistics...")
    stats = calculate_stats(all_games)

    print("\n=== Statistics Summary ===")
    print(f"Bankruptcy by prompt: {stats['bankruptcy_by_prompt']}")
    print(f"Goal escalation by prompt: {stats['goal_escalation_by_prompt']}")
    print(f"Option distribution by prompt:")
    for p in ['BASE', 'G', 'M', 'GM']:
        print(f"  {p}: {stats['option_dist_by_prompt'][p]}")

    print("\nCreating figure...")
    fig = create_figure(stats)

    # Save figure
    output_dir = '/home/ubuntu/llm_addiction/rebuttal_analysis/figures'
    os.makedirs(output_dir, exist_ok=True)

    fig.savefig(f'{output_dir}/prompt_effects_3panel.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(f'{output_dir}/prompt_effects_3panel.pdf',
                bbox_inches='tight', facecolor='white')

    print(f"\nFigures saved to:")
    print(f"  - {output_dir}/prompt_effects_3panel.png")
    print(f"  - {output_dir}/prompt_effects_3panel.pdf")
    plt.close()


if __name__ == '__main__':
    main()
