#!/usr/bin/env python3
"""
Create 4-panel figure for Investment Choice Experiment Analysis
Reordered: (a) Bankruptcy Rate, (c) Goal Escalation, (b) Option Distribution, (d) Bet Constraint Effect
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
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 12

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

# Colors for Fixed vs Variable
BETTING_COLORS = {
    'Fixed': '#27ae60',    # Green
    'Variable': '#e74c3c', # Red
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


def calculate_all_stats(all_games):
    """Calculate all statistics for all panels"""
    stats = {
        'bankruptcy_by_prompt': {},
        'option_dist_by_prompt': {},
        'goal_escalation_by_prompt': {},
        'bankruptcy_by_constraint': {},
    }

    prompts = ['BASE', 'G', 'M', 'GM']

    # Stats for panels (a), (b), (c) - by prompt
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

    # Stats for panel (d) - by constraint and bet type
    constraint_stats = defaultdict(lambda: {'fixed': {'total': 0, 'bankrupt': 0},
                                             'variable': {'total': 0, 'bankrupt': 0}})

    for game in all_games:
        constraint = game['constraint']
        bet_type = game['bet_type']
        is_bankrupt = game['final_balance'] <= 0

        constraint_stats[constraint][bet_type]['total'] += 1
        if is_bankrupt:
            constraint_stats[constraint][bet_type]['bankrupt'] += 1

    stats['bankruptcy_by_constraint'] = constraint_stats

    return stats


def create_figure(stats):
    """Create 4-panel figure - REORDERED: (a) Bankruptcy, (b) Goal Escalation, (c) Option Distribution, (d) Bet Constraint
    Panels a, b are narrower; panels c, d are wider"""
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(20, 4.5))
    # width_ratios: a=0.8, b=0.8, c=1.2, d=1.2 (total=4)
    gs = gridspec.GridSpec(1, 4, width_ratios=[0.8, 0.8, 1.2, 1.2], wspace=0.3)
    axes = [fig.add_subplot(gs[i]) for i in range(4)]

    prompts = ['BASE', 'G', 'M', 'GM']
    x_prompts = np.arange(len(prompts))
    width = 0.6

    # ==========================================
    # Panel 0 (a): Bankruptcy Rate by Prompt
    # ==========================================
    ax = axes[0]
    bankruptcy_rates = [stats['bankruptcy_by_prompt'][p] for p in prompts]
    colors = [PROMPT_COLORS[p] for p in prompts]

    bars = ax.bar(x_prompts, bankruptcy_rates, color=colors, width=width, edgecolor='black', linewidth=1)
    ax.set_ylabel('Bankruptcy Rate (%)')
    ax.set_title('(a) Bankruptcy Rate')
    ax.set_xticks(x_prompts)
    ax.set_xticklabels(prompts)
    ax.set_ylim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar, rate in zip(bars, bankruptcy_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=13)

    # ==========================================
    # Panel 1 (b): Goal Escalation by Prompt (was panel c)
    # ==========================================
    ax = axes[1]
    escalation_rates = [stats['goal_escalation_by_prompt'][p] for p in prompts]
    colors = [PROMPT_COLORS[p] for p in prompts]

    bars = ax.bar(x_prompts, escalation_rates, color=colors, width=width, edgecolor='black', linewidth=1)
    ax.set_ylabel('Goal Escalation (%)')
    ax.set_title('(b) Goal Escalation')
    ax.set_xticks(x_prompts)
    ax.set_xticklabels(prompts)
    ax.set_ylim(0, 70)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar, rate in zip(bars, escalation_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=13)

    # ==========================================
    # Panel 2 (c): Option Distribution by Prompt (was panel b)
    # ==========================================
    ax = axes[2]

    bottom = np.zeros(len(prompts))
    for opt in [1, 2, 3, 4]:
        values = [stats['option_dist_by_prompt'][p][opt] for p in prompts]
        color = OPTION_COLORS[f'Option {opt}']
        bars = ax.bar(x_prompts, values, width, bottom=bottom, label=f'Option {opt}',
                      color=color, edgecolor='black', linewidth=1)

        # Add percentage labels for significant portions
        for i, (val, b) in enumerate(zip(values, bottom)):
            if val > 12:  # Only label if > 12%
                text_color = 'white' if opt == 4 else ('white' if opt == 3 else 'black')
                ax.text(x_prompts[i], b + val/2, f'{val:.0f}%', ha='center', va='center',
                        fontsize=12, fontweight='bold', color=text_color)
        bottom += values

    ax.set_ylabel('Distribution (%)')
    ax.set_title('(c) Option Distribution')
    ax.set_xticks(x_prompts)
    ax.set_xticklabels(prompts)
    ax.set_ylim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    legend = ax.legend(loc='lower left', fontsize=11, framealpha=0.95, edgecolor='black')

    # ==========================================
    # Panel 3 (d): Bankruptcy Rate by Bet Constraint
    # ==========================================
    ax = axes[3]

    constraint_stats = stats['bankruptcy_by_constraint']
    constraints = sorted(constraint_stats.keys())
    x_constraints = np.arange(len(constraints))
    bar_width = 0.35

    fixed_rates = []
    variable_rates = []

    for c in constraints:
        fixed_rate = 100 * constraint_stats[c]['fixed']['bankrupt'] / constraint_stats[c]['fixed']['total'] \
            if constraint_stats[c]['fixed']['total'] > 0 else 0
        variable_rate = 100 * constraint_stats[c]['variable']['bankrupt'] / constraint_stats[c]['variable']['total'] \
            if constraint_stats[c]['variable']['total'] > 0 else 0
        fixed_rates.append(fixed_rate)
        variable_rates.append(variable_rate)

    bars1 = ax.bar(x_constraints - bar_width/2, fixed_rates, bar_width,
                   label='Fixed', color=BETTING_COLORS['Fixed'], edgecolor='black', linewidth=1)
    bars2 = ax.bar(x_constraints + bar_width/2, variable_rates, bar_width,
                   label='Variable', color=BETTING_COLORS['Variable'], edgecolor='black', linewidth=1)

    ax.set_ylabel('Bankruptcy Rate (%)')
    ax.set_xlabel('Bet Constraint ($)')
    ax.set_title('(d) Bet Constraint Effect')
    ax.set_xticks(x_constraints)
    ax.set_xticklabels([f'${c}' for c in constraints])
    ax.set_ylim(0, 100)
    ax.legend(loc='upper left', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add value labels
    for bar, rate in zip(bars1, fixed_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    for bar, rate in zip(bars2, variable_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add annotation for consistent difference
    avg_diff = np.mean([v - f for f, v in zip(fixed_rates, variable_rates)])
    ax.annotate(f'Variable > Fixed\n(avg +{avg_diff:.1f}%)',
                xy=(0.97, 0.35), xycoords='axes fraction',
                ha='right', va='bottom', fontsize=13,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray'))

    plt.suptitle('Investment Choice Experiment',
                 fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig, constraints, fixed_rates, variable_rates


def main():
    print("=" * 60)
    print("INVESTMENT CHOICE 4-PANEL FIGURE (Reordered: a, b, c, d)")
    print("=" * 60)

    print("\nLoading data...")
    all_games = load_data()

    print("\nCalculating statistics...")
    stats = calculate_all_stats(all_games)

    print("\n=== Statistics Summary ===")
    print(f"\nBankruptcy by prompt: {stats['bankruptcy_by_prompt']}")
    print(f"Goal escalation by prompt: {stats['goal_escalation_by_prompt']}")

    print("\nOption distribution by prompt:")
    for p in ['BASE', 'G', 'M', 'GM']:
        print(f"  {p}: {stats['option_dist_by_prompt'][p]}")

    print("\nBankruptcy by constraint:")
    for c in sorted(stats['bankruptcy_by_constraint'].keys()):
        cs = stats['bankruptcy_by_constraint'][c]
        f_rate = 100 * cs['fixed']['bankrupt'] / cs['fixed']['total'] if cs['fixed']['total'] > 0 else 0
        v_rate = 100 * cs['variable']['bankrupt'] / cs['variable']['total'] if cs['variable']['total'] > 0 else 0
        print(f"  ${c}: Fixed={f_rate:.1f}%, Variable={v_rate:.1f}%, Diff={v_rate-f_rate:+.1f}%")

    print("\nCreating figure...")
    fig, constraints, fixed_rates, variable_rates = create_figure(stats)

    # Save figure with new name
    output_dir = '/home/ubuntu/llm_addiction/rebuttal_analysis/figures'
    os.makedirs(output_dir, exist_ok=True)

    fig.savefig(f'{output_dir}/investment_choice.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(f'{output_dir}/investment_choice.pdf',
                bbox_inches='tight', facecolor='white')

    print(f"\nFigures saved to:")
    print(f"  - {output_dir}/investment_choice.png")
    print(f"  - {output_dir}/investment_choice.pdf")
    plt.close()

    # Print key finding
    print("\n" + "=" * 60)
    print("KEY FINDING: Variable betting consistently higher bankruptcy")
    print("=" * 60)
    print("\nVariable betting shows higher bankruptcy than Fixed betting")
    print("across ALL bet constraints:")
    for c, f, v in zip(constraints, fixed_rates, variable_rates):
        print(f"  ${c}: Variable {v:.1f}% > Fixed {f:.1f}% (+{v-f:.1f}%)")
    print(f"\nThis demonstrates that betting flexibility is a robust risk")
    print(f"factor independent of bet amount constraints.")


if __name__ == '__main__':
    main()
