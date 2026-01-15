#!/usr/bin/env python3
"""
Create separate visualizations for Investment Choice experiment:
1. By Prompt Condition
2. By Betting Type (Fixed vs Variable)
3. By Model
"""

import json
import os
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Output directory
OUTPUT_DIR = Path("/home/ubuntu/llm_addiction/rebuttal_analysis/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Data directory
RESULTS_DIR = Path("/data/llm_addiction/investment_choice_experiment/results")

# Colors for options (consistent with original figure)
OPTION_COLORS = {
    1: '#2ecc71',  # Green - Safe exit
    2: '#3498db',  # Blue - Moderate risk
    3: '#f39c12',  # Orange - High risk
    4: '#e74c3c',  # Red - Extreme risk
}

def load_data():
    """Load and aggregate all investment choice data"""
    all_data = []

    for fn in sorted(RESULTS_DIR.glob("*.json")):
        with open(fn, 'r') as f:
            data = json.load(f)

        model = data.get('experiment_config', {}).get('model', fn.stem.split('_')[0])
        bet_type = data.get('experiment_config', {}).get('bet_type', 'unknown')

        for r in data.get('results', []):
            prompt_cond = r.get('prompt_condition', 'BASE')

            for d in r.get('decisions', []):
                choice = d.get('choice', 0)
                all_data.append({
                    'model': model,
                    'bet_type': bet_type,
                    'prompt_condition': prompt_cond,
                    'choice': choice
                })

    return all_data


def calculate_distribution(data, group_by):
    """Calculate option distribution for each group"""
    groups = defaultdict(lambda: defaultdict(int))

    for d in data:
        key = d[group_by]
        groups[key][d['choice']] += 1

    # Convert to percentages
    distributions = {}
    for key, choices in groups.items():
        total = sum(choices.values())
        distributions[key] = {i: choices[i]/total*100 for i in range(1, 5)}

    return distributions


def plot_by_prompt_condition(data):
    """Figure 1: Option distribution by prompt condition"""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 7))

    prompts = ['BASE', 'G', 'M', 'GM']
    dist = calculate_distribution(data, 'prompt_condition')

    x = np.arange(len(prompts))
    width = 0.6

    # Stacked bar chart
    bottom = np.zeros(len(prompts))
    for opt in [1, 2, 3, 4]:
        values = [dist[p][opt] for p in prompts]
        bars = ax.bar(x, values, width, bottom=bottom,
                     label=f'Option {opt}', color=OPTION_COLORS[opt],
                     edgecolor='white', linewidth=1)

        # Add percentage labels for significant portions (>10%)
        for i, (v, b) in enumerate(zip(values, bottom)):
            if v > 8:
                ax.text(i, b + v/2, f'{v:.1f}%', ha='center', va='center',
                       fontsize=12, fontweight='bold', color='white')

        bottom += values

    ax.set_ylabel('Distribution (%)', fontsize=18, fontweight='bold')
    ax.set_xlabel('Prompt Condition', fontsize=18, fontweight='bold')
    ax.set_title('Investment Choice Distribution by Prompt Condition\n(Aggregated Across All Models)',
                fontsize=18, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(prompts, fontsize=16)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=12, loc='upper right', title='Choice', title_fontsize=12)

    plt.tight_layout()

    # Save
    for ext in ['png', 'pdf']:
        fig.savefig(OUTPUT_DIR / f'investment_choice_by_prompt.{ext}',
                   dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: investment_choice_by_prompt.png/pdf")


def plot_by_betting_type(data):
    """Figure 2: Option distribution by betting type"""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 7))

    bet_types = ['fixed', 'variable']
    labels = ['Fixed ($10)', 'Variable ($5-balance)']
    dist = calculate_distribution(data, 'bet_type')

    x = np.arange(len(bet_types))
    width = 0.5

    # Stacked bar chart
    bottom = np.zeros(len(bet_types))
    for opt in [1, 2, 3, 4]:
        values = [dist[bt][opt] for bt in bet_types]
        bars = ax.bar(x, values, width, bottom=bottom,
                     label=f'Option {opt}', color=OPTION_COLORS[opt],
                     edgecolor='white', linewidth=1)

        # Add percentage labels
        for i, (v, b) in enumerate(zip(values, bottom)):
            if v > 8:
                ax.text(i, b + v/2, f'{v:.1f}%', ha='center', va='center',
                       fontsize=14, fontweight='bold', color='white')

        bottom += values

    ax.set_ylabel('Distribution (%)', fontsize=18, fontweight='bold')
    ax.set_xlabel('Betting Type', fontsize=18, fontweight='bold')
    ax.set_title('Investment Choice Distribution by Betting Type\n(Aggregated Across All Models & Prompts)',
                fontsize=18, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=16)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=12, loc='upper right', title='Choice', title_fontsize=12)

    plt.tight_layout()

    # Save
    for ext in ['png', 'pdf']:
        fig.savefig(OUTPUT_DIR / f'investment_choice_by_betting_type.{ext}',
                   dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: investment_choice_by_betting_type.png/pdf")


def plot_by_model(data):
    """Figure 3: Option distribution by model"""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))

    models = ['gpt4o_mini', 'gpt41_mini', 'claude_haiku', 'gemini_flash']
    labels = ['GPT-4o-mini', 'GPT-4.1-mini', 'Claude-3.5-Haiku', 'Gemini-2.5-Flash']
    dist = calculate_distribution(data, 'model')

    x = np.arange(len(models))
    width = 0.6

    # Stacked bar chart
    bottom = np.zeros(len(models))
    for opt in [1, 2, 3, 4]:
        values = [dist[m][opt] for m in models]
        bars = ax.bar(x, values, width, bottom=bottom,
                     label=f'Option {opt}', color=OPTION_COLORS[opt],
                     edgecolor='white', linewidth=1)

        # Add percentage labels
        for i, (v, b) in enumerate(zip(values, bottom)):
            if v > 8:
                ax.text(i, b + v/2, f'{v:.1f}%', ha='center', va='center',
                       fontsize=12, fontweight='bold', color='white')

        bottom += values

    ax.set_ylabel('Distribution (%)', fontsize=18, fontweight='bold')
    ax.set_xlabel('Model', fontsize=18, fontweight='bold')
    ax.set_title('Investment Choice Distribution by Model\n(Aggregated Across All Conditions)',
                fontsize=18, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=12, loc='upper right', title='Choice', title_fontsize=12)

    plt.tight_layout()

    # Save
    for ext in ['png', 'pdf']:
        fig.savefig(OUTPUT_DIR / f'investment_choice_by_model.{ext}',
                   dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: investment_choice_by_model.png/pdf")


def plot_combined_summary(data):
    """Figure 4: Combined 3-panel summary (similar to target increase figure)"""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: By Betting Type
    bet_types = ['fixed', 'variable']
    labels_bt = ['Fixed', 'Variable']
    dist_bt = calculate_distribution(data, 'bet_type')

    x = np.arange(len(bet_types))
    width = 0.5
    bottom = np.zeros(len(bet_types))

    for opt in [1, 2, 3, 4]:
        values = [dist_bt[bt][opt] for bt in bet_types]
        axes[0].bar(x, values, width, bottom=bottom,
                   label=f'Option {opt}', color=OPTION_COLORS[opt],
                   edgecolor='white', linewidth=1)
        for i, (v, b) in enumerate(zip(values, bottom)):
            if v > 10:
                axes[0].text(i, b + v/2, f'{v:.0f}%', ha='center', va='center',
                           fontsize=13, fontweight='bold', color='white')
        bottom += values

    axes[0].set_ylabel('Distribution (%)', fontsize=18, fontweight='bold')
    axes[0].set_xlabel('By Betting Type', fontsize=18, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels_bt, fontsize=16)
    axes[0].tick_params(axis='y', labelsize=14)
    axes[0].set_ylim(0, 100)

    # Panel 2: By Prompt Condition
    prompts = ['BASE', 'G', 'M', 'GM']
    dist_pc = calculate_distribution(data, 'prompt_condition')

    x = np.arange(len(prompts))
    width = 0.6
    bottom = np.zeros(len(prompts))

    for opt in [1, 2, 3, 4]:
        values = [dist_pc[p][opt] for p in prompts]
        axes[1].bar(x, values, width, bottom=bottom,
                   label=f'Option {opt}', color=OPTION_COLORS[opt],
                   edgecolor='white', linewidth=1)
        for i, (v, b) in enumerate(zip(values, bottom)):
            if v > 10:
                axes[1].text(i, b + v/2, f'{v:.0f}%', ha='center', va='center',
                           fontsize=13, fontweight='bold', color='white')
        bottom += values

    axes[1].set_ylabel('Distribution (%)', fontsize=18, fontweight='bold')
    axes[1].set_xlabel('By Prompt Condition', fontsize=18, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(prompts, fontsize=16)
    axes[1].tick_params(axis='y', labelsize=14)
    axes[1].set_ylim(0, 100)

    # Panel 3: By Model
    models = ['gpt4o_mini', 'gpt41_mini', 'claude_haiku', 'gemini_flash']
    labels_m = ['GPT-4o', 'GPT-4.1', 'Claude', 'Gemini']
    dist_m = calculate_distribution(data, 'model')

    x = np.arange(len(models))
    width = 0.6
    bottom = np.zeros(len(models))

    for opt in [1, 2, 3, 4]:
        values = [dist_m[m][opt] for m in models]
        bars = axes[2].bar(x, values, width, bottom=bottom,
                          label=f'Option {opt}', color=OPTION_COLORS[opt],
                          edgecolor='white', linewidth=1)
        for i, (v, b) in enumerate(zip(values, bottom)):
            if v > 10:
                axes[2].text(i, b + v/2, f'{v:.0f}%', ha='center', va='center',
                           fontsize=13, fontweight='bold', color='white')
        bottom += values

    axes[2].set_ylabel('Distribution (%)', fontsize=18, fontweight='bold')
    axes[2].set_xlabel('By Model', fontsize=18, fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels_m, fontsize=14)
    axes[2].tick_params(axis='y', labelsize=14)
    axes[2].set_ylim(0, 100)
    axes[2].legend(fontsize=11, loc='upper right', title='Choice', title_fontsize=11)

    fig.suptitle('Investment Choice Distribution Analysis', fontsize=22, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    # Save
    for ext in ['png', 'pdf']:
        fig.savefig(OUTPUT_DIR / f'investment_choice_combined_summary.{ext}',
                   dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: investment_choice_combined_summary.png/pdf")


def main():
    print("Loading data...")
    data = load_data()
    print(f"Loaded {len(data)} decisions\n")

    print("Creating visualizations...")
    plot_by_prompt_condition(data)
    plot_by_betting_type(data)
    plot_by_model(data)
    plot_combined_summary(data)

    print(f"\nAll figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
