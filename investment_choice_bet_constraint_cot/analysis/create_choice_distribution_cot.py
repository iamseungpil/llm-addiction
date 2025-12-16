"""
Create Investment Choice Distribution figure for bet_constraint_cot experiment
Layout: 2 rows × 4 columns (Fixed Betting on top, Variable Betting on bottom)
Data source: /data/llm_addiction/investment_choice_bet_constraint_cot/results/
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Data paths - Using extended_cot data (100% complete, 6400 games)
RESULTS_DIR = Path("/data/llm_addiction/investment_choice_extended_cot/results")
OUTPUT_DIR = Path("/home/ubuntu/llm_addiction/investment_choice_extended_cot/analysis")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

MODEL_NAMES = {
    'gpt4o_mini': 'GPT-4o-mini',
    'gpt41_mini': 'GPT-4.1-mini',
    'claude_haiku': 'Claude-3.5-Haiku',
    'gemini_flash': 'Gemini-2.0-Flash'
}

CONDITIONS = ['BASE', 'G', 'M', 'GM']

def load_all_data():
    """Load all experiment results - deduplicated (latest file per combination only)"""
    print("Loading experiment data...")

    # Group files by (model, bet_type, bet_constraint) and keep only latest
    seen_combos = {}
    for f in sorted(os.listdir(RESULTS_DIR), reverse=True):  # Latest first
        if f.endswith('.json'):
            with open(RESULTS_DIR / f) as fp:
                data = json.load(fp)
                config = data['experiment_config']
                key = (config['model'], config['bet_type'], int(config['bet_constraint']))
                if key not in seen_combos:  # Keep first (latest) only
                    seen_combos[key] = (f, data)

    # Aggregate by model_bettype
    all_data = defaultdict(list)
    for key, (filename, data) in seen_combos.items():
        model, bet_type, bet_constraint = key
        agg_key = f"{model}_{bet_type}"
        all_data[agg_key].extend(data['results'])
        n_games = len(data['results'])
        print(f"  {filename}: {n_games} games")

    # Summary
    print(f"\nTotal combinations loaded:")
    for key, games in sorted(all_data.items()):
        print(f"  {key}: {len(games)} games")

    return all_data

def extract_choices(all_data):
    """Extract all choices by model, betting type, and condition"""
    print("\nExtracting choices...")

    choices_data = defaultdict(list)

    for key, games in all_data.items():
        model, bet_type = key.rsplit('_', 1)

        for game in games:
            condition = game['prompt_condition']

            # Extract all choices
            for decision in game['decisions']:
                choice = decision['choice']
                choices_data[(model, bet_type, condition)].append(choice)

    print(f"  Extracted {sum(len(v) for v in choices_data.values())} total decisions")

    return choices_data

def calculate_choice_distribution(choices):
    """Calculate percentage distribution of choices 1-4"""
    total = len(choices)
    if total == 0:
        return [0, 0, 0, 0]

    dist = [choices.count(i) / total * 100 for i in range(1, 5)]
    return dist

def create_revised_figure(choices_data):
    """
    Create 2×4 layout:
    - Top row: Fixed Betting (4 models)
    - Bottom row: Variable Betting (4 models)
    """
    print("\nCreating figure (2 rows × 4 columns)...")

    fig, axes = plt.subplots(2, 4, figsize=(18, 7))

    models = ['gpt4o_mini', 'gpt41_mini', 'claude_haiku', 'gemini_flash']
    colors = ['#2ecc71', '#f1c40f', '#f39c12', '#e74c3c']  # Green, Yellow, Orange, Red
    bar_width = 0.65

    # Check which models have data
    available_models = set()
    for (model, bet_type, condition), choices in choices_data.items():
        if len(choices) > 0:
            available_models.add(model)

    print(f"  Available models: {available_models}")

    # Top row: Fixed Betting
    for i, model in enumerate(models):
        ax = axes[0, i]
        bet_type = 'fixed'

        # Prepare data
        x_pos = np.arange(len(CONDITIONS))

        # Calculate distributions for each condition
        distributions = []
        for condition in CONDITIONS:
            choices = choices_data[(model, bet_type, condition)]
            dist = calculate_choice_distribution(choices)
            distributions.append(dist)

        distributions = np.array(distributions).T

        # Create stacked bar chart
        bottoms = np.zeros(len(CONDITIONS))
        for choice_idx in range(4):
            ax.bar(x_pos, distributions[choice_idx],
                   width=bar_width,
                   label=f'Option {choice_idx+1}',
                   color=colors[choice_idx],
                   bottom=bottoms,
                   edgecolor='white',
                   linewidth=0.8)
            bottoms += distributions[choice_idx]

        # Formatting
        ax.set_title(f'{MODEL_NAMES.get(model, model)}', fontsize=24, fontweight='bold', pad=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(CONDITIONS, fontsize=22)
        ax.set_ylim([0, 100])
        if i == 0:
            ax.set_ylabel('Distribution (%)', fontsize=22, fontweight='bold', labelpad=10)
        ax.tick_params(axis='y', labelsize=20)
        ax.grid(axis='y', alpha=0.3)

    # Bottom row: Variable Betting
    legend_handles = []
    legend_labels = []
    for i, model in enumerate(models):
        ax = axes[1, i]
        bet_type = 'variable'

        # Prepare data
        x_pos = np.arange(len(CONDITIONS))

        # Calculate distributions for each condition
        distributions = []
        for condition in CONDITIONS:
            choices = choices_data[(model, bet_type, condition)]
            dist = calculate_choice_distribution(choices)
            distributions.append(dist)

        distributions = np.array(distributions).T

        # Create stacked bar chart
        bottoms = np.zeros(len(CONDITIONS))
        for choice_idx in range(4):
            bars = ax.bar(x_pos, distributions[choice_idx],
                   width=bar_width,
                   label=f'Option {choice_idx+1}',
                   color=colors[choice_idx],
                   bottom=bottoms,
                   edgecolor='white',
                   linewidth=0.8)
            bottoms += distributions[choice_idx]

            if i == len(models) - 1:
                legend_handles.append(bars)
                legend_labels.append(f'Option {choice_idx+1}')

        # Formatting
        ax.set_xticks(x_pos)
        ax.set_xticklabels(CONDITIONS, fontsize=22)
        ax.set_ylim([0, 100])
        if i == 0:
            ax.set_ylabel('Distribution (%)', fontsize=22, fontweight='bold', labelpad=10)
        ax.tick_params(axis='y', labelsize=20)
        ax.grid(axis='y', alpha=0.3)

    # Add figure-level legend
    fig.legend(legend_handles, legend_labels,
               loc='center right',
               fontsize=20,
               framealpha=0.9,
               bbox_to_anchor=(1.0, 0.47))

    # Add row labels
    fig.text(0.045, 0.66, 'Fixed Betting', fontsize=24, fontweight='bold',
             rotation=90, va='center', ha='center')
    fig.text(0.045, 0.26, 'Variable Betting', fontsize=24, fontweight='bold',
             rotation=90, va='center', ha='center')

    # Add centered X-axis label
    fig.text(0.5, 0.02, 'Prompt Condition', fontsize=22, fontweight='bold',
             ha='center', va='center')

    # Main title
    fig.suptitle('Investment Choice Distribution by Model, Betting Type, and Prompt Condition',
                 fontsize=28, fontweight='bold', y=0.95)

    plt.tight_layout(rect=[0.06, 0.04, 0.88, 0.96])

    # Save both PNG and PDF
    png_path = OUTPUT_DIR / 'investment_choice_distributions_cot.png'
    pdf_path = OUTPUT_DIR / 'investment_choice_distributions_cot.pdf'

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"  Saved PNG: {png_path}")

    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"  Saved PDF: {pdf_path}")

    plt.close()

def print_statistics(choices_data):
    """Print summary statistics"""
    print("\nSummary Statistics:")
    print("="*80)

    models = ['gpt4o_mini', 'gpt41_mini', 'claude_haiku', 'gemini_flash']

    for model in models:
        print(f"\n{MODEL_NAMES.get(model, model)}:")
        print(f"  {'Condition':<15} {'Fixed n':>10} {'Variable n':>12} {'Opt4 (F)':>10} {'Opt4 (V)':>10}")
        print("  " + "-"*60)

        for condition in CONDITIONS:
            choices_fixed = choices_data[(model, 'fixed', condition)]
            choices_var = choices_data[(model, 'variable', condition)]

            opt4_fixed = choices_fixed.count(4) / len(choices_fixed) * 100 if choices_fixed else 0
            opt4_var = choices_var.count(4) / len(choices_var) * 100 if choices_var else 0

            print(f"  {condition:<15} {len(choices_fixed):>10} {len(choices_var):>12} {opt4_fixed:>9.1f}% {opt4_var:>9.1f}%")

def main():
    print("="*80)
    print("Investment Choice Distribution (bet_constraint_cot)")
    print("="*80)

    # Load data
    all_data = load_all_data()

    # Extract choices
    choices_data = extract_choices(all_data)

    # Create figure
    create_revised_figure(choices_data)

    # Print statistics
    print_statistics(choices_data)

    print("\n" + "="*80)
    print("Complete!")
    print(f"Output: {OUTPUT_DIR}")
    print("="*80)

if __name__ == "__main__":
    main()
