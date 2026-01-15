"""
Create revised Investment Choice Distribution figure
Layout: 2 rows × 4 columns (Fixed Betting on top, Variable Betting on bottom)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Data paths
RESULTS_DIR = Path("/data/llm_addiction/investment_choice_experiment/results")
OUTPUT_DIR = Path("/home/ubuntu/llm_addiction/investment_choice_experiment/analysis")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Result files
RESULT_FILES = {
    'gpt4o_mini_fixed': 'gpt4o_mini_fixed_20251119_042406.json',
    'gpt4o_mini_variable': 'gpt4o_mini_variable_20251119_035805.json',
    'gpt41_mini_fixed': 'gpt41_mini_fixed_20251119_032133.json',
    'gpt41_mini_variable': 'gpt41_mini_variable_20251119_022306.json',
    'claude_haiku_fixed': 'claude_haiku_fixed_20251119_044100.json',
    'claude_haiku_variable': 'claude_haiku_variable_20251119_035809.json',
    'gemini_flash_fixed': 'gemini_flash_fixed_20251119_110752.json',
    'gemini_flash_variable': 'gemini_flash_variable_20251119_043257.json'
}

MODEL_NAMES = {
    'gpt4o_mini': 'GPT-4o-mini',
    'gpt41_mini': 'GPT-4.1-mini',
    'claude_haiku': 'Claude-3.5-Haiku',
    'gemini_flash': 'Gemini-2.5-Flash'
}

CONDITIONS = ['BASE', 'G', 'M', 'GM']

def load_all_data():
    """Load all experiment results"""
    print("📂 Loading experiment data...")
    all_data = {}

    for key, filename in RESULT_FILES.items():
        filepath = RESULTS_DIR / filename
        with open(filepath) as f:
            data = json.load(f)
            all_data[key] = data
            n_games = len(data['results'])
            print(f"  ✓ {key:30} {n_games:>3} games")

    return all_data

def extract_choices(all_data):
    """Extract all choices by model, betting type, and condition"""
    print("\n📊 Extracting choices...")

    choices_data = defaultdict(list)

    for key, data in all_data.items():
        model, bet_type = key.rsplit('_', 1)

        for game in data['results']:
            condition = game['prompt_condition']

            # Extract all choices
            for decision in game['decisions']:
                choice = decision['choice']
                choices_data[(model, bet_type, condition)].append(choice)

    print(f"  ✓ Extracted {sum(len(v) for v in choices_data.values())} total decisions")

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
    print("\n🎨 Creating revised figure (2 rows × 4 columns)...")

    fig, axes = plt.subplots(2, 4, figsize=(18, 7))

    models = ['gpt4o_mini', 'gpt41_mini', 'claude_haiku', 'gemini_flash']
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']  # Green, Blue, Orange, Red
    bar_width = 0.65  # Reduce bar width

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

        distributions = np.array(distributions).T  # Shape: (4 choices, 4 conditions)

        # Create stacked bar chart with reduced width
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

        # Formatting with larger fonts
        ax.set_title(f'{MODEL_NAMES[model]}', fontsize=24, fontweight='bold', pad=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(CONDITIONS, fontsize=22)
        ax.set_ylim([0, 100])
        # Only show Y-label on leftmost subplot
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

        # Create stacked bar chart with reduced width
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

            # Collect legend handles and labels from last subplot
            if i == len(models) - 1:
                legend_handles.append(bars)
                legend_labels.append(f'Option {choice_idx+1}')

        # Formatting with larger fonts
        ax.set_xticks(x_pos)
        ax.set_xticklabels(CONDITIONS, fontsize=22)
        ax.set_ylim([0, 100])
        # Only show Y-label on leftmost subplot
        if i == 0:
            ax.set_ylabel('Distribution (%)', fontsize=22, fontweight='bold', labelpad=10)
        ax.tick_params(axis='y', labelsize=20)
        ax.grid(axis='y', alpha=0.3)

    # Add figure-level legend on the right side
    fig.legend(legend_handles, legend_labels,
               loc='center right',
               fontsize=20,
               framealpha=0.9,
               bbox_to_anchor=(1.0, 0.47))

    # Add row labels (Fixed Betting / Variable Betting)
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
    png_path = OUTPUT_DIR / 'investment_choice_distributions_revised.png'
    pdf_path = OUTPUT_DIR / 'investment_choice_distributions_revised.pdf'

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved PNG: {png_path}")

    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"  ✓ Saved PDF: {pdf_path}")

    plt.close()

def main():
    print("="*80)
    print("Creating Revised Investment Choice Distribution Figure")
    print("="*80)

    # Load data
    all_data = load_all_data()

    # Extract choices
    choices_data = extract_choices(all_data)

    # Create revised figure
    create_revised_figure(choices_data)

    print("\n" + "="*80)
    print("✅ Figure creation complete!")
    print(f"📁 Saved to: {OUTPUT_DIR}")
    print("="*80)

if __name__ == "__main__":
    main()
