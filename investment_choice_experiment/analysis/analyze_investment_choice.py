"""
Investment Choice Experiment Analysis
Analyzes risk-taking behavior across 4 LLMs with different betting types and prompt conditions
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from scipy import stats

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Data paths
RESULTS_DIR = Path("/data/llm_addiction/investment_choice_experiment/results")
OUTPUT_DIR = Path("/home/ubuntu/llm_addiction/investment_choice_experiment/analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

# Result files (validated complete versions - 1600 total experiments)
# All files are from Nov 19, 2025 (latest verified complete data)
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
    first_round_choices = defaultdict(list)

    for key, data in all_data.items():
        model, bet_type = key.rsplit('_', 1)

        for game in data['results']:
            condition = game['prompt_condition']

            # Extract all choices
            for decision in game['decisions']:
                choice = decision['choice']
                choices_data[(model, bet_type, condition)].append(choice)

                # First round only
                if decision['round'] == 1:
                    first_round_choices[(model, bet_type, condition)].append(choice)

    print(f"  ✓ Extracted {sum(len(v) for v in choices_data.values())} total decisions")
    print(f"  ✓ Extracted {sum(len(v) for v in first_round_choices.values())} first-round decisions")

    return choices_data, first_round_choices

def calculate_risk_score(choices):
    """Calculate average risk score (1-4 scale)"""
    return np.mean(choices)

def calculate_choice_distribution(choices):
    """Calculate percentage distribution of choices 1-4"""
    total = len(choices)
    if total == 0:
        return [0, 0, 0, 0]

    dist = [choices.count(i) / total * 100 for i in range(1, 5)]
    return dist

def create_figure1_choice_distributions(choices_data):
    """
    Figure 1: Choice Distribution by Model, Betting Type, and Prompt Condition
    4x2 subplot grid (4 models × 2 betting types)
    """
    print("\n🎨 Creating Figure 1: Choice Distributions...")

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    fig.suptitle('Investment Choice Distribution by Model, Betting Type, and Prompt Condition',
                 fontsize=16, fontweight='bold', y=0.995)

    models = ['gpt4o_mini', 'gpt41_mini', 'claude_haiku', 'gemini_flash']
    bet_types = ['fixed', 'variable']
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']  # Green, Blue, Orange, Red

    for i, model in enumerate(models):
        for j, bet_type in enumerate(bet_types):
            ax = axes[i, j]

            # Prepare data
            x_pos = np.arange(len(CONDITIONS))
            width = 0.2

            # Calculate distributions for each condition
            distributions = []
            for condition in CONDITIONS:
                choices = choices_data[(model, bet_type, condition)]
                dist = calculate_choice_distribution(choices)
                distributions.append(dist)

            distributions = np.array(distributions).T  # Shape: (4 choices, 4 conditions)

            # Create stacked bar chart
            bottoms = np.zeros(len(CONDITIONS))
            for choice_idx in range(4):
                ax.bar(x_pos, distributions[choice_idx],
                       label=f'Option {choice_idx+1}',
                       color=colors[choice_idx],
                       bottom=bottoms)
                bottoms += distributions[choice_idx]

            # Formatting
            ax.set_xlabel('Prompt Condition', fontsize=10)
            ax.set_ylabel('Choice Distribution (%)', fontsize=10)
            ax.set_title(f'{MODEL_NAMES[model]} - {bet_type.capitalize()} Betting',
                        fontsize=11, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(CONDITIONS)
            ax.set_ylim([0, 100])
            ax.grid(axis='y', alpha=0.3)

            # Add legend only to top-right subplot
            if i == 0 and j == 1:
                ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'investment_choice_distributions.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()

def create_figure2_risk_score_heatmap(choices_data):
    """
    Figure 2: Risk Score Heatmap
    Shows average risk score (1-4) across all conditions
    """
    print("\n🎨 Creating Figure 2: Risk Score Heatmap...")

    models = ['gpt4o_mini', 'gpt41_mini', 'claude_haiku', 'gemini_flash']
    bet_types = ['fixed', 'variable']

    # Prepare data matrix
    columns = [f'{cond}-{bt[0].upper()}' for cond in CONDITIONS for bt in bet_types]
    risk_matrix = []

    for model in models:
        row = []
        for condition in CONDITIONS:
            for bet_type in bet_types:
                choices = choices_data[(model, bet_type, condition)]
                risk_score = calculate_risk_score(choices)
                row.append(risk_score)
        risk_matrix.append(row)

    risk_matrix = np.array(risk_matrix)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(16, 6))

    sns.heatmap(risk_matrix,
                annot=True,
                fmt='.2f',
                cmap='RdYlGn_r',  # Red (high risk) to Green (low risk)
                vmin=1.0,
                vmax=4.0,
                xticklabels=columns,
                yticklabels=[MODEL_NAMES[m] for m in models],
                cbar_kws={'label': 'Average Risk Score (1=Safe, 4=Risky)'},
                ax=ax)

    ax.set_title('Risk-Taking Behavior Heatmap: Average Choice Score by Model and Condition',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Condition (F=Fixed, V=Variable)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'investment_risk_score_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()

def create_figure3_first_round_choices(first_round_choices):
    """
    Figure 3: First-Round Choice Distribution
    Analyzes initial decision patterns without history effects
    """
    print("\n🎨 Creating Figure 3: First-Round Choice Patterns...")

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('First-Round Investment Choices by Model and Condition',
                 fontsize=16, fontweight='bold', y=0.995)

    models = ['gpt4o_mini', 'gpt41_mini', 'claude_haiku', 'gemini_flash']
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']

    for i, model in enumerate(models):
        # Fixed betting
        ax = axes[0, i]
        data_fixed = []
        for condition in CONDITIONS:
            choices = first_round_choices[(model, 'fixed', condition)]
            dist = calculate_choice_distribution(choices)
            data_fixed.append(dist)

        data_fixed = np.array(data_fixed).T
        x_pos = np.arange(len(CONDITIONS))
        bottoms = np.zeros(len(CONDITIONS))

        for choice_idx in range(4):
            ax.bar(x_pos, data_fixed[choice_idx],
                   label=f'Option {choice_idx+1}',
                   color=colors[choice_idx],
                   bottom=bottoms)
            bottoms += data_fixed[choice_idx]

        ax.set_title(f'{MODEL_NAMES[model]} - Fixed', fontsize=11, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(CONDITIONS)
        ax.set_ylim([0, 100])
        ax.set_ylabel('Distribution (%)', fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        if i == 3:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)

        # Variable betting
        ax = axes[1, i]
        data_var = []
        for condition in CONDITIONS:
            choices = first_round_choices[(model, 'variable', condition)]
            dist = calculate_choice_distribution(choices)
            data_var.append(dist)

        data_var = np.array(data_var).T
        bottoms = np.zeros(len(CONDITIONS))

        for choice_idx in range(4):
            ax.bar(x_pos, data_var[choice_idx],
                   color=colors[choice_idx],
                   bottom=bottoms)
            bottoms += data_var[choice_idx]

        ax.set_title(f'{MODEL_NAMES[model]} - Variable', fontsize=11, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(CONDITIONS)
        ax.set_ylim([0, 100])
        ax.set_xlabel('Condition', fontsize=10)
        ax.set_ylabel('Distribution (%)', fontsize=10)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'investment_first_round_choices.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()

def print_statistics(choices_data):
    """Print summary statistics"""
    print("\n📈 Summary Statistics:")
    print("="*80)

    models = ['gpt4o_mini', 'gpt41_mini', 'claude_haiku', 'gemini_flash']

    for model in models:
        print(f"\n{MODEL_NAMES[model]}:")
        print(f"  {'Condition':<15} {'Fixed Risk':>12} {'Variable Risk':>15} {'Option 4 (F)':>12} {'Option 4 (V)':>12}")
        print("  " + "-"*70)

        for condition in CONDITIONS:
            choices_fixed = choices_data[(model, 'fixed', condition)]
            choices_var = choices_data[(model, 'variable', condition)]

            risk_fixed = calculate_risk_score(choices_fixed)
            risk_var = calculate_risk_score(choices_var)

            opt4_fixed = choices_fixed.count(4) / len(choices_fixed) * 100
            opt4_var = choices_var.count(4) / len(choices_var) * 100

            print(f"  {condition:<15} {risk_fixed:>12.3f} {risk_var:>15.3f} {opt4_fixed:>11.1f}% {opt4_var:>11.1f}%")

def main():
    print("="*80)
    print("Investment Choice Experiment Analysis")
    print("="*80)

    # Load data
    all_data = load_all_data()

    # Extract choices
    choices_data, first_round_choices = extract_choices(all_data)

    # Create figures
    create_figure1_choice_distributions(choices_data)
    create_figure2_risk_score_heatmap(choices_data)
    create_figure3_first_round_choices(first_round_choices)

    # Print statistics
    print_statistics(choices_data)

    print("\n" + "="*80)
    print("✅ Analysis complete!")
    print(f"📁 Figures saved to: {OUTPUT_DIR}")
    print("="*80)

if __name__ == "__main__":
    main()
