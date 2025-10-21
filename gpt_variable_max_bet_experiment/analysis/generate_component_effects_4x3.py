#!/usr/bin/env python3
"""
Generate Component Effects figure as 4x3 grid (4 bet amounts × 3 metrics).

Shows how each prompt component (G, M, P, R, W) affects outcomes
for each bet amount separately.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')

# Paths
DATA_FILE = Path('/home/ubuntu/llm_addiction/gpt_variable_max_bet_experiment/analysis/combined_data_complete.csv')
OUTPUT_DIR = Path('/home/ubuntu/llm_addiction/gpt_variable_max_bet_experiment/analysis/figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color scheme
COLORS = {
    'fixed': '#1f77b4',    # Blue
    'variable': '#d62728'  # Red
}


def generate_component_effects_4x3(df: pd.DataFrame):
    """
    Component Effects: 4x3 Grid.

    Rows: 4 bet amounts (10, 30, 50, 70)
    Columns: 3 metrics (Bankruptcy, Total Bet, Rounds)

    Each subplot shows component effects for that amount and metric.
    """
    print("Generating Component Effects (4x3 Grid)...")

    fig, axes = plt.subplots(4, 3, figsize=(18, 16))

    bet_amounts = [10, 30, 50, 70]
    components = ['G', 'M', 'P', 'R', 'W']

    metrics = [
        ('is_bankrupt', 'Bankruptcy (%)', 100),
        ('total_bet', 'Total Bet ($)', 1),
        ('total_rounds', 'Rounds', 1)
    ]

    for row_idx, amount in enumerate(bet_amounts):
        for col_idx, (metric, ylabel, scale) in enumerate(metrics):
            ax = axes[row_idx, col_idx]

            x = np.arange(len(components))
            width = 0.35

            effects_fixed = []
            effects_var = []

            for component in components:
                for bet_type, effects_list in [('fixed', effects_fixed), ('variable', effects_var)]:
                    # Filter by bet amount and type
                    subset = df[(df['bet_type'] == bet_type) & (df['bet_amount'] == amount)]

                    if len(subset) == 0:
                        effects_list.append(0)
                        continue

                    # Component effect: with vs without
                    with_comp = subset[subset['prompt_combo'].str.contains(component, na=False)]
                    without_comp = subset[~subset['prompt_combo'].str.contains(component, na=False)]

                    if len(with_comp) > 0 and len(without_comp) > 0:
                        effect = (with_comp[metric].mean() - without_comp[metric].mean()) * scale
                    else:
                        effect = 0

                    effects_list.append(effect)

            # Plot bars
            ax.bar(x - width/2, effects_fixed, width, label='Fixed',
                  color=COLORS['fixed'], alpha=0.8)
            ax.bar(x + width/2, effects_var, width, label='Variable',
                  color=COLORS['variable'], alpha=0.8)

            # Formatting
            if row_idx == 0:
                ax.set_title(ylabel, fontsize=13, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(f'${amount}', fontsize=12, fontweight='bold',
                             rotation=0, labelpad=30, va='center')

            ax.set_xticks(x)
            ax.set_xticklabels(components, fontsize=11)
            ax.axhline(0, color='black', linewidth=1, alpha=0.5)
            ax.grid(True, alpha=0.3, axis='y')

            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=10, loc='upper left')

            # Hide x-labels except bottom row
            if row_idx < 3:
                ax.set_xticklabels([])

    # Global labels
    fig.text(0.5, 0.02, 'Prompt Components', ha='center', fontsize=14, fontweight='bold')
    fig.text(0.02, 0.5, 'Bet Amount ($)', va='center', rotation='vertical',
             fontsize=14, fontweight='bold')

    plt.suptitle('Component Effects: 4 Bet Amounts × 3 Metrics',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.99])

    png_path = OUTPUT_DIR / '5_component_effects_fixed_vs_variable.png'
    pdf_path = OUTPUT_DIR / '5_component_effects_fixed_vs_variable.pdf'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"  ✅ Saved: {png_path.name}")


def main():
    """Main execution."""
    print("="*80)
    print("Generating Component Effects Figure (4x3 Grid)")
    print("="*80)

    # Load data
    print(f"\nLoading data from: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    print(f"  ✅ Loaded {len(df)} experiments")

    # Generate figure
    print("\n" + "="*80)
    generate_component_effects_4x3(df)

    print("\n" + "="*80)
    print(f"✅ Figure saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == '__main__':
    main()
