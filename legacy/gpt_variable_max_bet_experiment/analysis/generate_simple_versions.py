#!/usr/bin/env python3
"""
Generate simple 1x3 and 1x2 versions of the figures.

These show overall fixed vs variable comparison without breaking down by amount.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr

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


def generate_component_effects_1x3(df: pd.DataFrame):
    """
    Component Effects: Simple 1x3 (overall fixed vs variable).
    """
    print("Generating Component Effects (1x3 Simple)...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    components = ['G', 'M', 'P', 'R', 'W']
    metrics = [
        ('is_bankrupt', 'Bankruptcy Effect (%)', 100),
        ('total_bet', 'Total Bet Effect ($)', 1),
        ('total_rounds', 'Rounds Effect', 1)
    ]

    for ax, (metric, ylabel, scale) in zip(axes, metrics):
        x = np.arange(len(components))
        width = 0.35

        effects_fixed = []
        effects_var = []

        for component in components:
            for bet_type, effects_list in [('fixed', effects_fixed), ('variable', effects_var)]:
                subset = df[df['bet_type'] == bet_type]

                with_comp = subset[subset['prompt_combo'].str.contains(component, na=False)]
                without_comp = subset[~subset['prompt_combo'].str.contains(component, na=False)]

                if len(with_comp) > 0 and len(without_comp) > 0:
                    effect = (with_comp[metric].mean() - without_comp[metric].mean()) * scale
                else:
                    effect = 0

                effects_list.append(effect)

        ax.bar(x - width/2, effects_fixed, width, label='Fixed',
              color=COLORS['fixed'], alpha=0.8)
        ax.bar(x + width/2, effects_var, width, label='Variable',
              color=COLORS['variable'], alpha=0.8)

        ax.set_xlabel('Prompt Component', fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(components, fontsize=14)
        ax.axhline(0, color='black', linewidth=1, alpha=0.5)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Component Effects: Fixed vs Variable', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()

    png_path = OUTPUT_DIR / '5_component_effects_fixed_vs_variable_simple.png'
    pdf_path = OUTPUT_DIR / '5_component_effects_fixed_vs_variable_simple.pdf'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"  ✅ Saved: {png_path.name}")


def generate_correlation_1x2(df: pd.DataFrame):
    """
    Irrationality-Bankruptcy Correlation: Simple 1x2 (overall fixed vs variable).
    """
    print("Generating Irrationality-Bankruptcy Correlation (1x2 Simple)...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for idx, bet_type in enumerate(['fixed', 'variable']):
        subset = df[df['bet_type'] == bet_type]

        # Aggregate by prompt combo
        agg = subset.groupby('prompt_combo').agg({
            'composite': 'mean',
            'is_bankrupt': 'mean'
        }).reset_index()
        agg['bankruptcy_rate'] = agg['is_bankrupt'] * 100

        ax = axes[idx]
        color = COLORS[bet_type]

        ax.scatter(agg['composite'], agg['bankruptcy_rate'],
                  alpha=0.7, s=100, color=color)

        # Regression line and correlation
        if len(agg) > 1:
            try:
                r, p = pearsonr(agg['composite'], agg['bankruptcy_rate'])

                if not np.isnan(r):
                    z = np.polyfit(agg['composite'], agg['bankruptcy_rate'], 1)
                    p_line = np.poly1d(z)
                    x_line = np.linspace(agg['composite'].min(), agg['composite'].max(), 100)
                    ax.plot(x_line, p_line(x_line),
                           color=color, linestyle='-', linewidth=2.5, alpha=0.8)

                    ax.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.3e}',
                           transform=ax.transAxes, verticalalignment='top', fontsize=12,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            except:
                pass

        ax.set_xlabel('Composite Irrationality Index', fontsize=14, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Bankruptcy Rate (%)', fontsize=14, fontweight='bold')
        ax.set_title(f'{bet_type.capitalize()} Betting', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Irrationality-Bankruptcy Correlation', fontsize=18, fontweight='bold', y=1.0)
    plt.tight_layout()

    png_path = OUTPUT_DIR / '6_irrationality_bankruptcy_correlation_simple.png'
    pdf_path = OUTPUT_DIR / '6_irrationality_bankruptcy_correlation_simple.pdf'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"  ✅ Saved: {png_path.name}")


def main():
    """Main execution."""
    print("="*80)
    print("Generating Simple (1x3, 1x2) Versions")
    print("="*80)

    # Load data
    print(f"\nLoading data from: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    print(f"  ✅ Loaded {len(df)} experiments")

    # Generate figures
    print("\n" + "="*80)
    generate_component_effects_1x3(df)
    generate_correlation_1x2(df)

    print("\n" + "="*80)
    print(f"✅ Simple versions saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == '__main__':
    main()
