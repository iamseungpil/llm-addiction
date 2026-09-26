#!/usr/bin/env python3
"""
Generate Irrationality-Bankruptcy Correlation figure as 4x2 grid.

Rows: 4 bet amounts (10, 30, 50, 70)
Columns: 2 bet types (Fixed, Variable)

Shows correlation between composite irrationality index and bankruptcy rate
for each amount and type separately.
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


def generate_correlation_4x2(df: pd.DataFrame):
    """
    Irrationality-Bankruptcy Correlation: 4x2 Grid.

    Rows: 4 bet amounts (10, 30, 50, 70)
    Columns: Fixed vs Variable

    Each subplot shows correlation for that specific amount and type.
    """
    print("Generating Irrationality-Bankruptcy Correlation (4x2 Grid)...")

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))

    bet_amounts = [10, 30, 50, 70]
    bet_types = ['fixed', 'variable']

    for row_idx, amount in enumerate(bet_amounts):
        for col_idx, bet_type in enumerate(bet_types):
            ax = axes[row_idx, col_idx]

            # Filter data for this amount and type
            subset = df[(df['bet_type'] == bet_type) & (df['bet_amount'] == amount)]

            if len(subset) == 0:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=14)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            # Aggregate by prompt combo
            agg = subset.groupby('prompt_combo').agg({
                'composite': 'mean',
                'is_bankrupt': 'mean'
            }).reset_index()
            agg['bankruptcy_rate'] = agg['is_bankrupt'] * 100

            # Plot scatter
            color = COLORS[bet_type]
            ax.scatter(agg['composite'], agg['bankruptcy_rate'],
                      alpha=0.7, s=80, color=color)

            # Regression line and correlation
            if len(agg) > 1:
                try:
                    r, p = pearsonr(agg['composite'], agg['bankruptcy_rate'])

                    if not np.isnan(r):
                        # Fit line
                        z = np.polyfit(agg['composite'], agg['bankruptcy_rate'], 1)
                        p_line = np.poly1d(z)
                        x_line = np.linspace(agg['composite'].min(), agg['composite'].max(), 100)
                        ax.plot(x_line, p_line(x_line),
                               color=color, linestyle='-', linewidth=2.5, alpha=0.8)

                        # Add correlation text
                        ax.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.2e}',
                               transform=ax.transAxes, verticalalignment='top',
                               fontsize=11,
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                except (ValueError, np.linalg.LinAlgError):
                    pass

            # Formatting
            if row_idx == 0:
                ax.set_title(bet_type.capitalize(), fontsize=14, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(f'${amount}', fontsize=12, fontweight='bold',
                             rotation=0, labelpad=30, va='center')

            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', labelsize=10)

            # Set consistent x-axis
            if col_idx == 0:  # Fixed
                ax.set_xlim(-0.01, 0.35)
            else:  # Variable
                ax.set_xlim(0.19, 0.33)

    # Global labels
    fig.text(0.5, 0.02, 'Composite Irrationality Index', ha='center',
             fontsize=14, fontweight='bold')
    fig.text(0.02, 0.5, 'Bet Amount ($)', va='center', rotation='vertical',
             fontsize=14, fontweight='bold')
    fig.text(0.98, 0.5, 'Bankruptcy Rate (%)', va='center', rotation='vertical',
             fontsize=14, fontweight='bold')

    plt.suptitle('Irrationality-Bankruptcy Correlation: 4 Bet Amounts × 2 Types',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.99])

    png_path = OUTPUT_DIR / '6_irrationality_bankruptcy_correlation.png'
    pdf_path = OUTPUT_DIR / '6_irrationality_bankruptcy_correlation.pdf'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"  ✅ Saved: {png_path.name}")

    # Print correlation summary
    print("\n📊 Correlation Summary:")
    print("-" * 60)
    for amount in bet_amounts:
        print(f"\n${amount}:")
        for bet_type in bet_types:
            subset = df[(df['bet_type'] == bet_type) & (df['bet_amount'] == amount)]
            if len(subset) > 0:
                agg = subset.groupby('prompt_combo').agg({
                    'composite': 'mean',
                    'is_bankrupt': 'mean'
                }).reset_index()

                if len(agg) > 1:
                    r, p = pearsonr(agg['composite'], agg['is_bankrupt'])
                    print(f"  {bet_type:8s}: r = {r:6.3f}, p = {p:.2e}")


def main():
    """Main execution."""
    print("="*80)
    print("Generating Irrationality-Bankruptcy Correlation (4x2 Grid)")
    print("="*80)

    # Load data
    print(f"\nLoading data from: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    print(f"  ✅ Loaded {len(df)} experiments")

    # Generate figure
    print("\n" + "="*80)
    generate_correlation_4x2(df)

    print("\n" + "="*80)
    print(f"✅ Figure saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == '__main__':
    main()
