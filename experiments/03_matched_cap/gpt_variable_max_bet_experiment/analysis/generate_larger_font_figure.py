#!/usr/bin/env python3
"""
Generate larger font version of combined figure for paper.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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


def generate_figure_combined_large_font(df: pd.DataFrame):
    """
    Combined Figure with LARGER fonts for publication.

    Layout: 1x4
    - Panel A: Bankruptcy Rate
    - Panel B: Game Persistence
    - Panel C: Average Bet Size
    - Panel D: Composite Irrationality Index
    """
    print("Generating Combined Figure with LARGER FONTS...")

    fig, axes = plt.subplots(1, 4, figsize=(32, 7))

    all_amounts = [10, 30, 50, 70]

    # Prepare data
    subset_var = df[df['bet_type'] == 'variable']
    subset_fixed = df[df['bet_type'] == 'fixed']

    # Panel A: Bankruptcy Rate
    ax = axes[0]

    stats_var = subset_var.groupby('bet_amount').agg({
        'is_bankrupt': 'mean'
    }).reset_index()
    stats_var['bankruptcy_rate'] = stats_var['is_bankrupt'] * 100

    stats_fixed = subset_fixed.groupby('bet_amount').agg({
        'is_bankrupt': 'mean'
    }).reset_index()
    stats_fixed['bankruptcy_rate'] = stats_fixed['is_bankrupt'] * 100

    # Variable - layered style
    ax.plot(stats_var['bet_amount'], stats_var['bankruptcy_rate'],
           color=COLORS['variable'], linewidth=3, alpha=0.9, zorder=2)
    ax.scatter(stats_var['bet_amount'], stats_var['bankruptcy_rate'],
              marker='s', s=180, facecolors='white', edgecolors='none', linewidths=0, zorder=3)
    ax.scatter(stats_var['bet_amount'], stats_var['bankruptcy_rate'],
              marker='s', s=120, facecolors='none', edgecolors=COLORS['variable'], linewidths=2.5, zorder=4, label='Variable')

    # Fixed - layered style
    ax.plot(stats_fixed['bet_amount'], stats_fixed['bankruptcy_rate'],
           color=COLORS['fixed'], linewidth=3, alpha=0.9, zorder=2)
    ax.scatter(stats_fixed['bet_amount'], stats_fixed['bankruptcy_rate'],
              marker='o', s=180, facecolors='white', edgecolors='none', linewidths=0, zorder=3)
    ax.scatter(stats_fixed['bet_amount'], stats_fixed['bankruptcy_rate'],
              marker='o', s=120, facecolors='none', edgecolors=COLORS['fixed'], linewidths=2.5, zorder=4, label='Fixed')

    ax.set_ylabel('Bankruptcy Rate (%)', fontsize=44, fontweight='bold')
    ax.set_title('Bankruptcy Rate', fontsize=46, fontweight='bold', pad=20)
    ax.legend(fontsize=36, loc='center right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(all_amounts)
    ax.tick_params(axis='both', which='major', labelsize=40)

    # Panel B: Average Rounds
    ax = axes[1]

    stats_var = subset_var.groupby('bet_amount').agg({
        'total_rounds': 'mean'
    }).reset_index()

    stats_fixed = subset_fixed.groupby('bet_amount').agg({
        'total_rounds': 'mean'
    }).reset_index()

    # Variable - layered style
    ax.plot(stats_var['bet_amount'], stats_var['total_rounds'],
           color=COLORS['variable'], linewidth=3, alpha=0.9, zorder=2)
    ax.scatter(stats_var['bet_amount'], stats_var['total_rounds'],
              marker='s', s=180, facecolors='white', edgecolors='none', linewidths=0, zorder=3)
    ax.scatter(stats_var['bet_amount'], stats_var['total_rounds'],
              marker='s', s=120, facecolors='none', edgecolors=COLORS['variable'], linewidths=2.5, zorder=4, label='Variable')

    # Fixed - layered style
    ax.plot(stats_fixed['bet_amount'], stats_fixed['total_rounds'],
           color=COLORS['fixed'], linewidth=3, alpha=0.9, zorder=2)
    ax.scatter(stats_fixed['bet_amount'], stats_fixed['total_rounds'],
              marker='o', s=180, facecolors='white', edgecolors='none', linewidths=0, zorder=3)
    ax.scatter(stats_fixed['bet_amount'], stats_fixed['total_rounds'],
              marker='o', s=120, facecolors='none', edgecolors=COLORS['fixed'], linewidths=2.5, zorder=4, label='Fixed')

    ax.set_ylabel('Average Rounds', fontsize=44, fontweight='bold')
    ax.set_title('Game Persistence', fontsize=46, fontweight='bold', pad=20)
    ax.legend(fontsize=36, loc='center right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(all_amounts)
    ax.tick_params(axis='both', which='major', labelsize=40)

    # Panel C: Average Bet Size
    ax = axes[2]

    # Fixed: bet size = bet amount
    stats_fixed = subset_fixed.groupby('bet_amount').size().reset_index(name='count')
    stats_fixed['avg_bet'] = stats_fixed['bet_amount']

    # Variable: actual average bet
    stats_var = subset_var.groupby('bet_amount').agg({
        'avg_bet': 'mean'
    }).reset_index()

    # Variable - layered style
    ax.plot(stats_var['bet_amount'], stats_var['avg_bet'],
           color=COLORS['variable'], linewidth=3, alpha=0.9, zorder=2)
    ax.scatter(stats_var['bet_amount'], stats_var['avg_bet'],
              marker='s', s=180, facecolors='white', edgecolors='none', linewidths=0, zorder=3)
    ax.scatter(stats_var['bet_amount'], stats_var['avg_bet'],
              marker='s', s=120, facecolors='none', edgecolors=COLORS['variable'], linewidths=2.5, zorder=4, label='Variable (< Max)')

    # Fixed - layered style
    ax.plot(stats_fixed['bet_amount'], stats_fixed['avg_bet'],
           color=COLORS['fixed'], linewidth=3, alpha=0.9, zorder=2)
    ax.scatter(stats_fixed['bet_amount'], stats_fixed['avg_bet'],
              marker='o', s=180, facecolors='white', edgecolors='none', linewidths=0, zorder=3)
    ax.scatter(stats_fixed['bet_amount'], stats_fixed['avg_bet'],
              marker='o', s=120, facecolors='none', edgecolors=COLORS['fixed'], linewidths=2.5, zorder=4, label='Fixed (= Max)')

    ax.set_ylabel('Average Bet ($)', fontsize=44, fontweight='bold')
    ax.set_title('Average Bet Size', fontsize=46, fontweight='bold', pad=20)
    ax.legend(fontsize=36, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(all_amounts)
    ax.tick_params(axis='both', which='major', labelsize=40)

    # Panel D: Composite Irrationality Index
    ax = axes[3]

    stats_var = subset_var.groupby('bet_amount').agg({
        'composite': 'mean'
    }).reset_index()

    stats_fixed = subset_fixed.groupby('bet_amount').agg({
        'composite': 'mean'
    }).reset_index()

    # Variable - layered style
    ax.plot(stats_var['bet_amount'], stats_var['composite'],
           color=COLORS['variable'], linewidth=3, alpha=0.9, zorder=2)
    ax.scatter(stats_var['bet_amount'], stats_var['composite'],
              marker='s', s=180, facecolors='white', edgecolors='none', linewidths=0, zorder=3)
    ax.scatter(stats_var['bet_amount'], stats_var['composite'],
              marker='s', s=120, facecolors='none', edgecolors=COLORS['variable'],
              linewidths=2.5, zorder=4, label='Variable')

    # Fixed - layered style
    ax.plot(stats_fixed['bet_amount'], stats_fixed['composite'],
           color=COLORS['fixed'], linewidth=3, alpha=0.9, zorder=2)
    ax.scatter(stats_fixed['bet_amount'], stats_fixed['composite'],
              marker='o', s=180, facecolors='white', edgecolors='none', linewidths=0, zorder=3)
    ax.scatter(stats_fixed['bet_amount'], stats_fixed['composite'],
              marker='o', s=120, facecolors='none', edgecolors=COLORS['fixed'],
              linewidths=2.5, zorder=4, label='Fixed')

    ax.set_ylabel('COMPOSITE', fontsize=44, fontweight='bold')
    ax.set_title('Composite Irrationality Index', fontsize=46, fontweight='bold', pad=20)
    ax.legend(fontsize=36, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(all_amounts)
    ax.tick_params(axis='both', which='major', labelsize=40)

    plt.suptitle('Fixed vs. Variable Betting: Comparison Across Bet Amounts',
                 fontsize=50, fontweight='bold', y=1.00)

    # Add single x-axis label at the bottom center
    fig.text(0.5, -0.04, 'Bet Amount ($)', ha='center', fontsize=46, fontweight='bold')

    plt.tight_layout()

    png_path = OUTPUT_DIR / 'combined_fixed_vs_variable_with_composite.png'
    pdf_path = OUTPUT_DIR / 'combined_fixed_vs_variable_with_composite.pdf'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"  ✅ Saved: {png_path.name}")


def main():
    """Main execution."""
    print("="*80)
    print("Generating LARGER FONT Combined Figure")
    print("="*80)

    # Load data
    print(f"\nLoading data from: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    print(f"  ✅ Loaded {len(df)} experiments")
    print(f"     Fixed: {len(df[df['bet_type']=='fixed'])} experiments")
    print(f"     Variable: {len(df[df['bet_type']=='variable'])} experiments")

    # Generate figure
    generate_figure_combined_large_font(df)

    print("\n" + "="*80)
    print(f"✅ Figure saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == '__main__':
    main()
