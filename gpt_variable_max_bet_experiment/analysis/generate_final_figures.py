#!/usr/bin/env python3
"""
Generate FINAL corrected figures for GPT Fixed vs Variable Betting Analysis.

Changes:
1. Figure 1: Show ALL 4 amounts ($10, $30, $50, $70) for both fixed and variable
2. Figure 2: Simple line plot (금액별 irrationality 증감)
3. Figure 3: Complexity trend with $10 data (simple 1x3)

Data sources:
- Fixed: 6,400 experiments ($10: 1,600 from fixed_parsing, $30/$50/$70: 4,800)
- Variable: 5,150 experiments (incomplete: ~1,250 missing)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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


def generate_figure_1_final(df: pd.DataFrame):
    """
    Figure 1: Fixed vs Variable Direct Comparison.

    - Shows all 4 amounts [10, 30, 50, 70] for both fixed and variable
    - Upgraded visual style with layered markers (like complexity trend figure)
    """
    print("Generating Figure 1 (UPGRADED): Direct Comparison...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # All amounts to show on x-axis
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
           color=COLORS['variable'], linewidth=2, alpha=0.9, zorder=2)
    ax.scatter(stats_var['bet_amount'], stats_var['bankruptcy_rate'],
              marker='s', s=120, facecolors='white', edgecolors='none', linewidths=0, zorder=3)
    ax.scatter(stats_var['bet_amount'], stats_var['bankruptcy_rate'],
              marker='s', s=80, facecolors='none', edgecolors=COLORS['variable'], linewidths=1.5, zorder=4, label='Variable')

    # Fixed - layered style
    ax.plot(stats_fixed['bet_amount'], stats_fixed['bankruptcy_rate'],
           color=COLORS['fixed'], linewidth=2, alpha=0.9, zorder=2)
    ax.scatter(stats_fixed['bet_amount'], stats_fixed['bankruptcy_rate'],
              marker='o', s=120, facecolors='white', edgecolors='none', linewidths=0, zorder=3)
    ax.scatter(stats_fixed['bet_amount'], stats_fixed['bankruptcy_rate'],
              marker='o', s=80, facecolors='none', edgecolors=COLORS['fixed'], linewidths=1.5, zorder=4, label='Fixed')

    ax.set_ylabel('Bankruptcy Rate (%)', fontsize=20, fontweight='bold')
    ax.set_title('Bankruptcy Rate', fontsize=22, fontweight='bold')
    ax.legend(fontsize=18)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(all_amounts)
    ax.tick_params(axis='both', which='major', labelsize=18)

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
           color=COLORS['variable'], linewidth=2, alpha=0.9, zorder=2)
    ax.scatter(stats_var['bet_amount'], stats_var['total_rounds'],
              marker='s', s=120, facecolors='white', edgecolors='none', linewidths=0, zorder=3)
    ax.scatter(stats_var['bet_amount'], stats_var['total_rounds'],
              marker='s', s=80, facecolors='none', edgecolors=COLORS['variable'], linewidths=1.5, zorder=4, label='Variable')

    # Fixed - layered style
    ax.plot(stats_fixed['bet_amount'], stats_fixed['total_rounds'],
           color=COLORS['fixed'], linewidth=2, alpha=0.9, zorder=2)
    ax.scatter(stats_fixed['bet_amount'], stats_fixed['total_rounds'],
              marker='o', s=120, facecolors='white', edgecolors='none', linewidths=0, zorder=3)
    ax.scatter(stats_fixed['bet_amount'], stats_fixed['total_rounds'],
              marker='o', s=80, facecolors='none', edgecolors=COLORS['fixed'], linewidths=1.5, zorder=4, label='Fixed')

    ax.set_ylabel('Average Rounds', fontsize=20, fontweight='bold')
    ax.set_title('Game Persistence', fontsize=22, fontweight='bold')
    ax.legend(fontsize=18)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(all_amounts)
    ax.tick_params(axis='both', which='major', labelsize=18)

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
           color=COLORS['variable'], linewidth=2, alpha=0.9, zorder=2)
    ax.scatter(stats_var['bet_amount'], stats_var['avg_bet'],
              marker='s', s=120, facecolors='white', edgecolors='none', linewidths=0, zorder=3)
    ax.scatter(stats_var['bet_amount'], stats_var['avg_bet'],
              marker='s', s=80, facecolors='none', edgecolors=COLORS['variable'], linewidths=1.5, zorder=4, label='Variable (< Max)')

    # Fixed - layered style
    ax.plot(stats_fixed['bet_amount'], stats_fixed['avg_bet'],
           color=COLORS['fixed'], linewidth=2, alpha=0.9, zorder=2)
    ax.scatter(stats_fixed['bet_amount'], stats_fixed['avg_bet'],
              marker='o', s=120, facecolors='white', edgecolors='none', linewidths=0, zorder=3)
    ax.scatter(stats_fixed['bet_amount'], stats_fixed['avg_bet'],
              marker='o', s=80, facecolors='none', edgecolors=COLORS['fixed'], linewidths=1.5, zorder=4, label='Fixed (= Max)')

    ax.set_ylabel('Average Bet ($)', fontsize=20, fontweight='bold')
    ax.set_title('Average Bet Size', fontsize=22, fontweight='bold')
    ax.legend(fontsize=18)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(all_amounts)
    ax.tick_params(axis='both', which='major', labelsize=18)

    plt.suptitle('Fixed vs. Variable Betting: Comparison Across Bet Amounts',
                 fontsize=26, fontweight='bold', y=1.00)

    # Add single x-axis label at the bottom center (like complexity trend figure)
    fig.text(0.5, -0.02, 'Bet Amount ($)', ha='center', fontsize=22, fontweight='bold')

    plt.tight_layout()

    png_path = OUTPUT_DIR / '1_fixed_vs_variable_comparison.png'
    pdf_path = OUTPUT_DIR / '1_fixed_vs_variable_comparison.pdf'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"  ✅ Saved: {png_path.name}")


def generate_figure_2_final(df: pd.DataFrame):
    """
    Figure 2: Irrationality Indices by Bet Amount (1x4 horizontal layout).

    - Upgraded visual style with layered markers (like complexity trend figure)
    - Horizontal layout (1 row, 4 columns)
    - Panel A: i_ev
    - Panel B: i_lc
    - Panel C: i_eb
    - Panel D: composite
    """
    print("Generating Figure 2 (UPGRADED): Irrationality Indices...")

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    metrics = [
        ('i_ev', 'Expected Value Ignorance (I_EV)', 'I_EV'),
        ('i_lc', 'Loss Chasing (I_LC)', 'I_LC'),
        ('i_eb', 'Extreme Betting (I_EB)', 'I_EB'),
        ('composite', 'Composite Irrationality Index', 'COMPOSITE')
    ]

    all_amounts = [10, 30, 50, 70]

    # Prepare data
    subset_var = df[df['bet_type'] == 'variable']
    subset_fixed = df[df['bet_type'] == 'fixed']

    for idx, (metric, title, ylabel) in enumerate(metrics):
        ax = axes[idx]

        # Calculate statistics
        stats_var = subset_var.groupby('bet_amount').agg({
            metric: 'mean'
        }).reset_index()

        stats_fixed = subset_fixed.groupby('bet_amount').agg({
            metric: 'mean'
        }).reset_index()

        # Variable - layered style
        ax.plot(stats_var['bet_amount'], stats_var[metric],
               color=COLORS['variable'], linewidth=2, alpha=0.9, zorder=2)
        ax.scatter(stats_var['bet_amount'], stats_var[metric],
                  marker='s', s=120, facecolors='white', edgecolors='none', linewidths=0, zorder=3)
        ax.scatter(stats_var['bet_amount'], stats_var[metric],
                  marker='s', s=80, facecolors='none', edgecolors=COLORS['variable'],
                  linewidths=1.5, zorder=4, label='Variable')

        # Fixed - layered style
        ax.plot(stats_fixed['bet_amount'], stats_fixed[metric],
               color=COLORS['fixed'], linewidth=2, alpha=0.9, zorder=2)
        ax.scatter(stats_fixed['bet_amount'], stats_fixed[metric],
                  marker='o', s=120, facecolors='white', edgecolors='none', linewidths=0, zorder=3)
        ax.scatter(stats_fixed['bet_amount'], stats_fixed[metric],
                  marker='o', s=80, facecolors='none', edgecolors=COLORS['fixed'],
                  linewidths=1.5, zorder=4, label='Fixed')

        ax.set_ylabel(ylabel, fontsize=28, fontweight='bold')
        ax.set_title(title, fontsize=30, fontweight='bold')
        ax.legend(fontsize=24, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(all_amounts)
        ax.tick_params(axis='both', which='major', labelsize=26)

    plt.suptitle('Irrationality Indices by Bet Amount', fontsize=36, fontweight='bold', y=1.02)

    # Add single x-axis label at the bottom center
    fig.text(0.5, -0.02, 'Bet Amount ($)', ha='center', fontsize=30, fontweight='bold')

    plt.tight_layout()

    png_path = OUTPUT_DIR / '2_irrationality_index_by_amount.png'
    pdf_path = OUTPUT_DIR / '2_irrationality_index_by_amount.pdf'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"  ✅ Saved: {png_path.name}")


def generate_figure_combined(df: pd.DataFrame):
    """
    Combined Figure: Figure 1 (3 panels) + Composite Index from Figure 2.

    Layout: 1x4
    - Panel A: Bankruptcy Rate
    - Panel B: Game Persistence
    - Panel C: Average Bet Size
    - Panel D: Composite Irrationality Index
    """
    print("Generating Combined Figure (1x4)...")

    fig, axes = plt.subplots(1, 4, figsize=(28, 6))

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
           color=COLORS['variable'], linewidth=2, alpha=0.9, zorder=2)
    ax.scatter(stats_var['bet_amount'], stats_var['bankruptcy_rate'],
              marker='s', s=120, facecolors='white', edgecolors='none', linewidths=0, zorder=3)
    ax.scatter(stats_var['bet_amount'], stats_var['bankruptcy_rate'],
              marker='s', s=80, facecolors='none', edgecolors=COLORS['variable'], linewidths=1.5, zorder=4, label='Variable')

    # Fixed - layered style
    ax.plot(stats_fixed['bet_amount'], stats_fixed['bankruptcy_rate'],
           color=COLORS['fixed'], linewidth=2, alpha=0.9, zorder=2)
    ax.scatter(stats_fixed['bet_amount'], stats_fixed['bankruptcy_rate'],
              marker='o', s=120, facecolors='white', edgecolors='none', linewidths=0, zorder=3)
    ax.scatter(stats_fixed['bet_amount'], stats_fixed['bankruptcy_rate'],
              marker='o', s=80, facecolors='none', edgecolors=COLORS['fixed'], linewidths=1.5, zorder=4, label='Fixed')

    ax.set_ylabel('Bankruptcy Rate (%)', fontsize=28, fontweight='bold')
    ax.set_title('Bankruptcy Rate', fontsize=30, fontweight='bold')
    ax.legend(fontsize=24)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(all_amounts)
    ax.tick_params(axis='both', which='major', labelsize=26)

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
           color=COLORS['variable'], linewidth=2, alpha=0.9, zorder=2)
    ax.scatter(stats_var['bet_amount'], stats_var['total_rounds'],
              marker='s', s=120, facecolors='white', edgecolors='none', linewidths=0, zorder=3)
    ax.scatter(stats_var['bet_amount'], stats_var['total_rounds'],
              marker='s', s=80, facecolors='none', edgecolors=COLORS['variable'], linewidths=1.5, zorder=4, label='Variable')

    # Fixed - layered style
    ax.plot(stats_fixed['bet_amount'], stats_fixed['total_rounds'],
           color=COLORS['fixed'], linewidth=2, alpha=0.9, zorder=2)
    ax.scatter(stats_fixed['bet_amount'], stats_fixed['total_rounds'],
              marker='o', s=120, facecolors='white', edgecolors='none', linewidths=0, zorder=3)
    ax.scatter(stats_fixed['bet_amount'], stats_fixed['total_rounds'],
              marker='o', s=80, facecolors='none', edgecolors=COLORS['fixed'], linewidths=1.5, zorder=4, label='Fixed')

    ax.set_ylabel('Average Rounds', fontsize=28, fontweight='bold')
    ax.set_title('Game Persistence', fontsize=30, fontweight='bold')
    ax.legend(fontsize=24)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(all_amounts)
    ax.tick_params(axis='both', which='major', labelsize=26)

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
           color=COLORS['variable'], linewidth=2, alpha=0.9, zorder=2)
    ax.scatter(stats_var['bet_amount'], stats_var['avg_bet'],
              marker='s', s=120, facecolors='white', edgecolors='none', linewidths=0, zorder=3)
    ax.scatter(stats_var['bet_amount'], stats_var['avg_bet'],
              marker='s', s=80, facecolors='none', edgecolors=COLORS['variable'], linewidths=1.5, zorder=4, label='Variable (< Max)')

    # Fixed - layered style
    ax.plot(stats_fixed['bet_amount'], stats_fixed['avg_bet'],
           color=COLORS['fixed'], linewidth=2, alpha=0.9, zorder=2)
    ax.scatter(stats_fixed['bet_amount'], stats_fixed['avg_bet'],
              marker='o', s=120, facecolors='white', edgecolors='none', linewidths=0, zorder=3)
    ax.scatter(stats_fixed['bet_amount'], stats_fixed['avg_bet'],
              marker='o', s=80, facecolors='none', edgecolors=COLORS['fixed'], linewidths=1.5, zorder=4, label='Fixed (= Max)')

    ax.set_ylabel('Average Bet ($)', fontsize=28, fontweight='bold')
    ax.set_title('Average Bet Size', fontsize=30, fontweight='bold')
    ax.legend(fontsize=24)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(all_amounts)
    ax.tick_params(axis='both', which='major', labelsize=26)

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
           color=COLORS['variable'], linewidth=2, alpha=0.9, zorder=2)
    ax.scatter(stats_var['bet_amount'], stats_var['composite'],
              marker='s', s=120, facecolors='white', edgecolors='none', linewidths=0, zorder=3)
    ax.scatter(stats_var['bet_amount'], stats_var['composite'],
              marker='s', s=80, facecolors='none', edgecolors=COLORS['variable'],
              linewidths=1.5, zorder=4, label='Variable')

    # Fixed - layered style
    ax.plot(stats_fixed['bet_amount'], stats_fixed['composite'],
           color=COLORS['fixed'], linewidth=2, alpha=0.9, zorder=2)
    ax.scatter(stats_fixed['bet_amount'], stats_fixed['composite'],
              marker='o', s=120, facecolors='white', edgecolors='none', linewidths=0, zorder=3)
    ax.scatter(stats_fixed['bet_amount'], stats_fixed['composite'],
              marker='o', s=80, facecolors='none', edgecolors=COLORS['fixed'],
              linewidths=1.5, zorder=4, label='Fixed')

    ax.set_ylabel('COMPOSITE', fontsize=28, fontweight='bold')
    ax.set_title('Composite Irrationality Index', fontsize=30, fontweight='bold')
    ax.legend(fontsize=24)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(all_amounts)
    ax.tick_params(axis='both', which='major', labelsize=26)

    plt.suptitle('Fixed vs. Variable Betting: Comparison Across Bet Amounts',
                 fontsize=36, fontweight='bold', y=1.02)

    # Add single x-axis label at the bottom center
    fig.text(0.5, -0.02, 'Bet Amount ($)', ha='center', fontsize=30, fontweight='bold')

    plt.tight_layout()

    png_path = OUTPUT_DIR / 'combined_fixed_vs_variable_with_composite.png'
    pdf_path = OUTPUT_DIR / 'combined_fixed_vs_variable_with_composite.pdf'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"  ✅ Saved: {png_path.name}")


def generate_figure_3_final(df: pd.DataFrame):
    """
    Figure 3: Complexity Trend (Simple 1x3 like Image #5 + $10 data).

    - Panel A: Bankruptcy Rate vs Complexity
    - Panel B: Total Rounds vs Complexity
    - Panel C: Total Bet vs Complexity

    Shows both fixed and variable, with all variable amounts including $10.
    """
    print("Generating Figure 3 (FINAL): Complexity Trend (1x3)...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    metrics = [
        ('is_bankrupt', 'Bankruptcy Rate (%)', lambda x: x * 100),
        ('total_rounds', 'Average Rounds', lambda x: x),
        ('total_bet', 'Average Total Bet ($)', lambda x: x)
    ]

    for ax, (metric, ylabel, transform) in zip(axes, metrics):
        for bet_type in ['fixed', 'variable']:
            subset = df[df['bet_type'] == bet_type]
            stats = subset.groupby('complexity').agg({
                metric: 'mean'
            }).reset_index()
            stats[metric] = stats[metric].apply(transform)

            ax.plot(stats['complexity'], stats[metric],
                   marker='o', color=COLORS[bet_type],
                   label=bet_type.capitalize(),
                   linewidth=2.5, markersize=10, alpha=0.8)

            # Add trend line
            if len(stats) > 1:
                z = np.polyfit(stats['complexity'], stats[metric], 1)
                p = np.poly1d(z)
                ax.plot(stats['complexity'], p(stats['complexity']),
                       color=COLORS[bet_type], linestyle='--', alpha=0.5, linewidth=1.5)

        ax.set_xlabel('Prompt Complexity (# components)', fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.2, 5.2)
        ax.set_xticks(range(6))

    plt.suptitle('Complexity Effects: Fixed vs Variable Betting',
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()

    png_path = OUTPUT_DIR / '3_complexity_trend_fixed_vs_variable.png'
    pdf_path = OUTPUT_DIR / '3_complexity_trend_fixed_vs_variable.pdf'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"  ✅ Saved: {png_path.name}")


def main():
    """Main execution."""
    print("="*80)
    print("Generating FINAL Figures (Correct Layout)")
    print("="*80)

    # Load data
    print(f"\nLoading data from: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    print(f"  ✅ Loaded {len(df)} experiments")
    print(f"     Fixed: {len(df[df['bet_type']=='fixed'])} experiments")
    print(f"     Variable: {len(df[df['bet_type']=='variable'])} experiments")

    # Verify data
    print("\n📊 Data Verification:")
    print(f"  Fixed bet amounts: {sorted(df[df['bet_type']=='fixed']['bet_amount'].unique())}")
    print(f"  Variable bet amounts: {sorted(df[df['bet_type']=='variable']['bet_amount'].unique())}")
    print(f"  ⚠️  Note: $10 exists ONLY in variable betting (no fixed $10 data)")

    # Generate figures
    print("\n" + "="*80)
    print("Generating figures...")
    print("="*80)
    generate_figure_1_final(df)
    generate_figure_2_final(df)
    generate_figure_3_final(df)

    print("\n" + "="*80)
    print(f"✅ All figures saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == '__main__':
    main()
