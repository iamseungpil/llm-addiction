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
    - Fixed $10 now available from corrected parsing (1,600 experiments)
    """
    print("Generating Figure 1 (FINAL): Direct Comparison...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # All amounts to show on x-axis
    all_amounts = [10, 30, 50, 70]
    fixed_amounts = [10, 30, 50, 70]  # All 4 amounts now available
    variable_amounts = [10, 30, 50, 70]

    # Panel A: Bankruptcy Rate
    ax = axes[0]

    # Variable (all 4 amounts)
    subset_var = df[df['bet_type'] == 'variable']
    stats_var = subset_var.groupby('bet_amount').agg({
        'is_bankrupt': 'mean'
    }).reset_index()
    stats_var['bankruptcy_rate'] = stats_var['is_bankrupt'] * 100

    ax.plot(stats_var['bet_amount'], stats_var['bankruptcy_rate'],
           marker='s', color=COLORS['variable'], label='Variable',
           linewidth=2.5, markersize=10, alpha=0.8)

    # Fixed (only 30, 50, 70)
    subset_fixed = df[df['bet_type'] == 'fixed']
    stats_fixed = subset_fixed.groupby('bet_amount').agg({
        'is_bankrupt': 'mean'
    }).reset_index()
    stats_fixed['bankruptcy_rate'] = stats_fixed['is_bankrupt'] * 100

    ax.plot(stats_fixed['bet_amount'], stats_fixed['bankruptcy_rate'],
           marker='o', color=COLORS['fixed'], label='Fixed',
           linewidth=2.5, markersize=10, alpha=0.8)

    ax.set_xlabel('Bet Amount ($)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Bankruptcy Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title('Bankruptcy Rate', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(all_amounts)

    # Panel B: Average Rounds
    ax = axes[1]

    stats_var = subset_var.groupby('bet_amount').agg({
        'total_rounds': 'mean'
    }).reset_index()

    stats_fixed = subset_fixed.groupby('bet_amount').agg({
        'total_rounds': 'mean'
    }).reset_index()

    ax.plot(stats_var['bet_amount'], stats_var['total_rounds'],
           marker='s', color=COLORS['variable'], label='Variable',
           linewidth=2.5, markersize=10, alpha=0.8)
    ax.plot(stats_fixed['bet_amount'], stats_fixed['total_rounds'],
           marker='o', color=COLORS['fixed'], label='Fixed',
           linewidth=2.5, markersize=10, alpha=0.8)

    ax.set_xlabel('Bet Amount ($)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Rounds', fontsize=14, fontweight='bold')
    ax.set_title('Game Persistence', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(all_amounts)

    # Panel C: Average Bet Size
    ax = axes[2]

    # Fixed: bet size = bet amount
    stats_fixed = subset_fixed.groupby('bet_amount').size().reset_index(name='count')
    stats_fixed['avg_bet'] = stats_fixed['bet_amount']

    # Variable: actual average bet
    stats_var = subset_var.groupby('bet_amount').agg({
        'avg_bet': 'mean'
    }).reset_index()

    ax.plot(stats_var['bet_amount'], stats_var['avg_bet'],
           marker='s', color=COLORS['variable'], label='Variable (< Max)',
           linewidth=2.5, markersize=10, alpha=0.8)
    ax.plot(stats_fixed['bet_amount'], stats_fixed['avg_bet'],
           marker='o', color=COLORS['fixed'], label='Fixed (= Max)',
           linewidth=2.5, markersize=10, alpha=0.8)

    ax.set_xlabel('Max Bet Amount ($)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Actual Average Bet ($)', fontsize=14, fontweight='bold')
    ax.set_title('Actual Bet Size', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(all_amounts)

    plt.suptitle('Fixed vs Variable Betting: Direct Comparison',
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()

    png_path = OUTPUT_DIR / '1_fixed_vs_variable_comparison.png'
    pdf_path = OUTPUT_DIR / '1_fixed_vs_variable_comparison.pdf'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"  ✅ Saved: {png_path.name}")


def generate_figure_2_final(df: pd.DataFrame):
    """
    Figure 2: Irrationality Indices by Bet Amount (Simple 2x2 line plots).

    Like Image #4: Shows how each index changes with bet amount.
    - Panel A: i_ev
    - Panel B: i_lc (corrected)
    - Panel C: i_eb
    - Panel D: composite
    """
    print("Generating Figure 2 (FINAL): Irrationality Line Plots...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    metrics = [
        ('i_ev', 'Expected Value Ignorance (i_ev)'),
        ('i_lc', 'Loss Chasing (i_lc)'),
        ('i_eb', 'Extreme Betting (i_eb)'),
        ('composite', 'Composite Irrationality Index')
    ]

    all_amounts = [10, 30, 50, 70]

    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx]

        # Variable betting (all 4 amounts)
        subset_var = df[df['bet_type'] == 'variable']
        stats_var = subset_var.groupby('bet_amount').agg({
            metric: 'mean'
        }).reset_index()

        ax.plot(stats_var['bet_amount'], stats_var[metric],
               marker='s', color=COLORS['variable'], label='Variable',
               linewidth=2.5, markersize=10, alpha=0.8)

        # Fixed betting (only 30, 50, 70)
        subset_fixed = df[df['bet_type'] == 'fixed']
        stats_fixed = subset_fixed.groupby('bet_amount').agg({
            metric: 'mean'
        }).reset_index()

        ax.plot(stats_fixed['bet_amount'], stats_fixed[metric],
               marker='o', color=COLORS['fixed'], label='Fixed',
               linewidth=2.5, markersize=10, alpha=0.8)

        ax.set_xlabel('Bet Amount ($)', fontsize=13, fontweight='bold')
        ax.set_ylabel(metric.upper() if metric != 'composite' else 'COMPOSITE',
                     fontsize=13, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(all_amounts)

    plt.suptitle('Irrationality Indices by Bet Amount', fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()

    png_path = OUTPUT_DIR / '2_irrationality_index_by_amount.png'
    pdf_path = OUTPUT_DIR / '2_irrationality_index_by_amount.pdf'
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
