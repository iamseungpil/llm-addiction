#!/usr/bin/env python3
"""
Generate CORRECTED comparison figures for GPT Fixed vs Variable Betting Analysis.

Key improvements:
1. Uses corrected loss chasing definition (ratio-based)
2. Adds $10 data to visualizations
3. Figure 2: 4x3 grid (4 bet amounts × 3 panels)
4. Figure 3: 4x3 grid (4 bet amounts × 3 metrics)
5. NO hard-coding - all values from real data
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


def generate_figure_1_corrected(df: pd.DataFrame):
    """
    Figure 1 (CORRECTED): Fixed vs Variable Direct Comparison with $10 data.

    Panel A: Bankruptcy Rate
    Panel B: Average Rounds
    Panel C: Average Bet Size

    Shows all 4 variable amounts [10, 30, 50, 70] and 3 fixed amounts [30, 50, 70]
    """
    print("Generating Figure 1 (CORRECTED): Direct Comparison with $10 data...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Get bet amounts
    fixed_amounts = sorted(df[df['bet_type'] == 'fixed']['bet_amount'].unique())
    variable_amounts = sorted(df[df['bet_type'] == 'variable']['bet_amount'].unique())

    print(f"  Fixed amounts: {fixed_amounts}")
    print(f"  Variable amounts: {variable_amounts}")

    # Panel A: Bankruptcy Rate
    ax = axes[0]

    # Fixed betting (only 30, 50, 70)
    subset_fixed = df[df['bet_type'] == 'fixed']
    stats_fixed = subset_fixed.groupby('bet_amount').agg({
        'is_bankrupt': 'mean'
    }).reset_index()
    stats_fixed['bankruptcy_rate'] = stats_fixed['is_bankrupt'] * 100

    ax.plot(stats_fixed['bet_amount'], stats_fixed['bankruptcy_rate'],
           marker='o', color=COLORS['fixed'], label='Fixed',
           linewidth=2.5, markersize=10, alpha=0.8)

    # Variable betting (all 4 amounts including $10)
    subset_var = df[df['bet_type'] == 'variable']
    stats_var = subset_var.groupby('bet_amount').agg({
        'is_bankrupt': 'mean'
    }).reset_index()
    stats_var['bankruptcy_rate'] = stats_var['is_bankrupt'] * 100

    ax.plot(stats_var['bet_amount'], stats_var['bankruptcy_rate'],
           marker='s', color=COLORS['variable'], label='Variable',
           linewidth=2.5, markersize=10, alpha=0.8)

    ax.set_xlabel('Bet Amount ($)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Bankruptcy Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title('Bankruptcy Rate', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(variable_amounts)  # Show all 4 amounts

    # Panel B: Average Rounds
    ax = axes[1]

    stats_fixed = subset_fixed.groupby('bet_amount').agg({
        'total_rounds': 'mean'
    }).reset_index()

    stats_var = subset_var.groupby('bet_amount').agg({
        'total_rounds': 'mean'
    }).reset_index()

    ax.plot(stats_fixed['bet_amount'], stats_fixed['total_rounds'],
           marker='o', color=COLORS['fixed'], label='Fixed',
           linewidth=2.5, markersize=10, alpha=0.8)
    ax.plot(stats_var['bet_amount'], stats_var['total_rounds'],
           marker='s', color=COLORS['variable'], label='Variable',
           linewidth=2.5, markersize=10, alpha=0.8)

    ax.set_xlabel('Bet Amount ($)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Rounds', fontsize=14, fontweight='bold')
    ax.set_title('Game Persistence', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(variable_amounts)

    # Panel C: Average Bet Size
    ax = axes[2]

    # Fixed: bet size = bet amount (always)
    stats_fixed = subset_fixed.groupby('bet_amount').size().reset_index(name='count')
    stats_fixed['avg_bet'] = stats_fixed['bet_amount']

    # Variable: actual average bet (computed from round_details)
    stats_var = subset_var.groupby('bet_amount').agg({
        'avg_bet': 'mean'
    }).reset_index()

    ax.plot(stats_fixed['bet_amount'], stats_fixed['avg_bet'],
           marker='o', color=COLORS['fixed'], label='Fixed (= Max)',
           linewidth=2.5, markersize=10, alpha=0.8)
    ax.plot(stats_var['bet_amount'], stats_var['avg_bet'],
           marker='s', color=COLORS['variable'], label='Variable (< Max)',
           linewidth=2.5, markersize=10, alpha=0.8)

    ax.set_xlabel('Max Bet Amount ($)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Actual Average Bet ($)', fontsize=14, fontweight='bold')
    ax.set_title('Actual Bet Size', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(variable_amounts)

    plt.suptitle('Fixed vs Variable Betting: Direct Comparison (with $10)',
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()

    png_path = OUTPUT_DIR / '1_fixed_vs_variable_comparison_corrected.png'
    pdf_path = OUTPUT_DIR / '1_fixed_vs_variable_comparison_corrected.pdf'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"  ✅ Saved: {png_path.name}")


def generate_figure_2_corrected(df: pd.DataFrame):
    """
    Figure 2 (CORRECTED): Irrationality Indices by Bet Amount - 4x3 Grid.

    Rows: 4 bet amounts (10, 30, 50, 70)
    Columns: 3 panels (i_ev, i_lc, i_eb)

    Each subplot shows fixed vs variable for that specific amount.
    """
    print("Generating Figure 2 (CORRECTED): 4x3 Irrationality Grid...")

    fig, axes = plt.subplots(4, 3, figsize=(15, 16))

    variable_amounts = sorted(df[df['bet_type'] == 'variable']['bet_amount'].unique())
    metrics = [
        ('i_ev', 'i_ev'),
        ('i_lc', 'i_lc (corrected)'),
        ('i_eb', 'i_eb')
    ]

    for row_idx, amount in enumerate(variable_amounts):
        for col_idx, (metric, label) in enumerate(metrics):
            ax = axes[row_idx, col_idx]

            # Get data for this amount
            var_data = df[(df['bet_type'] == 'variable') & (df['bet_amount'] == amount)]
            fixed_data = df[(df['bet_type'] == 'fixed') & (df['bet_amount'] == amount)]

            # Group by prompt combo
            var_stats = var_data.groupby('prompt_combo').agg({metric: 'mean'}).reset_index()
            fixed_stats = fixed_data.groupby('prompt_combo').agg({metric: 'mean'}).reset_index() if len(fixed_data) > 0 else pd.DataFrame()

            # Plot variable
            if not var_stats.empty:
                x_var = np.arange(len(var_stats))
                ax.bar(x_var, var_stats[metric], alpha=0.7, color=COLORS['variable'],
                      label='Variable', width=0.8)

            # Overlay fixed (if available)
            if not fixed_stats.empty:
                # Match conditions
                matched = []
                for _, row in var_stats.iterrows():
                    combo = row['prompt_combo']
                    fixed_val = fixed_stats[fixed_stats['prompt_combo'] == combo][metric]
                    matched.append(fixed_val.iloc[0] if len(fixed_val) > 0 else 0)

                ax.plot(x_var, matched, marker='o', color=COLORS['fixed'],
                       label='Fixed', linewidth=2, markersize=6, alpha=0.8)

            # Formatting
            if row_idx == 0:
                ax.set_title(label, fontsize=12, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(f'${amount}', fontsize=12, fontweight='bold', rotation=0,
                             labelpad=30, va='center')
            if row_idx == 0 and col_idx == 2:
                ax.legend(fontsize=9, loc='upper right')

            ax.set_ylim(0, 1.0)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_xticks([])  # Hide x-ticks for cleaner look

    # Global labels
    fig.text(0.5, 0.02, 'Prompt Conditions', ha='center', fontsize=14, fontweight='bold')
    fig.text(0.02, 0.5, 'Bet Amount ($)', va='center', rotation='vertical', fontsize=14, fontweight='bold')

    plt.suptitle('Irrationality Indices: 4 Bet Amounts × 3 Metrics',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.99])

    png_path = OUTPUT_DIR / '2_irrationality_index_by_amount_corrected.png'
    pdf_path = OUTPUT_DIR / '2_irrationality_index_by_amount_corrected.pdf'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"  ✅ Saved: {png_path.name}")


def generate_figure_3_corrected(df: pd.DataFrame):
    """
    Figure 3 (CORRECTED): Complexity Effects by Bet Amount - 4x3 Grid.

    Rows: 4 bet amounts (10, 30, 50, 70)
    Columns: 3 metrics (Bankruptcy Rate, Rounds, Total Bet)

    Each subplot shows complexity trend for fixed vs variable at that amount.
    """
    print("Generating Figure 3 (CORRECTED): 4x3 Complexity Grid...")

    fig, axes = plt.subplots(4, 3, figsize=(18, 16))

    variable_amounts = sorted(df[df['bet_type'] == 'variable']['bet_amount'].unique())
    metrics = [
        ('is_bankrupt', 'Bankruptcy Rate (%)', lambda x: x * 100),
        ('total_rounds', 'Avg Rounds', lambda x: x),
        ('total_bet', 'Avg Total Bet ($)', lambda x: x)
    ]

    for row_idx, amount in enumerate(variable_amounts):
        for col_idx, (metric, ylabel, transform) in enumerate(metrics):
            ax = axes[row_idx, col_idx]

            for bet_type in ['fixed', 'variable']:
                subset = df[(df['bet_type'] == bet_type) & (df['bet_amount'] == amount)]

                if len(subset) == 0:
                    continue

                stats = subset.groupby('complexity').agg({
                    metric: 'mean'
                }).reset_index()
                stats[metric] = stats[metric].apply(transform)

                ax.plot(stats['complexity'], stats[metric],
                       marker='o', color=COLORS[bet_type],
                       label=bet_type.capitalize() if (row_idx == 0 and col_idx == 0) else '',
                       linewidth=2.5, markersize=8, alpha=0.8)

                # Trend line
                if len(stats) > 1:
                    z = np.polyfit(stats['complexity'], stats[metric], 1)
                    p = np.poly1d(z)
                    ax.plot(stats['complexity'], p(stats['complexity']),
                           color=COLORS[bet_type], linestyle='--', alpha=0.4, linewidth=1.5)

            # Formatting
            if row_idx == 0:
                ax.set_title(ylabel, fontsize=12, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(f'${amount}', fontsize=12, fontweight='bold', rotation=0,
                             labelpad=30, va='center')
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=10, loc='upper left')

            ax.grid(True, alpha=0.3)
            ax.set_xlim(-0.2, 5.2)
            ax.set_xticks(range(6))
            if row_idx < 3:  # Hide x-tick labels except bottom row
                ax.set_xticklabels([])

    # Global labels
    fig.text(0.5, 0.02, 'Prompt Complexity (# components)', ha='center', fontsize=14, fontweight='bold')
    fig.text(0.02, 0.5, 'Bet Amount ($)', va='center', rotation='vertical', fontsize=14, fontweight='bold')

    plt.suptitle('Complexity Effects: 4 Bet Amounts × 3 Metrics',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.99])

    png_path = OUTPUT_DIR / '3_complexity_trend_fixed_vs_variable_corrected.png'
    pdf_path = OUTPUT_DIR / '3_complexity_trend_fixed_vs_variable_corrected.pdf'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"  ✅ Saved: {png_path.name}")


def main():
    """Main execution."""
    print("="*80)
    print("Generating CORRECTED Figures (with i_lc corrected + $10 data)")
    print("="*80)

    # Load data
    print(f"\nLoading data from: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    print(f"  ✅ Loaded {len(df)} experiments")
    print(f"     Fixed: {len(df[df['bet_type']=='fixed'])} experiments")
    print(f"     Variable: {len(df[df['bet_type']=='variable'])} experiments")

    # Verify data integrity
    print("\n📊 Data Verification:")
    print(f"  Fixed bet amounts: {sorted(df[df['bet_type']=='fixed']['bet_amount'].unique())}")
    print(f"  Variable bet amounts: {sorted(df[df['bet_type']=='variable']['bet_amount'].unique())}")

    # Generate corrected figures
    print("\n" + "="*80)
    print("Generating figures...")
    print("="*80)
    generate_figure_1_corrected(df)
    generate_figure_2_corrected(df)
    generate_figure_3_corrected(df)

    print("\n" + "="*80)
    print(f"✅ All CORRECTED figures saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == '__main__':
    main()
