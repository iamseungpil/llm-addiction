#!/usr/bin/env python3
"""
Generate all comparison figures for GPT Fixed vs Variable Betting Analysis.

Generates 6 publication-quality figures:
1. Fixed vs Variable Direct Comparison (3-panel)
2. Irrationality Index by Bet Amount (4-panel)
3. Complexity Trend Comparison (3-panel)
4. Bankruptcy Heatmap (2-panel)
5. Component Effects Comparison (3-panel)
6. Irrationality-Bankruptcy Correlation (2-panel)
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

def complexity_from_combo(combo: str) -> int:
    """Calculate prompt complexity from combo string."""
    if not combo or combo == 'BASE':
        return 0
    components = ['G', 'M', 'P', 'R', 'W']
    return sum(1 for c in components if c in combo)


def generate_figure_1(df: pd.DataFrame):
    """
    Figure 1: Fixed vs Variable Direct Comparison (3-panel).

    Panel A: Bankruptcy Rate by Bet Amount
    Panel B: Average Rounds by Bet Amount
    Panel C: Average Bet Size
    """
    print("Generating Figure 1: Direct Comparison...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Get shared bet amounts
    fixed_amounts = sorted(df[df['bet_type'] == 'fixed']['bet_amount'].unique())
    variable_amounts = sorted(df[df['bet_type'] == 'variable']['bet_amount'].unique())
    common_amounts = sorted(set(fixed_amounts) & set(variable_amounts))

    # Panel A: Bankruptcy Rate
    ax = axes[0]
    for bet_type in ['fixed', 'variable']:
        subset = df[df['bet_type'] == bet_type]
        stats = subset.groupby('bet_amount').agg({
            'is_bankrupt': 'mean'
        }).reset_index()
        stats['bankruptcy_rate'] = stats['is_bankrupt'] * 100

        # Only plot common amounts
        stats = stats[stats['bet_amount'].isin(common_amounts)]

        ax.plot(stats['bet_amount'], stats['bankruptcy_rate'],
               marker='o', color=COLORS[bet_type], label=bet_type.capitalize(),
               linewidth=2.5, markersize=10, alpha=0.8)

    ax.set_xlabel('Bet Amount ($)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Bankruptcy Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title('Bankruptcy Rate', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(common_amounts)

    # Panel B: Average Rounds
    ax = axes[1]
    for bet_type in ['fixed', 'variable']:
        subset = df[df['bet_type'] == bet_type]
        stats = subset.groupby('bet_amount').agg({
            'total_rounds': 'mean'
        }).reset_index()

        stats = stats[stats['bet_amount'].isin(common_amounts)]

        ax.plot(stats['bet_amount'], stats['total_rounds'],
               marker='s', color=COLORS[bet_type], label=bet_type.capitalize(),
               linewidth=2.5, markersize=10, alpha=0.8)

    ax.set_xlabel('Bet Amount ($)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Rounds', fontsize=14, fontweight='bold')
    ax.set_title('Game Persistence', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(common_amounts)

    # Panel C: Average Bet Size
    ax = axes[2]

    # Fixed: bet size = bet amount
    subset_fixed = df[df['bet_type'] == 'fixed']
    stats_fixed = subset_fixed.groupby('bet_amount').size().reset_index(name='count')
    stats_fixed['avg_bet'] = stats_fixed['bet_amount']  # Fixed bet = bet_amount
    stats_fixed = stats_fixed[stats_fixed['bet_amount'].isin(common_amounts)]

    # Variable: actual average bet
    subset_var = df[df['bet_type'] == 'variable']
    stats_var = subset_var.groupby('bet_amount').agg({
        'avg_bet': 'mean'
    }).reset_index()
    stats_var = stats_var[stats_var['bet_amount'].isin(variable_amounts)]

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

    plt.suptitle('Fixed vs Variable Betting: Direct Comparison', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()

    png_path = OUTPUT_DIR / '1_fixed_vs_variable_comparison.png'
    pdf_path = OUTPUT_DIR / '1_fixed_vs_variable_comparison.pdf'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved: {png_path.name}")


def generate_figure_2(df: pd.DataFrame):
    """
    Figure 2: Irrationality Index by Bet Amount (4-panel).

    Panel A: i_ev
    Panel B: i_lc
    Panel C: i_eb
    Panel D: composite
    """
    print("Generating Figure 2: Irrationality Indices...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    metrics = [
        ('i_ev', 'Expected Value Ignorance (i_ev)'),
        ('i_lc', 'Loss Chasing (i_lc)'),
        ('i_eb', 'Extreme Betting (i_eb)'),
        ('composite', 'Composite Irrationality Index')
    ]

    variable_amounts = sorted(df[df['bet_type'] == 'variable']['bet_amount'].unique())
    fixed_amounts = sorted(df[df['bet_type'] == 'fixed']['bet_amount'].unique())

    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx]

        # Fixed betting
        subset_fixed = df[df['bet_type'] == 'fixed']
        stats_fixed = subset_fixed.groupby('bet_amount').agg({
            metric: 'mean'
        }).reset_index()
        stats_fixed = stats_fixed[stats_fixed['bet_amount'].isin(fixed_amounts)]

        # Variable betting
        subset_var = df[df['bet_type'] == 'variable']
        stats_var = subset_var.groupby('bet_amount').agg({
            metric: 'mean'
        }).reset_index()
        stats_var = stats_var[stats_var['bet_amount'].isin(variable_amounts)]

        ax.plot(stats_fixed['bet_amount'], stats_fixed[metric],
               marker='o', color=COLORS['fixed'], label='Fixed',
               linewidth=2.5, markersize=10, alpha=0.8)
        ax.plot(stats_var['bet_amount'], stats_var[metric],
               marker='s', color=COLORS['variable'], label='Variable',
               linewidth=2.5, markersize=10, alpha=0.8)

        ax.set_xlabel('Bet Amount ($)', fontsize=13, fontweight='bold')
        ax.set_ylabel(metric.upper(), fontsize=13, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(variable_amounts)

    plt.suptitle('Irrationality Indices by Bet Amount', fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()

    png_path = OUTPUT_DIR / '2_irrationality_index_by_amount.png'
    pdf_path = OUTPUT_DIR / '2_irrationality_index_by_amount.pdf'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved: {png_path.name}")


def generate_figure_3(df: pd.DataFrame):
    """
    Figure 3: Complexity Trend Comparison (3-panel).

    Panel A: Bankruptcy Rate vs Complexity
    Panel B: Total Rounds vs Complexity
    Panel C: Total Bet vs Complexity
    """
    print("Generating Figure 3: Complexity Trend...")

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
                   marker='o', color=COLORS[bet_type], label=bet_type.capitalize(),
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

    plt.suptitle('Complexity Effects: Fixed vs Variable Betting', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()

    png_path = OUTPUT_DIR / '3_complexity_trend_fixed_vs_variable.png'
    pdf_path = OUTPUT_DIR / '3_complexity_trend_fixed_vs_variable.pdf'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved: {png_path.name}")


def generate_figure_4(df: pd.DataFrame):
    """
    Figure 4: Bankruptcy Heatmap (2-panel side-by-side).

    Left: Fixed Betting
    Right: Variable Betting
    """
    print("Generating Figure 4: Bankruptcy Heatmap...")

    # Get all prompt combos
    all_combos = sorted(df['prompt_combo'].unique(), key=lambda x: (complexity_from_combo(x), x))

    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

    for idx, bet_type in enumerate(['fixed', 'variable']):
        subset = df[df['bet_type'] == bet_type]
        amounts = sorted(subset['bet_amount'].unique())

        # Create pivot table
        pivot_data = []
        for amount in amounts:
            row = []
            for combo in all_combos:
                group = subset[(subset['bet_amount'] == amount) & (subset['prompt_combo'] == combo)]
                bankruptcy_rate = group['is_bankrupt'].mean() * 100 if len(group) > 0 else 0
                row.append(bankruptcy_rate)
            pivot_data.append(row)

        pivot_df = pd.DataFrame(pivot_data, columns=all_combos, index=amounts)

        # Plot heatmap
        ax = axes[idx]
        sns.heatmap(pivot_df, cmap='YlOrRd', vmin=0, vmax=60, annot=False,
                   fmt='.0f', cbar_kws={'label': 'Bankruptcy Rate (%)'}, ax=ax)

        ax.set_xlabel('Prompt Combination', fontsize=14, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Bet Amount ($)', fontsize=14, fontweight='bold')
        ax.set_title(f'{bet_type.capitalize()} Betting', fontsize=16, fontweight='bold')

        # Rotate x labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)

    plt.suptitle('Bankruptcy Rate Heatmap: Fixed vs Variable', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()

    png_path = OUTPUT_DIR / '4_bankruptcy_heatmap_fixed_vs_variable.png'
    pdf_path = OUTPUT_DIR / '4_bankruptcy_heatmap_fixed_vs_variable.pdf'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved: {png_path.name}")


def generate_figure_5(df: pd.DataFrame):
    """
    Figure 5: Component Effects Comparison (3-panel).

    Panel A: Bankruptcy Effect
    Panel B: Total Bet Effect
    Panel C: Rounds Effect
    """
    print("Generating Figure 5: Component Effects...")

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
            # Map R to search string
            search_comp = component

            for bet_type, effects_list in [('fixed', effects_fixed), ('variable', effects_var)]:
                subset = df[df['bet_type'] == bet_type]
                with_comp = subset[subset['prompt_combo'].str.contains(search_comp, na=False)]
                without_comp = subset[~subset['prompt_combo'].str.contains(search_comp, na=False)]

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

    png_path = OUTPUT_DIR / '5_component_effects_fixed_vs_variable.png'
    pdf_path = OUTPUT_DIR / '5_component_effects_fixed_vs_variable.pdf'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved: {png_path.name}")


def generate_figure_6(df: pd.DataFrame):
    """
    Figure 6: Irrationality-Bankruptcy Correlation (2-panel).

    Left: Fixed Betting
    Right: Variable Betting
    """
    print("Generating Figure 6: Irrationality-Bankruptcy Correlation...")

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
        ax.scatter(agg['composite'], agg['bankruptcy_rate'],
                  alpha=0.7, s=100, color=COLORS[bet_type])

        # Regression line and correlation
        if len(agg) > 1:
            try:
                r, p = pearsonr(agg['composite'], agg['bankruptcy_rate'])

                if not np.isnan(r):
                    z = np.polyfit(agg['composite'], agg['bankruptcy_rate'], 1)
                    p_line = np.poly1d(z)
                    x_line = np.linspace(agg['composite'].min(), agg['composite'].max(), 100)
                    ax.plot(x_line, p_line(x_line),
                           color=COLORS[bet_type], linestyle='-', linewidth=2.5, alpha=0.8)

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

    png_path = OUTPUT_DIR / '6_irrationality_bankruptcy_correlation.png'
    pdf_path = OUTPUT_DIR / '6_irrationality_bankruptcy_correlation.pdf'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved: {png_path.name}")


def main():
    """Main execution."""
    print("="*70)
    print("Generating All Figures")
    print("="*70)

    # Load data
    print(f"\nLoading data from: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    print(f"  Loaded {len(df)} experiments")
    print(f"  Fixed: {len(df[df['bet_type']=='fixed'])}")
    print(f"  Variable: {len(df[df['bet_type']=='variable'])}")

    # Generate all figures
    print("\n" + "="*70)
    generate_figure_1(df)
    generate_figure_2(df)
    generate_figure_3(df)
    generate_figure_4(df)
    generate_figure_5(df)
    generate_figure_6(df)

    print("\n" + "="*70)
    print(f"All figures saved to: {OUTPUT_DIR}")
    print("="*70)


if __name__ == '__main__':
    main()
