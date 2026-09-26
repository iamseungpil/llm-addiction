#!/usr/bin/env python3
"""
Create Gemini-only complexity figure with statistical significance testing
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')

# Set larger font sizes
plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18
})

def create_gemini_only_complexity_figure():
    """Create complexity figure with only Gemini data and statistical tests"""
    print("Creating Gemini-only complexity figure with statistical analysis...")

    # Load all Gemini data files
    data_files = [
        '/data/llm_addiction/gemini_experiment/gemini_experiment_20250920_042809.json',  # Main file (41MB)
        '/data/llm_addiction/gemini_experiment/gemini_experiment_20250920_024529.json',  # Additional (851KB)
    ]

    all_experiments = []
    total_loaded = 0

    for file_path in data_files:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
            experiments = data.get('results', [])
            all_experiments.extend(experiments)
            total_loaded += len(experiments)

    print(f"Analyzing {total_loaded} Gemini-2.5-Flash experiments")

    # Process Gemini data
    gemini_analysis = []
    for result in all_experiments:
        prompt_combo = result.get('prompt_combo', 'BASE')
        if not prompt_combo or prompt_combo == 'BASE':
            complexity = 0
        else:
            complexity = sum(1 for comp in ['G', 'M', 'P', 'R', 'W'] if comp in prompt_combo)

        gemini_analysis.append({
            'complexity': complexity,
            'prompt': prompt_combo,
            'is_bankrupt': result.get('is_bankrupt', False),
            'total_rounds': result.get('total_rounds', 0),
            'total_bet': result.get('total_bet', 0)
        })

    # Convert to DataFrame for analysis
    gemini_df = pd.DataFrame(gemini_analysis)

    # Group by complexity and calculate metrics
    complexity_stats = []
    for complexity in sorted(gemini_df['complexity'].unique()):
        subset = gemini_df[gemini_df['complexity'] == complexity]
        n_experiments = len(subset)
        bankruptcy_rate = subset['is_bankrupt'].mean() * 100
        avg_rounds = subset['total_rounds'].mean()
        avg_bet = subset['total_bet'].mean()

        complexity_stats.append({
            'complexity': complexity,
            'n_experiments': n_experiments,
            'bankruptcy_rate': bankruptcy_rate,
            'avg_rounds': avg_rounds,
            'avg_bet': avg_bet
        })

    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Gemini-2.5-Flash: Prompt Complexity Effects', fontsize=24, fontweight='bold')

    complexities = [stat['complexity'] for stat in complexity_stats]
    bankruptcy_rates = [stat['bankruptcy_rate'] for stat in complexity_stats]
    avg_rounds = [stat['avg_rounds'] for stat in complexity_stats]
    avg_bets = [stat['avg_bet'] for stat in complexity_stats]
    n_experiments = [stat['n_experiments'] for stat in complexity_stats]

    # Panel 1: Bankruptcy Rate vs Complexity
    bars1 = ax1.bar(complexities, bankruptcy_rates, alpha=0.7, color='#2E86AB')
    ax1.set_xlabel('Prompt Complexity', fontweight='bold')
    ax1.set_ylabel('Bankruptcy Rate (%)', fontweight='bold')
    ax1.set_title('Bankruptcy vs Complexity', fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Add sample size labels on bars
    for i, (bar, n) in enumerate(zip(bars1, n_experiments)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'N={n}', ha='center', va='bottom', fontsize=14)

    # Panel 2: Average Rounds vs Complexity
    bars2 = ax2.bar(complexities, avg_rounds, alpha=0.7, color='#A23B72')
    ax2.set_xlabel('Prompt Complexity', fontweight='bold')
    ax2.set_ylabel('Average Rounds', fontweight='bold')
    ax2.set_title('Rounds vs Complexity', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Average Total Bet vs Complexity
    bars3 = ax3.bar(complexities, avg_bets, alpha=0.7, color='#F18F01')
    ax3.set_xlabel('Prompt Complexity', fontweight='bold')
    ax3.set_ylabel('Average Total Bet ($)', fontweight='bold')
    ax3.set_title('Total Bet vs Complexity', fontweight='bold')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # Statistical analysis
    print("\nStatistical Analysis:")
    print("===================")

    # Linear regression for bankruptcy rate vs complexity
    slope_br, intercept_br, r_br, p_br, se_br = stats.linregress(complexities, bankruptcy_rates)
    print(f"Bankruptcy Rate vs Complexity:")
    print(f"  Linear regression: y = {slope_br:.3f}x + {intercept_br:.3f}")
    print(f"  Correlation coefficient (r): {r_br:.3f}")
    print(f"  P-value: {p_br:.6f}")
    print(f"  R-squared: {r_br**2:.3f}")

    # Linear regression for avg rounds vs complexity
    slope_r, intercept_r, r_r, p_r, se_r = stats.linregress(complexities, avg_rounds)
    print(f"\nAverage Rounds vs Complexity:")
    print(f"  Linear regression: y = {slope_r:.3f}x + {intercept_r:.3f}")
    print(f"  Correlation coefficient (r): {r_r:.3f}")
    print(f"  P-value: {p_r:.6f}")
    print(f"  R-squared: {r_r**2:.3f}")

    # Linear regression for avg bet vs complexity
    slope_b, intercept_b, r_b, p_b, se_b = stats.linregress(complexities, avg_bets)
    print(f"\nAverage Total Bet vs Complexity:")
    print(f"  Linear regression: y = {slope_b:.3f}x + {intercept_b:.3f}")
    print(f"  Correlation coefficient (r): {r_b:.3f}")
    print(f"  P-value: {p_b:.6f}")
    print(f"  R-squared: {r_b**2:.3f}")

    # Chi-square test for bankruptcy distribution
    bankruptcy_counts = gemini_df.groupby('complexity')['is_bankrupt'].sum()
    total_counts = gemini_df.groupby('complexity').size()
    non_bankruptcy_counts = total_counts - bankruptcy_counts

    if len(bankruptcy_counts) > 1 and bankruptcy_counts.sum() > 0:
        contingency_table = np.array([bankruptcy_counts.values, non_bankruptcy_counts.values])
        chi2, p_chi2 = stats.chi2_contingency(contingency_table)[:2]
        print(f"\nChi-square test for bankruptcy distribution:")
        print(f"  Chi-square statistic: {chi2:.3f}")
        print(f"  P-value: {p_chi2:.6f}")

    # Save figure
    output_path_pdf = '/home/ubuntu/llm_addiction/gemini_experiment/results/gemini_complexity_trend.pdf'
    output_path_png = '/home/ubuntu/llm_addiction/gemini_experiment/results/gemini_complexity_trend.png'

    os.makedirs(os.path.dirname(output_path_pdf), exist_ok=True)
    plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white')

    print(f"\n✅ Gemini-only complexity figure created with statistical analysis!")

    # Print summary table
    print(f"\nComplexity Analysis Summary:")
    print(f"===========================")
    print(f"{'Complexity':<11} {'N':<7} {'Bankruptcy %':<13} {'Avg Rounds':<11} {'Avg Bet ($)':<11}")
    print("-" * 60)
    for stat in complexity_stats:
        print(f"{stat['complexity']:<11} {stat['n_experiments']:<7} {stat['bankruptcy_rate']:<13.1f} {stat['avg_rounds']:<11.1f} {stat['avg_bet']:<11.1f}")

    plt.close()
    return complexity_stats

if __name__ == "__main__":
    create_gemini_only_complexity_figure()