#!/usr/bin/env python3
"""
LLaMA Complexity Trends Analysis
Analyze prompt complexity effects on behavior
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18
})

def load_llama_data():
    """Load LLaMA data"""
    print("Loading LLaMA data...")
    with open('/data/llm_addiction/experiment_0_llama_corrected/final_llama_20251004_021106.json', 'r') as f:
        data = json.load(f)
    return data['results']

def analyze_complexity(experiments):
    """Analyze prompt complexity effects"""
    print("Analyzing complexity trends...")

    complexity_data = []
    for exp in experiments:
        prompt_combo = exp.get('prompt_combo', 'BASE')
        if not prompt_combo or prompt_combo == 'BASE':
            complexity = 0
        else:
            complexity = sum(1 for comp in ['G', 'M', 'P', 'R', 'W'] if comp in prompt_combo)

        complexity_data.append({
            'complexity': complexity,
            'prompt': prompt_combo,
            'is_bankrupt': exp['outcome'] == 'bankruptcy',
            'total_rounds': exp['total_rounds'],
            'total_bet': sum(h['bet'] for h in exp.get('history', []))
        })

    df = pd.DataFrame(complexity_data)

    # Group by complexity
    complexity_stats = []
    for complexity in sorted(df['complexity'].unique()):
        subset = df[df['complexity'] == complexity]
        complexity_stats.append({
            'complexity': complexity,
            'n': len(subset),
            'bankruptcy_rate': subset['is_bankrupt'].mean() * 100,
            'bankruptcy_se': subset['is_bankrupt'].sem() * 100,
            'avg_rounds': subset['total_rounds'].mean(),
            'rounds_se': subset['total_rounds'].sem(),
            'avg_bet': subset['total_bet'].mean(),
            'bet_se': subset['total_bet'].sem()
        })

    return complexity_stats

def create_complexity_figure(stats):
    """Create complexity trends figure"""
    print("Creating complexity figure...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    complexities = [s['complexity'] for s in stats]

    # Plot 1: Bankruptcy Rate
    ax = axes[0]
    y = [s['bankruptcy_rate'] for s in stats]
    yerr = [s['bankruptcy_se'] for s in stats]
    ax.errorbar(complexities, y, yerr=yerr, marker='o', markersize=10,
                linewidth=2, capsize=5, capthick=2, color='#2E86AB')
    ax.set_xlabel('Prompt Complexity', fontsize=20)
    ax.set_ylabel('Bankruptcy Rate (%)', fontsize=20)
    ax.set_title('LLaMA: Complexity vs Bankruptcy', fontsize=22)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(complexities)

    # Plot 2: Average Rounds
    ax = axes[1]
    y = [s['avg_rounds'] for s in stats]
    yerr = [s['rounds_se'] for s in stats]
    ax.errorbar(complexities, y, yerr=yerr, marker='s', markersize=10,
                linewidth=2, capsize=5, capthick=2, color='#A23B72')
    ax.set_xlabel('Prompt Complexity', fontsize=20)
    ax.set_ylabel('Average Rounds', fontsize=20)
    ax.set_title('LLaMA: Complexity vs Rounds', fontsize=22)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(complexities)

    # Plot 3: Total Bet
    ax = axes[2]
    y = [s['avg_bet'] for s in stats]
    yerr = [s['bet_se'] for s in stats]
    ax.errorbar(complexities, y, yerr=yerr, marker='^', markersize=10,
                linewidth=2, capsize=5, capthick=2, color='#F18F01')
    ax.set_xlabel('Prompt Complexity', fontsize=20)
    ax.set_ylabel('Average Total Bet ($)', fontsize=20)
    ax.set_title('LLaMA: Complexity vs Betting', fontsize=22)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(complexities)

    plt.tight_layout()
    output_path = '/home/ubuntu/llm_addiction/experiment_0_llama_gemma_restart/analysis/figures/llama/complexity_trends.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def statistical_tests(complexity_stats):
    """Perform ANOVA and correlation tests"""
    print("\nStatistical Tests:")

    complexities = [s['complexity'] for s in complexity_stats]
    bankruptcy_rates = [s['bankruptcy_rate'] for s in complexity_stats]

    # Correlation
    if len(complexities) > 2:
        r, p = stats.pearsonr(complexities, bankruptcy_rates)
        print(f"  Correlation (Complexity vs Bankruptcy): r={r:.3f}, p={p:.3f}")

    # Print summary
    print("\nComplexity Summary:")
    for s in complexity_stats:
        print(f"  Level {s['complexity']}: {s['bankruptcy_rate']:.1f}% bankruptcy (n={s['n']})")

def save_csv(complexity_stats):
    """Save to CSV"""
    import csv
    output_path = '/home/ubuntu/llm_addiction/experiment_0_llama_gemma_restart/analysis/figures/llama/complexity_trends.csv'

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=complexity_stats[0].keys())
        writer.writeheader()
        writer.writerows(complexity_stats)

    print(f"Saved: {output_path}")

if __name__ == '__main__':
    experiments = load_llama_data()
    complexity_stats = analyze_complexity(experiments)
    create_complexity_figure(complexity_stats)
    statistical_tests(complexity_stats)
    save_csv(complexity_stats)
    print("✅ LLaMA Complexity Trends analysis complete!")
