#!/usr/bin/env python3
"""
LLaMA Component Effects Analysis
Analyze individual prompt component effects (G, M, P, R, W)
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
    'font.size': 22,
    'axes.titlesize': 24,
    'axes.labelsize': 22,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20
})

def load_llama_data():
    """Load LLaMA data"""
    print("Loading LLaMA data...")
    with open('/data/llm_addiction/experiment_0_llama_corrected/final_llama_20251004_021106.json', 'r') as f:
        data = json.load(f)
    return data['results']

def analyze_component_effects(experiments):
    """Analyze individual component effects"""
    print("Analyzing component effects...")

    components = ['G', 'M', 'P', 'R', 'W']
    component_names = {
        'G': 'G (Goal)',
        'M': 'M (Maximize)',
        'P': 'P (Probability)',
        'R': 'H (Hidden)',  # R → H for consistency
        'W': 'W (Winnings)'
    }

    results = {}

    for component in components:
        with_component = []
        without_component = []

        for exp in experiments:
            prompt_combo = exp.get('prompt_combo', 'BASE')
            history = exp.get('history', [])

            metrics = {
                'is_bankrupt': exp['outcome'] == 'bankruptcy',
                'total_rounds': exp['total_rounds'],
                'total_bet': sum(h['bet'] for h in history) if history else 0
            }

            if component in prompt_combo:
                with_component.append(metrics)
            else:
                without_component.append(metrics)

        # Calculate differences
        if with_component and without_component:
            with_df = pd.DataFrame(with_component)
            without_df = pd.DataFrame(without_component)

            # Bankruptcy rate
            bankruptcy_with = with_df['is_bankrupt'].mean() * 100
            bankruptcy_without = without_df['is_bankrupt'].mean() * 100
            bankruptcy_diff = bankruptcy_with - bankruptcy_without
            bankruptcy_se = np.sqrt(with_df['is_bankrupt'].sem()**2 + without_df['is_bankrupt'].sem()**2) * 100

            # Rounds
            rounds_with = with_df['total_rounds'].mean()
            rounds_without = without_df['total_rounds'].mean()
            rounds_diff = rounds_with - rounds_without
            rounds_se = np.sqrt(with_df['total_rounds'].sem()**2 + without_df['total_rounds'].sem()**2)

            # Total bet
            bet_with = with_df['total_bet'].mean()
            bet_without = without_df['total_bet'].mean()
            bet_diff = bet_with - bet_without
            bet_se = np.sqrt(with_df['total_bet'].sem()**2 + without_df['total_bet'].sem()**2)

            # T-tests
            _, bankruptcy_p = stats.ttest_ind(with_df['is_bankrupt'], without_df['is_bankrupt'])
            _, rounds_p = stats.ttest_ind(with_df['total_rounds'], without_df['total_rounds'])
            _, bet_p = stats.ttest_ind(with_df['total_bet'], without_df['total_bet'])

            results[component] = {
                'name': component_names[component],
                'bankruptcy_diff': bankruptcy_diff,
                'bankruptcy_se': bankruptcy_se,
                'bankruptcy_p': bankruptcy_p,
                'rounds_diff': rounds_diff,
                'rounds_se': rounds_se,
                'rounds_p': rounds_p,
                'bet_diff': bet_diff,
                'bet_se': bet_se,
                'bet_p': bet_p,
                'n_with': len(with_component),
                'n_without': len(without_component)
            }

    return results

def create_component_figure(results):
    """Create component effects figure"""
    print("Creating component effects figure...")

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    components = ['G', 'M', 'P', 'R', 'W']
    x_pos = np.arange(len(components))

    # Plot 1: Bankruptcy Rate Difference
    ax = axes[0]
    y = [results[c]['bankruptcy_diff'] for c in components]
    yerr = [results[c]['bankruptcy_se'] for c in components]
    colors = ['#2E86AB' if val > 0 else '#A23B72' for val in y]

    bars = ax.bar(x_pos, y, yerr=yerr, capsize=5, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.5)
    ax.set_xlabel('Prompt Component', fontsize=22)
    ax.set_ylabel('Δ Bankruptcy Rate (%)', fontsize=22)
    ax.set_title('LLaMA: Component Effects on Bankruptcy', fontsize=24)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([results[c]['name'].split()[0] for c in components])
    ax.grid(True, alpha=0.3, axis='y')

    # Add significance stars
    for i, c in enumerate(components):
        p = results[c]['bankruptcy_p']
        if p < 0.001:
            ax.text(i, y[i] + yerr[i], '***', ha='center', va='bottom', fontsize=20)
        elif p < 0.01:
            ax.text(i, y[i] + yerr[i], '**', ha='center', va='bottom', fontsize=20)
        elif p < 0.05:
            ax.text(i, y[i] + yerr[i], '*', ha='center', va='bottom', fontsize=20)

    # Plot 2: Rounds Difference
    ax = axes[1]
    y = [results[c]['rounds_diff'] for c in components]
    yerr = [results[c]['rounds_se'] for c in components]
    colors = ['#F18F01' if val > 0 else '#C73E1D' for val in y]

    bars = ax.bar(x_pos, y, yerr=yerr, capsize=5, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.5)
    ax.set_xlabel('Prompt Component', fontsize=22)
    ax.set_ylabel('Δ Average Rounds', fontsize=22)
    ax.set_title('LLaMA: Component Effects on Rounds', fontsize=24)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([results[c]['name'].split()[0] for c in components])
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Bet Difference
    ax = axes[2]
    y = [results[c]['bet_diff'] for c in components]
    yerr = [results[c]['bet_se'] for c in components]
    colors = ['#6A994E' if val > 0 else '#BC4749' for val in y]

    bars = ax.bar(x_pos, y, yerr=yerr, capsize=5, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.5)
    ax.set_xlabel('Prompt Component', fontsize=22)
    ax.set_ylabel('Δ Total Bet ($)', fontsize=22)
    ax.set_title('LLaMA: Component Effects on Betting', fontsize=24)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([results[c]['name'].split()[0] for c in components])
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = '/home/ubuntu/llm_addiction/experiment_0_llama_gemma_restart/analysis/figures/llama/component_effects.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def print_summary(results):
    """Print summary statistics"""
    print("\nComponent Effects Summary:")
    print("-" * 80)
    print(f"{'Component':<12} {'Δ Bankruptcy':<15} {'p-value':<10} {'n (with/without)'}")
    print("-" * 80)

    for comp in ['G', 'M', 'P', 'R', 'W']:
        r = results[comp]
        sig = '***' if r['bankruptcy_p'] < 0.001 else '**' if r['bankruptcy_p'] < 0.01 else '*' if r['bankruptcy_p'] < 0.05 else ''
        print(f"{r['name']:<12} {r['bankruptcy_diff']:>6.2f}%{sig:<7} p={r['bankruptcy_p']:.4f}  {r['n_with']}/{r['n_without']}")

def save_csv(results):
    """Save to CSV"""
    import csv
    output_path = '/home/ubuntu/llm_addiction/experiment_0_llama_gemma_restart/analysis/figures/llama/component_effects.csv'

    rows = []
    for comp in ['G', 'M', 'P', 'R', 'W']:
        r = results[comp]
        rows.append({
            'component': r['name'],
            'bankruptcy_diff': r['bankruptcy_diff'],
            'bankruptcy_se': r['bankruptcy_se'],
            'bankruptcy_p': r['bankruptcy_p'],
            'rounds_diff': r['rounds_diff'],
            'rounds_se': r['rounds_se'],
            'rounds_p': r['rounds_p'],
            'bet_diff': r['bet_diff'],
            'bet_se': r['bet_se'],
            'bet_p': r['bet_p'],
            'n_with': r['n_with'],
            'n_without': r['n_without']
        })

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {output_path}")

if __name__ == '__main__':
    experiments = load_llama_data()
    results = analyze_component_effects(experiments)
    create_component_figure(results)
    print_summary(results)
    save_csv(results)
    print("✅ LLaMA Component Effects analysis complete!")
