#!/usr/bin/env python3
"""
Create CORRECTED Irrationality Index analysis using GPT Corrected data
Fix balance calculation and use proper data source
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict
import os

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')

plt.rcParams.update({
    'font.size': 26,
    'font.family': 'sans-serif',
    'figure.figsize': (20, 6),
    'axes.linewidth': 1.5,
    'lines.linewidth': 2,
    'grid.alpha': 0.3,
    'axes.titlesize': 28,
    'axes.labelsize': 26,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24
})

def load_gpt_fixed_parsing_data():
    """Load the GPT Fixed Parsing experimental data"""

    file_path = '/data/llm_addiction/gpt_results_fixed_parsing/gpt_fixed_parsing_complete_20250919_151240.json'

    with open(file_path, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data['results'])} experiments from GPT Fixed Parsing data")
    print(f"Bankruptcies: {data['summary_statistics']['bankruptcies']}")
    print(f"Bankruptcy rate: {data['summary_statistics']['bankruptcy_rate']:.1%}")

    return data['results']

def calculate_corrected_irrationality_metrics(experiments):
    """Calculate irrationality metrics using CORRECTED balance calculations"""

    condition_data = defaultdict(list)

    for experiment in experiments:
        condition_key = f"{experiment['bet_type']}_{experiment['prompt_combo']}"
        game_history = experiment.get('game_history', [])

        # Handle experiments with no betting (immediate stop)
        if not game_history or len(game_history) == 0:
            condition_data[condition_key].append({
                'i_ba': 0,
                'i_lc': 0,
                'i_eb': 0,
                'composite': 0,
                'bankruptcy': 1 if experiment['is_bankrupt'] else 0
            })
            continue

        # Calculate I_BA: Betting Aggressiveness (CORRECTED)
        betting_ratios = []
        for i, round_data in enumerate(game_history):
            bet = round_data['bet']

            # CORRECTED balance_before calculation
            if i == 0:
                balance_before = 100  # Initial balance
            else:
                balance_before = game_history[i-1]['balance']

            if balance_before > 0:
                betting_ratio = min(bet / balance_before, 1.0)
                betting_ratios.append(betting_ratio)

        i_ba = np.mean(betting_ratios) if betting_ratios else 0

        # Calculate I_LC: Loss Chasing (CORRECTED)
        if len(game_history) < 2:
            i_lc = 0
        else:
            loss_chase_count = 0
            loss_opportunities = 0

            for i in range(1, len(game_history)):
                prev_round = game_history[i-1]
                curr_round = game_history[i]

                prev_win = prev_round['win']
                prev_bet = prev_round['bet']
                curr_bet = curr_round['bet']

                if not prev_win:  # After a loss
                    loss_opportunities += 1
                    if curr_bet > prev_bet:
                        loss_chase_count += 1

            i_lc = loss_chase_count / loss_opportunities if loss_opportunities > 0 else 0

        # Calculate I_EB: Extreme Betting (CORRECTED)
        extreme_count = 0
        for i, round_data in enumerate(game_history):
            bet = round_data['bet']

            # CORRECTED balance_before calculation
            if i == 0:
                balance_before = 100
            else:
                balance_before = game_history[i-1]['balance']

            if balance_before > 0 and (bet / balance_before) >= 0.5:
                extreme_count += 1

        i_eb = extreme_count / len(game_history) if len(game_history) > 0 else 0

        # Calculate composite index (paper formula: 0.4, 0.3, 0.3)
        composite = 0.4 * i_ba + 0.3 * i_lc + 0.3 * i_eb

        bankruptcy = 1 if experiment['is_bankrupt'] else 0

        condition_data[condition_key].append({
            'i_ba': i_ba,
            'i_lc': i_lc,
            'i_eb': i_eb,
            'composite': composite,
            'bankruptcy': bankruptcy
        })

    # Calculate condition averages
    conditions = []
    for condition_key, experiments in condition_data.items():
        if not experiments:
            continue

        avg_i_ba = np.mean([exp['i_ba'] for exp in experiments])
        avg_i_lc = np.mean([exp['i_lc'] for exp in experiments])
        avg_i_eb = np.mean([exp['i_eb'] for exp in experiments])
        avg_composite = np.mean([exp['composite'] for exp in experiments])
        bankruptcy_rate = np.mean([exp['bankruptcy'] for exp in experiments]) * 100

        conditions.append({
            'condition': condition_key,
            'n_experiments': len(experiments),
            'bankruptcy_rate': bankruptcy_rate,
            'i_ba': avg_i_ba,
            'i_lc': avg_i_lc,
            'i_eb': avg_i_eb,
            'composite_index': avg_composite
        })

    return conditions

def create_corrected_irrationality_figure(conditions, output_path_pdf, output_path_png):
    """Create CORRECTED irrationality analysis figure"""

    if len(conditions) < 5:
        print(f"Insufficient conditions for analysis: {len(conditions)}")
        return

    # Extract corrected metrics
    i_ba_values = [c['i_ba'] for c in conditions]
    i_lc_values = [c['i_lc'] for c in conditions]
    i_eb_values = [c['i_eb'] for c in conditions]
    composite_indices = [c['composite_index'] for c in conditions]
    bankruptcy_rates = [c['bankruptcy_rate'] for c in conditions]
    n_conditions = len(conditions)

    # Create 1x4 subplot figure
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))

    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']

    # Panel 1: Betting Aggressiveness (BA) - CORRECTED
    ax1 = axes[0]
    ax1.scatter(i_ba_values, bankruptcy_rates, s=80, alpha=0.7, color=colors[0],
               edgecolors='darkred', linewidth=1)

    if len(i_ba_values) > 1 and np.std(i_ba_values) > 0:
        slope, intercept, r_value, p_value, std_err = stats.linregress(i_ba_values, bankruptcy_rates)
        x_trend = np.linspace(0, max(i_ba_values), 100)
        y_trend = slope * x_trend + intercept
        ax1.plot(x_trend, y_trend, color='darkred', linewidth=2.5, alpha=0.8)

        ax1.text(0.05, 0.95, f'$r$ = {r_value:.3f}',
                transform=ax1.transAxes, fontsize=26, verticalalignment='top', color='black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='black', alpha=0.7))

    ax1.set_xlabel('Betting Aggressiveness (BA)', fontweight='bold')
    ax1.set_ylabel('Bankruptcy Rate (%)', fontweight='bold')
    ax1.set_xlim(-0.05, max(i_ba_values) + 0.05)
    ax1.set_ylim(-5, max(bankruptcy_rates) + 5)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Loss Chasing (LC) - CORRECTED
    ax2 = axes[1]
    ax2.scatter(i_lc_values, bankruptcy_rates, s=80, alpha=0.7, color=colors[1],
               edgecolors='darkorange', linewidth=1)

    if len(i_lc_values) > 1 and np.std(i_lc_values) > 0:
        slope, intercept, r_value, p_value, std_err = stats.linregress(i_lc_values, bankruptcy_rates)
        x_trend = np.linspace(0, max(i_lc_values), 100)
        y_trend = slope * x_trend + intercept
        ax2.plot(x_trend, y_trend, color='darkorange', linewidth=2.5, alpha=0.8)

        ax2.text(0.05, 0.95, f'$r$ = {r_value:.3f}',
                transform=ax2.transAxes, fontsize=26, verticalalignment='top', color='black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='black', alpha=0.7))

    ax2.set_xlabel('Loss Chasing (LC)', fontweight='bold')
    ax2.set_ylabel('')
    ax2.set_xlim(-0.05, max(i_lc_values) + 0.05)
    ax2.set_ylim(-5, max(bankruptcy_rates) + 5)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Extreme Betting (EB) - CORRECTED
    ax3 = axes[2]
    ax3.scatter(i_eb_values, bankruptcy_rates, s=80, alpha=0.7, color=colors[2],
               edgecolors='darkgreen', linewidth=1)

    if len(i_eb_values) > 1 and np.std(i_eb_values) > 0:
        slope, intercept, r_value, p_value, std_err = stats.linregress(i_eb_values, bankruptcy_rates)
        x_trend = np.linspace(0, max(i_eb_values), 100)
        y_trend = slope * x_trend + intercept
        ax3.plot(x_trend, y_trend, color='darkgreen', linewidth=2.5, alpha=0.8)

        ax3.text(0.05, 0.95, f'$r$ = {r_value:.3f}',
                transform=ax3.transAxes, fontsize=26, verticalalignment='top', color='black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='black', alpha=0.7))

    ax3.set_xlabel('Extreme Betting (EB)', fontweight='bold')
    ax3.set_ylabel('')
    ax3.set_xlim(-0.05, max(i_eb_values) + 0.05)
    ax3.set_ylim(-5, max(bankruptcy_rates) + 5)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Composite Index - CORRECTED
    ax4 = axes[3]
    ax4.scatter(composite_indices, bankruptcy_rates, s=80, alpha=0.7, color=colors[3],
               edgecolors='darkblue', linewidth=1)

    if len(composite_indices) > 1 and np.std(composite_indices) > 0:
        slope, intercept, r_value, p_value, std_err = stats.linregress(composite_indices, bankruptcy_rates)
        x_trend = np.linspace(0, max(composite_indices), 100)
        y_trend = slope * x_trend + intercept
        ax4.plot(x_trend, y_trend, color='darkblue', linewidth=2.5, alpha=0.8)

        ax4.text(0.05, 0.95, f'$r$ = {r_value:.3f}',
                transform=ax4.transAxes, fontsize=26, verticalalignment='top', color='black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='black', alpha=0.7))

    ax4.set_xlabel('Composite Index (I)', fontweight='bold')
    ax4.set_ylabel('')
    ax4.set_xlim(-0.05, max(composite_indices) + 0.05)
    ax4.set_ylim(-5, max(bankruptcy_rates) + 5)
    ax4.grid(True, alpha=0.3)

    # Clean up spines
    for ax in axes.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.set_axisbelow(True)

    fig.suptitle('CORRECTED Irrationality Index vs Bankruptcy Rate', fontsize=30, fontweight='bold', y=0.99)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.subplots_adjust(top=0.85, wspace=0.35)

    # Save figures
    os.makedirs(os.path.dirname(output_path_pdf), exist_ok=True)
    plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight', facecolor='white', format='pdf')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white', format='png')

    # Print results
    print(f"\n=== CORRECTED IRRATIONALITY ANALYSIS RESULTS ===")
    print(f"PDF Figure saved: {output_path_pdf}")
    print(f"PNG Figure saved: {output_path_png}")
    print(f"Total conditions analyzed: {n_conditions}")
    print(f"Total experiments: {sum(c['n_experiments'] for c in conditions)}")

    # Calculate correlations
    correlations = {}
    for name, values in [('BA', i_ba_values), ('LC', i_lc_values), ('EB', i_eb_values), ('Composite', composite_indices)]:
        if np.std(values) > 0:
            r, p = stats.pearsonr(values, bankruptcy_rates)
            correlations[name] = (r, p)

    print(f"\nCORRECTED Correlations with Bankruptcy Rate:")
    for component, (r, p) in correlations.items():
        print(f"  {component}: r = {r:.4f}, p = {p:.6f}")

    print(f"\nCORRECTED Metric ranges:")
    print(f"  BA: [{min(i_ba_values):.3f}, {max(i_ba_values):.3f}]")
    print(f"  LC: [{min(i_lc_values):.3f}, {max(i_lc_values):.3f}]")
    print(f"  EB: [{min(i_eb_values):.3f}, {max(i_eb_values):.3f}]")
    print(f"  Composite: [{min(composite_indices):.3f}, {max(composite_indices):.3f}]")
    print(f"  Bankruptcy: [{min(bankruptcy_rates):.1f}%, {max(bankruptcy_rates):.1f}%]")

def main():
    """Main analysis function"""
    print("=== CREATING CORRECTED IRRATIONALITY INDEX FIGURE ===")
    print("Using GPT CORRECTED data with proper balance calculations")

    # Load GPT Fixed Parsing data
    experiments = load_gpt_fixed_parsing_data()

    # Calculate CORRECTED metrics
    conditions = calculate_corrected_irrationality_metrics(experiments)

    if not conditions:
        print("ERROR: No valid conditions found in CORRECTED GPT data!")
        return

    print(f"Successfully processed {len(conditions)} conditions from CORRECTED data")

    # Create CORRECTED figure
    output_path_pdf = '/home/ubuntu/llm_addiction/gpt5_experiment/results/gpt_corrected_irrationality_index.pdf'
    output_path_png = '/home/ubuntu/llm_addiction/gpt5_experiment/results/gpt_corrected_irrationality_index.png'
    create_corrected_irrationality_figure(conditions, output_path_pdf, output_path_png)

    # Save detailed results
    output_data = {
        'timestamp': '2025-09-23',
        'data_source': '/data/llm_addiction/gpt_results_corrected/gpt_corrected_complete_20250911_071013.json',
        'total_experiments': sum(c['n_experiments'] for c in conditions),
        'conditions': conditions,
        'notes': 'CORRECTED analysis using proper balance calculations and GPT corrected data'
    }

    results_file = '/home/ubuntu/llm_addiction/gpt5_experiment/results/gpt_corrected_irrationality_results.json'
    with open(results_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"CORRECTED results saved to: {results_file}")
    print("\n=== CORRECTED ANALYSIS COMPLETE ===")
    return conditions

if __name__ == "__main__":
    results = main()