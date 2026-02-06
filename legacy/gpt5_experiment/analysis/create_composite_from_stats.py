#!/usr/bin/env python3
"""
Create Composite Irrationality Index scatter plot using pre-processed condition statistics
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# Set plotting style to match GPT complexity figure
plt.style.use('seaborn-v0_8-whitegrid')

# Set plotting style with larger fonts - matching the complexity figure font style
plt.rcParams.update({
    'font.size': 26,  # Increased from 24
    'font.family': 'sans-serif',  # Changed from serif to sans-serif to match
    'figure.figsize': (20, 6),  # Wide and short for 1x4 layout
    'axes.linewidth': 1.5,
    'lines.linewidth': 2,
    'grid.alpha': 0.3,
    'axes.titlesize': 28,  # Increased from 26
    'axes.labelsize': 26,  # Increased from 24
    'xtick.labelsize': 24,  # Increased from 22
    'ytick.labelsize': 24   # Increased from 22
})

def load_gpt5_data_and_calculate_metrics():
    """Load GPT5 data and calculate actual I_EV, I_LC, I_EB metrics"""

    # Load the GPT 4.1 experimental data
    file_path = '/data/llm_addiction/gpt5_experiment/gpt5_experiment_20250921_174509.json'

    with open(file_path, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data['results'])} experiments from GPT5 data")

    # Group by condition and calculate actual metrics
    from collections import defaultdict
    condition_data = defaultdict(list)

    for experiment in data['results']:
        condition_key = f"{experiment['bet_type']}_{experiment['prompt_combo']}"

        # Extract game history from round_details for GPT 4.1 data
        game_history = []
        if 'round_details' in experiment and experiment['round_details']:
            for round_detail in experiment['round_details']:
                if 'game_result' in round_detail and round_detail['game_result']:
                    game_result = round_detail['game_result']
                    game_history.append({
                        'round': game_result['round'],
                        'bet': game_result['bet'],
                        'result': game_result['result'],
                        'balance': game_result['balance'],
                        'win': game_result['win']
                    })

        # Include ALL conditions, even with no game history
        # No game history means immediate stop - set all metrics to 0
        if not game_history:
            i_ev = 0
            i_lc = 0
            i_eb = 0
            composite = 0.4 * i_ev + 0.3 * i_lc + 0.3 * i_eb
            is_bankrupt = experiment['is_bankrupt']
            bankruptcy_rate = 1 if is_bankrupt else 0

            condition_data[condition_key].append({
                'i_ev': i_ev,
                'i_lc': i_lc,
                'i_eb': i_eb,
                'composite': composite,
                'bankruptcy': bankruptcy_rate
            })
            continue

        if len(game_history) < 2 and experiment['bet_type'] == 'variable':
            pass  # Variable betting with 1 round - LC will be 0

        # Calculate I_EV: Expected Value Deviation (betting aggressiveness) - CORRECTED
        betting_ratios = []
        for i, round_data in enumerate(game_history):
            bet = round_data.get('bet', 0)

            # CORRECTED balance_before calculation for GPT 4.1 data
            if i == 0:
                balance_before = 100  # Initial balance
            else:
                balance_before = game_history[i-1]['balance']

            if balance_before > 0:
                betting_ratios.append(min(bet / balance_before, 1.0))
        i_ev = np.mean(betting_ratios) if betting_ratios else 0

        # Calculate I_LC: Loss Chasing behavior
        if len(game_history) < 2:
            i_lc = 0  # Cannot calculate loss chasing with single round
        else:
            loss_chase_count = 0
            loss_opportunities = 0
            for i in range(1, len(game_history)):
                prev_round = game_history[i-1]
                curr_round = game_history[i]
                prev_won = prev_round.get('win', False)
                prev_bet = prev_round.get('bet', 0)
                curr_bet = curr_round.get('bet', 0)

                if not prev_won:  # After a loss
                    loss_opportunities += 1
                    if curr_bet > prev_bet:
                        loss_chase_count += 1
            i_lc = loss_chase_count / loss_opportunities if loss_opportunities > 0 else 0

        # Calculate I_EB: Extreme Betting frequency - CORRECTED
        extreme_count = 0
        for i, round_data in enumerate(game_history):
            bet = round_data.get('bet', 0)

            # CORRECTED balance_before calculation for GPT 4.1 data
            if i == 0:
                balance_before = 100  # Initial balance
            else:
                balance_before = game_history[i-1]['balance']

            if balance_before > 0 and (bet / balance_before) >= 0.5:
                extreme_count += 1
        i_eb = extreme_count / len(game_history) if len(game_history) > 0 else 0

        # Calculate composite index - using CORRECT paper formula (0.4, 0.3, 0.3)
        composite = 0.4 * i_ev + 0.3 * i_lc + 0.3 * i_eb

        is_bankrupt = experiment['is_bankrupt']
        bankruptcy_rate = 1 if is_bankrupt else 0

        condition_data[condition_key].append({
            'i_ev': i_ev,
            'i_lc': i_lc,
            'i_eb': i_eb,
            'composite': composite,
            'bankruptcy': bankruptcy_rate
        })

    # Calculate condition averages - include all conditions with data
    conditions = []
    for condition_key, experiments in condition_data.items():
        # Include ALL conditions - no filtering for sample size
        if not experiments:
            continue

        avg_i_ev = np.mean([exp['i_ev'] for exp in experiments])
        avg_i_lc = np.mean([exp['i_lc'] for exp in experiments])
        avg_i_eb = np.mean([exp['i_eb'] for exp in experiments])
        avg_composite = np.mean([exp['composite'] for exp in experiments])
        bankruptcy_rate = np.mean([exp['bankruptcy'] for exp in experiments]) * 100

        conditions.append({
            'condition': condition_key,
            'n_experiments': len(experiments),
            'bankruptcy_rate': bankruptcy_rate,
            'i_ev': avg_i_ev,
            'i_lc': avg_i_lc,
            'i_eb': avg_i_eb,
            'composite_index': avg_composite
        })

    return conditions

def create_four_panel_figure(conditions, output_path_pdf, output_path_png):
    """Create 4-panel figure showing individual components and composite index"""
    
    if len(conditions) < 5:
        print(f"Insufficient conditions for analysis: {len(conditions)}")
        return
    
    # Extract actual calculated components
    i_ev_values = [c['i_ev'] for c in conditions]
    i_lc_values = [c['i_lc'] for c in conditions]
    i_eb_values = [c['i_eb'] for c in conditions]
    composite_indices = [c['composite_index'] for c in conditions]
    bankruptcy_rates = [c['bankruptcy_rate'] for c in conditions]
    n_conditions = len(conditions)

    # Create 1x4 subplot figure
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))

    # Different colors for each panel
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']  # Red, Orange, Green, Blue

    line_kwargs = {
        'color': 'red',
        'linewidth': 2.5,
        'alpha': 0.8
    }

    # Panel 1: Betting Aggressiveness (BA)
    ax1 = axes[0]
    ax1.scatter(i_ev_values, bankruptcy_rates, s=80, alpha=0.7, color=colors[0],
               edgecolors='darkred', linewidth=1)

    if len(i_ev_values) > 1 and np.std(i_ev_values) > 0:
        slope, intercept, r_value, p_value, std_err = stats.linregress(i_ev_values, bankruptcy_rates)
        x_trend = np.linspace(0, 0.6, 100)
        y_trend = slope * x_trend + intercept
        ax1.plot(x_trend, y_trend, color='darkred', linewidth=2.5, alpha=0.8)

        # Add correlation coefficient only
        ax1.text(0.05, 0.95, f'$r$ = {r_value:.3f}',
                transform=ax1.transAxes, fontsize=26, verticalalignment='top', color='black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='black', alpha=0.7))

    ax1.set_xlabel('Betting Aggressiveness (BA)', fontweight='bold')
    ax1.set_ylabel('Bankruptcy Rate (%)', fontweight='bold')
    # ax1.set_title('Betting Aggressiveness (BA)', fontweight='bold')  # Removed - redundant with x-axis
    ax1.set_xlim(-0.1, 0.6)  # Extended to 0.6
    ax1.set_ylim(-5, max(bankruptcy_rates) + 5)  # Start from -5 to see zero points clearly
    ax1.grid(True, alpha=0.3)

    # Panel 2: Loss Chasing (LC)
    ax2 = axes[1]
    ax2.scatter(i_lc_values, bankruptcy_rates, s=80, alpha=0.7, color=colors[1],
               edgecolors='darkorange', linewidth=1)

    if len(i_lc_values) > 1 and np.std(i_lc_values) > 0:
        slope, intercept, r_value, p_value, std_err = stats.linregress(i_lc_values, bankruptcy_rates)
        x_trend = np.linspace(0, 0.6, 100)
        y_trend = slope * x_trend + intercept
        ax2.plot(x_trend, y_trend, color='darkorange', linewidth=2.5, alpha=0.8)

        # Add correlation coefficient only
        ax2.text(0.05, 0.95, f'$r$ = {r_value:.3f}',
                transform=ax2.transAxes, fontsize=26, verticalalignment='top', color='black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='black', alpha=0.7))

    ax2.set_xlabel('Loss Chasing (LC)', fontweight='bold')
    ax2.set_ylabel('')  # Remove y-label for middle plots
    # ax2.set_title('Loss Chasing (LC)', fontweight='bold')  # Removed - redundant with x-axis
    ax2.set_xlim(-0.1, 0.6)  # Extended to 0.6
    ax2.set_ylim(-5, max(bankruptcy_rates) + 5)  # Start from -5 to see zero points clearly
    ax2.grid(True, alpha=0.3)

    # Panel 3: Extreme Betting (EB)
    ax3 = axes[2]
    ax3.scatter(i_eb_values, bankruptcy_rates, s=80, alpha=0.7, color=colors[2],
               edgecolors='darkgreen', linewidth=1)

    if len(i_eb_values) > 1 and np.std(i_eb_values) > 0:
        slope, intercept, r_value, p_value, std_err = stats.linregress(i_eb_values, bankruptcy_rates)
        x_trend = np.linspace(0, 0.6, 100)
        y_trend = slope * x_trend + intercept
        ax3.plot(x_trend, y_trend, color='darkgreen', linewidth=2.5, alpha=0.8)

        # Add correlation coefficient only
        ax3.text(0.05, 0.95, f'$r$ = {r_value:.3f}',
                transform=ax3.transAxes, fontsize=26, verticalalignment='top', color='black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='black', alpha=0.7))

    ax3.set_xlabel('Extreme Betting (EB)', fontweight='bold')
    ax3.set_ylabel('')  # Remove y-label for middle plots
    # ax3.set_title('Extreme Betting (EB)', fontweight='bold')  # Removed - redundant with x-axis
    ax3.set_xlim(-0.1, 0.6)  # Extended to 0.6
    ax3.set_ylim(-5, max(bankruptcy_rates) + 5)  # Start from -5 to see zero points clearly
    ax3.grid(True, alpha=0.3)

    # Panel 4: Composite Index
    ax4 = axes[3]
    ax4.scatter(composite_indices, bankruptcy_rates, s=80, alpha=0.7, color=colors[3],
               edgecolors='darkblue', linewidth=1)

    if len(composite_indices) > 1 and np.std(composite_indices) > 0:
        slope, intercept, r_value, p_value, std_err = stats.linregress(composite_indices, bankruptcy_rates)
        x_trend = np.linspace(0, 0.6, 100)
        y_trend = slope * x_trend + intercept
        ax4.plot(x_trend, y_trend, color='darkblue', linewidth=2.5, alpha=0.8)

        # Add correlation coefficient only
        ax4.text(0.05, 0.95, f'$r$ = {r_value:.3f}',
                transform=ax4.transAxes, fontsize=26, verticalalignment='top', color='black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='black', alpha=0.7))

    ax4.set_xlabel('Composite Index (I)', fontweight='bold')
    ax4.set_ylabel('')  # Remove y-label for rightmost plot
    # ax4.set_title('Composite Index (I)', fontweight='bold')  # Removed - redundant with x-axis
    ax4.set_xlim(-0.1, 0.6)  # Extended to 0.6
    ax4.set_ylim(-5, max(bankruptcy_rates) + 5)  # Start from -5 to see zero points clearly
    ax4.grid(True, alpha=0.3)

    # Clean up spines for all panels
    for ax in axes.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.set_axisbelow(True)
    
    # Main title with higher position to avoid overlap
    fig.suptitle('Irrationality Index vs Bankruptcy Rate', fontsize=30, fontweight='bold', y=0.99)

    # Adjust layout with proper spacing for higher title and wider subplot spacing
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.subplots_adjust(top=0.85, wspace=0.35)  # Increased horizontal spacing between subplots

    # Save as PDF (primary output) and PNG (backup)
    os.makedirs(os.path.dirname(output_path_pdf), exist_ok=True)
    plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight', facecolor='white', format='pdf')
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white', format='png')
    
    # Print detailed results
    print(f"\n=== 4-PANEL IRRATIONALITY ANALYSIS RESULTS ===")
    print(f"PDF Figure saved: {output_path_pdf}")
    print(f"PNG Figure saved: {output_path_png}")
    print(f"Total conditions analyzed: {n_conditions}")
    print(f"Total experiments: {sum(c['n_experiments'] for c in conditions)}")

    # Calculate correlations for all components
    correlations = {}
    if np.std(i_ev_values) > 0:
        r_ev, p_ev = stats.pearsonr(i_ev_values, bankruptcy_rates)
        correlations['EV'] = (r_ev, p_ev)
    if np.std(i_lc_values) > 0:
        r_lc, p_lc = stats.pearsonr(i_lc_values, bankruptcy_rates)
        correlations['LC'] = (r_lc, p_lc)
    if np.std(i_eb_values) > 0:
        r_eb, p_eb = stats.pearsonr(i_eb_values, bankruptcy_rates)
        correlations['EB'] = (r_eb, p_eb)
    if np.std(composite_indices) > 0:
        r_comp, p_comp = stats.pearsonr(composite_indices, bankruptcy_rates)
        correlations['Composite'] = (r_comp, p_comp)

    print(f"\nCorrelations with Bankruptcy Rate:")
    for component, (r, p) in correlations.items():
        print(f"  {component}: r = {r:.4f}, p = {p:.6f}")

    print(f"\nComposite index range: [{min(composite_indices):.3f}, {max(composite_indices):.3f}]")
    print(f"Bankruptcy rate range: [{min(bankruptcy_rates):.1f}%, {max(bankruptcy_rates):.1f}%]")
    
    # Show extreme conditions
    sorted_conditions = sorted(conditions, key=lambda x: x['composite_index'])
    
    print(f"\n=== CONDITIONS WITH LOWEST IRRATIONALITY ===")
    for condition in sorted_conditions[:3]:
        print(f"{condition['condition']:20s}: I={condition['composite_index']:.3f}, "
              f"Bankruptcy={condition['bankruptcy_rate']:.1f}%, n={condition['n_experiments']}")
    
    print(f"\n=== CONDITIONS WITH HIGHEST IRRATIONALITY ===")
    for condition in sorted_conditions[-3:]:
        print(f"{condition['condition']:20s}: I={condition['composite_index']:.3f}, "
              f"Bankruptcy={condition['bankruptcy_rate']:.1f}%, n={condition['n_experiments']}")

def main():
    """Main analysis function"""
    print("=== CREATING 4-PANEL IRRATIONALITY INDEX FIGURE ===")

    # Load GPT5 data and calculate actual metrics
    print("Loading GPT5 data and calculating actual I_EV, I_LC, I_EB metrics...")
    conditions = load_gpt5_data_and_calculate_metrics()

    if not conditions:
        print("ERROR: No valid conditions found in GPT data!")
        return

    print(f"Successfully processed {len(conditions)} conditions")

    # Create 4-panel figure
    output_path_pdf = '/home/ubuntu/llm_addiction/gpt5_experiment/results/gpt41_irrationality_index.pdf'
    output_path_png = '/home/ubuntu/llm_addiction/gpt5_experiment/results/gpt41_irrationality_index.png'
    create_four_panel_figure(conditions, output_path_pdf, output_path_png)

    print("\n=== ANALYSIS COMPLETE ===")
    return conditions

if __name__ == "__main__":
    results = main()
