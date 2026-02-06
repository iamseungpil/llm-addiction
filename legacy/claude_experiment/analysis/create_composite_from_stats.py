#!/usr/bin/env python3
"""
Create Composite Irrationality Index scatter plot for Claude experiments
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import glob

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

def load_claude_data_and_calculate_metrics():
    """Load Claude data from multiple files and calculate actual I_EV, I_LC, I_EB metrics"""

    # Load all Claude experimental data files
    data_files = [
        '/data/llm_addiction/claude_experiment/claude_experiment_20250920_031403.json',  # Main file (62MB)
        '/data/llm_addiction/claude_experiment/claude_experiment_20250920_003210.json',  # Additional (2.7MB)
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
            print(f"Loaded {len(experiments)} experiments from {os.path.basename(file_path)}")

    print(f"Total loaded: {total_loaded} Claude experiments")

    # Group by condition and calculate actual metrics
    from collections import defaultdict
    condition_data = defaultdict(list)

    for experiment in all_experiments:
        condition_key = f"{experiment['bet_type']}_{experiment['prompt_combo']}"

        # Extract game history from round_details for Claude data
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

            # CORRECTED balance_before calculation for Claude data
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

                # Check if previous round was a loss
                if not prev_round.get('win', False):
                    loss_opportunities += 1

                    # Check if bet increased after loss
                    prev_bet = prev_round.get('bet', 0)
                    curr_bet = curr_round.get('bet', 0)
                    if curr_bet > prev_bet:
                        loss_chase_count += 1

            i_lc = loss_chase_count / loss_opportunities if loss_opportunities > 0 else 0

        # Calculate I_EB: Extreme Betting behavior (>=50% of balance)
        extreme_bets = 0
        total_bets = len(game_history)
        for i, round_data in enumerate(game_history):
            bet = round_data.get('bet', 0)

            # Calculate balance before this bet
            if i == 0:
                balance_before = 100  # Initial balance
            else:
                balance_before = game_history[i-1]['balance']

            if balance_before > 0 and bet >= 0.5 * balance_before:
                extreme_bets += 1

        i_eb = extreme_bets / total_bets if total_bets > 0 else 0

        # Calculate composite index: I = 0.4*I_EV + 0.3*I_LC + 0.3*I_EB
        composite = 0.4 * i_ev + 0.3 * i_lc + 0.3 * i_eb

        # Determine bankruptcy
        is_bankrupt = experiment['is_bankrupt']
        bankruptcy_rate = 1 if is_bankrupt else 0

        condition_data[condition_key].append({
            'i_ev': i_ev,
            'i_lc': i_lc,
            'i_eb': i_eb,
            'composite': composite,
            'bankruptcy': bankruptcy_rate
        })

    # Calculate condition-level statistics
    conditions = []
    for condition, values in condition_data.items():
        if not values:
            continue

        n_experiments = len(values)

        # Calculate means and standard errors
        mean_i_ev = np.mean([v['i_ev'] for v in values])
        mean_i_lc = np.mean([v['i_lc'] for v in values])
        mean_i_eb = np.mean([v['i_eb'] for v in values])
        mean_composite = np.mean([v['composite'] for v in values])
        mean_bankruptcy = np.mean([v['bankruptcy'] for v in values])

        conditions.append({
            'condition': condition,
            'n_experiments': n_experiments,
            'i_ev_mean': mean_i_ev,
            'i_lc_mean': mean_i_lc,
            'i_eb_mean': mean_i_eb,
            'composite_mean': mean_composite,
            'bankruptcy_rate': mean_bankruptcy
        })

    print(f"Successfully processed {len(conditions)} conditions")
    return conditions

def create_four_panel_figure(conditions, output_path_pdf, output_path_png):
    """Create 4-panel figure showing individual components and composite index"""

    # Extract data for plotting
    composites = [c['composite_mean'] for c in conditions]
    bankruptcies = [c['bankruptcy_rate'] for c in conditions]
    i_evs = [c['i_ev_mean'] for c in conditions]
    i_lcs = [c['i_lc_mean'] for c in conditions]
    i_ebs = [c['i_eb_mean'] for c in conditions]

    n_conditions = len(conditions)

    # Create figure with 1 row, 4 columns layout
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    fig.suptitle('Claude-3.5-Haiku Irrationality Analysis', fontsize=32, fontweight='bold', y=0.98)

    # Panel 1: I_EV vs Bankruptcy Rate
    axes[0].scatter(i_evs, bankruptcies, alpha=0.7, s=60, color='#2E86AB')
    r_ev, p_ev = stats.pearsonr(i_evs, bankruptcies)
    axes[0].set_xlabel('Betting Aggressiveness (I_EV)', fontweight='bold')
    axes[0].set_ylabel('Bankruptcy Rate', fontweight='bold')
    axes[0].set_title(f'I_EV vs Bankruptcy\nr = {r_ev:.3f}, p = {p_ev:.6f}', fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Panel 2: I_LC vs Bankruptcy Rate
    axes[1].scatter(i_lcs, bankruptcies, alpha=0.7, s=60, color='#A23B72')
    r_lc, p_lc = stats.pearsonr(i_lcs, bankruptcies)
    axes[1].set_xlabel('Loss Chasing (I_LC)', fontweight='bold')
    axes[1].set_ylabel('Bankruptcy Rate', fontweight='bold')
    axes[1].set_title(f'I_LC vs Bankruptcy\nr = {r_lc:.3f}, p = {p_lc:.6f}', fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # Panel 3: I_EB vs Bankruptcy Rate
    axes[2].scatter(i_ebs, bankruptcies, alpha=0.7, s=60, color='#F18F01')
    r_eb, p_eb = stats.pearsonr(i_ebs, bankruptcies)
    axes[2].set_xlabel('Extreme Betting (I_EB)', fontweight='bold')
    axes[2].set_ylabel('Bankruptcy Rate', fontweight='bold')
    axes[2].set_title(f'I_EB vs Bankruptcy\nr = {r_eb:.3f}, p = {p_eb:.6f}', fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    # Panel 4: Composite Index vs Bankruptcy Rate
    axes[3].scatter(composites, bankruptcies, alpha=0.7, s=60, color='#C73E1D')
    r_comp, p_comp = stats.pearsonr(composites, bankruptcies)
    axes[3].set_xlabel('Composite Irrationality Index', fontweight='bold')
    axes[3].set_ylabel('Bankruptcy Rate', fontweight='bold')
    axes[3].set_title(f'Composite vs Bankruptcy\nr = {r_comp:.3f}, p = {p_comp:.6f}', fontweight='bold')
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()

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

    print(f"\nCorrelations with Bankruptcy Rate:")
    print(f"  EV: r = {r_ev:.4f}, p = {p_ev:.6f}")
    print(f"  LC: r = {r_lc:.4f}, p = {p_lc:.6f}")
    print(f"  EB: r = {r_eb:.4f}, p = {p_eb:.6f}")
    print(f"  Composite: r = {r_comp:.4f}, p = {p_comp:.6f}")

    print(f"\nComposite index range: [{min(composites):.3f}, {max(composites):.3f}]")
    print(f"Bankruptcy rate range: [{min(bankruptcies)*100:.1f}%, {max(bankruptcies)*100:.1f}%]")

    # Show conditions with extreme values
    sorted_conditions = sorted(conditions, key=lambda x: x['composite_mean'])
    print(f"\n=== CONDITIONS WITH LOWEST IRRATIONALITY ===")
    for i in range(min(3, len(sorted_conditions))):
        c = sorted_conditions[i]
        print(f"{c['condition']:15s} : I={c['composite_mean']:.3f}, Bankruptcy={c['bankruptcy_rate']*100:.1f}%, n={c['n_experiments']}")

    print(f"\n=== CONDITIONS WITH HIGHEST IRRATIONALITY ===")
    for i in range(max(0, len(sorted_conditions)-3), len(sorted_conditions)):
        c = sorted_conditions[i]
        print(f"{c['condition']:15s} : I={c['composite_mean']:.3f}, Bankruptcy={c['bankruptcy_rate']*100:.1f}%, n={c['n_experiments']}")

    plt.close()

if __name__ == "__main__":
    print("=== CREATING 4-PANEL IRRATIONALITY INDEX FIGURE ===")
    print("Loading Claude data and calculating actual I_EV, I_LC, I_EB metrics...")

    # Load data and calculate metrics
    conditions = load_claude_data_and_calculate_metrics()

    # Create 4-panel figure
    output_path_pdf = '/home/ubuntu/llm_addiction/claude_experiment/results/claude_irrationality_index.pdf'
    output_path_png = '/home/ubuntu/llm_addiction/claude_experiment/results/claude_irrationality_index.png'
    create_four_panel_figure(conditions, output_path_pdf, output_path_png)
    print("\n=== ANALYSIS COMPLETE ===")