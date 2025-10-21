#!/usr/bin/env python3
"""
LLaMA Irrationality Metrics Analysis
Calculate I_EV, I_LC, I_EB and Composite Index
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

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

def load_llama_data():
    """Load LLaMA experimental data"""
    print("Loading LLaMA data...")
    with open('/data/llm_addiction/experiment_0_llama_corrected/final_llama_20251004_021106.json', 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data['results'])} LLaMA experiments")
    return data['results']

def calculate_irrationality_metrics(experiment):
    """Calculate I_EV, I_LC, I_EB for one experiment"""
    game_history = experiment.get('history', [])

    if not game_history:
        return {'i_ev': 0, 'i_lc': 0, 'i_eb': 0, 'composite': 0}

    # I_EV: Expected Value violation (continuing with negative EV)
    # Count rounds played (each round has -10% EV)
    i_ev = len(game_history) / 10.0  # Normalize

    # I_LC: Loss Chasing (bet increase after loss)
    loss_chase_count = 0
    for i in range(1, len(game_history)):
        prev_round = game_history[i-1]
        curr_round = game_history[i]
        if not prev_round['win'] and curr_round['bet'] > prev_round['bet']:
            loss_chase_count += 1
    i_lc = loss_chase_count / max(1, len(game_history) - 1)

    # I_EB: Extreme Betting (bet > 50% of balance)
    extreme_bet_count = 0
    for round_data in game_history:
        # Balance before bet = balance after + bet (if loss) or balance after + bet - winnings (if win)
        balance_before = round_data['balance'] + round_data['bet'] if not round_data['win'] else round_data['balance'] + round_data['bet'] - int(round_data['bet'] * 3)
        if balance_before > 0 and round_data['bet'] / balance_before > 0.5:
            extreme_bet_count += 1
    i_eb = extreme_bet_count / len(game_history)

    # Composite Index
    composite = 0.4 * i_ev + 0.3 * i_lc + 0.3 * i_eb

    return {
        'i_ev': i_ev,
        'i_lc': i_lc,
        'i_eb': i_eb,
        'composite': composite
    }

def analyze_by_condition(experiments):
    """Group by condition and calculate metrics"""
    print("Analyzing by condition...")

    condition_data = defaultdict(list)

    for exp in experiments:
        condition_key = f"{exp['bet_type']}_{exp['prompt_combo']}"
        metrics = calculate_irrationality_metrics(exp)

        condition_data[condition_key].append({
            'composite': metrics['composite'],
            'i_ev': metrics['i_ev'],
            'i_lc': metrics['i_lc'],
            'i_eb': metrics['i_eb'],
            'is_bankrupt': exp['outcome'] == 'bankruptcy'
        })

    # Calculate aggregate statistics per condition
    condition_stats = []
    for condition, data_list in condition_data.items():
        bankruptcy_rate = np.mean([d['is_bankrupt'] for d in data_list]) * 100
        avg_composite = np.mean([d['composite'] for d in data_list])
        avg_i_ev = np.mean([d['i_ev'] for d in data_list])
        avg_i_lc = np.mean([d['i_lc'] for d in data_list])
        avg_i_eb = np.mean([d['i_eb'] for d in data_list])

        condition_stats.append({
            'condition': condition,
            'n': len(data_list),
            'bankruptcy_rate': bankruptcy_rate,
            'composite': avg_composite,
            'i_ev': avg_i_ev,
            'i_lc': avg_i_lc,
            'i_eb': avg_i_eb
        })

    return condition_stats

def create_scatter_plot(condition_stats):
    """Create composite index vs bankruptcy rate scatter plot"""
    print("Creating scatter plot...")

    fig, axes = plt.subplots(1, 4, figsize=(28, 6))

    # Plot 1: Composite Index
    ax = axes[0]
    x = [s['composite'] for s in condition_stats]
    y = [s['bankruptcy_rate'] for s in condition_stats]
    ax.scatter(x, y, s=150, alpha=0.6, color='#2E86AB')

    # Linear regression
    if len(x) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        x_line = np.linspace(min(x), max(x), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'r--', linewidth=2, alpha=0.8,
                label=f'R²={r_value**2:.3f}, p={p_value:.3f}')
        ax.legend(fontsize=20)

    ax.set_xlabel('Composite Irrationality Index', fontsize=24)
    ax.set_ylabel('Bankruptcy Rate (%)', fontsize=24)
    ax.set_title('LLaMA: Composite Index', fontsize=26)
    ax.grid(True, alpha=0.3)

    # Plot 2: I_EV
    ax = axes[1]
    x = [s['i_ev'] for s in condition_stats]
    y = [s['bankruptcy_rate'] for s in condition_stats]
    ax.scatter(x, y, s=150, alpha=0.6, color='#A23B72')
    if len(x) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        x_line = np.linspace(min(x), max(x), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'r--', linewidth=2, alpha=0.8,
                label=f'R²={r_value**2:.3f}, p={p_value:.3f}')
        ax.legend(fontsize=20)
    ax.set_xlabel('I_EV (EV Violation)', fontsize=24)
    ax.set_ylabel('Bankruptcy Rate (%)', fontsize=24)
    ax.set_title('LLaMA: I_EV', fontsize=26)
    ax.grid(True, alpha=0.3)

    # Plot 3: I_LC
    ax = axes[2]
    x = [s['i_lc'] for s in condition_stats]
    y = [s['bankruptcy_rate'] for s in condition_stats]
    ax.scatter(x, y, s=150, alpha=0.6, color='#F18F01')
    if len(x) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        x_line = np.linspace(min(x), max(x), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'r--', linewidth=2, alpha=0.8,
                label=f'R²={r_value**2:.3f}, p={p_value:.3f}')
        ax.legend(fontsize=20)
    ax.set_xlabel('I_LC (Loss Chasing)', fontsize=24)
    ax.set_ylabel('Bankruptcy Rate (%)', fontsize=24)
    ax.set_title('LLaMA: I_LC', fontsize=26)
    ax.grid(True, alpha=0.3)

    # Plot 4: I_EB
    ax = axes[3]
    x = [s['i_eb'] for s in condition_stats]
    y = [s['bankruptcy_rate'] for s in condition_stats]
    ax.scatter(x, y, s=150, alpha=0.6, color='#C73E1D')
    if len(x) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        x_line = np.linspace(min(x), max(x), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'r--', linewidth=2, alpha=0.8,
                label=f'R²={r_value**2:.3f}, p={p_value:.3f}')
        ax.legend(fontsize=20)
    ax.set_xlabel('I_EB (Extreme Betting)', fontsize=24)
    ax.set_ylabel('Bankruptcy Rate (%)', fontsize=24)
    ax.set_title('LLaMA: I_EB', fontsize=26)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = '/home/ubuntu/llm_addiction/experiment_0_llama_gemma_restart/analysis/figures/llama/irrationality_metrics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def save_csv(condition_stats):
    """Save results to CSV"""
    import csv
    output_path = '/home/ubuntu/llm_addiction/experiment_0_llama_gemma_restart/analysis/figures/llama/irrationality_metrics.csv'

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['condition', 'n', 'bankruptcy_rate', 'composite', 'i_ev', 'i_lc', 'i_eb'])
        writer.writeheader()
        writer.writerows(condition_stats)

    print(f"Saved: {output_path}")

if __name__ == '__main__':
    experiments = load_llama_data()
    condition_stats = analyze_by_condition(experiments)
    create_scatter_plot(condition_stats)
    save_csv(condition_stats)
    print("✅ LLaMA Irrationality Metrics analysis complete!")
