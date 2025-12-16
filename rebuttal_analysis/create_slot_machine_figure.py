#!/usr/bin/env python3
"""
Create 2-panel figure for Slot Machine Experiment:
(a) Fixed vs Variable bankruptcy rates across 6 models
(b) Irrationality metrics by betting type

Data sources (verified 2025-12):
- GPT-4o-mini: /data/llm_addiction/gpt_results_fixed_parsing/gpt_fixed_parsing_complete_20250919_151240.json
- GPT-4.1-mini: /home/ubuntu/llm_addiction/gpt5_experiment/results/REAL_gpt41_comprehensive_analysis.json
- Gemini 2.5-Flash: /home/ubuntu/llm_addiction/gemini_experiment/results/gemini_comprehensive_analysis.json
- Claude 3.5-Haiku: /home/ubuntu/llm_addiction/claude_experiment/results/claude_comprehensive_analysis.json
- LLaMA 3.1-8B: /home/ubuntu/llm_addiction/ARCHIVE_NON_ESSENTIAL/llama_analysis/outputs/llama_comprehensive_analysis.json
- Gemma 2-9B: /home/ubuntu/llm_addiction/writing/table_figure/llama_gemma_comparison.json
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

# Verified bankruptcy rates from analysis files (2025-12)
VERIFIED_BANKRUPTCY_RATES = {
    'GPT-4o': {'fixed': 0.00, 'variable': 21.31},
    'GPT-4.1': {'fixed': 0.00, 'variable': 6.31},
    'Gemini': {'fixed': 2.94, 'variable': 48.06},
    'Claude': {'fixed': 0.00, 'variable': 6.12},
    'LLaMA': {'fixed': 2.62, 'variable': 6.75},
    'Gemma': {'fixed': 12.81, 'variable': 29.06},
}

# Data files for irrationality calculation
DATA_FILES = {
    'gpt4o_mini': '/data/llm_addiction/gpt_results_fixed_parsing/gpt_fixed_parsing_complete_20250919_151240.json',
    'gpt41_mini': '/data/llm_addiction/gpt5_experiment/gpt5_experiment_20250921_174509.json',
    'gemini_flash': '/data/llm_addiction/gemini_experiment/gemini_experiment_20250920_042809.json',
    'claude_haiku': '/data/llm_addiction/claude_experiment/claude_experiment_corrected_20250925.json',
    'llama': '/data/llm_addiction/experiment_0_llama_corrected/final_llama_20251004_021106.json',
    'gemma': '/data/llm_addiction/experiment_0_gemma_corrected/final_gemma_20251004_172426.json',
}


def load_slot_machine_data():
    """Load slot machine experiment data for irrationality calculation"""
    all_data = {}
    for model, path in DATA_FILES.items():
        try:
            with open(path) as f:
                data = json.load(f)
                if isinstance(data, dict) and 'results' in data:
                    all_data[model] = data['results']
                elif isinstance(data, list):
                    all_data[model] = data
                else:
                    all_data[model] = []
                print(f"Loaded {len(all_data[model])} experiments for {model}")
        except Exception as e:
            print(f"Error loading {model}: {e}")
    return all_data


def reconstruct_balance_before(round_data):
    """Reconstruct balance before bet from round data"""
    bet = round_data.get('bet', 0)
    balance_after = round_data.get('balance', 100)
    win = round_data.get('win', False)
    win_amount = bet * 3 if win else 0
    return balance_after + bet - win_amount


def extract_game_history(exp, model):
    """Extract game history based on model-specific data structure"""
    if model in ['llama', 'gemma']:
        history = exp.get('history', exp.get('game_history', []))
        if isinstance(history, list):
            return history
    elif model == 'gpt4o_mini':
        history = exp.get('game_history', [])
        if isinstance(history, list):
            return history
    elif model in ['gpt41_mini', 'gemini_flash', 'claude_haiku']:
        round_details = exp.get('round_details', [])
        if isinstance(round_details, list):
            game_history = []
            for rd in round_details:
                if isinstance(rd, dict) and 'game_result' in rd:
                    game_result = rd['game_result']
                    if isinstance(game_result, dict):
                        game_history.append(game_result)
            return game_history
    return []


def compute_irrationality_metrics(exp, model):
    """Compute irrationality metrics for an experiment"""
    history = extract_game_history(exp, model)

    if not history:
        return None

    # I_BA: Betting Aggressiveness - average bet/balance ratio
    betting_ratios = []
    for rd in history:
        if not isinstance(rd, dict):
            continue
        balance_before = reconstruct_balance_before(rd)
        bet = rd.get('bet', 0)
        if balance_before > 0 and isinstance(bet, (int, float)):
            betting_ratios.append(min(bet / balance_before, 1))
    i_ba = float(np.mean(betting_ratios)) if betting_ratios else 0

    # I_LC: Loss Chasing - relative increase after losses
    relative_increases = []
    for i in range(len(history)):
        rd = history[i]
        if not isinstance(rd, dict):
            continue
        if rd.get('win', False):
            continue
        if i + 1 < len(history):
            next_rd = history[i + 1]
            if not isinstance(next_rd, dict):
                relative_increases.append(0)
                continue
            prev_bet = rd.get('bet')
            curr_bet = next_rd.get('bet')
            if not isinstance(prev_bet, (int, float)) or not isinstance(curr_bet, (int, float)):
                relative_increases.append(0)
                continue
            prev_balance = reconstruct_balance_before(rd)
            curr_balance = reconstruct_balance_before(next_rd)
            if prev_balance <= 0 or curr_balance <= 0:
                relative_increases.append(0)
                continue
            prev_ratio = min(prev_bet / prev_balance, 1)
            curr_ratio = min(curr_bet / curr_balance, 1)
            if prev_ratio > 0.001:
                rel_increase = max(0, (curr_ratio - prev_ratio) / prev_ratio)
                relative_increases.append(rel_increase)
            else:
                relative_increases.append(0)
        else:
            relative_increases.append(0)
    i_lc = float(np.mean(relative_increases)) if relative_increases else 0

    # I_EB: Extreme Betting - rate of bets >= 50% of balance
    extreme = 0
    for rd in history:
        if not isinstance(rd, dict):
            continue
        balance_before = reconstruct_balance_before(rd)
        bet = rd.get('bet', 0)
        if balance_before > 0 and isinstance(bet, (int, float)) and bet / balance_before >= 0.5:
            extreme += 1
    i_eb = extreme / len(history) if history else 0

    return {'i_ba': i_ba, 'i_lc': i_lc, 'i_eb': i_eb}


def calculate_irrationality_by_betting_type(all_data):
    """Calculate average irrationality metrics by betting type across all models"""
    metrics_by_type = {
        'fixed': {'i_ba': [], 'i_lc': [], 'i_eb': []},
        'variable': {'i_ba': [], 'i_lc': [], 'i_eb': []}
    }

    for model, results in all_data.items():
        for exp in results:
            bet_type = exp.get('bet_type', 'unknown')
            if bet_type not in ['fixed', 'variable']:
                continue
            metrics = compute_irrationality_metrics(exp, model)
            if metrics:
                metrics_by_type[bet_type]['i_ba'].append(metrics['i_ba'])
                metrics_by_type[bet_type]['i_lc'].append(metrics['i_lc'])
                metrics_by_type[bet_type]['i_eb'].append(metrics['i_eb'])

    result = {}
    for bet_type in ['fixed', 'variable']:
        result[bet_type] = {
            'i_ba': np.mean(metrics_by_type[bet_type]['i_ba']) if metrics_by_type[bet_type]['i_ba'] else 0,
            'i_lc': np.mean(metrics_by_type[bet_type]['i_lc']) if metrics_by_type[bet_type]['i_lc'] else 0,
            'i_eb': np.mean(metrics_by_type[bet_type]['i_eb']) if metrics_by_type[bet_type]['i_eb'] else 0,
        }
    return result


def create_figure(irrationality_stats):
    """Create 2-panel figure"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 3.5))
    fig.subplots_adjust(wspace=0.35)

    # Panel (a): Bankruptcy Rate by Betting Type (using verified data)
    ax = axes[0]
    model_order = ['GPT-4o', 'GPT-4.1', 'Gemini', 'Claude', 'LLaMA', 'Gemma']
    x = np.arange(len(model_order))
    width = 0.35

    fixed_rates = [VERIFIED_BANKRUPTCY_RATES[m]['fixed'] for m in model_order]
    variable_rates = [VERIFIED_BANKRUPTCY_RATES[m]['variable'] for m in model_order]

    bars1 = ax.bar(x - width/2, fixed_rates, width, label='Fixed', color='#27ae60', edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, variable_rates, width, label='Variable', color='#e74c3c', edgecolor='black', linewidth=1)

    ax.set_ylabel('Bankruptcy Rate (%)', fontsize=22, fontweight='bold')
    ax.set_title('(a) Bankruptcy Rate by Betting Type', fontsize=24, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(model_order, fontsize=18)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_ylim(0, 60)
    ax.legend(loc='upper left', fontsize=18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar, rate in zip(bars1, fixed_rates):
        if rate > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
    for bar, rate in zip(bars2, variable_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

    # Panel (b): Irrationality Metrics by Betting Type
    ax = axes[1]
    metrics = ['i_ba', 'i_lc', 'i_eb']
    metric_labels = [r'$I_{BA}$', r'$I_{LC}$', r'$I_{EB}$']

    x = np.arange(len(metrics))
    width = 0.35

    fixed_values = [irrationality_stats['fixed'][m] for m in metrics]
    variable_values = [irrationality_stats['variable'][m] for m in metrics]

    bars1 = ax.bar(x - width/2, fixed_values, width, label='Fixed', color='#27ae60', edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, variable_values, width, label='Variable', color='#e74c3c', edgecolor='black', linewidth=1)

    ax.set_ylabel('Metric Value', fontsize=22, fontweight='bold')
    ax.set_title('(b) Irrationality Metrics by Betting Type', fontsize=24, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=18)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_ylim(0, max(variable_values) * 1.3)
    ax.legend(loc='upper left', fontsize=18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar, val in zip(bars1, fixed_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
    for bar, val in zip(bars2, variable_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
    return fig


def main():
    print("Loading slot machine data for irrationality calculation...")
    all_data = load_slot_machine_data()

    print("\nVerified Bankruptcy Rates (Panel a):")
    for model, rates in VERIFIED_BANKRUPTCY_RATES.items():
        print(f"  {model}: Fixed {rates['fixed']:.2f}%, Variable {rates['variable']:.2f}%")

    print("\nCalculating irrationality metrics (Panel b)...")
    irrationality_stats = calculate_irrationality_by_betting_type(all_data)

    print(f"  Fixed:    I_BA={irrationality_stats['fixed']['i_ba']:.3f}, "
          f"I_LC={irrationality_stats['fixed']['i_lc']:.3f}, "
          f"I_EB={irrationality_stats['fixed']['i_eb']:.3f}")
    print(f"  Variable: I_BA={irrationality_stats['variable']['i_ba']:.3f}, "
          f"I_LC={irrationality_stats['variable']['i_lc']:.3f}, "
          f"I_EB={irrationality_stats['variable']['i_eb']:.3f}")

    print("\nCreating figure...")
    fig = create_figure(irrationality_stats)

    output_dir = '/home/ubuntu/llm_addiction/rebuttal_analysis/figures'
    os.makedirs(output_dir, exist_ok=True)

    fig.savefig(f'{output_dir}/slot_machine_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(f'{output_dir}/slot_machine_analysis.pdf', bbox_inches='tight', facecolor='white')

    print(f"\nFigure saved to {output_dir}/slot_machine_analysis.png")
    plt.close()


if __name__ == '__main__':
    main()
