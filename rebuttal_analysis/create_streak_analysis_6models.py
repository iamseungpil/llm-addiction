#!/usr/bin/env python3
"""
Generate Streak Analysis figure with all 6 LLMs:
- 4 API models: GPT-4o-mini, GPT-4.1-mini, Gemini-2.5-Flash, Claude-3.5-Haiku
- 2 Open-weight models: LLaMA-3.1-8B, Gemma-2-9B
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
import glob

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14

# Data paths
API_MODEL_PATHS = {
    'gpt4o_mini': '/data/llm_addiction/gpt_results_fixed_parsing/gpt_fixed_parsing_complete_20250919_151240.json',
    'gpt41_mini': '/data/llm_addiction/gpt5_experiment/gpt5_experiment_20250921_174509.json',
    'gemini_flash': '/data/llm_addiction/gemini_experiment/gemini_experiment_20250920_042809.json',
    'claude_haiku': '/data/llm_addiction/claude_experiment/claude_experiment_corrected_20250925.json',
}

OPEN_MODEL_DIRS = {
    'llama': '/data/llm_addiction/experiment_0_llama_corrected/',
    'gemma': '/data/llm_addiction/experiment_0_gemma_corrected/',
}

MODEL_DISPLAY = {
    'gpt4o_mini': 'GPT-4o-mini',
    'gpt41_mini': 'GPT-4.1-mini',
    'gemini_flash': 'Gemini-2.5-Flash',
    'claude_haiku': 'Claude-3.5-Haiku',
    'llama': 'LLaMA-3.1-8B',
    'gemma': 'Gemma-2-9B',
}

def load_api_model_data(model_name, path):
    """Load data from API models"""
    with open(path) as f:
        data = json.load(f)

    experiments = data.get('experiments', data.get('results', []))
    return experiments

def load_open_model_data(model_name, directory):
    """Load data from open-weight models (LLaMA/Gemma)"""
    all_experiments = []
    json_files = sorted(glob.glob(f"{directory}/*.json"))

    # Use the final checkpoint
    if json_files:
        final_file = max(json_files, key=lambda x: int(x.split('_')[-2]) if '_' in x else 0)
        with open(final_file) as f:
            experiments = json.load(f)
        all_experiments.extend(experiments)

    return all_experiments

def extract_history_api(exp, model):
    """Extract game history from API model experiment"""
    if model == 'gpt4o_mini':
        return exp.get('game_history', [])
    else:  # gpt41_mini, gemini_flash, claude_haiku
        round_details = exp.get('round_details', [])
        history = []
        for rd in round_details:
            if isinstance(rd, dict) and 'game_result' in rd:
                game_result = rd['game_result']
                if isinstance(game_result, dict):
                    history.append(game_result)
        return history

def extract_history_open(exp):
    """Extract game history from open-weight model experiment"""
    return exp.get('history', [])

def calculate_streak_metrics_for_experiment(history):
    """Calculate streak-based metrics for a single experiment"""
    if not history or len(history) < 2:
        return None

    metrics = {
        'win_streak': defaultdict(lambda: {'bet_increase': 0, 'total': 0, 'continue': 0}),
        'loss_streak': defaultdict(lambda: {'bet_increase': 0, 'total': 0, 'continue': 0}),
    }

    # Track streaks
    current_streak_type = None
    current_streak_length = 0

    for i, round_data in enumerate(history):
        is_win = round_data.get('win', False)
        current_bet = round_data.get('bet', 10)

        if i == 0:
            current_streak_type = 'win' if is_win else 'loss'
            current_streak_length = 1
            continue

        prev_bet = history[i-1].get('bet', 10)

        # Check if streak continues
        if (is_win and current_streak_type == 'win') or (not is_win and current_streak_type == 'loss'):
            current_streak_length += 1
        else:
            # Streak broke - record metrics
            streak_key = f'{current_streak_type}_streak'
            metrics[streak_key][current_streak_length]['total'] += 1
            metrics[streak_key][current_streak_length]['continue'] += 1  # They continued

            if current_bet > prev_bet:
                metrics[streak_key][current_streak_length]['bet_increase'] += 1

            # Start new streak
            current_streak_type = 'win' if is_win else 'loss'
            current_streak_length = 1

    return metrics

def aggregate_metrics(all_metrics):
    """Aggregate metrics across all experiments"""
    result = {
        'win': defaultdict(lambda: {'bet_increase': 0, 'total': 0, 'continue': 0}),
        'loss': defaultdict(lambda: {'bet_increase': 0, 'total': 0, 'continue': 0}),
    }

    for exp_metrics in all_metrics:
        if exp_metrics is None:
            continue
        for streak_type in ['win', 'loss']:
            streak_key = f'{streak_type}_streak'
            for length, counts in exp_metrics[streak_key].items():
                result[streak_type][length]['bet_increase'] += counts['bet_increase']
                result[streak_type][length]['total'] += counts['total']
                result[streak_type][length]['continue'] += counts['continue']

    return result

def process_all_models():
    """Process all 6 models and return aggregated metrics"""
    all_model_metrics = {}

    # Process API models
    for model_name, path in API_MODEL_PATHS.items():
        print(f"Processing {model_name}...")
        try:
            experiments = load_api_model_data(model_name, path)
            metrics_list = []

            for exp in experiments:
                # Only variable betting
                bet_type = exp.get('bet_type', exp.get('betting_type', 'unknown'))
                if 'variable' not in bet_type.lower():
                    continue

                history = extract_history_api(exp, model_name)
                metrics = calculate_streak_metrics_for_experiment(history)
                if metrics:
                    metrics_list.append(metrics)

            all_model_metrics[model_name] = aggregate_metrics(metrics_list)
            print(f"  Processed {len(metrics_list)} variable betting experiments")
        except Exception as e:
            print(f"  Error: {e}")

    # Process open-weight models
    for model_name, directory in OPEN_MODEL_DIRS.items():
        print(f"Processing {model_name}...")
        try:
            experiments = load_open_model_data(model_name, directory)
            metrics_list = []

            for exp in experiments:
                bet_type = exp.get('bet_type', 'unknown')
                if bet_type != 'variable':
                    continue

                history = extract_history_open(exp)
                metrics = calculate_streak_metrics_for_experiment(history)
                if metrics:
                    metrics_list.append(metrics)

            all_model_metrics[model_name] = aggregate_metrics(metrics_list)
            print(f"  Processed {len(metrics_list)} variable betting experiments")
        except Exception as e:
            print(f"  Error: {e}")

    return all_model_metrics

def calculate_averaged_metrics(all_model_metrics):
    """Calculate averaged metrics across all models"""
    averaged = {
        'win': defaultdict(lambda: {'rate': [], 'continue_rate': []}),
        'loss': defaultdict(lambda: {'rate': [], 'continue_rate': []}),
    }

    for model_name, metrics in all_model_metrics.items():
        for streak_type in ['win', 'loss']:
            for length in range(1, 6):
                data = metrics[streak_type][length]
                if data['total'] > 10:  # Minimum sample size
                    bet_rate = data['bet_increase'] / data['total']
                    cont_rate = data['continue'] / data['total']
                    averaged[streak_type][length]['rate'].append(bet_rate)
                    averaged[streak_type][length]['continue_rate'].append(cont_rate)

    # Calculate mean and std
    result = {
        'win': {'rate': [], 'rate_std': [], 'continue': [], 'continue_std': []},
        'loss': {'rate': [], 'rate_std': [], 'continue': [], 'continue_std': []},
    }

    for streak_type in ['win', 'loss']:
        for length in range(1, 6):
            rates = averaged[streak_type][length]['rate']
            cont_rates = averaged[streak_type][length]['continue_rate']

            if rates:
                result[streak_type]['rate'].append(np.mean(rates))
                result[streak_type]['rate_std'].append(np.std(rates))
            else:
                result[streak_type]['rate'].append(0)
                result[streak_type]['rate_std'].append(0)

            if cont_rates:
                result[streak_type]['continue'].append(np.mean(cont_rates))
                result[streak_type]['continue_std'].append(np.std(cont_rates))
            else:
                result[streak_type]['continue'].append(0)
                result[streak_type]['continue_std'].append(0)

    return result

def create_figure(averaged_metrics):
    """Create the streak analysis figure"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plt.subplots_adjust(wspace=0.4)

    streak_lengths = np.arange(1, 6)
    x = np.arange(len(streak_lengths))
    width = 0.35

    colors = {'win': '#2ecc71', 'loss': '#e74c3c'}

    # Panel (a): Bet Increase Rate
    ax = axes[0]
    win_rates = averaged_metrics['win']['rate']
    win_std = averaged_metrics['win']['rate_std']
    loss_rates = averaged_metrics['loss']['rate']
    loss_std = averaged_metrics['loss']['rate_std']

    bars1 = ax.bar(x - width/2, win_rates, width, yerr=win_std, label='Win Streak',
                   color=colors['win'], capsize=3, error_kw={'linewidth': 1.5})
    bars2 = ax.bar(x + width/2, loss_rates, width, yerr=loss_std, label='Loss Streak',
                   color=colors['loss'], capsize=3, error_kw={'linewidth': 1.5})

    ax.set_xlabel('Consecutive Streak Length')
    ax.set_ylabel('Bet Increase Rate')
    ax.set_title('(a) Bet Increase Rate')
    ax.set_xticks(x)
    ax.set_xticklabels(streak_lengths)
    ax.set_ylim(0, 0.5)
    ax.legend(loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel (b): Continuation Rate
    ax = axes[1]
    win_cont = averaged_metrics['win']['continue']
    win_cont_std = averaged_metrics['win']['continue_std']
    loss_cont = averaged_metrics['loss']['continue']
    loss_cont_std = averaged_metrics['loss']['continue_std']

    bars1 = ax.bar(x - width/2, win_cont, width, yerr=win_cont_std, label='Win Streak',
                   color=colors['win'], capsize=3, error_kw={'linewidth': 1.5})
    bars2 = ax.bar(x + width/2, loss_cont, width, yerr=loss_cont_std, label='Loss Streak',
                   color=colors['loss'], capsize=3, error_kw={'linewidth': 1.5})

    ax.set_xlabel('Consecutive Streak Length')
    ax.set_ylabel('Continuation Rate')
    ax.set_title('(b) Continuation Rate')
    ax.set_xticks(x)
    ax.set_xticklabels(streak_lengths)
    ax.set_ylim(0, 1.0)
    ax.legend(loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    return fig

def main():
    print("Processing all 6 models for streak analysis...")
    all_model_metrics = process_all_models()

    print("\nCalculating averaged metrics...")
    averaged_metrics = calculate_averaged_metrics(all_model_metrics)

    print("\nBet Increase Rate by streak length:")
    print("  Win:", [f"{r:.3f}" for r in averaged_metrics['win']['rate']])
    print("  Loss:", [f"{r:.3f}" for r in averaged_metrics['loss']['rate']])

    print("\nContinuation Rate by streak length:")
    print("  Win:", [f"{r:.3f}" for r in averaged_metrics['win']['continue']])
    print("  Loss:", [f"{r:.3f}" for r in averaged_metrics['loss']['continue']])

    print("\nCreating figure...")
    fig = create_figure(averaged_metrics)

    output_dir = '/home/ubuntu/llm_addiction/rebuttal_analysis/figures'
    fig.savefig(f'{output_dir}/streak_analysis_6models.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(f'{output_dir}/streak_analysis_6models.pdf', bbox_inches='tight', facecolor='white')
    print(f"\nFigure saved to {output_dir}/streak_analysis_6models.png")
    plt.close()

if __name__ == '__main__':
    main()
