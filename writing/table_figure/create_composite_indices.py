#!/usr/bin/env python3
"""Generate 4-model composite irrationality indices comparison figure."""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

plt.style.use('seaborn-v0_8-whitegrid')

MODEL_PATTERNS = {
    # 'llama': 'SPECIAL_LLAMA_LOADER',
    'gpt4mini': '/data/llm_addiction/gpt_results_fixed_parsing/gpt_fixed_parsing_complete_20250919_151240.json',
    'gpt41mini': '/data/llm_addiction/gpt5_experiment/gpt5_experiment_20250921_174509.json',
    'gemini25flash': '/data/llm_addiction/gemini_experiment/gemini_experiment_20250920_042809.json',
    'claude': '/data/llm_addiction/claude_experiment/claude_experiment_corrected_20250925.json',
}

MODEL_DISPLAY_NAMES = {
    'gpt4mini': 'GPT-4o-mini',
    'gpt41mini': 'GPT-4.1-mini',
    'gemini25flash': 'Gemini-2.5-Flash',
    'claude': 'Claude-3.5-Haiku',
}

OUTPUT_DIR = Path('/home/ubuntu/llm_addiction/writing/figures/losschasing')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_llama_complete_data():
    """Load complete LLaMA dataset using llama_analysis data loader"""
    sys.path.append('/home/ubuntu/llm_addiction/llama_analysis')

    try:
        from data_loader import load_llama_results
        print("Loading complete LLaMA dataset (6,400 experiments)...")
        llama_results = load_llama_results()
        print(f"Successfully loaded {len(llama_results)} LLaMA experiments")
        return llama_results
    except Exception as e:
        print(f"Error loading complete LLaMA data: {e}")
        fallback_path = '/data/llm_addiction/results/exp1_missing_complete_20250820_090040.json'
        print(f"Using fallback: {fallback_path}")
        with open(fallback_path, 'r') as f:
            data = json.load(f)
        return data.get('results', [])


def load_experiments() -> pd.DataFrame:
    records = []

    for model, pattern in MODEL_PATTERNS.items():
        if model == 'llama' and pattern == 'SPECIAL_LLAMA_LOADER':
            results = load_llama_complete_data()
            for exp in results:
                exp_record = exp.copy()
                exp_record['model'] = model
                exp_record['source_file'] = 'llama_complete_dataset'
                records.append(exp_record)
        else:
            try:
                with open(pattern, 'r') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error loading {model}: {e}")
                continue

            results = data.get('results') or []
            for exp in results:
                exp_record = exp.copy()
                exp_record['model'] = model
                exp_record['source_file'] = pattern
                records.append(exp_record)

    if not records:
        raise ValueError('No experiment records found. Check file paths.')
    return pd.DataFrame(records)


def reconstruct_balance_before(round_data):
    bet = round_data.get('bet', 0)
    balance_after = round_data.get('balance', 100)
    win = round_data.get('win', False)
    win_amount = bet * 3 if win else 0
    return balance_after + bet - win_amount


def extract_game_history(exp, model):
    """Extract game history based on model-specific data structure"""
    if model in ['llama', 'gpt4mini']:
        history = exp.get('game_history')
        if isinstance(history, list):
            return history
    elif model in ['gpt41mini', 'gemini25flash', 'claude']:
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


def compute_loss_chasing_rate(history) -> float:
    """Compute rate of increased bet-to-balance ratios following losses."""
    chase_events = 0
    opportunities = 0

    for i in range(1, len(history)):
        prev = history[i - 1]
        curr = history[i]
        if not isinstance(prev, dict) or not isinstance(curr, dict):
            continue
        if prev.get('win', False):
            continue

        prev_bet = prev.get('bet')
        curr_bet = curr.get('bet')
        if not isinstance(prev_bet, (int, float)) or not isinstance(curr_bet, (int, float)):
            continue

        prev_balance = reconstruct_balance_before(prev)
        curr_balance = reconstruct_balance_before(curr)
        if prev_balance <= 0 or curr_balance <= 0:
            continue

        prev_ratio = min(prev_bet / prev_balance, 1)
        curr_ratio = min(curr_bet / curr_balance, 1)

        opportunities += 1
        if curr_ratio > prev_ratio:
            chase_events += 1

    return chase_events / opportunities if opportunities else 0.0


def compute_behavior_metrics(exp, model) -> dict:
    """Compute behavioral metrics with model-aware history extraction"""
    history = extract_game_history(exp, model)

    if not isinstance(history, list) or not history:
        return {'i_ev': 0, 'i_lc': 0, 'i_eb': 0}

    betting_ratios = []
    for rd in history:
        if not isinstance(rd, dict):
            continue
        balance_before = reconstruct_balance_before(rd)
        bet = rd.get('bet', 0)
        if balance_before > 0 and isinstance(bet, (int, float)):
            betting_ratios.append(min(bet / balance_before, 1))
    i_ev = float(np.mean(betting_ratios)) if betting_ratios else 0

    i_lc = compute_loss_chasing_rate(history)

    extreme = 0
    for rd in history:
        if not isinstance(rd, dict):
            continue
        balance_before = reconstruct_balance_before(rd)
        bet = rd.get('bet', 0)
        if balance_before > 0 and isinstance(bet, (int, float)) and bet / balance_before >= 0.5:
            extreme += 1
    i_eb = extreme / len(history) if history else 0

    return {'i_ev': i_ev, 'i_lc': i_lc, 'i_eb': i_eb}


def generate_composite_indices(df: pd.DataFrame):
    rows = []
    for _, exp in df.iterrows():
        model = exp['model']
        metrics = compute_behavior_metrics(exp, model)
        rows.append({
            'model': model,
            'condition': exp.get('prompt_combo', 'BASE'),
            'bet_type': exp.get('bet_type', 'unknown'),
            **metrics,
            'bankruptcy': 1 if exp.get('is_bankrupt') else 0
        })

    metrics_df = pd.DataFrame(rows)
    if metrics_df.empty:
        print('No composite metrics available - skipping figure')
        return

    metrics_df['composite'] = 0.4 * metrics_df['i_ev'] + 0.3 * metrics_df['i_lc'] + 0.3 * metrics_df['i_eb']

    models = metrics_df['model'].unique()
    n_models = len(models)

    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 3.5), sharey=True)
    if n_models == 1:
        axes = [axes]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, (ax, model) in enumerate(zip(axes, models)):
        subset = metrics_df[metrics_df['model'] == model]
        agg = subset.groupby('condition').agg({
            'composite': 'mean',
            'bankruptcy': 'mean'
        }).reset_index()

        color = colors[i % len(colors)]
        ax.scatter(agg['composite'], agg['bankruptcy'] * 100, alpha=0.7, s=80, color=color)

        if len(agg) > 1:
            try:
                r, p = pearsonr(agg['composite'], agg['bankruptcy'])

                if not np.isnan(r):
                    z = np.polyfit(agg['composite'], agg['bankruptcy'] * 100, 1)
                    p_line = np.poly1d(z)
                    x_line = np.linspace(agg['composite'].min(), agg['composite'].max(), 100)
                    ax.plot(x_line, p_line(x_line), color=color, linestyle='-', linewidth=2.5, alpha=0.8)

                    ax.text(0.05, 0.95, f'r = {r:.3f}',
                           transform=ax.transAxes, verticalalignment='top', fontsize=24,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            except (np.linalg.LinAlgError, ValueError):
                pass

        display_name = MODEL_DISPLAY_NAMES.get(model, model)
        ax.set_title(f'{display_name}', fontsize=26, fontweight='bold')
        if i == 0:
            ax.set_ylabel('Bankruptcy Rate (%)', fontsize=24, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=22)
        ax.grid(True, alpha=0.3)

        # Set consistent x-axis start, flexible end
        current_xlim = ax.get_xlim()
        ax.set_xlim(-0.02, current_xlim[1])

    plt.suptitle('Irrationality-Bankruptcy Correlation Across Models', fontsize=28, fontweight='bold', y=0.98)
    plt.tight_layout(w_pad=3.0)

    fig.text(0.5, -0.02, 'Composite Irrationality Index', ha='center', fontsize=24, fontweight='bold')

    png_path = OUTPUT_DIR / '4model_composite_indices.png'
    pdf_path = OUTPUT_DIR / '4model_composite_indices.pdf'

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"Figure saved to:")
    print(f"  PNG: {png_path}")
    print(f"  PDF: {pdf_path}")


def main():
    print("Loading experiment data...")
    df = load_experiments()
    print(f"Loaded {len(df)} experiments from {df['model'].nunique()} models")

    print("\nGenerating composite indices figure...")
    generate_composite_indices(df)
    print("\nDone!")


if __name__ == '__main__':
    main()
