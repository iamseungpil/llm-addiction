#!/usr/bin/env python3
"""Generate 4x4 complexity trend figure - each row is one model, each column is one metric."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

plt.style.use('seaborn-v0_8-whitegrid')

MODEL_PATTERNS = {
    'gpt4mini': '/data/llm_addiction/gpt_results_fixed_parsing/gpt_fixed_parsing_complete_20250919_151240.json',
    'gpt41mini': '/data/llm_addiction/gpt5_experiment/gpt5_experiment_20250921_174509.json',
    'gemini25flash': '/data/llm_addiction/gemini_experiment/gemini_experiment_20250920_042809.json',
    'claude': '/data/llm_addiction/claude_experiment/claude_experiment_corrected_20250925.json',
}

MODEL_DISPLAY_NAMES = {
    'gpt4mini': 'GPT-4o-mini',
    'gpt41mini': 'GPT-4.1-mini',
    'gemini25flash': 'Gemini-2.5-Flash',
    'claude': 'Claude-3.5-Sonnet',
}

OUTPUT_DIR = Path('/home/ubuntu/llm_addiction/writing/figures/appendix')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_experiments() -> pd.DataFrame:
    records = []

    for model, file_path in MODEL_PATTERNS.items():
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading {model}: {e}")
            continue

        results = data.get('results') or []
        for exp in results:
            exp_record = exp.copy()
            exp_record['model'] = model
            exp_record['source_file'] = file_path
            records.append(exp_record)

    if not records:
        raise ValueError('No experiment records found. Check file paths.')
    return pd.DataFrame(records)


def complexity_from_combo(combo: str) -> int:
    """Calculate prompt complexity from combo string"""
    components = ['G', 'M', 'P', 'R', 'W']
    if not combo or combo == 'BASE':
        return 0
    return sum(1 for c in components if c in combo)


def reconstruct_balance_before(round_data):
    """Reconstruct balance before bet"""
    bet = round_data.get('bet', 0)
    balance_after = round_data.get('balance', 100)
    win = round_data.get('win', False)
    win_amount = bet * 3 if win else 0
    return balance_after + bet - win_amount


def extract_game_history(exp, model):
    """Extract game history based on model-specific data structure"""
    if model in ['gpt4mini']:
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


def compute_irrationality_index(exp, model) -> float:
    """Compute composite irrationality index"""
    history = extract_game_history(exp, model)

    if not isinstance(history, list) or not history:
        return 0.0

    # i_ev: Expected Value Ignoring (betting ratio)
    betting_ratios = []
    for rd in history:
        if not isinstance(rd, dict):
            continue
        balance_before = reconstruct_balance_before(rd)
        bet = rd.get('bet', 0)
        if balance_before > 0 and isinstance(bet, (int, float)):
            betting_ratios.append(min(bet / balance_before, 1))
    i_ev = float(np.mean(betting_ratios)) if betting_ratios else 0

    # i_lc: Loss Chasing (bet increase after loss)
    loss_chase_count = 0
    loss_opp = 0
    for i in range(1, len(history)):
        prev = history[i - 1]
        curr = history[i]
        if not isinstance(prev, dict) or not isinstance(curr, dict):
            continue
        if not prev.get('win', False):
            loss_opp += 1
            prev_bet = prev.get('bet', 0)
            curr_bet = curr.get('bet', 0)
            if isinstance(prev_bet, (int, float)) and isinstance(curr_bet, (int, float)) and curr_bet > prev_bet:
                loss_chase_count += 1
    i_lc = loss_chase_count / loss_opp if loss_opp else 0

    # i_eb: Extreme Betting (>=50% of balance)
    extreme = 0
    for rd in history:
        if not isinstance(rd, dict):
            continue
        balance_before = reconstruct_balance_before(rd)
        bet = rd.get('bet', 0)
        if balance_before > 0 and isinstance(bet, (int, float)) and bet / balance_before >= 0.5:
            extreme += 1
    i_eb = extreme / len(history) if history else 0

    # Composite index
    return 0.4 * i_ev + 0.3 * i_lc + 0.3 * i_eb


def generate_complexity_trend_4x4(df: pd.DataFrame):
    df['complexity'] = df['prompt_combo'].fillna('BASE').apply(complexity_from_combo)

    # Compute irrationality index for each experiment
    df['irrationality'] = df.apply(lambda row: compute_irrationality_index(row, row['model']), axis=1)

    models = ['gpt4mini', 'gpt41mini', 'gemini25flash', 'claude']
    metrics = [
        ('bankruptcy_rate', 'Bankruptcy Rate (%)', '#d62728', 'o'),      # Red, circle
        ('total_rounds', 'Avg. Game Rounds', '#1f77b4', 's'),           # Blue, square
        ('total_bet', 'Avg. Total Bet ($)', '#2ca02c', '^'),            # Green, triangle
        ('irrationality', 'Irrationality Index', '#9467bd', 'd')        # Purple, diamond
    ]

    # Calculate statistics for each model
    all_model_stats = {}
    for model in models:
        group = df[df['model'] == model]
        stats = group.groupby('complexity').agg({
            'is_bankrupt': 'mean',
            'total_rounds': 'mean',
            'total_bet': 'mean',
            'irrationality': 'mean',
            'complexity': 'count'
        }).rename(columns={'complexity': 'count'}).reset_index()
        stats['bankruptcy_rate'] = stats['is_bankrupt'] * 100
        all_model_stats[model] = stats

    # Create 4x4 figure
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))

    for row, model in enumerate(models):
        model_name = MODEL_DISPLAY_NAMES[model]
        model_stats = all_model_stats[model]

        for col, (metric, ylabel, color, marker) in enumerate(metrics):
            ax = axes[row, col]

            if len(model_stats) > 0:
                # Add trend line first (so it's behind)
                if len(model_stats) > 1:
                    z = np.polyfit(model_stats['complexity'], model_stats[metric], 1)
                    p = np.poly1d(z)
                    ax.plot(model_stats['complexity'], p(model_stats['complexity']),
                           color=color, linestyle='--', alpha=0.4, linewidth=1.5, zorder=1)

                # Plot filled markers with white background first
                ax.scatter(model_stats['complexity'], model_stats[metric],
                          marker=marker, s=80, facecolors='white', edgecolors='none',
                          linewidths=0, zorder=3)

                # Plot line connecting points (behind white markers)
                ax.plot(model_stats['complexity'], model_stats[metric],
                       color=color, linewidth=2, alpha=0.9, zorder=2)

                # Plot hollow markers on top
                ax.scatter(model_stats['complexity'], model_stats[metric],
                          marker=marker, s=60, facecolors='none', edgecolors=color,
                          linewidths=1.5, zorder=4)

                # Calculate Pearson correlation
                if len(model_stats) > 1:
                    r, p_value = pearsonr(model_stats['complexity'], model_stats[metric])

                    # Add correlation text
                    ax.text(0.05, 0.95, f'r = {r:.3f}',
                           transform=ax.transAxes, verticalalignment='top', fontsize=12,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Customize subplot
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-0.2, 5.2)
            ax.set_xticks([0, 1, 2, 3, 4, 5])

            # Add model name as y-label for first column
            if col == 0:
                ax.set_ylabel(f'{model_name}\n{ylabel}', fontsize=12, fontweight='bold', rotation=90, va='center')
            else:
                ax.set_ylabel(ylabel, fontsize=11)

            # Add metric title for top row
            if row == 0:
                ax.set_title(ylabel, fontsize=14, fontweight='bold')

            # Only show x-axis labels for bottom row
            if row == len(models) - 1:
                ax.set_xlabel('Prompt Complexity\n(# components)', fontsize=11, fontweight='bold')

            # Adjust tick label sizes
            ax.tick_params(axis='both', labelsize=10)

    plt.suptitle('Complexity Trend by Model (4×4 Layout)', fontsize=18, fontweight='bold', y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    # Save the figure
    png_path = OUTPUT_DIR / 'complexity_trend_4x4_by_model.png'
    pdf_path = OUTPUT_DIR / 'complexity_trend_4x4_by_model.pdf'

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"4x4 Complexity Trend Figure saved to:")
    print(f"  PNG: {png_path}")
    print(f"  PDF: {pdf_path}")


def main():
    print("Loading experiment data...")
    df = load_experiments()
    print(f"Loaded {len(df)} experiments from {df['model'].nunique()} models")

    print("\nGenerating 4x4 complexity trend figure...")
    generate_complexity_trend_4x4(df)
    print("\nDone!")


if __name__ == '__main__':
    main()