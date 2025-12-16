#!/usr/bin/env python3
"""Generate individual component effects charts for each model separately."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')

MODEL_PATTERNS = {
    'GPT-4o-mini': '/data/llm_addiction/gpt_results_fixed_parsing/gpt_fixed_parsing_complete_20250919_151240.json',
    'GPT-4.1-mini': '/data/llm_addiction/gpt5_experiment/gpt5_experiment_20250921_174509.json',
    'Gemini-2.5-Flash': '/data/llm_addiction/gemini_experiment/gemini_experiment_20250920_042809.json',
    'Claude-3.5-Haiku': '/data/llm_addiction/claude_experiment/claude_experiment_corrected_20250925.json',
}

MODEL_SHORT_NAMES = {
    'GPT-4o-mini': 'gpt4mini',
    'GPT-4.1-mini': 'gpt41mini',
    'Gemini-2.5-Flash': 'gemini25flash',
    'Claude-3.5-Haiku': 'claude'
}

OUTPUT_DIR = Path('/home/ubuntu/llm_addiction/writing/figures')
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
            exp_record['model_short'] = MODEL_SHORT_NAMES[model]
            exp_record['source_file'] = file_path
            records.append(exp_record)

    if not records:
        raise ValueError('No experiment records found. Check file paths.')
    return pd.DataFrame(records)


def reconstruct_balance_before(round_data):
    """Reconstruct balance before bet"""
    bet = round_data.get('bet', 0)
    balance_after = round_data.get('balance', 100)
    win = round_data.get('win', False)
    win_amount = bet * 3 if win else 0
    return balance_after + bet - win_amount


def extract_game_history(exp, model_short):
    """Extract game history based on model-specific data structure"""
    if model_short in ['gpt4mini']:
        history = exp.get('game_history')
        if isinstance(history, list):
            return history
    elif model_short in ['gpt41mini', 'gemini25flash', 'claude']:
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


def compute_irrationality_index(exp, model_short) -> float:
    """Compute composite irrationality index"""
    history = extract_game_history(exp, model_short)

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


def generate_comprehensive_model_chart(df: pd.DataFrame):
    """Generate one large chart with all models and metrics (3x4 grid, no Irrationality Index)"""
    components = ['G', 'M', 'P', 'H', 'W']
    metrics = ['bankruptcy_effect', 'bet_effect', 'rounds_effect']
    metric_titles = ['Bankruptcy Effect (%)', 'Total Bet Effect ($)', 'Rounds Effect']
    model_order = ['GPT-4o-mini', 'GPT-4.1-mini', 'Gemini-2.5-Flash', 'Claude-3.5-Haiku']

    # Compute irrationality index for all experiments
    df['irrationality'] = df.apply(lambda row: compute_irrationality_index(row, row['model_short']), axis=1)

    # Unified color scheme for all models (Green/Red)
    colors = {'fixed': '#2ca02c', 'variable': '#d62728'}

    # Create 3x4 grid (3 metrics x 4 models) - no Irrationality Index
    fig, axes = plt.subplots(3, 4, figsize=(20, 14), sharex=True)

    for model_idx, model in enumerate(model_order):
        model_group = df[df['model'] == model]
        if model_group.empty:
            continue
        print(f"Processing {model}...")

        # Calculate component effects for this model
        records = []
        for bet_type in ['fixed', 'variable']:
            bet_group = model_group[model_group['bet_type'] == bet_type]

            for component in components:
                # Map H back to R for data matching
                search_component = 'R' if component == 'H' else component
                with_comp = bet_group[bet_group['prompt_combo'].fillna('').str.contains(search_component)]
                without_comp = bet_group[~bet_group['prompt_combo'].fillna('').str.contains(search_component)]

                if with_comp.empty or without_comp.empty:
                    continue

                effect_row = {
                    'bet_type': bet_type,
                    'component': component,
                    'bankruptcy_effect': (with_comp['is_bankrupt'].fillna(False).mean() - without_comp['is_bankrupt'].fillna(False).mean()) * 100,
                    'bet_effect': with_comp['total_bet'].fillna(0).mean() - without_comp['total_bet'].fillna(0).mean(),
                    'rounds_effect': with_comp['total_rounds'].fillna(0).mean() - without_comp['total_rounds'].fillna(0).mean(),
                    'irrationality_effect': with_comp['irrationality'].fillna(0).mean() - without_comp['irrationality'].fillna(0).mean()
                }
                records.append(effect_row)

        if not records:
            print(f"No component effects found for {model}")
            continue

        effect_df = pd.DataFrame(records)

        # Plot in the 3x4 grid (metrics as rows, models as columns)

        for metric_idx, metric in enumerate(metrics):
            ax = axes[metric_idx, model_idx]  # Transposed indices

            # Create grouped bar chart
            x = np.arange(len(components))
            width = 0.3

            for i, bet_type in enumerate(['fixed', 'variable']):
                bet_data = effect_df[effect_df['bet_type'] == bet_type]
                values = []
                for comp in components:
                    comp_data = bet_data[bet_data['component'] == comp]
                    if len(comp_data) > 0:
                        values.append(comp_data[metric].iloc[0])
                    else:
                        values.append(0)

                label = 'Fixed' if bet_type == 'fixed' else 'Variable'
                ax.bar(x + i * width - width/2, values, width, label=label,
                      color=colors[bet_type], alpha=0.8)

            # Y-axis labels only on leftmost column
            if model_idx == 0:
                ax.set_ylabel(metric_titles[metric_idx], fontsize=24, fontweight='bold',
                             labelpad=10, va='center')
            else:
                ax.set_ylabel('')

            # Column titles only on top row (model names)
            if metric_idx == 0:
                ax.set_title(model, fontsize=26, fontweight='bold', pad=15)

            ax.set_xticks(x)
            ax.set_xticklabels(components, fontsize=18)
            ax.axhline(0, color='black', linewidth=1, alpha=0.5)
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(axis='y', labelsize=16)

            # Set y-axis limits (adjusted for Gemini max values)
            if metric == 'bankruptcy_effect':
                ax.set_ylim(-20, 70)  # Gemini max: 58.6%
            elif metric == 'bet_effect':
                ax.set_ylim(-200, 350)  # Current scale is sufficient
            elif metric == 'rounds_effect':
                ax.set_ylim(-10, 18)  # Current scale is sufficient

            # Legend on all top row charts (each model has different colors)
            if metric_idx == 0:
                ax.legend(fontsize=20, loc='best')

    plt.suptitle('Component Effects by Betting Type - All Models Comparison', fontsize=28, fontweight='bold', y=1.00)
    plt.tight_layout()

    # Add common x-axis label at bottom center (moved further down)
    fig.text(0.5, -0.02, 'Prompt Components', ha='center', fontsize=24, fontweight='bold')

    # Save comprehensive chart
    png_path = OUTPUT_DIR / 'component_effects_all_models_3x4.png'
    pdf_path = OUTPUT_DIR / 'component_effects_all_models_3x4.pdf'

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"Comprehensive chart saved:")
    print(f"  PNG: {png_path}")
    print(f"  PDF: {pdf_path}")


def main():
    print("Loading experiment data...")
    df = load_experiments()
    print(f"Loaded {len(df)} experiments from {df['model'].nunique()} models")

    print(f"Models found: {df['model'].unique().tolist()}")
    print(f"\nGenerating comprehensive 3x4 model comparison chart (no Irrationality Index)...")

    generate_comprehensive_model_chart(df)

    print("\nDone!")


if __name__ == '__main__':
    main()