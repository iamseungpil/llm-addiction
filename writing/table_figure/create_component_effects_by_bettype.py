#!/usr/bin/env python3
"""Generate component effects comparison by betting type (Fixed vs Variable)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')

MODEL_PATTERNS = {
    'gpt4mini': '/data/llm_addiction/gpt_results_fixed_parsing/gpt_fixed_parsing_complete_20250919_151240.json',
    'gpt41mini': '/data/llm_addiction/gpt5_experiment/gpt5_experiment_20250921_174509.json',
    'gemini25flash': '/data/llm_addiction/gemini_experiment/gemini_experiment_20250920_042809.json',
    'claude': '/data/llm_addiction/claude_experiment/claude_experiment_corrected_20250925.json',
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


def generate_component_effects_by_bettype(df: pd.DataFrame):
    components = ['G', 'M', 'P', 'H', 'W']

    # Compute irrationality index for all experiments
    df['irrationality'] = df.apply(lambda row: compute_irrationality_index(row, row['model']), axis=1)

    records = []
    for model, model_group in df.groupby('model'):
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
                    'model': model,
                    'bet_type': bet_type,
                    'component': component,
                    'bankruptcy_effect': (with_comp['is_bankrupt'].fillna(False).mean() - without_comp['is_bankrupt'].fillna(False).mean()) * 100,
                    'bet_effect': with_comp['total_bet'].fillna(0).mean() - without_comp['total_bet'].fillna(0).mean(),
                    'rounds_effect': with_comp['total_rounds'].fillna(0).mean() - without_comp['total_rounds'].fillna(0).mean(),
                    'irrationality_effect': with_comp['irrationality'].fillna(0).mean() - without_comp['irrationality'].fillna(0).mean()
                }
                records.append(effect_row)

    effect_df = pd.DataFrame(records)
    if effect_df.empty:
        print('No component effects to plot')
        return

    # Average across all models for each bet type with std
    avg_effects = effect_df.groupby(['bet_type', 'component']).agg({
        'bankruptcy_effect': ['mean', 'std'],
        'bet_effect': ['mean', 'std'],
        'rounds_effect': ['mean', 'std'],
        'irrationality_effect': ['mean', 'std']
    }).reset_index()

    # Flatten column names
    avg_effects.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in avg_effects.columns.values]

    # Create figure with 4 subplots
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5), sharex=True)
    metric_titles = ['Bankruptcy Effect (%)', 'Total Bet Effect ($)', 'Rounds Effect', 'Irrationality Effect']
    colors = {'fixed': '#2ca02c', 'variable': '#d62728'}  # Green for Fixed, Red for Variable

    for idx, (ax, metric, title) in enumerate(zip(axes, ['bankruptcy_effect', 'bet_effect', 'rounds_effect', 'irrationality_effect'], metric_titles)):
        components = ['G', 'M', 'P', 'H', 'W']

        # Create grouped bar chart
        x = np.arange(len(components))
        width = 0.25

        for i, bet_type in enumerate(['fixed', 'variable']):
            bet_data = avg_effects[avg_effects['bet_type'] == bet_type]
            values = []
            bar_colors = []
            for comp in components:
                comp_data = bet_data[bet_data['component'] == comp]
                if len(comp_data) > 0:
                    values.append(comp_data[f'{metric}_mean'].iloc[0])
                else:
                    values.append(0)

                # Color scheme: Fixed=Green, Variable=Red, but P&H Variable=Light Red
                if bet_type == 'variable' and comp in ['P', 'H']:
                    bar_colors.append('#ff9999')  # Light red for P, H Variable
                else:
                    bar_colors.append(colors[bet_type])

            label = 'Fixed' if bet_type == 'fixed' else 'Variable'
            ax.bar(x + i * width - width/2, values, width, label=label,
                  color=bar_colors, alpha=0.8)

        ax.set_ylabel(title, fontsize=22, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(components, fontsize=22)
        ax.axhline(0, color='black', linewidth=1, alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='y', labelsize=16)

        # Set y-axis limits to show all data with padding
        if metric == 'bankruptcy_effect':
            ax.set_ylim(-8, 28)
        elif metric == 'bet_effect':
            ax.set_ylim(-120, 220)
        elif metric == 'rounds_effect':
            ax.set_ylim(-6, 10)
        elif metric == 'irrationality_effect':
            ax.set_ylim(-0.05, 0.15)

        # Only show legend on leftmost chart
        if idx == 0:
            ax.legend(fontsize=16, loc='best')

    plt.suptitle('Component Effects by Betting Type (Averaged Across Models)', fontsize=30, fontweight='bold', y=0.98)
    plt.tight_layout(w_pad=3.0)

    # Add common x-axis label at bottom center
    fig.text(0.5, -0.02, 'Prompt Components', ha='center', fontsize=24, fontweight='bold')

    png_path = OUTPUT_DIR / 'component_effects_by_bettype.png'
    pdf_path = OUTPUT_DIR / 'component_effects_by_bettype.pdf'

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

    print("\nGenerating component effects by betting type figure...")
    generate_component_effects_by_bettype(df)
    print("\nDone!")


if __name__ == '__main__':
    main()