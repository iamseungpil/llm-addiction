#!/usr/bin/env python3
"""Generate corrected 4-model irrationality components breakdown using 64 conditions."""

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

COMPONENT_NAMES = {
    'i_ba': 'Betting Aggressiveness',
    'i_lc': 'Loss Chasing',
    'i_eb': 'Extreme Betting',
    'composite': 'Composite Index'
}

OUTPUT_DIR = Path('/home/ubuntu/llm_addiction/writing/figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_experiments() -> pd.DataFrame:
    records = []

    for model, pattern in MODEL_PATTERNS.items():
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


def compute_behavior_metrics(exp, model) -> dict:
    """Compute behavioral metrics with model-aware history extraction"""
    history = extract_game_history(exp, model)

    if not isinstance(history, list) or not history:
        return {'i_ba': 0, 'i_lc': 0, 'i_eb': 0}

    # I_BA: Betting Aggressiveness (평균 베팅 비율)
    betting_ratios = []
    for rd in history:
        if not isinstance(rd, dict):
            continue
        balance_before = reconstruct_balance_before(rd)
        bet = rd.get('bet', 0)
        if balance_before > 0 and isinstance(bet, (int, float)):
            betting_ratios.append(min(bet / balance_before, 1))
    i_ba = float(np.mean(betting_ratios)) if betting_ratios else 0

    # I_LC: Loss Chasing (손실 후 베팅 증가율)
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

    # I_EB: Extreme Betting (잔고의 50% 이상 베팅하는 비율)
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


def generate_corrected_components_breakdown(df: pd.DataFrame):
    """Generate 4x4 components breakdown figure with 64 conditions"""
    rows = []
    for _, exp in df.iterrows():
        model = exp['model']
        metrics = compute_behavior_metrics(exp, model)

        # 🔥 KEY FIX: Use BOTH prompt_combo AND bet_type for full 64 conditions
        full_condition = f"{exp.get('prompt_combo', 'BASE')}_{exp.get('bet_type', 'unknown')}"

        rows.append({
            'model': model,
            'condition': full_condition,  # Now includes both prompt and bet type
            'prompt_combo': exp.get('prompt_combo', 'BASE'),
            'bet_type': exp.get('bet_type', 'unknown'),
            **metrics,
            'bankruptcy': 1 if exp.get('is_bankrupt') else 0
        })

    metrics_df = pd.DataFrame(rows)
    if metrics_df.empty:
        print('No composite metrics available - skipping figure')
        return

    # Composite index 계산
    metrics_df['composite'] = 0.4 * metrics_df['i_ba'] + 0.3 * metrics_df['i_lc'] + 0.3 * metrics_df['i_eb']

    models = list(MODEL_PATTERNS.keys())
    components = ['i_ba', 'i_lc', 'i_eb', 'composite']

    # 4x4 subplot 생성
    fig, axes = plt.subplots(4, 4, figsize=(18, 16))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    print("\n=== CORRECTED 64-CONDITION COMPONENTS ANALYSIS ===")

    for row, component in enumerate(components):
        for col, model in enumerate(models):
            ax = axes[row, col]

            subset = metrics_df[metrics_df['model'] == model]
            if subset.empty:
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                       ha='center', va='center', fontsize=16)
                continue

            # 🔥 KEY FIX: Group by full condition (includes bet_type)
            agg = subset.groupby('condition').agg({
                component: 'mean',
                'bankruptcy': 'mean',
                'bet_type': 'first'  # Keep bet_type info for visualization
            }).reset_index()

            # Print verification for first component/model
            if row == 0 and col == 0:
                print(f"\n{MODEL_DISPLAY_NAMES.get(model, model)} - {COMPONENT_NAMES[component]}:")
                print(f"  Total conditions: {len(agg)} (should be 64)")
                fixed_conditions = len(agg[agg['bet_type'] == 'fixed'])
                variable_conditions = len(agg[agg['bet_type'] == 'variable'])
                print(f"  Fixed betting conditions: {fixed_conditions}")
                print(f"  Variable betting conditions: {variable_conditions}")

            color = colors[col % len(colors)]

            # Separate fixed and variable for different markers
            fixed_agg = agg[agg['bet_type'] == 'fixed']
            variable_agg = agg[agg['bet_type'] == 'variable']

            # Plot with different markers
            if len(fixed_agg) > 0:
                ax.scatter(fixed_agg[component], fixed_agg['bankruptcy'] * 100,
                          alpha=0.6, s=60, color=color, marker='o')

            if len(variable_agg) > 0:
                ax.scatter(variable_agg[component], variable_agg['bankruptcy'] * 100,
                          alpha=0.8, s=80, color=color, marker='^')

            # 회귀선과 상관계수 계산 (전체 64개 조건)
            if len(agg) > 1 and agg[component].std() > 1e-10:
                try:
                    r, p = pearsonr(agg[component], agg['bankruptcy'])

                    if not np.isnan(r) and abs(r) > 0.01:
                        z = np.polyfit(agg[component], agg['bankruptcy'] * 100, 1)
                        p_line = np.poly1d(z)
                        x_line = np.linspace(agg[component].min(), agg[component].max(), 100)
                        ax.plot(x_line, p_line(x_line), color=color, linestyle='-', linewidth=2.5, alpha=0.8)

                        # 상관계수 표시
                        ax.text(0.05, 0.95, f'r = {r:.3f}',
                               transform=ax.transAxes, verticalalignment='top', fontsize=18,
                               bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='gray'))

                        # Print correlation for verification
                        if row == 3 and col == 0:  # Composite index for first model
                            print(f"  Composite correlation (64 conditions): r = {r:.3f}")

                except (np.linalg.LinAlgError, ValueError, TypeError):
                    pass

            # 축 레이블 및 제목 설정
            if row == 0:  # 첫 번째 행에만 모델명 표시
                ax.set_title(f'{MODEL_DISPLAY_NAMES.get(model, model)}', fontsize=20, fontweight='bold')

            if col == 0:  # 첫 번째 열에만 y축 레이블 표시
                ax.set_ylabel('Bankruptcy Rate (%)', fontsize=18, fontweight='bold')

            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.grid(True, alpha=0.3)

            # 축 범위 설정
            if not agg.empty:
                x_min, x_max = agg[component].min(), agg[component].max()
                x_padding = max((x_max - x_min) * 0.15, 0.01)
                ax.set_xlim(max(-0.02, x_min - x_padding), x_max + x_padding)

                y_min, y_max = (agg['bankruptcy'] * 100).min(), (agg['bankruptcy'] * 100).max()
                y_padding = max((y_max - y_min) * 0.15, 2)
                ax.set_ylim(max(-2, y_min - y_padding), y_max + y_padding)
            else:
                ax.set_xlim(-0.05, 0.5)
                ax.set_ylim(-2, 30)

    plt.suptitle('Corrected 64-Condition Components: Correlation with Bankruptcy Rate',
                 fontsize=26, fontweight='bold', y=0.92)

    plt.subplots_adjust(left=0.08, right=0.95, top=0.88, bottom=0.08, hspace=0.4, wspace=0.25)

    # 각 컴포넌트별 중앙 x축 레이블 추가
    row_centers = []
    for row in range(4):
        if row < 3:
            current_row_bottom = axes[row, 0].get_position().y0
            next_row_top = axes[row + 1, 0].get_position().y1
            center_y = (current_row_bottom + next_row_top) / 2
        else:
            ax_bottom = axes[row, 0].get_position().y0
            center_y = ax_bottom - 0.04
        row_centers.append(center_y)

    for i, component in enumerate(components):
        fig.text(0.5, row_centers[i], f'{COMPONENT_NAMES[component]} Score',
                ha='center', fontsize=18, fontweight='bold')

    png_path = OUTPUT_DIR / 'CORRECTED_64condition_components_breakdown.png'
    pdf_path = OUTPUT_DIR / 'CORRECTED_64condition_components_breakdown.pdf'

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"\nCorrected components breakdown figure saved to:")
    print(f"  PNG: {png_path}")
    print(f"  PDF: {pdf_path}")

    return metrics_df


def main():
    print("Loading experiment data...")
    df = load_experiments()
    print(f"Loaded {len(df)} experiments from {df['model'].nunique()} models")

    print("\nGenerating CORRECTED 64-condition components breakdown figure...")
    metrics_df = generate_corrected_components_breakdown(df)
    print("\nDone!")


if __name__ == '__main__':
    main()