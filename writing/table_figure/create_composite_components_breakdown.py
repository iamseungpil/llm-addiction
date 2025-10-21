#!/usr/bin/env python3
"""Generate 4-model irrationality components breakdown figure (4x4 = 16 panels)."""

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


def generate_components_breakdown(df: pd.DataFrame):
    """Generate 4x4 components breakdown figure"""
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

    # Composite index 계산
    metrics_df['composite'] = 0.4 * metrics_df['i_ba'] + 0.3 * metrics_df['i_lc'] + 0.3 * metrics_df['i_eb']

    models = list(MODEL_PATTERNS.keys())
    components = ['i_ba', 'i_lc', 'i_eb', 'composite']

    # 개별 차트별 축 범위 계산 (각각 맞춤형 설정)

    # 4x4 subplot 생성 (행 간격 확대)
    fig, axes = plt.subplots(4, 4, figsize=(18, 16))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for row, component in enumerate(components):
        for col, model in enumerate(models):
            ax = axes[row, col]

            subset = metrics_df[metrics_df['model'] == model]
            if subset.empty:
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                       ha='center', va='center', fontsize=16)
                continue

            agg = subset.groupby('condition').agg({
                component: 'mean',
                'bankruptcy': 'mean'
            }).reset_index()

            color = colors[col % len(colors)]
            ax.scatter(agg[component], agg['bankruptcy'] * 100, alpha=0.7, s=80, color=color)

            # 회귀선과 상관계수 계산
            if len(agg) > 1 and agg[component].std() > 1e-10:
                try:
                    r, p = pearsonr(agg[component], agg['bankruptcy'])

                    if not np.isnan(r) and abs(r) > 0.01:
                        z = np.polyfit(agg[component], agg['bankruptcy'] * 100, 1)
                        p_line = np.poly1d(z)
                        x_line = np.linspace(agg[component].min(), agg[component].max(), 100)
                        ax.plot(x_line, p_line(x_line), color=color, linestyle='-', linewidth=2.5, alpha=0.8)

                        # 상관계수 표시 (폰트 크기 증가)
                        ax.text(0.05, 0.95, f'r = {r:.3f}',
                               transform=ax.transAxes, verticalalignment='top', fontsize=18,
                               bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='gray'))
                except (np.linalg.LinAlgError, ValueError, TypeError):
                    pass

            # 축 레이블 및 제목 설정 (모든 폰트 크기 증가)
            if row == 0:  # 첫 번째 행에만 모델명 표시
                ax.set_title(f'{MODEL_DISPLAY_NAMES.get(model, model)}', fontsize=20, fontweight='bold')

            if col == 0:  # 첫 번째 열에만 y축 레이블 표시
                ax.set_ylabel('Bankruptcy Rate (%)', fontsize=18, fontweight='bold')

            # 개별 차트의 x축 레이블 제거 (중앙 통합 레이블로 대체)

            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.grid(True, alpha=0.3)

            # 개별 차트별 맞춤형 축 범위 설정
            if not agg.empty:
                # X축 범위: 데이터에 맞게 개별 설정
                x_min, x_max = agg[component].min(), agg[component].max()
                x_padding = max((x_max - x_min) * 0.15, 0.01)  # 최소 패딩 보장
                ax.set_xlim(max(-0.02, x_min - x_padding), x_max + x_padding)

                # Y축 범위: 데이터에 맞게 개별 설정
                y_min, y_max = (agg['bankruptcy'] * 100).min(), (agg['bankruptcy'] * 100).max()
                y_padding = max((y_max - y_min) * 0.15, 2)  # 최소 2% 패딩 보장
                ax.set_ylim(max(-2, y_min - y_padding), y_max + y_padding)
            else:
                # 데이터가 없는 경우 기본 범위
                ax.set_xlim(-0.05, 0.5)
                ax.set_ylim(-2, 30)

    plt.suptitle('Irrationality Components Breakdown: Correlation with Bankruptcy Rate',
                 fontsize=26, fontweight='bold', y=0.92)

    # 행 간격을 넓혀서 레이블 공간 확보
    plt.subplots_adjust(left=0.08, right=0.95, top=0.88, bottom=0.08, hspace=0.4, wspace=0.25)

    # 각 컴포넌트별 중앙 x축 레이블 추가 (행과 행 사이의 정확한 중간점에 배치)
    row_centers = []
    for row in range(4):
        if row < 3:  # 첫 3개 행의 경우 - 현재 행의 아래와 다음 행의 위 사이 중점
            current_row_bottom = axes[row, 0].get_position().y0
            next_row_top = axes[row + 1, 0].get_position().y1
            center_y = (current_row_bottom + next_row_top) / 2
        else:  # 마지막 행인 경우 - 차트 아래쪽
            ax_bottom = axes[row, 0].get_position().y0
            center_y = ax_bottom - 0.04
        row_centers.append(center_y)

    for i, component in enumerate(components):
        fig.text(0.5, row_centers[i], f'{COMPONENT_NAMES[component]} Score',
                ha='center', fontsize=18, fontweight='bold')

    png_path = OUTPUT_DIR / '4model_irrationality_components_breakdown.png'
    pdf_path = OUTPUT_DIR / '4model_irrationality_components_breakdown.pdf'

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"Components breakdown figure saved to:")
    print(f"  PNG: {png_path}")
    print(f"  PDF: {pdf_path}")

    # 통계 요약 출력
    print(f"\n=== COMPONENT STATISTICS SUMMARY ===")
    for model in models:
        subset = metrics_df[metrics_df['model'] == model]
        if not subset.empty:
            print(f"\n{MODEL_DISPLAY_NAMES.get(model, model)}:")
            print(f"  Experiments: {len(subset)}")
            print(f"  Bankruptcy Rate: {subset['bankruptcy'].mean():.3f}")
            for comp in ['i_ba', 'i_lc', 'i_eb', 'composite']:
                mean_val = subset[comp].mean()
                std_val = subset[comp].std()
                print(f"  {COMPONENT_NAMES[comp]}: {mean_val:.3f} ± {std_val:.3f}")


def main():
    print("Loading experiment data...")
    df = load_experiments()
    print(f"Loaded {len(df)} experiments from {df['model'].nunique()} models")

    print("\nGenerating irrationality components breakdown figure...")
    generate_components_breakdown(df)
    print("\nDone!")


if __name__ == '__main__':
    main()