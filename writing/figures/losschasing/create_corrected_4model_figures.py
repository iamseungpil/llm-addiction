#!/usr/bin/env python3
"""
Regenerate all 4model figures with corrected loss chasing definition.

Corrected i_lc definition:
- OLD: Fraction of times bet amount increased after loss
- NEW: Fraction of times (bet/balance) ratio increased after loss

This better captures risk escalation even when balance decreases.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

OUTPUT_DIR = Path('/home/ubuntu/llm_addiction/writing/figures/losschasing')
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
    """Reconstruct balance before bet from round data."""
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


def compute_i_lc_corrected(history) -> float:
    """
    CORRECTED Loss Chasing Index.

    Computes: (# times bet/balance ratio increased after loss) / (# loss opportunities)

    This better captures risk escalation:
    - Even if bet stays same, ratio increases as balance decreases
    - Measures actual risk increase, not just nominal bet increase
    """
    if len(history) < 2:
        return 0.0

    chase_events = 0
    opportunities = 0

    for i in range(1, len(history)):
        prev = history[i - 1]
        curr = history[i]

        if not isinstance(prev, dict) or not isinstance(curr, dict):
            continue

        # Only consider losses in previous round
        if prev.get('win', False):
            continue

        prev_bet = prev.get('bet')
        curr_bet = curr.get('bet')
        if not isinstance(prev_bet, (int, float)) or not isinstance(curr_bet, (int, float)):
            continue
        if prev_bet <= 0 or curr_bet <= 0:
            continue

        # Reconstruct balances
        prev_balance = reconstruct_balance_before(prev)
        curr_balance = reconstruct_balance_before(curr)

        if prev_balance <= 0 or curr_balance <= 0:
            continue

        # Compute betting ratios
        prev_ratio = min(prev_bet / prev_balance, 1.0)
        curr_ratio = min(curr_bet / curr_balance, 1.0)

        opportunities += 1

        # Check if ratio increased (= risk escalation)
        if curr_ratio > prev_ratio:
            chase_events += 1

    return chase_events / opportunities if opportunities > 0 else 0.0


def compute_behavior_metrics(exp, model) -> dict:
    """Compute behavioral metrics with CORRECTED i_lc"""
    history = extract_game_history(exp, model)

    if not isinstance(history, list) or not history:
        return {'i_ev': 0, 'i_lc': 0, 'i_eb': 0}

    # i_ev: Expected Value Ignorance
    betting_ratios = []
    for rd in history:
        if not isinstance(rd, dict):
            continue
        balance_before = reconstruct_balance_before(rd)
        bet = rd.get('bet', 0)
        if balance_before > 0 and isinstance(bet, (int, float)):
            betting_ratios.append(min(bet / balance_before, 1))
    i_ev = float(np.mean(betting_ratios)) if betting_ratios else 0

    # i_lc: Loss Chasing (CORRECTED)
    i_lc = compute_i_lc_corrected(history)

    # i_eb: Extreme Betting
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


def complexity_from_combo(combo: str) -> int:
    """Calculate prompt complexity from combo string"""
    components = ['G', 'M', 'P', 'R', 'W']
    if not combo or combo == 'BASE':
        return 0
    return sum(1 for c in components if c in combo)


def generate_composite_indices(df: pd.DataFrame):
    """Generate composite indices figure (CORRECTED)."""
    print("Generating: 4model_composite_indices_corrected.png")

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

    models = ['gpt4mini', 'gpt41mini', 'gemini25flash', 'claude']
    n_models = len(models)

    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 3.5), sharey=True)

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

        current_xlim = ax.get_xlim()
        ax.set_xlim(-0.02, current_xlim[1])

    plt.suptitle('Irrationality-Bankruptcy Correlation (Corrected i_lc)', fontsize=28, fontweight='bold', y=0.98)
    plt.tight_layout(w_pad=3.0)

    fig.text(0.5, -0.02, 'Composite Irrationality Index', ha='center', fontsize=24, fontweight='bold')

    png_path = OUTPUT_DIR / '4model_composite_indices_corrected.png'
    pdf_path = OUTPUT_DIR / '4model_composite_indices_corrected.pdf'

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved: {png_path}")


def generate_complexity_trend(df: pd.DataFrame):
    """Generate complexity trend figure (CORRECTED)."""
    print("Generating: 4model_complexity_trend_corrected.png")

    df['complexity'] = df['prompt_combo'].fillna('BASE').apply(complexity_from_combo)

    stats_rows = []
    for model, group in df.groupby('model'):
        stats = group.groupby('complexity').agg({
            'is_bankrupt': 'mean',
            'total_rounds': 'mean',
            'total_bet': 'mean',
            'complexity': 'count'
        }).rename(columns={'complexity': 'count'}).reset_index()
        stats['bankruptcy_rate'] = stats['is_bankrupt'] * 100
        stats['model'] = model
        stats_rows.append(stats)

    plot_df = pd.concat(stats_rows, ignore_index=True)
    if plot_df.empty:
        print('No complexity data available')
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']

    model_order = ['gpt4mini', 'gpt41mini', 'gemini25flash', 'claude']

    metrics = [
        ('bankruptcy_rate', 'Bankruptcy Rate (%)', 'Bankruptcy Rate'),
        ('total_rounds', 'Average Game Rounds', 'Game Persistence'),
        ('total_bet', 'Average Total Bet ($)', 'Total Bet Size')
    ]

    for ax, (metric, ylabel, title) in zip(axes, metrics):
        for i, model in enumerate(model_order):
            group = plot_df[plot_df['model'] == model]
            if group.empty:
                continue
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            label = MODEL_DISPLAY_NAMES.get(model, model)

            ax.plot(group['complexity'], group[metric],
                   marker=marker, color=color, label=label, linewidth=2.5,
                   markersize=8, alpha=0.8)

            if len(group) > 1:
                z = np.polyfit(group['complexity'], group[metric], 1)
                p = np.poly1d(z)
                ax.plot(group['complexity'], p(group['complexity']),
                       color=color, linestyle='--', alpha=0.5, linewidth=1)

        ax.set_xlabel('Prompt Complexity (# components)', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        if ax == axes[0]:
            ax.legend(fontsize=11, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.1, 5.1)

    plt.suptitle('4-Model Complexity Effects (Corrected i_lc)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    png_path = OUTPUT_DIR / '4model_complexity_trend_corrected.png'
    pdf_path = OUTPUT_DIR / '4model_complexity_trend_corrected.pdf'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved: {png_path}")


def generate_component_effects(df: pd.DataFrame):
    """Generate component effects figure (CORRECTED)."""
    print("Generating: 4model_component_effects_corrected.png")

    components = ['G', 'M', 'P', 'R', 'W']

    records = []
    for model, model_group in df.groupby('model'):
        for component in components:
            with_comp = model_group[model_group['prompt_combo'].fillna('').str.contains(component)]
            without_comp = model_group[~model_group['prompt_combo'].fillna('').str.contains(component)]

            if with_comp.empty or without_comp.empty:
                continue

            effect_row = {
                'model': model,
                'component': component,
                'bankruptcy_effect': (with_comp['is_bankrupt'].fillna(False).mean() - without_comp['is_bankrupt'].fillna(False).mean()) * 100,
                'bet_effect': with_comp['total_bet'].fillna(0).mean() - without_comp['total_bet'].fillna(0).mean(),
                'rounds_effect': with_comp['total_rounds'].fillna(0).mean() - without_comp['total_rounds'].fillna(0).mean()
            }
            records.append(effect_row)

    effect_df = pd.DataFrame(records)
    if effect_df.empty:
        print('No component effects to plot')
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    metric_titles = ['Bankruptcy Effect (%)', 'Total Bet Effect ($)', 'Rounds Effect']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for idx, (ax, metric, title) in enumerate(zip(axes, ['bankruptcy_effect', 'bet_effect', 'rounds_effect'], metric_titles)):
        models = ['gpt4mini', 'gpt41mini', 'gemini25flash', 'claude']
        components = ['G', 'M', 'P', 'R', 'W']

        x = np.arange(len(components))
        width = 0.2
        for i, model in enumerate(models):
            model_data = effect_df[effect_df['model'] == model]
            values = []
            for comp in components:
                comp_data = model_data[model_data['component'] == comp]
                values.append(comp_data[metric].iloc[0] if len(comp_data) > 0 else 0)

            display_name = MODEL_DISPLAY_NAMES.get(model, model)
            ax.bar(x + i * width, values, width, label=display_name,
                  color=colors[i % len(colors)], alpha=0.8)

        ax.set_title(title, fontsize=20, fontweight='bold')
        ax.set_ylabel(title, fontsize=18, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(components, fontsize=16)
        ax.axhline(0, color='black', linewidth=1, alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='y', labelsize=16)

        if metric == 'bankruptcy_effect':
            ax.set_ylim(-15, 35)
        elif metric == 'bet_effect':
            ax.set_ylim(-130, 180)
        elif metric == 'rounds_effect':
            ax.set_ylim(-6, 7)

        if idx == 0:
            ax.legend(loc='upper right', fontsize=14)

    plt.suptitle('4-Model Component Effects (Corrected i_lc)', fontsize=22, fontweight='bold', y=0.98)
    plt.tight_layout(w_pad=3.0)

    fig.text(0.5, -0.02, 'Prompt Components', ha='center', fontsize=18, fontweight='bold')

    png_path = OUTPUT_DIR / '4model_component_effects_corrected.png'
    pdf_path = OUTPUT_DIR / '4model_component_effects_corrected.pdf'

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved: {png_path}")


def generate_components_breakdown(df: pd.DataFrame):
    """Generate irrationality components breakdown (CORRECTED)."""
    print("Generating: 4model_irrationality_components_breakdown_corrected.png")

    rows = []
    for _, exp in df.iterrows():
        model = exp['model']
        metrics = compute_behavior_metrics(exp, model)
        rows.append({
            'model': model,
            'condition': exp.get('prompt_combo', 'BASE'),
            **metrics
        })

    metrics_df = pd.DataFrame(rows)
    if metrics_df.empty:
        return

    models = ['gpt4mini', 'gpt41mini', 'gemini25flash', 'claude']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for idx, model in enumerate(models):
        subset = metrics_df[metrics_df['model'] == model]
        agg = subset.groupby('condition').agg({
            'i_ev': 'mean',
            'i_lc': 'mean',
            'i_eb': 'mean'
        }).reset_index()

        ax = axes[idx]
        conditions = agg['condition'].tolist()
        x = np.arange(len(conditions))
        width = 0.25

        ax.bar(x - width, agg['i_ev'], width, label='i_ev (EV Ignorance)', color='#8dd3c7', alpha=0.8)
        ax.bar(x, agg['i_lc'], width, label='i_lc (Loss Chasing)', color='#fb8072', alpha=0.8)
        ax.bar(x + width, agg['i_eb'], width, label='i_eb (Extreme Betting)', color='#bebada', alpha=0.8)

        display_name = MODEL_DISPLAY_NAMES.get(model, model)
        ax.set_title(display_name, fontsize=18, fontweight='bold')
        ax.set_ylabel('Index Value', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=45, ha='right', fontsize=8)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Irrationality Components Breakdown (Corrected i_lc)', fontsize=20, fontweight='bold', y=0.995)
    plt.tight_layout()

    png_path = OUTPUT_DIR / '4model_irrationality_components_breakdown_corrected.png'
    pdf_path = OUTPUT_DIR / '4model_irrationality_components_breakdown_corrected.pdf'

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved: {png_path}")


def main():
    print("="*70)
    print("Regenerating 4model Figures with CORRECTED Loss Chasing Definition")
    print("="*70)
    print("\nCorrected i_lc:")
    print("  OLD: (# bet increases after loss) / (# losses)")
    print("  NEW: (# bet/balance ratio increases after loss) / (# losses)")
    print("\nThis better captures risk escalation!\n")
    print("="*70)

    print("\nLoading experiment data...")
    df = load_experiments()
    print(f"Loaded {len(df)} experiments from {df['model'].nunique()} models")

    print("\nGenerating figures...")
    generate_composite_indices(df)
    generate_complexity_trend(df)
    generate_component_effects(df)
    generate_components_breakdown(df)

    print("\n" + "="*70)
    print("All figures saved to:")
    print(f"  {OUTPUT_DIR}")
    print("="*70)


if __name__ == '__main__':
    main()
