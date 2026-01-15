#!/usr/bin/env python3
"""Generate 3x4 grid showing complexity trend for each model across 3 metrics (no Irrationality Index)."""

import json
import sys
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

MODEL_DISPLAY_NAMES = {
    'gpt4mini': 'GPT-4o-mini',
    'gpt41mini': 'GPT-4.1-mini',
    'gemini25flash': 'Gemini-2.5-Flash',
    'claude': 'Claude-3.5-Haiku',
}

# Color scheme for each model
MODEL_COLORS = {
    'gpt4mini': '#1f77b4',      # Blue
    'gpt41mini': '#ff7f0e',     # Orange
    'gemini25flash': '#2ca02c',  # Green
    'claude': '#d62728',        # Red
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


def complexity_from_combo(combo: str) -> int:
    """Calculate prompt complexity from combo string"""
    components = ['G', 'M', 'P', 'R', 'W']
    if not combo or combo == 'BASE':
        return 0
    return sum(1 for c in components if c in combo)


def generate_complexity_trend_3x4(df: pd.DataFrame):
    df['complexity'] = df['prompt_combo'].fillna('BASE').apply(complexity_from_combo)

    # Prepare data for all models
    stats_by_model = {}
    for model, group in df.groupby('model'):
        stats = group.groupby('complexity').agg({
            'is_bankrupt': 'mean',
            'total_rounds': 'mean',
            'total_bet': 'mean',
            'complexity': 'count'
        }).rename(columns={'complexity': 'count'}).reset_index()
        stats['bankruptcy_rate'] = stats['is_bankrupt'] * 100
        stats_by_model[model] = stats

    if not stats_by_model:
        print('No complexity data available')
        return

    # Create 3x4 grid figure (removed Irrationality Index row)
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))

    # Metrics to plot (no Irrationality Index)
    metrics = [
        ('bankruptcy_rate', 'Bankruptcy Rate (%)', 'o'),  # Circle
        ('total_rounds', 'Game Rounds', 's'),             # Square
        ('total_bet', 'Total Bet ($)', '^'),              # Triangle
    ]

    model_names = list(MODEL_PATTERNS.keys())

    for row, (metric, ylabel, marker) in enumerate(metrics):
        for col, model in enumerate(model_names):
            ax = axes[row, col]

            if model not in stats_by_model:
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                       ha='center', va='center', fontsize=16)
                ax.set_xlim(-0.2, 5.2)
                continue

            model_data = stats_by_model[model]
            color = MODEL_COLORS[model]

            if len(model_data) == 0:
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                       ha='center', va='center', fontsize=16)
                ax.set_xlim(-0.2, 5.2)
                continue

            # Add trend line first (behind points)
            if len(model_data) > 1:
                z = np.polyfit(model_data['complexity'], model_data[metric], 1)
                p = np.poly1d(z)
                ax.plot(model_data['complexity'], p(model_data['complexity']),
                       color=color, linestyle='--', alpha=0.4, linewidth=1.5, zorder=1)

            # Plot line connecting points (behind markers)
            ax.plot(model_data['complexity'], model_data[metric],
                   color=color, linewidth=2, alpha=0.9, zorder=2)

            # Plot filled white markers first (background)
            ax.scatter(model_data['complexity'], model_data[metric],
                      marker=marker, s=100, facecolors='white', edgecolors='none',
                      linewidths=0, zorder=3)

            # Plot hollow colored markers on top
            ax.scatter(model_data['complexity'], model_data[metric],
                      marker=marker, s=80, facecolors='none', edgecolors=color,
                      linewidths=2, zorder=4)

            # Calculate and display correlation
            if len(model_data) > 1:
                from scipy.stats import pearsonr
                r, p_value = pearsonr(model_data['complexity'], model_data[metric])

                # Add correlation text
                ax.text(0.05, 0.95, f'r = {r:.3f}',
                       transform=ax.transAxes, verticalalignment='top', fontsize=22,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Formatting
            ax.set_xlim(-0.2, 5.2)
            ax.set_xticks([0, 1, 2, 3, 4, 5])
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', which='major', labelsize=16)

            # Add titles only for the top row
            if row == 0:
                ax.set_title(MODEL_DISPLAY_NAMES[model], fontsize=24, fontweight='bold', pad=10)

    # Add overall title
    plt.suptitle('Gambling Behavior vs Prompt Complexity by Model',
                 fontsize=28, fontweight='bold', y=0.995)

    # Add single x-axis label at the bottom center
    fig.text(0.5, 0.02, 'Prompt Complexity (# components)', ha='center', fontsize=24, fontweight='bold')

    # Add y-axis labels manually with individual adjustments
    y_labels = ['Bankruptcy Rate (%)', 'Game Rounds', 'Total Bet ($)']
    y_adjustments = [0.04, 0.0, -0.02]  # Individual adjustments for each label

    for i, (ylabel, adjustment) in enumerate(zip(y_labels, y_adjustments)):
        # Get the actual y-center position of the subplot and apply adjustment
        ax_pos = axes[i, 0].get_position()
        y_center = ax_pos.y0 + ax_pos.height / 2 + adjustment
        fig.text(0.06, y_center, ylabel, ha='center', va='center', rotation=90,
                fontsize=22, fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.08, left=0.10, hspace=0.25, wspace=0.25)

    # Save PNG
    png_path = OUTPUT_DIR / '4model_complexity_trend_3x4.png'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"PNG saved to {png_path}")

    # Save PDF
    pdf_path = OUTPUT_DIR / '4model_complexity_trend_3x4.pdf'
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"PDF saved to {pdf_path}")

    plt.close(fig)


def main():
    print("Loading experiment data...")
    df = load_experiments()
    print(f"Loaded {len(df)} experiments from {df['model'].nunique()} models")

    print("\nGenerating 3x4 complexity trend figure (no Irrationality Index)...")
    generate_complexity_trend_3x4(df)
    print("\nDone!")


if __name__ == '__main__':
    main()
