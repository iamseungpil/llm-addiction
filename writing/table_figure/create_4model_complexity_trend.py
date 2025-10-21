#!/usr/bin/env python3
"""Generate 4-model complexity trend comparison figure."""

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
    'claude': 'Claude-3.5-Sonnet',
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


def generate_complexity_trend(df: pd.DataFrame):
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

    # Create 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
    markers = ['o', 's', '^', 'D']  # Circle, Square, Triangle, Diamond

    # Define model order to match previous figures
    model_order = ['gpt4mini', 'gpt41mini', 'gemini25flash', 'claude']

    # Metrics to plot
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

            # Add trend line
            if len(group) > 1:
                z = np.polyfit(group['complexity'], group[metric], 1)
                p = np.poly1d(z)
                ax.plot(group['complexity'], p(group['complexity']),
                       color=color, linestyle='--', alpha=0.5, linewidth=1)

        ax.set_xlabel('Prompt Complexity (# components)', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        if ax == axes[0]:  # Only show legend on first subplot
            ax.legend(fontsize=11, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.1, 5.1)

    plt.suptitle('4-Model Complexity Effects Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save PNG
    png_path = OUTPUT_DIR / '4model_complexity_trend.png'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"PNG saved to {png_path}")

    # Save PDF
    pdf_path = OUTPUT_DIR / '4model_complexity_trend.pdf'
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"PDF saved to {pdf_path}")

    plt.close(fig)


def main():
    print("Loading experiment data...")
    df = load_experiments()
    print(f"Loaded {len(df)} experiments from {df['model'].nunique()} models")

    print("\nGenerating complexity trend figure...")
    generate_complexity_trend(df)
    print("\nDone!")


if __name__ == '__main__':
    main()