#!/usr/bin/env python3
"""Generate 4x3 component effects figure - each row is one model, each column is one metric."""

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


def generate_component_effects_4x3(df: pd.DataFrame):
    components = ['G', 'M', 'P', 'H', 'W']
    models = ['gpt4mini', 'gpt41mini', 'gemini25flash', 'claude']
    metrics = ['bankruptcy_effect', 'bet_effect', 'rounds_effect']
    metric_titles = ['Bankruptcy Effect (%)', 'Total Bet Effect ($)', 'Rounds Effect']

    # Calculate component effects for each model
    all_effects = {}
    for model in models:
        model_group = df[df['model'] == model]
        all_effects[model] = {}

        for component in components:
            # Map H back to R for data matching
            search_component = 'R' if component == 'H' else component
            with_comp = model_group[model_group['prompt_combo'].fillna('').str.contains(search_component)]
            without_comp = model_group[~model_group['prompt_combo'].fillna('').str.contains(search_component)]

            if with_comp.empty or without_comp.empty:
                all_effects[model][component] = {
                    'bankruptcy_effect': 0,
                    'bet_effect': 0,
                    'rounds_effect': 0
                }
                continue

            all_effects[model][component] = {
                'bankruptcy_effect': (with_comp['is_bankrupt'].fillna(False).mean() - without_comp['is_bankrupt'].fillna(False).mean()) * 100,
                'bet_effect': with_comp['total_bet'].fillna(0).mean() - without_comp['total_bet'].fillna(0).mean(),
                'rounds_effect': with_comp['total_rounds'].fillna(0).mean() - without_comp['total_rounds'].fillna(0).mean()
            }

    # Create 4x3 figure
    fig, axes = plt.subplots(4, 3, figsize=(18, 16), sharex=True)
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']  # Different colors for each component

    for row, model in enumerate(models):
        model_name = MODEL_DISPLAY_NAMES[model]

        for col, (metric, metric_title) in enumerate(zip(metrics, metric_titles)):
            ax = axes[row, col]

            # Extract values for this model and metric
            values = []
            for component in components:
                values.append(all_effects[model][component][metric])

            # Create bar chart for this model and metric
            bars = ax.bar(components, values, color=colors, alpha=0.8)

            # Customize subplot
            ax.axhline(0, color='black', linewidth=1, alpha=0.5)
            ax.grid(True, alpha=0.3, axis='y')

            # Set y-axis limits based on metric type
            if metric == 'bankruptcy_effect':
                ax.set_ylim(-15, 35)
            elif metric == 'bet_effect':
                ax.set_ylim(-130, 180)
            elif metric == 'rounds_effect':
                ax.set_ylim(-6, 7)

            # Add model name as y-label for first column
            if col == 0:
                ax.set_ylabel(model_name, fontsize=14, fontweight='bold', rotation=90, va='center')

            # Add metric title for top row
            if row == 0:
                ax.set_title(metric_title, fontsize=16, fontweight='bold')

            # Only show x-axis labels for bottom row
            if row == len(models) - 1:
                ax.set_xlabel('Prompt Components', fontsize=12, fontweight='bold')

            # Adjust tick label sizes
            ax.tick_params(axis='both', labelsize=11)

    plt.suptitle('Component Effects by Model (4×3 Layout)', fontsize=20, fontweight='bold', y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    # Save the figure
    png_path = OUTPUT_DIR / 'component_effects_4x3_by_model.png'
    pdf_path = OUTPUT_DIR / 'component_effects_4x3_by_model.pdf'

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"4x3 Component Effects Figure saved to:")
    print(f"  PNG: {png_path}")
    print(f"  PDF: {pdf_path}")


def main():
    print("Loading experiment data...")
    df = load_experiments()
    print(f"Loaded {len(df)} experiments from {df['model'].nunique()} models")

    print("\nGenerating 4x3 component effects figure...")
    generate_component_effects_4x3(df)
    print("\nDone!")


if __name__ == '__main__':
    main()