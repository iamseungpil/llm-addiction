#!/usr/bin/env python3
"""Generate 4-model component effects comparison figure."""

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


def generate_component_effects(df: pd.DataFrame):
    components = ['G', 'M', 'P', 'H', 'W']

    records = []
    for model, model_group in df.groupby('model'):
        for component in components:
            # Map H back to R for data matching
            search_component = 'R' if component == 'H' else component
            with_comp = model_group[model_group['prompt_combo'].fillna('').str.contains(search_component)]
            without_comp = model_group[~model_group['prompt_combo'].fillna('').str.contains(search_component)]

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

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    metric_titles = ['Bankruptcy Effect (%)', 'Total Bet Effect ($)', 'Rounds Effect']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red

    for idx, (ax, metric, title) in enumerate(zip(axes, ['bankruptcy_effect', 'bet_effect', 'rounds_effect'], metric_titles)):
        models = ['gpt4mini', 'gpt41mini', 'gemini25flash', 'claude']
        components = ['G', 'M', 'P', 'H', 'W']

        # Create grouped bar chart
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

        # Set y-axis limits to make charts more compact
        if metric == 'bankruptcy_effect':
            ax.set_ylim(-15, 35)
        elif metric == 'bet_effect':
            ax.set_ylim(-130, 180)
        elif metric == 'rounds_effect':
            ax.set_ylim(-6, 7)

        if idx == 0:  # Only show legend on first subplot
            ax.legend(loc='upper right', fontsize=14)

    plt.suptitle('4-Model Component Effects Comparison', fontsize=22, fontweight='bold', y=0.98)
    plt.tight_layout(w_pad=3.0)

    # Add common x-axis label at bottom center
    fig.text(0.5, -0.02, 'Prompt Components', ha='center', fontsize=18, fontweight='bold')

    png_path = OUTPUT_DIR / '4model_component_effects.png'
    pdf_path = OUTPUT_DIR / '4model_component_effects.pdf'

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

    print("\nGenerating component effects figure...")
    generate_component_effects(df)
    print("\nDone!")


if __name__ == '__main__':
    main()