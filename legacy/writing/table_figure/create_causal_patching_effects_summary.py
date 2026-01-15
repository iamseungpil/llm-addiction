#!/usr/bin/env python3
"""
Create two-panel figure showing causal patching effects and layer distribution
Uses real experimental data from GPU 4 & 5 without hardcoding
"""

import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set plotting style to match SAE feature separation figure
plt.style.use('seaborn-v0_8-whitegrid')

SUMMARY_CSV = Path('/home/ubuntu/llm_addiction/analysis/exp2_feature_group_summary.csv')


def _extract_layer(feature_id: str) -> int:
    prefix = feature_id.split('-')[0]
    if prefix.startswith('L'):
        return int(prefix[1:])
    raise ValueError(f"Unexpected feature id format: {feature_id}")


def load_experimental_data() -> pd.DataFrame:
    """Load aggregated feature effects from latest experiment-2 analysis."""

    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Summary CSV not found: {SUMMARY_CSV}")

    df = pd.read_csv(SUMMARY_CSV)
    if df.empty:
        raise ValueError("Summary CSV is empty.")

    df['layer'] = df['feature'].apply(_extract_layer)
    print(f"Loaded {len(df)} features from summary CSV ({df['feature'].nunique()} unique ids).")
    return df

def create_patching_effects_figure(df: pd.DataFrame) -> None:
    """Create average patching effects comparison figure using aggregated CSV."""

    # Match font settings from SAE feature separation figure
    plt.rcParams.update({
        'font.size': 20,
        'font.family': 'sans-serif',
        'font.weight': 'normal',
        'axes.titlesize': 22,
        'axes.labelsize': 20,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18,
        'axes.linewidth': 1.5,
        'lines.linewidth': 2,
        'grid.alpha': 0.3
    })

    safe_group = df[df['classified_as'] == 'safe']
    risky_group = df[df['classified_as'] == 'risky']

    # 정확한 컬럼 매핑 (기존 이미지와 일치하도록)
    avg_effects = {
        'safe': {
            'safe_stop': safe_group['safe_stop_delta'].mean(),
            'risky_stop': safe_group['risky_stop_delta'].mean(),
            'risky_bankruptcy': safe_group['risky_bankruptcy_delta'].mean(),
        },
        'risky': {
            'safe_stop': risky_group['safe_stop_delta'].mean(),
            'risky_stop': risky_group['risky_stop_delta'].mean(),
            'risky_bankruptcy': risky_group['risky_bankruptcy_delta'].mean(),
        },
    }

    # Calculate standard errors for error bars
    safe_sem = {
        'safe_stop': safe_group['safe_stop_delta'].sem(),
        'risky_stop': safe_group['risky_stop_delta'].sem(),
        'risky_bankruptcy': safe_group['risky_bankruptcy_delta'].sem(),
    }
    risky_sem = {
        'safe_stop': risky_group['safe_stop_delta'].sem(),
        'risky_stop': risky_group['risky_stop_delta'].sem(),
        'risky_bankruptcy': risky_group['risky_bankruptcy_delta'].sem(),
    }

    print("Average effects (safe group):", avg_effects['safe'])
    print("Average effects (risky group):", avg_effects['risky'])

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # 3개 카테고리: Stopping Rate first, then context in parentheses
    categories = ['Stopping Rate\n(Safe Context)', 'Stopping Rate\n(Risky Context)', 'Bankruptcy Rate\n(Risky Context)']
    x_positions = np.arange(len(categories))
    bar_width = 0.28

    # Safe Features (green) values and errors
    safe_values = [
        avg_effects['safe']['safe_stop'],
        avg_effects['safe']['risky_stop'],
        avg_effects['safe']['risky_bankruptcy']
    ]
    safe_errors = [
        safe_sem['safe_stop'],
        safe_sem['risky_stop'],
        safe_sem['risky_bankruptcy']
    ]

    # Risky Features (red) values and errors
    risky_values = [
        avg_effects['risky']['safe_stop'],
        avg_effects['risky']['risky_stop'],
        avg_effects['risky']['risky_bankruptcy']
    ]
    risky_errors = [
        risky_sem['safe_stop'],
        risky_sem['risky_stop'],
        risky_sem['risky_bankruptcy']
    ]

    # Create bars
    bars_safe = ax.bar(x_positions - bar_width/2, safe_values, bar_width,
                       yerr=safe_errors, capsize=5,
                       label='Safe Features', color='#2ca02c', alpha=0.8,
                       edgecolor='black', linewidth=1)

    bars_risky = ax.bar(x_positions + bar_width/2, risky_values, bar_width,
                        yerr=risky_errors, capsize=5,
                        label='Risky Features', color='#d62728', alpha=0.8,
                        edgecolor='black', linewidth=1)

    # Formatting
    ax.set_ylabel('Change in Rate', fontweight='bold', fontsize=20)
    ax.set_xticks(x_positions)

    # Custom labels with different font sizes
    for i, (x_pos, category) in enumerate(zip(x_positions, categories)):
        parts = category.split('\n')
        main_text = parts[0]  # "Stopping Rate" or "Bankruptcy Rate"
        context_text = parts[1]  # "(Safe Context)" etc.

        # Main text (larger) - positioned below chart area
        ax.text(x_pos, -0.235, main_text, ha='center', va='center',
                fontsize=18, fontweight='bold', transform=ax.transData)
        # Context text (smaller) - positioned further below main text
        ax.text(x_pos, -0.265, context_text, ha='center', va='center',
                fontsize=16, fontweight='normal', color='gray', transform=ax.transData)

    # Remove default labels
    ax.set_xticklabels([])
    ax.set_ylim(-0.21, 0.35)  # Back to original limits

    # Adjust margins for tighter layout
    plt.subplots_adjust(bottom=0.12, top=0.85)
    ax.axhline(0, color='black', linewidth=1)
    ax.grid(axis='y', alpha=0.3)

    # Add white background to x-axis labels
    for i, label in enumerate(ax.get_xticklabels()):
        label.set_bbox(dict(boxstyle='round,pad=0.3', facecolor='white',
                           edgecolor='white', alpha=0.9))

    # Legend with black border and larger font - no shadow
    legend = ax.legend(fontsize=20, frameon=True, fancybox=True, shadow=False)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1)

    # Add value labels on bars
    for bars, values in [(bars_safe, safe_values), (bars_risky, risky_values)]:
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.,
                   height + (0.01 if height >= 0 else -0.02),
                   f'{value:+.1%}', ha='center',
                   va='bottom' if height >= 0 else 'top',
                   fontweight='bold', fontsize=16)

    plt.suptitle(
        'Effects of Feature Activation on Gambling Behavior',
        fontsize=24, fontweight='bold', y=0.92
    )

    plt.tight_layout()

    output_path_png = '/home/ubuntu/llm_addiction/writing/figures/causal_patching_average_effects.png'
    output_path_pdf = '/home/ubuntu/llm_addiction/writing/figures/causal_patching_average_effects.pdf'

    os.makedirs(os.path.dirname(output_path_png), exist_ok=True)
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight', facecolor='white')

    print("Patching effects figure saved:")
    print("  PNG:", output_path_png)
    print("  PDF:", output_path_pdf)

    plt.close()

def create_layer_distribution_figure(df: pd.DataFrame) -> None:
    """Create layer distribution chart with safe vs risky counts."""

    # Match font settings from SAE feature separation figure exactly
    plt.rcParams.update({
        'font.size': 24,
        'font.family': 'sans-serif',
        'font.weight': 'normal',
        'axes.titlesize': 26,
        'axes.labelsize': 24,
        'xtick.labelsize': 22,
        'ytick.labelsize': 22,
        'legend.fontsize': 20,
        'axes.linewidth': 1.5,
        'lines.linewidth': 2,
        'grid.alpha': 0.3
    })

    grouped = df.groupby(['layer', 'classified_as']).size().unstack(fill_value=0)
    layers = sorted(grouped.index)
    safe_counts = grouped.get('safe', pd.Series(0, index=layers))
    risky_counts = grouped.get('risky', pd.Series(0, index=layers))

    print("Layer distribution summary:")
    for layer in layers:
        total = grouped.loc[layer].sum()
        print(
            f"  Layer {layer}: safe={safe_counts.loc[layer]}, risky={risky_counts.loc[layer]}, total={total}"
        )

    fig, ax = plt.subplots(figsize=(13, 6.5))
    indices = np.arange(len(layers))
    bar_width = 0.6

    safe_bars = ax.bar(
        indices,
        safe_counts.values,
        bar_width,
        color='#2ca02c',
        edgecolor='black',
        label='Safe Features',
    )
    risky_bars = ax.bar(
        indices,
        risky_counts.values,
        bar_width,
        bottom=safe_counts.values,
        color='#d62728',
        edgecolor='black',
        label='Risky Features',
    )

    ax.set_xticks(indices)
    ax.set_xticklabels([f'L{layer}' for layer in layers], rotation=0, fontweight='bold')  # rotation=0 for straight text
    ax.set_ylabel('Feature Count', fontweight='bold')
    ax.set_title(
        'Layer-wise Distribution of Causal Features',
        fontweight='bold',
        pad=20
    )
    ax.grid(axis='y', alpha=0.3)

    # Legend with black border - moved to upper left
    legend = ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=False)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1)

    total_counts = safe_counts.values + risky_counts.values
    max_total = total_counts.max() if len(total_counts) else 0
    ax.set_ylim(0, 110)  # Increased from 98 to 110 to accommodate L29 bar
    upper_padding = 5  # Fixed padding for text positioning

    for idx, (safe_val, risky_val, total) in enumerate(
        zip(safe_counts.values, risky_counts.values, total_counts)
    ):
        if safe_val > 0:
            ax.text(
                indices[idx],
                safe_val / 2,
                str(int(safe_val)),
                ha='center',
                va='center',
                fontsize=16,
                color='white',
                fontweight='bold',
            )
        if risky_val > 0:
            ax.text(
                indices[idx],
                safe_val + risky_val / 2,
                str(int(risky_val)),
                ha='center',
                va='center',
                fontsize=16,
                color='white',
                fontweight='bold',
            )
        if total > 0:
            ax.text(
                indices[idx],
                safe_val + risky_val + upper_padding * 0.2,
                str(int(total)),
                ha='center',
                va='bottom',
                fontsize=18,
                fontweight='bold',
            )

    plt.tight_layout()

    output_path_png = '/home/ubuntu/llm_addiction/writing/figures/causal_features_layer_distribution_corrected.png'
    output_path_pdf = '/home/ubuntu/llm_addiction/writing/figures/causal_features_layer_distribution_corrected.pdf'

    os.makedirs(os.path.dirname(output_path_png), exist_ok=True)
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight', facecolor='white')

    print("Layer distribution figure saved:")
    print("  PNG:", output_path_png)
    print("  PDF:", output_path_pdf)

    plt.close()

def main():
    """Main function to create both figures"""

    print("=== Creating Causal Patching Analysis Figures ===")
    print("Using real experimental data (no hardcoding)")

    # Load experimental data
    df = load_experimental_data()

    if df.empty:
        print("ERROR: No experimental data loaded!")
        return

    print(f"\nProcessing {len(df)} causal features...")

    # Create figures
    create_patching_effects_figure(df)
    create_layer_distribution_figure(df)

    print("\n=== Analysis Complete ===")
    print("Both figures created based on real GPU 4 & 5 experimental results")

if __name__ == "__main__":
    main()
