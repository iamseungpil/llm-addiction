#!/usr/bin/env python3
"""
L1-31 실험 결과 시각화 (TRUE 4-way consistency)
원본 441개 분석과 동일한 형식으로 생성 (NO HARDCODING, NO HALLUCINATION)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 데이터 경로 (CORRECT 일관성 분석 결과)
TRUE_SAFE_CSV = Path("/home/ubuntu/llm_addiction/analysis/CORRECT_consistent_safe_features.csv")
TRUE_RISKY_CSV = Path("/home/ubuntu/llm_addiction/analysis/CORRECT_consistent_risky_features.csv")
OUTPUT_DIR = Path("/home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/analysis/figures")

# 원본 figure와 동일한 스타일 설정
plt.style.use('seaborn-v0_8-whitegrid')


def load_true_4way_data():
    """Load CORRECT consistent features (NO HARDCODING)"""
    if not TRUE_SAFE_CSV.exists() or not TRUE_RISKY_CSV.exists():
        raise FileNotFoundError(
            f"CORRECT consistent CSV files not found:\n"
            f"  Safe: {TRUE_SAFE_CSV}\n"
            f"  Risky: {TRUE_RISKY_CSV}\n"
            f"Run CORRECT_consistent_features.py first!"
        )

    safe_df = pd.read_csv(TRUE_SAFE_CSV)
    risky_df = pd.read_csv(TRUE_RISKY_CSV)

    print(f"Loaded CORRECT consistent features:")
    print(f"  Safe: {len(safe_df)} features")
    print(f"  Risky: {len(risky_df)} features")
    print(f"  Total: {len(safe_df) + len(risky_df)} features")

    return safe_df, risky_df


def create_average_effects_figure(safe_df, risky_df):
    """
    Figure 1: Average patching effects comparison
    원본: /home/ubuntu/llm_addiction/writing/figures/causal_patching_average_effects.png
    """
    # Font settings (match original)
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

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Calculate average effects (NO HARDCODING - from actual data)
    safe_avg = {
        'safe_stop': safe_df['safe_stop_delta'].mean(),
        'risky_stop': safe_df['risky_stop_delta'].mean(),
        'risky_bankruptcy': safe_df['risky_bankruptcy_delta'].mean(),
    }

    risky_avg = {
        'safe_stop': risky_df['safe_stop_delta'].mean(),
        'risky_stop': risky_df['risky_stop_delta'].mean(),
        'risky_bankruptcy': risky_df['risky_bankruptcy_delta'].mean(),
    }

    # Calculate standard errors
    safe_sem = {
        'safe_stop': safe_df['safe_stop_delta'].sem(),
        'risky_stop': safe_df['risky_stop_delta'].sem(),
        'risky_bankruptcy': safe_df['risky_bankruptcy_delta'].sem(),
    }

    risky_sem = {
        'safe_stop': risky_df['safe_stop_delta'].sem(),
        'risky_stop': risky_df['risky_stop_delta'].sem(),
        'risky_bankruptcy': risky_df['risky_bankruptcy_delta'].sem(),
    }

    print(f"\nAverage effects (Safe features, n={len(safe_df)}):")
    for key, val in safe_avg.items():
        print(f"  {key}: {val:+.4f} (SEM: {safe_sem[key]:.4f})")

    print(f"\nAverage effects (Risky features, n={len(risky_df)}):")
    for key, val in risky_avg.items():
        print(f"  {key}: {val:+.4f} (SEM: {risky_sem[key]:.4f})")

    # 3 categories (match original)
    categories = [
        'Stopping Rate\n(Safe Context)',
        'Stopping Rate\n(Risky Context)',
        'Bankruptcy Rate\n(Risky Context)'
    ]
    x_positions = np.arange(len(categories))
    bar_width = 0.28

    # Safe Features values
    safe_values = [
        safe_avg['safe_stop'],
        safe_avg['risky_stop'],
        safe_avg['risky_bankruptcy']
    ]
    safe_errors = [
        safe_sem['safe_stop'],
        safe_sem['risky_stop'],
        safe_sem['risky_bankruptcy']
    ]

    # Risky Features values
    risky_values = [
        risky_avg['safe_stop'],
        risky_avg['risky_stop'],
        risky_avg['risky_bankruptcy']
    ]
    risky_errors = [
        risky_sem['safe_stop'],
        risky_sem['risky_stop'],
        risky_sem['risky_bankruptcy']
    ]

    # Create bars (match original colors)
    bars_safe = ax.bar(
        x_positions - bar_width/2, safe_values, bar_width,
        yerr=safe_errors, capsize=5,
        label='Safe Features', color='#2ca02c', alpha=0.8,
        edgecolor='black', linewidth=1
    )

    bars_risky = ax.bar(
        x_positions + bar_width/2, risky_values, bar_width,
        yerr=risky_errors, capsize=5,
        label='Risky Features', color='#d62728', alpha=0.8,
        edgecolor='black', linewidth=1
    )

    # Formatting (match original)
    ax.set_ylabel('Change in Rate', fontweight='bold', fontsize=20)
    ax.set_xticks(x_positions)

    # Custom labels (match original)
    for i, (x_pos, category) in enumerate(zip(x_positions, categories)):
        parts = category.split('\n')
        main_text = parts[0]
        context_text = parts[1]

        ax.text(x_pos, -0.56, main_text, ha='center', va='center',
                fontsize=18, fontweight='bold', transform=ax.transData)
        ax.text(x_pos, -0.62, context_text, ha='center', va='center',
                fontsize=16, fontweight='normal', color='gray', transform=ax.transData)

    ax.set_xticklabels([])
    ax.set_ylim(-0.50, 0.25)

    plt.subplots_adjust(bottom=0.18, top=0.92)
    ax.axhline(0, color='black', linewidth=1)
    ax.grid(axis='y', alpha=0.3)

    # Legend (match original)
    legend = ax.legend(fontsize=20, frameon=True, fancybox=True, shadow=False)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1)

    # Value labels on bars
    for bars, values in [(bars_safe, safe_values), (bars_risky, risky_values)]:
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + (0.01 if height >= 0 else -0.02),
                f'{value:+.1%}',
                ha='center',
                va='bottom' if height >= 0 else 'top',
                fontweight='bold', fontsize=16
            )

    plt.suptitle(
        'Effects of Feature Activation on Gambling Behavior (L1–30)',
        fontsize=24, fontweight='bold', y=0.96
    )

    plt.tight_layout()

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_png = OUTPUT_DIR / 'L1_30_causal_patching_average_effects.png'
    output_pdf = OUTPUT_DIR / 'L1_30_causal_patching_average_effects.pdf'

    plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight', facecolor='white')

    print(f"\n✅ Average effects figure saved:")
    print(f"   PNG: {output_png}")
    print(f"   PDF: {output_pdf}")

    plt.close()


def create_layer_distribution_figure(safe_df, risky_df, show_bar_labels=True):
    """
    Figure 2: Layer-wise distribution
    원본: /home/ubuntu/llm_addiction/writing/figures/causal_features_layer_distribution_corrected.png

    Args:
        show_bar_labels: If True, show white numbers inside bars. If False, only show totals above bars.
    """
    # Font settings (match original)
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

    # Extract layer from feature name (NO HARDCODING)
    def get_layer(feature_str):
        return int(feature_str.split('-')[0][1:])

    safe_df['layer'] = safe_df['feature'].apply(get_layer)
    risky_df['layer'] = risky_df['feature'].apply(get_layer)

    # Count by layer
    safe_counts = safe_df['layer'].value_counts().sort_index()
    risky_counts = risky_df['layer'].value_counts().sort_index()

    # Get all layers present in data
    all_layers = sorted(set(safe_counts.index) | set(risky_counts.index))

    # Fill missing layers with 0
    safe_counts = safe_counts.reindex(all_layers, fill_value=0)
    risky_counts = risky_counts.reindex(all_layers, fill_value=0)

    print(f"\nLayer distribution (TRUE 4-way):")
    for layer in all_layers:
        total = safe_counts[layer] + risky_counts[layer]
        print(f"  L{layer}: safe={safe_counts[layer]}, risky={risky_counts[layer]}, total={total}")

    fig, ax = plt.subplots(figsize=(18, 6.5))
    indices = np.arange(len(all_layers))
    bar_width = 0.6

    # Stacked bars (match original)
    safe_bars = ax.bar(
        indices, safe_counts.values, bar_width,
        color='#2ca02c', edgecolor='black', label='Safe Features'
    )

    risky_bars = ax.bar(
        indices, risky_counts.values, bar_width,
        bottom=safe_counts.values,
        color='#d62728', edgecolor='black', label='Risky Features'
    )

    # X-axis labels
    ax.set_xticks(indices)
    ax.set_xticklabels([f'{layer}' for layer in all_layers], rotation=0, fontweight='bold')
    ax.set_ylabel('Feature Count', fontweight='bold', fontsize=28)
    ax.set_title(
        'Layer-wise Distribution of Causal Features (L1–30)',
        fontweight='bold', pad=20, fontsize=34
    )
    ax.grid(axis='y', alpha=0.3)

    # Legend (match original)
    legend = ax.legend(loc='upper left', fontsize=24, frameon=True, fancybox=True, shadow=False)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1)

    # Set y-limit with padding
    total_counts = safe_counts.values + risky_counts.values
    max_total = total_counts.max() if len(total_counts) > 0 else 0
    ax.set_ylim(0, max_total * 1.15)

    # Add value labels
    upper_padding = 5
    for idx, (safe_val, risky_val) in enumerate(zip(safe_counts.values, risky_counts.values)):
        total = safe_val + risky_val

        # Safe count (inside green bar) - optional
        if show_bar_labels and safe_val > 0:
            ax.text(
                indices[idx], safe_val / 2, str(int(safe_val)),
                ha='center', va='center', fontsize=16,
                color='white', fontweight='bold'
            )

        # Risky count (inside red bar) - optional
        if show_bar_labels and risky_val > 0:
            ax.text(
                indices[idx], safe_val + risky_val / 2, str(int(risky_val)),
                ha='center', va='center', fontsize=16,
                color='white', fontweight='bold'
            )

        # Total count (above bar) - always show
        if total > 0:
            ax.text(
                indices[idx], safe_val + risky_val + upper_padding * 0.2,
                str(int(total)),
                ha='center', va='bottom', fontsize=18, fontweight='bold'
            )

    plt.tight_layout()

    # Save
    suffix = '' if show_bar_labels else '_no_bar_labels'
    output_png = OUTPUT_DIR / f'L1_30_causal_features_layer_distribution{suffix}.png'
    output_pdf = OUTPUT_DIR / f'L1_30_causal_features_layer_distribution{suffix}.pdf'

    plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight', facecolor='white')

    print(f"\n✅ Layer distribution figure saved:")
    print(f"   PNG: {output_png}")
    print(f"   PDF: {output_pdf}")

    plt.close()


def main():
    print("=" * 80)
    print("L1-31 Experiment Figure Generation (CORRECT Consistent Features)")
    print("NO HARDCODING - Using actual experimental data only")
    print("=" * 80)

    # Load data
    safe_df, risky_df = load_true_4way_data()

    # Create figures
    print("\n" + "=" * 80)
    print("Creating Figure 1: Average Effects")
    print("=" * 80)
    create_average_effects_figure(safe_df, risky_df)

    print("\n" + "=" * 80)
    print("Creating Figure 2: Layer Distribution (with bar labels)")
    print("=" * 80)
    create_layer_distribution_figure(safe_df, risky_df, show_bar_labels=True)

    print("\n" + "=" * 80)
    print("Creating Figure 2: Layer Distribution (without bar labels)")
    print("=" * 80)
    create_layer_distribution_figure(safe_df, risky_df, show_bar_labels=False)

    print("\n" + "=" * 80)
    print("✅ All figures generated successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
