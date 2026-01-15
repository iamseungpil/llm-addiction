#!/usr/bin/env python3
"""
Generate figures for rebuttal analysis using REPARSED data
Based on corrected parsing (not balance values, but actual bet decisions)

Key differences from original:
- Original CORRECT_consistent: 640 safe, 2,147 risky (parsing captured balance values)
- REPARSED: 23 safe, 89 risky (correct bet decision parsing)

Figures:
1. Layer-wise distribution of safe/risky features
2. Behavioral effects comparison (safe vs risky features)
3. Feature-feature correlation analysis
4. Word-feature association examples
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from scipy import stats

# Set style - no grid
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.grid'] = False

OUTPUT_DIR = Path("/home/ubuntu/llm_addiction/rebuttal_analysis/figures_REPARSED")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# REPARSED Data paths
REPARSED_DIR = Path("/data/llm_addiction/experiment_2_multilayer_patching/reparsed")
SAFE_JSON = REPARSED_DIR / "safe_features_20251125_043600.json"
RISKY_JSON = REPARSED_DIR / "risky_features_20251125_043600.json"
CLASSIFIED_JSON = REPARSED_DIR / "classified_features_20251125_043600.json"
PHASE1_DIR = Path("/data/llm_addiction/experiment_pathway_token_analysis/results/phase1_patching_REPARSED")


def load_reparsed_data():
    """Load REPARSED safe and risky feature data"""
    with open(SAFE_JSON, 'r') as f:
        safe_data = json.load(f)
    with open(RISKY_JSON, 'r') as f:
        risky_data = json.load(f)
    with open(CLASSIFIED_JSON, 'r') as f:
        classified_data = json.load(f)
    return safe_data, risky_data, classified_data


def figure1_layer_distribution(safe_data, risky_data):
    """Figure 1: Layer-wise distribution of safe/risky features (L1-30)"""
    print("Creating Figure 1: Layer distribution (REPARSED)...")

    # Extract layer counts
    safe_layers = defaultdict(int)
    risky_layers = defaultdict(int)

    for feat in safe_data['features']:
        safe_layers[feat['layer']] += 1

    for feat in risky_data['features']:
        risky_layers[feat['layer']] += 1

    # Create full range L1-30
    layers = list(range(1, 31))
    safe_vals = [safe_layers.get(l, 0) for l in layers]
    risky_vals = [risky_layers.get(l, 0) for l in layers]
    totals = [s + r for s, r in zip(safe_vals, risky_vals)]

    # Create figure - matching reference style
    fig, ax = plt.subplots(figsize=(16, 4.5))

    x = np.arange(len(layers))
    width = 0.75

    n_safe = safe_data['count']
    n_risky = risky_data['count']

    # Stacked bar chart - with black edge, no alpha
    bars1 = ax.bar(x, risky_vals, width, label=f'Risky Features (n={n_risky})',
                   color='#e74c3c', edgecolor='black', linewidth=1)
    bars2 = ax.bar(x, safe_vals, width, bottom=risky_vals, label=f'Safe Features (n={n_safe})',
                   color='#27ae60', edgecolor='black', linewidth=1)

    # Add total counts on top
    for i, total in enumerate(totals):
        if total > 0:
            ax.text(i, total + 0.3, f'{total}', ha='center', va='bottom',
                    fontsize=18, fontweight='bold')

    ax.set_xlabel('Layer', fontsize=24, fontweight='bold')
    ax.set_ylabel('Number of Features', fontsize=24, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{l}' for l in layers], fontsize=18)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_ylim(0, max(totals) * 1.2 if totals else 10)

    # Remove grid and top/right spines
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    legend = ax.legend(loc='upper left', fontsize=18, framealpha=0.95, edgecolor='black')

    # Title
    ax.set_title('Layer-wise Distribution of Causal Features',
                 fontsize=28, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.subplots_adjust(left=0.06, right=0.98)
    plt.savefig(OUTPUT_DIR / "figure1_layer_distribution_REPARSED.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "figure1_layer_distribution_REPARSED.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {OUTPUT_DIR / 'figure1_layer_distribution_REPARSED.pdf'}")

    # Print statistics
    print(f"\n  Layer Distribution Summary (REPARSED):")
    for l in layers:
        if safe_layers[l] > 0 or risky_layers[l] > 0:
            print(f"  - L{l}: Safe={safe_layers[l]}, Risky={risky_layers[l]}")


def figure2_behavioral_effects(safe_data, risky_data):
    """Figure 2: Behavioral effects comparison (3-panel with bankruptcy)"""
    print("\nCreating Figure 2: Behavioral effects (REPARSED with Bankruptcy)...")

    # Load pre-calculated behavioral effects from bankruptcy calculation script
    effects_file = OUTPUT_DIR / "behavioral_effects_REPARSED.json"
    if effects_file.exists():
        with open(effects_file) as f:
            effects_data = json.load(f)

        # Extract values from pre-calculated data
        sf = effects_data['safe_features']
        rf = effects_data['risky_features']

        safe_safe_stop = sf['safe_stop']['mean'] * 100
        safe_risky_stop = sf['risky_stop']['mean'] * 100
        safe_bankruptcy = sf['risky_bankruptcy']['mean'] * 100

        risky_safe_stop = rf['safe_stop']['mean'] * 100
        risky_risky_stop = rf['risky_stop']['mean'] * 100
        risky_bankruptcy = rf['risky_bankruptcy']['mean'] * 100

        # Standard errors
        safe_safe_se = sf['safe_stop']['se'] * 100
        safe_risky_se = sf['risky_stop']['se'] * 100
        safe_bank_se = sf['risky_bankruptcy']['se'] * 100
        risky_safe_se = rf['safe_stop']['se'] * 100
        risky_risky_se = rf['risky_stop']['se'] * 100
        risky_bank_se = rf['risky_bankruptcy']['se'] * 100

        n_safe = sf['n']
        n_risky = rf['n']
    else:
        print("  WARNING: behavioral_effects_REPARSED.json not found, using JSON causality data")
        # Fallback to original method (without bankruptcy)
        safe_effects_list = [feat['causality'] for feat in safe_data['features']]
        risky_effects_list = [feat['causality'] for feat in risky_data['features']]

        safe_safe_stop = np.mean([e['safe_effect_size'] for e in safe_effects_list]) * 100
        safe_risky_stop = np.mean([e['risky_effect_size'] for e in safe_effects_list]) * 100
        safe_bankruptcy = 0

        risky_safe_stop = np.mean([e['safe_effect_size'] for e in risky_effects_list]) * 100
        risky_risky_stop = np.mean([e['risky_effect_size'] for e in risky_effects_list]) * 100
        risky_bankruptcy = 0

        n_safe = safe_data['count']
        n_risky = risky_data['count']
        safe_safe_se = safe_risky_se = safe_bank_se = 0
        risky_safe_se = risky_risky_se = risky_bank_se = 0

    # Create figure - wide horizontal layout with 3 panels
    fig, ax = plt.subplots(figsize=(10, 4.5))

    categories = ['Safe Context\nStop Rate', 'Risky Context\nStop Rate', 'Risky Context\nBankruptcy']
    x = np.arange(len(categories))
    width = 0.35

    safe_vals = [safe_safe_stop, safe_risky_stop, safe_bankruptcy]
    risky_vals = [risky_safe_stop, risky_risky_stop, risky_bankruptcy]
    safe_errs = [safe_safe_se, safe_risky_se, safe_bank_se]
    risky_errs = [risky_safe_se, risky_risky_se, risky_bank_se]

    # Create bars
    bars1 = ax.bar(x - width/2, safe_vals, width, yerr=safe_errs,
                   label=f'Safe Features (n={n_safe})', color='#27ae60',
                   alpha=0.85, capsize=5, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, risky_vals, width, yerr=risky_errs,
                   label=f'Risky Features (n={n_risky})', color='#e74c3c',
                   alpha=0.85, capsize=5, edgecolor='black', linewidth=1)

    # Add value labels
    for bar, val in zip(bars1, safe_vals):
        height = bar.get_height()
        ax.annotate(f'{val:+.1f}%',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3 if height >= 0 else -12),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=10, fontweight='bold')

    for bar, val in zip(bars2, risky_vals):
        height = bar.get_height()
        ax.annotate(f'{val:+.1f}%',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3 if height >= 0 else -12),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=10, fontweight='bold')

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Effect Size (%)', fontsize=14, fontweight='bold')
    ax.set_title('Behavioral Effects of Safe vs Risky Feature Patching', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.legend(loc='upper left', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set symmetric y-limits
    all_vals = safe_vals + risky_vals
    y_max = max(abs(v) for v in all_vals) * 1.3
    ax.set_ylim(-y_max, y_max)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure2_behavioral_effects_REPARSED.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "figure2_behavioral_effects_REPARSED.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {OUTPUT_DIR / 'figure2_behavioral_effects_REPARSED.pdf'}")
    print(f"\n  Effect Summary (REPARSED with Bankruptcy):")
    print(f"  - Safe features (n={n_safe}): Stop {safe_safe_stop:+.1f}%/{safe_risky_stop:+.1f}%, Bankruptcy {safe_bankruptcy:+.1f}%")
    print(f"  - Risky features (n={n_risky}): Stop {risky_safe_stop:+.1f}%/{risky_risky_stop:+.1f}%, Bankruptcy {risky_bankruptcy:+.1f}%")


def figure3_comparison(safe_data, risky_data):
    """Figure 3: Comparison between original and REPARSED"""
    print("\nCreating Figure 3: Original vs REPARSED comparison...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Feature counts
    ax1 = axes[0]
    categories = ['Safe', 'Risky']
    original = [640, 2147]
    reparsed = [safe_data['count'], risky_data['count']]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax1.bar(x - width/2, original, width, label='Original (Balance Parsing)', color='#3498db', alpha=0.85)
    bars2 = ax1.bar(x + width/2, reparsed, width, label='REPARSED (Bet Parsing)', color='#e74c3c', alpha=0.85)

    ax1.set_ylabel('Number of Features')
    ax1.set_title('Feature Count Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.set_yscale('log')

    for bar, val in zip(bars1, original):
        ax1.annotate(f'{val}', xy=(bar.get_x() + bar.get_width()/2, val),
                     ha='center', va='bottom', fontsize=10)
    for bar, val in zip(bars2, reparsed):
        ax1.annotate(f'{val}', xy=(bar.get_x() + bar.get_width()/2, val),
                     ha='center', va='bottom', fontsize=10)

    # Panel 2: Effect sizes
    ax2 = axes[1]
    orig_effects = ['+9.1% / +8.7%', '-41.3% / -41.5%']

    # Calculate REPARSED effects
    safe_effects = [f['causality']['safe_effect_size'] for f in safe_data['features']]
    risky_effects = [f['causality']['risky_effect_size'] for f in risky_data['features']]

    safe_avg = np.mean(safe_effects) * 100 if safe_effects else 0
    risky_avg = np.mean(risky_effects) * 100 if risky_effects else 0

    repar_effects = [f'+{safe_avg:.1f}%', f'{risky_avg:+.1f}%']

    cell_text = [
        ['Original', '+9.1%', '-41.3%'],
        ['REPARSED', f'+{safe_avg:.1f}%', f'{risky_avg:+.1f}%']
    ]

    ax2.axis('off')
    table = ax2.table(
        cellText=cell_text,
        colLabels=['Method', 'Safe Feature Effect', 'Risky Feature Effect'],
        cellLoc='center',
        loc='center',
        colColours=['#f0f0f0']*3
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    ax2.set_title('Effect Size Comparison\n(Stop Rate Change)', pad=20)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure3_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "figure3_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {OUTPUT_DIR / 'figure3_comparison.pdf'}")


def figure4_parsing_error_explanation():
    """Figure 4: Explanation of parsing error"""
    print("\nCreating Figure 4: Parsing error explanation...")

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')

    explanation = """
PARSING ERROR ANALYSIS
======================

Original Response Example:
--------------------------
"1
Bet $10
Round 4: Bet $10, Loss, Balance $130

Choose:
1) Bet $10
2) Stop

Choice: 2
Stop

Final score: $130"

Dollar amounts found: ['10', '10', '130', '10', '130']


PARSING METHODS:
----------------

Original Method (amounts[-1]):
  Returns: $130 (WRONG - this is balance, not bet!)

Correct Method (first bet amount):
  Returns: $10 (CORRECT - actual betting decision)


IMPACT ON RESULTS:
------------------

                    Original Parsing    Correct Parsing
                    (Balance Values)    (Bet Decisions)
---------------------------------------------------------
Safe Features:          640                 23          (-96.4%)
Risky Features:       2,147                 89          (-95.9%)
Feature Overlap:         0%                  0%


CONCLUSION:
-----------
The original parsing captured noise (balance/score values)
rather than actual betting decisions. This led to:
1. Massive overestimation of causal features
2. Inflated effect sizes
3. False positive features that are not actually causal
"""

    ax.text(0.05, 0.95, explanation, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.savefig(OUTPUT_DIR / "figure4_parsing_error.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "figure4_parsing_error.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {OUTPUT_DIR / 'figure4_parsing_error.pdf'}")


def main():
    print("=" * 60)
    print("Generating REPARSED Figures")
    print("=" * 60)

    # Load REPARSED data
    safe_data, risky_data, classified_data = load_reparsed_data()

    print(f"\nLoaded REPARSED data:")
    print(f"  Safe features: {safe_data['count']}")
    print(f"  Risky features: {risky_data['count']}")
    print(f"  Classification breakdown:")
    for k, v in classified_data['classification_counts'].items():
        print(f"    {k}: {v}")

    # Generate figures
    figure1_layer_distribution(safe_data, risky_data)
    figure2_behavioral_effects(safe_data, risky_data)
    figure3_comparison(safe_data, risky_data)
    figure4_parsing_error_explanation()

    print("\n" + "=" * 60)
    print("All REPARSED figures generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
