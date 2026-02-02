#!/usr/bin/env python3
"""
Improved SAE Condition Comparison Visualization
Publication-quality figures with enhanced clarity
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import seaborn as sns

# Set publication-ready style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def load_results(model_type='llama'):
    """Load the analysis results"""
    results_dir = Path('/mnt/c/Users/oollccddss/git/llm-addiction/additional_experiments/sae_condition_comparison/results')

    # Try summary file first
    summary_files = list(results_dir.glob(f'condition_comparison_summary_{model_type}_*.json'))
    if summary_files:
        with open(summary_files[0]) as f:
            return json.load(f)

    # For Gemma, load individual files and construct data structure
    vs_files = sorted(results_dir.glob(f'variable_vs_fixed_{model_type}_*.json'))
    fw_files = sorted(results_dir.glob(f'four_way_{model_type}_*.json'))

    if not vs_files or not fw_files:
        raise FileNotFoundError(f"No result files found for {model_type}")

    # Load individual results
    with open(vs_files[-1]) as f:
        vs_data = json.load(f)
    with open(fw_files[-1]) as f:
        fw_data = json.load(f)

    # Extract top features from all_results
    all_vs = vs_data.get('all_results', [])
    all_fw = fw_data.get('all_results', [])

    def safe_float(x):
        return float(x) if isinstance(x, str) else x

    # Sort and get top features
    significant_vs = [r for r in all_vs if r.get('fdr_significant', False)]
    higher_var = sorted([r for r in significant_vs if safe_float(r['cohens_d']) > 0], key=lambda x: -safe_float(x['cohens_d']))
    higher_fix = sorted([r for r in significant_vs if safe_float(r['cohens_d']) < 0], key=lambda x: safe_float(x['cohens_d']))

    sorted_fw = sorted(all_fw, key=lambda x: -safe_float(x.get('eta_squared', 0)))

    # Construct the expected data structure
    return {
        'model_type': model_type,
        'data_summary': {
            'total_games': 3200,
            'variable': 1600,
            'fixed': 1600,
            'variable_bankrupt': 465 if model_type == 'gemma' else 108,
            'variable_safe': 1135 if model_type == 'gemma' else 1492,
            'fixed_bankrupt': 205 if model_type == 'gemma' else 42,
            'fixed_safe': 1395 if model_type == 'gemma' else 1558,
            'variable_bankruptcy_rate': 0.290625 if model_type == 'gemma' else 0.0675,
            'fixed_bankruptcy_rate': 0.128125 if model_type == 'gemma' else 0.02625
        },
        'variable_vs_fixed': {
            'summary': vs_data['summary'],
            'top_higher_in_variable': higher_var[:50],
            'top_higher_in_fixed': higher_fix[:50]
        },
        'four_way': {
            'summary': fw_data['summary'],
            'top_features': sorted_fw[:50]
        }
    }


def create_improved_heatmap(data, n_features=20, model_name='LLaMA-3.1-8B', model_type='llama'):
    """
    Improved 4-way heatmap with better visual clarity
    Focus on Z-score normalized view only for clarity
    """
    # Get top features from four-way analysis
    top_features = data['four_way']['top_features'][:n_features]

    # Prepare data matrix
    feature_names = []
    activation_matrix = []
    eta_squared_values = []

    for feat in top_features:
        layer = feat['layer']
        fid = feat['feature_id']
        feature_names.append(f"L{layer}-{fid}")

        means = feat['group_means']
        row = [
            means['variable_bankrupt'],
            means['variable_safe'],
            means['fixed_bankrupt'],
            means['fixed_safe']
        ]
        activation_matrix.append(row)

        # Convert eta_squared to float if it's a string
        eta = float(feat['eta_squared']) if isinstance(feat['eta_squared'], str) else feat['eta_squared']
        eta_squared_values.append(eta)

    activation_matrix = np.array(activation_matrix)

    # Normalize each row for better visualization (z-score)
    normalized = (activation_matrix - activation_matrix.mean(axis=1, keepdims=True)) / \
                 (activation_matrix.std(axis=1, keepdims=True) + 1e-8)

    # Create figure with single panel (Z-score only)
    fig, ax = plt.subplots(figsize=(8, 10))

    # Use diverging colormap with clear center
    im = ax.imshow(normalized, aspect='auto', cmap='RdBu_r', vmin=-2.5, vmax=2.5)

    # Set ticks and labels
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(['Bankrupt', 'Safe', 'Bankrupt', 'Safe'],
                       fontsize=11, fontweight='bold')
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=9)

    # Add effect size (η²) next to feature names
    for i, (name, eta) in enumerate(zip(feature_names, eta_squared_values)):
        ax.text(3.6, i, f'η²={eta:.3f}', va='center', fontsize=8,
                color='gray', style='italic')

    ax.set_xlabel('Condition', fontsize=13, fontweight='bold')
    ax.set_ylabel('SAE Feature (ranked by η²)', fontsize=13, fontweight='bold')
    ax.set_title(f'SAE Feature Activation Patterns ({model_name})',
                 fontsize=14, fontweight='bold', pad=35)

    # Add Variable/Fixed labels below title, above heatmap
    # Using negative y values to position above the heatmap (row 0)
    # More negative = higher up (closer to title), less negative = lower (closer to heatmap)
    ax.text(0.5, -0.9, 'Variable Betting', ha='center', fontsize=12,
            fontweight='bold', color='darkblue', va='center')
    ax.text(2.5, -0.9, 'Fixed Betting', ha='center', fontsize=12,
            fontweight='bold', color='darkred', va='center')

    # Add colorbar with clear labeling
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.12)
    cbar.set_label('Normalized Activation\n(Z-score within feature)',
                   fontsize=11, fontweight='bold')
    cbar.ax.axhline(y=0, color='black', linewidth=1, linestyle='-', alpha=0.3)

    # Add grid lines for clarity
    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(feature_names), 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1.5)

    # Add STRONG vertical line separating Variable and Fixed
    ax.axvline(x=1.5, color='black', linewidth=3, linestyle='--', alpha=0.8)


    plt.tight_layout()

    save_file = f'/mnt/c/Users/oollccddss/git/llm-addiction/additional_experiments/sae_condition_comparison/results/fig1_improved_heatmap_{model_type}.png'
    plt.savefig(save_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: fig1_improved_heatmap_{model_type}.png")

    return fig


def create_improved_scatter(data, model_name='LLaMA-3.1-8B', model_type='llama'):
    """
    Improved scatter plot with normalized axes and clearer interpretation zones
    """
    top_features = data['four_way']['top_features'][:50]

    bet_effects = []
    outcome_effects = []
    eta_squared = []
    labels = []

    for feat in top_features:
        means = feat['group_means']
        vb = means['variable_bankrupt']
        vs = means['variable_safe']
        fb = means['fixed_bankrupt']
        fs = means['fixed_safe']

        # Bet type effect: |mean(Variable) - mean(Fixed)|
        bet_eff = abs((vb + vs)/2 - (fb + fs)/2)
        # Outcome effect: |mean(Bankrupt) - mean(Safe)|
        out_eff = abs((vb + fb)/2 - (vs + fs)/2)

        bet_effects.append(bet_eff)
        outcome_effects.append(out_eff)

        eta = float(feat['eta_squared']) if isinstance(feat['eta_squared'], str) else feat['eta_squared']
        eta_squared.append(eta)
        labels.append(f"L{feat['layer']}-{feat['feature_id']}")

    bet_effects = np.array(bet_effects)
    outcome_effects = np.array(outcome_effects)
    eta_squared = np.array(eta_squared)

    # Calculate ratio for coloring
    ratios = bet_effects / (outcome_effects + 1e-8)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 9))

    # Add shaded regions FIRST (background)
    max_val = max(max(bet_effects), max(outcome_effects)) * 1.05

    # Bet Type dominant region (above diagonal)
    ax.fill_between([0, max_val], [0, max_val], [max_val, max_val],
                    alpha=0.15, color='purple', label='Bet Type > Outcome', zorder=1)

    # Outcome dominant region (below diagonal)
    ax.fill_between([0, max_val], [0, 0], [0, max_val],
                    alpha=0.15, color='orange', label='Outcome > Bet Type', zorder=1)

    # Add diagonal line (equal effect)
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, linewidth=2,
            label='Equal Effect Line', zorder=2)

    # Scatter plot with size based on eta-squared
    sizes = 30 + 150 * (eta_squared / eta_squared.max())

    scatter = ax.scatter(outcome_effects, bet_effects,
                        c=eta_squared, cmap='viridis',
                        s=sizes, alpha=0.7,
                        edgecolors='black', linewidth=1.0, zorder=3)

    # Add labels for top 10 features
    for i in range(min(10, len(labels))):
        # Offset calculation for better readability
        offset_x = 5 if outcome_effects[i] < max_val * 0.7 else -35
        offset_y = 5 if bet_effects[i] < max_val * 0.7 else -10

        ax.annotate(labels[i], (outcome_effects[i], bet_effects[i]),
                   xytext=(offset_x, offset_y), textcoords='offset points',
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='gray', alpha=0.8))

    # Labels and title
    ax.set_xlabel('Outcome Effect: |Mean(Bankrupt) - Mean(Safe)|',
                  fontsize=13, fontweight='bold')
    ax.set_ylabel('Bet Type Effect: |Mean(Variable) - Mean(Fixed)|',
                  fontsize=13, fontweight='bold')
    ax.set_title(f'Feature Encoding Strategy ({model_name})',
                 fontsize=14, fontweight='bold', pad=15)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Effect Size (η²)', fontsize=12, fontweight='bold')

    # Legend
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

    # Set limits with padding
    ax.set_xlim(-max_val*0.02, max_val)
    ax.set_ylim(-max_val*0.02, max_val)

    # Add grid for easier reading
    ax.grid(True, alpha=0.2, linestyle=':', zorder=0)

    plt.tight_layout()

    save_file = f'/mnt/c/Users/oollccddss/git/llm-addiction/additional_experiments/sae_condition_comparison/results/fig3_improved_scatter_{model_type}.png'
    plt.savefig(save_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: fig3_improved_scatter_{model_type}.png")

    return fig


def main(model_type='llama'):
    """Generate improved figures for specified model"""
    print(f"\n{'='*60}")
    print(f"Generating IMPROVED visualizations for {model_type.upper()}")
    print(f"{'='*60}\n")

    print(f"Loading results for {model_type}...")
    data = load_results(model_type)

    model_name = 'LLaMA-3.1-8B' if model_type == 'llama' else 'Gemma-2-9B-IT'

    print("\n" + "="*60)
    print(f"Figure 1: Improved Four-Way Heatmap ({model_name})")
    print("="*60)
    create_improved_heatmap(data, model_name=model_name, model_type=model_type)

    print("\n" + "="*60)
    print(f"Figure 3: Improved Bet Type vs Outcome Scatter ({model_name})")
    print("="*60)
    create_improved_scatter(data, model_name=model_name, model_type=model_type)

    print("\n" + "="*60)
    print(f"All improved figures generated successfully for {model_name}!")
    print("="*60)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        model_type = sys.argv[1]
    else:
        model_type = 'llama'
    main(model_type)
