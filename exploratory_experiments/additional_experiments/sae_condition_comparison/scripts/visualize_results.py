#!/usr/bin/env python3
"""
SAE Condition Comparison Visualization
Generates publication-ready figures for the paper
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Set publication-ready style
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
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


def create_four_way_heatmap(data, n_features=20, save_path=None, model_name='LLaMA-3.1-8B', model_type='llama'):
    """
    Create a 4-way heatmap showing feature activations across conditions

    Rows: Top features (sorted by eta-squared from four-way analysis)
    Columns: 4 conditions (VB, VS, FB, FS)
    """
    # Get top features from four-way analysis
    top_features = data['four_way']['top_features'][:n_features]

    # Prepare data matrix
    feature_names = []
    activation_matrix = []

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

    activation_matrix = np.array(activation_matrix)

    # Normalize each row for better visualization (z-score)
    normalized = (activation_matrix - activation_matrix.mean(axis=1, keepdims=True)) / \
                 (activation_matrix.std(axis=1, keepdims=True) + 1e-8)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 10), gridspec_kw={'width_ratios': [1, 1]})

    # Left: Raw values heatmap
    ax1 = axes[0]
    im1 = ax1.imshow(activation_matrix, aspect='auto', cmap='YlOrRd')
    ax1.set_xticks([0, 1, 2, 3])
    ax1.set_xticklabels(['Variable\nBankrupt', 'Variable\nSafe', 'Fixed\nBankrupt', 'Fixed\nSafe'])
    ax1.set_yticks(range(len(feature_names)))
    ax1.set_yticklabels(feature_names)
    ax1.set_title('(A) Raw Activation Values', fontweight='bold', pad=10)
    ax1.set_xlabel('Condition')
    ax1.set_ylabel('SAE Feature')

    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Mean Activation')

    # Add grid lines
    ax1.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, len(feature_names), 1), minor=True)
    ax1.grid(which='minor', color='white', linestyle='-', linewidth=0.5)

    # Right: Normalized (z-score) heatmap
    ax2 = axes[1]
    im2 = ax2.imshow(normalized, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
    ax2.set_xticks([0, 1, 2, 3])
    ax2.set_xticklabels(['Variable\nBankrupt', 'Variable\nSafe', 'Fixed\nBankrupt', 'Fixed\nSafe'])
    ax2.set_yticks(range(len(feature_names)))
    ax2.set_yticklabels(feature_names)
    ax2.set_title('(B) Normalized (Z-score)', fontweight='bold', pad=10)
    ax2.set_xlabel('Condition')

    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('Z-score')

    ax2.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, len(feature_names), 1), minor=True)
    ax2.grid(which='minor', color='white', linestyle='-', linewidth=0.5)

    # Add vertical line separating Variable and Fixed
    for ax in axes:
        ax.axvline(x=1.5, color='black', linewidth=2, linestyle='--')

    plt.suptitle(f'Four-Way SAE Feature Activation Comparison ({model_name})\n'
                 'Top 20 Features by Effect Size (η²)',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()

    save_file = f'/mnt/c/Users/oollccddss/git/llm-addiction/additional_experiments/sae_condition_comparison/results/fig1_four_way_heatmap_{model_type}.png'
    plt.savefig(save_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: fig1_four_way_heatmap_{model_type}.png")

    return fig


def create_layer_effect_plot(data, save_path=None, model_name='LLaMA-3.1-8B', model_type='llama'):
    """
    Create layer-wise effect size distribution plot
    Shows which layers have the strongest Variable vs Fixed differences
    """
    from collections import defaultdict

    # Collect effect sizes by layer from variable_vs_fixed analysis
    layer_effects_var = defaultdict(list)  # Higher in variable
    layer_effects_fix = defaultdict(list)  # Higher in fixed

    for feat in data['variable_vs_fixed']['top_higher_in_variable']:
        d = float(feat['cohens_d']) if isinstance(feat['cohens_d'], str) else feat['cohens_d']
        layer_effects_var[feat['layer']].append(abs(d))

    for feat in data['variable_vs_fixed']['top_higher_in_fixed']:
        d = float(feat['cohens_d']) if isinstance(feat['cohens_d'], str) else feat['cohens_d']
        layer_effects_fix[feat['layer']].append(abs(d))

    # Get all layers
    all_layers = sorted(set(layer_effects_var.keys()) | set(layer_effects_fix.keys()))

    # Calculate mean effect size per layer
    var_means = [np.mean(layer_effects_var.get(l, [0])) for l in all_layers]
    fix_means = [np.mean(layer_effects_fix.get(l, [0])) for l in all_layers]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(all_layers))
    width = 0.35

    bars1 = ax.bar(x - width/2, var_means, width, label='Higher in Variable', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width/2, fix_means, width, label='Higher in Fixed', color='#3498db', alpha=0.8)

    ax.set_xlabel('Layer', fontsize=14)
    ax.set_ylabel('Mean |Cohen\'s d|', fontsize=14)
    ax.set_title(f'Layer-wise Effect Size Distribution ({model_name})\nTop 50 Features per Direction',
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{l}' for l in all_layers], rotation=45, ha='right')
    ax.legend()

    ax.axhline(y=2.0, color='gray', linestyle='--', alpha=0.5, label='Large effect (d=2.0)')

    plt.tight_layout()

    save_file = f'/mnt/c/Users/oollccddss/git/llm-addiction/additional_experiments/sae_condition_comparison/results/fig2_layer_effect_size_{model_type}.png'
    plt.savefig(save_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: fig2_layer_effect_size_{model_type}.png")

    return fig


def create_bet_vs_outcome_scatter(data, save_path=None, model_name='LLaMA-3.1-8B', model_type='llama'):
    """
    Scatter plot showing Bet Type effect vs Outcome effect for each feature
    Demonstrates that Bet Type dominates
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

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Color by eta-squared
    scatter = ax.scatter(outcome_effects, bet_effects, c=eta_squared,
                        cmap='viridis', s=100, alpha=0.7, edgecolors='black', linewidth=0.5)

    # Add diagonal line (equal effect)
    max_val = max(max(bet_effects), max(outcome_effects))
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Equal effect')

    # Add labels for top 10 features
    for i in range(min(10, len(labels))):
        ax.annotate(labels[i], (outcome_effects[i], bet_effects[i]),
                   xytext=(5, 5), textcoords='offset points', fontsize=9, alpha=0.8)

    ax.set_xlabel('Outcome Effect (|Bankrupt - Safe|)', fontsize=14)
    ax.set_ylabel('Bet Type Effect (|Variable - Fixed|)', fontsize=14)
    ax.set_title(f'Bet Type vs Outcome Effect for SAE Features ({model_name})\n'
                 'Points above diagonal: Bet Type dominates',
                 fontsize=14, fontweight='bold')

    cbar = plt.colorbar(scatter)
    cbar.set_label('η² (Effect Size)', fontsize=12)

    # Add shaded region
    ax.fill_between([0, max_val], [0, max_val], [max_val, max_val],
                    alpha=0.1, color='blue', label='Bet Type > Outcome')

    ax.legend(loc='lower right')
    ax.set_xlim(0, max(outcome_effects) * 1.1)
    ax.set_ylim(0, max(bet_effects) * 1.1)

    plt.tight_layout()

    save_file = f'/mnt/c/Users/oollccddss/git/llm-addiction/additional_experiments/sae_condition_comparison/results/fig3_bet_vs_outcome_scatter_{model_type}.png'
    plt.savefig(save_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: fig3_bet_vs_outcome_scatter_{model_type}.png")

    return fig


def create_top_features_bar(data, n_features=15, save_path=None, model_name='LLaMA-3.1-8B', model_type='llama'):
    """
    Bidirectional bar chart showing top features higher in Variable vs Fixed
    """
    var_features = data['variable_vs_fixed']['top_higher_in_variable'][:n_features]
    fix_features = data['variable_vs_fixed']['top_higher_in_fixed'][:n_features]

    fig, ax = plt.subplots(figsize=(12, 10))

    # Prepare data
    var_labels = [f"L{f['layer']}-{f['feature_id']}" for f in var_features]
    var_d = [float(f['cohens_d']) if isinstance(f['cohens_d'], str) else f['cohens_d'] for f in var_features]

    fix_labels = [f"L{f['layer']}-{f['feature_id']}" for f in fix_features]
    fix_d = [float(f['cohens_d']) if isinstance(f['cohens_d'], str) else f['cohens_d'] for f in fix_features]

    # Combine and sort
    all_labels = var_labels + fix_labels
    all_d = var_d + fix_d

    # Sort by absolute value
    sorted_idx = np.argsort(np.abs(all_d))[::-1]
    all_labels = [all_labels[i] for i in sorted_idx]
    all_d = [all_d[i] for i in sorted_idx]

    # Create colors based on direction
    colors = ['#e74c3c' if d > 0 else '#3498db' for d in all_d]

    y_pos = np.arange(len(all_labels))

    bars = ax.barh(y_pos, all_d, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(all_labels)
    ax.set_xlabel("Cohen's d", fontsize=14)
    ax.set_ylabel('SAE Feature', fontsize=14)
    ax.set_title(f"Top Differential SAE Features ({model_name})\nVariable vs Fixed Betting Conditions",
                 fontsize=14, fontweight='bold')

    # Add vertical line at 0
    ax.axvline(x=0, color='black', linewidth=1)

    # Add legend
    var_patch = mpatches.Patch(color='#e74c3c', label='Higher in Variable')
    fix_patch = mpatches.Patch(color='#3498db', label='Higher in Fixed')
    ax.legend(handles=[var_patch, fix_patch], loc='lower right')

    # Add effect size guidelines
    ax.axvline(x=0.8, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=-0.8, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=2.0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=-2.0, color='gray', linestyle='--', alpha=0.5)

    ax.text(0.85, len(all_labels)-1, 'Large\n(0.8)', fontsize=9, alpha=0.6)
    ax.text(2.05, len(all_labels)-1, 'Very Large\n(2.0)', fontsize=9, alpha=0.6)

    plt.tight_layout()

    save_file = f'/mnt/c/Users/oollccddss/git/llm-addiction/additional_experiments/sae_condition_comparison/results/fig4_top_features_bar_{model_type}.png'
    plt.savefig(save_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: fig4_top_features_bar_{model_type}.png")

    return fig


def main(model_type='llama'):
    """Generate all figures for specified model"""
    print(f"Loading results for {model_type}...")
    data = load_results(model_type)

    model_name = 'LLaMA-3.1-8B' if model_type == 'llama' else 'Gemma-2-9B-IT'
    results_dir = '/mnt/c/Users/oollccddss/git/llm-addiction/additional_experiments/sae_condition_comparison/results'

    print("\n" + "="*60)
    print(f"Generating Figure 1: Four-Way Heatmap ({model_name})")
    print("="*60)
    create_four_way_heatmap(data, model_name=model_name, model_type=model_type)

    print("\n" + "="*60)
    print(f"Generating Figure 2: Layer Effect Size ({model_name})")
    print("="*60)
    create_layer_effect_plot(data, model_name=model_name, model_type=model_type)

    print("\n" + "="*60)
    print(f"Generating Figure 3: Bet Type vs Outcome Scatter ({model_name})")
    print("="*60)
    create_bet_vs_outcome_scatter(data, model_name=model_name, model_type=model_type)

    print("\n" + "="*60)
    print(f"Generating Figure 4: Top Features Bar Chart ({model_name})")
    print("="*60)
    create_top_features_bar(data, model_name=model_name, model_type=model_type)

    print("\n" + "="*60)
    print(f"All figures generated successfully for {model_name}!")
    print("="*60)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        model_type = sys.argv[1]
    else:
        model_type = 'llama'
    main(model_type)
