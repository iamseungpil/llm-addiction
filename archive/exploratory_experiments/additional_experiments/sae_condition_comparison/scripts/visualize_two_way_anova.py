#!/usr/bin/env python3
"""
Visualization for Two-Way ANOVA Results

Creates comprehensive figures showing:
1. Effect size comparison (Bet Type, Outcome, Interaction)
2. Main effects scatter plot
3. Interaction patterns
4. Layer-wise effect distribution
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

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
    """Load Two-Way ANOVA results"""
    results_dir = Path('/mnt/c/Users/oollccddss/git/llm-addiction/additional_experiments/sae_condition_comparison/results')

    # Find latest results file
    result_files = sorted(results_dir.glob(f'two_way_anova_{model_type}_*.json'))
    if not result_files:
        raise FileNotFoundError(f"No Two-Way ANOVA results found for {model_type}")

    with open(result_files[-1]) as f:
        return json.load(f)


def create_effect_size_comparison(data, model_name='LLaMA-3.1-8B', model_type='llama'):
    """
    Figure 1: Effect Size Comparison
    Bar plot comparing η² for Bet Type, Outcome, and Interaction
    """
    results = data['all_results']

    # Extract eta-squared values
    eta_bet = [r['bet_type_effect']['eta_squared'] for r in results]
    eta_outcome = [r['outcome_effect']['eta_squared'] for r in results]
    eta_interaction = [r['interaction_effect']['eta_squared'] for r in results]

    # Top 100 features by total effect
    total_eta = [eta_bet[i] + eta_outcome[i] + eta_interaction[i] for i in range(len(results))]
    top_indices = np.argsort(total_eta)[-100:]

    eta_bet_top = [eta_bet[i] for i in top_indices]
    eta_outcome_top = [eta_outcome[i] for i in top_indices]
    eta_interaction_top = [eta_interaction[i] for i in top_indices]

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Distribution (all features)
    ax = axes[0]
    positions = [1, 2, 3]
    bp = ax.boxplot(
        [eta_bet, eta_outcome, eta_interaction],
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=False
    )

    colors = ['#3498db', '#e74c3c', '#2ecc71']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add violin plot overlay
    parts = ax.violinplot(
        [eta_bet, eta_outcome, eta_interaction],
        positions=positions,
        widths=0.8,
        showmeans=False,
        showextrema=False
    )
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.3)

    ax.set_xticks(positions)
    ax.set_xticklabels(['Bet Type', 'Outcome', 'Interaction'], fontweight='bold')
    ax.set_ylabel('Effect Size (η²)', fontweight='bold')
    ax.set_title(f'Effect Size Distribution - All Features\n({model_name})', fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(max(eta_bet), max(eta_outcome), max(eta_interaction)) * 1.1)

    # Add mean lines
    means = [np.mean(eta_bet), np.mean(eta_outcome), np.mean(eta_interaction)]
    for pos, mean, color in zip(positions, means, colors):
        ax.hlines(mean, pos - 0.4, pos + 0.4, colors=color, linewidth=3, linestyle='--', alpha=0.8)

    # Panel B: Top 100 features
    ax = axes[1]
    x = np.arange(len(top_indices))
    width = 0.8

    # Stacked bar chart
    ax.bar(x, eta_bet_top, width, label='Bet Type', color='#3498db', alpha=0.8)
    ax.bar(x, eta_outcome_top, width, bottom=eta_bet_top, label='Outcome', color='#e74c3c', alpha=0.8)
    bottom = [eta_bet_top[i] + eta_outcome_top[i] for i in range(len(eta_bet_top))]
    ax.bar(x, eta_interaction_top, width, bottom=bottom, label='Interaction', color='#2ecc71', alpha=0.8)

    ax.set_xlabel('Feature Rank (by total η²)', fontweight='bold')
    ax.set_ylabel('Effect Size (η²)', fontweight='bold')
    ax.set_title(f'Effect Decomposition - Top 100 Features\n({model_name})', fontweight='bold', pad=15)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    save_file = f'/mnt/c/Users/oollccddss/git/llm-addiction/additional_experiments/sae_condition_comparison/results/figures/two_way_anova_effect_comparison_{model_type}.png'
    plt.savefig(save_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_file}")

    return fig


def create_main_effects_scatter(data, model_name='LLaMA-3.1-8B', model_type='llama'):
    """
    Figure 2: Main Effects Scatter Plot
    Compare Bet Type η² vs Outcome η² (similar to existing Figure 3, but using proper ANOVA)
    """
    results = data['all_results']

    # Extract values
    eta_bet = np.array([r['bet_type_effect']['eta_squared'] for r in results])
    eta_outcome = np.array([r['outcome_effect']['eta_squared'] for r in results])
    eta_interaction = np.array([r['interaction_effect']['eta_squared'] for r in results])

    # Total effect for coloring
    total_eta = eta_bet + eta_outcome + eta_interaction

    # Top 50 features
    top_indices = np.argsort(total_eta)[-50:]

    fig, ax = plt.subplots(figsize=(10, 9))

    # Add shaded regions
    max_val = max(max(eta_bet), max(eta_outcome)) * 1.05

    # Bet Type dominant region (above diagonal)
    ax.fill_between([0, max_val], [0, max_val], [max_val, max_val],
                    alpha=0.15, color='blue', label='Bet Type > Outcome', zorder=1)

    # Outcome dominant region (below diagonal)
    ax.fill_between([0, max_val], [0, 0], [0, max_val],
                    alpha=0.15, color='red', label='Outcome > Bet Type', zorder=1)

    # Diagonal line
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, linewidth=2,
            label='Equal Effect', zorder=2)

    # Scatter all features (small, gray)
    ax.scatter(eta_outcome, eta_bet, c='gray', s=10, alpha=0.3, zorder=3)

    # Highlight top 50 features
    sizes = 30 + 150 * (total_eta[top_indices] / total_eta.max())
    scatter = ax.scatter(
        eta_outcome[top_indices],
        eta_bet[top_indices],
        c=total_eta[top_indices],
        cmap='viridis',
        s=sizes,
        alpha=0.7,
        edgecolors='black',
        linewidth=1.0,
        zorder=4
    )

    # Annotate top 10
    for i in range(min(10, len(top_indices))):
        idx = top_indices[-(i+1)]
        layer = results[idx]['layer']
        fid = results[idx]['feature_id']
        label = f"L{layer}-{fid}"

        offset_x = 5 if eta_outcome[idx] < max_val * 0.7 else -35
        offset_y = 5 if eta_bet[idx] < max_val * 0.7 else -10

        ax.annotate(label, (eta_outcome[idx], eta_bet[idx]),
                   xytext=(offset_x, offset_y), textcoords='offset points',
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor='gray', alpha=0.8))

    ax.set_xlabel('Outcome Main Effect (η²)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Bet Type Main Effect (η²)', fontsize=13, fontweight='bold')
    ax.set_title(f'Two-Way ANOVA: Main Effects Comparison\n({model_name})',
                 fontsize=14, fontweight='bold', pad=15)

    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Total Effect Size (η²)', fontsize=12, fontweight='bold')

    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.set_xlim(-max_val*0.02, max_val)
    ax.set_ylim(-max_val*0.02, max_val)
    ax.grid(True, alpha=0.2, linestyle=':', zorder=0)

    plt.tight_layout()

    save_file = f'/mnt/c/Users/oollccddss/git/llm-addiction/additional_experiments/sae_condition_comparison/results/figures/two_way_anova_main_effects_scatter_{model_type}.png'
    plt.savefig(save_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_file}")

    return fig


def create_interaction_patterns(data, model_name='LLaMA-3.1-8B', model_type='llama'):
    """
    Figure 3: Interaction Patterns
    Show features with significant interaction effects
    """
    results = data['all_results']

    # Find features with significant interaction (p < 0.05)
    sig_interactions = [r for r in results if r['interaction_effect']['p_value'] < 0.05]

    # Sort by interaction eta-squared
    sig_interactions.sort(key=lambda x: x['interaction_effect']['eta_squared'], reverse=True)

    # Take top 12 for visualization
    top_n = min(12, len(sig_interactions))
    top_interactions = sig_interactions[:top_n]

    if top_n == 0:
        print(f"No significant interactions found for {model_type}")
        return None

    # Create figure with subplots
    n_cols = 4
    n_rows = (top_n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten() if top_n > 1 else [axes]

    for i, feat in enumerate(top_interactions):
        ax = axes[i]

        means = feat['group_means']
        vb = means['variable_bankrupt']
        vs = means['variable_safe']
        fb = means['fixed_bankrupt']
        fs = means['fixed_safe']

        # Plot interaction
        x = [0, 1]
        variable_line = [vb, vs]
        fixed_line = [fb, fs]

        ax.plot(x, variable_line, 'o-', color='#3498db', linewidth=2.5, markersize=10,
                label='Variable', alpha=0.8)
        ax.plot(x, fixed_line, 's-', color='#e74c3c', linewidth=2.5, markersize=10,
                label='Fixed', alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(['Bankrupt', 'Safe'], fontweight='bold')
        ax.set_ylabel('Mean Activation', fontweight='bold')
        ax.set_title(f"L{feat['layer']}-{feat['feature_id']}\n" +
                    f"η²_int={feat['interaction_effect']['eta_squared']:.3f}, " +
                    f"p={feat['interaction_effect']['p_value']:.4f}",
                    fontsize=10, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(alpha=0.3, linestyle='--')

    # Hide unused subplots
    for i in range(top_n, len(axes)):
        axes[i].axis('off')

    fig.suptitle(f'Significant Interaction Effects (Top {top_n})\n{model_name}',
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    save_file = f'/mnt/c/Users/oollccddss/git/llm-addiction/additional_experiments/sae_condition_comparison/results/figures/two_way_anova_interactions_{model_type}.png'
    plt.savefig(save_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_file}")

    return fig


def create_layer_wise_effects(data, model_name='LLaMA-3.1-8B', model_type='llama'):
    """
    Figure 4: Layer-wise Effect Distribution
    Heatmap showing mean effect sizes per layer
    """
    results = data['all_results']

    # Group by layer
    layers = sorted(set([r['layer'] for r in results]))
    layer_stats = {}

    for layer in layers:
        layer_feats = [r for r in results if r['layer'] == layer]

        layer_stats[layer] = {
            'mean_eta_bet': np.mean([f['bet_type_effect']['eta_squared'] for f in layer_feats]),
            'mean_eta_outcome': np.mean([f['outcome_effect']['eta_squared'] for f in layer_feats]),
            'mean_eta_interaction': np.mean([f['interaction_effect']['eta_squared'] for f in layer_feats]),
            'count': len(layer_feats)
        }

    # Create matrix for heatmap
    matrix = []
    for layer in layers:
        stats = layer_stats[layer]
        matrix.append([
            stats['mean_eta_bet'],
            stats['mean_eta_outcome'],
            stats['mean_eta_interaction']
        ])

    matrix = np.array(matrix)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, len(layers) * 0.4 + 2))

    im = ax.imshow(matrix.T, aspect='auto', cmap='YlOrRd', vmin=0)

    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f"L{l}" for l in layers], rotation=0, fontsize=9)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Bet Type', 'Outcome', 'Interaction'], fontweight='bold')

    ax.set_xlabel('Layer', fontweight='bold', fontsize=12)
    ax.set_title(f'Mean Effect Sizes by Layer\n({model_name})',
                 fontweight='bold', fontsize=14, pad=15)

    # Add values as text
    for i in range(len(layers)):
        for j in range(3):
            text = ax.text(i, j, f'{matrix[i, j]:.3f}',
                         ha="center", va="center", color="black" if matrix[i, j] < 0.5 else "white",
                         fontsize=8)

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Mean η²', fontweight='bold', fontsize=11)

    plt.tight_layout()

    save_file = f'/mnt/c/Users/oollccddss/git/llm-addiction/additional_experiments/sae_condition_comparison/results/figures/two_way_anova_layer_effects_{model_type}.png'
    plt.savefig(save_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_file}")

    return fig


def main(model_type='llama'):
    """Generate all Two-Way ANOVA visualizations"""
    print(f"\n{'='*60}")
    print(f"Generating Two-Way ANOVA Visualizations for {model_type.upper()}")
    print(f"{'='*60}\n")

    # Load results
    print(f"Loading results for {model_type}...")
    data = load_results(model_type)

    model_name = 'LLaMA-3.1-8B' if model_type == 'llama' else 'Gemma-2-9B-IT'

    print("\n" + "="*60)
    print("Figure 1: Effect Size Comparison")
    print("="*60)
    create_effect_size_comparison(data, model_name, model_type)

    print("\n" + "="*60)
    print("Figure 2: Main Effects Scatter Plot")
    print("="*60)
    create_main_effects_scatter(data, model_name, model_type)

    print("\n" + "="*60)
    print("Figure 3: Interaction Patterns")
    print("="*60)
    create_interaction_patterns(data, model_name, model_type)

    print("\n" + "="*60)
    print("Figure 4: Layer-wise Effect Distribution")
    print("="*60)
    create_layer_wise_effects(data, model_name, model_type)

    print("\n" + "="*60)
    print(f"All Two-Way ANOVA figures generated for {model_name}!")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize Two-Way ANOVA Results")
    parser.add_argument('--model', type=str, default='llama', choices=['llama', 'gemma'],
                        help='Model type')
    args = parser.parse_args()

    main(args.model)
