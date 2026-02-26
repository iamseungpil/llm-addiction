#!/usr/bin/env python3
"""
Two-Way ANOVA Results: Figure 1 Style Heatmap
Creates publication-quality heatmaps similar to the original Figure 1
but using Two-Way ANOVA results with flexible ranking options.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
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


def load_two_way_anova_results(model_type='llama'):
    """Load Two-Way ANOVA results"""
    results_dir = Path('/mnt/c/Users/oollccddss/git/llm-addiction/additional_experiments/sae_condition_comparison/results')

    # Find the most recent two-way ANOVA result file
    anova_files = sorted(results_dir.glob(f'two_way_anova_{model_type}_*.json'))

    if not anova_files:
        raise FileNotFoundError(f"No Two-Way ANOVA results found for {model_type}")

    with open(anova_files[-1]) as f:
        return json.load(f)


def create_heatmap_by_ranking(data, ranking_mode='total', n_features=20,
                              model_name='LLaMA-3.1-8B', model_type='llama'):
    """
    Create Figure 1 style heatmap from Two-Way ANOVA results.

    Args:
        data: Two-Way ANOVA results dictionary
        ranking_mode: How to rank features
            - 'total': Sum of all eta-squared (bet_type + outcome + interaction)
            - 'bet_type': Bet Type effect eta-squared
            - 'outcome': Outcome effect eta-squared
            - 'interaction': Interaction effect eta-squared
        n_features: Number of top features to display
        model_name: Model name for title
        model_type: Model type (llama/gemma)
    """
    all_results = data['all_results']

    # Calculate ranking criterion
    for feat in all_results:
        bet_eta = feat['bet_type_effect']['eta_squared']
        out_eta = feat['outcome_effect']['eta_squared']
        int_eta = feat['interaction_effect']['eta_squared']

        if ranking_mode == 'total':
            feat['ranking_score'] = bet_eta + out_eta + int_eta
        elif ranking_mode == 'bet_type':
            feat['ranking_score'] = bet_eta
        elif ranking_mode == 'outcome':
            feat['ranking_score'] = out_eta
        elif ranking_mode == 'interaction':
            feat['ranking_score'] = int_eta
        else:
            raise ValueError(f"Unknown ranking_mode: {ranking_mode}")

    # Sort by ranking score
    sorted_features = sorted(all_results, key=lambda x: -x['ranking_score'])
    top_features = sorted_features[:n_features]

    # Prepare data matrix
    feature_names = []
    activation_matrix = []
    effect_info = []

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

        # Store effect sizes for annotation
        bet_eta = feat['bet_type_effect']['eta_squared']
        out_eta = feat['outcome_effect']['eta_squared']
        int_eta = feat['interaction_effect']['eta_squared']
        effect_info.append({
            'bet': bet_eta,
            'out': out_eta,
            'int': int_eta,
            'total': bet_eta + out_eta + int_eta
        })

    activation_matrix = np.array(activation_matrix)

    # Z-score normalization (row-wise)
    normalized = (activation_matrix - activation_matrix.mean(axis=1, keepdims=True)) / \
                 (activation_matrix.std(axis=1, keepdims=True) + 1e-8)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 12))

    # Heatmap with diverging colormap
    im = ax.imshow(normalized, aspect='auto', cmap='RdBu_r', vmin=-2.5, vmax=2.5)

    # Set ticks and labels
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(['Bankrupt', 'Safe', 'Bankrupt', 'Safe'],
                      fontsize=11, fontweight='bold')
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=9)

    # Add effect size annotations on the right
    for i, (name, eff) in enumerate(zip(feature_names, effect_info)):
        if ranking_mode == 'total':
            text = f"B:{eff['bet']:.3f} O:{eff['out']:.3f} I:{eff['int']:.3f}"
        elif ranking_mode == 'bet_type':
            text = f"η²={eff['bet']:.3f}"
        elif ranking_mode == 'outcome':
            text = f"η²={eff['out']:.3f}"
        elif ranking_mode == 'interaction':
            text = f"η²={eff['int']:.3f}"

        ax.text(4.2, i, text, va='center', fontsize=7,
               color='gray', style='italic')

    # Labels and title
    ax.set_xlabel('Condition', fontsize=13, fontweight='bold')
    ax.set_ylabel(f'SAE Feature (ranked by {ranking_mode.replace("_", " ").title()} η²)',
                 fontsize=13, fontweight='bold')

    ranking_title = {
        'total': 'Total Effect',
        'bet_type': 'Bet Type Effect',
        'outcome': 'Outcome Effect',
        'interaction': 'Interaction Effect'
    }[ranking_mode]

    ax.set_title(f'Two-Way ANOVA: {ranking_title} ({model_name})',
                fontsize=14, fontweight='bold', pad=35)

    # Add Variable/Fixed labels
    ax.text(0.5, -0.9, 'Variable Betting', ha='center', fontsize=12,
           fontweight='bold', color='darkblue', va='center')
    ax.text(2.5, -0.9, 'Fixed Betting', ha='center', fontsize=12,
           fontweight='bold', color='darkred', va='center')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.15)
    cbar.set_label('Normalized Activation\n(Z-score within feature)',
                  fontsize=11, fontweight='bold')
    cbar.ax.axhline(y=0, color='black', linewidth=1, linestyle='-', alpha=0.3)

    # Grid lines
    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(feature_names), 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1.5)

    # Vertical separator
    ax.axvline(x=1.5, color='black', linewidth=3, linestyle='--', alpha=0.8)

    plt.tight_layout()

    # Save
    save_file = f'/mnt/c/Users/oollccddss/git/llm-addiction/additional_experiments/sae_condition_comparison/results/two_way_anova_heatmap_{ranking_mode}_{model_type}.png'
    plt.savefig(save_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_file}")

    return fig


def create_all_heatmaps(model_type='llama'):
    """Create heatmaps for all ranking modes"""
    print(f"\n{'='*70}")
    print(f"Creating Two-Way ANOVA Heatmaps for {model_type.upper()}")
    print(f"{'='*70}\n")

    # Load data
    print(f"Loading Two-Way ANOVA results for {model_type}...")
    data = load_two_way_anova_results(model_type)

    model_name = 'LLaMA-3.1-8B' if model_type == 'llama' else 'Gemma-2-9B-IT'

    ranking_modes = ['total', 'bet_type', 'outcome', 'interaction']

    for mode in ranking_modes:
        print(f"\n{'='*70}")
        print(f"Creating heatmap ranked by: {mode.replace('_', ' ').title()}")
        print(f"{'='*70}")
        create_heatmap_by_ranking(data, ranking_mode=mode,
                                 n_features=20,
                                 model_name=model_name,
                                 model_type=model_type)

    print(f"\n{'='*70}")
    print(f"All heatmaps created successfully for {model_name}!")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Create Figure 1 style heatmaps from Two-Way ANOVA results")
    parser.add_argument('--model', type=str, default='llama', choices=['llama', 'gemma'],
                       help='Model type')
    parser.add_argument('--ranking', type=str, default='all',
                       choices=['all', 'total', 'bet_type', 'outcome', 'interaction'],
                       help='Ranking mode (default: all)')
    parser.add_argument('--n-features', type=int, default=20,
                       help='Number of features to display (default: 20)')

    args = parser.parse_args()

    if args.ranking == 'all':
        create_all_heatmaps(args.model)
    else:
        data = load_two_way_anova_results(args.model)
        model_name = 'LLaMA-3.1-8B' if args.model == 'llama' else 'Gemma-2-9B-IT'
        create_heatmap_by_ranking(data, ranking_mode=args.ranking,
                                 n_features=args.n_features,
                                 model_name=model_name,
                                 model_type=args.model)


if __name__ == '__main__':
    main()
