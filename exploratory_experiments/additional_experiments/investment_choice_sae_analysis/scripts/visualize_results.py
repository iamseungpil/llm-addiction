"""
Visualize Phase 2 correlation analysis results.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def load_results(results_file: Path) -> dict:
    """Load correlation analysis results."""
    with open(results_file, 'r') as f:
        return json.load(f)


def plot_significant_features_by_layer(results: dict, save_path: Path):
    """Plot number of significant features by layer."""
    layers = []
    binary_counts = []
    multiclass_counts = []

    for layer_key, layer_data in sorted(results.items()):
        if not layer_key.startswith('layer_'):
            continue

        layer_num = layer_data['layer']
        layers.append(layer_num)

        # Binary analysis
        if 'binary' in layer_data:
            binary_counts.append(layer_data['binary']['n_significant_features'])
        else:
            binary_counts.append(0)

        # Multi-class analysis
        if 'multiclass' in layer_data:
            multiclass_counts.append(layer_data['multiclass']['n_significant_features'])
        else:
            multiclass_counts.append(0)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(layers))
    width = 0.35

    ax.bar(x - width/2, binary_counts, width, label='Binary (Safe vs Risky)', alpha=0.8)
    ax.bar(x + width/2, multiclass_counts, width, label='Multi-class (4-way)', alpha=0.8)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Number of Significant Features', fontsize=12)
    ax.set_title('Significant SAE Features by Layer', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_top_features_heatmap(results: dict, save_path: Path, analysis_type: str = 'binary'):
    """Plot heatmap of top feature activations."""
    # Select a representative layer (highest number of significant features)
    best_layer = None
    max_features = 0

    for layer_key, layer_data in results.items():
        if not layer_key.startswith('layer_'):
            continue

        if analysis_type in layer_data:
            n_sig = layer_data[analysis_type].get('n_significant_features', 0)
            if n_sig > max_features:
                max_features = n_sig
                best_layer = layer_data

    if best_layer is None:
        print(f"No data found for {analysis_type} analysis")
        return

    layer_num = best_layer['layer']
    analysis_data = best_layer[analysis_type]

    # Extract feature means
    if analysis_type == 'binary':
        feature_means = analysis_data.get('feature_means', {})
        if not feature_means:
            print("No feature means found")
            return

        # Get top 20 features (10 safe + 10 risky)
        safe_feats = analysis_data.get('safe_features', [])[:10]
        risky_feats = analysis_data.get('risky_features', [])[:10]

        all_feats = [(f[0], f[1], 'Safe') for f in safe_feats] + \
                    [(f[0], f[1], 'Risky') for f in risky_feats]

        if not all_feats:
            print("No features to plot")
            return

        # Create heatmap data
        heatmap_data = []
        feature_labels = []

        for feat_id, cohens_d, feat_type in all_feats:
            if str(feat_id) in feature_means:
                means = feature_means[str(feat_id)]
                # Binary labels: 0 = Risky, 1 = Safe
                risky_mean = means.get('0', 0)
                safe_mean = means.get('1', 0)
                heatmap_data.append([risky_mean, safe_mean])
                feature_labels.append(f"{feat_type} F{feat_id}\n(d={cohens_d:.2f})")

        heatmap_data = np.array(heatmap_data)

        # Plot
        fig, ax = plt.subplots(figsize=(6, 12))

        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            xticklabels=['Risky (2/3/4)', 'Safe (1)'],
            yticklabels=feature_labels,
            cbar_kws={'label': 'Mean Activation'},
            ax=ax
        )

        ax.set_title(f'Top SAE Features - Layer {layer_num}\n(Binary: Safe vs Risky)',
                     fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()


def plot_choice_distribution(results: dict, save_path: Path):
    """Plot choice distribution effect sizes across layers."""
    layers = []
    avg_effect_sizes = []

    for layer_key, layer_data in sorted(results.items()):
        if not layer_key.startswith('layer_'):
            continue

        layer_num = layer_data['layer']

        if 'multiclass' in layer_data:
            top_features = layer_data['multiclass'].get('top_features', [])
            if top_features:
                # Average eta-squared of top 10 features
                avg_eta = np.mean([f[1] for f in top_features[:10]])
                layers.append(layer_num)
                avg_effect_sizes.append(avg_eta)

    if not layers:
        print("No multi-class data found")
        return

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(layers, avg_effect_sizes, marker='o', linewidth=2, markersize=8)
    ax.fill_between(layers, 0, avg_effect_sizes, alpha=0.3)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Average Effect Size (η²)', fontsize=12)
    ax.set_title('Choice Prediction Strength Across Layers\n(Top 10 Features)',
                 fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Visualize correlation analysis results')
    parser.add_argument('--results', type=str, required=True,
                        help='Path to correlation_analysis JSON file')
    parser.add_argument('--output_dir', type=str, default='results/visualizations',
                        help='Output directory for plots')

    args = parser.parse_args()

    # Load results
    results_file = Path(args.results)
    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        return

    results = load_results(results_file)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded results from: {results_file}")
    print(f"Output directory: {output_dir}")
    print()

    # Generate plots
    print("Generating plots...")

    # 1. Significant features by layer
    plot_significant_features_by_layer(
        results,
        output_dir / 'significant_features_by_layer.png'
    )

    # 2. Top features heatmap
    plot_top_features_heatmap(
        results,
        output_dir / 'top_features_heatmap_binary.png',
        analysis_type='binary'
    )

    # 3. Choice distribution effect
    plot_choice_distribution(
        results,
        output_dir / 'choice_prediction_by_layer.png'
    )

    print()
    print("Visualization complete!")


if __name__ == '__main__':
    main()
