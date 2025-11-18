#!/usr/bin/env python3
"""
Image 4: Layer-wise Feature Evolution (FIXED)

Visualizes how feature activations and characteristics evolve across layers 1-31.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from collections import defaultdict

plt.style.use('seaborn-v0_8-darkgrid')

def load_experiment1_data():
    """Load Experiment 1 Layer 1-31 feature extraction data"""
    print("Loading Experiment 1 data...")

    file_path = Path("/data/llm_addiction/experiment_1_L1_31_extraction/L1_31_features_FINAL_20250930_220003.json")

    with open(file_path, 'r') as f:
        data = json.load(f)

    print(f"Total experiments processed: {data['total_experiments_processed']}")
    print(f"Total layers analyzed: {data['total_layers']}")
    print(f"Total significant features: {data['total_significant_features']}")

    return data

def extract_layer_statistics(data):
    """Extract statistics from each layer"""
    print("\nExtracting layer statistics...")

    layer_stats = {}

    for layer_num_str, layer_data in data['layer_results'].items():
        layer_num = int(layer_num_str)

        layer_stats[layer_num] = {
            'n_features': layer_data['n_features'],
            'n_bankrupt': layer_data['n_bankrupt'],
            'n_safe': layer_data['n_safe'],
            'n_significant': layer_data['n_significant']
        }

        # Extract Cohen's d values from significant features
        cohens_d_values = []
        if 'significant_features' in layer_data and isinstance(layer_data['significant_features'], list):
            for feat in layer_data['significant_features']:
                if isinstance(feat, dict) and 'cohen_d' in feat:
                    cohens_d_values.append(feat['cohen_d'])

        layer_stats[layer_num]['cohens_d_values'] = cohens_d_values
        layer_stats[layer_num]['mean_cohens_d'] = np.mean(cohens_d_values) if cohens_d_values else 0
        layer_stats[layer_num]['std_cohens_d'] = np.std(cohens_d_values) if cohens_d_values else 0

    print(f"Extracted statistics for {len(layer_stats)} layers")

    return layer_stats

def create_visualization(layer_stats):
    """Create layer-wise evolution visualization"""

    # Create figure with 4 subplots
    fig = plt.figure(figsize=(20, 12))

    layers = sorted(layer_stats.keys())

    # Subplot 1: Number of significant features per layer
    ax1 = plt.subplot(2, 2, 1)

    n_significant = [layer_stats[l]['n_significant'] for l in layers]

    bars = ax1.bar(layers, n_significant, color=plt.cm.viridis(np.linspace(0, 1, len(layers))),
                   alpha=0.7, edgecolor='black', linewidth=0.5)

    ax1.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Significant Features', fontsize=12, fontweight='bold')
    ax1.set_title('Significant Features Across Layers', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Subplot 2: Bankrupt vs Safe features
    ax2 = plt.subplot(2, 2, 2)

    n_bankrupt = [layer_stats[l]['n_bankrupt'] for l in layers]
    n_safe = [layer_stats[l]['n_safe'] for l in layers]

    x = np.arange(len(layers))
    width = 0.35

    ax2.bar(x - width/2, n_bankrupt, width, label='Bankrupt-associated',
           color='#e57373', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.bar(x + width/2, n_safe, width, label='Safe-associated',
           color='#64b5f6', alpha=0.7, edgecolor='black', linewidth=0.5)

    ax2.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Features', fontsize=12, fontweight='bold')
    ax2.set_title('Bankrupt vs Safe Features by Layer', fontsize=14, fontweight='bold')
    ax2.set_xticks(x[::3])  # Show every 3rd layer
    ax2.set_xticklabels(layers[::3])
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # Subplot 3: Cohen's d distribution across layers
    ax3 = plt.subplot(2, 2, 3)

    mean_cohens = [layer_stats[l]['mean_cohens_d'] for l in layers]
    std_cohens = [layer_stats[l]['std_cohens_d'] for l in layers]

    ax3.plot(layers, mean_cohens, 'o-', linewidth=2, markersize=6,
            label='Mean Cohen\'s d', color='#9c27b0')
    ax3.fill_between(layers,
                     [m - s for m, s in zip(mean_cohens, std_cohens)],
                     [m + s for m, s in zip(mean_cohens, std_cohens)],
                     alpha=0.3, color='#9c27b0', label='±1 Std Dev')

    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Cohen\'s d', fontsize=12, fontweight='bold')
    ax3.set_title('Effect Size Evolution Across Layers', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Subplot 4: Layer statistics table
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')

    # Create summary table (every 3rd layer)
    table_data = []
    table_data.append(['Layer', 'Total', 'Significant', 'Bankrupt', 'Safe', 'Ratio'])

    for layer in layers[::3]:
        stats = layer_stats[layer]
        ratio = f"{100*stats['n_significant']/stats['n_features']:.1f}%" if stats['n_features'] > 0 else "N/A"

        table_data.append([
            f"L{layer}",
            f"{stats['n_features']:,}",
            f"{stats['n_significant']:,}",
            f"{stats['n_bankrupt']:,}",
            f"{stats['n_safe']:,}",
            ratio
        ])

    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.12, 0.15, 0.18, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header row
    for i in range(6):
        table[(0, i)].set_facecolor('#e0e0e0')
        table[(0, i)].set_text_props(weight='bold')

    ax4.set_title('Layer Statistics Summary (Every 3rd Layer)', fontsize=14, fontweight='bold', pad=20)

    # Overall title
    total_sig = sum(layer_stats[l]['n_significant'] for l in layers)
    total_feat = sum(layer_stats[l]['n_features'] for l in layers)

    fig.suptitle(f'Experiment 1: Layer-wise Feature Evolution Analysis\n' +
                f'Total: {total_sig:,} significant features from {total_feat:,} features across {len(layers)} layers',
                fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig

def main():
    print("="*80)
    print("Image 4: Layer-wise Feature Evolution (FIXED)")
    print("="*80)
    print()

    # Load data
    data = load_experiment1_data()

    # Extract layer statistics
    layer_stats = extract_layer_statistics(data)

    # Print summary
    print("\nLayer Summary:")
    layers = sorted(layer_stats.keys())
    for layer in layers[:10]:
        stats = layer_stats[layer]
        print(f"  Layer {layer:2d}: {stats['n_significant']:4,} significant / {stats['n_features']:6,} total ({100*stats['n_significant']/stats['n_features']:.1f}%)")
    print("  ...")

    # Create visualization
    print("\nCreating visualization...")
    fig = create_visualization(layer_stats)

    # Save
    output_dir = Path("/home/ubuntu/llm_addiction/experiment_pathway_token_analysis/analysis/images")

    png_path = output_dir / "04_layer_evolution.png"
    pdf_path = output_dir / "04_layer_evolution.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')

    print(f"\n✅ Saved visualization:")
    print(f"   PNG: {png_path}")
    print(f"   PDF: {pdf_path}")
    print()

if __name__ == '__main__':
    main()
