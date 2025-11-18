#!/usr/bin/env python3
"""
Image 4: Layer-wise Feature Evolution (Experiment 1)

Visualizes how feature activations evolve across layers 1-31.
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

    if not file_path.exists():
        print(f"⚠️  File not found: {file_path}")
        print("    Trying alternative file...")
        file_path = Path("/data/llm_addiction/experiment_1_L1_31_extraction/L1_31_features_FINAL_20250930_204420.json")

    with open(file_path, 'r') as f:
        data = json.load(f)

    print(f"Loaded data with {len(data)} entries")

    # Parse layer-wise activations
    layer_activations = defaultdict(list)

    for entry in data:
        if 'layer_features' in entry:
            for layer_name, features in entry['layer_features'].items():
                layer_num = int(layer_name.replace('layer_', ''))

                # Extract feature activations
                for feature_data in features:
                    if isinstance(feature_data, dict) and 'activation' in feature_data:
                        activation = feature_data['activation']
                        layer_activations[layer_num].append(activation)

    print(f"Extracted activations for {len(layer_activations)} layers")

    return layer_activations

def create_visualization(layer_activations):
    """Create layer-wise evolution visualization"""

    # Create figure with 3 subplots
    fig = plt.figure(figsize=(20, 12))

    # Subplot 1: Box plot of layer activations
    ax1 = plt.subplot(2, 2, 1)

    layers = sorted(layer_activations.keys())
    box_data = [layer_activations[layer] for layer in layers]

    bp = ax1.boxplot(box_data, positions=layers, widths=0.6,
                     patch_artist=True, showfliers=False)

    # Color boxes by layer depth
    colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax1.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Feature Activation', fontsize=12, fontweight='bold')
    ax1.set_title('Feature Activation Distribution Across Layers', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Subplot 2: Mean and std evolution
    ax2 = plt.subplot(2, 2, 2)

    means = [np.mean(layer_activations[layer]) for layer in layers]
    stds = [np.std(layer_activations[layer]) for layer in layers]

    ax2.plot(layers, means, 'o-', linewidth=2, markersize=6, label='Mean Activation', color='#1f77b4')
    ax2.fill_between(layers,
                     [m - s for m, s in zip(means, stds)],
                     [m + s for m, s in zip(means, stds)],
                     alpha=0.3, color='#1f77b4', label='±1 Std Dev')

    ax2.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean Feature Activation', fontsize=12, fontweight='bold')
    ax2.set_title('Mean Activation Evolution', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Activation percentiles
    ax3 = plt.subplot(2, 2, 3)

    percentiles = [10, 25, 50, 75, 90]
    percentile_data = defaultdict(list)

    for layer in layers:
        acts = layer_activations[layer]
        for p in percentiles:
            percentile_data[p].append(np.percentile(acts, p))

    colors_p = ['#d32f2f', '#f57c00', '#fbc02d', '#7cb342', '#1976d2']
    for p, color in zip(percentiles, colors_p):
        ax3.plot(layers, percentile_data[p], 'o-', linewidth=2, markersize=4,
                label=f'{p}th Percentile', color=color, alpha=0.7)

    ax3.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Feature Activation', fontsize=12, fontweight='bold')
    ax3.set_title('Activation Percentiles Across Layers', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=9, loc='best')
    ax3.grid(True, alpha=0.3)

    # Subplot 4: Layer statistics table
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')

    # Calculate statistics
    table_data = []
    table_data.append(['Layer', 'Mean', 'Std', 'Min', 'Max', 'N'])

    # Show every 3rd layer to fit in table
    for layer in layers[::3]:
        acts = layer_activations[layer]
        table_data.append([
            f"L{layer}",
            f"{np.mean(acts):.3f}",
            f"{np.std(acts):.3f}",
            f"{np.min(acts):.3f}",
            f"{np.max(acts):.3f}",
            f"{len(acts):,}"
        ])

    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.12, 0.15, 0.15, 0.15, 0.15, 0.18])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header row
    for i in range(6):
        table[(0, i)].set_facecolor('#e0e0e0')
        table[(0, i)].set_text_props(weight='bold')

    ax4.set_title('Layer Statistics Summary (Every 3rd Layer)', fontsize=14, fontweight='bold', pad=20)

    # Overall title
    total_activations = sum(len(acts) for acts in layer_activations.values())
    fig.suptitle(f'Experiment 1: Layer-wise Feature Evolution Analysis\n' +
                f'Total Activations: {total_activations:,} across {len(layers)} layers',
                fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig

def main():
    print("="*80)
    print("Image 4: Layer-wise Feature Evolution")
    print("="*80)
    print()

    # Load data
    layer_activations = load_experiment1_data()

    if not layer_activations:
        print("❌ No data found!")
        return

    # Print summary statistics
    print("\nSummary Statistics:")
    layers = sorted(layer_activations.keys())
    print(f"  Layers covered: {layers[0]} to {layers[-1]}")
    print(f"  Total layers: {len(layers)}")

    total_activations = sum(len(acts) for acts in layer_activations.values())
    print(f"  Total activations: {total_activations:,}")

    print("\n  Layer-wise activation counts:")
    for layer in layers[:10]:
        print(f"    Layer {layer:2d}: {len(layer_activations[layer]):,} activations")
    print("    ...")

    # Create visualization
    print("\nCreating visualization...")
    fig = create_visualization(layer_activations)

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
