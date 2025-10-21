#!/usr/bin/env python3
"""
Phase 1 Analysis: Sparsity Evolution Across Layers
Measure how feature sparsity changes from L1 to L31
"""

import json
import numpy as np
import matplotlib.pyplot as plt

def load_pathway_data():
    """Load Experiment 1 pathway data"""
    print("Loading pathway data...")
    with open('/data/llm_addiction/experiment_1_layer_pathway_L1_31/final_pathway_L1_31_20251001_165207.json', 'r') as f:
        data = json.load(f)
    return data['results']

def analyze_sparsity(results):
    """Analyze sparsity evolution across layers"""
    print("\nAnalyzing sparsity evolution...")

    sparsity_stats = {}

    for layer_idx in range(1, 32):
        layer_name = f'L{layer_idx}'

        all_features = []

        for game in results:
            if game['round_data']:
                last_round = game['round_data'][-1]
                if layer_name in last_round['features']:
                    features = np.array(last_round['features'][layer_name])
                    all_features.append(features)

        if len(all_features) > 0:
            all_features = np.array(all_features)

            # Sparsity metrics
            zero_fraction = np.mean(all_features == 0)
            near_zero_fraction = np.mean(np.abs(all_features) < 0.01)

            # Activation statistics
            nonzero_activations = all_features[all_features != 0]
            if len(nonzero_activations) > 0:
                mean_nonzero = np.mean(nonzero_activations)
                max_activation = np.max(all_features)
            else:
                mean_nonzero = 0
                max_activation = 0

            sparsity_stats[layer_name] = {
                'layer': layer_idx,
                'sparsity': float(zero_fraction),
                'near_zero_sparsity': float(near_zero_fraction),
                'mean_activation': float(np.mean(all_features)),
                'mean_nonzero_activation': float(mean_nonzero),
                'max_activation': float(max_activation),
                'n_samples': len(all_features)
            }

            print(f"{layer_name}: Sparsity={zero_fraction*100:.1f}%, "
                  f"Mean={np.mean(all_features):.4f}, Max={max_activation:.2f}")

    return sparsity_stats

def create_sparsity_plots(sparsity_stats):
    """Create sparsity evolution plots"""
    print("\nCreating sparsity evolution plots...")

    layers = sorted(sparsity_stats.keys(), key=lambda x: int(x[1:]))
    layer_nums = [int(l[1:]) for l in layers]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Sparsity
    ax = axes[0, 0]
    sparsity = [sparsity_stats[l]['sparsity'] * 100 for l in layers]
    ax.plot(layer_nums, sparsity, 'o-', linewidth=2, markersize=6, color='#2E86AB')
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Sparsity (%)', fontsize=12)
    ax.set_title('Feature Sparsity Across Layers', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhspan(98, 100, alpha=0.2, color='red', label='Very Sparse (>98%)')
    ax.legend()

    # Plot 2: Mean Activation
    ax = axes[0, 1]
    mean_act = [sparsity_stats[l]['mean_activation'] for l in layers]
    ax.plot(layer_nums, mean_act, 'o-', linewidth=2, markersize=6, color='#A23B72')
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Mean Activation', fontsize=12)
    ax.set_title('Mean Feature Activation Across Layers', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Highlight critical layers
    for layer in [8, 9, 10, 11]:
        ax.axvline(layer, color='orange', linestyle='--', alpha=0.3, linewidth=1)
    ax.text(9.5, max(mean_act)*0.9, 'Critical\nLayers', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))

    # Plot 3: Max Activation
    ax = axes[1, 0]
    max_act = [sparsity_stats[l]['max_activation'] for l in layers]
    ax.plot(layer_nums, max_act, 'o-', linewidth=2, markersize=6, color='#F18F01')
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Max Activation', fontsize=12)
    ax.set_title('Maximum Feature Activation Across Layers', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 4: Mean Non-zero Activation
    ax = axes[1, 1]
    mean_nonzero = [sparsity_stats[l]['mean_nonzero_activation'] for l in layers]
    ax.plot(layer_nums, mean_nonzero, 'o-', linewidth=2, markersize=6, color='#C73E1D')
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Mean Non-zero Activation', fontsize=12)
    ax.set_title('Mean Non-zero Feature Activation Across Layers', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = '/home/ubuntu/llm_addiction/experiment_1_layer_pathway_L1_31/sparsity_evolution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()

def save_results(sparsity_stats):
    """Save sparsity analysis results"""
    output_file = '/home/ubuntu/llm_addiction/experiment_1_layer_pathway_L1_31/sparsity_evolution_results.json'

    with open(output_file, 'w') as f:
        json.dump(sparsity_stats, f, indent=2)

    print(f"Saved results: {output_file}")

def print_summary(sparsity_stats):
    """Print sparsity summary"""
    print("\n" + "=" * 80)
    print("SPARSITY EVOLUTION SUMMARY")
    print("=" * 80)

    # Least sparse layers
    sorted_layers = sorted(sparsity_stats.items(),
                          key=lambda x: x[1]['sparsity'])

    print("\nLEAST SPARSE LAYERS (Most active):")
    for layer, stats in sorted_layers[:5]:
        print(f"{layer}: {stats['sparsity']*100:.2f}% sparse, "
              f"Mean={stats['mean_activation']:.4f}, Max={stats['max_activation']:.2f}")

    # Most sparse layers
    print("\nMOST SPARSE LAYERS (Least active):")
    for layer, stats in sorted_layers[-5:]:
        print(f"{layer}: {stats['sparsity']*100:.2f}% sparse, "
              f"Mean={stats['mean_activation']:.4f}, Max={stats['max_activation']:.2f}")

    # Compare critical layers
    print("\nCRITICAL LAYERS (L8, L9, L10, L11, L31):")
    for layer_name in ['L8', 'L9', 'L10', 'L11', 'L31']:
        stats = sparsity_stats[layer_name]
        print(f"{layer_name}: {stats['sparsity']*100:.2f}% sparse, "
              f"Mean={stats['mean_activation']:.4f}, Max={stats['max_activation']:.2f}")

if __name__ == '__main__':
    print("=" * 80)
    print("PHASE 1: SPARSITY EVOLUTION ANALYSIS")
    print("=" * 80)

    results = load_pathway_data()
    print(f"Loaded {len(results)} games")

    sparsity_stats = analyze_sparsity(results)
    create_sparsity_plots(sparsity_stats)
    save_results(sparsity_stats)
    print_summary(sparsity_stats)

    print("\n✅ Phase 1 Sparsity Evolution Analysis Complete!")
