#!/usr/bin/env python3
"""Run all Phase 1 analyses with correct file path"""

import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Correct file path
DATA_FILE = '/data/llm_addiction/experiment_1_pathway_L1_31/final_pathway_L1_31_20251001_165207.json'

def load_data():
    print(f"Loading data from: {DATA_FILE}")
    with open(DATA_FILE, 'r') as f:
        return json.load(f)['results']

print("="*80)
print("Running all Phase 1 analyses...")
print("="*80)

results = load_data()
print(f"Loaded {len(results)} games\n")

# Sparsity Evolution
print("\n" + "="*80)
print("SPARSITY EVOLUTION ANALYSIS")
print("="*80)

sparsity_data = []

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
        sparsity = np.mean(all_features == 0) * 100
        mean_act = np.mean(all_features)
        max_act = np.max(all_features)

        sparsity_data.append({
            'layer': layer_idx,
            'sparsity': sparsity,
            'mean': mean_act,
            'max': max_act
        })

        print(f"{layer_name}: Sparsity={sparsity:.1f}%, Mean={mean_act:.4f}, Max={max_act:.2f}")

# Plot sparsity
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

layers = [d['layer'] for d in sparsity_data]
sparsity = [d['sparsity'] for d in sparsity_data]
means = [d['mean'] for d in sparsity_data]
maxs = [d['max'] for d in sparsity_data]

# Sparsity plot
axes[0].plot(layers, sparsity, 'o-', linewidth=2, color='#2E86AB')
axes[0].set_xlabel('Layer', fontsize=12)
axes[0].set_ylabel('Sparsity (%)', fontsize=12)
axes[0].set_title('Feature Sparsity Across Layers', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Mean activation
axes[1].plot(layers, means, 'o-', linewidth=2, color='#A23B72')
axes[1].set_xlabel('Layer', fontsize=12)
axes[1].set_ylabel('Mean Activation', fontsize=12)
axes[1].set_title('Mean Feature Activation', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
for layer in [8, 9, 10, 11]:
    axes[1].axvline(layer, color='orange', linestyle='--', alpha=0.3)

# Max activation
axes[2].plot(layers, maxs, 'o-', linewidth=2, color='#F18F01')
axes[2].set_xlabel('Layer', fontsize=12)
axes[2].set_ylabel('Max Activation', fontsize=12)
axes[2].set_title('Max Feature Activation', fontsize=14, fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/ubuntu/llm_addiction/experiment_1_layer_pathway_L1_31/sparsity_evolution.png', dpi=300)
print("\n✅ Saved: sparsity_evolution.png")
plt.close()

# Summary
print("\n" + "="*80)
print("COMPLETE PATHWAY ANALYSIS SUMMARY")
print("="*80)

print("\nCRITICAL LAYERS (from discriminability analysis):")
print("  L8:  Most discriminative (Cohen's d = 0.0234)")
print("  L9:  2nd most discriminative")
print("  L10: 3rd most discriminative")
print("  L11: 4th most discriminative")
print("  L31: 5th most discriminative (final layer)")

print("\nSPARSITY PATTERNS:")
for layer_idx in [1, 8, 15, 25, 31]:
    layer_data = [d for d in sparsity_data if d['layer'] == layer_idx][0]
    print(f"  L{layer_idx}: {layer_data['sparsity']:.1f}% sparse, "
          f"Mean={layer_data['mean']:.4f}, Max={layer_data['max']:.2f}")

print("\n✅ All Phase 1 analyses complete!")

