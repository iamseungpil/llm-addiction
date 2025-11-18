#!/usr/bin/env python3
"""
Phase 1 Analysis: Layer-wise Discriminability
Measure how well each layer separates bankruptcy vs safe decisions
"""

import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path

def load_pathway_data():
    """Load Experiment 1 pathway data"""
    print("Loading pathway data...")
    with open('/data/llm_addiction/experiment_1_pathway_L1_31/final_pathway_L1_31_20251001_165207.json', 'r') as f:
        data = json.load(f)
    return data['results']

def extract_layer_features(results, layer_name):
    """Extract features from specific layer for all games"""
    bankruptcy_features = []
    safe_features = []

    for game in results:
        outcome = game['outcome']

        # Get last decision features
        if game['round_data']:
            last_round = game['round_data'][-1]
            if layer_name in last_round['features']:
                features = np.array(last_round['features'][layer_name])

                if outcome == 'bankruptcy':
                    bankruptcy_features.append(features)
                else:
                    safe_features.append(features)

    return np.array(bankruptcy_features), np.array(safe_features)

def measure_discriminability(bankruptcy_features, safe_features):
    """Measure how well features separate two groups using linear classifier"""

    # Combine data
    X = np.vstack([bankruptcy_features, safe_features])
    y = np.array([1] * len(bankruptcy_features) + [0] * len(safe_features))

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train linear classifier with cross-validation
    clf = LogisticRegression(max_iter=1000, random_state=42)
    scores = cross_val_score(clf, X_scaled, y, cv=5)

    return {
        'mean_accuracy': scores.mean(),
        'std_accuracy': scores.std(),
        'scores': scores.tolist()
    }

def analyze_all_layers(results):
    """Analyze discriminability for all layers"""
    print("\nAnalyzing layer-wise discriminability...")

    layer_results = {}

    for layer_idx in range(1, 32):
        layer_name = f'L{layer_idx}'
        print(f"  Analyzing {layer_name}...", end=" ")

        bankruptcy_feats, safe_feats = extract_layer_features(results, layer_name)

        if len(bankruptcy_feats) > 0 and len(safe_feats) > 0:
            disc_results = measure_discriminability(bankruptcy_feats, safe_feats)
            layer_results[layer_name] = disc_results
            print(f"Accuracy: {disc_results['mean_accuracy']:.3f} ± {disc_results['std_accuracy']:.3f}")
        else:
            print("Insufficient data")

    return layer_results

def plot_discriminability(layer_results):
    """Plot layer-wise discriminability"""
    layers = sorted(layer_results.keys(), key=lambda x: int(x[1:]))
    accuracies = [layer_results[l]['mean_accuracy'] for l in layers]
    stds = [layer_results[l]['std_accuracy'] for l in layers]

    plt.figure(figsize=(14, 6))

    x = [int(l[1:]) for l in layers]
    plt.plot(x, accuracies, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    plt.fill_between(x,
                     np.array(accuracies) - np.array(stds),
                     np.array(accuracies) + np.array(stds),
                     alpha=0.3, color='#2E86AB')

    plt.axhline(0.5, color='red', linestyle='--', label='Chance (50%)', alpha=0.5)
    plt.xlabel('Layer', fontsize=14)
    plt.ylabel('Classification Accuracy', fontsize=14)
    plt.title('Layer-wise Discriminability: Bankruptcy vs Safe Decisions', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = '/home/ubuntu/llm_addiction/experiment_1_layer_pathway_L1_31/layer_discriminability.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot: {output_path}")
    plt.close()

def save_results(layer_results):
    """Save analysis results"""
    output_file = '/home/ubuntu/llm_addiction/experiment_1_layer_pathway_L1_31/layer_discriminability_results.json'

    with open(output_file, 'w') as f:
        json.dump(layer_results, f, indent=2)

    print(f"Saved results: {output_file}")

if __name__ == '__main__':
    print("=" * 80)
    print("PHASE 1: LAYER-WISE DISCRIMINABILITY ANALYSIS")
    print("=" * 80)

    results = load_pathway_data()
    print(f"Loaded {len(results)} games")

    bankruptcy_count = sum(1 for r in results if r['outcome'] == 'bankruptcy')
    safe_count = sum(1 for r in results if r['outcome'] == 'voluntary_stop')
    print(f"Bankruptcy: {bankruptcy_count}, Safe: {safe_count}")

    layer_results = analyze_all_layers(results)
    plot_discriminability(layer_results)
    save_results(layer_results)

    # Print top layers
    sorted_layers = sorted(layer_results.items(),
                          key=lambda x: x[1]['mean_accuracy'],
                          reverse=True)

    print("\n" + "=" * 80)
    print("TOP 5 MOST DISCRIMINATIVE LAYERS")
    print("=" * 80)
    for layer, result in sorted_layers[:5]:
        print(f"{layer}: {result['mean_accuracy']:.3f} ± {result['std_accuracy']:.3f}")

    print("\n✅ Phase 1 Analysis Complete!")
