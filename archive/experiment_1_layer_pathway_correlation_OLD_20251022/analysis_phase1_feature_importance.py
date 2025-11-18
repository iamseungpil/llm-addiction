#!/usr/bin/env python3
"""
Phase 1 Analysis: Feature Importance per Layer
Identify which specific features are most discriminative in each critical layer
"""

import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path

def load_pathway_data():
    """Load Experiment 1 pathway data"""
    print("Loading pathway data...")
    with open('/data/llm_addiction/experiment_1_pathway_L1_31/final_pathway_L1_31_20251001_165207.json', 'r') as f:
        data = json.load(f)
    return data['results']

def extract_top_features_per_layer(results, top_k=20):
    """Extract top K most discriminative features for each layer"""
    print("\nExtracting top features per layer...")

    critical_layers = ['L8', 'L9', 'L10', 'L11', 'L31']  # Focus on critical layers
    layer_top_features = {}

    for layer_name in critical_layers:
        print(f"\n  Analyzing {layer_name}...")

        bankruptcy_features = []
        safe_features = []

        for game in results:
            if game['round_data']:
                last_round = game['round_data'][-1]
                if layer_name in last_round['features']:
                    features = np.array(last_round['features'][layer_name])

                    if game['outcome'] == 'bankruptcy':
                        bankruptcy_features.append(features)
                    else:
                        safe_features.append(features)

        if len(bankruptcy_features) == 0 or len(safe_features) == 0:
            continue

        bankruptcy_features = np.array(bankruptcy_features)
        safe_features = np.array(safe_features)

        # Calculate Cohen's d for each feature
        b_mean = np.mean(bankruptcy_features, axis=0)
        s_mean = np.mean(safe_features, axis=0)
        pooled_std = np.sqrt((np.var(bankruptcy_features, axis=0) + np.var(safe_features, axis=0)) / 2)
        cohen_d = (b_mean - s_mean) / (pooled_std + 1e-10)

        # Calculate t-statistics
        t_stats = []
        p_values = []
        for i in range(len(b_mean)):
            t, p = stats.ttest_ind(bankruptcy_features[:, i], safe_features[:, i])
            t_stats.append(t)
            p_values.append(p)

        t_stats = np.array(t_stats)
        p_values = np.array(p_values)

        # Get top K features by |Cohen's d|
        abs_cohen_d = np.abs(cohen_d)
        top_indices = np.argsort(abs_cohen_d)[::-1][:top_k]

        top_features = []
        for idx in top_indices:
            top_features.append({
                'feature_id': int(idx),
                'cohen_d': float(cohen_d[idx]),
                'abs_cohen_d': float(abs_cohen_d[idx]),
                't_statistic': float(t_stats[idx]),
                'p_value': float(p_values[idx]),
                'bankruptcy_mean': float(b_mean[idx]),
                'safe_mean': float(s_mean[idx]),
                'direction': 'risky' if cohen_d[idx] > 0 else 'safe'
            })

        layer_top_features[layer_name] = top_features

        # Print summary
        print(f"    Top feature: {layer_name}-{top_features[0]['feature_id']}")
        print(f"      Cohen's d: {top_features[0]['cohen_d']:.3f}")
        print(f"      p-value: {top_features[0]['p_value']:.6f}")
        print(f"      Direction: {top_features[0]['direction']}")

    return layer_top_features

def create_feature_importance_plot(layer_top_features):
    """Create visualization of top features per layer"""
    print("\nCreating feature importance plot...")

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))

    layers = ['L8', 'L9', 'L10', 'L11', 'L31']

    for idx, layer_name in enumerate(layers):
        ax = axes[idx]
        features = layer_top_features[layer_name][:10]  # Top 10

        feature_ids = [f['feature_id'] for f in features]
        cohen_ds = [f['cohen_d'] for f in features]
        colors = ['#C73E1D' if d > 0 else '#2E86AB' for d in cohen_ds]

        y_pos = np.arange(len(feature_ids))
        ax.barh(y_pos, cohen_ds, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{layer_name}-{fid}" for fid in feature_ids], fontsize=8)
        ax.set_xlabel("Cohen's d", fontsize=10)
        ax.set_title(f"{layer_name} Top Features", fontsize=12, fontweight='bold')
        ax.axvline(0, color='black', linewidth=1, linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    output_path = '/home/ubuntu/llm_addiction/experiment_1_layer_pathway_L1_31/feature_importance_critical_layers.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()

def save_results(layer_top_features):
    """Save feature importance results"""
    output_file = '/home/ubuntu/llm_addiction/experiment_1_layer_pathway_L1_31/feature_importance_results.json'

    with open(output_file, 'w') as f:
        json.dump(layer_top_features, f, indent=2)

    print(f"Saved results: {output_file}")

def print_top_features_summary(layer_top_features):
    """Print summary of top features"""
    print("\n" + "=" * 100)
    print("TOP 5 MOST DISCRIMINATIVE FEATURES PER CRITICAL LAYER")
    print("=" * 100)

    for layer_name in ['L8', 'L9', 'L10', 'L11', 'L31']:
        features = layer_top_features[layer_name][:5]
        print(f"\n{layer_name}:")
        print("-" * 100)
        cohen_header = "Cohen's d"
        print(f"{'Rank':<6} {'Feature':<15} {cohen_header:<12} {'p-value':<12} {'Direction':<10} {'Bankruptcy Mean':<15} {'Safe Mean':<15}")
        print("-" * 100)

        for i, feat in enumerate(features, 1):
            print(f"{i:<6} {layer_name}-{feat['feature_id']:<10} "
                  f"{feat['cohen_d']:>10.3f}  {feat['p_value']:>10.6f}  "
                  f"{feat['direction']:<10} {feat['bankruptcy_mean']:>14.6f}  {feat['safe_mean']:>14.6f}")

if __name__ == '__main__':
    print("=" * 100)
    print("PHASE 1: FEATURE IMPORTANCE RANKING PER LAYER")
    print("=" * 100)

    results = load_pathway_data()
    print(f"Loaded {len(results)} games")

    bankruptcy_count = sum(1 for r in results if r['outcome'] == 'bankruptcy')
    safe_count = sum(1 for r in results if r['outcome'] == 'voluntary_stop')
    print(f"Bankruptcy: {bankruptcy_count}, Safe: {safe_count}")

    layer_top_features = extract_top_features_per_layer(results, top_k=20)
    create_feature_importance_plot(layer_top_features)
    save_results(layer_top_features)
    print_top_features_summary(layer_top_features)

    print("\n✅ Phase 1 Feature Importance Analysis Complete!")
