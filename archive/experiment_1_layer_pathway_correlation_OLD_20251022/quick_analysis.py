#!/usr/bin/env python3
"""Quick analysis using simple metrics (not full classification)"""

import json
import numpy as np
from scipy import stats

def load_data():
    with open('/data/llm_addiction/experiment_1_pathway_L1_31/final_pathway_L1_31_20251001_165207.json', 'r') as f:
        return json.load(f)['results']

def analyze_layer_separation(results):
    """Use Cohen's d to measure separation (faster than classification)"""
    print("Analyzing layer-wise separation (Cohen's d)...")

    layer_stats = {}

    for layer_idx in range(1, 32):
        layer_name = f'L{layer_idx}'

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

        if len(bankruptcy_features) > 0 and len(safe_features) > 0:
            bankruptcy_features = np.array(bankruptcy_features)
            safe_features = np.array(safe_features)

            # Mean features for each group
            b_mean = np.mean(bankruptcy_features, axis=0)
            s_mean = np.mean(safe_features, axis=0)

            # Pooled std
            pooled_std = np.sqrt((np.var(bankruptcy_features, axis=0) + np.var(safe_features, axis=0)) / 2)

            # Cohen's d per feature
            cohen_d = (b_mean - s_mean) / (pooled_std + 1e-10)

            # Aggregate metrics
            layer_stats[layer_name] = {
                'mean_abs_cohen_d': float(np.mean(np.abs(cohen_d))),
                'max_abs_cohen_d': float(np.max(np.abs(cohen_d))),
                'n_strong_features': int(np.sum(np.abs(cohen_d) > 0.5)),
                'n_samples_bankruptcy': len(bankruptcy_features),
                'n_samples_safe': len(safe_features)
            }

            print(f"{layer_name}: Mean |d|={layer_stats[layer_name]['mean_abs_cohen_d']:.4f}, "
                  f"Max |d|={layer_stats[layer_name]['max_abs_cohen_d']:.2f}, "
                  f"Strong={layer_stats[layer_name]['n_strong_features']}")

    return layer_stats

def main():
    results = load_data()
    print(f"Loaded {len(results)} games")

    bankruptcy = sum(1 for r in results if r['outcome'] == 'bankruptcy')
    print(f"Bankruptcy: {bankruptcy}, Safe: {len(results) - bankruptcy}\n")

    layer_stats = analyze_layer_separation(results)

    # Save results
    with open('/home/ubuntu/llm_addiction/experiment_1_layer_pathway_L1_31/quick_analysis_results.json', 'w') as f:
        json.dump(layer_stats, f, indent=2)

    # Top layers
    sorted_layers = sorted(layer_stats.items(),
                          key=lambda x: x[1]['mean_abs_cohen_d'],
                          reverse=True)

    print("\n" + "="*80)
    print("TOP 10 MOST DISCRIMINATIVE LAYERS (by mean |Cohen's d|)")
    print("="*80)
    for i, (layer, stats) in enumerate(sorted_layers[:10], 1):
        print(f"{i}. {layer}: Mean |d|={stats['mean_abs_cohen_d']:.4f}, "
              f"Max |d|={stats['max_abs_cohen_d']:.2f}, "
              f"Strong features={stats['n_strong_features']}")

    print("\n✅ Analysis complete!")

if __name__ == '__main__':
    main()
