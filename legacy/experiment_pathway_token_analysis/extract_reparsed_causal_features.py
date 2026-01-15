#!/usr/bin/env python3
"""
Extract causal features from reparsed L1-31 experiment results
for use in pathway token analysis
"""

import json
from pathlib import Path
from collections import defaultdict

def main():
    # Load reparsed causal features
    reparsed_file = Path("/data/llm_addiction/experiment_2_multilayer_patching/reparsed/reparsed_causal_features_20251125_043558.json")

    print(f"Loading reparsed causal features from: {reparsed_file}")
    with open(reparsed_file, 'r') as f:
        reparsed_data = json.load(f)

    total_features = reparsed_data['total_causal_features']
    causal_features = reparsed_data['causal_features']

    print(f"Total causal features: {total_features}")

    # Extract feature list in pathway analysis format
    feature_list = []
    layer_counts = defaultdict(int)

    for feat in causal_features:
        layer = feat['layer']
        feature_id = feat['feature_id']

        feature_list.append({
            'layer': layer,
            'feature_id': feature_id
        })

        layer_counts[layer] += 1

    # Get unique layers
    layers = sorted(set(f['layer'] for f in feature_list))

    # Create output structure
    output = {
        'source': 'REPARSED L1-31 experiment (2025-11-25)',
        'total_features': len(feature_list),
        'layers': layers,
        'layer_distribution': dict(layer_counts),
        'features': feature_list
    }

    # Save to pathway analysis directory
    output_file = Path("/home/ubuntu/llm_addiction/experiment_pathway_token_analysis/causal_features_list_REPARSED.json")
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Saved to: {output_file}")
    print(f"\nLayer distribution:")
    for layer in sorted(layer_counts.keys()):
        print(f"  Layer {layer:2d}: {layer_counts[layer]:4d} features")

    # Print comparison with old list
    old_file = Path("/home/ubuntu/llm_addiction/experiment_pathway_token_analysis/causal_features_list.json")
    if old_file.exists():
        with open(old_file, 'r') as f:
            old_data = json.load(f)

        print(f"\n📊 Comparison:")
        print(f"  Old total: {old_data['total_features']} features")
        print(f"  New total: {output['total_features']} features")
        print(f"  Difference: {output['total_features'] - old_data['total_features']:+d} features")

if __name__ == "__main__":
    main()
