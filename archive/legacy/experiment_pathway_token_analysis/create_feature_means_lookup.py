#!/usr/bin/env python3
"""
Extract safe_mean and bankrupt_mean (risky_mean) values for causal features
from L1_31 extraction data
"""

import json
from pathlib import Path
from tqdm.auto import tqdm

def main():
    # Load L1_31 feature extraction results
    l1_31_file = Path("/data/llm_addiction/experiment_1_L1_31_extraction/L1_31_features_FINAL_20250930_204420.json")

    print(f"Loading L1_31 data from {l1_31_file}...")
    with open(l1_31_file, 'r') as f:
        l1_31_data = json.load(f)

    # Load causal features list
    causal_file = Path("/home/ubuntu/llm_addiction/experiment_pathway_token_analysis/causal_features_list.json")
    with open(causal_file, 'r') as f:
        causal_data = json.load(f)

    causal_features = causal_data['features']
    print(f"Total causal features to extract: {len(causal_features)}")

    # Create lookup dictionary: {f"L{layer}-{feature_id}": {"safe_mean": x, "risky_mean": y}}
    feature_means = {}
    missing_features = []

    for feat in tqdm(causal_features, desc="Extracting means"):
        layer = feat['layer']
        feature_id = feat['feature_id']
        feature_name = f"L{layer}-{feature_id}"

        # Find this feature in L1_31 data
        layer_results = l1_31_data['layer_results'].get(str(layer))

        if not layer_results:
            missing_features.append(feature_name)
            continue

        # Find matching feature_idx
        found = False
        for feature_data in layer_results['significant_features']:
            if feature_data['feature_idx'] == feature_id:
                feature_means[feature_name] = {
                    'safe_mean': float(feature_data['safe_mean']),
                    'risky_mean': float(feature_data['bankrupt_mean'])
                }
                found = True
                break

        if not found:
            missing_features.append(feature_name)

    print(f"\n✅ Successfully extracted {len(feature_means)} feature means")
    print(f"❌ Missing {len(missing_features)} features")

    if missing_features:
        print(f"Sample missing features: {missing_features[:10]}")

    # Save lookup dictionary
    output_file = Path("/home/ubuntu/llm_addiction/experiment_pathway_token_analysis/feature_means_lookup.json")
    with open(output_file, 'w') as f:
        json.dump({
            'source': 'L1_31_features_FINAL_20250930_204420',
            'total_features': len(feature_means),
            'missing_features': missing_features,
            'feature_means': feature_means
        }, f, indent=2)

    print(f"\n✅ Saved to {output_file}")

    # Print sample
    sample_features = list(feature_means.items())[:5]
    print("\nSample features:")
    for fname, means in sample_features:
        print(f"  {fname}: safe={means['safe_mean']:.6f}, risky={means['risky_mean']:.6f}")

if __name__ == "__main__":
    main()
