#!/usr/bin/env python3
"""
Extract safe_mean and risky_mean values for REPARSED causal features
from L1_31 extraction data

Note: Feature activations themselves are unchanged by reparsing.
Reparsing only affects the behavioral outcome classification.
Therefore, we can use the original feature means from L1-31 extraction.
"""

import json
from pathlib import Path
from tqdm.auto import tqdm

def main():
    # Load L1_31 feature extraction results (UNCHANGED by reparsing)
    l1_31_file = Path("/data/llm_addiction/experiment_1_L1_31_extraction/L1_31_features_FINAL_20250930_220003.json")

    print(f"Loading L1_31 data from {l1_31_file}...")
    with open(l1_31_file, 'r') as f:
        l1_31_data = json.load(f)

    # Load REPARSED causal features list
    causal_file = Path("/home/ubuntu/llm_addiction/experiment_pathway_token_analysis/causal_features_list_REPARSED.json")
    print(f"Loading reparsed causal features from {causal_file}...")
    with open(causal_file, 'r') as f:
        causal_data = json.load(f)

    causal_features = causal_data['features']
    print(f"Total reparsed causal features to extract: {len(causal_features)}")

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
        print(f"⚠️ Sample missing features: {missing_features[:10]}")
        print(f"⚠️ Total missing: {len(missing_features)}")

    # Save lookup dictionary
    output_file = Path("/home/ubuntu/llm_addiction/experiment_pathway_token_analysis/feature_means_lookup_REPARSED.json")
    with open(output_file, 'w') as f:
        json.dump({
            'source': 'L1_31_features_FINAL_20250930_220003 (activation values)',
            'causal_features_source': 'REPARSED L1-31 experiment (2025-11-25)',
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

    # Compare with old lookup
    old_file = Path("/home/ubuntu/llm_addiction/experiment_pathway_token_analysis/feature_means_lookup.json")
    if old_file.exists():
        with open(old_file, 'r') as f:
            old_data = json.load(f)

        print(f"\n📊 Comparison:")
        print(f"  Old features with means: {old_data['total_features']}")
        print(f"  New features with means: {len(feature_means)}")
        print(f"  Difference: {len(feature_means) - old_data['total_features']:+d}")

if __name__ == "__main__":
    main()
