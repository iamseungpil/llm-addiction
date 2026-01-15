#!/usr/bin/env python3
"""
Feature Means 변환 스크립트

L1_31_features_CONVERTED_20251111.json (13,434개)를
Phase 1이 읽을 수 있는 형식으로 변환합니다.

변환 내용:
1. layer_results[layer]['significant_features'] → feature_means[L{layer}-{feature_idx}]
2. bankrupt_mean → risky_mean
3. 2,510개 reparsed causal features만 추출
"""

import json
from pathlib import Path

def main():
    # 1. Load reparsed causal features (2,510개)
    print("Loading reparsed causal features...")
    reparsed_file = Path("/home/ubuntu/llm_addiction/experiment_pathway_token_analysis/causal_features_list_REPARSED.json")
    with open(reparsed_file, 'r') as f:
        reparsed_data = json.load(f)

    reparsed_features = set()
    for feat in reparsed_data['features']:
        layer = feat['layer']
        feature_id = feat['feature_id']
        reparsed_features.add((layer, feature_id))

    print(f"  Total reparsed causal features: {len(reparsed_features)}")

    # 2. Load CONVERTED file (13,434개)
    print("\nLoading CONVERTED feature means...")
    converted_file = Path("/data/llm_addiction/experiment_1_L1_31_extraction/L1_31_features_CONVERTED_20251111.json")
    with open(converted_file, 'r') as f:
        converted_data = json.load(f)

    # 3. Build feature_means dict in Phase 1 expected format
    print("\nConverting to Phase 1 format...")
    feature_means = {}
    matched = 0
    missing = []

    for layer_key, layer_data in converted_data['layer_results'].items():
        layer = int(layer_key)
        for feat in layer_data['significant_features']:
            feature_id = feat['feature_idx']

            # Only include reparsed causal features
            if (layer, feature_id) not in reparsed_features:
                continue

            feature_name = f"L{layer}-{feature_id}"

            feature_means[feature_name] = {
                'safe_mean': feat['safe_mean'],
                'risky_mean': feat['bankrupt_mean'],  # bankrupt_mean → risky_mean
                'cohen_d': feat.get('cohen_d', 0.0),
                'p_value': feat.get('p_value', 1.0)
            }
            matched += 1

    # 4. Check for missing features
    for layer, feature_id in reparsed_features:
        feature_name = f"L{layer}-{feature_id}"
        if feature_name not in feature_means:
            missing.append(feature_name)

    print(f"\n=== Conversion Results ===")
    print(f"Reparsed causal features: {len(reparsed_features)}")
    print(f"Matched (have means): {matched}")
    print(f"Missing (no means): {len(missing)}")
    print(f"Coverage: {100*matched/len(reparsed_features):.1f}%")

    if missing:
        print(f"\nFirst 10 missing features:")
        for feat in missing[:10]:
            print(f"  {feat}")

    # 5. Save output
    output = {
        'source': 'L1_31_features_CONVERTED_20251111.json (converted for Phase 1)',
        'causal_features_source': 'REPARSED L1-31 experiment (2025-11-25)',
        'total_features': len(feature_means),
        'missing_count': len(missing),
        'missing_features': missing,
        'feature_means': feature_means
    }

    output_file = Path("/home/ubuntu/llm_addiction/experiment_pathway_token_analysis/feature_means_lookup_REPARSED_FULL.json")
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Saved to: {output_file}")
    print(f"   Total features with means: {len(feature_means)}")

    # 6. Verify format
    print(f"\n=== Sample Entry ===")
    sample_key = list(feature_means.keys())[0]
    print(f"Key: {sample_key}")
    print(f"Value: {feature_means[sample_key]}")

if __name__ == "__main__":
    main()
