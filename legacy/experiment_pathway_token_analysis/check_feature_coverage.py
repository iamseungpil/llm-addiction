#!/usr/bin/env python3
"""Check how many reparsed features have means in existing file"""

import json

print("Loading reparsed causal features...")
with open('/data/llm_addiction/experiment_2_multilayer_patching/reparsed/reparsed_causal_features_20251125_043558.json') as f:
    reparsed = json.load(f)
causal_features = reparsed['causal_features']
print(f"Loaded {len(causal_features)} reparsed causal features")

print("\nLoading existing means from L1_31_features_CONVERTED_20251111.json...")
with open('/data/llm_addiction/experiment_1_L1_31_extraction/L1_31_features_CONVERTED_20251111.json') as f:
    converted = json.load(f)

# Build set of existing features
existing_features = set()
for layer_key, layer_data in converted['layer_results'].items():
    layer = int(layer_key)
    for feat in layer_data['significant_features']:
        feature_id = feat['feature_idx']
        existing_features.add((layer, feature_id))

print(f"Loaded {len(existing_features)} features with means")

# Check overlap
matched = 0
missing = []

for feat in causal_features:
    layer = feat['layer']
    feature_id = feat['feature_id']

    if (layer, feature_id) in existing_features:
        matched += 1
    else:
        missing.append(f"L{layer}-{feature_id}")

coverage_pct = 100 * matched / len(causal_features)

print("\n" + "="*60)
print("COVERAGE ANALYSIS RESULTS")
print("="*60)
print(f"✅ Matched: {matched}/{len(causal_features)} ({coverage_pct:.1f}%)")
print(f"❌ Missing: {len(missing)}")

if coverage_pct >= 80:
    print(f"\n✅ GOOD COVERAGE ({coverage_pct:.1f}%) - Can proceed with pathway analysis")
elif coverage_pct >= 50:
    print(f"\n⚠️  MODERATE COVERAGE ({coverage_pct:.1f}%) - Consider extracting missing means")
else:
    print(f"\n❌ LOW COVERAGE ({coverage_pct:.1f}%) - Must extract missing means")

if len(missing) > 0 and len(missing) <= 50:
    print(f"\nFirst 20 missing features:")
    for feat in missing[:20]:
        print(f"  {feat}")
elif len(missing) > 50:
    print(f"\nFirst 20 missing features (out of {len(missing)} total):")
    for feat in missing[:20]:
        print(f"  {feat}")

# Save missing features list
if len(missing) > 0:
    output = {
        'total_reparsed': len(causal_features),
        'matched': matched,
        'missing_count': len(missing),
        'coverage_percent': coverage_pct,
        'missing_features': missing
    }

    output_path = '/home/ubuntu/llm_addiction/experiment_pathway_token_analysis/missing_features.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n💾 Missing features list saved to: {output_path}")
