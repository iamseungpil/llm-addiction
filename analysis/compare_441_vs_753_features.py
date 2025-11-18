#!/usr/bin/env python3
"""Compare 441-feature analysis (GPU 4/5) vs 753-feature L1_31 re-analysis"""

import pandas as pd
from pathlib import Path

def main():
    print("=" * 80)
    print("COMPARING 441 (Original) vs 753 (L1_31 Re-analysis) CAUSAL FEATURES")
    print("=" * 80)

    # Load CSVs
    csv_441 = Path("/home/ubuntu/llm_addiction/analysis/exp2_feature_group_summary.csv")
    csv_753 = Path("/home/ubuntu/llm_addiction/analysis/exp2_L1_31_feature_group_summary_BASELINE.csv")

    df_441 = pd.read_csv(csv_441)
    df_753 = pd.read_csv(csv_753)

    print(f"\nLoaded:")
    print(f"  Original (GPU 4/5): {len(df_441)} total features")
    print(f"  L1_31 re-analysis: {len(df_753)} total features")

    # Filter to causal only
    df_441_causal = df_441[df_441['classified_as'].isin(['safe', 'risky'])].copy()
    df_753_causal = df_753[df_753['classified_as'].isin(['safe', 'risky'])].copy()

    print(f"\nCausal features only:")
    print(f"  Original: {len(df_441_causal)} causal (441 expected)")
    print(f"  L1_31: {len(df_753_causal)} causal (753 expected)")

    # Find overlapping features
    features_441 = set(df_441_causal['feature'])
    features_753 = set(df_753_causal['feature'])

    overlap = features_441 & features_753
    only_441 = features_441 - features_753
    only_753 = features_753 - features_441

    print(f"\n{'=' * 80}")
    print(f"OVERLAP ANALYSIS:")
    print(f"  Common causal features (ROBUST): {len(overlap)}")
    print(f"  Only in Original (441): {len(only_441)}")
    print(f"  Only in L1_31 (753): {len(only_753)}")
    print(f"  Total unique causal features: {len(features_441 | features_753)}")
    print(f"{'=' * 80}")

    # Check classification agreement in overlap
    if len(overlap) > 0:
        print(f"\nClassification agreement in {len(overlap)} overlapping features:")

        # Create lookup dictionaries
        classification_441 = df_441_causal.set_index('feature')['classified_as'].to_dict()
        classification_753 = df_753_causal.set_index('feature')['classified_as'].to_dict()

        agreement = 0
        disagreement = 0
        for feat in overlap:
            if classification_441[feat] == classification_753[feat]:
                agreement += 1
            else:
                disagreement += 1

        print(f"  Agreement: {agreement}/{len(overlap)} ({100*agreement/len(overlap):.1f}%)")
        print(f"  Disagreement: {disagreement}/{len(overlap)} ({100*disagreement/len(overlap):.1f}%)")

        # Show disagreements
        if disagreement > 0:
            print(f"\n  Disagreements (features classified differently):")
            for feat in sorted(overlap):
                c_441 = classification_441[feat]
                c_753 = classification_753[feat]
                if c_441 != c_753:
                    print(f"    {feat}: Original={c_441}, L1_31={c_753}")

    # Layer-wise breakdown
    print(f"\n{'=' * 80}")
    print("LAYER-WISE BREAKDOWN:")
    print(f"{'Layer':<8} {'Overlap':<10} {'Only 441':<12} {'Only 753':<12} {'Total Unique':<12}")
    print("-" * 80)

    for layer in [25, 26, 27, 28, 29, 30, 31]:
        overlap_layer = [f for f in overlap if f.startswith(f'L{layer}-')]
        only_441_layer = [f for f in only_441 if f.startswith(f'L{layer}-')]
        only_753_layer = [f for f in only_753 if f.startswith(f'L{layer}-')]
        total_layer = len(overlap_layer) + len(only_441_layer) + len(only_753_layer)

        print(f"L{layer:<7} {len(overlap_layer):<10} {len(only_441_layer):<12} {len(only_753_layer):<12} {total_layer:<12}")

    print("-" * 80)
    print(f"{'TOTAL':<8} {len(overlap):<10} {len(only_441):<12} {len(only_753):<12} {len(features_441 | features_753):<12}")

    # Save robust features (overlap with agreement)
    if len(overlap) > 0:
        robust_features = []
        for feat in overlap:
            if classification_441[feat] == classification_753[feat]:
                robust_features.append({
                    'feature': feat,
                    'classification': classification_441[feat],
                    'source': 'both_analyses_agree'
                })

        robust_df = pd.DataFrame(robust_features)
        output_path = Path("/home/ubuntu/llm_addiction/analysis/robust_causal_features_OVERLAP.csv")
        robust_df.to_csv(output_path, index=False)

        print(f"\n✅ Saved {len(robust_features)} robust causal features (overlap + agreement) to:")
        print(f"   {output_path}")

    # Create comprehensive feature list with source tracking
    all_features_data = []

    for feat in features_441 | features_753:
        in_441 = feat in features_441
        in_753 = feat in features_753

        if in_441 and in_753:
            c_441 = classification_441[feat]
            c_753 = classification_753[feat]
            if c_441 == c_753:
                source = "both_agree"
                classification = c_441
            else:
                source = "both_disagree"
                classification = f"441:{c_441}, 753:{c_753}"
        elif in_441:
            source = "only_441"
            classification = classification_441[feat]
        else:
            source = "only_753"
            classification = classification_753[feat]

        all_features_data.append({
            'feature': feat,
            'in_441': in_441,
            'in_753': in_753,
            'source': source,
            'classification': classification
        })

    comprehensive_df = pd.DataFrame(all_features_data).sort_values('feature')
    comprehensive_path = Path("/home/ubuntu/llm_addiction/analysis/comprehensive_causal_features_comparison.csv")
    comprehensive_df.to_csv(comprehensive_path, index=False)

    print(f"\n✅ Saved comprehensive feature comparison ({len(all_features_data)} features) to:")
    print(f"   {comprehensive_path}")

    print(f"\n{'=' * 80}")
    print("SUMMARY:")
    print(f"  ✅ {len(overlap)} features found in BOTH analyses (robust)")
    print(f"     - {agreement if len(overlap) > 0 else 0} agree on classification (high confidence)")
    print(f"     - {disagreement if len(overlap) > 0 else 0} disagree on classification (needs review)")
    print(f"  📊 {len(only_441)} features unique to Original (441)")
    print(f"  📊 {len(only_753)} features unique to L1_31 (753)")
    print(f"  🎯 Total unique causal features discovered: {len(features_441 | features_753)}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
