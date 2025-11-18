#!/usr/bin/env python3
"""
Post-Hoc Global FDR Correction for L1-31 SAE Extraction
Converts layer-wise FDR results to global FDR (compatible with reference methodology)
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import false_discovery_control
from statsmodels.stats.multitest import multipletests
from datetime import datetime
import argparse

def load_checkpoint_results(checkpoint_paths):
    """Load all layer results from checkpoint files"""
    all_layer_results = {}

    for path in checkpoint_paths:
        if not Path(path).exists():
            print(f"⚠️  Checkpoint not found: {path}")
            continue

        with open(path, 'r') as f:
            data = json.load(f)

        layer_results = data.get('layer_results', {})

        for layer_id, layer_data in layer_results.items():
            if layer_id in all_layer_results:
                print(f"⚠️  Layer {layer_id} already loaded, skipping duplicate")
                continue
            all_layer_results[layer_id] = layer_data

        print(f"✅ Loaded {len(layer_results)} layers from {Path(path).name}")

    return all_layer_results

def extract_all_features(layer_results):
    """Extract all features with their p-values across all layers"""
    all_features = []

    for layer_id, layer_data in sorted(layer_results.items(), key=lambda x: int(x[0])):
        significant_features = layer_data.get('significant_features', [])

        for feature in significant_features:
            feature_record = {
                'layer': int(layer_id),
                'feature_idx': feature['feature_idx'],
                'p_value': feature['p_value'],
                'cohen_d': feature['cohen_d'],
                'bankrupt_mean': feature['bankrupt_mean'],
                'safe_mean': feature['safe_mean'],
                'bankrupt_std': feature['bankrupt_std'],
                'safe_std': feature['safe_std'],
                'p_corrected_layer': feature.get('p_corrected', feature['p_value'])  # Original layer-wise correction
            }
            all_features.append(feature_record)

    return all_features

def apply_global_fdr(all_features, alpha=0.05, method='fdr_bh'):
    """Apply global FDR correction across all layers"""

    if not all_features:
        print("❌ No features to correct")
        return []

    # Extract p-values
    p_values = [f['p_value'] for f in all_features]

    print(f"\n{'='*60}")
    print(f"GLOBAL FDR CORRECTION")
    print(f"{'='*60}")
    print(f"Total features: {len(p_values)}")
    print(f"FDR method: {method}")
    print(f"Alpha: {alpha}")

    # Apply Benjamini-Hochberg FDR correction
    rejected, p_corrected, alpha_sidak, alpha_bonf = multipletests(
        p_values,
        method=method,
        alpha=alpha
    )

    # Assign corrected p-values and rejection decisions
    selected_features = []

    for i, feature in enumerate(all_features):
        feature['p_corrected_global'] = p_corrected[i]
        feature['rejected_global'] = bool(rejected[i])

        if rejected[i]:
            selected_features.append(feature)

    print(f"\nResults:")
    print(f"  Original features (layer-wise FDR): {len(all_features)}")
    print(f"  Selected features (global FDR α={alpha}): {len(selected_features)}")
    print(f"  Reduction: {len(all_features) - len(selected_features)} ({100*(1-len(selected_features)/len(all_features)):.1f}%)")

    # Layer breakdown
    layer_counts_before = {}
    layer_counts_after = {}

    for feature in all_features:
        layer = feature['layer']
        layer_counts_before[layer] = layer_counts_before.get(layer, 0) + 1

    for feature in selected_features:
        layer = feature['layer']
        layer_counts_after[layer] = layer_counts_after.get(layer, 0) + 1

    print(f"\nFeatures by layer (before → after):")
    for layer in sorted(layer_counts_before.keys()):
        before = layer_counts_before[layer]
        after = layer_counts_after.get(layer, 0)
        reduction = 100 * (1 - after/before) if before > 0 else 0
        print(f"  Layer {layer:2d}: {before:4d} → {after:4d} ({reduction:5.1f}% reduction)")

    return selected_features

def save_results(selected_features, output_path, metadata):
    """Save globally-corrected results"""

    # Sort by absolute Cohen's d
    selected_features.sort(key=lambda x: abs(x['cohen_d']), reverse=True)

    # Prepare output
    output_data = {
        'metadata': metadata,
        'n_features': len(selected_features),
        'features': selected_features,
        'layer_summary': {}
    }

    # Layer summary
    for feature in selected_features:
        layer = feature['layer']
        if layer not in output_data['layer_summary']:
            output_data['layer_summary'][layer] = {
                'n_features': 0,
                'max_cohen_d': 0,
                'min_p_value': 1.0
            }

        output_data['layer_summary'][layer]['n_features'] += 1
        output_data['layer_summary'][layer]['max_cohen_d'] = max(
            output_data['layer_summary'][layer]['max_cohen_d'],
            abs(feature['cohen_d'])
        )
        output_data['layer_summary'][layer]['min_p_value'] = min(
            output_data['layer_summary'][layer]['min_p_value'],
            feature['p_corrected_global']
        )

    # Save
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✅ Results saved to: {output_path}")
    print(f"   File size: {Path(output_path).stat().st_size / 1024:.1f} KB")

    return output_path

def main():
    parser = argparse.ArgumentParser(description='Apply global FDR correction to L1-31 SAE features')
    parser.add_argument('--checkpoints', nargs='+', required=True,
                       help='Checkpoint JSON files (batch1, batch2, batch3)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSON file path')
    parser.add_argument('--alpha', type=float, default=0.05,
                       help='FDR alpha threshold (default: 0.05)')
    parser.add_argument('--method', type=str, default='fdr_bh',
                       choices=['fdr_bh', 'fdr_by', 'fdr_tsbh', 'fdr_tsbky'],
                       help='FDR correction method (default: fdr_bh)')

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"POST-HOC GLOBAL FDR CORRECTION")
    print(f"{'='*60}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Checkpoints: {len(args.checkpoints)}")
    for cp in args.checkpoints:
        print(f"  - {Path(cp).name}")
    print(f"Output: {args.output}")
    print(f"FDR α: {args.alpha}")
    print(f"Method: {args.method}")

    # Step 1: Load checkpoint results
    print(f"\n{'='*60}")
    print(f"STEP 1: LOADING CHECKPOINTS")
    print(f"{'='*60}")

    layer_results = load_checkpoint_results(args.checkpoints)
    print(f"\n✅ Loaded {len(layer_results)} layers total")

    # Step 2: Extract all features
    print(f"\n{'='*60}")
    print(f"STEP 2: EXTRACTING FEATURES")
    print(f"{'='*60}")

    all_features = extract_all_features(layer_results)
    print(f"✅ Extracted {len(all_features)} features with layer-wise FDR")

    # Step 3: Apply global FDR
    print(f"\n{'='*60}")
    print(f"STEP 3: APPLYING GLOBAL FDR")
    print(f"{'='*60}")

    selected_features = apply_global_fdr(all_features, alpha=args.alpha, method=args.method)

    # Step 4: Save results
    print(f"\n{'='*60}")
    print(f"STEP 4: SAVING RESULTS")
    print(f"{'='*60}")

    metadata = {
        'timestamp': datetime.now().isoformat(),
        'source_checkpoints': [str(cp) for cp in args.checkpoints],
        'fdr_method': args.method,
        'fdr_alpha': args.alpha,
        'correction_type': 'global_multilayer',
        'total_layers': len(layer_results),
        'layers': sorted([int(k) for k in layer_results.keys()]),
        'features_before_global_fdr': len(all_features),
        'features_after_global_fdr': len(selected_features)
    }

    output_path = save_results(selected_features, args.output, metadata)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Global FDR correction complete!")
    print(f"   Input: {len(all_features)} features (layer-wise FDR)")
    print(f"   Output: {len(selected_features)} features (global FDR α={args.alpha})")
    print(f"   File: {output_path}")
    print(f"\nNext steps:")
    print(f"1. Generate feature means lookup:")
    print(f"   python create_feature_means_lookup.py \\")
    print(f"       --features {output_path} \\")
    print(f"       --experiments /data/llm_addiction/results/exp1_*.json \\")
    print(f"       --output L1_31_feature_means_lookup.json")
    print(f"\n2. Update downstream scripts with new file paths")
    print(f"3. Validate compatibility with L25-31 reference data")

if __name__ == '__main__':
    main()
