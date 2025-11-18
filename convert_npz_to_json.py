#!/usr/bin/env python3
"""
Convert new NPZ format to old JSON format for Experiment 2 compatibility

Input:  /data/llm_addiction/results/L1_31_GLOBAL_FDR_features_20251110_214621.npz
Output: /data/llm_addiction/experiment_1_L1_31_extraction/L1_31_features_CONVERTED_20251111.json

This converter ensures compatibility with experiment_2_L1_31_top300.py
which expects the old JSON format.
"""
import numpy as np
import json
from datetime import datetime
from pathlib import Path

def convert_npz_to_json():
    """Convert NPZ format to JSON format"""

    # Input NPZ file
    npz_file = '/data/llm_addiction/results/L1_31_GLOBAL_FDR_features_20251110_214621.npz'
    print(f"Loading NPZ file: {npz_file}")
    npz = np.load(npz_file)

    # Create JSON structure
    output = {
        'timestamp': datetime.now().isoformat(),
        'source_file': npz_file,
        'conversion_date': '2025-11-11',
        'conversion_script': __file__,
        'description': 'Converted from Global FDR NPZ format to JSON format for Experiment 2',
        'total_layers': 31,
        'layer_results': {}
    }

    total_features = 0

    print("\nProcessing layers:")
    print("-" * 60)

    # Process each layer
    for layer in range(1, 32):
        key_prefix = f'layer_{layer}_'
        indices_key = key_prefix + 'indices'

        if indices_key not in npz.files:
            print(f"Layer {layer:2d}: Not found (skipping)")
            continue

        # Extract arrays
        indices = npz[indices_key]
        cohen_d = npz[key_prefix + 'cohen_d']
        p_values = npz[key_prefix + 'p_values']
        bankrupt_mean = npz[key_prefix + 'bankrupt_mean']
        safe_mean = npz[key_prefix + 'safe_mean']
        bankrupt_std = npz[key_prefix + 'bankrupt_std']
        safe_std = npz[key_prefix + 'safe_std']

        # Create feature list
        features = []
        for i in range(len(indices)):
            features.append({
                'feature_idx': int(indices[i]),
                'cohen_d': float(cohen_d[i]),
                'p_value': float(p_values[i]),
                'bankrupt_mean': float(bankrupt_mean[i]),
                'safe_mean': float(safe_mean[i]),
                'bankrupt_std': float(bankrupt_std[i]),
                'safe_std': float(safe_std[i])
            })

        # Sort by |Cohen's d| descending (already sorted, but ensuring)
        features.sort(key=lambda x: abs(x['cohen_d']), reverse=True)

        # Add to output
        output['layer_results'][str(layer)] = {
            'layer': layer,
            'n_features': 32768,  # Total possible features in SAE
            'n_significant': len(features),
            'significant_features': features
        }

        total_features += len(features)

        # Print progress
        top_cohen_d = features[0]['cohen_d'] if features else 0
        print(f"Layer {layer:2d}: {len(features):4d} features (top |d|={abs(top_cohen_d):.3f})")

    output['total_significant_features'] = total_features

    # Create output directory if needed
    output_dir = Path('/data/llm_addiction/experiment_1_L1_31_extraction')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    output_file = output_dir / 'L1_31_features_CONVERTED_20251111.json'
    print(f"\nSaving JSON file: {output_file}")

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    # Get file size
    file_size = output_file.stat().st_size
    size_mb = file_size / (1024 * 1024)

    print("-" * 60)
    print(f"\n✅ Conversion complete!")
    print(f"   Total features: {total_features:,}")
    print(f"   Total layers: {len(output['layer_results'])}")
    print(f"   Output file: {output_file}")
    print(f"   File size: {size_mb:.2f} MB")

    # Verification
    print(f"\n🔍 Verification:")
    with open(output_file, 'r') as f:
        verify = json.load(f)
    print(f"   ✓ JSON is valid and loadable")
    print(f"   ✓ Contains {verify['total_significant_features']} features")
    print(f"   ✓ Contains {len(verify['layer_results'])} layers")

    # Sample data
    print(f"\n📊 Sample Layer 1 data:")
    layer_1 = verify['layer_results']['1']
    print(f"   Features: {layer_1['n_significant']}")
    if layer_1['significant_features']:
        sample = layer_1['significant_features'][0]
        print(f"   Top feature: idx={sample['feature_idx']}, d={sample['cohen_d']:.3f}, p={sample['p_value']:.2e}")

    return output_file

if __name__ == '__main__':
    convert_npz_to_json()
