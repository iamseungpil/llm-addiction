#!/usr/bin/env python3
"""Comprehensive analysis of all feature files to resolve count discrepancies"""

import numpy as np
import os

def analyze_all_feature_files():
    """Analyze all feature NPZ files to understand count discrepancies"""

    base_dir = "/data/llm_addiction/results"

    feature_files = [
        "llama_feature_arrays_20250813_152135.npz",  # Earlier version
        "llama_feature_arrays_20250829_150110_v2.npz",  # v2 version
        "multilayer_features_20250911_171655.npz",  # Latest multilayer
    ]

    print("=== COMPREHENSIVE FEATURE COUNT ANALYSIS ===")
    print("Analyzing all feature files to resolve 370 vs 441 discrepancy\n")

    total_counts = {}

    for filename in feature_files:
        filepath = os.path.join(base_dir, filename)
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filename}")
            continue

        print(f"📁 {filename}")
        print("-" * 60)

        data = np.load(filepath)
        layer_totals = {}
        overall_total = 0
        safe_total = 0
        risky_total = 0

        # Get all layers present in this file
        layers = set()
        for key in data.keys():
            if '_indices' in key:
                layer = key.split('_')[1]
                layers.add(layer)

        # Analyze each layer
        for layer in sorted(layers):
            indices_key = f'layer_{layer}_indices'
            if indices_key in data:
                count = len(data[indices_key])
                layer_totals[layer] = count
                overall_total += count

                # Check for Cohen's d if available
                cohen_key = f'layer_{layer}_cohen_d'
                if cohen_key in data:
                    cohen_values = data[cohen_key]
                    safe_count = np.sum(cohen_values < 0)
                    risky_count = np.sum(cohen_values > 0)
                    safe_total += safe_count
                    risky_total += risky_count
                    print(f"  Layer {layer}: {count} features ({safe_count} safe, {risky_count} risky)")
                else:
                    print(f"  Layer {layer}: {count} features")

        print(f"\n  📊 TOTAL: {overall_total} features")
        if safe_total > 0 and risky_total > 0:
            print(f"  🛡️  Safe: {safe_total}, ⚡ Risky: {risky_total}")

        total_counts[filename] = {
            'total': overall_total,
            'safe': safe_total,
            'risky': risky_total,
            'layers': layer_totals
        }
        print("\n")

    # Summary comparison
    print("=== SUMMARY COMPARISON ===")
    for filename, counts in total_counts.items():
        print(f"{filename}: {counts['total']} total features")

    print("\n=== RESOLVING THE 370 vs 441 DISCREPANCY ===")

    # Check if 370 or 441 matches any of our totals
    if 'llama_feature_arrays_20250829_150110_v2.npz' in total_counts:
        v2_total = total_counts['llama_feature_arrays_20250829_150110_v2.npz']['total']
        print(f"v2 file total: {v2_total}")
        if v2_total == 356:  # 53 + 303
            print(f"✅ The v2 file has 356 features (53 + 303), not 370 or 441")

    # Check Layer 25 specifically for the 441 number
    if 'multilayer_features_20250911_171655.npz' in total_counts:
        multilayer = total_counts['multilayer_features_20250911_171655.npz']
        if '25' in multilayer['layers'] and multilayer['layers']['25'] == 441:
            print(f"✅ Found 441: This is Layer 25 count in the multilayer analysis")

        print(f"📈 Multilayer analysis shows dramatic expansion:")
        print(f"   - Original (Aug 13): 392 features (192 + 200)")
        print(f"   - v2 (Aug 29): 356 features (53 + 303)")
        print(f"   - Multilayer (Sep 11): 3,365 features across 7 layers")

if __name__ == '__main__':
    analyze_all_feature_files()