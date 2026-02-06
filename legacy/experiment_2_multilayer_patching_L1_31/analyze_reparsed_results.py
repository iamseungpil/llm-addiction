#!/usr/bin/env python3
"""
Analyze reparsed response logs to identify causal features

This script:
1. Loads reparsed response data
2. Recalculates causal effects using improved parsing
3. Identifies causal features with statistical tests
4. Compares with original results

Usage:
    python analyze_reparsed_results.py --input reparsed_responses_YYYYMMDD_HHMMSS.json

Output:
    - reparsed_causal_features_YYYYMMDD_HHMMSS.json
    - reparsed_vs_original_comparison_YYYYMMDD_HHMMSS.json
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np
from scipy import stats
from tqdm import tqdm


def calculate_cohen_d(group1, group2):
    """Calculate Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return 0.0

    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def analyze_causality(trial_data: dict) -> dict:
    """
    Analyze if feature shows causal effect using statistical tests

    trial_data format: {condition_name: [bet_amounts]} where 0 = stop
    """
    causality_results = {
        'is_causal_safe': False,
        'is_causal_risky': False,
        'safe_effect_size': 0,
        'risky_effect_size': 0,
        'safe_p_value': 1.0,
        'risky_p_value': 1.0,
        'interpretation': 'no_effect'
    }

    # Test 1: Safe prompt causality (stop rate changes)
    try:
        safe_baseline_bets = trial_data.get('safe_baseline', [])
        safe_with_safe_bets = trial_data.get('safe_with_safe_patch', [])
        safe_with_risky_bets = trial_data.get('safe_with_risky_patch', [])

        if len(safe_baseline_bets) >= 10 and len(safe_with_safe_bets) >= 10 and len(safe_with_risky_bets) >= 10:
            # Convert to stop rates (bet=0 means stop)
            baseline_stop_rate = sum(1 for bet in safe_baseline_bets if bet == 0) / len(safe_baseline_bets)
            safe_patch_stop_rate = sum(1 for bet in safe_with_safe_bets if bet == 0) / len(safe_with_safe_bets)
            risky_patch_stop_rate = sum(1 for bet in safe_with_risky_bets if bet == 0) / len(safe_with_risky_bets)

            # Test if safe patching increases stop rate, risky patching decreases it
            safe_effect = safe_patch_stop_rate - baseline_stop_rate
            risky_effect = risky_patch_stop_rate - baseline_stop_rate

            # Statistical tests
            # Chi-square test for safe patch
            safe_stops = sum(1 for bet in safe_with_safe_bets if bet == 0)
            safe_bets = len(safe_with_safe_bets) - safe_stops
            baseline_stops = sum(1 for bet in safe_baseline_bets if bet == 0)
            baseline_bets = len(safe_baseline_bets) - baseline_stops

            contingency_safe = [[safe_stops, safe_bets], [baseline_stops, baseline_bets]]
            if all(sum(row) > 0 for row in contingency_safe):
                chi2_safe, p_safe = stats.chi2_contingency(contingency_safe)[:2]
            else:
                p_safe = 1.0

            # Chi-square test for risky patch
            risky_stops = sum(1 for bet in safe_with_risky_bets if bet == 0)
            risky_bets = len(safe_with_risky_bets) - risky_stops
            contingency_risky = [[risky_stops, risky_bets], [baseline_stops, baseline_bets]]

            if all(sum(row) > 0 for row in contingency_risky):
                chi2_risky, p_risky = stats.chi2_contingency(contingency_risky)[:2]
            else:
                p_risky = 1.0

            # Determine if safe context shows causal effect
            # Store effect size with direction preserved
            causality_results['safe_effect_size'] = safe_effect
            causality_results['safe_p_value'] = p_safe

            # Causal if statistically significant with sufficient effect size (either direction)
            if p_safe < 0.05 and abs(safe_effect) > 0.1:
                causality_results['is_causal_safe'] = True

    except Exception as e:
        print(f"Warning: Safe context test failed: {e}")

    # Test 2: Risky prompt causality (high bet rate changes)
    try:
        risky_baseline_bets = trial_data.get('risky_baseline', [])
        risky_with_safe_bets = trial_data.get('risky_with_safe_patch', [])
        risky_with_risky_bets = trial_data.get('risky_with_risky_patch', [])

        if len(risky_baseline_bets) >= 10 and len(risky_with_safe_bets) >= 10 and len(risky_with_risky_bets) >= 10:
            # High bet threshold (>= $20)
            HIGH_BET_THRESHOLD = 20

            baseline_high_rate = sum(1 for bet in risky_baseline_bets if bet >= HIGH_BET_THRESHOLD) / len(risky_baseline_bets)
            safe_patch_high_rate = sum(1 for bet in risky_with_safe_bets if bet >= HIGH_BET_THRESHOLD) / len(risky_with_safe_bets)
            risky_patch_high_rate = sum(1 for bet in risky_with_risky_bets if bet >= HIGH_BET_THRESHOLD) / len(risky_with_risky_bets)

            # Test if risky patching increases high bet rate
            risky_effect = risky_patch_high_rate - baseline_high_rate

            # Chi-square test
            risky_high = sum(1 for bet in risky_with_risky_bets if bet >= HIGH_BET_THRESHOLD)
            risky_low = len(risky_with_risky_bets) - risky_high
            baseline_high = sum(1 for bet in risky_baseline_bets if bet >= HIGH_BET_THRESHOLD)
            baseline_low = len(risky_baseline_bets) - baseline_high

            contingency = [[risky_high, risky_low], [baseline_high, baseline_low]]
            if all(sum(row) > 0 for row in contingency):
                chi2, p_risky = stats.chi2_contingency(contingency)[:2]
            else:
                p_risky = 1.0

            # Store effect size with direction preserved
            causality_results['risky_effect_size'] = risky_effect
            causality_results['risky_p_value'] = p_risky

            # Causal if statistically significant with sufficient effect size (either direction)
            if p_risky < 0.05 and abs(risky_effect) > 0.1:
                causality_results['is_causal_risky'] = True

    except Exception as e:
        print(f"Warning: Risky context test failed: {e}")

    # Determine interpretation
    if causality_results['is_causal_safe'] and causality_results['is_causal_risky']:
        causality_results['interpretation'] = 'both_contexts_causal'
    elif causality_results['is_causal_safe']:
        causality_results['interpretation'] = 'safe_context_causal'
    elif causality_results['is_causal_risky']:
        causality_results['interpretation'] = 'risky_context_causal'
    else:
        causality_results['interpretation'] = 'no_causal_effect'

    return causality_results


def analyze_reparsed_data(reparsed_file: Path, output_dir: Path, original_results_dir: Path = None):
    """
    Analyze reparsed data to identify causal features
    """
    print("=" * 80)
    print("CAUSAL FEATURE ANALYSIS (REPARSED DATA)")
    print("=" * 80)

    # Load reparsed data
    print(f"\nLoading reparsed data from {reparsed_file}")
    with open(reparsed_file) as f:
        reparsed_data = json.load(f)

    print(f"Loaded data for {len(reparsed_data)} features")

    # Analyze each feature
    results = []
    causal_features = []

    print("\nAnalyzing causal effects...")
    for feature_name, conditions_data in tqdm(reparsed_data.items(), desc="Processing features"):
        # Extract layer and feature_id
        if '-' in feature_name:
            layer_str, feature_id_str = feature_name.split('-')
            layer = int(layer_str.replace('L', ''))
            feature_id = int(feature_id_str)
        else:
            continue

        # Organize trial data by condition, excluding invalid reparsed entries
        trial_data = {}
        for condition, trials in conditions_data.items():
            trial_data[condition] = [
                t['reparsed_bet']
                for t in trials
                if t.get('reparsed_valid', False)
            ]

        # Analyze causality
        causality = analyze_causality(trial_data)

        # Store result
        result = {
            'feature': feature_name,
            'layer': layer,
            'feature_id': feature_id,
            'causality': causality,
            'trial_counts': {
                condition: len(trials)
                for condition, trials in trial_data.items()
            }
        }

        results.append(result)

        # Identify causal features
        if causality['is_causal_safe'] or causality['is_causal_risky']:
            causal_features.append(result)

    # Sort by effect size
    causal_features.sort(
        key=lambda x: abs(x['causality'].get('safe_effect_size', 0)) + abs(x['causality'].get('risky_effect_size', 0)),
        reverse=True
    )

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save all results
    all_results_file = output_dir / f'reparsed_all_features_{timestamp}.json'
    print(f"\nSaving all results to {all_results_file}")

    with open(all_results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'total_features': len(results),
            'causal_features': len(causal_features),
            'results': results
        }, f, indent=2)

    # Save causal features only
    causal_file = output_dir / f'reparsed_causal_features_{timestamp}.json'
    print(f"Saving causal features to {causal_file}")

    with open(causal_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'total_causal_features': len(causal_features),
            'causal_features': causal_features
        }, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)

    print(f"\nTotal features analyzed: {len(results)}")
    print(f"Causal features identified: {len(causal_features)}")
    print(f"Causal rate: {100 * len(causal_features) / len(results):.1f}%")

    # Breakdown by context
    safe_only = sum(1 for f in causal_features if f['causality']['is_causal_safe'] and not f['causality']['is_causal_risky'])
    risky_only = sum(1 for f in causal_features if f['causality']['is_causal_risky'] and not f['causality']['is_causal_safe'])
    both = sum(1 for f in causal_features if f['causality']['is_causal_safe'] and f['causality']['is_causal_risky'])

    print(f"\nCausality breakdown:")
    print(f"  Safe context only: {safe_only}")
    print(f"  Risky context only: {risky_only}")
    print(f"  Both contexts: {both}")

    # Top causal features
    print(f"\nTop 10 causal features:")
    for i, feature in enumerate(causal_features[:10], 1):
        causality = feature['causality']
        print(f"  {i}. {feature['feature']}")
        print(f"     Safe effect: {causality['safe_effect_size']:.3f} (p={causality['safe_p_value']:.4f})")
        print(f"     Risky effect: {causality['risky_effect_size']:.3f} (p={causality['risky_p_value']:.4f})")

    print(f"\n✅ Analysis complete!")
    print(f"   All results: {all_results_file}")
    print(f"   Causal features: {causal_file}")
    print("=" * 80)

    return all_results_file, causal_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze reparsed response logs')
    parser.add_argument('--input', type=str, help='Reparsed responses JSON file')
    parser.add_argument('--output_dir', type=str,
                       default='/data/llm_addiction/experiment_2_multilayer_patching/reparsed',
                       help='Output directory')

    args = parser.parse_args()

    # Find latest reparsed file if not specified
    if not args.input:
        output_dir = Path(args.output_dir)
        reparsed_files = sorted(output_dir.glob('reparsed_responses_*.json'))
        if not reparsed_files:
            print("❌ No reparsed files found. Run reparse_response_logs.py first.")
            exit(1)
        args.input = str(reparsed_files[-1])
        print(f"Using latest reparsed file: {args.input}")

    # Run analysis
    reparsed_file = Path(args.input)
    output_dir = Path(args.output_dir)

    analyze_reparsed_data(reparsed_file, output_dir)

    print(f"\n📊 Next step: Run compare_parsing_methods.py to compare with original results")
