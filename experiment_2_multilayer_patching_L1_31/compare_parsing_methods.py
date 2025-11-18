#!/usr/bin/env python3
"""
Compare original vs reparsed causal feature results

This script:
1. Loads original checkpoint results
2. Loads reparsed causal feature results
3. Compares which features are identified as causal
4. Generates detailed comparison report

Usage:
    python compare_parsing_methods.py --reparsed reparsed_causal_features_YYYYMMDD_HHMMSS.json

Output:
    - parsing_comparison_report_YYYYMMDD_HHMMSS.json
    - parsing_comparison_report_YYYYMMDD_HHMMSS.txt (human-readable)
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict


def load_original_results(checkpoint_dir: Path):
    """
    Load original causal feature results from checkpoints
    """
    print("Loading original checkpoint results...")

    # Find all checkpoints
    checkpoints = sorted(checkpoint_dir.glob('checkpoint_L*.json'))

    # Get latest checkpoint per layer
    layer_checkpoints = {}
    for ckpt in checkpoints:
        with open(ckpt) as f:
            data = json.load(f)

        layer_range = data.get('layer_range', '')
        if layer_range not in layer_checkpoints:
            layer_checkpoints[layer_range] = ckpt
        else:
            # Compare timestamps
            current_ts = ckpt.stem.split('_')[-2] + '_' + ckpt.stem.split('_')[-1]
            existing_ts = layer_checkpoints[layer_range].stem.split('_')[-2] + '_' + layer_checkpoints[layer_range].stem.split('_')[-1]

            if current_ts > existing_ts:
                layer_checkpoints[layer_range] = ckpt

    # Load all results
    all_results = []
    for ckpt in layer_checkpoints.values():
        with open(ckpt) as f:
            data = json.load(f)
            all_results.extend(data.get('results', []))

    print(f"Loaded {len(all_results)} features from {len(layer_checkpoints)} checkpoints")

    return all_results


def compare_results(original_results, reparsed_results):
    """
    Compare original vs reparsed causal feature identification
    """
    print("\n" + "=" * 80)
    print("COMPARING ORIGINAL VS REPARSED RESULTS")
    print("=" * 80)

    # Create lookup dictionaries
    original_by_feature = {}
    for result in original_results:
        feature = result.get('feature')
        if feature:
            original_by_feature[feature] = result

    reparsed_by_feature = {}
    for result in reparsed_results:
        feature = result.get('feature')
        if feature:
            reparsed_by_feature[feature] = result

    # Find common features
    common_features = set(original_by_feature.keys()) & set(reparsed_by_feature.keys())
    print(f"\nFeatures in both datasets: {len(common_features)}")

    # Categorize features
    categories = {
        'both_causal': [],
        'only_original_causal': [],
        'only_reparsed_causal': [],
        'neither_causal': [],
        'agreement': 0,
        'disagreement': 0
    }

    for feature in common_features:
        original = original_by_feature[feature]
        reparsed = reparsed_by_feature[feature]

        original_causal = (
            original.get('causality', {}).get('is_causal_safe', False) or
            original.get('causality', {}).get('is_causal_risky', False)
        )

        reparsed_causal = (
            reparsed.get('causality', {}).get('is_causal_safe', False) or
            reparsed.get('causality', {}).get('is_causal_risky', False)
        )

        if original_causal and reparsed_causal:
            categories['both_causal'].append({
                'feature': feature,
                'original': original,
                'reparsed': reparsed
            })
            categories['agreement'] += 1
        elif original_causal and not reparsed_causal:
            categories['only_original_causal'].append({
                'feature': feature,
                'original': original,
                'reparsed': reparsed
            })
            categories['disagreement'] += 1
        elif not original_causal and reparsed_causal:
            categories['only_reparsed_causal'].append({
                'feature': feature,
                'original': original,
                'reparsed': reparsed
            })
            categories['disagreement'] += 1
        else:
            categories['neither_causal'].append({
                'feature': feature,
                'original': original,
                'reparsed': reparsed
            })
            categories['agreement'] += 1

    # Calculate statistics
    total = len(common_features)
    agreement_rate = 100 * categories['agreement'] / total if total > 0 else 0

    stats = {
        'total_features': total,
        'both_causal': len(categories['both_causal']),
        'only_original_causal': len(categories['only_original_causal']),
        'only_reparsed_causal': len(categories['only_reparsed_causal']),
        'neither_causal': len(categories['neither_causal']),
        'agreement': categories['agreement'],
        'disagreement': categories['disagreement'],
        'agreement_rate': agreement_rate
    }

    return categories, stats


def generate_report(categories, stats, output_dir: Path):
    """
    Generate detailed comparison report
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # JSON report
    json_report = output_dir / f'parsing_comparison_report_{timestamp}.json'

    report_data = {
        'timestamp': timestamp,
        'statistics': stats,
        'categories': {
            'both_causal': categories['both_causal'][:50],  # Top 50
            'only_original_causal': categories['only_original_causal'][:50],
            'only_reparsed_causal': categories['only_reparsed_causal'][:50]
        }
    }

    with open(json_report, 'w') as f:
        json.dump(report_data, f, indent=2)

    # Human-readable text report
    txt_report = output_dir / f'parsing_comparison_report_{timestamp}.txt'

    with open(txt_report, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PARSING METHOD COMPARISON REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Generated: {timestamp}\n\n")

        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total features compared: {stats['total_features']}\n")
        f.write(f"Agreement rate: {stats['agreement_rate']:.1f}%\n\n")

        f.write("CAUSAL FEATURE IDENTIFICATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Both methods identified as causal: {stats['both_causal']}\n")
        f.write(f"Only original identified as causal: {stats['only_original_causal']}\n")
        f.write(f"Only reparsed identified as causal: {stats['only_reparsed_causal']}\n")
        f.write(f"Neither method identified as causal: {stats['neither_causal']}\n\n")

        # Detailed breakdown of disagreements
        f.write("=" * 80 + "\n")
        f.write("DISAGREEMENTS: Original Causal, Reparsed Non-Causal\n")
        f.write("=" * 80 + "\n\n")

        for i, item in enumerate(categories['only_original_causal'][:20], 1):
            f.write(f"{i}. {item['feature']}\n")
            orig_c = item['original']['causality']
            repr_c = item['reparsed']['causality']
            f.write(f"   Original: safe_p={orig_c.get('safe_p_value', 1.0):.4f}, ")
            f.write(f"risky_p={orig_c.get('risky_p_value', 1.0):.4f}\n")
            f.write(f"   Reparsed: safe_p={repr_c.get('safe_p_value', 1.0):.4f}, ")
            f.write(f"risky_p={repr_c.get('risky_p_value', 1.0):.4f}\n\n")

        f.write("=" * 80 + "\n")
        f.write("DISAGREEMENTS: Original Non-Causal, Reparsed Causal\n")
        f.write("=" * 80 + "\n\n")

        for i, item in enumerate(categories['only_reparsed_causal'][:20], 1):
            f.write(f"{i}. {item['feature']}\n")
            orig_c = item['original']['causality']
            repr_c = item['reparsed']['causality']
            f.write(f"   Original: safe_p={orig_c.get('safe_p_value', 1.0):.4f}, ")
            f.write(f"risky_p={orig_c.get('risky_p_value', 1.0):.4f}\n")
            f.write(f"   Reparsed: safe_p={repr_c.get('safe_p_value', 1.0):.4f}, ")
            f.write(f"risky_p={repr_c.get('risky_p_value', 1.0):.4f}\n\n")

        f.write("=" * 80 + "\n")
        f.write("INTERPRETATION\n")
        f.write("=" * 80 + "\n\n")

        if stats['agreement_rate'] >= 90:
            f.write("✅ HIGH AGREEMENT (>90%)\n")
            f.write("The two parsing methods produce highly consistent results.\n")
            f.write("Parsing differences have minimal impact on causal feature identification.\n")
        elif stats['agreement_rate'] >= 70:
            f.write("⚠️  MODERATE AGREEMENT (70-90%)\n")
            f.write("Some differences exist between parsing methods.\n")
            f.write("Both results should be reported to demonstrate robustness.\n")
        else:
            f.write("❌ LOW AGREEMENT (<70%)\n")
            f.write("Significant differences between parsing methods.\n")
            f.write("Parsing method choice substantially affects conclusions.\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"\n✅ Reports generated:")
    print(f"   JSON: {json_report}")
    print(f"   Text: {txt_report}")

    return json_report, txt_report


def main():
    parser = argparse.ArgumentParser(description='Compare original vs reparsed results')
    parser.add_argument('--reparsed', type=str, help='Reparsed causal features JSON file')
    parser.add_argument('--checkpoint_dir', type=str,
                       default='/data/llm_addiction/experiment_2_multilayer_patching',
                       help='Original checkpoint directory')
    parser.add_argument('--output_dir', type=str,
                       default='/data/llm_addiction/experiment_2_multilayer_patching/reparsed',
                       help='Output directory')

    args = parser.parse_args()

    # Find latest reparsed file if not specified
    if not args.reparsed:
        output_dir = Path(args.output_dir)
        reparsed_files = sorted(output_dir.glob('reparsed_causal_features_*.json'))
        if not reparsed_files:
            print("❌ No reparsed causal feature files found. Run analyze_reparsed_results.py first.")
            exit(1)
        args.reparsed = str(reparsed_files[-1])
        print(f"Using latest reparsed file: {args.reparsed}")

    # Load data
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)

    original_results = load_original_results(checkpoint_dir)

    print(f"\nLoading reparsed results from {args.reparsed}")
    with open(args.reparsed) as f:
        reparsed_data = json.load(f)
    reparsed_results = reparsed_data.get('causal_features', [])

    # Also load all reparsed results for complete comparison
    all_reparsed_file = Path(args.reparsed).parent / args.reparsed.replace('causal_features', 'all_features')
    if all_reparsed_file.exists():
        with open(all_reparsed_file) as f:
            all_reparsed_data = json.load(f)
        all_reparsed_results = all_reparsed_data.get('results', [])
    else:
        all_reparsed_results = reparsed_results

    # Compare
    categories, stats = compare_results(original_results, all_reparsed_results)

    # Print summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    print(f"\nTotal features compared: {stats['total_features']}")
    print(f"Agreement rate: {stats['agreement_rate']:.1f}%\n")

    print("Causal feature identification:")
    print(f"  Both methods: {stats['both_causal']}")
    print(f"  Only original: {stats['only_original_causal']}")
    print(f"  Only reparsed: {stats['only_reparsed_causal']}")
    print(f"  Neither: {stats['neither_causal']}")

    # Generate report
    json_report, txt_report = generate_report(categories, stats, output_dir)

    print("\n" + "=" * 80)
    print("✅ Comparison complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
