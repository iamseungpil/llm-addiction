#!/usr/bin/env python3
"""
Classify features as SAFE or RISKY based on effect directions

This script performs post-processing on reparsed results to correctly
classify features based on their directional effects:

SAFE FEATURE:
- Safe context: Increases stop rate (safe_effect > 0)
- Risky context: Decreases high bet rate (risky_effect < 0)

RISKY FEATURE:
- Safe context: Decreases stop rate (safe_effect < 0)
- Risky context: Increases high bet rate (risky_effect > 0)

Usage:
    python classify_safe_risky_features.py --input reparsed_all_features_*.json

Output:
    - classified_features_YYYYMMDD_HHMMSS.json (all features with classification)
    - safe_features_YYYYMMDD_HHMMSS.json (only safe features)
    - risky_features_YYYYMMDD_HHMMSS.json (only risky features)
"""

import json
import argparse
from pathlib import Path
from datetime import datetime


def classify_feature(feature_result: dict) -> dict:
    """
    Classify a feature as SAFE or RISKY based on effect directions.
    
    Returns enhanced result with classification fields:
        - feature_type: 'SAFE', 'RISKY', 'MIXED', or 'NON_CAUSAL'
        - classification_confidence: 'high', 'medium', 'low'
        - classification_reason: explanation
    """
    causality = feature_result.get('causality', {})
    
    safe_effect = causality.get('safe_effect_size', 0)
    risky_effect = causality.get('risky_effect_size', 0)
    safe_p = causality.get('safe_p_value', 1.0)
    risky_p = causality.get('risky_p_value', 1.0)
    
    # Thresholds
    P_THRESHOLD = 0.05
    EFFECT_THRESHOLD = 0.1
    
    # Check statistical significance
    safe_significant = safe_p < P_THRESHOLD
    risky_significant = risky_p < P_THRESHOLD
    
    # Check effect size magnitude
    safe_strong = abs(safe_effect) > EFFECT_THRESHOLD
    risky_strong = abs(risky_effect) > EFFECT_THRESHOLD
    
    # Classification logic
    classification = {
        'feature_type': 'NON_CAUSAL',
        'classification_confidence': 'none',
        'classification_reason': 'no_significant_effects',
        'safe_direction': 'none',
        'risky_direction': 'none'
    }
    
    # Determine directions
    if safe_strong:
        classification['safe_direction'] = 'increases_stop' if safe_effect > 0 else 'decreases_stop'
    
    if risky_strong:
        classification['risky_direction'] = 'increases_risk' if risky_effect > 0 else 'decreases_risk'
    
    # SAFE FEATURE: stop↑ (safe_effect > 0) & risk↓ (risky_effect < 0)
    if (safe_significant and safe_effect > EFFECT_THRESHOLD and
        risky_significant and risky_effect < -EFFECT_THRESHOLD):
        classification['feature_type'] = 'SAFE'
        classification['classification_confidence'] = 'high'
        classification['classification_reason'] = 'both_contexts_promote_safety'
    
    # RISKY FEATURE: stop↓ (safe_effect < 0) & risk↑ (risky_effect > 0)
    elif (safe_significant and safe_effect < -EFFECT_THRESHOLD and
          risky_significant and risky_effect > EFFECT_THRESHOLD):
        classification['feature_type'] = 'RISKY'
        classification['classification_confidence'] = 'high'
        classification['classification_reason'] = 'both_contexts_promote_risk'
    
    # SAFE-leaning: Only safe context shows safety effect
    elif safe_significant and safe_effect > EFFECT_THRESHOLD:
        classification['feature_type'] = 'SAFE'
        classification['classification_confidence'] = 'medium'
        classification['classification_reason'] = 'safe_context_only_promotes_safety'
    
    # RISKY-leaning: Only risky context shows risk effect
    elif risky_significant and risky_effect > EFFECT_THRESHOLD:
        classification['feature_type'] = 'RISKY'
        classification['classification_confidence'] = 'medium'
        classification['classification_reason'] = 'risky_context_only_promotes_risk'
    
    # MIXED: Contradictory effects
    elif ((safe_significant and risky_significant) and
          ((safe_effect > 0 and risky_effect > 0) or
           (safe_effect < 0 and risky_effect < 0))):
        classification['feature_type'] = 'MIXED'
        classification['classification_confidence'] = 'low'
        classification['classification_reason'] = 'contradictory_effects_in_both_contexts'
    
    # CAUSAL but unclear direction
    elif safe_significant or risky_significant:
        classification['feature_type'] = 'CAUSAL_UNCLEAR'
        classification['classification_confidence'] = 'low'
        classification['classification_reason'] = 'significant_but_unclear_direction'
    
    # Add classification to result
    result = feature_result.copy()
    result['classification'] = classification
    
    return result


def classify_all_features(input_file: Path, output_dir: Path):
    """
    Classify all features and save results.
    """
    print("=" * 80)
    print("FEATURE CLASSIFICATION")
    print("=" * 80)
    
    # Load reparsed results
    print(f"\nLoading reparsed results from {input_file}")
    with open(input_file) as f:
        data = json.load(f)
    
    all_results = data.get('results', [])
    print(f"Loaded {len(all_results)} features")
    
    # Classify each feature
    print("\nClassifying features...")
    classified_results = []
    
    classification_counts = {
        'SAFE': 0,
        'RISKY': 0,
        'MIXED': 0,
        'CAUSAL_UNCLEAR': 0,
        'NON_CAUSAL': 0
    }
    
    for result in all_results:
        classified = classify_feature(result)
        classified_results.append(classified)
        
        feature_type = classified['classification']['feature_type']
        classification_counts[feature_type] = classification_counts.get(feature_type, 0) + 1
    
    # Print summary
    print("\n" + "=" * 80)
    print("CLASSIFICATION SUMMARY")
    print("=" * 80)
    
    total = len(classified_results)
    print(f"\nTotal features: {total}")
    print(f"\nClassification breakdown:")
    for ftype, count in sorted(classification_counts.items(), key=lambda x: -x[1]):
        print(f"  {ftype:20s}: {count:5d} ({100*count/total:5.1f}%)")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # All features with classification
    all_file = output_dir / f'classified_features_{timestamp}.json'
    print(f"\nSaving all classified features to {all_file}")
    with open(all_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'total_features': total,
            'classification_counts': classification_counts,
            'results': classified_results
        }, f, indent=2)
    
    # Safe features only
    safe_features = [r for r in classified_results 
                     if r['classification']['feature_type'] == 'SAFE']
    safe_file = output_dir / f'safe_features_{timestamp}.json'
    print(f"Saving safe features to {safe_file}")
    with open(safe_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'count': len(safe_features),
            'features': safe_features
        }, f, indent=2)
    
    # Risky features only
    risky_features = [r for r in classified_results 
                      if r['classification']['feature_type'] == 'RISKY']
    risky_file = output_dir / f'risky_features_{timestamp}.json'
    print(f"Saving risky features to {risky_file}")
    with open(risky_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'count': len(risky_features),
            'features': risky_features
        }, f, indent=2)
    
    print("\n" + "=" * 80)
    print("✅ Classification complete!")
    print("=" * 80)
    
    print(f"\nOutput files:")
    print(f"  All features: {all_file}")
    print(f"  Safe features: {safe_file} ({len(safe_features)} features)")
    print(f"  Risky features: {risky_file} ({len(risky_features)} features)")
    
    return all_file, safe_file, risky_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify features as SAFE or RISKY')
    parser.add_argument('--input', type=str, help='Reparsed all features JSON file')
    parser.add_argument('--output_dir', type=str,
                       default='/data/llm_addiction/experiment_2_multilayer_patching/reparsed',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Find latest reparsed file if not specified
    if not args.input:
        output_dir = Path(args.output_dir)
        reparsed_files = sorted(output_dir.glob('reparsed_all_features_*.json'))
        if not reparsed_files:
            print("❌ No reparsed feature files found. Run analyze_reparsed_results.py first.")
            exit(1)
        args.input = str(reparsed_files[-1])
        print(f"Using latest reparsed file: {args.input}")
    
    input_file = Path(args.input)
    output_dir = Path(args.output_dir)
    
    # Run classification
    classify_all_features(input_file, output_dir)
