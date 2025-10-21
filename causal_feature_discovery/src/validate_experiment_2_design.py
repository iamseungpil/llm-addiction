#!/usr/bin/env python3
"""
Experiment 2 Design Validation
Tests the new 3-condition design on a small subset of features to validate methodology
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
import torch

# Set GPU for validation
os.environ['CUDA_VISIBLE_DEVICES'] = '6'  # Use GPU 6 for validation

sys.path.append('/home/ubuntu/llm_addiction/causal_feature_discovery/src')
from experiment_2_multilayer_population_mean import MultiLayerPopulationMeanExperiment

def run_validation():
    """Run validation on top 10 features"""
    print("="*80)
    print("EXPERIMENT 2 DESIGN VALIDATION")
    print("="*80)
    
    # Initialize experiment
    experiment = MultiLayerPopulationMeanExperiment()
    
    # Reduce trials for faster validation
    experiment.n_trials = 10  # Quick validation
    
    # Load models
    experiment.load_models()
    
    # Load features
    features = experiment.load_features()
    
    # Test top 10 features (highest Cohen's d)
    test_features = features[:10]
    
    print(f"\nValidating on top {len(test_features)} features:")
    for i, f in enumerate(test_features):
        print(f"  {i+1}. L{f['layer']}-{f['feature_id']}: Cohen's d={f['cohen_d']:.3f}")
    
    print(f"\nQuick validation settings:")
    print(f"  Trials per condition: {experiment.n_trials}")
    print(f"  Conditions: {experiment.conditions}")
    print(f"  Expected runtime: ~15 minutes")
    
    # Run experiment on subset
    results, causal_bet, causal_stop = experiment.run_experiment(0, len(test_features))
    
    # Validation analysis
    print("\n" + "="*50)
    print("VALIDATION RESULTS")
    print("="*50)
    
    # Check if 3-condition design is working
    features_with_results = {}
    for result in results:
        key = f"L{result['layer']}-{result['feature_id']}"
        if key not in features_with_results:
            features_with_results[key] = []
        features_with_results[key].append(result)
    
    print(f"Features tested: {len(features_with_results)}")
    print(f"Causal features (betting): {len(causal_bet)}")
    print(f"Causal features (stopping): {len(causal_stop)}")
    
    # Analyze condition effects
    print("\nCondition Effect Analysis:")
    for feature_key, feature_results in features_with_results.items():
        risky_results = [r for r in feature_results if r['prompt_type'] == 'risky']
        
        if len(risky_results) == 3:  # All 3 conditions present
            safe_bet = next(r['avg_bet'] for r in risky_results if r['condition'] == 'safe_mean')
            baseline_bet = next(r['avg_bet'] for r in risky_results if r['condition'] == 'baseline')
            risky_bet = next(r['avg_bet'] for r in risky_results if r['condition'] == 'risky_mean')
            
            print(f"  {feature_key}: safe=${safe_bet:.1f} < baseline=${baseline_bet:.1f} < risky=${risky_bet:.1f}")
            
            # Check if ordering is correct (for risk-increasing features)
            expected_order = safe_bet <= baseline_bet <= risky_bet
            print(f"    Expected ordering: {'✅' if expected_order else '❌'}")
    
    # Feature quality check
    print(f"\nFeature Quality Check:")
    if test_features:
        cohen_ds = [f['cohen_d'] for f in test_features]
        print(f"  Cohen's d range: {min(cohen_ds):.3f} to {max(cohen_ds):.3f}")
        print(f"  Average |Cohen's d|: {np.mean([abs(d) for d in cohen_ds]):.3f}")
        
        p_values = [f['p_value'] for f in test_features]
        print(f"  P-value range: {min(p_values):.2e} to {max(p_values):.2e}")
        print(f"  All p < 0.001: {'✅' if all(p < 0.001 for p in p_values) else '❌'}")
    
    print("\n" + "="*50)
    print("VALIDATION COMPLETE")
    print("="*50)
    
    if len(causal_bet) > 0 or len(causal_stop) > 0:
        print("✅ Design validation successful!")
        print("   3-condition methodology is detecting causal features")
        print("   Ready for full-scale experiment")
    else:
        print("⚠️  No causal features detected in validation")
        print("   May need to adjust causality criteria")
        print("   Review results before full-scale run")
    
    return results, causal_bet, causal_stop

if __name__ == "__main__":
    validation_results = run_validation()