#!/usr/bin/env python3
"""
Quick timing test for single feature with optimized parameters
"""

import os
import sys
import time
import numpy as np

# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

sys.path.append('/home/ubuntu/llm_addiction/causal_feature_discovery/src')
from experiment_2_final_correct import FinalCorrectPopulationMeanExperiment

def test_feature_timing():
    print("="*60)
    print("SINGLE FEATURE TIMING TEST (15 trials per condition)")
    print("="*60)
    
    # Initialize experiment
    experiment = FinalCorrectPopulationMeanExperiment(gpu_id=6, results_dir='/data/llm_addiction/results')
    
    # Load models
    print("Loading models...")
    start_time = time.time()
    experiment.load_models()
    load_time = time.time() - start_time
    print(f"Model loading time: {load_time:.1f}s")
    
    # Load a single high-effect feature for testing
    features_data = np.load('/data/llm_addiction/results/multilayer_features_20250911_171655.npz')
    
    # Get first feature from layer 30 (typically high effect)
    layer_30_indices = features_data['layer_30_indices']
    layer_30_cohen_d = features_data['layer_30_cohen_d']
    layer_30_safe_means = features_data['layer_30_safe_means']
    layer_30_bankrupt_means = features_data['layer_30_bankrupt_means']
    
    # Test with the highest Cohen's d feature from layer 30
    max_idx = np.argmax(np.abs(layer_30_cohen_d))
    
    test_feature = {
        'layer': 30,
        'feature_id': int(layer_30_indices[max_idx]),
        'cohen_d': float(layer_30_cohen_d[max_idx]),
        'safe_mean': float(layer_30_safe_means[max_idx]),
        'bankrupt_mean': float(layer_30_bankrupt_means[max_idx])
    }
    
    print(f"\nTesting feature: L{test_feature['layer']}-{test_feature['feature_id']}")
    print(f"Cohen's d: {test_feature['cohen_d']:.3f}")
    print(f"Trials per condition: {experiment.n_trials}")
    print(f"Total trials per feature: {6 * experiment.n_trials}")
    
    # Time the feature test
    print("\nStarting feature test...")
    start_time = time.time()
    
    result = experiment.test_single_feature(test_feature)
    
    end_time = time.time()
    test_time = end_time - start_time
    
    print(f"\nFeature test completed!")
    print(f"Time taken: {test_time:.1f}s ({test_time/60:.1f} minutes)")
    print(f"Time per trial: {test_time/(6*experiment.n_trials):.1f}s")
    
    # Extrapolate to full experiment
    total_features = 3365
    estimated_total = test_time * total_features
    
    print(f"\nExtrapolation to full experiment:")
    print(f"Total features: {total_features}")
    print(f"Estimated total time: {estimated_total/3600:.1f} hours ({estimated_total/(3600*24):.1f} days)")
    print(f"With 2 GPUs: {estimated_total/(2*3600):.1f} hours ({estimated_total/(2*3600*24):.1f} days)")
    
    # Show causality result
    print(f"\nCausality result:")
    print(f"Is causal: {result.get('interpretation', 'unknown')}")
    if 'safe_p_value' in result:
        print(f"Safe p-value: {result['safe_p_value']:.4f}")
    if 'risky_p_value' in result:
        print(f"Risky p-value: {result['risky_p_value']:.4f}")

if __name__ == "__main__":
    test_feature_timing()