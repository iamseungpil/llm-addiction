#!/usr/bin/env python3
"""
Analyze all_results from population mean patching to verify actual data
"""

import json
import numpy as np
from collections import defaultdict

def analyze_all_results():
    """Analyze all_results from both GPU files"""
    
    # Load both result files
    gpu4_file = '/data/llm_addiction/results/patching_population_mean_final_20250905_150612.json'
    gpu5_file = '/data/llm_addiction/results/patching_population_mean_final_20250905_085027.json'
    
    print("Loading all_results...")
    
    with open(gpu4_file, 'r') as f:
        gpu4_data = json.load(f)
    
    with open(gpu5_file, 'r') as f:
        gpu5_data = json.load(f)
    
    # Combine all results
    all_results = gpu4_data['all_results'] + gpu5_data['all_results']
    
    print(f"Total results: {len(all_results)}")
    print(f"GPU 4 results: {len(gpu4_data['all_results'])}")
    print(f"GPU 5 results: {len(gpu5_data['all_results'])}")
    
    # Analyze structure
    if all_results:
        sample_result = all_results[0]
        print(f"\nSample result structure:")
        for key, value in sample_result.items():
            if isinstance(value, dict):
                print(f"  {key}: {type(value)} with keys {list(value.keys())}")
            else:
                print(f"  {key}: {type(value)} = {value}")
    
    # Group by feature
    feature_groups = defaultdict(list)
    for result in all_results:
        key = f"L{result['layer']}-{result['feature_id']}"
        feature_groups[key].append(result)
    
    print(f"\nUnique features tested: {len(feature_groups)}")
    
    # Analyze scales and conditions
    scales = set()
    prompt_types = set()
    for result in all_results:
        scales.add(result['scale'])
        prompt_types.add(result['prompt_type'])
    
    print(f"Scales used: {sorted(scales)}")
    print(f"Prompt types: {sorted(prompt_types)}")
    
    # Check a specific feature that showed perfect correlation
    target_features = ['L30-22675', 'L30-22072', 'L30-8387']
    
    for feature_key in target_features:
        if feature_key in feature_groups:
            print(f"\n" + "="*60)
            print(f"DETAILED ANALYSIS: {feature_key}")
            print("="*60)
            
            feature_results = feature_groups[feature_key]
            
            # Separate by prompt type
            risky_results = [r for r in feature_results if r['prompt_type'] == 'risky']
            safe_results = [r for r in feature_results if r['prompt_type'] == 'safe']
            
            # Analyze risky results
            if risky_results:
                print(f"\nRISKY PROMPT ({len(risky_results)} results):")
                risky_results.sort(key=lambda x: x['scale'])
                
                for r in risky_results:
                    print(f"  Scale {r['scale']}: avg_bet=${r['avg_bet']:.2f}, stop_rate={r['stop_rate']:.3f}, "
                          f"trials={r['n_trials']}, patched_value={r.get('patched_value', 'None')}")
                
                # Check causality
                if risky_results[0].get('causality'):
                    causality = risky_results[0]['causality']
                    print(f"\n  CAUSALITY DATA:")
                    print(f"    risky_bet_correlation: {causality.get('risky_bet_correlation', 'N/A')}")
                    print(f"    safe_bet_correlation: {causality.get('safe_bet_correlation', 'N/A')}")
                    print(f"    risky_stop_correlation: {causality.get('risky_stop_correlation', 'N/A')}")
                    print(f"    safe_stop_correlation: {causality.get('safe_stop_correlation', 'N/A')}")
                    print(f"    bet_effect_risky: {causality.get('bet_effect_risky', 'N/A')}")
                    print(f"    bet_effect_safe: {causality.get('bet_effect_safe', 'N/A')}")
                    print(f"    is_causal_bet: {causality.get('is_causal_bet', 'N/A')}")
                    print(f"    is_causal_stop: {causality.get('is_causal_stop', 'N/A')}")
            
            # Analyze safe results
            if safe_results:
                print(f"\nSAFE PROMPT ({len(safe_results)} results):")
                safe_results.sort(key=lambda x: x['scale'])
                
                for r in safe_results:
                    print(f"  Scale {r['scale']}: avg_bet=${r['avg_bet']:.2f}, stop_rate={r['stop_rate']:.3f}, "
                          f"trials={r['n_trials']}, patched_value={r.get('patched_value', 'None')}")
    
    # Check if images match actual data
    print(f"\n" + "="*60)
    print("CHECKING IMAGE DATA CONSISTENCY")
    print("="*60)
    
    # Look for features mentioned in images: L30-21866, L30-22274, L30-22675, L30-320, L30-3003, L30-30536
    image_features = ['L30-21866', 'L30-22274', 'L30-22675', 'L30-320', 'L30-3003', 'L30-30536']
    
    actual_causal_features = []
    for feature_key, results in feature_groups.items():
        if results and results[0].get('causality', {}).get('is_causal_bet', False):
            actual_causal_features.append(feature_key)
    
    print(f"Features mentioned in images: {image_features}")
    print(f"Actually causal features (first 10): {actual_causal_features[:10]}")
    
    # Check if image features are actually causal
    for feat in image_features:
        if feat in feature_groups:
            causality = feature_groups[feat][0].get('causality', {})
            is_causal = causality.get('is_causal_bet', False) or causality.get('is_causal_stop', False)
            print(f"  {feat}: is_causal = {is_causal}")
        else:
            print(f"  {feat}: NOT FOUND in results")
    
    # Summary statistics
    print(f"\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    causal_bet_count = 0
    causal_stop_count = 0
    total_features = 0
    
    for feature_key, results in feature_groups.items():
        if results:
            total_features += 1
            causality = results[0].get('causality', {})
            if causality.get('is_causal_bet', False):
                causal_bet_count += 1
            if causality.get('is_causal_stop', False):
                causal_stop_count += 1
    
    print(f"Total features tested: {total_features}")
    print(f"Causal features (betting): {causal_bet_count}")
    print(f"Causal features (stop): {causal_stop_count}")
    print(f"Causal rate (betting): {causal_bet_count/total_features*100:.1f}%")
    print(f"Causal rate (stop): {causal_stop_count/total_features*100:.1f}%")

if __name__ == '__main__':
    analyze_all_results()