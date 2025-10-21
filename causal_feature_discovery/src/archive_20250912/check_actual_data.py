#!/usr/bin/env python3
"""
Check actual data that's causing p=0.000000 and r=1.000 results
"""

import json
import numpy as np
from scipy.stats import spearmanr

def check_specific_feature():
    """Check specific feature data that shows perfect correlation"""
    
    # Load GPU 5 results
    gpu5_file = '/data/llm_addiction/results/patching_population_mean_final_20250905_085027.json'
    
    with open(gpu5_file, 'r') as f:
        gpu5_data = json.load(f)
    
    # Find L25-1130 data (first one with perfect correlation)
    target_results = []
    for result in gpu5_data['all_results']:
        if result['layer'] == 25 and result['feature_id'] == 1130:
            target_results.append(result)
    
    print("="*80)
    print("CHECKING L25-1130 DATA (Perfect Correlation)")
    print("="*80)
    
    # Separate by prompt type
    risky_results = [r for r in target_results if r['prompt_type'] == 'risky']
    safe_results = [r for r in target_results if r['prompt_type'] == 'safe']
    
    print(f"Risky results: {len(risky_results)}")
    print(f"Safe results: {len(safe_results)}")
    
    if risky_results:
        print("\nRISKY PROMPT DATA:")
        for r in risky_results:
            print(f"  Scale {r['scale']}: avg_bet=${r['avg_bet']:.1f}, stop_rate={r['stop_rate']:.3f}")
        
        scales = [r['scale'] for r in risky_results]
        bets = [r['avg_bet'] for r in risky_results]
        stops = [r['stop_rate'] for r in risky_results]
        
        print(f"\n  Scales: {scales}")
        print(f"  Bets: {bets}")
        print(f"  Stops: {stops}")
        
        # Check correlation manually
        bet_corr, bet_p = spearmanr(scales, bets)
        stop_corr, stop_p = spearmanr(scales, stops)
        
        print(f"\n  Bet correlation: r={bet_corr:.6f}, p={bet_p:.6f}")
        print(f"  Stop correlation: r={stop_corr:.6f}, p={stop_p:.6f}")
    
    if safe_results:
        print("\nSAFE PROMPT DATA:")
        for r in safe_results:
            print(f"  Scale {r['scale']}: avg_bet=${r['avg_bet']:.1f}, stop_rate={r['stop_rate']:.3f}")
        
        scales = [r['scale'] for r in safe_results]
        bets = [r['avg_bet'] for r in safe_results]
        stops = [r['stop_rate'] for r in safe_results]
        
        print(f"\n  Scales: {scales}")
        print(f"  Bets: {bets}")
        print(f"  Stops: {stops}")
        
        # Check correlation manually
        bet_corr, bet_p = spearmanr(scales, bets)
        stop_corr, stop_p = spearmanr(scales, stops)
        
        print(f"\n  Bet correlation: r={bet_corr:.6f}, p={bet_p:.6f}")
        print(f"  Stop correlation: r={stop_corr:.6f}, p={stop_p:.6f}")
    
    # Check stored causality data
    if target_results:
        causality = target_results[0]['causality']
        print("\nSTORED CAUSALITY DATA:")
        for key, value in causality.items():
            print(f"  {key}: {value}")

def check_multiple_features():
    """Check multiple features to understand the pattern"""
    
    gpu5_file = '/data/llm_addiction/results/patching_population_mean_final_20250905_085027.json'
    
    with open(gpu5_file, 'r') as f:
        gpu5_data = json.load(f)
    
    # Group by feature
    feature_groups = {}
    for result in gpu5_data['all_results']:
        key = f"L{result['layer']}-{result['feature_id']}"
        if key not in feature_groups:
            feature_groups[key] = []
        feature_groups[key].append(result)
    
    print("\n" + "="*80)
    print("CHECKING MULTIPLE FEATURES FOR PATTERN")
    print("="*80)
    
    perfect_corr_count = 0
    total_features = 0
    
    for feature_key, results in list(feature_groups.items())[:10]:  # Check first 10
        
        risky_results = [r for r in results if r['prompt_type'] == 'risky']
        safe_results = [r for r in results if r['prompt_type'] == 'safe']
        
        if len(risky_results) != 3 or len(safe_results) != 3:
            continue
            
        total_features += 1
        
        # Check risky data
        risky_scales = [r['scale'] for r in risky_results]
        risky_bets = [r['avg_bet'] for r in risky_results]
        
        # Check if data is monotonic (perfect order)
        is_monotonic_bet = all(risky_bets[i] <= risky_bets[i+1] for i in range(len(risky_bets)-1)) or \
                           all(risky_bets[i] >= risky_bets[i+1] for i in range(len(risky_bets)-1))
        
        bet_corr, bet_p = spearmanr(risky_scales, risky_bets)
        
        print(f"\n{feature_key}:")
        print(f"  Scales: {risky_scales}")
        print(f"  Bets: {risky_bets}")
        print(f"  Monotonic: {is_monotonic_bet}")
        print(f"  Correlation: r={bet_corr:.6f}, p={bet_p:.6f}")
        
        if abs(bet_corr) == 1.0:
            perfect_corr_count += 1
    
    print(f"\nSUMMARY:")
    print(f"  Perfect correlations: {perfect_corr_count}/{total_features}")
    print(f"  Rate: {perfect_corr_count/total_features*100:.1f}%")

def understand_spearman_with_3_points():
    """Understand why Spearman gives perfect correlation with 3 points"""
    
    print("\n" + "="*80)
    print("UNDERSTANDING SPEARMAN WITH 3 POINTS")
    print("="*80)
    
    # Test cases
    test_cases = [
        ([0.5, 1.0, 1.5], [10, 15, 20], "Perfect increasing"),
        ([0.5, 1.0, 1.5], [20, 15, 10], "Perfect decreasing"),
        ([0.5, 1.0, 1.5], [10, 20, 15], "Non-monotonic"),
        ([0.5, 1.0, 1.5], [15, 15, 15], "Constant"),
        ([0.5, 1.0, 1.5], [10, 15, 15], "Partial increase"),
    ]
    
    for scales, values, description in test_cases:
        corr, p = spearmanr(scales, values)
        print(f"\n{description}:")
        print(f"  X: {scales}")
        print(f"  Y: {values}")
        print(f"  Spearman r: {corr:.6f}, p: {p:.6f}")
        
        # Show ranks
        from scipy.stats import rankdata
        x_ranks = rankdata(scales)
        y_ranks = rankdata(values)
        print(f"  X ranks: {x_ranks}")
        print(f"  Y ranks: {y_ranks}")

if __name__ == '__main__':
    check_specific_feature()
    check_multiple_features()
    understand_spearman_with_3_points()