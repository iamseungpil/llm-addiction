#!/usr/bin/env python3
"""
Debug why intervention response curves show flat lines
Check if the causal feature grouping logic is working correctly
"""

import json
import numpy as np
from collections import defaultdict

def debug_intervention_response():
    """Debug the intervention response curve generation"""
    
    print("="*80)
    print("DEBUGGING INTERVENTION RESPONSE CURVES")
    print("="*80)
    
    # Load actual patching results from both GPUs
    print("Loading data...")
    
    with open('/data/llm_addiction/results/patching_population_mean_final_20250905_150612.json', 'r') as f:
        gpu4_data = json.load(f)
    
    with open('/data/llm_addiction/results/patching_population_mean_final_20250905_085027.json', 'r') as f:
        gpu5_data = json.load(f)
    
    print(f"GPU4 causal betting features: {len(gpu4_data.get('causal_features_bet', []))}")
    print(f"GPU5 causal betting features: {len(gpu5_data.get('causal_features_bet', []))}")
    
    # Combine all results
    all_results = []
    if 'all_results' in gpu4_data:
        all_results.extend(gpu4_data['all_results'])
    if 'all_results' in gpu5_data:
        all_results.extend(gpu5_data['all_results'])
    
    print(f"Total results: {len(all_results)}")
    
    # Get causal features the SAME WAY as the figure code
    causal_features = set()
    if 'causal_features_bet' in gpu4_data:
        for feat in gpu4_data['causal_features_bet']:
            causal_features.add((feat['layer'], feat['feature_id']))
    if 'causal_features_bet' in gpu5_data:
        for feat in gpu5_data['causal_features_bet']:
            causal_features.add((feat['layer'], feat['feature_id']))
    
    print(f"Causal features identified: {len(causal_features)}")
    print(f"First 5 causal features: {list(causal_features)[:5]}")
    
    # Check data grouping by scale
    scale_data = defaultdict(lambda: {'causal': [], 'non_causal': []})
    
    causal_count = 0
    non_causal_count = 0
    
    for result in all_results:
        layer = result['layer']
        feature_id = result['feature_id']
        scale = result['scale']
        avg_bet = result.get('avg_bet', 0)
        stop_rate = result.get('stop_rate', 0)
        
        # Skip invalid results
        if avg_bet is None or stop_rate is None:
            continue
            
        # Check if causal
        is_causal = (layer, feature_id) in causal_features
        
        if is_causal:
            causal_count += 1
        else:
            non_causal_count += 1
            
        feature_type = 'causal' if is_causal else 'non_causal'
        scale_data[scale][feature_type].append({
            'avg_bet': avg_bet,
            'stop_rate': stop_rate * 100,
            'risk_taking_prob': (1 - stop_rate) * 100
        })
    
    print(f"\nData classification:")
    print(f"  Causal results: {causal_count}")
    print(f"  Non-causal results: {non_causal_count}")
    print(f"  Total: {causal_count + non_causal_count}")
    
    # Analyze by scale
    scales = sorted(scale_data.keys())
    print(f"\nScales found: {scales}")
    
    for scale in scales:
        causal_data = scale_data[scale]['causal']
        non_causal_data = scale_data[scale]['non_causal']
        
        print(f"\nScale {scale}:")
        print(f"  Causal features: {len(causal_data)} results")
        print(f"  Non-causal features: {len(non_causal_data)} results")
        
        if causal_data:
            causal_bets = [d['avg_bet'] for d in causal_data]
            print(f"  Causal avg bet: {np.mean(causal_bets):.2f} ± {np.std(causal_bets):.2f}")
            print(f"  Causal bet range: {min(causal_bets):.1f} - {max(causal_bets):.1f}")
        
        if non_causal_data:
            non_causal_bets = [d['avg_bet'] for d in non_causal_data]
            print(f"  Non-causal avg bet: {np.mean(non_causal_bets):.2f} ± {np.std(non_causal_bets):.2f}")
            print(f"  Non-causal bet range: {min(non_causal_bets):.1f} - {max(non_causal_bets):.1f}")
    
    # Check specific causal features vs actual results
    print(f"\n" + "="*60)
    print("CHECKING CAUSAL FEATURE CLASSIFICATION")
    print("="*60)
    
    # Check first few results to see if they're being classified correctly
    for i, result in enumerate(all_results[:10]):
        layer = result['layer']
        feature_id = result['feature_id']
        is_causal = (layer, feature_id) in causal_features
        causality = result.get('causality', {})
        actual_causal = causality.get('is_causal_bet', False)
        
        print(f"Result {i}: L{layer}-{feature_id}")
        print(f"  Classified as causal: {is_causal}")
        print(f"  Actually causal (from result): {actual_causal}")
        print(f"  Match: {is_causal == actual_causal}")
        print(f"  Scale: {result['scale']}, Bet: ${result['avg_bet']:.1f}")
        print()
    
    # Final calculation as in figure code
    causal_avg_bets = []
    non_causal_avg_bets = []
    
    for scale in scales:
        causal_data = scale_data[scale]['causal']
        non_causal_data = scale_data[scale]['non_causal']
        
        if causal_data:
            causal_avg_bets.append(np.mean([d['avg_bet'] for d in causal_data]))
        else:
            causal_avg_bets.append(0)
        
        if non_causal_data:
            non_causal_avg_bets.append(np.mean([d['avg_bet'] for d in non_causal_data]))
        else:
            non_causal_avg_bets.append(0)
    
    print(f"FINAL RESULTS (as shown in figure):")
    print(f"Scales: {scales}")
    print(f"Causal avg bets: {causal_avg_bets}")
    print(f"Non-causal avg bets: {non_causal_avg_bets}")
    
    # Check if there's variation
    if len(set(causal_avg_bets)) == 1:
        print(f"❌ PROBLEM: Causal bets are all the same value!")
    else:
        print(f"✅ Causal bets show variation: {min(causal_avg_bets):.1f} to {max(causal_avg_bets):.1f}")
    
    if len(set(non_causal_avg_bets)) == 1:
        print(f"❌ PROBLEM: Non-causal bets are all the same value!")
    else:
        print(f"✅ Non-causal bets show variation: {min(non_causal_avg_bets):.1f} to {max(non_causal_avg_bets):.1f}")

if __name__ == '__main__':
    debug_intervention_response()