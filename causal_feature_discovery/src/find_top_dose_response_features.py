#!/usr/bin/env python3
"""
Find top features with largest dose-response effects for individual analysis
"""

import json
import numpy as np
from collections import defaultdict

def find_top_dose_response_features():
    """Find features with largest dose-response effects"""
    
    print("Finding top dose-response features...")
    
    # Load results from both GPUs
    gpu4_file = '/data/llm_addiction/results/patching_population_mean_final_20250905_150612.json'
    gpu5_file = '/data/llm_addiction/results/patching_population_mean_final_20250905_085027.json'
    
    with open(gpu4_file, 'r') as f:
        gpu4_data = json.load(f)
    
    with open(gpu5_file, 'r') as f:
        gpu5_data = json.load(f)
    
    # Combine all results
    all_results = gpu4_data['all_results'] + gpu5_data['all_results']
    
    # Group by feature
    feature_groups = defaultdict(list)
    for result in all_results:
        key = f"L{result['layer']}-{result['feature_id']}"
        feature_groups[key].append(result)
    
    print(f"Total features: {len(feature_groups)}")
    
    # Calculate dose-response effects for each feature
    feature_effects = []
    
    for feature_key, results in feature_groups.items():
        
        # Separate by prompt type
        risky_results = [r for r in results if r['prompt_type'] == 'risky']
        safe_results = [r for r in results if r['prompt_type'] == 'safe']
        
        if len(risky_results) < 3 or len(safe_results) < 3:
            continue
        
        # Sort by scale
        risky_results.sort(key=lambda x: x['scale'])
        safe_results.sort(key=lambda x: x['scale'])
        
        # Calculate betting effect (max - min across scales)
        risky_bets = [r['avg_bet'] for r in risky_results]
        safe_bets = [r['avg_bet'] for r in safe_results]
        
        risky_bet_effect = max(risky_bets) - min(risky_bets)
        safe_bet_effect = max(safe_bets) - min(safe_bets)
        max_bet_effect = max(risky_bet_effect, safe_bet_effect)
        
        # Calculate stop rate effect
        risky_stops = [r['stop_rate'] for r in risky_results]
        safe_stops = [r['stop_rate'] for r in safe_results]
        
        risky_stop_effect = abs(max(risky_stops) - min(risky_stops))
        safe_stop_effect = abs(max(safe_stops) - min(safe_stops))
        max_stop_effect = max(risky_stop_effect, safe_stop_effect)
        
        # Check if causal
        causality = results[0].get('causality', {})
        is_causal = causality.get('is_causal_bet', False) or causality.get('is_causal_stop', False)
        
        # Store feature info
        layer = results[0]['layer']
        feature_id = results[0]['feature_id']
        
        feature_effects.append({
            'feature_key': feature_key,
            'layer': layer,
            'feature_id': feature_id,
            'bet_effect': max_bet_effect,
            'stop_effect': max_stop_effect,
            'combined_effect': max_bet_effect + max_stop_effect * 100,  # Scale stop effect
            'is_causal': is_causal,
            'risky_bets': risky_bets,
            'safe_bets': safe_bets,
            'risky_stops': risky_stops,
            'safe_stops': safe_stops,
            'scales': [r['scale'] for r in risky_results]
        })
    
    # Sort by combined effect
    feature_effects.sort(key=lambda x: x['combined_effect'], reverse=True)
    
    print(f"Features with effects calculated: {len(feature_effects)}")
    
    # Show top features
    print(f"\nTop 20 features with largest dose-response effects:")
    print("="*80)
    
    for i, feat in enumerate(feature_effects[:20], 1):
        print(f"{i:2d}. {feat['feature_key']:12s} | Bet: ${feat['bet_effect']:6.1f} | Stop: {feat['stop_effect']:5.3f} | Causal: {feat['is_causal']}")
        print(f"    Risky bets: {[f'{b:.1f}' for b in feat['risky_bets']]}")
        print(f"    Safe bets:  {[f'{b:.1f}' for b in feat['safe_bets']]}")
        print()
    
    # Filter for causal features only
    causal_features = [f for f in feature_effects if f['is_causal']]
    causal_features.sort(key=lambda x: x['combined_effect'], reverse=True)
    
    print(f"\nTop 10 CAUSAL features with largest effects:")
    print("="*80)
    
    for i, feat in enumerate(causal_features[:10], 1):
        print(f"{i:2d}. {feat['feature_key']:12s} | Bet: ${feat['bet_effect']:6.1f} | Stop: {feat['stop_effect']:5.3f}")
        print(f"    Risky bets: {[f'{b:.1f}' for b in feat['risky_bets']]}")
        print(f"    Safe bets:  {[f'{b:.1f}' for b in feat['safe_bets']]}")
        print()
    
    # Save top features for plotting
    top_features = causal_features[:6]  # Top 6 for plotting
    
    output_data = {
        'timestamp': '20250910_top_dose_response',
        'top_features': top_features,
        'total_features_analyzed': len(feature_effects),
        'causal_features_count': len(causal_features)
    }
    
    output_file = '/data/llm_addiction/results/top_dose_response_features_20250910.json'
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✅ Top features saved to: {output_file}")
    
    return top_features

if __name__ == '__main__':
    find_top_dose_response_features()