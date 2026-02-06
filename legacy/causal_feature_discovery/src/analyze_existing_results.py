#!/usr/bin/env python3
"""
Analyze existing population mean patching results with proper p-value filtering
"""

import json
import numpy as np
from scipy import stats

def analyze_existing_results():
    """Re-analyze existing results with proper p-value filtering"""
    
    # Load GPU 4 and GPU 5 results
    gpu4_file = '/data/llm_addiction/results/patching_population_mean_final_20250905_150612.json'
    gpu5_file = '/data/llm_addiction/results/patching_population_mean_final_20250905_085027.json'
    
    print("Loading existing results...")
    
    with open(gpu4_file, 'r') as f:
        gpu4_data = json.load(f)
    
    with open(gpu5_file, 'r') as f:
        gpu5_data = json.load(f)
    
    print(f"GPU 4: {len(gpu4_data['all_results'])} results")
    print(f"GPU 5: {len(gpu5_data['all_results'])} results")
    
    # Combine all results
    all_results = gpu4_data['all_results'] + gpu5_data['all_results']
    
    # Group by feature
    feature_groups = {}
    for result in all_results:
        key = f"L{result['layer']}-{result['feature_id']}"
        if key not in feature_groups:
            feature_groups[key] = []
        feature_groups[key].append(result)
    
    print(f"Total features tested: {len(feature_groups)}")
    
    # Re-analyze with proper p-value filtering
    valid_causal_bet = []
    valid_causal_stop = []
    
    for feature_key, results in feature_groups.items():
        
        # Separate by prompt type
        risky_results = [r for r in results if r['prompt_type'] == 'risky']
        safe_results = [r for r in results if r['prompt_type'] == 'safe']
        
        if len(risky_results) < 3 or len(safe_results) < 3:
            continue
            
        # Extract data
        scales = [r['scale'] for r in risky_results]
        risky_bets = [r['avg_bet'] for r in risky_results]
        risky_stops = [r['stop_rate'] for r in risky_results]
        safe_bets = [r['avg_bet'] for r in safe_results]
        safe_stops = [r['stop_rate'] for r in safe_results]
        
        # Check causality in existing results
        causality = results[0]['causality']
        
        # Get actual correlation and p-values
        risky_bet_corr = causality.get('risky_bet_correlation', 0)
        safe_bet_corr = causality.get('safe_bet_correlation', 0)
        risky_stop_corr = causality.get('risky_stop_correlation', 0)
        safe_stop_corr = causality.get('safe_stop_correlation', 0)
        
        # RECALCULATE p-values (since they're missing from stored results)
        from scipy.stats import spearmanr
        
        _, risky_bet_p = spearmanr(scales, risky_bets)
        _, safe_bet_p = spearmanr(scales, safe_bets)
        _, risky_stop_p = spearmanr(scales, risky_stops)
        _, safe_stop_p = spearmanr(scales, safe_stops)
        
        # Effect sizes
        bet_effect_risky = max(risky_bets) - min(risky_bets)
        bet_effect_safe = max(safe_bets) - min(safe_bets)
        stop_effect_risky = abs(max(risky_stops) - min(risky_stops))
        stop_effect_safe = abs(max(safe_stops) - min(safe_stops))
        
        # CORRECTED causality criteria with p-values
        is_causal_bet = ((abs(risky_bet_corr) > 0.5 and risky_bet_p < 0.05) or 
                         (abs(safe_bet_corr) > 0.5 and safe_bet_p < 0.05)) and \
                        (bet_effect_risky > 5 or bet_effect_safe > 5)
        
        is_causal_stop = ((abs(risky_stop_corr) > 0.5 and risky_stop_p < 0.05) or 
                          (abs(safe_stop_corr) > 0.5 and safe_stop_p < 0.05)) and \
                         (stop_effect_risky > 0.1 or stop_effect_safe > 0.1)
        
        # Store valid causal features
        layer = results[0]['layer']
        feature_id = results[0]['feature_id']
        
        if is_causal_bet:
            valid_causal_bet.append({
                'layer': layer,
                'feature_id': feature_id,
                'bet_correlation': max(abs(risky_bet_corr), abs(safe_bet_corr)),
                'bet_p_value': min(risky_bet_p, safe_bet_p),
                'bet_effect': max(bet_effect_risky, bet_effect_safe),
                'risky_bet_corr': risky_bet_corr,
                'risky_bet_p': risky_bet_p,
                'safe_bet_corr': safe_bet_corr,
                'safe_bet_p': safe_bet_p
            })
        
        if is_causal_stop:
            valid_causal_stop.append({
                'layer': layer,
                'feature_id': feature_id,
                'stop_correlation': max(abs(risky_stop_corr), abs(safe_stop_corr)),
                'stop_p_value': min(risky_stop_p, safe_stop_p),
                'stop_effect': max(stop_effect_risky, stop_effect_safe),
                'risky_stop_corr': risky_stop_corr,
                'risky_stop_p': risky_stop_p,
                'safe_stop_corr': safe_stop_corr,
                'safe_stop_p': safe_stop_p
            })
    
    # Sort by p-value (most significant first)
    valid_causal_bet.sort(key=lambda x: x['bet_p_value'])
    valid_causal_stop.sort(key=lambda x: x['stop_p_value'])
    
    # Print results
    print("\n" + "="*80)
    print("CORRECTED ANALYSIS WITH P-VALUE FILTERING")
    print("="*80)
    
    print(f"\nOriginal claims:")
    print(f"  GPU 4: {gpu4_data['summary']['n_causal_bet']} betting + {gpu4_data['summary']['n_causal_stop']} stop")
    print(f"  GPU 5: {gpu5_data['summary']['n_causal_bet']} betting + {gpu5_data['summary']['n_causal_stop']} stop")
    
    print(f"\nCORRECTED results (with p < 0.05):")
    print(f"  Valid causal features (betting): {len(valid_causal_bet)}")
    print(f"  Valid causal features (stop): {len(valid_causal_stop)}")
    
    # Union of all causal features
    all_causal_features = set()
    for feat in valid_causal_bet:
        all_causal_features.add(f"L{feat['layer']}-{feat['feature_id']}")
    for feat in valid_causal_stop:
        all_causal_features.add(f"L{feat['layer']}-{feat['feature_id']}")
    
    print(f"  Total unique causal features: {len(all_causal_features)}")
    
    # Show top 10 most significant
    print(f"\nTop 10 Most Significant Causal Features (Betting):")
    for i, feat in enumerate(valid_causal_bet[:10], 1):
        print(f"  {i}. L{feat['layer']}-{feat['feature_id']}: p={feat['bet_p_value']:.6f}, r={feat['bet_correlation']:.3f}, effect=${feat['bet_effect']:.1f}")
    
    print(f"\nTop 10 Most Significant Causal Features (Stop Rate):")
    for i, feat in enumerate(valid_causal_stop[:10], 1):
        print(f"  {i}. L{feat['layer']}-{feat['feature_id']}: p={feat['stop_p_value']:.6f}, r={feat['stop_correlation']:.3f}, effect={feat['stop_effect']:.3f}")
    
    # Save corrected results
    corrected_results = {
        'timestamp': '20250910_corrected_analysis',
        'original_claims': {
            'gpu4_bet': gpu4_data['summary']['n_causal_bet'],
            'gpu4_stop': gpu4_data['summary']['n_causal_stop'],
            'gpu5_bet': gpu5_data['summary']['n_causal_bet'],
            'gpu5_stop': gpu5_data['summary']['n_causal_stop']
        },
        'corrected_results': {
            'n_causal_bet': len(valid_causal_bet),
            'n_causal_stop': len(valid_causal_stop),
            'n_causal_total': len(all_causal_features)
        },
        'valid_causal_bet': valid_causal_bet,
        'valid_causal_stop': valid_causal_stop
    }
    
    output_file = '/data/llm_addiction/results/corrected_population_patching_analysis_20250910.json'
    with open(output_file, 'w') as f:
        json.dump(corrected_results, f, indent=2)
    
    print(f"\n✅ Corrected analysis saved to: {output_file}")
    
    return valid_causal_bet, valid_causal_stop

if __name__ == '__main__':
    analyze_existing_results()