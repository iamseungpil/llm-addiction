#!/usr/bin/env python3
import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def analyze_reward_choice_results(results_file):
    """Analyze experiment 3 reward choice results"""
    print("=" * 80)
    print("Experiment 3 Reward Choice Results Analysis")
    print("=" * 80)
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    config = data['config']
    results = data['results']
    
    print(f"Configuration:")
    print(f"  n_valid: {config['n_valid']}")
    print(f"  scales: {config['scales']}")
    print(f"  pure_betting_count: {config['pure_betting_count']}")
    print(f"  pure_stopping_count: {config['pure_stopping_count']}")
    print()
    
    # Analyze each feature type
    for feature_type in ['pure_betting', 'pure_stopping']:
        print(f"\n{'='*60}")
        print(f"Analysis: {feature_type.upper()}")
        print('='*60)
        
        feature_results = results[feature_type]
        scales = []
        choice_1_probs = []
        choice_2_probs = []
        
        for scale in config['scales']:
            scale_str = str(scale)
            if scale_str in feature_results:
                result = feature_results[scale_str]
                scales.append(scale if scale != 'no_patch' else 0.0)
                choice_1_probs.append(result['choice_probs']['1'])
                choice_2_probs.append(result['choice_probs']['2'])
                
                print(f"Scale {scale_str:8}: Choice 1: {result['choice_probs']['1']:.3f}, Choice 2: {result['choice_probs']['2']:.3f}")
        
        # Statistical analysis
        print(f"\nStatistical Analysis for {feature_type}:")
        
        # Correlation analysis
        if len(scales) > 2:
            # Remove no_patch for correlation
            numeric_scales = [s for s in scales if s != 0.0]
            numeric_choice2 = [choice_2_probs[i] for i, s in enumerate(scales) if s != 0.0]
            
            if len(numeric_scales) > 2:
                correlation, p_value = stats.spearmanr(numeric_scales, numeric_choice2)
                print(f"  Spearman correlation (scale vs choice 2 prob): {correlation:.4f} (p={p_value:.4f})")
        
        # Compare no_patch vs extreme scales
        if 'no_patch' in feature_results and '5.0' in feature_results:
            no_patch_choices = feature_results['no_patch']['choices']
            extreme_choices = feature_results['5.0']['choices']
            
            # Count choice 2 selections
            no_patch_choice2 = sum(1 for c in no_patch_choices if c == '2')
            extreme_choice2 = sum(1 for c in extreme_choices if c == '2')
            
            # Chi-square test (only if both conditions have some choice 2 selections)
            if no_patch_choice2 > 0 and extreme_choice2 > 0:
                contingency = [[no_patch_choice2, len(no_patch_choices) - no_patch_choice2],
                              [extreme_choice2, len(extreme_choices) - extreme_choice2]]
                
                chi2, p_chi2 = stats.chi2_contingency(contingency)[:2]
                print(f"  Chi-square test (no_patch vs 5.0x): χ²={chi2:.4f}, p={p_chi2:.4f}")
            else:
                print(f"  Chi-square test: Cannot compute (zero frequencies)")
                print(f"    no_patch choice 2: {no_patch_choice2}/{len(no_patch_choices)}")
                print(f"    extreme choice 2: {extreme_choice2}/{len(extreme_choices)}")
            
            # Effect size (difference in proportions)
            effect_size = (extreme_choice2 / len(extreme_choices)) - (no_patch_choice2 / len(no_patch_choices))
            print(f"  Effect size (Δ choice 2 prob): {effect_size:.4f}")
    
    return data

def main():
    results_file = "/data/llm_addiction/results/exp3_corrected_reward_choice_gpu2_20250909_233743.json"
    analyze_reward_choice_results(results_file)

if __name__ == "__main__":
    main()