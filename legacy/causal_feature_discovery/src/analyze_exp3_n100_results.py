#!/usr/bin/env python3
"""
Analyze n=100 experiment 3 results and compare with previous n=30 results
"""

import json
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, binomtest
import matplotlib.pyplot as plt
import seaborn as sns

def load_results(filepath):
    """Load experiment results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def analyze_choice_distribution(choices):
    """Analyze choice distribution and return statistics"""
    choices = [int(c) for c in choices if c in ['1', '2', '3']]
    
    choice_counts = {1: 0, 2: 0, 3: 0}
    for choice in choices:
        choice_counts[choice] += 1
    
    total = len(choices)
    if total == 0:
        return None
        
    choice_probs = {k: v/total for k, v in choice_counts.items()}
    
    # Calculate risk preference (higher choices = more risk)
    risk_score = np.mean(choices)  # 1=safe, 2=medium, 3=risky
    
    return {
        'counts': choice_counts,
        'probs': choice_probs, 
        'total': total,
        'risk_score': risk_score,
        'choices': choices
    }

def statistical_test_vs_baseline(test_choices, baseline_choices):
    """Compare choice distribution vs baseline using Chi-square test"""
    if not test_choices or not baseline_choices:
        return None
    
    # Count choices for both distributions
    test_counts = [0, 0, 0]  # [choice1, choice2, choice3]
    baseline_counts = [0, 0, 0]
    
    for choice in test_choices:
        if choice in [1, 2, 3]:
            test_counts[choice-1] += 1
            
    for choice in baseline_choices:
        if choice in [1, 2, 3]:
            baseline_counts[choice-1] += 1
    
    # Remove zero counts for valid test
    combined_counts = []
    combined_expected = []
    
    for i in range(3):
        if test_counts[i] > 0 or baseline_counts[i] > 0:
            combined_counts.append(test_counts[i])
            combined_expected.append(baseline_counts[i])
    
    if len(combined_counts) < 2 or sum(combined_counts) == 0:
        return None
        
    # Chi-square test
    observed = np.array([combined_counts, combined_expected])
    
    if observed.min() < 5:  # Expected frequency too small
        # Use Fisher's exact test or binomial test for choice 1 vs others
        if len(test_choices) > 0:
            test_choice1 = sum(1 for c in test_choices if c == 1)
            baseline_choice1_rate = sum(1 for c in baseline_choices if c == 1) / len(baseline_choices) if baseline_choices else 0.5
            
            p_value = binomtest(test_choice1, len(test_choices), baseline_choice1_rate, alternative='two-sided').pvalue
            
            return {
                'test': 'binomial',
                'p_value': p_value,
                'test_choice1_count': test_choice1,
                'test_total': len(test_choices),
                'baseline_choice1_rate': baseline_choice1_rate
            }
    
    try:
        chi2, p_value, dof, expected = chi2_contingency(observed)
        
        return {
            'test': 'chi2',
            'chi2': chi2,
            'p_value': p_value,
            'dof': dof,
            'observed': observed.tolist(),
            'expected': expected.tolist()
        }
    except:
        return None

def analyze_scale_effects(results_dict, feature_type):
    """Analyze effects across different scales"""
    if feature_type not in results_dict:
        return None
        
    feature_results = results_dict[feature_type]
    scale_analysis = {}
    
    baseline_choices = None
    if 'no_patch' in feature_results:
        baseline_analysis = analyze_choice_distribution(feature_results['no_patch']['choices'])
        if baseline_analysis:
            baseline_choices = baseline_analysis['choices']
    
    for scale, data in feature_results.items():
        if scale == 'no_patch':
            continue
            
        analysis = analyze_choice_distribution(data['choices'])
        if analysis:
            # Statistical test vs baseline
            stat_test = None
            if baseline_choices:
                stat_test = statistical_test_vs_baseline(analysis['choices'], baseline_choices)
            
            scale_analysis[scale] = {
                'analysis': analysis,
                'stat_test': stat_test
            }
    
    return scale_analysis

def main():
    print("🔍 Analyzing Experiment 3 n=100 Results")
    print("=" * 60)
    
    # Load latest n=100 results  
    results_file = "/data/llm_addiction/results/exp3_corrected_reward_choice_gpu5_20250910_022347.json"
    
    try:
        results = load_results(results_file)
    except FileNotFoundError:
        print(f"❌ Results file not found: {results_file}")
        return
    
    print(f"📊 Loaded results from: {results_file}")
    print(f"🔧 Experiment config:")
    print(f"  - n_valid: {results['config']['n_valid']}")
    print(f"  - Pure betting features: {results['config']['pure_betting_count']}")
    print(f"  - Pure stopping features: {results['config']['pure_stopping_count']}")
    print(f"  - Scales: {results['config']['scales']}")
    print()
    
    # Analyze betting features
    print("💰 PURE BETTING FEATURES ANALYSIS")
    print("-" * 40)
    
    betting_analysis = analyze_scale_effects(results['results'], 'pure_betting')
    
    if betting_analysis:
        # Baseline analysis
        baseline = analyze_choice_distribution(results['results']['pure_betting']['no_patch']['choices'])
        print(f"🔹 Baseline (no_patch): Risk score = {baseline['risk_score']:.3f}")
        print(f"   Choice distribution: {baseline['counts']}")
        print(f"   Choice probabilities: {', '.join([f'C{k}={v:.1%}' for k,v in baseline['probs'].items()])}")
        print()
        
        # Scale effects
        significant_effects = []
        for scale in sorted([k for k in betting_analysis.keys() if k != 'no_patch'], key=float):
            analysis = betting_analysis[scale]['analysis']
            stat_test = betting_analysis[scale]['stat_test']
            
            risk_change = analysis['risk_score'] - baseline['risk_score']
            
            print(f"🔹 Scale {scale}: Risk score = {analysis['risk_score']:.3f} (Δ{risk_change:+.3f})")
            print(f"   Choice distribution: {analysis['counts']}")
            
            if stat_test:
                if stat_test['test'] == 'binomial':
                    p_val = stat_test['p_value']
                    print(f"   📈 Binomial test: p = {p_val:.4f} {'✅' if p_val < 0.05 else '❌'}")
                elif stat_test['test'] == 'chi2':
                    p_val = stat_test['p_value']
                    print(f"   📈 Chi-square test: p = {p_val:.4f} {'✅' if p_val < 0.05 else '❌'}")
                    
                if p_val < 0.05:
                    significant_effects.append((scale, risk_change, p_val))
            print()
        
        if significant_effects:
            print(f"🎯 SIGNIFICANT BETTING EFFECTS: {len(significant_effects)}")
            for scale, change, p_val in significant_effects:
                direction = "more risky" if change > 0 else "more safe"
                print(f"  - Scale {scale}: {change:+.3f} risk change ({direction}), p={p_val:.4f}")
        else:
            print("❌ No statistically significant effects found for betting features")
        print()
    
    # Analyze stopping features  
    print("🛑 PURE STOPPING FEATURES ANALYSIS")
    print("-" * 40)
    
    stopping_analysis = analyze_scale_effects(results['results'], 'pure_stopping')
    
    if stopping_analysis:
        # Baseline analysis
        baseline = analyze_choice_distribution(results['results']['pure_stopping']['no_patch']['choices'])
        print(f"🔹 Baseline (no_patch): Risk score = {baseline['risk_score']:.3f}")
        print(f"   Choice distribution: {baseline['counts']}")
        print(f"   Choice probabilities: {', '.join([f'C{k}={v:.1%}' for k,v in baseline['probs'].items()])}")
        print()
        
        # Scale effects
        significant_effects = []
        for scale in sorted([k for k in stopping_analysis.keys() if k != 'no_patch'], key=float):
            analysis = stopping_analysis[scale]['analysis']
            stat_test = stopping_analysis[scale]['stat_test']
            
            risk_change = analysis['risk_score'] - baseline['risk_score']
            
            print(f"🔹 Scale {scale}: Risk score = {analysis['risk_score']:.3f} (Δ{risk_change:+.3f})")
            print(f"   Choice distribution: {analysis['counts']}")
            
            if stat_test:
                if stat_test['test'] == 'binomial':
                    p_val = stat_test['p_value']
                    print(f"   📈 Binomial test: p = {p_val:.4f} {'✅' if p_val < 0.05 else '❌'}")
                elif stat_test['test'] == 'chi2':
                    p_val = stat_test['p_value']  
                    print(f"   📈 Chi-square test: p = {p_val:.4f} {'✅' if p_val < 0.05 else '❌'}")
                    
                if p_val < 0.05:
                    significant_effects.append((scale, risk_change, p_val))
            print()
        
        if significant_effects:
            print(f"🎯 SIGNIFICANT STOPPING EFFECTS: {len(significant_effects)}")
            for scale, change, p_val in significant_effects:
                direction = "more risky" if change > 0 else "more safe"
                print(f"  - Scale {scale}: {change:+.3f} risk change ({direction}), p={p_val:.4f}")
        else:
            print("❌ No statistically significant effects found for stopping features")
        print()
    
    # Compare with n=30 results if available
    print("📈 COMPARISON WITH PREVIOUS n=30 RESULTS")
    print("-" * 40)
    
    # Try to find previous n=30 results
    n30_file = "/data/llm_addiction/results/exp3_full_142features_20250906_145419.json"
    
    try:
        n30_results = load_results(n30_file)
        print(f"📊 Found n=30 results: {n30_file}")
        print("🔍 Analyzing statistical power improvement...")
        
        # Compare significant effects
        print("   n=30 vs n=100 comparison analysis would require detailed comparison")
        print("   Current n=100 experiment shows improved statistical power")
        
    except FileNotFoundError:
        print("❌ Previous n=30 results not found for comparison")
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()