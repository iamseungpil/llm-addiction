#!/usr/bin/env python3
"""
Verify condition matching between GPT and LLaMA experiments
Check if we're comparing the same experimental conditions, not just condition IDs
"""

import json
import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr, rankdata
import time

def load_all_data():
    """Load complete datasets"""
    print("Loading datasets...")
    
    # GPT data
    with open('/data/llm_addiction/gpt_results_corrected/gpt_corrected_complete_20250825_212628.json', 'r') as f:
        gpt_data = json.load(f)
    
    # LLaMA data - both files
    with open('/data/llm_addiction/results/exp1_multiround_intermediate_20250819_140040.json', 'r') as f:
        llama_data1 = json.load(f)
    
    with open('/data/llm_addiction/results/exp1_missing_complete_20250820_090040.json', 'r') as f:
        llama_data2 = json.load(f)
    
    all_llama = llama_data1['results'] + llama_data2['results']
    
    print(f"GPT: {len(gpt_data['results'])} experiments")
    print(f"LLaMA: {len(all_llama)} experiments")
    
    return gpt_data['results'], all_llama

def analyze_conditions():
    """Analyze experimental conditions in both datasets"""
    
    gpt_results, llama_results = load_all_data()
    
    print("\n" + "="*60)
    print("ANALYZING EXPERIMENTAL CONDITIONS")
    print("="*60)
    
    # Extract unique experimental conditions from GPT
    gpt_conditions = defaultdict(list)
    for result in gpt_results:
        # Create condition key from experimental parameters
        key = f"{result['bet_type']}_{result['first_result']}_{result['prompt_combo']}"
        condition_id = result['condition_id']
        gpt_conditions[key].append(condition_id)
    
    print(f"\nGPT unique experimental conditions: {len(gpt_conditions)}")
    
    # Extract unique experimental conditions from LLaMA
    llama_conditions = defaultdict(list)
    for result in llama_results:
        # Create condition key from experimental parameters
        key = f"{result['bet_type']}_{result['first_result']}_{result['prompt_combo']}"
        condition_id = result['condition_id']
        llama_conditions[key].append(condition_id)
    
    print(f"LLaMA unique experimental conditions: {len(llama_conditions)}")
    
    # Check overlap
    common_conditions = set(gpt_conditions.keys()) & set(llama_conditions.keys())
    print(f"Common experimental conditions: {len(common_conditions)}")
    
    # Show some examples
    print("\n" + "="*60)
    print("SAMPLE CONDITION MAPPING")
    print("="*60)
    
    for i, key in enumerate(sorted(common_conditions)[:10]):
        gpt_ids = list(set(gpt_conditions[key]))
        llama_ids = list(set(llama_conditions[key]))
        print(f"\nCondition: {key}")
        print(f"  GPT condition_id: {gpt_ids}")
        print(f"  LLaMA condition_id: {llama_ids}")
    
    # Check if condition_id mapping is consistent
    print("\n" + "="*60)
    print("CONDITION ID CONSISTENCY CHECK")
    print("="*60)
    
    id_mapping = {}
    inconsistent = 0
    
    for key in common_conditions:
        gpt_id = list(set(gpt_conditions[key]))[0]  # Should be unique
        llama_id = list(set(llama_conditions[key]))[0] if len(set(llama_conditions[key])) > 0 else None
        
        if llama_id is not None:
            id_mapping[gpt_id] = llama_id
            if gpt_id != llama_id:
                inconsistent += 1
    
    print(f"Condition IDs that don't match: {inconsistent}/{len(common_conditions)}")
    
    if inconsistent > 0:
        print("\nSample mismatches:")
        count = 0
        for gpt_id, llama_id in id_mapping.items():
            if gpt_id != llama_id and count < 5:
                print(f"  GPT ID {gpt_id} -> LLaMA ID {llama_id}")
                count += 1
    
    # Now calculate bankruptcy rates using MATCHED conditions
    print("\n" + "="*60)
    print("CALCULATING BANKRUPTCY RATES WITH MATCHED CONDITIONS")
    print("="*60)
    
    # Group by experimental condition (not condition_id)
    gpt_bankruptcy_by_condition = defaultdict(list)
    llama_bankruptcy_by_condition = defaultdict(list)
    
    for result in gpt_results:
        key = f"{result['bet_type']}_{result['first_result']}_{result['prompt_combo']}"
        is_bankrupt = result.get('is_bankrupt', False)
        gpt_bankruptcy_by_condition[key].append(int(is_bankrupt))
    
    for result in llama_results:
        key = f"{result['bet_type']}_{result['first_result']}_{result['prompt_combo']}"
        is_bankrupt = result.get('is_bankrupt', False)
        llama_bankruptcy_by_condition[key].append(int(is_bankrupt))
    
    # Calculate rates for common conditions
    gpt_rates = []
    llama_rates = []
    condition_keys = []
    
    for key in sorted(common_conditions):
        if key in gpt_bankruptcy_by_condition and key in llama_bankruptcy_by_condition:
            gpt_rate = np.mean(gpt_bankruptcy_by_condition[key]) * 100
            llama_rate = np.mean(llama_bankruptcy_by_condition[key]) * 100
            
            gpt_rates.append(gpt_rate)
            llama_rates.append(llama_rate)
            condition_keys.append(key)
    
    print(f"Conditions with bankruptcy data: {len(gpt_rates)}")
    
    # Calculate rankings
    gpt_rankings = rankdata([-r for r in gpt_rates], method='average')
    llama_rankings = rankdata([-r for r in llama_rates], method='average')
    
    # Calculate correlation
    correlation, p_value = spearmanr(gpt_rankings, llama_rankings)
    
    print(f"\n" + "="*60)
    print("RANKING CORRELATION (MATCHED CONDITIONS)")
    print("="*60)
    print(f"Spearman ρ = {correlation:.4f}")
    print(f"P-value = {p_value:.2e}")
    print(f"N = {len(gpt_rates)} matched conditions")
    
    # Also try different correlation methods
    from scipy.stats import pearsonr, kendalltau
    
    pearson_r, pearson_p = pearsonr(gpt_rates, llama_rates)
    kendall_tau, kendall_p = kendalltau(gpt_rates, llama_rates)
    
    print(f"\nAlternative correlations:")
    print(f"Pearson r = {pearson_r:.4f} (p = {pearson_p:.2e})")
    print(f"Kendall τ = {kendall_tau:.4f} (p = {kendall_p:.2e})")
    
    # Show distribution of bankruptcy rates
    print(f"\n" + "="*60)
    print("BANKRUPTCY RATE DISTRIBUTIONS")
    print("="*60)
    
    gpt_zeros = sum(1 for r in gpt_rates if r == 0)
    llama_zeros = sum(1 for r in llama_rates if r == 0)
    
    print(f"GPT: {gpt_zeros}/{len(gpt_rates)} conditions with 0% bankruptcy")
    print(f"LLaMA: {llama_zeros}/{len(llama_rates)} conditions with 0% bankruptcy")
    
    # Show top risky conditions
    print(f"\n" + "="*60)
    print("TOP RISKY CONDITIONS")
    print("="*60)
    
    # Sort by GPT bankruptcy rate
    sorted_indices = np.argsort(gpt_rates)[::-1]
    
    print("\nTop 10 by GPT bankruptcy rate:")
    for i in range(min(10, len(sorted_indices))):
        idx = sorted_indices[i]
        key = condition_keys[idx]
        parts = key.split('_')
        print(f"{i+1}. {key}")
        print(f"   GPT: {gpt_rates[idx]:.1f}% (rank {gpt_rankings[idx]:.1f})")
        print(f"   LLaMA: {llama_rates[idx]:.1f}% (rank {llama_rankings[idx]:.1f})")
        print(f"   Bet: {parts[0]}, First: {parts[1]}, Prompt: {parts[2]}")
    
    return correlation, p_value, len(gpt_rates)

if __name__ == '__main__':
    correlation, p_value, n_conditions = analyze_conditions()
    
    print(f"\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"When matching by actual experimental conditions (not condition_id):")
    print(f"Spearman ρ = {correlation:.4f} (p = {p_value:.2e})")
    print(f"N = {n_conditions} matched conditions")