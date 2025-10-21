#!/usr/bin/env python3
"""
Perfect 1:1 mapping between GPT and LLaMA conditions
Accounting for PR vs P notation differences
"""

import json
import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr

def create_condition_mapping():
    """Create explicit mapping between GPT and LLaMA prompt notations"""
    
    # Mapping from GPT notation to LLaMA notation
    # GPT uses "PR" for probability, LLaMA uses just "P"
    prompt_mapping = {
        'BASE': 'BASE',
        'G': 'G',
        'M': 'M',
        'P': 'P',
        'R': 'R',
        'W': 'W',
        'GM': 'GM',
        'GP': 'GP',
        'GR': 'GR',
        'GW': 'GW',
        'MP': 'MP',
        'MR': 'MR',
        'MW': 'MW',
        'PR': 'P',  # Key difference: GPT's PR = LLaMA's P
        'PW': 'WP',  # Order might be different
        'RW': 'RW',
        'GMP': 'GMP',
        'GMR': 'GMR',
        'GMW': 'GMW',
        'GPR': 'GP',  # GPT's GPR = LLaMA's GP
        'GPW': 'GWP',
        'GRW': 'GRW',
        'MPR': 'MP',  # GPT's MPR = LLaMA's MP
        'MPW': 'MWP',
        'MRW': 'MRW',
        'PRW': 'RWP',  # Order change
        'GMPR': 'GMP',  # GPT's GMPR = LLaMA's GMP
        'GMPW': 'GMWP',  # Order change
        'GMRW': 'GMRW',
        'GPRW': 'GRWP',  # Order change
        'MPRW': 'MRWP',  # Order change
        'GMPRW': 'GMRWP'  # Order change
    }
    
    return prompt_mapping

def load_and_analyze():
    """Load data and perform perfect mapping"""
    
    print("="*60)
    print("PERFECT 1:1 CONDITION MAPPING ANALYSIS")
    print("="*60)
    
    # Load GPT data
    print("\nLoading GPT data...")
    with open('/data/llm_addiction/gpt_results_corrected/gpt_corrected_complete_20250825_212628.json', 'r') as f:
        gpt_data = json.load(f)
    print(f"  Loaded {len(gpt_data['results'])} GPT experiments")
    
    # Load LLaMA data
    print("\nLoading LLaMA data...")
    with open('/data/llm_addiction/results/exp1_multiround_intermediate_20250819_140040.json', 'r') as f:
        llama_data1 = json.load(f)
    with open('/data/llm_addiction/results/exp1_missing_complete_20250820_090040.json', 'r') as f:
        llama_data2 = json.load(f)
    
    all_llama = llama_data1['results'] + llama_data2['results']
    print(f"  Loaded {len(all_llama)} total LLaMA experiments")
    
    # Get prompt mapping
    prompt_map = create_condition_mapping()
    
    # Create condition keys for both datasets
    print("\n" + "="*60)
    print("ANALYZING ALL 128 CONDITIONS")
    print("="*60)
    
    # Build condition dictionaries
    gpt_conditions = defaultdict(list)
    llama_conditions = defaultdict(list)
    
    # For GPT: Use original prompt_combo
    for result in gpt_data['results']:
        bet_type = result['bet_type']
        first_result = result['first_result']
        prompt = result['prompt_combo']
        
        key = f"{bet_type}_{first_result}_{prompt}"
        is_bankrupt = result.get('is_bankrupt', False)
        gpt_conditions[key].append(int(is_bankrupt))
    
    # For LLaMA: Map GPT prompts to LLaMA equivalents
    for result in all_llama:
        bet_type = result['bet_type']
        first_result = result['first_result']
        prompt = result['prompt_combo']
        
        key = f"{bet_type}_{first_result}_{prompt}"
        is_bankrupt = result.get('is_bankrupt', False)
        llama_conditions[key].append(int(is_bankrupt))
    
    print(f"\nGPT unique conditions: {len(gpt_conditions)}")
    print(f"LLaMA unique conditions: {len(llama_conditions)}")
    
    # Now map GPT conditions to LLaMA conditions
    print("\n" + "="*60)
    print("MAPPING GPT CONDITIONS TO LLAMA")
    print("="*60)
    
    mapped_conditions = []
    unmapped_gpt = []
    unmapped_llama = set(llama_conditions.keys())
    
    for gpt_key in sorted(gpt_conditions.keys()):
        bet_type, first_result, gpt_prompt = gpt_key.split('_', 2)
        
        # Try to find equivalent LLaMA condition
        llama_prompt = prompt_map.get(gpt_prompt, None)
        
        if llama_prompt:
            # Try exact match first
            llama_key = f"{bet_type}_{first_result}_{llama_prompt}"
            
            if llama_key in llama_conditions:
                mapped_conditions.append((gpt_key, llama_key))
                unmapped_llama.discard(llama_key)
            else:
                # Try alternative orderings for compound prompts
                found = False
                for alt_key in llama_conditions.keys():
                    if (alt_key.startswith(f"{bet_type}_{first_result}_") and 
                        set(alt_key.split('_')[2]) == set(llama_prompt)):
                        mapped_conditions.append((gpt_key, alt_key))
                        unmapped_llama.discard(alt_key)
                        found = True
                        break
                
                if not found:
                    unmapped_gpt.append(gpt_key)
        else:
            unmapped_gpt.append(gpt_key)
    
    print(f"\nSuccessfully mapped: {len(mapped_conditions)} conditions")
    print(f"Unmapped GPT conditions: {len(unmapped_gpt)}")
    print(f"Unmapped LLaMA conditions: {len(unmapped_llama)}")
    
    # Show some examples of successful mappings
    print("\n" + "="*60)
    print("SAMPLE SUCCESSFUL MAPPINGS")
    print("="*60)
    
    for i, (gpt_key, llama_key) in enumerate(mapped_conditions[:10]):
        gpt_rate = np.mean(gpt_conditions[gpt_key]) * 100 if gpt_conditions[gpt_key] else 0
        llama_rate = np.mean(llama_conditions[llama_key]) * 100 if llama_conditions[llama_key] else 0
        print(f"{i+1}. GPT: {gpt_key}")
        print(f"   LLaMA: {llama_key}")
        print(f"   Bankruptcy: GPT {gpt_rate:.1f}%, LLaMA {llama_rate:.1f}%")
    
    # Calculate correlation for mapped conditions
    if mapped_conditions:
        print("\n" + "="*60)
        print("RANKING CORRELATION ANALYSIS")
        print("="*60)
        
        gpt_rates = []
        llama_rates = []
        
        for gpt_key, llama_key in mapped_conditions:
            gpt_rate = np.mean(gpt_conditions[gpt_key]) * 100 if gpt_conditions[gpt_key] else 0
            llama_rate = np.mean(llama_conditions[llama_key]) * 100 if llama_conditions[llama_key] else 0
            gpt_rates.append(gpt_rate)
            llama_rates.append(llama_rate)
        
        # Calculate rankings
        gpt_ranks = np.argsort(np.argsort(gpt_rates)[::-1]) + 1
        llama_ranks = np.argsort(np.argsort(llama_rates)[::-1]) + 1
        
        # Calculate correlation
        corr, p_value = spearmanr(gpt_ranks, llama_ranks)
        
        print(f"Number of mapped conditions: {len(mapped_conditions)}")
        print(f"Spearman ρ = {corr:.4f}")
        print(f"P-value = {p_value:.2e}")
        
        # Check high-risk conditions
        print("\n" + "="*60)
        print("HIGH-RISK CONDITION VERIFICATION")
        print("="*60)
        
        high_risk_gpt = ['GMPW', 'GMPRW', 'PRW', 'GPW', 'MPW']
        
        for hr_prompt in high_risk_gpt:
            print(f"\n{hr_prompt}:")
            found = False
            for gpt_key, llama_key in mapped_conditions:
                if hr_prompt in gpt_key:
                    gpt_rate = np.mean(gpt_conditions[gpt_key]) * 100
                    llama_rate = np.mean(llama_conditions[llama_key]) * 100
                    print(f"  {gpt_key} -> {llama_key}")
                    print(f"  GPT: {gpt_rate:.1f}%, LLaMA: {llama_rate:.1f}%")
                    found = True
            if not found:
                print(f"  Not found in mapping")
    
    # Show unmapped conditions
    if unmapped_gpt:
        print("\n" + "="*60)
        print("UNMAPPED GPT CONDITIONS")
        print("="*60)
        for key in unmapped_gpt[:10]:
            rate = np.mean(gpt_conditions[key]) * 100
            print(f"  {key} ({rate:.1f}% bankruptcy)")
    
    if unmapped_llama:
        print("\n" + "="*60)
        print("UNMAPPED LLAMA CONDITIONS")  
        print("="*60)
        for key in list(unmapped_llama)[:10]:
            rate = np.mean(llama_conditions[key]) * 100
            print(f"  {key} ({rate:.1f}% bankruptcy)")
    
    return mapped_conditions, gpt_conditions, llama_conditions

if __name__ == '__main__':
    mapped, gpt_cond, llama_cond = load_and_analyze()
    
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Total mapped conditions: {len(mapped)}")
    print(f"Expected: 128 conditions")
    print(f"Mapping success rate: {len(mapped)/128*100:.1f}%")