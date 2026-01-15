#!/usr/bin/env python3
"""
Find GMPW and similar high-risk conditions in both datasets
GPT uses PR notation (GMPW, GMPRW) while LLaMA uses P notation (GMWP, GMRWP)
"""

import json
import numpy as np
from collections import defaultdict

def load_all_data():
    """Load complete datasets"""
    print("Loading datasets...")
    
    # GPT data
    print("Loading GPT data...")
    with open('/data/llm_addiction/gpt_results_corrected/gpt_corrected_complete_20250825_212628.json', 'r') as f:
        gpt_data = json.load(f)
    print(f"  Loaded {len(gpt_data['results'])} GPT experiments")
    
    # LLaMA data - both files
    print("Loading LLaMA data (file 1)...")
    with open('/data/llm_addiction/results/exp1_multiround_intermediate_20250819_140040.json', 'r') as f:
        llama_data1 = json.load(f)
    print(f"  Loaded {len(llama_data1['results'])} experiments from file 1")
    
    print("Loading LLaMA data (file 2)...")
    with open('/data/llm_addiction/results/exp1_missing_complete_20250820_090040.json', 'r') as f:
        llama_data2 = json.load(f)
    print(f"  Loaded {len(llama_data2['results'])} experiments from file 2")
    
    all_llama = llama_data1['results'] + llama_data2['results']
    print(f"  Total LLaMA experiments: {len(all_llama)}")
    
    return gpt_data['results'], all_llama

def analyze_high_risk_conditions():
    """Find and match high-risk conditions between datasets"""
    
    gpt_results, llama_results = load_all_data()
    
    print("\n" + "="*60)
    print("HIGH-RISK CONDITION ANALYSIS")
    print("="*60)
    
    # Group by prompt_combo for easier analysis
    gpt_by_prompt = defaultdict(list)
    llama_by_prompt = defaultdict(list)
    
    for result in gpt_results:
        prompt = result['prompt_combo']
        is_bankrupt = result.get('is_bankrupt', False)
        gpt_by_prompt[prompt].append(int(is_bankrupt))
    
    for result in llama_results:
        prompt = result['prompt_combo']
        is_bankrupt = result.get('is_bankrupt', False)
        llama_by_prompt[prompt].append(int(is_bankrupt))
    
    # Calculate bankruptcy rates by prompt combo
    gpt_prompt_rates = {}
    for prompt, bankruptcies in gpt_by_prompt.items():
        if len(bankruptcies) > 0:
            gpt_prompt_rates[prompt] = (np.mean(bankruptcies) * 100, len(bankruptcies))
    
    llama_prompt_rates = {}
    for prompt, bankruptcies in llama_by_prompt.items():
        if len(bankruptcies) > 0:
            llama_prompt_rates[prompt] = (np.mean(bankruptcies) * 100, len(bankruptcies))
    
    # Sort by bankruptcy rate
    gpt_sorted = sorted(gpt_prompt_rates.items(), key=lambda x: x[1][0], reverse=True)
    llama_sorted = sorted(llama_prompt_rates.items(), key=lambda x: x[1][0], reverse=True)
    
    print("\n" + "="*60)
    print("TOP 10 RISKIEST PROMPTS - GPT")
    print("="*60)
    for i, (prompt, (rate, count)) in enumerate(gpt_sorted[:10]):
        print(f"{i+1}. {prompt:15} {rate:6.1f}% ({count} experiments)")
    
    print("\n" + "="*60)
    print("TOP 10 RISKIEST PROMPTS - LLaMA")
    print("="*60)
    for i, (prompt, (rate, count)) in enumerate(llama_sorted[:10]):
        print(f"{i+1}. {prompt:15} {rate:6.1f}% ({count} experiments)")
    
    # Check specific high-risk conditions
    print("\n" + "="*60)
    print("SEARCHING FOR GMPW AND SIMILAR CONDITIONS")
    print("="*60)
    
    # GPT uses PR notation
    gpt_high_risk = ['GMPW', 'GMPRW', 'PRW', 'GPW', 'MPW', 'GMPR', 'GPR', 'MPR']
    
    # LLaMA equivalent (P instead of PR)
    llama_equivalents = {
        'GMPW': 'GMWP',
        'GMPRW': 'GMRWP', 
        'PRW': 'RWP',
        'GPW': 'GWP',
        'MPW': 'MWP',
        'GMPR': 'GMRP',
        'GPR': 'GRP',
        'MPR': 'MRP'
    }
    
    print("\n" + "="*60)
    print("CONDITION MAPPING AND BANKRUPTCY RATES")
    print("="*60)
    print(f"{'GPT Condition':<12} {'LLaMA Equiv':<12} {'GPT Rate':<15} {'LLaMA Rate':<15}")
    print("-"*60)
    
    for gpt_cond in gpt_high_risk:
        llama_cond = llama_equivalents.get(gpt_cond, '?')
        
        # Get GPT rate
        if gpt_cond in gpt_prompt_rates:
            gpt_rate, gpt_n = gpt_prompt_rates[gpt_cond]
            gpt_str = f"{gpt_rate:.1f}% (n={gpt_n})"
        else:
            gpt_str = "Not found"
        
        # Get LLaMA rate  
        if llama_cond in llama_prompt_rates:
            llama_rate, llama_n = llama_prompt_rates[llama_cond]
            llama_str = f"{llama_rate:.1f}% (n={llama_n})"
        else:
            llama_str = "Not found"
        
        print(f"{gpt_cond:<12} {llama_cond:<12} {gpt_str:<15} {llama_str:<15}")
    
    # Now check by full experimental condition (bet_type + first_result + prompt)
    print("\n" + "="*60)
    print("FULL CONDITION ANALYSIS (Variable Betting Only)")
    print("="*60)
    
    # Group by full condition
    gpt_full_conditions = defaultdict(list)
    llama_full_conditions = defaultdict(list)
    
    for result in gpt_results:
        if result['bet_type'] == 'variable':
            key = f"{result['bet_type']}_{result['first_result']}_{result['prompt_combo']}"
            is_bankrupt = result.get('is_bankrupt', False)
            gpt_full_conditions[key].append(int(is_bankrupt))
    
    for result in llama_results:
        if result['bet_type'] == 'variable':
            key = f"{result['bet_type']}_{result['first_result']}_{result['prompt_combo']}"
            is_bankrupt = result.get('is_bankrupt', False)
            llama_full_conditions[key].append(int(is_bankrupt))
    
    # Calculate rates
    gpt_full_rates = {}
    for cond, bankruptcies in gpt_full_conditions.items():
        if len(bankruptcies) > 0:
            gpt_full_rates[cond] = np.mean(bankruptcies) * 100
    
    llama_full_rates = {}
    for cond, bankruptcies in llama_full_conditions.items():
        if len(bankruptcies) > 0:
            llama_full_rates[cond] = np.mean(bankruptcies) * 100
    
    # Check specific conditions
    print("\nChecking specific variable betting conditions:")
    print(f"{'Condition':<30} {'GPT Rate':<12} {'LLaMA Rate':<12}")
    print("-"*60)
    
    for gpt_prompt in ['GMPW', 'GMPRW', 'PRW']:
        llama_prompt = llama_equivalents.get(gpt_prompt, gpt_prompt)
        
        for first_result in ['W', 'L']:
            gpt_key = f"variable_{first_result}_{gpt_prompt}"
            llama_key = f"variable_{first_result}_{llama_prompt}"
            
            gpt_rate = gpt_full_rates.get(gpt_key, -1)
            llama_rate = llama_full_rates.get(llama_key, -1)
            
            if gpt_rate >= 0 or llama_rate >= 0:
                gpt_str = f"{gpt_rate:.1f}%" if gpt_rate >= 0 else "Not found"
                llama_str = f"{llama_rate:.1f}%" if llama_rate >= 0 else "Not found"
                print(f"{gpt_key:<30} {gpt_str:<12} {llama_str:<12}")

if __name__ == '__main__':
    analyze_high_risk_conditions()