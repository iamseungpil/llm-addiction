#!/usr/bin/env python3
"""
Analyze GPT-LLaMA correlation focusing on prompts (32 types)
Using the correct LLaMA data files
"""

import json
import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

def load_correct_data():
    """Load the correct, complete datasets"""
    print("Loading correct datasets...")
    
    # GPT data - confirmed correct
    with open('/data/llm_addiction/gpt_results_corrected/gpt_corrected_complete_20250825_212628.json', 'r') as f:
        gpt_data = json.load(f)
    print(f"✓ GPT: {len(gpt_data['results'])} experiments")
    
    # LLaMA data - the main file from Aug 19 (most recent complete)
    print("Loading LLaMA main file (14GB)...")
    with open('/data/llm_addiction/results/exp1_multiround_intermediate_20250819_140040.json', 'r') as f:
        llama_main = json.load(f)
    print(f"✓ LLaMA main: {len(llama_main['results'])} experiments")
    
    # LLaMA additional data
    print("Loading LLaMA additional file...")
    with open('/data/llm_addiction/results/exp1_missing_complete_20250820_090040.json', 'r') as f:
        llama_add = json.load(f)
    print(f"✓ LLaMA additional: {len(llama_add['results'])} experiments")
    
    all_llama = llama_main['results'] + llama_add['results']
    print(f"✓ Total LLaMA: {len(all_llama)} experiments")
    
    return gpt_data['results'], all_llama

def normalize_prompt(prompt):
    """Normalize GPT prompts to match LLaMA format"""
    # GPT uses PR for probability, LLaMA uses just P
    if 'PR' in prompt:
        # Handle different PR cases
        if prompt == 'PR':
            return 'P'
        elif prompt == 'PRW':
            return 'RWP'
        elif prompt == 'GPR':
            return 'GP'
        elif prompt == 'MPR':
            return 'MP'
        elif prompt == 'GMPR':
            return 'GMP'
        elif prompt == 'GPRW':
            return 'GRWP'
        elif prompt == 'MPRW':
            return 'MRWP'
        elif prompt == 'GMPRW':
            return 'GMRWP'
        else:
            # General case: replace PR with P
            return prompt.replace('PR', 'P')
    
    # Also handle order differences
    if prompt == 'GMPW':
        return 'GMWP'
    elif prompt == 'GPW':
        return 'GWP'
    elif prompt == 'MPW':
        return 'MWP'
    elif prompt == 'PW':
        return 'WP'
    
    return prompt

def analyze_prompt_correlation():
    """Analyze correlation for 32 prompt combinations"""
    
    gpt_results, llama_results = load_correct_data()
    
    print("\n" + "="*60)
    print("PROMPT-LEVEL ANALYSIS (32 types)")
    print("="*60)
    
    # Group by prompt (aggregating across bet types and first results)
    gpt_by_prompt = defaultdict(list)
    llama_by_prompt = defaultdict(list)
    
    # Process GPT data
    for result in gpt_results:
        prompt = normalize_prompt(result['prompt_combo'])
        is_bankrupt = result.get('is_bankrupt', False)
        gpt_by_prompt[prompt].append(int(is_bankrupt))
    
    # Process LLaMA data
    for result in llama_results:
        prompt = result['prompt_combo']
        is_bankrupt = result.get('is_bankrupt', False)
        llama_by_prompt[prompt].append(int(is_bankrupt))
    
    print(f"\nGPT unique prompts: {len(gpt_by_prompt)}")
    print(f"LLaMA unique prompts: {len(llama_by_prompt)}")
    
    # Find common prompts
    common_prompts = set(gpt_by_prompt.keys()) & set(llama_by_prompt.keys())
    print(f"Common prompts: {len(common_prompts)}")
    
    # Calculate bankruptcy rates for common prompts
    prompt_data = []
    for prompt in sorted(common_prompts):
        gpt_rate = np.mean(gpt_by_prompt[prompt]) * 100
        llama_rate = np.mean(llama_by_prompt[prompt]) * 100
        gpt_n = len(gpt_by_prompt[prompt])
        llama_n = len(llama_by_prompt[prompt])
        
        prompt_data.append({
            'prompt': prompt,
            'gpt_rate': gpt_rate,
            'llama_rate': llama_rate,
            'gpt_n': gpt_n,
            'llama_n': llama_n
        })
    
    # Sort by average bankruptcy rate
    prompt_data.sort(key=lambda x: (x['gpt_rate'] + x['llama_rate'])/2, reverse=True)
    
    print("\n" + "="*60)
    print("TOP 15 RISKIEST PROMPTS")
    print("="*60)
    print(f"{'Prompt':<10} {'GPT Rate':<12} {'LLaMA Rate':<12} {'Diff':<10} {'N (GPT/LLaMA)':<15}")
    print("-"*60)
    
    for i, data in enumerate(prompt_data[:15]):
        diff = data['gpt_rate'] - data['llama_rate']
        print(f"{data['prompt']:<10} {data['gpt_rate']:>6.1f}% "
              f"{data['llama_rate']:>11.1f}% {diff:>9.1f}% "
              f"{data['gpt_n']:>7}/{data['llama_n']:<7}")
    
    # Calculate correlations
    gpt_rates = [d['gpt_rate'] for d in prompt_data]
    llama_rates = [d['llama_rate'] for d in prompt_data]
    
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    
    spearman_rho, spearman_p = spearmanr(gpt_rates, llama_rates)
    pearson_r, pearson_p = pearsonr(gpt_rates, llama_rates)
    
    print(f"Spearman ρ = {spearman_rho:.4f} (p = {spearman_p:.4f})")
    print(f"Pearson r = {pearson_r:.4f} (p = {pearson_p:.4f})")
    
    # Analyze prompt complexity
    print("\n" + "="*60)
    print("COMPLEXITY ANALYSIS")
    print("="*60)
    
    complexity_gpt = defaultdict(list)
    complexity_llama = defaultdict(list)
    
    for data in prompt_data:
        prompt = data['prompt']
        if prompt == 'BASE':
            n_components = 0
        else:
            n_components = len(set(c for c in prompt if c in 'GMPRW'))
        
        complexity_gpt[n_components].append(data['gpt_rate'])
        complexity_llama[n_components].append(data['llama_rate'])
    
    print(f"{'Components':<12} {'GPT Mean':<12} {'LLaMA Mean':<12} {'N prompts':<10}")
    print("-"*50)
    
    for n in sorted(complexity_gpt.keys()):
        gpt_mean = np.mean(complexity_gpt[n])
        llama_mean = np.mean(complexity_llama[n])
        n_prompts = len(complexity_gpt[n])
        print(f"{n:<12} {gpt_mean:>6.1f}% {llama_mean:>11.1f}% {n_prompts:>9}")
    
    return prompt_data

def analyze_specific_components():
    """Analyze effect of specific prompt components"""
    
    gpt_results, llama_results = load_correct_data()
    
    print("\n" + "="*60)
    print("INDIVIDUAL COMPONENT EFFECTS")
    print("="*60)
    
    components = {
        'G': 'Goal (2x money)',
        'M': 'Maximize reward',
        'P': 'Probability info',
        'R': 'Hidden rules',
        'W': 'Reward info (3x)'
    }
    
    for comp, desc in components.items():
        gpt_with = []
        gpt_without = []
        llama_with = []
        llama_without = []
        
        # GPT
        for result in gpt_results:
            prompt = normalize_prompt(result['prompt_combo'])
            is_bankrupt = result.get('is_bankrupt', False)
            
            if prompt != 'BASE' and comp in prompt:
                gpt_with.append(int(is_bankrupt))
            else:
                gpt_without.append(int(is_bankrupt))
        
        # LLaMA
        for result in llama_results:
            prompt = result['prompt_combo']
            is_bankrupt = result.get('is_bankrupt', False)
            
            if prompt != 'BASE' and comp in prompt:
                llama_with.append(int(is_bankrupt))
            else:
                llama_without.append(int(is_bankrupt))
        
        gpt_with_rate = np.mean(gpt_with) * 100 if gpt_with else 0
        gpt_without_rate = np.mean(gpt_without) * 100 if gpt_without else 0
        llama_with_rate = np.mean(llama_with) * 100 if llama_with else 0
        llama_without_rate = np.mean(llama_without) * 100 if llama_without else 0
        
        print(f"\n{comp}: {desc}")
        print(f"  With {comp}:    GPT {gpt_with_rate:5.1f}% (n={len(gpt_with):4}), "
              f"LLaMA {llama_with_rate:5.1f}% (n={len(llama_with):5})")
        print(f"  Without {comp}: GPT {gpt_without_rate:5.1f}% (n={len(gpt_without):4}), "
              f"LLaMA {llama_without_rate:5.1f}% (n={len(llama_without):5})")
        print(f"  Effect:        GPT {gpt_with_rate - gpt_without_rate:+5.1f}%, "
              f"LLaMA {llama_with_rate - llama_without_rate:+5.1f}%")

if __name__ == '__main__':
    print("="*60)
    print("GPT-LLAMA PROMPT CORRELATION ANALYSIS")
    print("="*60)
    print("Using correct data files:")
    print("- GPT: gpt_corrected_complete_20250825_212628.json")
    print("- LLaMA: exp1_multiround_intermediate_20250819_140040.json (5780)")
    print("- LLaMA: exp1_missing_complete_20250820_090040.json (620)")
    print("="*60)
    
    prompt_data = analyze_prompt_correlation()
    analyze_specific_components()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
