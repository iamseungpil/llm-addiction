#!/usr/bin/env python3
"""
Set-based perfect mapping between GPT and LLaMA conditions
Maps conditions based on the SET of prompt components, not string order
"""

import json
import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

def parse_prompt_components(prompt):
    """
    Parse a prompt string into its component set
    E.g., "GMPW" -> {'G', 'M', 'P', 'W'}
    E.g., "GMPRW" -> {'G', 'M', 'P', 'R', 'W'} in GPT -> {'G', 'M', 'P', 'R', 'W'} same components
    
    IMPORTANT: 
    - GPT uses 'PR' where LLaMA uses just 'P' for probability
    - But 'R' (rules) is still a separate component
    - So GMPRW has all 5 components: G, M, P, R, W
    """
    components = set()
    
    # Handle BASE case
    if prompt == 'BASE':
        return {'BASE'}
    
    # First, handle known GPT patterns by replacing PR with P
    # This correctly identifies that PR means "P and R are both present"
    if 'PR' in prompt:
        # Replace PR with just P, but keep R separate
        prompt_fixed = prompt.replace('PR', 'P')
        # Now add back R if it was part of a compound
        if 'PRW' in prompt:
            # PRW means P + R + W
            prompt_fixed = prompt.replace('PRW', 'PWR').replace('PR', 'P')
        elif 'GMPR' in prompt and 'W' in prompt:
            # GMPRW means G + M + P + R + W
            prompt_fixed = 'GMPRW'
        elif prompt.endswith('PR'):
            # Ends with PR, just means P
            prompt_fixed = prompt.replace('PR', 'P')
        else:
            # General case: PR followed by something means P and R
            prompt_fixed = prompt.replace('PR', 'P')
            if 'R' not in prompt_fixed and len(prompt) > prompt.index('PR') + 2:
                # We had PR followed by more chars, so R is a component
                components.add('R')
    else:
        prompt_fixed = prompt
    
    # Now parse the fixed prompt character by character
    for char in prompt_fixed:
        if char in ['G', 'M', 'P', 'R', 'W']:
            components.add(char)
    
    return components

def create_condition_key_from_set(bet_type, first_result, components):
    """Create a canonical key from condition components"""
    # Sort components for consistent ordering
    sorted_components = sorted(components)
    return f"{bet_type}_{first_result}_{''.join(sorted_components)}"

def load_and_map_with_sets():
    """Load data and perform set-based mapping"""
    
    print("="*60)
    print("SET-BASED CONDITION MAPPING ANALYSIS")
    print("="*60)
    
    # Load GPT data
    print("\nLoading GPT data...")
    with open('/data/llm_addiction/gpt_results_corrected/gpt_corrected_complete_20250825_212628.json', 'r') as f:
        gpt_data = json.load(f)
    print(f"  Loaded {len(gpt_data['results'])} GPT experiments")
    
    # Load LLaMA data - both files for complete 6400
    print("\nLoading LLaMA data...")
    print("  File 1: exp1_multiround_intermediate_20250819_140040.json")
    with open('/data/llm_addiction/results/exp1_multiround_intermediate_20250819_140040.json', 'r') as f:
        llama_data1 = json.load(f)
    print(f"    Loaded {len(llama_data1['results'])} experiments")
    
    print("  File 2: exp1_missing_complete_20250820_090040.json")
    with open('/data/llm_addiction/results/exp1_missing_complete_20250820_090040.json', 'r') as f:
        llama_data2 = json.load(f)
    print(f"    Loaded {len(llama_data2['results'])} experiments")
    
    all_llama = llama_data1['results'] + llama_data2['results']
    print(f"\n  Total LLaMA experiments: {len(all_llama)}")
    
    # Build condition dictionaries using SET-based keys
    print("\n" + "="*60)
    print("BUILDING SET-BASED CONDITION MAPS")
    print("="*60)
    
    # GPT conditions with set-based keys
    gpt_by_set = defaultdict(list)
    gpt_original_keys = {}  # Track original prompt strings
    
    for result in gpt_data['results']:
        bet_type = result['bet_type']
        first_result = result['first_result']
        prompt = result['prompt_combo']
        
        # Parse prompt into components
        components = parse_prompt_components(prompt)
        set_key = create_condition_key_from_set(bet_type, first_result, components)
        
        # Store bankruptcy data
        is_bankrupt = result.get('is_bankrupt', False)
        gpt_by_set[set_key].append(int(is_bankrupt))
        
        # Track original prompt string
        if set_key not in gpt_original_keys:
            gpt_original_keys[set_key] = f"{bet_type}_{first_result}_{prompt}"
    
    print(f"GPT unique set-based conditions: {len(gpt_by_set)}")
    
    # LLaMA conditions with set-based keys
    llama_by_set = defaultdict(list)
    llama_original_keys = {}
    
    for result in all_llama:
        bet_type = result['bet_type']
        first_result = result['first_result']
        prompt = result['prompt_combo']
        
        # Parse prompt into components
        components = parse_prompt_components(prompt)
        set_key = create_condition_key_from_set(bet_type, first_result, components)
        
        # Store bankruptcy data
        is_bankrupt = result.get('is_bankrupt', False)
        llama_by_set[set_key].append(int(is_bankrupt))
        
        # Track original prompt string
        if set_key not in llama_original_keys:
            llama_original_keys[set_key] = f"{bet_type}_{first_result}_{prompt}"
    
    print(f"LLaMA unique set-based conditions: {len(llama_by_set)}")
    
    # Find perfect matches
    common_keys = set(gpt_by_set.keys()) & set(llama_by_set.keys())
    gpt_only = set(gpt_by_set.keys()) - set(llama_by_set.keys())
    llama_only = set(llama_by_set.keys()) - set(gpt_by_set.keys())
    
    print(f"\nPerfectly matched conditions: {len(common_keys)}")
    print(f"GPT-only conditions: {len(gpt_only)}")
    print(f"LLaMA-only conditions: {len(llama_only)}")
    
    # Show some examples of matches
    print("\n" + "="*60)
    print("SAMPLE MATCHED CONDITIONS")
    print("="*60)
    
    for i, key in enumerate(sorted(common_keys)[:10]):
        gpt_orig = gpt_original_keys[key]
        llama_orig = llama_original_keys[key]
        gpt_rate = np.mean(gpt_by_set[key]) * 100
        llama_rate = np.mean(llama_by_set[key]) * 100
        
        print(f"\n{i+1}. Set key: {key}")
        print(f"   GPT original: {gpt_orig}")
        print(f"   LLaMA original: {llama_orig}")
        print(f"   Bankruptcy: GPT {gpt_rate:.1f}% (n={len(gpt_by_set[key])}), LLaMA {llama_rate:.1f}% (n={len(llama_by_set[key])})")
    
    # Verify high-risk conditions
    print("\n" + "="*60)
    print("HIGH-RISK CONDITION VERIFICATION")
    print("="*60)
    
    high_risk_prompts = ['GMPW', 'GMPRW', 'PRW', 'GPW', 'MPW']
    
    for hr_prompt in high_risk_prompts:
        print(f"\n{hr_prompt}:")
        components = parse_prompt_components(hr_prompt)
        
        # Check both bet types and first results
        found_any = False
        for bet_type in ['fixed', 'variable']:
            for first_result in ['W', 'L']:
                set_key = create_condition_key_from_set(bet_type, first_result, components)
                
                if set_key in common_keys:
                    gpt_rate = np.mean(gpt_by_set[set_key]) * 100
                    llama_rate = np.mean(llama_by_set[set_key]) * 100
                    print(f"  {bet_type}_{first_result}: GPT {gpt_rate:.1f}%, LLaMA {llama_rate:.1f}%")
                    found_any = True
        
        if not found_any:
            print(f"  Not found in common conditions")
    
    # Calculate ranking correlation
    print("\n" + "="*60)
    print("RANKING CORRELATION ANALYSIS")
    print("="*60)
    
    gpt_rates = []
    llama_rates = []
    condition_labels = []
    
    for key in sorted(common_keys):
        gpt_rate = np.mean(gpt_by_set[key]) * 100
        llama_rate = np.mean(llama_by_set[key]) * 100
        gpt_rates.append(gpt_rate)
        llama_rates.append(llama_rate)
        condition_labels.append(key)
    
    # Calculate rankings
    gpt_ranks = np.argsort(np.argsort(gpt_rates)[::-1]) + 1
    llama_ranks = np.argsort(np.argsort(llama_rates)[::-1]) + 1
    
    # Calculate correlations
    spearman_rho, spearman_p = spearmanr(gpt_ranks, llama_ranks)
    
    print(f"Number of matched conditions: {len(common_keys)}")
    print(f"Spearman ρ = {spearman_rho:.4f} (p = {spearman_p:.2e})")
    
    # Also calculate Pearson on rates
    from scipy.stats import pearsonr
    pearson_r, pearson_p = pearsonr(gpt_rates, llama_rates)
    print(f"Pearson r (on rates) = {pearson_r:.4f} (p = {pearson_p:.2e})")
    
    # Show unmatched conditions
    if gpt_only:
        print("\n" + "="*60)
        print(f"GPT-ONLY CONDITIONS ({len(gpt_only)})")
        print("="*60)
        for key in sorted(gpt_only)[:5]:
            orig = gpt_original_keys[key]
            rate = np.mean(gpt_by_set[key]) * 100
            print(f"  {orig} ({rate:.1f}% bankruptcy)")
    
    if llama_only:
        print("\n" + "="*60)
        print(f"LLAMA-ONLY CONDITIONS ({len(llama_only)})")
        print("="*60)
        for key in sorted(llama_only)[:5]:
            orig = llama_original_keys[key]
            rate = np.mean(llama_by_set[key]) * 100
            print(f"  {orig} ({rate:.1f}% bankruptcy)")
    
    # Create visualization
    if len(common_keys) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Scatter plot of rankings
        ax1.scatter(gpt_ranks, llama_ranks, alpha=0.6)
        ax1.plot([1, len(gpt_ranks)], [1, len(llama_ranks)], 'r--', alpha=0.5)
        ax1.set_xlabel('GPT-4o-mini Rank', fontweight='bold')
        ax1.set_ylabel('LLaMA-3.1-8B Rank', fontweight='bold')
        ax1.set_title(f'Ranking Comparison (Spearman ρ = {spearman_rho:.3f})', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Direct rate comparison
        ax2.scatter(gpt_rates, llama_rates, alpha=0.6)
        ax2.plot([0, max(gpt_rates)], [0, max(llama_rates)], 'r--', alpha=0.5)
        ax2.set_xlabel('GPT-4o-mini Bankruptcy Rate (%)', fontweight='bold')
        ax2.set_ylabel('LLaMA-3.1-8B Bankruptcy Rate (%)', fontweight='bold')
        ax2.set_title(f'Rate Comparison (Pearson r = {pearson_r:.3f})', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Set-Based Condition Mapping Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = '/home/ubuntu/llm_addiction/writing/figures/set_based_mapping_analysis'
        plt.savefig(f'{output_path}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_path}.pdf', dpi=300, bbox_inches='tight')
        print(f"\n✅ Plot saved: {output_path}.png/pdf")
    
    return common_keys, gpt_by_set, llama_by_set

if __name__ == '__main__':
    common, gpt_conditions, llama_conditions = load_and_map_with_sets()
    
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Successfully matched: {len(common)}/{128} conditions ({len(common)/128*100:.1f}%)")
    print(f"Expected: 128 perfect 1:1 matches")
    
    if len(common) < 128:
        print(f"\n⚠️ WARNING: Only {len(common)} conditions matched!")
        print("This suggests differences in experimental design or data collection between GPT and LLaMA")