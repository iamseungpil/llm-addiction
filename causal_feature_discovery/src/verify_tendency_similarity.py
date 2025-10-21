#!/usr/bin/env python3
"""
Verify if GPT and LLaMA have similar tendencies/patterns despite different magnitudes
"""

import json
import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr, pearsonr, kendalltau
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def normalize_prompt(prompt):
    """Normalize GPT prompts to match LLaMA format"""
    if 'PR' in prompt:
        if prompt == 'PR': return 'P'
        elif prompt == 'PRW': return 'RWP'
        elif prompt == 'GPR': return 'GP'
        elif prompt == 'MPR': return 'MP'
        elif prompt == 'GMPR': return 'GMP'
        elif prompt == 'GPRW': return 'GRWP'
        elif prompt == 'MPRW': return 'MRWP'
        elif prompt == 'GMPRW': return 'GMRWP'
        else: return prompt.replace('PR', 'P')
    
    if prompt == 'GMPW': return 'GMWP'
    elif prompt == 'GPW': return 'GWP'
    elif prompt == 'MPW': return 'MWP'
    elif prompt == 'PW': return 'WP'
    
    return prompt

def verify_similar_tendencies():
    """Check if GPT and LLaMA have similar patterns despite different magnitudes"""
    
    print("="*60)
    print("TENDENCY SIMILARITY ANALYSIS")
    print("="*60)
    
    # Load data
    print("\nLoading datasets...")
    with open('/data/llm_addiction/gpt_results_corrected/gpt_corrected_complete_20250825_212628.json', 'r') as f:
        gpt_data = json.load(f)
    
    with open('/data/llm_addiction/results/exp1_multiround_intermediate_20250819_140040.json', 'r') as f:
        llama_data1 = json.load(f)
    with open('/data/llm_addiction/results/exp1_missing_complete_20250820_090040.json', 'r') as f:
        llama_data2 = json.load(f)
    
    gpt_results = gpt_data['results']
    llama_results = llama_data1['results'] + llama_data2['results']
    
    print(f"GPT: {len(gpt_results)}, LLaMA: {len(llama_results)}")
    
    # Method 1: Relative Risk Analysis
    print("\n" + "="*60)
    print("METHOD 1: RELATIVE RISK ANALYSIS")
    print("="*60)
    
    # Calculate baseline rates
    gpt_baseline = np.mean([r.get('is_bankrupt', False) for r in gpt_results]) * 100
    llama_baseline = np.mean([r.get('is_bankrupt', False) for r in llama_results]) * 100
    
    print(f"Baseline bankruptcy rates:")
    print(f"  GPT:   {gpt_baseline:.2f}%")
    print(f"  LLaMA: {llama_baseline:.2f}%")
    
    # Calculate relative risk for each condition
    relative_risks = []
    
    # Group by conditions
    gpt_conditions = defaultdict(list)
    llama_conditions = defaultdict(list)
    
    for r in gpt_results:
        key = f"{r['bet_type']}_{r['first_result']}_{normalize_prompt(r['prompt_combo'])}"
        gpt_conditions[key].append(int(r.get('is_bankrupt', False)))
    
    for r in llama_results:
        key = f"{r['bet_type']}_{r['first_result']}_{r['prompt_combo']}"
        llama_conditions[key].append(int(r.get('is_bankrupt', False)))
    
    common_conditions = set(gpt_conditions.keys()) & set(llama_conditions.keys())
    
    print(f"\nCommon conditions: {len(common_conditions)}")
    
    gpt_relative = []
    llama_relative = []
    
    for cond in common_conditions:
        # Calculate relative risk (rate / baseline)
        gpt_rate = np.mean(gpt_conditions[cond]) * 100
        llama_rate = np.mean(llama_conditions[cond]) * 100
        
        gpt_rel = gpt_rate / gpt_baseline if gpt_baseline > 0 else 0
        llama_rel = llama_rate / llama_baseline if llama_baseline > 0 else 0
        
        gpt_relative.append(gpt_rel)
        llama_relative.append(llama_rel)
    
    # Correlation of relative risks
    if len(gpt_relative) > 1:
        rho_relative, p_relative = spearmanr(gpt_relative, llama_relative)
        print(f"\nRelative risk correlation: ρ = {rho_relative:.4f} (p = {p_relative:.4f})")
    
    # Method 2: Z-Score Normalization
    print("\n" + "="*60)
    print("METHOD 2: Z-SCORE NORMALIZATION")
    print("="*60)
    
    gpt_rates = []
    llama_rates = []
    
    for cond in common_conditions:
        gpt_rate = np.mean(gpt_conditions[cond]) * 100
        llama_rate = np.mean(llama_conditions[cond]) * 100
        gpt_rates.append(gpt_rate)
        llama_rates.append(llama_rate)
    
    # Standardize (z-score)
    scaler = StandardScaler()
    gpt_z = scaler.fit_transform(np.array(gpt_rates).reshape(-1, 1)).flatten()
    llama_z = scaler.fit_transform(np.array(llama_rates).reshape(-1, 1)).flatten()
    
    rho_z, p_z = spearmanr(gpt_z, llama_z)
    pearson_z, p_pearson_z = pearsonr(gpt_z, llama_z)
    
    print(f"Z-normalized Spearman: ρ = {rho_z:.4f} (p = {p_z:.4f})")
    print(f"Z-normalized Pearson:  r = {pearson_z:.4f} (p = {p_pearson_z:.4f})")
    
    # Method 3: Risk Factor Analysis
    print("\n" + "="*60)
    print("METHOD 3: RISK FACTOR ANALYSIS")
    print("="*60)
    
    risk_factors = {
        'variable': [],
        'first_loss': [],
        'has_G': [],
        'has_M': [],
        'has_P': [],
        'has_R': [],
        'has_W': [],
        'complexity': []
    }
    
    # Calculate effect of each risk factor
    for model_name, results in [('GPT', gpt_results), ('LLaMA', llama_results)]:
        # Variable betting effect
        var_rate = np.mean([r.get('is_bankrupt', False) for r in results if r['bet_type'] == 'variable']) * 100
        fix_rate = np.mean([r.get('is_bankrupt', False) for r in results if r['bet_type'] == 'fixed']) * 100
        risk_factors['variable'].append(var_rate - fix_rate)
        
        # First loss effect
        loss_rate = np.mean([r.get('is_bankrupt', False) for r in results if r['first_result'] == 'L']) * 100
        win_rate = np.mean([r.get('is_bankrupt', False) for r in results if r['first_result'] == 'W']) * 100
        risk_factors['first_loss'].append(loss_rate - win_rate)
        
        # Component effects
        for comp in ['G', 'M', 'P', 'R', 'W']:
            key = f'has_{comp}'
            if model_name == 'GPT':
                with_comp = [r.get('is_bankrupt', False) for r in results 
                           if normalize_prompt(r['prompt_combo']) != 'BASE' and comp in normalize_prompt(r['prompt_combo'])]
                without_comp = [r.get('is_bankrupt', False) for r in results 
                              if normalize_prompt(r['prompt_combo']) == 'BASE' or comp not in normalize_prompt(r['prompt_combo'])]
            else:
                with_comp = [r.get('is_bankrupt', False) for r in results 
                           if r['prompt_combo'] != 'BASE' and comp in r['prompt_combo']]
                without_comp = [r.get('is_bankrupt', False) for r in results 
                              if r['prompt_combo'] == 'BASE' or comp not in r['prompt_combo']]
            
            with_rate = np.mean(with_comp) * 100 if with_comp else 0
            without_rate = np.mean(without_comp) * 100 if without_comp else 0
            risk_factors[key].append(with_rate - without_rate)
    
    print(f"{'Risk Factor':<15} {'GPT Effect':<12} {'LLaMA Effect':<12} {'Same Direction?'}")
    print("-"*55)
    for factor, effects in risk_factors.items():
        if len(effects) == 2:
            gpt_effect, llama_effect = effects
            same_dir = (gpt_effect * llama_effect > 0)  # Same sign
            print(f"{factor:<15} {gpt_effect:>+10.2f}% {llama_effect:>+11.2f}% {'✓' if same_dir else '✗'}")
    
    # Method 4: Ranking Consistency
    print("\n" + "="*60)
    print("METHOD 4: RANKING CONSISTENCY")
    print("="*60)
    
    # Get top 10 risky conditions for each model
    gpt_ranked = sorted([(cond, np.mean(gpt_conditions[cond])*100) for cond in common_conditions], 
                       key=lambda x: x[1], reverse=True)
    llama_ranked = sorted([(cond, np.mean(llama_conditions[cond])*100) for cond in common_conditions],
                         key=lambda x: x[1], reverse=True)
    
    gpt_top10 = set([c[0] for c in gpt_ranked[:10]])
    llama_top10 = set([c[0] for c in llama_ranked[:10]])
    
    overlap = gpt_top10 & llama_top10
    print(f"Top 10 overlap: {len(overlap)}/10 conditions")
    
    if overlap:
        print("\nShared high-risk conditions:")
        for cond in overlap:
            gpt_rate = next(r[1] for r in gpt_ranked if r[0] == cond)
            llama_rate = next(r[1] for r in llama_ranked if r[0] == cond)
            print(f"  {cond}: GPT {gpt_rate:.1f}%, LLaMA {llama_rate:.1f}%")
    
    # Method 5: Pattern Matching
    print("\n" + "="*60)
    print("METHOD 5: PATTERN MATCHING")
    print("="*60)
    
    # Check if both models show same patterns
    patterns = {
        'Variable > Fixed': [],
        'Loss > Win': [],
        'Complex > Simple': [],
        'Goal increases risk': []
    }
    
    for model_name, results in [('GPT', gpt_results), ('LLaMA', llama_results)]:
        # Variable > Fixed
        var = np.mean([r.get('is_bankrupt', False) for r in results if r['bet_type'] == 'variable'])
        fix = np.mean([r.get('is_bankrupt', False) for r in results if r['bet_type'] == 'fixed'])
        patterns['Variable > Fixed'].append(var > fix)
        
        # Loss > Win
        loss = np.mean([r.get('is_bankrupt', False) for r in results if r['first_result'] == 'L'])
        win = np.mean([r.get('is_bankrupt', False) for r in results if r['first_result'] == 'W'])
        patterns['Loss > Win'].append(loss > win)
        
        # Complex > Simple
        if model_name == 'GPT':
            simple = [r.get('is_bankrupt', False) for r in results 
                     if normalize_prompt(r['prompt_combo']) in ['BASE', 'G', 'M', 'P', 'R', 'W']]
            complex = [r.get('is_bankrupt', False) for r in results 
                      if len(normalize_prompt(r['prompt_combo'])) > 2]
        else:
            simple = [r.get('is_bankrupt', False) for r in results 
                     if r['prompt_combo'] in ['BASE', 'G', 'M', 'P', 'R', 'W']]
            complex = [r.get('is_bankrupt', False) for r in results 
                      if len(r['prompt_combo']) > 2]
        
        simple_rate = np.mean(simple) if simple else 0
        complex_rate = np.mean(complex) if complex else 0
        patterns['Complex > Simple'].append(complex_rate > simple_rate)
        
        # Goal increases risk
        if model_name == 'GPT':
            with_g = [r.get('is_bankrupt', False) for r in results 
                     if 'G' in normalize_prompt(r['prompt_combo'])]
            without_g = [r.get('is_bankrupt', False) for r in results 
                        if 'G' not in normalize_prompt(r['prompt_combo'])]
        else:
            with_g = [r.get('is_bankrupt', False) for r in results 
                     if 'G' in r['prompt_combo']]
            without_g = [r.get('is_bankrupt', False) for r in results 
                        if 'G' not in r['prompt_combo']]
        
        patterns['Goal increases risk'].append(np.mean(with_g) > np.mean(without_g))
    
    print("Pattern agreement:")
    agreement_count = 0
    for pattern, results in patterns.items():
        if len(results) == 2:
            agrees = results[0] == results[1]
            agreement_count += agrees
            print(f"  {pattern:<20}: {'✓ Agree' if agrees else '✗ Disagree'}")
    
    print(f"\nOverall pattern agreement: {agreement_count}/{len(patterns)} ({agreement_count/len(patterns)*100:.0f}%)")
    
    # Final Summary
    print("\n" + "="*60)
    print("SUMMARY: TENDENCY SIMILARITY")
    print("="*60)
    
    print(f"1. Relative risk correlation: ρ = {rho_relative:.3f}")
    print(f"2. Z-normalized correlation:  ρ = {rho_z:.3f}")
    print(f"3. Risk factor agreement:     {sum(1 for f, e in risk_factors.items() if len(e)==2 and e[0]*e[1]>0)}/{len([f for f in risk_factors if len(risk_factors[f])==2])}")
    print(f"4. Top-10 overlap:            {len(overlap)}/10")
    print(f"5. Pattern agreement:         {agreement_count}/{len(patterns)}")
    
    if rho_relative > 0.3 or rho_z > 0.3 or agreement_count >= 3:
        print("\n✓ CONCLUSION: Models show SIMILAR TENDENCIES despite different magnitudes")
    else:
        print("\n✗ CONCLUSION: Models show DIFFERENT TENDENCIES")

if __name__ == '__main__':
    verify_similar_tendencies()