#!/usr/bin/env python3
"""
Analyze LLaMA bankruptcy patterns - which conditions lead to higher bankruptcy?
"""

import json
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_llama_patterns():
    """Deep dive into LLaMA bankruptcy patterns"""
    
    print("="*60)
    print("LLAMA BANKRUPTCY PATTERN ANALYSIS")
    print("="*60)
    
    # Load LLaMA data
    print("\nLoading LLaMA data...")
    with open('/data/llm_addiction/results/exp1_multiround_intermediate_20250819_140040.json', 'r') as f:
        data1 = json.load(f)
    with open('/data/llm_addiction/results/exp1_missing_complete_20250820_090040.json', 'r') as f:
        data2 = json.load(f)
    
    all_results = data1['results'] + data2['results']
    print(f"Total experiments: {len(all_results)}")
    
    # Calculate total bankruptcies
    total_bankruptcies = sum(1 for r in all_results if r.get('is_bankrupt', False))
    print(f"Total bankruptcies: {total_bankruptcies} ({total_bankruptcies/len(all_results)*100:.1f}%)")
    
    # 1. Analyze by betting type
    print("\n" + "="*60)
    print("1. BETTING TYPE ANALYSIS")
    print("="*60)
    
    fixed_bankruptcies = []
    variable_bankruptcies = []
    
    for result in all_results:
        is_bankrupt = result.get('is_bankrupt', False)
        if result['bet_type'] == 'fixed':
            fixed_bankruptcies.append(int(is_bankrupt))
        else:
            variable_bankruptcies.append(int(is_bankrupt))
    
    fixed_rate = np.mean(fixed_bankruptcies) * 100
    variable_rate = np.mean(variable_bankruptcies) * 100
    
    print(f"Fixed betting:    {fixed_rate:.2f}% (n={len(fixed_bankruptcies)}, {sum(fixed_bankruptcies)} bankruptcies)")
    print(f"Variable betting: {variable_rate:.2f}% (n={len(variable_bankruptcies)}, {sum(variable_bankruptcies)} bankruptcies)")
    
    # 2. Analyze by first result
    print("\n" + "="*60)
    print("2. FIRST RESULT ANALYSIS")
    print("="*60)
    
    first_win_bankruptcies = []
    first_loss_bankruptcies = []
    
    for result in all_results:
        is_bankrupt = result.get('is_bankrupt', False)
        if result['first_result'] == 'W':
            first_win_bankruptcies.append(int(is_bankrupt))
        else:
            first_loss_bankruptcies.append(int(is_bankrupt))
    
    win_rate = np.mean(first_win_bankruptcies) * 100
    loss_rate = np.mean(first_loss_bankruptcies) * 100
    
    print(f"First Win:  {win_rate:.2f}% (n={len(first_win_bankruptcies)}, {sum(first_win_bankruptcies)} bankruptcies)")
    print(f"First Loss: {loss_rate:.2f}% (n={len(first_loss_bankruptcies)}, {sum(first_loss_bankruptcies)} bankruptcies)")
    
    # 3. Analyze by prompt combination
    print("\n" + "="*60)
    print("3. PROMPT COMBINATION ANALYSIS")
    print("="*60)
    
    prompt_bankruptcies = defaultdict(list)
    for result in all_results:
        prompt = result['prompt_combo']
        is_bankrupt = result.get('is_bankrupt', False)
        prompt_bankruptcies[prompt].append(int(is_bankrupt))
    
    # Calculate rates and sort
    prompt_rates = {}
    for prompt, bankruptcies in prompt_bankruptcies.items():
        rate = np.mean(bankruptcies) * 100
        count = sum(bankruptcies)
        prompt_rates[prompt] = (rate, count, len(bankruptcies))
    
    sorted_prompts = sorted(prompt_rates.items(), key=lambda x: x[1][0], reverse=True)
    
    print("\nTop 15 highest bankruptcy prompts:")
    print(f"{'Prompt':<10} {'Rate':<8} {'Count':<8} {'Total':<8}")
    print("-"*40)
    for prompt, (rate, count, total) in sorted_prompts[:15]:
        print(f"{prompt:<10} {rate:>6.2f}% {count:>7} {total:>7}")
    
    # 4. Combined condition analysis
    print("\n" + "="*60)
    print("4. COMBINED CONDITION ANALYSIS (Top 20)")
    print("="*60)
    
    condition_bankruptcies = defaultdict(list)
    for result in all_results:
        key = f"{result['bet_type']}_{result['first_result']}_{result['prompt_combo']}"
        is_bankrupt = result.get('is_bankrupt', False)
        condition_bankruptcies[key].append(int(is_bankrupt))
    
    condition_rates = {}
    for condition, bankruptcies in condition_bankruptcies.items():
        if len(bankruptcies) > 0:
            rate = np.mean(bankruptcies) * 100
            count = sum(bankruptcies)
            condition_rates[condition] = (rate, count, len(bankruptcies))
    
    sorted_conditions = sorted(condition_rates.items(), key=lambda x: x[1][0], reverse=True)
    
    print(f"{'Condition':<30} {'Rate':<8} {'Count':<8} {'Total':<8}")
    print("-"*60)
    for condition, (rate, count, total) in sorted_conditions[:20]:
        if rate > 0:  # Only show conditions with bankruptcies
            bet, first, prompt = condition.split('_', 2)
            print(f"{condition:<30} {rate:>6.2f}% {count:>7} {total:>7}")
    
    # 5. Pattern analysis
    print("\n" + "="*60)
    print("5. PATTERN ANALYSIS")
    print("="*60)
    
    # Check if variable + first loss is particularly risky
    var_loss_bankruptcies = []
    var_win_bankruptcies = []
    fix_loss_bankruptcies = []
    fix_win_bankruptcies = []
    
    for result in all_results:
        is_bankrupt = result.get('is_bankrupt', False)
        if result['bet_type'] == 'variable':
            if result['first_result'] == 'L':
                var_loss_bankruptcies.append(int(is_bankrupt))
            else:
                var_win_bankruptcies.append(int(is_bankrupt))
        else:
            if result['first_result'] == 'L':
                fix_loss_bankruptcies.append(int(is_bankrupt))
            else:
                fix_win_bankruptcies.append(int(is_bankrupt))
    
    print("\nInteraction effects:")
    print(f"Variable + Loss: {np.mean(var_loss_bankruptcies)*100:.2f}% ({sum(var_loss_bankruptcies)}/{len(var_loss_bankruptcies)})")
    print(f"Variable + Win:  {np.mean(var_win_bankruptcies)*100:.2f}% ({sum(var_win_bankruptcies)}/{len(var_win_bankruptcies)})")
    print(f"Fixed + Loss:    {np.mean(fix_loss_bankruptcies)*100:.2f}% ({sum(fix_loss_bankruptcies)}/{len(fix_loss_bankruptcies)})")
    print(f"Fixed + Win:     {np.mean(fix_win_bankruptcies)*100:.2f}% ({sum(fix_win_bankruptcies)}/{len(fix_win_bankruptcies)})")
    
    # 6. Prompt component analysis
    print("\n" + "="*60)
    print("6. PROMPT COMPONENT EFFECTS")
    print("="*60)
    
    components = ['G', 'M', 'P', 'R', 'W']
    component_effects = {}
    
    for comp in components:
        with_comp = []
        without_comp = []
        
        for result in all_results:
            prompt = result['prompt_combo']
            is_bankrupt = result.get('is_bankrupt', False)
            
            if prompt != 'BASE' and comp in prompt:
                with_comp.append(int(is_bankrupt))
            else:
                without_comp.append(int(is_bankrupt))
        
        with_rate = np.mean(with_comp) * 100 if with_comp else 0
        without_rate = np.mean(without_comp) * 100 if without_comp else 0
        effect = with_rate - without_rate
        
        component_effects[comp] = {
            'with': with_rate,
            'without': without_rate,
            'effect': effect,
            'with_count': sum(with_comp),
            'without_count': sum(without_comp)
        }
    
    print(f"{'Component':<12} {'With':<10} {'Without':<10} {'Effect':<10} {'Bankruptcies (with/without)'}")
    print("-"*70)
    for comp in components:
        data = component_effects[comp]
        print(f"{comp:<12} {data['with']:>6.2f}% {data['without']:>9.2f}% {data['effect']:>+9.2f}% "
              f"{data['with_count']:>10}/{data['without_count']}")
    
    # 7. Complexity analysis
    print("\n" + "="*60)
    print("7. PROMPT COMPLEXITY ANALYSIS")
    print("="*60)
    
    complexity_bankruptcies = defaultdict(list)
    
    for result in all_results:
        prompt = result['prompt_combo']
        is_bankrupt = result.get('is_bankrupt', False)
        
        if prompt == 'BASE':
            n_components = 0
        else:
            n_components = len(set(c for c in prompt if c in 'GMPRW'))
        
        complexity_bankruptcies[n_components].append(int(is_bankrupt))
    
    print(f"{'# Components':<15} {'Rate':<10} {'Count':<10} {'Total':<10}")
    print("-"*50)
    for n in sorted(complexity_bankruptcies.keys()):
        bankruptcies = complexity_bankruptcies[n]
        rate = np.mean(bankruptcies) * 100
        count = sum(bankruptcies)
        total = len(bankruptcies)
        print(f"{n:<15} {rate:>6.2f}% {count:>9} {total:>9}")
    
    # 8. Find specific patterns
    print("\n" + "="*60)
    print("8. SPECIFIC HIGH-RISK PATTERNS")
    print("="*60)
    
    # Check for patterns in high-risk conditions
    high_risk_prompts = ['GMR', 'GP', 'GMRP', 'GRP']
    
    for prompt in high_risk_prompts:
        prompt_results = [r for r in all_results if r['prompt_combo'] == prompt]
        if prompt_results:
            bankruptcies = [r for r in prompt_results if r.get('is_bankrupt', False)]
            
            # Analyze what makes them bankrupt
            var_count = sum(1 for r in bankruptcies if r['bet_type'] == 'variable')
            loss_count = sum(1 for r in bankruptcies if r['first_result'] == 'L')
            
            print(f"\n{prompt}: {len(bankruptcies)}/{len(prompt_results)} bankruptcies")
            print(f"  - Variable betting: {var_count}/{len(bankruptcies)} ({var_count/len(bankruptcies)*100:.0f}%)")
            print(f"  - First loss: {loss_count}/{len(bankruptcies)} ({loss_count/len(bankruptcies)*100:.0f}%)")

if __name__ == '__main__':
    analyze_llama_patterns()