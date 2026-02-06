#!/usr/bin/env python3
"""
Component-wise correlation analysis between GPT and LLaMA
Analyze correlations for:
1. Prompt combinations only (32 types)
2. Betting type (fixed vs variable)
3. First result (W vs L)
4. Each combination separately
"""

import json
import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

def parse_prompt_components(prompt):
    """Parse GPT prompt into components, handling PR->P conversion"""
    if prompt == 'BASE':
        return prompt
    
    # Handle PR -> P conversion for GPT prompts
    if 'PR' in prompt:
        # GMPRW -> GMRWP, PRW -> RWP, etc.
        if 'PRW' in prompt:
            prompt = prompt.replace('PRW', 'RWP')
        elif prompt.endswith('PR'):
            prompt = prompt.replace('PR', 'P')
        else:
            prompt = prompt.replace('PR', 'P')
    
    return prompt

def load_data():
    """Load GPT and LLaMA data"""
    print("Loading datasets...")
    
    # GPT data
    with open('/data/llm_addiction/gpt_results_corrected/gpt_corrected_complete_20250825_212628.json', 'r') as f:
        gpt_data = json.load(f)
    
    # LLaMA data
    with open('/data/llm_addiction/results/exp1_multiround_intermediate_20250819_140040.json', 'r') as f:
        llama_data1 = json.load(f)
    with open('/data/llm_addiction/results/exp1_missing_complete_20250820_090040.json', 'r') as f:
        llama_data2 = json.load(f)
    
    all_llama = llama_data1['results'] + llama_data2['results']
    
    print(f"GPT: {len(gpt_data['results'])} experiments")
    print(f"LLaMA: {len(all_llama)} experiments")
    
    return gpt_data['results'], all_llama

def analyze_by_prompt_only():
    """Analyze correlation by prompt combinations only (ignoring bet type and first result)"""
    
    gpt_results, llama_results = load_data()
    
    print("\n" + "="*60)
    print("ANALYSIS 1: PROMPT COMBINATIONS ONLY (32 types)")
    print("="*60)
    
    # Group by prompt only
    gpt_by_prompt = defaultdict(list)
    llama_by_prompt = defaultdict(list)
    
    for result in gpt_results:
        prompt = parse_prompt_components(result['prompt_combo'])
        is_bankrupt = result.get('is_bankrupt', False)
        gpt_by_prompt[prompt].append(int(is_bankrupt))
    
    for result in llama_results:
        prompt = result['prompt_combo']
        is_bankrupt = result.get('is_bankrupt', False)
        llama_by_prompt[prompt].append(int(is_bankrupt))
    
    # Find common prompts
    common_prompts = set(gpt_by_prompt.keys()) & set(llama_by_prompt.keys())
    print(f"Common prompt combinations: {len(common_prompts)}")
    
    # Calculate bankruptcy rates
    gpt_rates = []
    llama_rates = []
    prompt_labels = []
    
    for prompt in sorted(common_prompts):
        gpt_rate = np.mean(gpt_by_prompt[prompt]) * 100
        llama_rate = np.mean(llama_by_prompt[prompt]) * 100
        gpt_rates.append(gpt_rate)
        llama_rates.append(llama_rate)
        prompt_labels.append(prompt)
    
    # Calculate correlations
    if len(gpt_rates) > 1:
        spearman_rho, spearman_p = spearmanr(gpt_rates, llama_rates)
        pearson_r, pearson_p = pearsonr(gpt_rates, llama_rates)
        
        print(f"\nCorrelation results:")
        print(f"Spearman ρ = {spearman_rho:.4f} (p = {spearman_p:.4f})")
        print(f"Pearson r = {pearson_r:.4f} (p = {pearson_p:.4f})")
        
        # Show top risky prompts
        print(f"\nTop 5 risky prompts (by average of GPT+LLaMA):")
        combined_rates = [(g+l)/2 for g, l in zip(gpt_rates, llama_rates)]
        top_indices = np.argsort(combined_rates)[-5:][::-1]
        
        for i, idx in enumerate(top_indices):
            print(f"{i+1}. {prompt_labels[idx]:10} - GPT: {gpt_rates[idx]:.1f}%, LLaMA: {llama_rates[idx]:.1f}%")
    
    return common_prompts, gpt_rates, llama_rates, prompt_labels

def analyze_by_betting_type():
    """Analyze correlation by betting type (fixed vs variable)"""
    
    gpt_results, llama_results = load_data()
    
    print("\n" + "="*60)
    print("ANALYSIS 2: BETTING TYPE (Fixed vs Variable)")
    print("="*60)
    
    # Group by betting type
    gpt_by_bet = {'fixed': [], 'variable': []}
    llama_by_bet = {'fixed': [], 'variable': []}
    
    for result in gpt_results:
        bet_type = result['bet_type']
        is_bankrupt = result.get('is_bankrupt', False)
        gpt_by_bet[bet_type].append(int(is_bankrupt))
    
    for result in llama_results:
        bet_type = result['bet_type']
        is_bankrupt = result.get('is_bankrupt', False)
        llama_by_bet[bet_type].append(int(is_bankrupt))
    
    # Calculate rates
    print("\nBankruptcy rates by betting type:")
    for bet_type in ['fixed', 'variable']:
        gpt_rate = np.mean(gpt_by_bet[bet_type]) * 100 if gpt_by_bet[bet_type] else 0
        llama_rate = np.mean(llama_by_bet[bet_type]) * 100 if llama_by_bet[bet_type] else 0
        print(f"{bet_type:8} - GPT: {gpt_rate:.1f}% (n={len(gpt_by_bet[bet_type])}), LLaMA: {llama_rate:.1f}% (n={len(llama_by_bet[bet_type])})")

def analyze_by_first_result():
    """Analyze correlation by first result (W vs L)"""
    
    gpt_results, llama_results = load_data()
    
    print("\n" + "="*60)
    print("ANALYSIS 3: FIRST RESULT (Win vs Loss)")
    print("="*60)
    
    # Group by first result
    gpt_by_first = {'W': [], 'L': []}
    llama_by_first = {'W': [], 'L': []}
    
    for result in gpt_results:
        first_result = result['first_result']
        is_bankrupt = result.get('is_bankrupt', False)
        gpt_by_first[first_result].append(int(is_bankrupt))
    
    for result in llama_results:
        first_result = result['first_result']
        is_bankrupt = result.get('is_bankrupt', False)
        llama_by_first[first_result].append(int(is_bankrupt))
    
    # Calculate rates
    print("\nBankruptcy rates by first result:")
    for first in ['W', 'L']:
        gpt_rate = np.mean(gpt_by_first[first]) * 100 if gpt_by_first[first] else 0
        llama_rate = np.mean(llama_by_first[first]) * 100 if llama_by_first[first] else 0
        print(f"First {first} - GPT: {gpt_rate:.1f}% (n={len(gpt_by_first[first])}), LLaMA: {llama_rate:.1f}% (n={len(llama_by_first[first])})")

def analyze_interactions():
    """Analyze interactions between components"""
    
    gpt_results, llama_results = load_data()
    
    print("\n" + "="*60)
    print("ANALYSIS 4: COMPONENT INTERACTIONS")
    print("="*60)
    
    # Analyze variable betting + first L combination
    gpt_var_L = []
    llama_var_L = []
    
    for result in gpt_results:
        if result['bet_type'] == 'variable' and result['first_result'] == 'L':
            gpt_var_L.append(int(result.get('is_bankrupt', False)))
    
    for result in llama_results:
        if result['bet_type'] == 'variable' and result['first_result'] == 'L':
            llama_var_L.append(int(result.get('is_bankrupt', False)))
    
    gpt_var_L_rate = np.mean(gpt_var_L) * 100 if gpt_var_L else 0
    llama_var_L_rate = np.mean(llama_var_L) * 100 if llama_var_L else 0
    
    print(f"\nVariable + First Loss combination:")
    print(f"GPT: {gpt_var_L_rate:.1f}% (n={len(gpt_var_L)})")
    print(f"LLaMA: {llama_var_L_rate:.1f}% (n={len(llama_var_L)})")
    
    # Analyze by prompt complexity (number of components)
    print("\n" + "="*60)
    print("ANALYSIS 5: PROMPT COMPLEXITY")
    print("="*60)
    
    gpt_by_complexity = defaultdict(list)
    llama_by_complexity = defaultdict(list)
    
    for result in gpt_results:
        prompt = result['prompt_combo']
        if prompt == 'BASE':
            complexity = 0
        else:
            # Count unique components
            components = set()
            for char in prompt:
                if char in ['G', 'M', 'P', 'R', 'W']:
                    components.add(char)
            complexity = len(components)
        
        is_bankrupt = result.get('is_bankrupt', False)
        gpt_by_complexity[complexity].append(int(is_bankrupt))
    
    for result in llama_results:
        prompt = result['prompt_combo']
        if prompt == 'BASE':
            complexity = 0
        else:
            components = set()
            for char in prompt:
                if char in ['G', 'M', 'P', 'R', 'W']:
                    components.add(char)
            complexity = len(components)
        
        is_bankrupt = result.get('is_bankrupt', False)
        llama_by_complexity[complexity].append(int(is_bankrupt))
    
    print("\nBankruptcy by prompt complexity (# of components):")
    for complexity in sorted(set(gpt_by_complexity.keys()) | set(llama_by_complexity.keys())):
        gpt_rate = np.mean(gpt_by_complexity[complexity]) * 100 if gpt_by_complexity[complexity] else 0
        llama_rate = np.mean(llama_by_complexity[complexity]) * 100 if llama_by_complexity[complexity] else 0
        print(f"{complexity} components - GPT: {gpt_rate:.1f}% (n={len(gpt_by_complexity[complexity])}), LLaMA: {llama_rate:.1f}% (n={len(llama_by_complexity[complexity])})")
    
    # Calculate correlation by complexity
    complexities = sorted(set(gpt_by_complexity.keys()) & set(llama_by_complexity.keys()))
    if len(complexities) > 1:
        gpt_complex_rates = [np.mean(gpt_by_complexity[c]) * 100 for c in complexities]
        llama_complex_rates = [np.mean(llama_by_complexity[c]) * 100 for c in complexities]
        
        rho, p = spearmanr(gpt_complex_rates, llama_complex_rates)
        print(f"\nCorrelation by complexity: ρ = {rho:.4f} (p = {p:.4f})")

def create_visualization(prompts, gpt_rates, llama_rates, labels):
    """Create visualization of prompt-wise correlation"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Scatter plot of bankruptcy rates
    ax1 = axes[0, 0]
    ax1.scatter(gpt_rates, llama_rates, alpha=0.6, s=50)
    
    # Add labels for high-risk prompts
    for i, (g, l, label) in enumerate(zip(gpt_rates, llama_rates, labels)):
        if g > 10 or l > 5:  # Label high-risk prompts
            ax1.annotate(label, (g, l), fontsize=8, alpha=0.7)
    
    ax1.plot([0, max(gpt_rates)], [0, max(llama_rates)], 'r--', alpha=0.5)
    ax1.set_xlabel('GPT-4o-mini Bankruptcy Rate (%)', fontweight='bold')
    ax1.set_ylabel('LLaMA-3.1-8B Bankruptcy Rate (%)', fontweight='bold')
    ax1.set_title('Prompt-wise Bankruptcy Rate Comparison', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Rank comparison
    ax2 = axes[0, 1]
    gpt_ranks = np.argsort(np.argsort(gpt_rates)[::-1]) + 1
    llama_ranks = np.argsort(np.argsort(llama_rates)[::-1]) + 1
    
    ax2.scatter(gpt_ranks, llama_ranks, alpha=0.6, s=50)
    ax2.plot([1, len(gpt_ranks)], [1, len(llama_ranks)], 'r--', alpha=0.5)
    ax2.set_xlabel('GPT-4o-mini Rank', fontweight='bold')
    ax2.set_ylabel('LLaMA-3.1-8B Rank', fontweight='bold')
    ax2.set_title('Prompt Risk Ranking Comparison', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Difference analysis
    ax3 = axes[1, 0]
    differences = [g - l for g, l in zip(gpt_rates, llama_rates)]
    sorted_indices = np.argsort(differences)[::-1]
    
    top_10 = sorted_indices[:10]
    ax3.barh(range(10), [differences[i] for i in top_10])
    ax3.set_yticks(range(10))
    ax3.set_yticklabels([labels[i] for i in top_10])
    ax3.set_xlabel('Difference (GPT - LLaMA) %', fontweight='bold')
    ax3.set_title('Top 10 Prompts: GPT More Risky Than LLaMA', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Plot 4: Component analysis
    ax4 = axes[1, 1]
    
    # Group prompts by key components
    has_G = [i for i, l in enumerate(labels) if 'G' in l]
    has_M = [i for i, l in enumerate(labels) if 'M' in l]
    has_P = [i for i, l in enumerate(labels) if 'P' in l and l != 'BASE']
    has_W = [i for i, l in enumerate(labels) if 'W' in l]
    
    component_effects = {
        'Goal (G)': (np.mean([gpt_rates[i] for i in has_G]), np.mean([llama_rates[i] for i in has_G])),
        'Maximize (M)': (np.mean([gpt_rates[i] for i in has_M]), np.mean([llama_rates[i] for i in has_M])),
        'Probability (P)': (np.mean([gpt_rates[i] for i in has_P]), np.mean([llama_rates[i] for i in has_P])),
        'Reward (W)': (np.mean([gpt_rates[i] for i in has_W]), np.mean([llama_rates[i] for i in has_W])),
    }
    
    x = np.arange(len(component_effects))
    width = 0.35
    
    gpt_means = [v[0] for v in component_effects.values()]
    llama_means = [v[1] for v in component_effects.values()]
    
    ax4.bar(x - width/2, gpt_means, width, label='GPT-4o-mini', alpha=0.8)
    ax4.bar(x + width/2, llama_means, width, label='LLaMA-3.1-8B', alpha=0.8)
    
    ax4.set_xlabel('Prompt Component', fontweight='bold')
    ax4.set_ylabel('Average Bankruptcy Rate (%)', fontweight='bold')
    ax4.set_title('Effect of Individual Components', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(component_effects.keys())
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Component-wise Correlation Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = '/home/ubuntu/llm_addiction/writing/figures/component_wise_correlation'
    plt.savefig(f'{output_path}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_path}.pdf', dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved: {output_path}.png/pdf")

if __name__ == '__main__':
    print("="*60)
    print("COMPONENT-WISE CORRELATION ANALYSIS")
    print("="*60)
    
    # Run all analyses
    prompts, gpt_rates, llama_rates, labels = analyze_by_prompt_only()
    analyze_by_betting_type()
    analyze_by_first_result()
    analyze_interactions()
    
    # Create visualization
    if len(prompts) > 0:
        create_visualization(prompts, gpt_rates, llama_rates, labels)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)