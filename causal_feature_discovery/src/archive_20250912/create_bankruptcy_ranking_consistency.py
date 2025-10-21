#!/usr/bin/env python3
"""
Create CORRECT ranking consistency plot based on BANKRUPTCY RATES
Calculate ranking by condition_id (1-128) based on bankruptcy rates
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.stats import spearmanr
from collections import defaultdict

# Set style for academic plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = ['DejaVu Sans', 'Liberation Sans']
plt.rcParams['font.size'] = 11

def create_bankruptcy_ranking_consistency():
    """Create ranking consistency based on bankruptcy rates by condition"""
    
    print("Creating bankruptcy ranking consistency plot...")
    
    # Load GPT results
    gpt_file = '/data/llm_addiction/gpt_results_corrected/gpt_corrected_complete_20250825_212628.json'
    with open(gpt_file, 'r') as f:
        gpt_data = json.load(f)
    
    gpt_results = gpt_data['results']
    print(f"GPT results: {len(gpt_results)}")
    
    # Load LLaMA results
    llama_file = '/data/llm_addiction/results/exp1_multiround_intermediate_20250819_140040.json'
    with open(llama_file, 'r') as f:
        llama_data = json.load(f)
    
    llama_results = llama_data['results']
    print(f"LLaMA results: {len(llama_results)}")
    
    # Calculate GPT bankruptcy rates by condition_id
    gpt_condition_stats = defaultdict(list)
    for result in gpt_results:
        condition_id = result.get('condition_id')
        is_bankrupt = result.get('is_bankrupt', False)
        
        # Convert bankruptcy to binary
        bankruptcy_value = 1 if is_bankrupt else 0
        gpt_condition_stats[condition_id].append(bankruptcy_value)
    
    # Calculate GPT bankruptcy rates
    gpt_bankruptcy_rates = {}
    for condition_id, bankruptcies in gpt_condition_stats.items():
        if bankruptcies:  # Only if we have data
            bankruptcy_rate = np.mean(bankruptcies)
            gpt_bankruptcy_rates[condition_id] = bankruptcy_rate
    
    print(f"GPT conditions with data: {len(gpt_bankruptcy_rates)}")
    
    # Calculate LLaMA bankruptcy rates by condition_id
    llama_condition_stats = defaultdict(list)
    for result in llama_results:
        condition_id = result.get('condition_id')
        is_bankrupt = result.get('is_bankrupt', False)
        
        # Convert bankruptcy to binary
        bankruptcy_value = 1 if is_bankrupt else 0
        llama_condition_stats[condition_id].append(bankruptcy_value)
    
    # Calculate LLaMA bankruptcy rates
    llama_bankruptcy_rates = {}
    for condition_id, bankruptcies in llama_condition_stats.items():
        if bankruptcies:  # Only if we have data
            bankruptcy_rate = np.mean(bankruptcies)
            llama_bankruptcy_rates[condition_id] = bankruptcy_rate
    
    print(f"LLaMA conditions with data: {len(llama_bankruptcy_rates)}")
    
    # Find common condition_ids
    common_conditions = set(gpt_bankruptcy_rates.keys()) & set(llama_bankruptcy_rates.keys())
    print(f"Common conditions: {len(common_conditions)}")
    
    if len(common_conditions) < 10:
        print("Warning: Very few common conditions!")
        print("GPT sample conditions:", sorted(list(gpt_bankruptcy_rates.keys()))[:10])
        print("LLaMA sample conditions:", sorted(list(llama_bankruptcy_rates.keys()))[:10])
    
    # Prepare data for ranking
    condition_ids = []
    gpt_rates = []
    llama_rates = []
    
    for condition_id in sorted(common_conditions):
        condition_ids.append(condition_id)
        gpt_rates.append(gpt_bankruptcy_rates[condition_id])
        llama_rates.append(llama_bankruptcy_rates[condition_id])
    
    print(f"Data prepared for {len(condition_ids)} conditions")
    
    # Show sample data for verification
    print("\nSample bankruptcy rates:")
    for i in range(min(10, len(condition_ids))):
        cid = condition_ids[i]
        gpt_rate = gpt_rates[i]
        llama_rate = llama_rates[i]
        print(f"Condition {cid}: GPT {gpt_rate:.3f}, LLaMA {llama_rate:.3f}")
    
    # Convert to rankings (1 = highest bankruptcy rate)
    gpt_rankings = np.argsort(np.argsort(gpt_rates)[::-1]) + 1  # Descending order
    llama_rankings = np.argsort(np.argsort(llama_rates)[::-1]) + 1  # Descending order
    
    # Calculate Spearman correlation
    correlation, p_value = spearmanr(gpt_rankings, llama_rankings)
    
    print(f"\nSpearman correlation: {correlation:.3f}")
    print(f"P-value: {p_value:.6f}")
    
    # Show ranking comparison for top/bottom conditions
    print(f"\nTop 5 conditions by GPT bankruptcy rate:")
    gpt_sorted_indices = np.argsort(gpt_rates)[::-1][:5]
    for idx in gpt_sorted_indices:
        cid = condition_ids[idx]
        gpt_rank = gpt_rankings[idx] 
        llama_rank = llama_rankings[idx]
        print(f"  Condition {cid}: GPT rank {gpt_rank}, LLaMA rank {llama_rank}, GPT rate {gpt_rates[idx]:.3f}, LLaMA rate {llama_rates[idx]:.3f}")
    
    print(f"\nTop 5 conditions by LLaMA bankruptcy rate:")
    llama_sorted_indices = np.argsort(llama_rates)[::-1][:5]
    for idx in llama_sorted_indices:
        cid = condition_ids[idx]
        gpt_rank = gpt_rankings[idx]
        llama_rank = llama_rankings[idx] 
        print(f"  Condition {cid}: GPT rank {gpt_rank}, LLaMA rank {llama_rank}, GPT rate {gpt_rates[idx]:.3f}, LLaMA rate {llama_rates[idx]:.3f}")
    
    # Create scatter plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Scatter plot with rankings
    scatter = ax.scatter(gpt_rankings, llama_rankings, 
                        alpha=0.7, s=60, c='#3498db', edgecolors='navy', linewidth=0.5)
    
    # Perfect correlation line
    max_rank = max(max(gpt_rankings), max(llama_rankings))
    ax.plot([1, max_rank], [1, max_rank], 'r--', alpha=0.8, linewidth=2, 
            label='Perfect Correlation')
    
    # Labels and title
    ax.set_xlabel('GPT-4o-mini Bankruptcy Rate Ranking', fontweight='bold', fontsize=12)
    ax.set_ylabel('LLaMA-3.1-8B Bankruptcy Rate Ranking', fontweight='bold', fontsize=12)
    ax.set_title(f'Bankruptcy Rate Ranking Consistency Across Models\nSpearman ρ = {correlation:.3f}', 
                fontweight='bold', fontsize=14)
    
    # Grid and styling
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Set equal aspect and limits
    ax.set_xlim(0.5, max_rank + 0.5)
    ax.set_ylim(0.5, max_rank + 0.5)
    ax.set_aspect('equal')
    
    # Add correlation text box
    p_text = f"p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"
    textstr = f'Spearman ρ = {correlation:.3f}\n{p_text}\nn = {len(common_conditions)} conditions'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    # Add interpretation
    if correlation > 0.8:
        interpretation = "Strong positive correlation"
    elif correlation > 0.6:
        interpretation = "Moderate positive correlation"
    elif correlation > 0.4:
        interpretation = "Weak positive correlation"
    elif correlation < -0.4:
        interpretation = "Negative correlation"
    else:
        interpretation = "Little correlation"
    
    ax.text(0.05, 0.80, interpretation, transform=ax.transAxes, fontsize=10,
            style='italic', verticalalignment='top')
    
    # Add explanation
    explanation = "Ranking: 1 = highest bankruptcy rate\nHigher rank = more dangerous condition"
    ax.text(0.05, 0.70, explanation, transform=ax.transAxes, fontsize=9,
            verticalalignment='top')
    
    plt.tight_layout()
    
    # Save the figure
    output_path = '/home/ubuntu/llm_addiction/writing/figures/bankruptcy_ranking_consistency.png'
    plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), format='pdf', dpi=300, bbox_inches='tight')
    
    print(f"\n✅ Bankruptcy ranking consistency plot saved: {output_path}")
    print(f"Final correlation: {correlation:.3f} (p = {p_value:.6f})")
    
    # Save detailed data for verification
    output_data = {
        'timestamp': '20250910_bankruptcy_ranking',
        'correlation': correlation,
        'p_value': p_value,
        'n_conditions': len(common_conditions),
        'gpt_bankruptcy_rates': {str(cid): rate for cid, rate in zip(condition_ids, gpt_rates)},
        'llama_bankruptcy_rates': {str(cid): rate for cid, rate in zip(condition_ids, llama_rates)},
        'gpt_rankings': gpt_rankings.tolist(),
        'llama_rankings': llama_rankings.tolist(),
        'condition_ids': condition_ids
    }
    
    data_file = '/data/llm_addiction/results/bankruptcy_ranking_consistency_20250910.json'
    with open(data_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Detailed data saved: {data_file}")
    
    return correlation, p_value

if __name__ == '__main__':
    create_bankruptcy_ranking_consistency()