#!/usr/bin/env python3
"""
Improved ranking analysis with tie-breaking
1. Sort by bankruptcy rate (descending)
2. Break ties using condition name (alphabetical)
3. Use multiple correlation metrics
"""

import json
import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr, kendalltau, pearsonr
import matplotlib.pyplot as plt

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
    
    return gpt_data['results'], all_llama

def calculate_bankruptcy_with_tiebreaking():
    """Calculate bankruptcy rates with consistent tie-breaking"""
    
    gpt_results, llama_results = load_all_data()
    
    print("\n" + "="*60)
    print("IMPROVED RANKING ANALYSIS WITH TIE-BREAKING")
    print("="*60)
    
    # Group by experimental condition string (for consistent tie-breaking)
    gpt_by_condition = defaultdict(list)
    llama_by_condition = defaultdict(list)
    
    # Use experimental parameters as key
    for result in gpt_results:
        key = f"{result['bet_type']}_{result['first_result']}_{result['prompt_combo']}"
        is_bankrupt = result.get('is_bankrupt', False)
        gpt_by_condition[key].append(int(is_bankrupt))
    
    for result in llama_results:
        key = f"{result['bet_type']}_{result['first_result']}_{result['prompt_combo']}"
        is_bankrupt = result.get('is_bankrupt', False)
        llama_by_condition[key].append(int(is_bankrupt))
    
    # Find common conditions
    common_conditions = sorted(set(gpt_by_condition.keys()) & set(llama_by_condition.keys()))
    print(f"Common experimental conditions: {len(common_conditions)}")
    
    # Calculate bankruptcy rates
    condition_data = []
    for condition in common_conditions:
        gpt_rate = np.mean(gpt_by_condition[condition]) * 100
        llama_rate = np.mean(llama_by_condition[condition]) * 100
        
        condition_data.append({
            'condition': condition,
            'gpt_rate': gpt_rate,
            'llama_rate': llama_rate
        })
    
    # Sort with tie-breaking: primary by rate (descending), secondary by condition name
    gpt_sorted = sorted(condition_data, key=lambda x: (-x['gpt_rate'], x['condition']))
    llama_sorted = sorted(condition_data, key=lambda x: (-x['llama_rate'], x['condition']))
    
    # Create ranking dictionaries
    gpt_rank_dict = {item['condition']: rank+1 for rank, item in enumerate(gpt_sorted)}
    llama_rank_dict = {item['condition']: rank+1 for rank, item in enumerate(llama_sorted)}
    
    # Get aligned rankings
    gpt_ranks = []
    llama_ranks = []
    conditions_list = []
    gpt_rates_list = []
    llama_rates_list = []
    
    for cond in common_conditions:
        gpt_ranks.append(gpt_rank_dict[cond])
        llama_ranks.append(llama_rank_dict[cond])
        conditions_list.append(cond)
        
        # Find rates
        for item in condition_data:
            if item['condition'] == cond:
                gpt_rates_list.append(item['gpt_rate'])
                llama_rates_list.append(item['llama_rate'])
                break
    
    # Calculate multiple correlation metrics
    print("\n" + "="*60)
    print("CORRELATION METRICS (WITH TIE-BREAKING)")
    print("="*60)
    
    # 1. Spearman rank correlation
    spearman_rho, spearman_p = spearmanr(gpt_ranks, llama_ranks)
    print(f"Spearman ρ = {spearman_rho:.4f} (p = {spearman_p:.2e})")
    
    # 2. Kendall's tau (better for ties)
    kendall_tau, kendall_p = kendalltau(gpt_ranks, llama_ranks)
    print(f"Kendall τ = {kendall_tau:.4f} (p = {kendall_p:.2e})")
    
    # 3. Pearson correlation on rates
    pearson_r, pearson_p = pearsonr(gpt_rates_list, llama_rates_list)
    print(f"Pearson r (rates) = {pearson_r:.4f} (p = {pearson_p:.2e})")
    
    # 4. Top-k agreement metric
    print("\n" + "="*60)
    print("TOP-K AGREEMENT ANALYSIS")
    print("="*60)
    
    for k in [5, 10, 20]:
        top_k_gpt = set([item['condition'] for item in gpt_sorted[:k]])
        top_k_llama = set([item['condition'] for item in llama_sorted[:k]])
        
        agreement = len(top_k_gpt & top_k_llama) / k * 100
        print(f"Top-{k} agreement: {agreement:.1f}% ({len(top_k_gpt & top_k_llama)}/{k} conditions)")
    
    # 5. Weighted rank correlation (emphasize high-risk conditions)
    weights = [1.0 / rank for rank in range(1, len(gpt_ranks) + 1)]
    weighted_corr = np.corrcoef(
        np.array(gpt_ranks) * weights,
        np.array(llama_ranks) * weights
    )[0, 1]
    print(f"\nWeighted correlation (emphasizing top ranks): {weighted_corr:.4f}")
    
    # Show examples of ranking differences
    print("\n" + "="*60)
    print("LARGEST RANKING DISCREPANCIES")
    print("="*60)
    
    discrepancies = []
    for i, cond in enumerate(conditions_list):
        disc = abs(gpt_ranks[i] - llama_ranks[i])
        discrepancies.append((disc, cond, gpt_ranks[i], llama_ranks[i], 
                            gpt_rates_list[i], llama_rates_list[i]))
    
    discrepancies.sort(reverse=True)
    
    print("\nTop 10 conditions with largest rank differences:")
    for i, (disc, cond, gpt_rank, llama_rank, gpt_rate, llama_rate) in enumerate(discrepancies[:10]):
        print(f"{i+1}. {cond[:50]}...")
        print(f"   Rank difference: {disc}")
        print(f"   GPT: rank {gpt_rank} ({gpt_rate:.1f}% bankruptcy)")
        print(f"   LLaMA: rank {llama_rank} ({llama_rate:.1f}% bankruptcy)")
    
    # Create visualization
    print("\n" + "="*60)
    print("CREATING IMPROVED VISUALIZATION")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Plot 1: Rank comparison with tie-breaking
    ax1 = axes[0, 0]
    colors = ['red' if gr > lr else 'blue' if gr < lr else 'gray' 
              for gr, lr in zip(gpt_rates_list, llama_rates_list)]
    ax1.scatter(gpt_ranks, llama_ranks, c=colors, alpha=0.6, s=30)
    ax1.plot([1, len(gpt_ranks)], [1, len(llama_ranks)], 'k--', alpha=0.3)
    ax1.set_xlabel('GPT-4o-mini Rank (with tie-breaking)', fontweight='bold')
    ax1.set_ylabel('LLaMA-3.1-8B Rank (with tie-breaking)', fontweight='bold')
    ax1.set_title(f'Rank Comparison (Spearman ρ = {spearman_rho:.3f})', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Direct rate comparison
    ax2 = axes[0, 1]
    ax2.scatter(gpt_rates_list, llama_rates_list, alpha=0.6)
    ax2.plot([0, max(gpt_rates_list)], [0, max(llama_rates_list)], 'k--', alpha=0.3)
    ax2.set_xlabel('GPT-4o-mini Bankruptcy Rate (%)', fontweight='bold')
    ax2.set_ylabel('LLaMA-3.1-8B Bankruptcy Rate (%)', fontweight='bold')
    ax2.set_title(f'Rate Comparison (Pearson r = {pearson_r:.3f})', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Rank difference distribution
    ax3 = axes[1, 0]
    rank_diffs = [abs(gr - lr) for gr, lr in zip(gpt_ranks, llama_ranks)]
    ax3.hist(rank_diffs, bins=30, edgecolor='black', alpha=0.7)
    ax3.set_xlabel('Absolute Rank Difference', fontweight='bold')
    ax3.set_ylabel('Number of Conditions', fontweight='bold')
    ax3.set_title(f'Rank Difference Distribution (Mean = {np.mean(rank_diffs):.1f})', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Cumulative agreement
    ax4 = axes[1, 1]
    k_values = range(1, min(51, len(gpt_ranks)+1))
    agreements = []
    for k in k_values:
        top_k_gpt = set([item['condition'] for item in gpt_sorted[:k]])
        top_k_llama = set([item['condition'] for item in llama_sorted[:k]])
        agreement = len(top_k_gpt & top_k_llama) / k * 100
        agreements.append(agreement)
    
    ax4.plot(k_values, agreements, linewidth=2)
    ax4.set_xlabel('Top-K Conditions', fontweight='bold')
    ax4.set_ylabel('Agreement (%)', fontweight='bold')
    ax4.set_title('Top-K Agreement Analysis', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 100)
    
    plt.suptitle('Improved Ranking Analysis with Tie-Breaking', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    output_path = '/home/ubuntu/llm_addiction/writing/figures/improved_ranking_analysis'
    plt.savefig(f'{output_path}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_path}.pdf', dpi=300, bbox_inches='tight')
    
    print(f"✅ Improved analysis plot saved: {output_path}.png/pdf")
    
    return spearman_rho, kendall_tau, pearson_r

if __name__ == '__main__':
    print("="*60)
    print("IMPROVED RANKING ANALYSIS WITH MULTIPLE METRICS")
    print("="*60)
    
    spearman, kendall, pearson = calculate_bankruptcy_with_tiebreaking()
    
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Spearman ρ (rank correlation): {spearman:.4f}")
    print(f"Kendall τ (better for ties): {kendall:.4f}")
    print(f"Pearson r (rate correlation): {pearson:.4f}")
    print("\n✅ Analysis complete!")