#!/usr/bin/env python3
"""
Re-analyze bankruptcy ranking consistency with full data
1. Check all LLaMA data (combine both files)
2. Exclude 0% bankruptcy conditions for clearer visualization
3. Calculate proper p-values
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, rankdata
from collections import defaultdict
import time

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = ['DejaVu Sans', 'Liberation Sans']
plt.rcParams['font.size'] = 11

def load_all_data():
    """Load complete GPT and LLaMA datasets"""
    
    print("="*60)
    print("Loading complete datasets...")
    print("="*60)
    
    # Load GPT data
    print("\n1. Loading GPT data...")
    start_time = time.time()
    with open('/data/llm_addiction/gpt_results_corrected/gpt_corrected_complete_20250825_212628.json', 'r') as f:
        gpt_data = json.load(f)
    print(f"   GPT data loaded: {len(gpt_data['results'])} experiments")
    print(f"   Time taken: {time.time() - start_time:.2f} seconds")
    
    # Load LLaMA data - BOTH files
    print("\n2. Loading LLaMA data (main file)...")
    start_time = time.time()
    with open('/data/llm_addiction/results/exp1_multiround_intermediate_20250819_140040.json', 'r') as f:
        llama_data1 = json.load(f)
    print(f"   Main file: {len(llama_data1['results'])} experiments")
    print(f"   Time taken: {time.time() - start_time:.2f} seconds")
    
    print("\n3. Loading LLaMA data (additional file)...")
    start_time = time.time()
    with open('/data/llm_addiction/results/exp1_missing_complete_20250820_090040.json', 'r') as f:
        llama_data2 = json.load(f)
    print(f"   Additional file: {len(llama_data2['results'])} experiments")
    print(f"   Time taken: {time.time() - start_time:.2f} seconds")
    
    # Combine LLaMA data
    all_llama_results = llama_data1['results'] + llama_data2['results']
    print(f"\n   Total LLaMA experiments: {len(all_llama_results)}")
    
    return gpt_data['results'], all_llama_results

def calculate_bankruptcy_rates(gpt_results, llama_results):
    """Calculate bankruptcy rates for each condition"""
    
    print("\n" + "="*60)
    print("Calculating bankruptcy rates by condition...")
    print("="*60)
    
    # GPT bankruptcy rates
    gpt_by_condition = defaultdict(list)
    for result in gpt_results:
        cid = result['condition_id']
        is_bankrupt = result.get('is_bankrupt', False)
        gpt_by_condition[cid].append(int(is_bankrupt))
    
    gpt_rates = {}
    for cid in range(128):
        if cid in gpt_by_condition:
            gpt_rates[cid] = np.mean(gpt_by_condition[cid]) * 100
            print(f"GPT Condition {cid}: {len(gpt_by_condition[cid])} trials, {gpt_rates[cid]:.1f}% bankruptcy", end='\r')
    
    print(f"\nGPT: {len(gpt_rates)} conditions with data")
    
    # LLaMA bankruptcy rates
    llama_by_condition = defaultdict(list)
    for result in llama_results:
        cid = result['condition_id']
        is_bankrupt = result.get('is_bankrupt', False)
        llama_by_condition[cid].append(int(is_bankrupt))
    
    llama_rates = {}
    for cid in range(128):
        if cid in llama_by_condition:
            llama_rates[cid] = np.mean(llama_by_condition[cid]) * 100
            print(f"LLaMA Condition {cid}: {len(llama_by_condition[cid])} trials, {llama_rates[cid]:.1f}% bankruptcy", end='\r')
    
    print(f"\nLLaMA: {len(llama_rates)} conditions with data")
    
    # Check distribution
    print("\n" + "="*60)
    print("Bankruptcy rate distributions:")
    print("="*60)
    
    # GPT distribution
    gpt_zero = sum(1 for rate in gpt_rates.values() if rate == 0)
    gpt_nonzero = sum(1 for rate in gpt_rates.values() if rate > 0)
    print(f"\nGPT: {gpt_zero} conditions with 0%, {gpt_nonzero} with >0%")
    
    # LLaMA distribution
    llama_zero = sum(1 for rate in llama_rates.values() if rate == 0)
    llama_nonzero = sum(1 for rate in llama_rates.values() if rate > 0)
    print(f"LLaMA: {llama_zero} conditions with 0%, {llama_nonzero} with >0%")
    
    return gpt_rates, llama_rates

def analyze_rankings(gpt_rates, llama_rates, exclude_zeros=False):
    """Analyze ranking consistency"""
    
    print("\n" + "="*60)
    print(f"Analyzing rankings (exclude_zeros={exclude_zeros})...")
    print("="*60)
    
    # Find common conditions
    common_cids = [cid for cid in range(128) 
                   if cid in gpt_rates and cid in llama_rates]
    
    if exclude_zeros:
        # Exclude conditions where BOTH models have 0% bankruptcy
        common_cids = [cid for cid in common_cids 
                      if not (gpt_rates[cid] == 0 and llama_rates[cid] == 0)]
    
    print(f"Common conditions: {len(common_cids)}")
    
    # Get rates for common conditions
    gpt_common = [gpt_rates[cid] for cid in common_cids]
    llama_common = [llama_rates[cid] for cid in common_cids]
    
    # Calculate rankings (higher rate = lower rank number = more risky)
    gpt_rankings = rankdata([-r for r in gpt_common], method='average')
    llama_rankings = rankdata([-r for r in llama_common], method='average')
    
    # Calculate Spearman correlation
    correlation, p_value = spearmanr(gpt_rankings, llama_rankings)
    
    print(f"\nSpearman ρ = {correlation:.4f}")
    print(f"P-value = {p_value:.2e}")
    
    # Show some examples
    print("\nTop 5 risky conditions (GPT ranking):")
    gpt_sorted = np.argsort(gpt_rankings)
    for i in range(min(5, len(gpt_sorted))):
        idx = gpt_sorted[i]
        cid = common_cids[idx]
        print(f"  CID {cid}: GPT rank {gpt_rankings[idx]:.1f} ({gpt_common[idx]:.1f}%), "
              f"LLaMA rank {llama_rankings[idx]:.1f} ({llama_common[idx]:.1f}%)")
    
    return common_cids, gpt_rankings, llama_rankings, gpt_common, llama_common, correlation, p_value

def create_plots(common_cids, gpt_rankings, llama_rankings, gpt_rates_common, llama_rates_common, 
                correlation, p_value, exclude_zeros=False):
    """Create ranking consistency plots"""
    
    suffix = "_no_zeros" if exclude_zeros else "_all"
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left plot: Scatter plot
    colors = []
    sizes = []
    for gpt_rate, llama_rate in zip(gpt_rates_common, llama_rates_common):
        if gpt_rate == 0 and llama_rate == 0:
            colors.append('#95a5a6')  # Gray for both zero
            sizes.append(20)
        elif gpt_rate > 20 or llama_rate > 20:
            colors.append('#e74c3c')  # Red for high risk
            sizes.append(80)
        elif gpt_rate > 10 or llama_rate > 10:
            colors.append('#f39c12')  # Orange for medium
            sizes.append(60)
        elif gpt_rate > 0 or llama_rate > 0:
            colors.append('#3498db')  # Blue for low risk
            sizes.append(40)
        else:
            colors.append('#2ecc71')  # Green
            sizes.append(30)
    
    # Add small random jitter to avoid overlapping points
    np.random.seed(42)
    jitter = 0.3
    gpt_jittered = gpt_rankings + np.random.normal(0, jitter, len(gpt_rankings))
    llama_jittered = llama_rankings + np.random.normal(0, jitter, len(llama_rankings))
    
    scatter = ax1.scatter(gpt_jittered, llama_jittered, c=colors, s=sizes, 
                         alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # Add diagonal line
    max_rank = max(max(gpt_rankings), max(llama_rankings))
    ax1.plot([1, max_rank], [1, max_rank], 'r--', alpha=0.5, linewidth=2, 
            label='Perfect Agreement')
    
    ax1.set_xlabel('GPT-4o-mini Bankruptcy Risk Ranking', fontsize=12, fontweight='bold')
    ax1.set_ylabel('LLaMA-3.1-8B Bankruptcy Risk Ranking', fontsize=12, fontweight='bold')
    
    title = 'Bankruptcy Risk Ranking Consistency'
    if exclude_zeros:
        title += '\n(Excluding conditions with 0% bankruptcy in both models)'
    ax1.set_title(title, fontsize=13, fontweight='bold')
    
    # Add statistics
    stats_text = f'Spearman ρ = {correlation:.4f}\n'
    if p_value < 0.001:
        stats_text += f'p < 0.001\n'
    else:
        stats_text += f'p = {p_value:.4f}\n'
    stats_text += f'n = {len(common_cids)} conditions'
    
    ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right')
    
    # Right plot: Histogram of bankruptcy rates
    ax2.hist([gpt_rates_common, llama_rates_common], bins=20, 
            label=['GPT-4o-mini', 'LLaMA-3.1-8B'], 
            color=['#3498db', '#e74c3c'], alpha=0.7)
    ax2.set_xlabel('Bankruptcy Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Conditions', fontsize=12, fontweight='bold')
    ax2.set_title('Bankruptcy Rate Distributions', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    output_path = f'/home/ubuntu/llm_addiction/writing/figures/bankruptcy_ranking_consistency_reanalyzed{suffix}'
    plt.savefig(f'{output_path}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_path}.pdf', dpi=300, bbox_inches='tight')
    
    print(f"\n✅ Plot saved: {output_path}.png/pdf")

def main():
    """Main analysis function"""
    
    print("\n" + "="*60)
    print("COMPLETE RANKING CONSISTENCY RE-ANALYSIS")
    print("="*60)
    
    # Load all data
    gpt_results, llama_results = load_all_data()
    
    # Calculate bankruptcy rates
    gpt_rates, llama_rates = calculate_bankruptcy_rates(gpt_results, llama_results)
    
    # Analysis 1: All conditions
    print("\n" + "="*60)
    print("ANALYSIS 1: ALL CONDITIONS")
    print("="*60)
    common_all, gpt_rank_all, llama_rank_all, gpt_rate_all, llama_rate_all, corr_all, p_all = \
        analyze_rankings(gpt_rates, llama_rates, exclude_zeros=False)
    
    create_plots(common_all, gpt_rank_all, llama_rank_all, gpt_rate_all, llama_rate_all,
                corr_all, p_all, exclude_zeros=False)
    
    # Analysis 2: Excluding zeros
    print("\n" + "="*60)
    print("ANALYSIS 2: EXCLUDING MUTUAL ZEROS")
    print("="*60)
    common_no0, gpt_rank_no0, llama_rank_no0, gpt_rate_no0, llama_rate_no0, corr_no0, p_no0 = \
        analyze_rankings(gpt_rates, llama_rates, exclude_zeros=True)
    
    create_plots(common_no0, gpt_rank_no0, llama_rank_no0, gpt_rate_no0, llama_rate_no0,
                corr_no0, p_no0, exclude_zeros=True)
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"All conditions: ρ = {corr_all:.4f} (p = {p_all:.2e}), n = {len(common_all)}")
    print(f"Excluding zeros: ρ = {corr_no0:.4f} (p = {p_no0:.2e}), n = {len(common_no0)}")
    print("\n✅ Complete re-analysis finished!")

if __name__ == '__main__':
    main()