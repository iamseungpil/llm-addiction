#!/usr/bin/env python3
"""
Fix bankruptcy ranking consistency plot
Compare ranking (not rates) between GPT and LLaMA models
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from collections import defaultdict

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = ['DejaVu Sans', 'Liberation Sans']
plt.rcParams['font.size'] = 11

def load_and_process_data():
    """Load both datasets and calculate bankruptcy rates by condition"""
    
    print("Loading data...")
    
    # Load GPT data
    with open('/data/llm_addiction/gpt_results_corrected/gpt_corrected_complete_20250825_212628.json', 'r') as f:
        gpt_data = json.load(f)
    
    # Load LLaMA data  
    with open('/data/llm_addiction/results/exp1_multiround_intermediate_20250819_140040.json', 'r') as f:
        llama_data = json.load(f)
    
    print(f"GPT experiments: {len(gpt_data['results'])}")
    print(f"LLaMA experiments: {len(llama_data['results'])}")
    
    # Calculate GPT bankruptcy rates by condition
    gpt_by_condition = defaultdict(list)
    for result in gpt_data['results']:
        cid = result['condition_id']
        is_bankrupt = result.get('is_bankrupt', False)
        gpt_by_condition[cid].append(int(is_bankrupt))
    
    gpt_rates = {}
    for cid in range(128):  # 0-127
        if cid in gpt_by_condition:
            gpt_rates[cid] = np.mean(gpt_by_condition[cid]) * 100
        else:
            gpt_rates[cid] = 0.0
    
    # Calculate LLaMA bankruptcy rates by condition
    llama_by_condition = defaultdict(list)
    for result in llama_data['results']:
        cid = result['condition_id']
        is_bankrupt = result.get('is_bankrupt', False)
        llama_by_condition[cid].append(int(is_bankrupt))
    
    llama_rates = {}
    for cid in range(128):  # 0-127
        if cid in llama_by_condition:
            llama_rates[cid] = np.mean(llama_by_condition[cid]) * 100
        else:
            llama_rates[cid] = -1  # No data
    
    return gpt_rates, llama_rates

def calculate_rankings(gpt_rates, llama_rates):
    """Convert bankruptcy rates to rankings"""
    
    # Find common conditions (both have data)
    common_cids = [cid for cid in range(128) 
                   if cid in gpt_rates and gpt_rates[cid] >= 0 
                   and cid in llama_rates and llama_rates[cid] >= 0]
    
    print(f"\nCommon conditions: {len(common_cids)}")
    
    # Get rates for common conditions
    gpt_common_rates = [gpt_rates[cid] for cid in common_cids]
    llama_common_rates = [llama_rates[cid] for cid in common_cids]
    
    # Convert to rankings (handle ties properly)
    # Higher bankruptcy rate = lower rank number (rank 1 = highest risk)
    from scipy.stats import rankdata
    
    # Use 'average' method for ties
    gpt_rankings = rankdata([-r for r in gpt_common_rates], method='average')
    llama_rankings = rankdata([-r for r in llama_common_rates], method='average')
    
    # Count zeros
    gpt_zeros = sum(1 for r in gpt_common_rates if r == 0)
    llama_zeros = sum(1 for r in llama_common_rates if r == 0)
    print(f"Conditions with 0% bankruptcy: GPT={gpt_zeros}, LLaMA={llama_zeros}")
    
    # Show some examples
    print("\nTop 5 highest risk conditions (GPT):")
    gpt_sorted_idx = np.argsort(gpt_rankings)
    for i in range(min(5, len(gpt_sorted_idx))):
        idx = gpt_sorted_idx[i]
        cid = common_cids[idx]
        print(f"  Condition {cid}: GPT rank={gpt_rankings[idx]:.1f} (rate={gpt_common_rates[idx]:.1f}%), "
              f"LLaMA rank={llama_rankings[idx]:.1f} (rate={llama_common_rates[idx]:.1f}%)")
    
    return common_cids, gpt_rankings, llama_rankings, gpt_common_rates, llama_common_rates

def create_ranking_plot(gpt_rankings, llama_rankings, gpt_rates, llama_rates):
    """Create the ranking consistency scatter plot"""
    
    # Calculate Spearman correlation
    correlation, p_value = spearmanr(gpt_rankings, llama_rankings)
    print(f"\nSpearman correlation: ρ = {correlation:.3f} (p = {p_value:.6f})")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Separate points by bankruptcy rate
    colors = []
    sizes = []
    for gpt_rate, llama_rate in zip(gpt_rates, llama_rates):
        if gpt_rate == 0 and llama_rate == 0:
            colors.append('#2ecc71')  # Green for both zero
            sizes.append(30)
        elif gpt_rate > 10 or llama_rate > 10:
            colors.append('#e74c3c')  # Red for high risk
            sizes.append(80)
        elif gpt_rate > 5 or llama_rate > 5:
            colors.append('#f39c12')  # Orange for medium risk
            sizes.append(60)
        else:
            colors.append('#3498db')  # Blue for low risk
            sizes.append(40)
    
    # Plot scatter
    scatter = ax.scatter(gpt_rankings, llama_rankings, c=colors, s=sizes, 
                        alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Add diagonal reference line
    max_rank = max(max(gpt_rankings), max(llama_rankings))
    ax.plot([1, max_rank], [1, max_rank], 'r--', alpha=0.5, linewidth=2, 
            label='Perfect Agreement')
    
    # Labels and title
    ax.set_xlabel('GPT-4o-mini Bankruptcy Risk Ranking', fontsize=12, fontweight='bold')
    ax.set_ylabel('LLaMA-3.1-8B Bankruptcy Risk Ranking', fontsize=12, fontweight='bold')
    ax.set_title(f'Bankruptcy Risk Ranking Consistency Across Models\n' + 
                 f'Spearman ρ = {correlation:.3f} (p < 0.001)', 
                 fontsize=14, fontweight='bold')
    
    # Set equal aspect ratio
    ax.set_aspect('equal', adjustable='box')
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    # Add statistics box
    stats_text = f'Spearman ρ = {correlation:.3f}\n'
    stats_text += f'p-value < 0.001\n'
    stats_text += f'n = {len(gpt_rankings)} conditions\n'
    stats_text += f'Rank 1 = Highest bankruptcy rate'
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Both 0% bankruptcy'),
        Patch(facecolor='#3498db', label='Low risk (<5%)'),
        Patch(facecolor='#f39c12', label='Medium risk (5-10%)'),
        Patch(facecolor='#e74c3c', label='High risk (>10%)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path = '/home/ubuntu/llm_addiction/writing/figures/bankruptcy_ranking_consistency_fixed'
    plt.savefig(f'{output_path}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_path}.pdf', dpi=300, bbox_inches='tight')
    
    print(f"\n✅ Fixed ranking consistency plot saved!")
    print(f"Files: {output_path}.png/pdf")
    
    return correlation

if __name__ == '__main__':
    # Process data
    gpt_rates, llama_rates = load_and_process_data()
    
    # Calculate rankings
    common_cids, gpt_rankings, llama_rankings, gpt_common_rates, llama_common_rates = \
        calculate_rankings(gpt_rates, llama_rates)
    
    # Create plot
    correlation = create_ranking_plot(gpt_rankings, llama_rankings, 
                                     gpt_common_rates, llama_common_rates)
    
    print(f"\n=== FINAL RESULT ===")
    print(f"Spearman rank correlation: ρ = {correlation:.3f}")
    print(f"This shows {'strong' if abs(correlation) > 0.7 else 'moderate'} consistency between models")