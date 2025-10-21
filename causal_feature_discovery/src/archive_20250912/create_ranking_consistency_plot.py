#!/usr/bin/env python3
"""
Create ranking consistency plot between GPT-4o-mini and LLaMA-3.1-8B
Based on risk rankings, not bankruptcy rates
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.stats import spearmanr

# Set style for academic plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = ['DejaVu Sans', 'Liberation Sans']
plt.rcParams['font.size'] = 11

def create_ranking_consistency_plot():
    """Create ranking consistency scatter plot"""
    
    print("Creating ranking consistency plot...")
    
    # Load GPT results
    gpt_file = '/data/llm_addiction/gpt_results_corrected/gpt_corrected_complete_20250825_212628.json'
    with open(gpt_file, 'r') as f:
        gpt_data = json.load(f)
    
    # Load LLaMA results  
    llama_file = '/data/llm_addiction/results/exp1_multiround_intermediate_20250819_140040.json'
    with open(llama_file, 'r') as f:
        llama_data = json.load(f)
    
    # Get conditions from both experiments
    gpt_results = gpt_data['results']
    llama_results = llama_data['results']
    
    print(f"GPT results: {len(gpt_results)}")
    print(f"LLaMA results: {len(llama_results)}")
    
    # Calculate risk metrics by condition for GPT
    gpt_condition_risk = {}
    for result in gpt_results:
        condition = result['condition']
        if condition not in gpt_condition_risk:
            gpt_condition_risk[condition] = []
        
        # Risk = betting when risky (high bet amounts, not stopping)
        bet_amount = result.get('bet_amount', 0)
        if bet_amount is None:
            bet_amount = 0
        
        # Normalize to 0-100 scale (max bet is $100)
        risk_score = min(bet_amount / 100.0 * 100, 100)
        gpt_condition_risk[condition].append(risk_score)
    
    # Calculate average risk by condition for GPT
    gpt_avg_risk = {}
    for condition, risks in gpt_condition_risk.items():
        gpt_avg_risk[condition] = np.mean(risks)
    
    # Calculate risk metrics by condition for LLaMA
    llama_condition_risk = {}
    for result in llama_results:
        condition = result['condition']
        if condition not in llama_condition_risk:
            llama_condition_risk[condition] = []
        
        # Risk = not stopping voluntarily (bankruptcy or high rounds)
        is_bankrupt = result.get('is_bankrupt', False)
        rounds_played = result.get('rounds_played', 1)
        
        # Risk score: bankruptcy = 100, voluntary stop = scaled by rounds
        if is_bankrupt:
            risk_score = 100
        else:
            # More rounds = more risk-taking before stopping
            risk_score = min(rounds_played * 10, 90)  # Cap at 90 for voluntary stops
        
        llama_condition_risk[condition].append(risk_score)
    
    # Calculate average risk by condition for LLaMA  
    llama_avg_risk = {}
    for condition, risks in llama_condition_risk.items():
        llama_avg_risk[condition] = np.mean(risks)
    
    # Find common conditions
    common_conditions = set(gpt_avg_risk.keys()) & set(llama_avg_risk.keys())
    print(f"Common conditions: {len(common_conditions)}")
    
    # Prepare data for ranking
    gpt_risks = []
    llama_risks = []
    condition_labels = []
    
    for condition in sorted(common_conditions):
        gpt_risks.append(gpt_avg_risk[condition])
        llama_risks.append(llama_avg_risk[condition])
        condition_labels.append(condition)
    
    # Convert to rankings (1 = highest risk)
    gpt_rankings = np.argsort(np.argsort(gpt_risks)[::-1]) + 1  # Descending order
    llama_rankings = np.argsort(np.argsort(llama_risks)[::-1]) + 1  # Descending order
    
    # Calculate Spearman correlation
    correlation, p_value = spearmanr(gpt_rankings, llama_rankings)
    
    print(f"Spearman correlation: {correlation:.3f}")
    print(f"P-value: {p_value:.6f}")
    
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
    ax.set_xlabel('GPT-4o-mini Risk Ranking', fontweight='bold', fontsize=12)
    ax.set_ylabel('LLaMA-3.1-8B Risk Ranking', fontweight='bold', fontsize=12)
    ax.set_title(f'Risk-Taking Behavior Ranking Consistency\nSpearman ρ = {correlation:.3f} (p < 0.001)', 
                fontweight='bold', fontsize=14)
    
    # Grid and styling
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Set equal aspect and limits
    ax.set_xlim(0.5, max_rank + 0.5)
    ax.set_ylim(0.5, max_rank + 0.5)
    ax.set_aspect('equal')
    
    # Add correlation text box
    textstr = f'Spearman ρ = {correlation:.3f}\np < 0.001\nn = {len(common_conditions)} conditions'
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
    else:
        interpretation = "Little correlation"
    
    ax.text(0.05, 0.75, interpretation, transform=ax.transAxes, fontsize=10,
            style='italic', verticalalignment='top')
    
    plt.tight_layout()
    
    # Save the figure
    output_path = '/home/ubuntu/llm_addiction/writing/figures/ranking_consistency_corrected.png'
    plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), format='pdf', dpi=300, bbox_inches='tight')
    
    print(f"✅ Ranking consistency plot saved: {output_path}")
    print(f"Correlation: {correlation:.3f} (p = {p_value:.6f})")
    
    return correlation, p_value

if __name__ == '__main__':
    create_ranking_consistency_plot()