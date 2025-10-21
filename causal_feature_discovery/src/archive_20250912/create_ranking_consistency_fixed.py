#!/usr/bin/env python3
"""
Create ranking consistency plot between GPT-4o-mini and LLaMA-3.1-8B
Fixed version with correct data structure understanding
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
    
    # Load GPT results (smaller file first)
    gpt_file = '/data/llm_addiction/gpt_results_corrected/gpt_corrected_complete_20250825_212628.json'
    with open(gpt_file, 'r') as f:
        gpt_data = json.load(f)
    
    # Check GPT data structure
    if 'results' in gpt_data:
        gpt_results = gpt_data['results']
    else:
        gpt_results = gpt_data
    
    print(f"GPT results: {len(gpt_results)}")
    
    # Check first few GPT results to understand structure
    if gpt_results:
        print("Sample GPT result keys:", list(gpt_results[0].keys()))
    
    # Calculate GPT risk by prompt combination
    gpt_condition_risk = {}
    for result in gpt_results:
        # Create condition key from prompt components
        bet_type = result.get('bet_type', 'unknown')
        first_result = result.get('first_result', 'unknown')
        prompt_combo = result.get('prompt_combo', 'unknown')
        
        condition = f"{bet_type}_{first_result}_{prompt_combo}"
        
        if condition not in gpt_condition_risk:
            gpt_condition_risk[condition] = []
        
        # Risk = betting amount (higher = more risky)
        bet_amount = result.get('bet_amount', 0)
        if bet_amount is None:
            bet_amount = 0
        
        # Normalize to 0-100 scale
        risk_score = min(bet_amount / 100.0 * 100, 100)
        gpt_condition_risk[condition].append(risk_score)
    
    # Calculate average risk by condition for GPT
    gpt_avg_risk = {}
    for condition, risks in gpt_condition_risk.items():
        if risks:  # Only if we have data
            gpt_avg_risk[condition] = np.mean(risks)
    
    print(f"GPT conditions: {len(gpt_avg_risk)}")
    
    # Load LLaMA results (use smaller sample first to test)
    llama_file = '/data/llm_addiction/results/exp1_multiround_intermediate_20250819_140040.json'
    with open(llama_file, 'r') as f:
        llama_data = json.load(f)
    
    if 'results' in llama_data:
        llama_results = llama_data['results']
    else:
        llama_results = llama_data
    
    print(f"LLaMA results: {len(llama_results)}")
    
    # Check LLaMA data structure
    if llama_results:
        print("Sample LLaMA result keys:", list(llama_results[0].keys()))
    
    # Calculate LLaMA risk by prompt combination  
    llama_condition_risk = {}
    for result in llama_results:
        # Create condition key from experiment parameters
        bet_type = result.get('bet_type', 'unknown')
        first_result = result.get('first_result', 'unknown') 
        prompt_combo = result.get('prompt_combo', 'unknown')
        
        condition = f"{bet_type}_{first_result}_{prompt_combo}"
        
        if condition not in llama_condition_risk:
            llama_condition_risk[condition] = []
        
        # Risk = bankruptcy or high round count
        is_bankrupt = result.get('is_bankrupt', False)
        rounds_played = result.get('rounds_played', 1)
        
        if is_bankrupt:
            risk_score = 100  # Maximum risk
        else:
            # More rounds = more risk before stopping
            risk_score = min(rounds_played * 5, 90)  # Cap at 90 for voluntary stops
        
        llama_condition_risk[condition].append(risk_score)
    
    # Calculate average risk by condition for LLaMA
    llama_avg_risk = {}
    for condition, risks in llama_condition_risk.items():
        if risks:  # Only if we have data
            llama_avg_risk[condition] = np.mean(risks)
    
    print(f"LLaMA conditions: {len(llama_avg_risk)}")
    
    # Find common conditions
    common_conditions = set(gpt_avg_risk.keys()) & set(llama_avg_risk.keys())
    print(f"Common conditions: {len(common_conditions)}")
    
    if len(common_conditions) < 10:
        print("Warning: Very few common conditions found!")
        print("GPT conditions sample:", list(gpt_avg_risk.keys())[:5])
        print("LLaMA conditions sample:", list(llama_avg_risk.keys())[:5])
    
    # Prepare data for ranking
    gpt_risks = []
    llama_risks = []
    condition_labels = []
    
    for condition in sorted(common_conditions):
        gpt_risks.append(gpt_avg_risk[condition])
        llama_risks.append(llama_avg_risk[condition])
        condition_labels.append(condition)
    
    if len(gpt_risks) < 5:
        print("Error: Not enough common conditions for meaningful correlation!")
        return None, None
    
    # Convert to rankings (1 = highest risk)
    gpt_rankings = np.argsort(np.argsort(gpt_risks)[::-1]) + 1
    llama_rankings = np.argsort(np.argsort(llama_risks)[::-1]) + 1
    
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
    ax.set_title(f'Risk-Taking Behavior Ranking Consistency\nSpearman ρ = {correlation:.3f}', 
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