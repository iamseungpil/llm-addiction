#!/usr/bin/env python3
"""
Create GPT-only complexity figure with statistical significance testing
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')

# Set larger font sizes
plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18
})

def create_gpt_only_complexity_figure():
    """Create complexity figure with only GPT data and statistical tests"""
    print("Creating GPT-only complexity figure with statistical analysis...")
    
    # Load latest GPT data (3,200 experiments)
    with open('/data/llm_addiction/gpt_results_fixed_parsing/gpt_fixed_parsing_complete_20250919_151240.json') as f:
        gpt_data = json.load(f)
    
    print(f"Analyzing {len(gpt_data['results'])} GPT-4o-mini experiments")
    
    # Process GPT data
    gpt_analysis = []
    for result in gpt_data['results']:
        prompt_combo = result.get('prompt_combo', 'BASE')
        if not prompt_combo or prompt_combo == 'BASE':
            complexity = 0
        else:
            complexity = sum(1 for comp in ['G', 'M', 'P', 'R', 'W'] if comp in prompt_combo)
        
        gpt_analysis.append({
            'complexity': complexity,
            'prompt': prompt_combo,
            'is_bankrupt': result.get('is_bankrupt', False),
            'total_rounds': result.get('total_rounds', 0),
            'total_bet': result.get('total_bet', 0)
        })
    
    df = pd.DataFrame(gpt_analysis)
    
    # Calculate statistics by complexity
    stats_by_complexity = df.groupby('complexity').agg({
        'is_bankrupt': ['count', 'sum', 'mean'],
        'total_rounds': 'mean',
        'total_bet': 'mean'
    }).round(3)
    
    # Extract data for plotting and statistics
    complexities = stats_by_complexity.index.tolist()
    sample_sizes = stats_by_complexity[('is_bankrupt', 'count')].tolist()
    bankruptcy_counts = stats_by_complexity[('is_bankrupt', 'sum')].tolist()
    bankruptcy_rates = (stats_by_complexity[('is_bankrupt', 'mean')] * 100).tolist()
    avg_rounds = stats_by_complexity[('total_rounds', 'mean')].tolist()
    avg_bets = stats_by_complexity[('total_bet', 'mean')].tolist()
    
    # Statistical tests
    print("\nStatistical Analysis:")
    print("===================")
    
    # Linear trend test for bankruptcy rate
    slope_bankruptcy, intercept_bankruptcy, r_bankruptcy, p_bankruptcy, se_bankruptcy = stats.linregress(complexities, bankruptcy_rates)
    print(f"Bankruptcy Rate vs Complexity:")
    print(f"  Linear regression: y = {slope_bankruptcy:.3f}x + {intercept_bankruptcy:.3f}")
    print(f"  Correlation coefficient (r): {r_bankruptcy:.3f}")
    print(f"  P-value: {p_bankruptcy:.6f}")
    print(f"  R-squared: {r_bankruptcy**2:.3f}")
    
    # Linear trend test for average rounds
    slope_rounds, intercept_rounds, r_rounds, p_rounds, se_rounds = stats.linregress(complexities, avg_rounds)
    print(f"\nAverage Rounds vs Complexity:")
    print(f"  Linear regression: y = {slope_rounds:.3f}x + {intercept_rounds:.3f}")
    print(f"  Correlation coefficient (r): {r_rounds:.3f}")
    print(f"  P-value: {p_rounds:.6f}")
    print(f"  R-squared: {r_rounds**2:.3f}")
    
    # Linear trend test for average betting
    slope_bets, intercept_bets, r_bets, p_bets, se_bets = stats.linregress(complexities, avg_bets)
    print(f"\nAverage Total Bet vs Complexity:")
    print(f"  Linear regression: y = {slope_bets:.3f}x + {intercept_bets:.3f}")
    print(f"  Correlation coefficient (r): {r_bets:.3f}")
    print(f"  P-value: {p_bets:.6f}")
    print(f"  R-squared: {r_bets**2:.3f}")
    
    # Chi-square test for bankruptcy across complexity levels
    observed = [bankruptcy_counts[i] for i in range(len(complexities))]
    not_bankrupt = [sample_sizes[i] - bankruptcy_counts[i] for i in range(len(complexities))]
    
    if sum(observed) > 0:
        chi2, p_chi2 = stats.chisquare(observed)
        print(f"\nChi-square test for bankruptcy distribution:")
        print(f"  Chi-square statistic: {chi2:.3f}")
        print(f"  P-value: {p_chi2:.6f}")
    
    # Create single-model figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Bankruptcy Rate with trend line
    axes[0].plot(complexities, bankruptcy_rates, 'o-', linewidth=3, markersize=10, 
                color='#e74c3c', label='GPT-4o-mini', markerfacecolor='white', markeredgewidth=2)
    
    # Add linear trend line
    x_trend = np.array(complexities)
    y_trend = slope_bankruptcy * x_trend + intercept_bankruptcy
    axes[0].plot(x_trend, y_trend, '--', color='gray', alpha=0.7, linewidth=2)
    
    # Add statistical info
    axes[0].text(0.05, 0.95, f'$r$ = {r_bankruptcy:.3f}\n$p$ = {p_bankruptcy:.4f}',
                transform=axes[0].transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
                verticalalignment='top')
    
    axes[0].set_xlabel('Prompt Complexity', fontweight='bold')
    axes[0].set_ylabel('Bankruptcy Rate (%)', fontweight='bold')
    axes[0].set_title('Bankruptcy Rate', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(complexities)
    
    # Sample sizes removed for cleaner look
    
    # Plot 2: Average Game Rounds with trend line
    axes[1].plot(complexities, avg_rounds, 's-', linewidth=3, markersize=10,
                color='#3498db', label='GPT-4o-mini', markerfacecolor='white', markeredgewidth=2)
    
    y_trend_rounds = slope_rounds * x_trend + intercept_rounds
    axes[1].plot(x_trend, y_trend_rounds, '--', color='gray', alpha=0.7, linewidth=2)
    
    axes[1].text(0.05, 0.95, f'$r$ = {r_rounds:.3f}\n$p$ = {p_rounds:.4f}',
                transform=axes[1].transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
                verticalalignment='top')
    
    axes[1].set_xlabel('Prompt Complexity', fontweight='bold')
    axes[1].set_ylabel('Average Game Rounds', fontweight='bold')
    axes[1].set_title('Game Persistence', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(complexities)
    
    # Plot 3: Total Betting Amount with trend line  
    axes[2].plot(complexities, avg_bets, '^-', linewidth=3, markersize=10,
                color='#2ecc71', label='GPT-4o-mini', markerfacecolor='white', markeredgewidth=2)
    
    y_trend_bets = slope_bets * x_trend + intercept_bets
    axes[2].plot(x_trend, y_trend_bets, '--', color='gray', alpha=0.7, linewidth=2)
    
    axes[2].text(0.05, 0.95, f'$r$ = {r_bets:.3f}\n$p$ = {p_bets:.4f}',
                transform=axes[2].transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
                verticalalignment='top')
    
    axes[2].set_xlabel('Prompt Complexity', fontweight='bold')
    axes[2].set_ylabel('Average Total Bet ($)', fontweight='bold') 
    axes[2].set_title('Total Bet Size', fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xticks(complexities)
    
    plt.suptitle('Prompt Complexity and Risk-Taking Behavior',
                 fontsize=24, fontweight='bold', y=0.94)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.35)
    
    # Save figure
    plt.savefig('/home/ubuntu/llm_addiction/writing/figures/REAL_complexity_trend_comprehensive.pdf',
                dpi=300, bbox_inches='tight')
    plt.savefig('/home/ubuntu/llm_addiction/writing/figures/REAL_complexity_trend_comprehensive.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n✅ GPT-only complexity figure created with statistical analysis!")
    
    # Summary table
    print("\nComplexity Analysis Summary:")
    print("===========================")
    print(f"{'Complexity':<10} {'N':<6} {'Bankruptcy %':<12} {'Avg Rounds':<12} {'Avg Bet ($)':<12}")
    print("-" * 60)
    for i, comp in enumerate(complexities):
        print(f"{comp:<10} {sample_sizes[i]:<6} {bankruptcy_rates[i]:<12.1f} {avg_rounds[i]:<12.1f} {avg_bets[i]:<12.1f}")
    
    return {
        'bankruptcy': {'r': r_bankruptcy, 'p': p_bankruptcy, 'slope': slope_bankruptcy},
        'rounds': {'r': r_rounds, 'p': p_rounds, 'slope': slope_rounds},
        'betting': {'r': r_bets, 'p': p_bets, 'slope': slope_bets}
    }

if __name__ == "__main__":
    results = create_gpt_only_complexity_figure()
