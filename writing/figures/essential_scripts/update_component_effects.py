#!/usr/bin/env python3
"""
Update REAL_component_effects.png to:
1. Remove sample size subplot (keep only 3 subplots)
2. Fix "+" prefix on negative numbers (improve readability)  
3. Add standard error bars for statistical significance
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
    'font.size': 22,
    'axes.titlesize': 24,
    'axes.labelsize': 22,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20
})

def load_gpt_data():
    """Load the latest GPT experimental data"""
    print("Loading GPT experimental data...")
    
    with open('/data/llm_addiction/gpt_results_fixed_parsing/gpt_fixed_parsing_complete_20250919_151240.json') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data['results'])} GPT experiments")
    return data['results']

def analyze_component_effects(gpt_results):
    """Analyze individual prompt component effects"""
    print("Analyzing individual prompt component effects...")
    
    # Components to analyze
    components = ['G', 'M', 'P', 'R', 'W']
    component_names = {
        'G': 'G',
        'M': 'M',
        'P': 'P',
        'R': 'H',
        'W': 'W'
    }
    
    results = {}
    
    for component in components:
        print(f"  Analyzing component: {component}")
        
        # Separate experiments with/without component
        with_component = []
        without_component = []
        
        for exp in gpt_results:
            prompt_combo = exp.get('prompt_combo', 'BASE')
            
            if component in prompt_combo:
                with_component.append(exp)
            else:
                without_component.append(exp)
        
        # Calculate metrics for each group
        def calc_metrics(experiments):
            if not experiments:
                return {'bankruptcy_rate': 0, 'avg_bet': 0, 'avg_rounds': 0, 'count': 0}
            
            bankruptcy_count = sum(1 for exp in experiments if exp.get('is_bankrupt', False))
            bankruptcy_rate = (bankruptcy_count / len(experiments)) * 100
            
            avg_bet = np.mean([exp.get('total_bet', 0) for exp in experiments])
            avg_rounds = np.mean([exp.get('total_rounds', 0) for exp in experiments])
            
            return {
                'bankruptcy_rate': bankruptcy_rate,
                'avg_bet': avg_bet, 
                'avg_rounds': avg_rounds,
                'count': len(experiments)
            }
        
        with_metrics = calc_metrics(with_component)
        without_metrics = calc_metrics(without_component)
        
        # Calculate effects and standard errors - CORRECTED for binary data
        def calc_effect_and_se(with_vals, without_vals, metric_name):
            if metric_name == 'is_bankrupt':
                # For binary data: use proportion difference standard error
                bankruptcy_with = sum(1 for exp in with_component if exp.get('is_bankrupt', False))
                bankruptcy_without = sum(1 for exp in without_component if exp.get('is_bankrupt', False))

                n_with = len(with_component)
                n_without = len(without_component)

                # Proportions (0-1 scale)
                p_with = bankruptcy_with / n_with if n_with > 0 else 0
                p_without = bankruptcy_without / n_without if n_without > 0 else 0

                # Effect in percentage points
                effect = (p_with - p_without) * 100

                # Proper standard error for proportion difference
                if n_with > 0 and n_without > 0:
                    se_with = np.sqrt(p_with * (1 - p_with) / n_with)
                    se_without = np.sqrt(p_without * (1 - p_without) / n_without)
                    combined_se = np.sqrt(se_with**2 + se_without**2) * 100  # Convert to percentage
                else:
                    combined_se = 0

            else:
                # For continuous data: use original method
                with_data = [exp.get(metric_name, 0) for exp in with_component]
                without_data = [exp.get(metric_name, 0) for exp in without_component]

                effect = np.mean(with_data) - np.mean(without_data)

                # Standard error for continuous data
                if len(with_data) > 1 and len(without_data) > 1:
                    se1 = np.std(with_data, ddof=1) / np.sqrt(len(with_data))
                    se2 = np.std(without_data, ddof=1) / np.sqrt(len(without_data))
                    combined_se = np.sqrt(se1**2 + se2**2)
                else:
                    combined_se = 0

            return effect, combined_se
        
        bankruptcy_effect, bankruptcy_se = calc_effect_and_se(with_component, without_component, 'is_bankrupt')
        bet_effect, bet_se = calc_effect_and_se(with_component, without_component, 'total_bet')
        rounds_effect, rounds_se = calc_effect_and_se(with_component, without_component, 'total_rounds')
        
        results[component] = {
            'name': component_names[component],
            'with_count': with_metrics['count'],
            'without_count': without_metrics['count'],
            'bankruptcy_effect': bankruptcy_effect,
            'bankruptcy_se': bankruptcy_se,
            'bet_effect': bet_effect,
            'bet_se': bet_se,
            'rounds_effect': rounds_effect,
            'rounds_se': rounds_se
        }
        
        print(f"    With {component}: {with_metrics['count']}, Without: {without_metrics['count']}")
        print(f"    Bankruptcy effect: {bankruptcy_effect:+.1f}% ± {bankruptcy_se:.1f}")
    
    return results

def create_updated_component_figure(component_results):
    """Create updated component effects figure (3 subplots, no sample sizes)"""
    print("Creating updated component effects figure...")
    
    # Extract data
    components = ['G', 'M', 'P', 'R', 'W']
    names = [component_results[c]['name'] for c in components]
    
    bankruptcy_effects = [component_results[c]['bankruptcy_effect'] for c in components]
    bankruptcy_ses = [component_results[c]['bankruptcy_se'] for c in components]
    
    bet_effects = [component_results[c]['bet_effect'] for c in components]
    bet_ses = [component_results[c]['bet_se'] for c in components]
    
    rounds_effects = [component_results[c]['rounds_effect'] for c in components]
    rounds_ses = [component_results[c]['rounds_se'] for c in components]
    
    # Create 1x3 horizontal layout
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Define colors
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

    # Subplot 1: Bankruptcy Rate Effects
    ax1 = axes[0]
    bars1 = ax1.bar(names, bankruptcy_effects, color=colors, alpha=0.8,
                    yerr=bankruptcy_ses, capsize=5, ecolor='black')
    ax1.set_ylabel('Bankruptcy Rate (%)', fontweight='bold')
    # ax1.set_title('Bankruptcy Rate', fontweight='bold')  # Removed - redundant with y-axis
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    # Set y-axis range based on data (roughly -3% to +8%)
    ax1.set_ylim(-4, 11)

    # Remove value labels on bars

    # Subplot 2: Game Duration Effects (MOVED from position 3 to 2)
    ax2 = axes[1]
    bars2 = ax2.bar(names, rounds_effects, color=colors, alpha=0.8,
                    yerr=rounds_ses, capsize=5, ecolor='black')
    ax2.set_ylabel('Game Rounds', fontweight='bold')
    # ax2.set_title('Game Duration', fontweight='bold')  # Removed - redundant with y-axis
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    # Set y-axis range for rounds (expanded to accommodate G value ~4.4)
    ax2.set_ylim(-0.5, 5.0)

    # Remove value labels on bars

    # Subplot 3: Betting Amount Effects (MOVED from position 2 to 3)
    ax3 = axes[2]
    bars3 = ax3.bar(names, bet_effects, color=colors, alpha=0.8,
                    yerr=bet_ses, capsize=5, ecolor='black')
    ax3.set_ylabel('Betting Amount ($)', fontweight='bold')
    # ax3.set_title('Betting Amount', fontweight='bold')  # Removed - redundant with y-axis
    # Add x-labels to all subplots for horizontal layout
    ax1.set_xlabel('Prompt Component', fontweight='bold')
    ax2.set_xlabel('Prompt Component', fontweight='bold')
    ax3.set_xlabel('Prompt Component', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    # Set y-axis range for betting amounts (expanded to accommodate G value ~73)
    ax3.set_ylim(-10, 80)
    
    # Remove value labels on bars
    
    # No need to remove subplot with 1x3 layout
    
    # Add simplified title
    plt.suptitle('Component Effects on Risk-Taking Behavior',
                 fontsize=26, fontweight='bold', y=0.78)

    # Adjust layout with padding for title and spacing between subplots
    plt.tight_layout(rect=[0, 0, 1, 0.82])
    plt.subplots_adjust(wspace=0.3)
    
    # Save figure with explicit format specification
    plt.savefig('/home/ubuntu/llm_addiction/writing/figures/REAL_component_effects.pdf',
                dpi=300, bbox_inches='tight', format='pdf', facecolor='white', edgecolor='none')
    plt.savefig('/home/ubuntu/llm_addiction/writing/figures/REAL_component_effects.png',
                dpi=300, bbox_inches='tight', format='png', facecolor='white', edgecolor='none')
    plt.close()
    
    # Print results
    print("\\n" + "="*60)
    print("✅ UPDATED COMPONENT EFFECTS RESULTS:")
    print("="*60)
    
    for comp in components:
        name = component_results[comp]['name']
        bank_eff = component_results[comp]['bankruptcy_effect']
        bank_se = component_results[comp]['bankruptcy_se']
        
        print(f"   {name:10}: {bank_eff:+.1f}% ± {bank_se:.1f} (bankruptcy)")
    
    print("\\n   ✅ Removed sample size subplot")
    print("   ✅ Fixed '+' prefix on negative numbers")
    print("   ✅ Added standard error bars")
    print("="*60)

def main():
    print("="*80)
    print("UPDATING COMPONENT EFFECTS FIGURE")
    print("="*80)
    
    # Load data
    gpt_results = load_gpt_data()
    
    # Analyze components
    component_results = analyze_component_effects(gpt_results)
    
    # Create updated figure
    create_updated_component_figure(component_results)
    
    print("\\n" + "="*80)
    print("✅ COMPONENT EFFECTS FIGURE UPDATED!")
    print("✅ Removed 4th subplot (sample sizes)")
    print("✅ Fixed negative number formatting (no '+' prefix)")
    print("✅ Added error bars for statistical significance")
    print("="*80)

if __name__ == "__main__":
    main()
