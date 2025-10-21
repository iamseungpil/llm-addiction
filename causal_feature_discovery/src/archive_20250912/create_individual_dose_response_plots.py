#!/usr/bin/env python3
"""
Create individual dose-response plots for top causal features
Show violin plots and before/after intervention effects
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns
from collections import defaultdict

# Set style for academic plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = ['DejaVu Sans', 'Liberation Sans']
plt.rcParams['font.size'] = 10

def create_individual_dose_response_plots():
    """Create individual feature dose-response analysis plots"""
    
    print("Creating individual dose-response plots...")
    
    # Load top features
    with open('/data/llm_addiction/results/top_dose_response_features_20250910.json', 'r') as f:
        data = json.load(f)
    
    top_features = data['top_features'][:6]  # Top 6 features
    
    # Load full results for detailed analysis
    gpu4_file = '/data/llm_addiction/results/patching_population_mean_final_20250905_150612.json'
    gpu5_file = '/data/llm_addiction/results/patching_population_mean_final_20250905_085027.json'
    
    with open(gpu4_file, 'r') as f:
        gpu4_data = json.load(f)
    
    with open(gpu5_file, 'r') as f:
        gpu5_data = json.load(f)
    
    all_results = gpu4_data['all_results'] + gpu5_data['all_results']
    
    # Group results by feature
    feature_data = defaultdict(list)
    for result in all_results:
        key = f"L{result['layer']}-{result['feature_id']}"
        feature_data[key].append(result)
    
    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, feat_info in enumerate(top_features):
        ax = axes[i]
        feature_key = feat_info['feature_key']
        results = feature_data[feature_key]
        
        # Separate by prompt type
        risky_results = [r for r in results if r['prompt_type'] == 'risky']
        safe_results = [r for r in results if r['prompt_type'] == 'safe']
        
        # Sort by scale
        risky_results.sort(key=lambda x: x['scale'])
        safe_results.sort(key=lambda x: x['scale'])
        
        # Extract data for plotting
        scales = [r['scale'] for r in risky_results]
        risky_bets = [r['avg_bet'] for r in risky_results]
        safe_bets = [r['avg_bet'] for r in safe_results]
        risky_stops = [r['stop_rate'] * 100 for r in risky_results]  # Convert to percentage
        safe_stops = [r['stop_rate'] * 100 for r in safe_results]
        
        # Plot betting amounts
        ax.plot(scales, risky_bets, 'o-', linewidth=3, markersize=8, 
                color='#e74c3c', label='Risky Prompt', alpha=0.9)
        ax.plot(scales, safe_bets, 's-', linewidth=3, markersize=8,
                color='#3498db', label='Safe Prompt', alpha=0.9)
        
        # Add value annotations
        for j, (scale, risky_bet, safe_bet) in enumerate(zip(scales, risky_bets, safe_bets)):
            ax.annotate(f'${risky_bet:.0f}', (scale, risky_bet), 
                       textcoords="offset points", xytext=(0,10), ha='center', 
                       fontsize=8, color='#e74c3c', fontweight='bold')
            ax.annotate(f'${safe_bet:.0f}', (scale, safe_bet), 
                       textcoords="offset points", xytext=(0,-15), ha='center',
                       fontsize=8, color='#3498db', fontweight='bold')
        
        # Styling
        ax.set_xlabel('Intervention Scale', fontweight='bold', fontsize=11)
        ax.set_ylabel('Average Betting Amount ($)', fontweight='bold', fontsize=11)
        ax.set_title(f'{feature_key}\nBet Effect: ${feat_info["bet_effect"]:.1f}', 
                    fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_xlim(min(scales) - 0.05, max(scales) + 0.05)
        
        # Add scale labels
        for scale in scales:
            ax.axvline(x=scale, color='gray', linestyle='--', alpha=0.2)
        
        # Add intervention annotations
        ax.text(scales[0], ax.get_ylim()[1]*0.95, f'{scales[0]}x\n(Safe)', 
                ha='center', va='top', fontsize=8, 
                bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', alpha=0.7))
        ax.text(scales[1], ax.get_ylim()[1]*0.95, f'{scales[1]}x\n(Baseline)', 
                ha='center', va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.7))
        ax.text(scales[2], ax.get_ylim()[1]*0.95, f'{scales[2]}x\n(Risky)', 
                ha='center', va='top', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='lightcoral', alpha=0.7))
    
    plt.suptitle('Individual Feature Dose-Response Curves\n(Top 6 Causal Features with Largest Effects)', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save the plot
    output_path = '/home/ubuntu/llm_addiction/writing/figures/individual_dose_response_plots.png'
    plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), format='pdf', dpi=300, bbox_inches='tight')
    
    print(f"✅ Individual dose-response plots saved: {output_path}")
    
    # Create violin plots for before/after comparison
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 12))
    axes2 = axes2.flatten()
    
    for i, feat_info in enumerate(top_features):
        ax = axes2[i]
        feature_key = feat_info['feature_key']
        results = feature_data[feature_key]
        
        # Prepare data for violin plot
        scale_0_5_bets = []
        scale_1_0_bets = []  # Baseline
        scale_1_5_bets = []
        
        for result in results:
            if result['scale'] == 0.5:
                scale_0_5_bets.append(result['avg_bet'])
            elif result['scale'] == 1.0:
                scale_1_0_bets.append(result['avg_bet'])
            elif result['scale'] == 1.5:
                scale_1_5_bets.append(result['avg_bet'])
        
        # Create violin plot data
        plot_data = []
        labels = []
        
        if scale_0_5_bets:
            plot_data.append(scale_0_5_bets)
            labels.append('0.5x\n(Safe)')
        
        if scale_1_0_bets:
            plot_data.append(scale_1_0_bets)
            labels.append('1.0x\n(Baseline)')
        
        if scale_1_5_bets:
            plot_data.append(scale_1_5_bets)
            labels.append('1.5x\n(Risky)')
        
        # Create violin plot
        if plot_data:
            parts = ax.violinplot(plot_data, positions=range(len(plot_data)), 
                                 showmeans=True, showmedians=True)
            
            # Color the violins
            colors = ['#3498db', '#f39c12', '#e74c3c']
            for j, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[j % len(colors)])
                pc.set_alpha(0.7)
        
        # Styling
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel('Betting Amount ($)', fontweight='bold', fontsize=11)
        ax.set_title(f'{feature_key}\nDistribution Across Scales', 
                    fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add mean values as text
        if plot_data:
            for j, data in enumerate(plot_data):
                mean_val = np.mean(data)
                ax.text(j, ax.get_ylim()[1]*0.9, f'μ=${mean_val:.1f}', 
                       ha='center', fontsize=9, fontweight='bold')
    
    plt.suptitle('Feature Intervention Effect Distributions\n(Violin Plots Showing Betting Amount Distributions)', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save violin plots
    violin_path = '/home/ubuntu/llm_addiction/writing/figures/individual_intervention_violins.png'
    plt.savefig(violin_path, format='png', dpi=300, bbox_inches='tight')
    plt.savefig(violin_path.replace('.png', '.pdf'), format='pdf', dpi=300, bbox_inches='tight')
    
    print(f"✅ Violin plots saved: {violin_path}")
    
    return top_features

if __name__ == '__main__':
    create_individual_dose_response_plots()