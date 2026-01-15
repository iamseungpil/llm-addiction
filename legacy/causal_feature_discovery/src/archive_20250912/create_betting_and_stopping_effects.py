#!/usr/bin/env python3
"""
Create comprehensive plots showing BOTH betting and stopping effects
- Betting Amount changes (bet effect)
- Stop Rate changes (stop effect) 
Show both effects clearly for individual features
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns
from collections import defaultdict

# Set style for academic plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = ['DejaVu Sans', 'Liberation Sans']
plt.rcParams['font.size'] = 9

def load_and_verify_stop_data():
    """Load and verify stop rate data alongside betting data"""
    
    print("Loading betting and stopping effect data...")
    
    # Load top features
    with open('/data/llm_addiction/results/top_dose_response_features_20250910.json', 'r') as f:
        data = json.load(f)
    
    top_features = data['top_features'][:6]
    
    # Load full results
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
    
    # Verify and extract both betting and stopping data
    verified_features = []
    
    for feat_info in top_features:
        feature_key = feat_info['feature_key']
        results = feature_data[feature_key]
        
        # Separate by prompt type
        risky_results = [r for r in results if r['prompt_type'] == 'risky']
        safe_results = [r for r in results if r['prompt_type'] == 'safe']
        
        if len(risky_results) != 3 or len(safe_results) != 3:
            continue
        
        # Sort by scale
        risky_results.sort(key=lambda x: x['scale'])
        safe_results.sort(key=lambda x: x['scale'])
        
        # Extract both betting and stopping data
        risky_scales = [r['scale'] for r in risky_results]
        safe_scales = [r['scale'] for r in safe_results]
        
        risky_bets = [r['avg_bet'] for r in risky_results]
        safe_bets = [r['avg_bet'] for r in safe_results]
        
        risky_stops = [r['stop_rate'] * 100 for r in risky_results]  # Convert to percentage
        safe_stops = [r['stop_rate'] * 100 for r in safe_results]
        
        # Verify no None values
        if any(x is None for x in risky_bets + safe_bets + risky_stops + safe_stops):
            continue
        
        # Calculate actual effects
        risky_bet_effect = max(risky_bets) - min(risky_bets)
        safe_bet_effect = max(safe_bets) - min(safe_bets)
        bet_effect = max(risky_bet_effect, safe_bet_effect)
        
        risky_stop_effect = abs(max(risky_stops) - min(risky_stops))
        safe_stop_effect = abs(max(safe_stops) - min(safe_stops))
        stop_effect = max(risky_stop_effect, safe_stop_effect)
        
        verified_features.append({
            'feature_key': feature_key,
            'layer': feat_info['layer'],
            'feature_id': feat_info['feature_id'],
            'bet_effect': bet_effect,
            'stop_effect': stop_effect,
            'risky_data': {
                'scales': risky_scales,
                'bets': risky_bets,
                'stops': risky_stops
            },
            'safe_data': {
                'scales': safe_scales,
                'bets': safe_bets,
                'stops': safe_stops
            }
        })
        
        print(f"✅ {feature_key}:")
        print(f"   Betting - Risky: {risky_bets}, Safe: {safe_bets}")
        print(f"   Stopping - Risky: {[f'{x:.1f}%' for x in risky_stops]}, Safe: {[f'{x:.1f}%' for x in safe_stops]}")
        print(f"   Effects - Bet: ${bet_effect:.1f}, Stop: {stop_effect:.1f}%")
        print()
    
    return verified_features

def create_betting_and_stopping_plots():
    """Create plots showing both betting and stopping effects"""
    
    features = load_and_verify_stop_data()
    
    # Create 3x4 subplot layout (3 rows, 4 columns)
    # Each feature gets 2 subplots: betting effect and stopping effect
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    for i, feat in enumerate(features):
        row = i
        
        # Betting effect subplot (left column of each pair)
        ax_bet = axes[row, 0] if i < 3 else axes[row-3, 2]
        # Stopping effect subplot (right column of each pair)
        ax_stop = axes[row, 1] if i < 3 else axes[row-3, 3]
        
        feature_key = feat['feature_key']
        risky_data = feat['risky_data']
        safe_data = feat['safe_data']
        
        # Plot betting effects
        ax_bet.plot(risky_data['scales'], risky_data['bets'], 'o-', 
                   linewidth=3, markersize=8, color='#e74c3c', 
                   label='Risky Context', alpha=0.9)
        ax_bet.plot(safe_data['scales'], safe_data['bets'], 's-', 
                   linewidth=3, markersize=8, color='#3498db', 
                   label='Safe Context', alpha=0.9)
        
        # Betting annotations
        for j, (scale, risky_bet, safe_bet) in enumerate(zip(
            risky_data['scales'], risky_data['bets'], safe_data['bets'])):
            ax_bet.annotate(f'${risky_bet:.0f}', (scale, risky_bet), 
                           textcoords="offset points", xytext=(0,10), ha='center', 
                           fontsize=7, color='#c0392b', fontweight='bold')
            ax_bet.annotate(f'${safe_bet:.0f}', (scale, safe_bet), 
                           textcoords="offset points", xytext=(0,-15), ha='center',
                           fontsize=7, color='#2980b9', fontweight='bold')
        
        # Betting plot styling
        ax_bet.set_xlabel('Intervention Scale', fontweight='bold', fontsize=9)
        ax_bet.set_ylabel('Betting Amount ($)', fontweight='bold', fontsize=9)
        ax_bet.set_title(f'{feature_key} - Betting Effect\nΔ${feat["bet_effect"]:.1f}', 
                        fontweight='bold', fontsize=10)
        ax_bet.grid(True, alpha=0.3)
        ax_bet.legend(fontsize=8)
        ax_bet.set_xlim(0.4, 1.6)
        
        # Plot stopping effects
        ax_stop.plot(risky_data['scales'], risky_data['stops'], 'o-', 
                    linewidth=3, markersize=8, color='#e74c3c', 
                    label='Risky Context', alpha=0.9)
        ax_stop.plot(safe_data['scales'], safe_data['stops'], 's-', 
                    linewidth=3, markersize=8, color='#3498db', 
                    label='Safe Context', alpha=0.9)
        
        # Stopping annotations
        for j, (scale, risky_stop, safe_stop) in enumerate(zip(
            risky_data['scales'], risky_data['stops'], safe_data['stops'])):
            ax_stop.annotate(f'{risky_stop:.0f}%', (scale, risky_stop), 
                            textcoords="offset points", xytext=(0,10), ha='center', 
                            fontsize=7, color='#c0392b', fontweight='bold')
            ax_stop.annotate(f'{safe_stop:.0f}%', (scale, safe_stop), 
                            textcoords="offset points", xytext=(0,-15), ha='center',
                            fontsize=7, color='#2980b9', fontweight='bold')
        
        # Stopping plot styling
        ax_stop.set_xlabel('Intervention Scale', fontweight='bold', fontsize=9)
        ax_stop.set_ylabel('Stop Rate (%)', fontweight='bold', fontsize=9)
        ax_stop.set_title(f'{feature_key} - Stopping Effect\nΔ{feat["stop_effect"]:.1f}%', 
                         fontweight='bold', fontsize=10)
        ax_stop.grid(True, alpha=0.3)
        ax_stop.legend(fontsize=8)
        ax_stop.set_xlim(0.4, 1.6)
        ax_stop.set_ylim(0, 100)
        
        # Add scale labels to both plots
        for ax in [ax_bet, ax_stop]:
            for scale in risky_data['scales']:
                ax.axvline(x=scale, color='gray', linestyle='--', alpha=0.2)
    
    plt.suptitle('Individual Feature Effects: Betting vs Stopping Behavior\n' + 
                 'Left: Betting Amount Changes | Right: Stop Rate Changes', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save the plot
    output_path = '/home/ubuntu/llm_addiction/writing/figures/betting_and_stopping_effects.png'
    plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), format='pdf', dpi=300, bbox_inches='tight')
    
    print(f"✅ Betting and stopping effects plot saved: {output_path}")
    
    return features

def create_dual_axis_plots():
    """Create plots with dual y-axes showing both effects on same plot"""
    
    features = load_and_verify_stop_data()
    
    # Create 2x3 subplot layout
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, feat in enumerate(features):
        ax = axes[i]
        
        feature_key = feat['feature_key']
        risky_data = feat['risky_data']
        safe_data = feat['safe_data']
        
        # Create second y-axis
        ax2 = ax.twinx()
        
        # Plot betting on primary axis (left)
        line1 = ax.plot(risky_data['scales'], risky_data['bets'], 'o-', 
                       linewidth=3, markersize=8, color='#e74c3c', 
                       label='Risky Betting', alpha=0.9)
        line2 = ax.plot(safe_data['scales'], safe_data['bets'], 's-', 
                       linewidth=3, markersize=8, color='#3498db', 
                       label='Safe Betting', alpha=0.9)
        
        # Plot stopping on secondary axis (right)
        line3 = ax2.plot(risky_data['scales'], risky_data['stops'], '^--', 
                        linewidth=2, markersize=6, color='#8e44ad', 
                        label='Risky Stopping', alpha=0.7)
        line4 = ax2.plot(safe_data['scales'], safe_data['stops'], 'v--', 
                        linewidth=2, markersize=6, color='#16a085', 
                        label='Safe Stopping', alpha=0.7)
        
        # Styling
        ax.set_xlabel('Intervention Scale', fontweight='bold', fontsize=10)
        ax.set_ylabel('Betting Amount ($)', fontweight='bold', fontsize=10, color='blue')
        ax2.set_ylabel('Stop Rate (%)', fontweight='bold', fontsize=10, color='purple')
        ax.set_title(f'{feature_key}\nBet Δ${feat["bet_effect"]:.1f} | Stop Δ{feat["stop_effect"]:.1f}%', 
                    fontweight='bold', fontsize=11)
        
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.4, 1.6)
        ax2.set_ylim(0, 100)
        
        # Color y-axis labels
        ax.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='purple')
        
        # Combined legend
        lines = line1 + line2 + line3 + line4
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, fontsize=8, loc='upper left')
    
    plt.suptitle('Dual-Axis Feature Effects: Betting (Blue) and Stopping (Purple)', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save dual axis plot
    dual_path = '/home/ubuntu/llm_addiction/writing/figures/dual_axis_betting_stopping.png'
    plt.savefig(dual_path, format='png', dpi=300, bbox_inches='tight')
    plt.savefig(dual_path.replace('.png', '.pdf'), format='pdf', dpi=300, bbox_inches='tight')
    
    print(f"✅ Dual-axis plot saved: {dual_path}")

if __name__ == '__main__':
    print("Creating comprehensive betting and stopping analysis...")
    features = create_betting_and_stopping_plots()
    create_dual_axis_plots()
    print("✅ All betting and stopping plots created!")
    print("\nKey insights:")
    for feat in features:
        print(f"- {feat['feature_key']}: Bet effect ${feat['bet_effect']:.1f}, Stop effect {feat['stop_effect']:.1f}%")