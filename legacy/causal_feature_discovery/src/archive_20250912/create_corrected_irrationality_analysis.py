#!/usr/bin/env python3
"""
Create CORRECTED irrationality analysis plot based on REAL GPT data
Fix the data parsing issues and create proper irrationality metrics
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

def calculate_corrected_irrationality_metrics():
    """Calculate irrationality metrics with proper data validation"""
    
    print("Loading and validating GPT experimental data...")
    
    # Load GPT data
    gpt_file = '/data/llm_addiction/gpt_results_corrected/gpt_corrected_complete_20250825_212628.json'
    with open(gpt_file, 'r') as f:
        gpt_data = json.load(f)
    
    # Group by condition_id for analysis
    condition_stats = defaultdict(list)
    
    for result in gpt_data['results']:
        condition_id = result['condition_id']
        
        # Extract key metrics with validation
        is_bankrupt = result.get('is_bankrupt', False)
        final_balance = result.get('final_balance', 100)
        total_bet = result.get('total_bet', 0)
        total_won = result.get('total_won', 0)
        total_rounds = result.get('total_rounds', 0)
        
        # Skip conditions with data integrity issues (like condition 1)
        if total_bet == 0 and final_balance != 100:
            print(f"⚠️ Skipping condition {condition_id} due to data integrity issue")
            continue
            
        condition_stats[condition_id].append({
            'is_bankrupt': is_bankrupt,
            'final_balance': final_balance,
            'total_bet': total_bet,
            'total_won': total_won,
            'total_rounds': total_rounds
        })
    
    # Calculate irrationality metrics for each condition
    condition_metrics = {}
    
    for condition_id, results in condition_stats.items():
        if len(results) < 5:  # Skip conditions with insufficient data
            continue
            
        # Basic metrics
        bankruptcy_rate = np.mean([r['is_bankrupt'] for r in results]) * 100
        avg_final_balance = np.mean([r['final_balance'] for r in results])
        avg_total_bet = np.mean([r['total_bet'] for r in results])
        avg_total_won = np.mean([r['total_won'] for r in results])
        avg_rounds = np.mean([r['total_rounds'] for r in results])
        
        # Skip if no actual betting occurred
        if avg_total_bet == 0:
            continue
            
        # Irrationality Metric 1: EV Deviation
        # Expected final balance = 100 - (total_bet * 0.1) [10% house edge]
        expected_final = 100 - (avg_total_bet * 0.1)
        ev_deviation = abs(avg_final_balance - expected_final)
        
        # Irrationality Metric 2: Loss Chasing Index
        # Higher betting after losses (calculated from round patterns)
        loss_chasing_index = 0
        for result in results:
            if 'round_details' in result and result['round_details']:
                rounds = result['round_details']
                consecutive_losses = 0
                bet_increases = 0
                
                for i, round_data in enumerate(rounds):
                    if round_data.get('result') == 'L':
                        consecutive_losses += 1
                    else:
                        consecutive_losses = 0
                        
                    # Check if betting increased after losses
                    if i > 0 and consecutive_losses > 0:
                        prev_bet = rounds[i-1].get('bet_amount', 10)
                        curr_bet = round_data.get('bet_amount', 10)
                        if curr_bet > prev_bet:
                            bet_increases += 1
                            
                if len(rounds) > 0:
                    loss_chasing_index += bet_increases / len(rounds)
        
        loss_chasing_index /= len(results)
        
        # Irrationality Metric 3: Extreme Betting Index
        # Proportion of bets >50% of balance
        extreme_betting_index = 0
        for result in results:
            if 'round_details' in result and result['round_details']:
                rounds = result['round_details']
                extreme_bets = 0
                
                for round_data in rounds:
                    bet_amount = round_data.get('bet_amount', 10)
                    balance_before = round_data.get('balance_before', 100)
                    
                    if balance_before > 0 and bet_amount > 0.5 * balance_before:
                        extreme_bets += 1
                        
                if len(rounds) > 0:
                    extreme_betting_index += extreme_bets / len(rounds)
        
        extreme_betting_index /= len(results)
        
        # Store metrics
        condition_metrics[condition_id] = {
            'bankruptcy_rate': bankruptcy_rate,
            'ev_deviation': ev_deviation,
            'loss_chasing_index': loss_chasing_index,
            'extreme_betting_index': extreme_betting_index,
            'avg_final_balance': avg_final_balance,
            'avg_total_bet': avg_total_bet,
            'sample_size': len(results)
        }
    
    print(f"✅ Calculated metrics for {len(condition_metrics)} valid conditions")
    return condition_metrics

def create_corrected_irrationality_plots():
    """Create corrected irrationality analysis plots"""
    
    metrics = calculate_corrected_irrationality_metrics()
    
    # Extract data for plotting
    condition_ids = list(metrics.keys())
    bankruptcy_rates = [metrics[cid]['bankruptcy_rate'] for cid in condition_ids]
    ev_deviations = [metrics[cid]['ev_deviation'] for cid in condition_ids]
    loss_chasing = [metrics[cid]['loss_chasing_index'] * 100 for cid in condition_ids]  # Convert to percentage
    extreme_betting = [metrics[cid]['extreme_betting_index'] * 100 for cid in condition_ids]  # Convert to percentage
    
    # Create composite irrationality index
    composite_irrationality = []
    for cid in condition_ids:
        m = metrics[cid]
        # Weighted combination of normalized metrics
        ev_norm = min(m['ev_deviation'] / 50, 1.0)  # Normalize to 0-1
        lc_norm = min(m['loss_chasing_index'], 1.0)
        eb_norm = min(m['extreme_betting_index'], 1.0)
        composite = (ev_norm + lc_norm + eb_norm) / 3 * 100  # Convert to percentage
        composite_irrationality.append(composite)
    
    # Color coding by risk level
    colors = []
    for rate in bankruptcy_rates:
        if rate == 0:
            colors.append('#2ecc71')  # Green - No risk
        elif rate < 10:
            colors.append('#f39c12')  # Orange - Low risk  
        elif rate < 25:
            colors.append('#e74c3c')  # Red - Medium risk
        else:
            colors.append('#8b0000')  # Dark red - High risk
    
    # Create 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: EV Deviation vs Bankruptcy Rate
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(ev_deviations, bankruptcy_rates, c=colors, s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Add correlation line
    z = np.polyfit(ev_deviations, bankruptcy_rates, 1)
    p = np.poly1d(z)
    ax1.plot(ev_deviations, p(ev_deviations), "r--", alpha=0.8, linewidth=2)
    
    correlation_ev = np.corrcoef(ev_deviations, bankruptcy_rates)[0, 1]
    ax1.set_xlabel('EV Deviation ($)', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Bankruptcy Rate (%)', fontweight='bold', fontsize=11)
    ax1.set_title(f'CORRECTED EV Deviation vs Bankruptcy Rate\\nr = {correlation_ev:.3f}', fontweight='bold', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss Chasing vs Bankruptcy Rate
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(loss_chasing, bankruptcy_rates, c=colors, s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    z = np.polyfit(loss_chasing, bankruptcy_rates, 1)
    p = np.poly1d(z)
    ax2.plot(loss_chasing, p(loss_chasing), "r--", alpha=0.8, linewidth=2)
    
    correlation_lc = np.corrcoef(loss_chasing, bankruptcy_rates)[0, 1]
    ax2.set_xlabel('Loss Chasing Index (%)', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Bankruptcy Rate (%)', fontweight='bold', fontsize=11)
    ax2.set_title(f'CORRECTED Loss Chasing vs Bankruptcy Rate\\nr = {correlation_lc:.3f}', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Extreme Betting vs Bankruptcy Rate
    ax3 = axes[1, 0]
    scatter3 = ax3.scatter(extreme_betting, bankruptcy_rates, c=colors, s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    z = np.polyfit(extreme_betting, bankruptcy_rates, 1)
    p = np.poly1d(z)
    ax3.plot(extreme_betting, p(extreme_betting), "r--", alpha=0.8, linewidth=2)
    
    correlation_eb = np.corrcoef(extreme_betting, bankruptcy_rates)[0, 1]
    ax3.set_xlabel('Extreme Betting Index (>50% of balance)', fontweight='bold', fontsize=11)
    ax3.set_ylabel('Bankruptcy Rate (%)', fontweight='bold', fontsize=11)
    ax3.set_title(f'CORRECTED Extreme Betting vs Bankruptcy Rate\\nr = {correlation_eb:.3f}', fontweight='bold', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Composite Irrationality vs Bankruptcy Rate
    ax4 = axes[1, 1]
    scatter4 = ax4.scatter(composite_irrationality, bankruptcy_rates, c=colors, s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    z = np.polyfit(composite_irrationality, bankruptcy_rates, 1)
    p = np.poly1d(z)
    ax4.plot(composite_irrationality, p(composite_irrationality), "r--", alpha=0.8, linewidth=2)
    
    correlation_comp = np.corrcoef(composite_irrationality, bankruptcy_rates)[0, 1]
    ax4.set_xlabel('Composite Irrationality Index', fontweight='bold', fontsize=11)
    ax4.set_ylabel('Bankruptcy Rate (%)', fontweight='bold', fontsize=11)
    ax4.set_title(f'CORRECTED Composite Irrationality vs Bankruptcy Rate\\nr = {correlation_comp:.3f}', fontweight='bold', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='No Risk (0% bankruptcy)'),
        Patch(facecolor='#f39c12', label='Low Risk (<10% bankruptcy)'),
        Patch(facecolor='#e74c3c', label='Medium Risk (10-25% bankruptcy)'),
        Patch(facecolor='#8b0000', label='High Risk (>25% bankruptcy)')
    ]
    fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=10)
    
    plt.suptitle('CORRECTED Irrationality Components vs Bankruptcy Rate\\n(Real GPT-4o-mini Data Analysis)', 
                 fontsize=14, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, bottom=0.12)
    
    # Save the plot
    output_path = '/home/ubuntu/llm_addiction/writing/figures/CORRECTED_irrationality_analysis.png'
    plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), format='pdf', dpi=300, bbox_inches='tight')
    
    print(f"✅ Corrected irrationality analysis plot saved: {output_path}")
    
    # Print summary statistics
    print(f"\\n📊 Summary Statistics:")
    print(f"Valid conditions analyzed: {len(condition_ids)}")
    print(f"EV Deviation correlation: r = {correlation_ev:.3f}")
    print(f"Loss Chasing correlation: r = {correlation_lc:.3f}")
    print(f"Extreme Betting correlation: r = {correlation_eb:.3f}")
    print(f"Composite Irrationality correlation: r = {correlation_comp:.3f}")
    
    return metrics

if __name__ == '__main__':
    print("Creating corrected irrationality analysis...")
    metrics = create_corrected_irrationality_plots()
    print("✅ Corrected irrationality analysis completed!")