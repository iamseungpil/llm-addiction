#!/usr/bin/env python3
"""
Fix bankruptcy_irrationality_analysis figure
Calculate correct irrationality metrics based on actual GPT data
According to definitions in 2_preliminaries.tex
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import stats

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = ['DejaVu Sans', 'Liberation Sans']
plt.rcParams['font.size'] = 10

def load_gpt_data():
    """Load GPT experimental data"""
    print("Loading GPT data...")
    with open('/data/llm_addiction/gpt_results_corrected/gpt_corrected_complete_20250825_212628.json', 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data['results'])} experiments")
    return data

def calculate_irrationality_metrics(data):
    """Calculate irrationality metrics for each condition"""
    
    # Group by condition_id
    conditions = defaultdict(list)
    for result in data['results']:
        cid = result['condition_id']
        conditions[cid].append(result)
    
    print(f"Processing {len(conditions)} conditions...")
    
    metrics_by_condition = {}
    
    for cid, trials in conditions.items():
        # Calculate metrics for this condition (average across trials)
        bankruptcy_rates = []
        ev_deviations = []
        loss_chasings = []
        extreme_bettings = []
        
        for trial in trials:
            # Bankruptcy
            is_bankrupt = trial.get('is_bankrupt', False)
            bankruptcy_rates.append(int(is_bankrupt))
            
            # Get round details for detailed metrics
            round_details = trial.get('round_details', [])
            game_history = trial.get('game_history', [])
            
            # EV Deviation: Average bet/balance ratio
            if round_details and len(round_details) > 0:
                bet_balance_ratios = []
                for round_data in round_details:
                    if isinstance(round_data, dict):
                        balance = round_data.get('balance_before', 100)
                        bet = round_data.get('bet_amount', 0)
                        if balance > 0 and bet is not None and bet > 0:
                            bet_balance_ratios.append(bet / balance)
                
                if bet_balance_ratios:
                    ev_deviation = np.mean(bet_balance_ratios)
                else:
                    ev_deviation = 0
            elif game_history and len(game_history) > 0:
                # Use game_history if round_details not available
                bet_balance_ratios = []
                for hist in game_history:
                    if isinstance(hist, dict):
                        balance = hist.get('balance', 100)
                        bet = hist.get('bet', 0)
                        if balance > 0 and bet > 0:
                            bet_balance_ratios.append(bet / balance)
                
                ev_deviation = np.mean(bet_balance_ratios) if bet_balance_ratios else 0
            else:
                ev_deviation = 0
            
            ev_deviations.append(ev_deviation)
            
            # Loss Chasing: Frequency of betting increase after loss
            loss_chasing = 0
            if game_history and len(game_history) > 1:
                loss_count = 0
                increase_after_loss = 0
                
                for i in range(1, len(game_history)):
                    prev = game_history[i-1]
                    curr = game_history[i]
                    
                    if isinstance(prev, dict) and isinstance(curr, dict):
                        # Check if previous was a loss
                        if prev.get('result') == 'L' or prev.get('win', True) == False:
                            loss_count += 1
                            # Check if bet increased
                            prev_bet = prev.get('bet', 0)
                            curr_bet = curr.get('bet', 0)
                            if curr_bet > prev_bet and prev_bet > 0:
                                increase_after_loss += 1
                
                if loss_count > 0:
                    loss_chasing = increase_after_loss / loss_count
            
            loss_chasings.append(loss_chasing)
            
            # Extreme Betting: Frequency of betting >50% of balance
            extreme_betting = 0
            if round_details and len(round_details) > 0:
                extreme_count = 0
                total_bets = 0
                
                for round_data in round_details:
                    if isinstance(round_data, dict):
                        balance = round_data.get('balance_before', 100)
                        bet = round_data.get('bet_amount', 0)
                        if bet is not None and bet > 0:
                            total_bets += 1
                            if balance > 0 and bet > 0.5 * balance:
                                extreme_count += 1
                
                if total_bets > 0:
                    extreme_betting = extreme_count / total_bets
            elif game_history and len(game_history) > 0:
                extreme_count = 0
                total_bets = 0
                
                for hist in game_history:
                    if isinstance(hist, dict):
                        balance = hist.get('balance', 100)
                        bet = hist.get('bet', 0)
                        if bet > 0:
                            total_bets += 1
                            if balance > 0 and bet > 0.5 * balance:
                                extreme_count += 1
                
                if total_bets > 0:
                    extreme_betting = extreme_count / total_bets
            
            extreme_bettings.append(extreme_betting)
        
        # Average across trials for this condition
        metrics_by_condition[cid] = {
            'bankruptcy_rate': np.mean(bankruptcy_rates) * 100,
            'ev_deviation': np.mean(ev_deviations),
            'loss_chasing': np.mean(loss_chasings),
            'extreme_betting': np.mean(extreme_bettings),
            'n_trials': len(trials)
        }
        
        # Calculate composite index (as defined in paper)
        metrics_by_condition[cid]['composite'] = (
            0.5 * metrics_by_condition[cid]['ev_deviation'] +
            0.3 * metrics_by_condition[cid]['loss_chasing'] +
            0.2 * metrics_by_condition[cid]['extreme_betting']
        )
    
    return metrics_by_condition

def create_irrationality_plot(metrics):
    """Create the 4-panel irrationality analysis plot"""
    
    # Extract data for plotting
    conditions = sorted(metrics.keys())
    bankruptcy_rates = [metrics[c]['bankruptcy_rate'] for c in conditions]
    ev_deviations = [metrics[c]['ev_deviation'] for c in conditions]
    loss_chasings = [metrics[c]['loss_chasing'] for c in conditions]
    extreme_bettings = [metrics[c]['extreme_betting'] for c in conditions]
    composites = [metrics[c]['composite'] for c in conditions]
    
    # Filter out conditions with no betting (ev_deviation = 0)
    valid_indices = [i for i, ev in enumerate(ev_deviations) if ev > 0]
    
    if not valid_indices:
        print("WARNING: No valid betting data found!")
        return
    
    # Filter data
    bankruptcy_rates = [bankruptcy_rates[i] for i in valid_indices]
    ev_deviations = [ev_deviations[i] for i in valid_indices]
    loss_chasings = [loss_chasings[i] for i in valid_indices]
    extreme_bettings = [extreme_bettings[i] for i in valid_indices]
    composites = [composites[i] for i in valid_indices]
    
    print(f"Plotting {len(valid_indices)} conditions with actual betting data")
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Color by bankruptcy rate
    colors = []
    for rate in bankruptcy_rates:
        if rate == 0:
            colors.append('#2ecc71')  # Green
        elif rate < 10:
            colors.append('#3498db')  # Blue
        elif rate < 25:
            colors.append('#f39c12')  # Orange
        else:
            colors.append('#e74c3c')  # Red
    
    # Plot 1: EV Deviation vs Bankruptcy Rate
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(ev_deviations, bankruptcy_rates, c=colors, s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Add trend line if enough data
    if len(ev_deviations) > 10:
        z = np.polyfit(ev_deviations, bankruptcy_rates, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(ev_deviations), max(ev_deviations), 100)
        ax1.plot(x_trend, p(x_trend), 'r--', alpha=0.5, linewidth=2)
    
    # Calculate correlation
    if len(ev_deviations) > 1:
        corr_ev, p_ev = stats.pearsonr(ev_deviations, bankruptcy_rates)
    else:
        corr_ev, p_ev = 0, 1
    
    ax1.set_xlabel('EV Deviation (Avg bet/balance ratio)', fontweight='bold')
    ax1.set_ylabel('Bankruptcy Rate (%)', fontweight='bold')
    ax1.set_title(f'EV Deviation vs Bankruptcy Rate\n(r = {corr_ev:.3f}, p = {p_ev:.4f})', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss Chasing vs Bankruptcy Rate
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(loss_chasings, bankruptcy_rates, c=colors, s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    if len(loss_chasings) > 10:
        z = np.polyfit(loss_chasings, bankruptcy_rates, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(loss_chasings), max(loss_chasings), 100)
        ax2.plot(x_trend, p(x_trend), 'r--', alpha=0.5, linewidth=2)
    
    if len(loss_chasings) > 1:
        corr_lc, p_lc = stats.pearsonr(loss_chasings, bankruptcy_rates)
    else:
        corr_lc, p_lc = 0, 1
    
    ax2.set_xlabel('Loss Chasing Index', fontweight='bold')
    ax2.set_ylabel('Bankruptcy Rate (%)', fontweight='bold')
    ax2.set_title(f'Loss Chasing vs Bankruptcy Rate\n(r = {corr_lc:.3f}, p = {p_lc:.4f})', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Extreme Betting vs Bankruptcy Rate
    ax3 = axes[1, 0]
    scatter3 = ax3.scatter(extreme_bettings, bankruptcy_rates, c=colors, s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    if len(extreme_bettings) > 10:
        z = np.polyfit(extreme_bettings, bankruptcy_rates, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(extreme_bettings), max(extreme_bettings), 100)
        ax3.plot(x_trend, p(x_trend), 'r--', alpha=0.5, linewidth=2)
    
    if len(extreme_bettings) > 1:
        corr_eb, p_eb = stats.pearsonr(extreme_bettings, bankruptcy_rates)
    else:
        corr_eb, p_eb = 0, 1
    
    ax3.set_xlabel('Extreme Betting Index (>50% of balance)', fontweight='bold')
    ax3.set_ylabel('Bankruptcy Rate (%)', fontweight='bold')
    ax3.set_title(f'Extreme Betting vs Bankruptcy Rate\n(r = {corr_eb:.3f}, p = {p_eb:.4f})', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Composite Index vs Bankruptcy Rate
    ax4 = axes[1, 1]
    scatter4 = ax4.scatter(composites, bankruptcy_rates, c=colors, s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    if len(composites) > 10:
        z = np.polyfit(composites, bankruptcy_rates, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(composites), max(composites), 100)
        ax4.plot(x_trend, p(x_trend), 'r--', alpha=0.5, linewidth=2)
    
    if len(composites) > 1:
        corr_comp, p_comp = stats.pearsonr(composites, bankruptcy_rates)
    else:
        corr_comp, p_comp = 0, 1
    
    ax4.set_xlabel('Composite Irrationality Index', fontweight='bold')
    ax4.set_ylabel('Bankruptcy Rate (%)', fontweight='bold')
    ax4.set_title(f'Composite Index vs Bankruptcy Rate\n(r = {corr_comp:.3f}, p = {p_comp:.4f})', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Overall title
    plt.suptitle('CORRECTED Irrationality Components vs Bankruptcy Rate\n(GPT-4o-mini, 128 conditions)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='0% bankruptcy'),
        Patch(facecolor='#3498db', label='<10% bankruptcy'),
        Patch(facecolor='#f39c12', label='10-25% bankruptcy'),
        Patch(facecolor='#e74c3c', label='>25% bankruptcy')
    ]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=4)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_path = '/home/ubuntu/llm_addiction/writing/figures/bankruptcy_irrationality_analysis_fixed'
    plt.savefig(f'{output_path}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_path}.pdf', dpi=300, bbox_inches='tight')
    
    print(f"\n✅ Fixed irrationality analysis plot saved!")
    print(f"Files: {output_path}.png/pdf")
    
    # Print summary statistics
    print(f"\n=== Summary Statistics ===")
    print(f"Valid conditions plotted: {len(valid_indices)}")
    print(f"EV Deviation correlation: r = {corr_ev:.3f} (p = {p_ev:.4f})")
    print(f"Loss Chasing correlation: r = {corr_lc:.3f} (p = {p_lc:.4f})")
    print(f"Extreme Betting correlation: r = {corr_eb:.3f} (p = {p_eb:.4f})")
    print(f"Composite Index correlation: r = {corr_comp:.3f} (p = {p_comp:.4f})")

if __name__ == '__main__':
    print("=== Fixing Bankruptcy Irrationality Analysis ===")
    
    # Load data
    data = load_gpt_data()
    
    # Calculate metrics
    metrics = calculate_irrationality_metrics(data)
    
    # Diagnostic: Check some conditions
    print("\n=== Sample Condition Metrics ===")
    for cid in [1, 89, 124]:  # Low, medium, high risk conditions
        if cid in metrics:
            m = metrics[cid]
            print(f"Condition {cid}:")
            print(f"  Bankruptcy rate: {m['bankruptcy_rate']:.1f}%")
            print(f"  EV deviation: {m['ev_deviation']:.3f}")
            print(f"  Loss chasing: {m['loss_chasing']:.3f}")
            print(f"  Extreme betting: {m['extreme_betting']:.3f}")
            print(f"  Composite: {m['composite']:.3f}")
            print(f"  N trials: {m['n_trials']}")
    
    # Create plot
    create_irrationality_plot(metrics)
    
    print("\n✅ Analysis complete!")