#!/usr/bin/env python3
"""
Image 5: Multi-round Patching Effect Timeline (FIXED)

Visualizes how feature patching effects evolve over multiple rounds
of the gambling game.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from collections import defaultdict

plt.style.use('seaborn-v0_8-darkgrid')

def load_experiment5_data():
    """Load Experiment 5 multi-round patching data"""
    print("Loading Experiment 5 data...")

    file_path = Path("/data/llm_addiction/experiment_5_multiround_patching/multiround_patching_final_20251005_205818.json")

    with open(file_path, 'r') as f:
        data = json.load(f)

    print(f"Timestamp: {data['timestamp']}")
    print(f"Total features: {data['total_features']}")
    print(f"Results count: {len(data['results'])}")

    return data

def extract_round_statistics(data):
    """Extract round-by-round statistics from all trials"""
    print("\nExtracting round statistics...")

    round_stats = defaultdict(lambda: {
        'safe_bets': [],
        'safe_balances': [],
        'risky_bets': [],
        'risky_balances': [],
        'safe_bankruptcies': 0,
        'risky_bankruptcies': 0
    })

    total_trials = 0

    for feature_result in data['results']:
        for condition_result in feature_result['results']:
            condition = condition_result['condition']

            # Determine if this is safe or risky patch
            is_safe = 'safe' in condition

            for trial in condition_result['trials']:
                total_trials += 1

                game_history = trial.get('game_history', [])

                for round_data in game_history:
                    round_num = round_data['round']
                    bet = round_data['bet']
                    balance = round_data['balance']

                    if is_safe:
                        round_stats[round_num]['safe_bets'].append(bet)
                        round_stats[round_num]['safe_balances'].append(balance)
                    else:
                        round_stats[round_num]['risky_bets'].append(bet)
                        round_stats[round_num]['risky_balances'].append(balance)

                # Count bankruptcies
                if trial.get('is_bankrupt', False):
                    if game_history:
                        final_round = game_history[-1]['round']
                        if is_safe:
                            round_stats[final_round]['safe_bankruptcies'] += 1
                        else:
                            round_stats[final_round]['risky_bankruptcies'] += 1

    print(f"Extracted statistics for {len(round_stats)} rounds from {total_trials} trials")

    return round_stats

def create_visualization(round_stats):
    """Create multi-round patching visualization"""

    # Create figure with 4 subplots
    fig = plt.figure(figsize=(20, 12))

    rounds = sorted(round_stats.keys())[:50]  # First 50 rounds

    # Subplot 1: Mean bet amount comparison
    ax1 = plt.subplot(2, 2, 1)

    safe_mean_bets = [np.mean(round_stats[r]['safe_bets']) if round_stats[r]['safe_bets'] else 0 for r in rounds]
    risky_mean_bets = [np.mean(round_stats[r]['risky_bets']) if round_stats[r]['risky_bets'] else 0 for r in rounds]

    ax1.plot(rounds, safe_mean_bets, 'o-', linewidth=2, markersize=4,
            color='#64b5f6', label='Safe Patch')
    ax1.plot(rounds, risky_mean_bets, 'o-', linewidth=2, markersize=4,
            color='#e57373', label='Risky Patch')

    ax1.set_xlabel('Round Number', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Bet Amount ($)', fontsize=12, fontweight='bold')
    ax1.set_title('Bet Amount: Safe vs Risky Patching', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Mean balance comparison
    ax2 = plt.subplot(2, 2, 2)

    safe_mean_balances = [np.mean(round_stats[r]['safe_balances']) if round_stats[r]['safe_balances'] else 0 for r in rounds]
    risky_mean_balances = [np.mean(round_stats[r]['risky_balances']) if round_stats[r]['risky_balances'] else 0 for r in rounds]

    ax2.plot(rounds, safe_mean_balances, 'o-', linewidth=2, markersize=4,
            color='#64b5f6', label='Safe Patch')
    ax2.plot(rounds, risky_mean_balances, 'o-', linewidth=2, markersize=4,
            color='#e57373', label='Risky Patch')

    ax2.axhline(y=100, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Initial Balance')
    ax2.set_xlabel('Round Number', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean Balance ($)', fontsize=12, fontweight='bold')
    ax2.set_title('Balance Evolution: Safe vs Risky Patching', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Cumulative bankruptcies
    ax3 = plt.subplot(2, 2, 3)

    cumulative_safe_bankruptcies = []
    cumulative_risky_bankruptcies = []

    total_safe = 0
    total_risky = 0

    for r in rounds:
        total_safe += round_stats[r]['safe_bankruptcies']
        total_risky += round_stats[r]['risky_bankruptcies']
        cumulative_safe_bankruptcies.append(total_safe)
        cumulative_risky_bankruptcies.append(total_risky)

    ax3.plot(rounds, cumulative_safe_bankruptcies, 'o-', linewidth=2, markersize=4,
            color='#64b5f6', label='Safe Patch')
    ax3.plot(rounds, cumulative_risky_bankruptcies, 'o-', linewidth=2, markersize=4,
            color='#e57373', label='Risky Patch')

    ax3.set_xlabel('Round Number', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Cumulative Bankruptcies', fontsize=12, fontweight='bold')
    ax3.set_title('Bankruptcy Accumulation: Safe vs Risky', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Subplot 4: Active trials per round
    ax4 = plt.subplot(2, 2, 4)

    safe_active = [len(round_stats[r]['safe_bets']) for r in rounds]
    risky_active = [len(round_stats[r]['risky_bets']) for r in rounds]

    ax4.plot(rounds, safe_active, 'o-', linewidth=2, markersize=4,
            color='#64b5f6', label='Safe Patch')
    ax4.plot(rounds, risky_active, 'o-', linewidth=2, markersize=4,
            color='#e57373', label='Risky Patch')

    ax4.set_xlabel('Round Number', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Active Trials', fontsize=12, fontweight='bold')
    ax4.set_title('Active Trials Per Round', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    # Overall title
    fig.suptitle(f'Experiment 5: Multi-round Patching Effect Timeline\n' +
                f'Safe vs Risky Feature Patching Comparison',
                fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig

def main():
    print("="*80)
    print("Image 5: Multi-round Patching Effect Timeline (FIXED)")
    print("="*80)
    print()

    # Load data
    data = load_experiment5_data()

    # Extract round statistics
    round_stats = extract_round_statistics(data)

    # Summary statistics
    rounds = sorted(round_stats.keys())

    print("\nSummary Statistics:")
    print(f"  Total rounds: {len(rounds)}")
    print(f"  Max round: {max(rounds)}")

    # Count total trials
    total_safe_trials = sum(len(round_stats[r]['safe_bets']) for r in rounds)
    total_risky_trials = sum(len(round_stats[r]['risky_bets']) for r in rounds)
    print(f"  Total safe patch trials: {total_safe_trials}")
    print(f"  Total risky patch trials: {total_risky_trials}")

    # Create visualization
    print("\nCreating visualization...")
    fig = create_visualization(round_stats)

    # Save
    output_dir = Path("/home/ubuntu/llm_addiction/experiment_pathway_token_analysis/analysis/images")

    png_path = output_dir / "05_multiround_patching_timeline.png"
    pdf_path = output_dir / "05_multiround_patching_timeline.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')

    print(f"\n✅ Saved visualization:")
    print(f"   PNG: {png_path}")
    print(f"   PDF: {pdf_path}")
    print()

if __name__ == '__main__':
    main()
