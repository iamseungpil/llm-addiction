#!/usr/bin/env python3
"""
Image 5: Multi-round Patching Effect Timeline

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

    file_path = Path("/data/llm_addiction/experiment_5_multiround_patching/multiround_patching_final_20251012_021759.json")

    if not file_path.exists():
        print(f"⚠️  File not found: {file_path}")
        print("    Trying alternative file...")
        file_path = Path("/data/llm_addiction/experiment_5_multiround_patching/multiround_patching_final_20251005_205818.json")

    with open(file_path, 'r') as f:
        data = json.load(f)

    print(f"Loaded data with {len(data)} entries")

    return data

def extract_round_statistics(data):
    """Extract round-by-round statistics"""
    print("\nExtracting round statistics...")

    round_stats = defaultdict(lambda: {
        'bets': [],
        'balances': [],
        'bankruptcies': 0,
        'stops': 0
    })

    for entry in data:
        if 'history' not in entry:
            continue

        history = entry['history']
        outcome = entry.get('outcome', 'unknown')

        for round_data in history:
            round_num = round_data['round']
            bet = round_data['bet']
            balance = round_data['balance']

            round_stats[round_num]['bets'].append(bet)
            round_stats[round_num]['balances'].append(balance)

        # Count final outcome at the last round
        if history:
            final_round = history[-1]['round']
            if outcome == 'bankruptcy':
                round_stats[final_round]['bankruptcies'] += 1
            elif outcome == 'voluntary_stop':
                round_stats[final_round]['stops'] += 1

    print(f"Extracted statistics for {len(round_stats)} rounds")

    return round_stats

def create_visualization(round_stats):
    """Create multi-round patching visualization"""

    # Create figure with 4 subplots
    fig = plt.figure(figsize=(20, 12))

    rounds = sorted(round_stats.keys())[:50]  # First 50 rounds

    # Subplot 1: Mean bet amount over rounds
    ax1 = plt.subplot(2, 2, 1)

    mean_bets = [np.mean(round_stats[r]['bets']) if round_stats[r]['bets'] else 0 for r in rounds]
    std_bets = [np.std(round_stats[r]['bets']) if round_stats[r]['bets'] else 0 for r in rounds]

    ax1.plot(rounds, mean_bets, 'o-', linewidth=2, markersize=5, color='#e57373', label='Mean Bet')
    ax1.fill_between(rounds,
                     [m - s for m, s in zip(mean_bets, std_bets)],
                     [m + s for m, s in zip(mean_bets, std_bets)],
                     alpha=0.3, color='#e57373', label='±1 Std Dev')

    ax1.set_xlabel('Round Number', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Bet Amount ($)', fontsize=12, fontweight='bold')
    ax1.set_title('Mean Bet Amount Over Rounds', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Mean balance over rounds
    ax2 = plt.subplot(2, 2, 2)

    mean_balances = [np.mean(round_stats[r]['balances']) if round_stats[r]['balances'] else 0 for r in rounds]
    std_balances = [np.std(round_stats[r]['balances']) if round_stats[r]['balances'] else 0 for r in rounds]

    ax2.plot(rounds, mean_balances, 'o-', linewidth=2, markersize=5, color='#64b5f6', label='Mean Balance')
    ax2.fill_between(rounds,
                     [m - s for m, s in zip(mean_balances, std_balances)],
                     [m + s for m, s in zip(mean_balances, std_balances)],
                     alpha=0.3, color='#64b5f6', label='±1 Std Dev')

    ax2.axhline(y=100, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Initial Balance')
    ax2.set_xlabel('Round Number', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Balance ($)', fontsize=12, fontweight='bold')
    ax2.set_title('Mean Balance Over Rounds', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Cumulative outcomes
    ax3 = plt.subplot(2, 2, 3)

    cumulative_bankruptcies = []
    cumulative_stops = []

    total_bankruptcies = 0
    total_stops = 0

    for r in rounds:
        total_bankruptcies += round_stats[r]['bankruptcies']
        total_stops += round_stats[r]['stops']
        cumulative_bankruptcies.append(total_bankruptcies)
        cumulative_stops.append(total_stops)

    ax3.plot(rounds, cumulative_bankruptcies, 'o-', linewidth=2, markersize=4,
            color='#d32f2f', label='Cumulative Bankruptcies')
    ax3.plot(rounds, cumulative_stops, 'o-', linewidth=2, markersize=4,
            color='#7cb342', label='Cumulative Stops')

    ax3.set_xlabel('Round Number', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Cumulative Count', fontsize=12, fontweight='bold')
    ax3.set_title('Cumulative Outcomes Over Rounds', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Subplot 4: Active games per round
    ax4 = plt.subplot(2, 2, 4)

    active_games = [len(round_stats[r]['bets']) for r in rounds]

    ax4.plot(rounds, active_games, 'o-', linewidth=2, markersize=5, color='#9c27b0', label='Active Games')
    ax4.set_xlabel('Round Number', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Number of Active Games', fontsize=12, fontweight='bold')
    ax4.set_title('Active Games Per Round', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    # Overall title
    total_games = len([e for e in data if 'history' in e])
    fig.suptitle(f'Experiment 5: Multi-round Patching Effect Timeline\n' +
                f'Total Games: {total_games}',
                fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig

def main():
    print("="*80)
    print("Image 5: Multi-round Patching Effect Timeline")
    print("="*80)
    print()

    # Load data
    data = load_experiment5_data()

    if not data:
        print("❌ No data found!")
        return

    # Extract round statistics
    round_stats = extract_round_statistics(data)

    # Summary statistics
    total_games = len([e for e in data if 'history' in e])
    bankruptcies = sum(1 for e in data if e.get('outcome') == 'bankruptcy')
    stops = sum(1 for e in data if e.get('outcome') == 'voluntary_stop')

    print("\nSummary Statistics:")
    print(f"  Total games: {total_games}")
    print(f"  Bankruptcies: {bankruptcies} ({100*bankruptcies/total_games:.1f}%)")
    print(f"  Voluntary stops: {stops} ({100*stops/total_games:.1f}%)")
    print(f"  Max rounds played: {max(round_stats.keys())}")

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
