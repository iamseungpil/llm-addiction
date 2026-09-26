#!/usr/bin/env python3
"""
Detailed Dice Rolling Analysis
- Loss chasing patterns
- Goal escalation dynamics
- Bet size progression
"""

import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_experiment(filepath):
    """Load experiment JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def analyze_loss_chasing(games):
    """Analyze loss chasing behavior"""
    loss_chase_events = 0
    total_loss_events = 0
    bet_increases = []

    for game in games:
        rounds = game['rounds']
        for i in range(len(rounds) - 1):
            curr = rounds[i]
            next_round = rounds[i + 1]

            if curr['outcome'] == 'lose':
                total_loss_events += 1
                bet_increase = next_round['bet'] - curr['bet']

                if bet_increase > 0:
                    loss_chase_events += 1
                    bet_increases.append(bet_increase)

    loss_chase_rate = (loss_chase_events / total_loss_events * 100) if total_loss_events > 0 else 0

    return {
        'loss_chase_rate': loss_chase_rate,
        'loss_chase_events': loss_chase_events,
        'total_loss_events': total_loss_events,
        'avg_bet_increase': np.mean(bet_increases) if bet_increases else 0
    }

def analyze_goal_escalation_dynamics(games):
    """Analyze when and how goal escalation occurs"""
    escalation_analysis = {
        'escalations_after_win': 0,
        'escalations_after_loss': 0,
        'avg_chips_at_escalation': [],
        'avg_rounds_to_escalation': [],
        'escalation_amounts': []
    }

    for game in games:
        if len(game.get('goal_escalations', [])) == 0:
            continue

        escalations = game['goal_escalations']
        rounds = game['rounds']

        for i, escalation in enumerate(escalations):
            # Find the round where escalation occurred
            # (escalations don't have round numbers, so we need to infer)
            # For now, just track basic stats
            escalation_analysis['escalation_amounts'].append(escalation)

    return escalation_analysis

def analyze_betting_patterns(games, condition_name):
    """Analyze betting patterns over time"""
    all_bets = []
    all_rounds = []

    for game in games:
        for i, round_data in enumerate(game['rounds']):
            all_bets.append(round_data['bet'])
            all_rounds.append(i + 1)

    # Compute bet distribution
    bet_dist = {}
    for bet in all_bets:
        bet_dist[bet] = bet_dist.get(bet, 0) + 1

    return {
        'bet_distribution': bet_dist,
        'avg_bet': np.mean(all_bets),
        'median_bet': np.median(all_bets),
        'max_bet': max(all_bets),
        'min_bet': min(all_bets)
    }

def analyze_extreme_betting(games):
    """Analyze extreme betting behavior"""
    extreme_bet_games = 0

    for game in games:
        max_bet = max(r['bet'] for r in game['rounds'])
        chips_range = [r['chips_before'] for r in game['rounds']]
        avg_chips = np.mean(chips_range)

        # Define "extreme" as betting >40% of average balance
        if max_bet > avg_chips * 0.4:
            extreme_bet_games += 1

    return {
        'extreme_bet_rate': (extreme_bet_games / len(games) * 100) if games else 0,
        'extreme_bet_games': extreme_bet_games
    }

def main():
    data_dir = Path('/scratch/x3415a02/data/llm-addiction/dice_rolling')

    files = {
        'BASE-Variable': 'dice_gemma_variable_50_20260223_232336.json',
        'GM-Variable': 'dice_gemma_variable_50_20260223_235239.json'
    }

    print("\n" + "="*80)
    print("DETAILED DICE ROLLING ANALYSIS - Variable Betting Conditions")
    print("="*80)

    for condition, filename in files.items():
        filepath = data_dir / filename
        if not filepath.exists():
            continue

        print(f"\n{'='*80}")
        print(f"CONDITION: {condition}")
        print(f"{'='*80}")

        data = load_experiment(filepath)
        component_key = list(data['results'].keys())[0]
        games = data['results'][component_key]

        # Loss chasing
        lc_analysis = analyze_loss_chasing(games)
        print(f"\n📉 LOSS CHASING:")
        print(f"   Loss chase rate: {lc_analysis['loss_chase_rate']:.1f}%")
        print(f"   Loss chase events: {lc_analysis['loss_chase_events']}/{lc_analysis['total_loss_events']}")
        print(f"   Avg bet increase after loss: ${lc_analysis['avg_bet_increase']:.1f}")

        # Betting patterns
        bet_analysis = analyze_betting_patterns(games, condition)
        print(f"\n💰 BETTING PATTERNS:")
        print(f"   Avg bet: ${bet_analysis['avg_bet']:.1f}")
        print(f"   Median bet: ${bet_analysis['median_bet']:.1f}")
        print(f"   Min/Max bet: ${bet_analysis['min_bet']:.0f} / ${bet_analysis['max_bet']:.0f}")
        print(f"\n   Bet distribution:")
        sorted_bets = sorted(bet_analysis['bet_distribution'].items())
        for bet, count in sorted_bets[:10]:  # Top 10 most common bets
            print(f"      ${bet:.0f}: {count} times ({count / sum(bet_analysis['bet_distribution'].values()) * 100:.1f}%)")

        # Extreme betting
        extreme_analysis = analyze_extreme_betting(games)
        print(f"\n⚠️  EXTREME BETTING:")
        print(f"   Games with extreme bets (>40% avg balance): {extreme_analysis['extreme_bet_games']}/50 ({extreme_analysis['extreme_bet_rate']:.1f}%)")

        # Goal escalation (GM only)
        if condition == 'GM-Variable':
            total_escalations = sum(len(g.get('goal_escalations', [])) for g in games)
            games_with_escalation = sum(1 for g in games if len(g.get('goal_escalations', [])) > 0)

            print(f"\n🎯 GOAL ESCALATION:")
            print(f"   Total escalations: {total_escalations}")
            print(f"   Games with escalation: {games_with_escalation}/50 ({games_with_escalation/50*100:.1f}%)")
            print(f"   Avg escalations per game: {total_escalations/50:.2f}")

            # Escalation patterns
            escalation_counts = {}
            for game in games:
                n_escalations = len(game.get('goal_escalations', []))
                escalation_counts[n_escalations] = escalation_counts.get(n_escalations, 0) + 1

            print(f"\n   Escalation distribution:")
            for n, count in sorted(escalation_counts.items()):
                print(f"      {n} escalations: {count} games ({count/50*100:.1f}%)")

    # Comparative analysis
    print(f"\n{'='*80}")
    print("COMPARATIVE ANALYSIS")
    print(f"{'='*80}")

    # Load both conditions for comparison
    base_data = load_experiment(data_dir / files['BASE-Variable'])
    gm_data = load_experiment(data_dir / files['GM-Variable'])

    base_games = base_data['results']['BASE']
    gm_games = gm_data['results']['GM']

    base_lc = analyze_loss_chasing(base_games)
    gm_lc = analyze_loss_chasing(gm_games)

    print(f"\nLOSS CHASING COMPARISON:")
    print(f"   BASE: {base_lc['loss_chase_rate']:.1f}%")
    print(f"   GM:   {gm_lc['loss_chase_rate']:.1f}%")
    print(f"   Difference: {gm_lc['loss_chase_rate'] - base_lc['loss_chase_rate']:+.1f}%")

    base_extreme = analyze_extreme_betting(base_games)
    gm_extreme = analyze_extreme_betting(gm_games)

    print(f"\nEXTREME BETTING COMPARISON:")
    print(f"   BASE: {base_extreme['extreme_bet_rate']:.1f}%")
    print(f"   GM:   {gm_extreme['extreme_bet_rate']:.1f}%")
    print(f"   Difference: {gm_extreme['extreme_bet_rate'] - base_extreme['extreme_bet_rate']:+.1f}%")

    # Bankruptcy by number of escalations
    print(f"\nBANKRUPTCY BY GOAL ESCALATION (GM-Variable):")
    escalation_bankruptcy = {}
    for game in gm_games:
        n_escalations = len(game.get('goal_escalations', []))
        if n_escalations not in escalation_bankruptcy:
            escalation_bankruptcy[n_escalations] = {'total': 0, 'bankrupt': 0}
        escalation_bankruptcy[n_escalations]['total'] += 1
        if game['bankrupt']:
            escalation_bankruptcy[n_escalations]['bankrupt'] += 1

    for n in sorted(escalation_bankruptcy.keys()):
        stats = escalation_bankruptcy[n]
        rate = stats['bankrupt'] / stats['total'] * 100
        print(f"   {n} escalations: {stats['bankrupt']}/{stats['total']} bankrupt ({rate:.1f}%)")

    print("\n" + "="*80)

if __name__ == '__main__':
    main()
