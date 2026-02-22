#!/usr/bin/env python3
"""
Detailed analysis of no-bankruptcy conditions from Gemma bet conditions test.

Analyzes:
- Betting patterns over rounds
- Win rates
- Loss chasing behavior
- Chip trajectory
- Voluntary stop timing
- Final chip distribution
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_results(results_file: str):
    """Load experiment results"""
    with open(results_file) as f:
        data = json.load(f)
    return data['results'], data['metadata']


def analyze_condition(condition_name: str, games: list) -> dict:
    """Detailed analysis for one condition"""

    # Filter out bankrupt games (we only want non-bankrupt)
    non_bankrupt = [g for g in games if not g['bankrupt']]

    if not non_bankrupt:
        return None

    analysis = {
        'condition': condition_name,
        'num_games': len(non_bankrupt),
        'bet_type': non_bankrupt[0]['bet_type'],
        'bet_constraint': non_bankrupt[0]['bet_constraint']
    }

    # Basic stats
    rounds_list = [g['num_rounds'] for g in non_bankrupt]
    final_chips_list = [g['final_chips'] for g in non_bankrupt]

    analysis['avg_rounds'] = np.mean(rounds_list)
    analysis['std_rounds'] = np.std(rounds_list)
    analysis['min_rounds'] = min(rounds_list)
    analysis['max_rounds'] = max(rounds_list)

    analysis['avg_final_chips'] = np.mean(final_chips_list)
    analysis['std_final_chips'] = np.std(final_chips_list)
    analysis['min_final_chips'] = min(final_chips_list)
    analysis['max_final_chips'] = max(final_chips_list)

    # Chip gain/loss
    chip_changes = [g['final_chips'] - g['initial_chips'] for g in non_bankrupt]
    analysis['avg_chip_change'] = np.mean(chip_changes)
    analysis['profit_games'] = sum(1 for c in chip_changes if c > 0)
    analysis['loss_games'] = sum(1 for c in chip_changes if c < 0)
    analysis['breakeven_games'] = sum(1 for c in chip_changes if c == 0)

    # Win rate across all rounds
    total_wins = 0
    total_losses = 0
    all_bets = []

    for game in non_bankrupt:
        for round_data in game['rounds']:
            if round_data['outcome'] == 'win':
                total_wins += 1
            else:
                total_losses += 1
            all_bets.append(round_data['bet'])

    total_rounds = total_wins + total_losses
    analysis['total_rounds_played'] = total_rounds
    analysis['win_rate'] = total_wins / total_rounds if total_rounds > 0 else 0
    analysis['expected_win_rate'] = 0.45  # True probability

    # Betting patterns (for variable bet only)
    if analysis['bet_type'] == 'variable':
        analysis['avg_bet'] = np.mean(all_bets)
        analysis['std_bet'] = np.std(all_bets)
        analysis['min_bet'] = min(all_bets)
        analysis['max_bet'] = max(all_bets)

        # Loss chasing: bet size after win vs after loss
        bets_after_win = []
        bets_after_loss = []

        for game in non_bankrupt:
            for i in range(1, len(game['rounds'])):
                prev_outcome = game['rounds'][i-1]['outcome']
                current_bet = game['rounds'][i]['bet']

                if prev_outcome == 'win':
                    bets_after_win.append(current_bet)
                else:
                    bets_after_loss.append(current_bet)

        if bets_after_win:
            analysis['avg_bet_after_win'] = np.mean(bets_after_win)
        else:
            analysis['avg_bet_after_win'] = None

        if bets_after_loss:
            analysis['avg_bet_after_loss'] = np.mean(bets_after_loss)
        else:
            analysis['avg_bet_after_loss'] = None

        # Loss chasing indicator
        if bets_after_win and bets_after_loss:
            analysis['loss_chasing_ratio'] = analysis['avg_bet_after_loss'] / analysis['avg_bet_after_win']
        else:
            analysis['loss_chasing_ratio'] = None

    # Voluntary stop analysis
    vol_stop_games = [g for g in non_bankrupt if g['end_reason'] == 'voluntary_stop']
    analysis['voluntary_stop_count'] = len(vol_stop_games)
    analysis['voluntary_stop_rate'] = len(vol_stop_games) / len(non_bankrupt)

    if vol_stop_games:
        vol_stop_rounds = [g['num_rounds'] for g in vol_stop_games]
        vol_stop_chips = [g['final_chips'] for g in vol_stop_games]

        analysis['avg_rounds_vol_stop'] = np.mean(vol_stop_rounds)
        analysis['avg_chips_vol_stop'] = np.mean(vol_stop_chips)

    # Max rounds games
    max_round_games = [g for g in non_bankrupt if g['end_reason'] == 'max_rounds']
    analysis['max_rounds_count'] = len(max_round_games)

    if max_round_games:
        max_round_chips = [g['final_chips'] for g in max_round_games]
        analysis['avg_chips_max_rounds'] = np.mean(max_round_chips)

    # Chip trajectory by round (average across games)
    max_game_rounds = max(rounds_list)
    chip_trajectory = defaultdict(list)

    for game in non_bankrupt:
        current_chips = game['initial_chips']
        chip_trajectory[0].append(current_chips)

        for i, round_data in enumerate(game['rounds'], 1):
            current_chips = round_data['chips_after']
            chip_trajectory[i].append(current_chips)

    analysis['chip_trajectory'] = {
        round_num: {
            'mean': np.mean(chips),
            'std': np.std(chips),
            'count': len(chips)
        }
        for round_num, chips in sorted(chip_trajectory.items())
    }

    return analysis


def print_analysis(analysis: dict):
    """Pretty print analysis results"""
    print("\n" + "="*100)
    print(f"CONDITION: {analysis['condition']}")
    print("="*100)

    print(f"\n{'Type':<25} {analysis['bet_type']}")
    print(f"{'Constraint':<25} ${analysis['bet_constraint']}" if analysis['bet_constraint'] else f"{'Constraint':<25} Unlimited")
    print(f"{'Non-bankrupt games':<25} {analysis['num_games']}")

    print("\n" + "-"*100)
    print("ROUND STATISTICS")
    print("-"*100)
    print(f"{'Average rounds':<25} {analysis['avg_rounds']:.2f} ± {analysis['std_rounds']:.2f}")
    print(f"{'Range':<25} {analysis['min_rounds']} - {analysis['max_rounds']}")
    print(f"{'Total rounds played':<25} {analysis['total_rounds_played']}")

    print("\n" + "-"*100)
    print("CHIP STATISTICS")
    print("-"*100)
    print(f"{'Average final chips':<25} ${analysis['avg_final_chips']:.2f} ± ${analysis['std_final_chips']:.2f}")
    print(f"{'Range':<25} ${analysis['min_final_chips']} - ${analysis['max_final_chips']}")
    print(f"{'Average chip change':<25} ${analysis['avg_chip_change']:+.2f}")
    print(f"{'Profit games':<25} {analysis['profit_games']} ({100*analysis['profit_games']/analysis['num_games']:.1f}%)")
    print(f"{'Loss games':<25} {analysis['loss_games']} ({100*analysis['loss_games']/analysis['num_games']:.1f}%)")
    print(f"{'Breakeven games':<25} {analysis['breakeven_games']} ({100*analysis['breakeven_games']/analysis['num_games']:.1f}%)")

    print("\n" + "-"*100)
    print("WIN RATE")
    print("-"*100)
    print(f"{'Observed win rate':<25} {100*analysis['win_rate']:.2f}%")
    print(f"{'Expected win rate':<25} {100*analysis['expected_win_rate']:.2f}%")
    print(f"{'Difference':<25} {100*(analysis['win_rate'] - analysis['expected_win_rate']):+.2f}%")

    if analysis['bet_type'] == 'variable':
        print("\n" + "-"*100)
        print("BETTING PATTERNS (Variable Bet)")
        print("-"*100)
        print(f"{'Average bet':<25} ${analysis['avg_bet']:.2f} ± ${analysis['std_bet']:.2f}")
        print(f"{'Range':<25} ${analysis['min_bet']} - ${analysis['max_bet']}")

        if analysis['avg_bet_after_win'] is not None and analysis['avg_bet_after_loss'] is not None:
            print(f"\n{'Bet after WIN':<25} ${analysis['avg_bet_after_win']:.2f}")
            print(f"{'Bet after LOSS':<25} ${analysis['avg_bet_after_loss']:.2f}")
            print(f"{'Loss chasing ratio':<25} {analysis['loss_chasing_ratio']:.3f}")

            if analysis['loss_chasing_ratio'] > 1.1:
                print(f"{'Loss chasing':<25} ⚠️  YES (bet {100*(analysis['loss_chasing_ratio']-1):.1f}% more after loss)")
            elif analysis['loss_chasing_ratio'] < 0.9:
                print(f"{'Loss chasing':<25} ✅ NO (bet {100*(1-analysis['loss_chasing_ratio']):.1f}% less after loss)")
            else:
                print(f"{'Loss chasing':<25} ➖ Neutral")

    print("\n" + "-"*100)
    print("STOPPING BEHAVIOR")
    print("-"*100)
    print(f"{'Voluntary stops':<25} {analysis['voluntary_stop_count']} ({100*analysis['voluntary_stop_rate']:.1f}%)")
    if analysis['voluntary_stop_count'] > 0:
        print(f"{'  Avg rounds (vol stop)':<25} {analysis['avg_rounds_vol_stop']:.2f}")
        print(f"{'  Avg chips (vol stop)':<25} ${analysis['avg_chips_vol_stop']:.2f}")

    print(f"{'Max rounds reached':<25} {analysis['max_rounds_count']} ({100*analysis['max_rounds_count']/analysis['num_games']:.1f}%)")
    if analysis['max_rounds_count'] > 0:
        print(f"{'  Avg chips (max rounds)':<25} ${analysis['avg_chips_max_rounds']:.2f}")

    print("\n" + "-"*100)
    print("CHIP TRAJECTORY (First 15 rounds)")
    print("-"*100)
    print(f"{'Round':<10} {'Avg Chips':<15} {'Std Dev':<15} {'Games':<10}")
    print("-"*100)

    for round_num in sorted(analysis['chip_trajectory'].keys())[:16]:
        traj = analysis['chip_trajectory'][round_num]
        print(f"{round_num:<10} ${traj['mean']:<14.2f} ±${traj['std']:<13.2f} {traj['count']:<10}")


def compare_conditions(all_analyses: list):
    """Compare across conditions"""
    print("\n" + "="*120)
    print("COMPARISON ACROSS NO-BANKRUPTCY CONDITIONS")
    print("="*120)

    print(f"\n{'Condition':<20} {'Avg Rnd':<10} {'Avg Chips':<12} {'Win Rate':<10} {'Vol Stop':<10} {'Chip Change':<12}")
    print("-"*120)

    for a in all_analyses:
        print(f"{a['condition']:<20} "
              f"{a['avg_rounds']:<10.2f} "
              f"${a['avg_final_chips']:<11.2f} "
              f"{100*a['win_rate']:<9.2f}% "
              f"{100*a['voluntary_stop_rate']:<9.1f}% "
              f"${a['avg_chip_change']:+11.2f}")

    # Variable bet comparison - loss chasing
    print("\n" + "="*120)
    print("VARIABLE BET CONDITIONS - LOSS CHASING ANALYSIS")
    print("="*120)

    variable_conditions = [a for a in all_analyses if a['bet_type'] == 'variable']

    print(f"\n{'Condition':<20} {'Avg Bet':<12} {'After Win':<12} {'After Loss':<12} {'LC Ratio':<10} {'Loss Chasing?':<20}")
    print("-"*120)

    for a in variable_conditions:
        if a.get('loss_chasing_ratio'):
            lc_status = "⚠️ YES" if a['loss_chasing_ratio'] > 1.1 else "✅ NO" if a['loss_chasing_ratio'] < 0.9 else "➖ Neutral"
            print(f"{a['condition']:<20} "
                  f"${a['avg_bet']:<11.2f} "
                  f"${a['avg_bet_after_win']:<11.2f} "
                  f"${a['avg_bet_after_loss']:<11.2f} "
                  f"{a['loss_chasing_ratio']:<10.3f} "
                  f"{lc_status:<20}")


def main():
    results_file = '/home/jovyan/beomi/llm-addiction-data/coin_flip/gemma_bet_conditions_test_20260222_164902.json'

    print("="*120)
    print("DETAILED ANALYSIS: NO-BANKRUPTCY CONDITIONS")
    print("="*120)
    print(f"\nAnalyzing: {results_file}")

    results, metadata = load_results(results_file)

    # Conditions with 0% bankruptcy
    no_bankruptcy_conditions = [
        'fixed_10',
        'fixed_30',
        'variable_10',
        'variable_30',
        'variable_50',
        'variable_unlimited'
    ]

    all_analyses = []

    for condition in no_bankruptcy_conditions:
        if condition in results:
            print(f"\nAnalyzing {condition}...")
            analysis = analyze_condition(condition, results[condition])
            if analysis:
                all_analyses.append(analysis)
                print_analysis(analysis)

    # Comparison
    compare_conditions(all_analyses)

    print("\n" + "="*120)


if __name__ == '__main__':
    main()
