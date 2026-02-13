#!/usr/bin/env python3
"""
Detailed Investment Choice Analysis
- First round choices
- Loss chasing behavior
- Multi-round games analysis
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def analyze_first_round_choices(results):
    """Analyze first round choice distribution"""
    by_condition = defaultdict(lambda: defaultdict(int))

    for game in results:
        condition = game['bet_type']
        history = game.get('history', [])

        if len(history) > 0:
            first_choice = history[0].get('choice')
            by_condition[condition][first_choice] += 1

    return by_condition

def analyze_loss_chasing(results):
    """Analyze if model increases risk after losses"""
    loss_chase_stats = defaultdict(lambda: {
        'after_loss': [],
        'after_win': [],
        'after_loss_count': 0,
        'after_win_count': 0
    })

    for game in results:
        condition = game['bet_type']
        history = game.get('history', [])

        for i in range(len(history) - 1):
            current_round = history[i]
            next_round = history[i + 1]

            current_choice = current_round.get('choice')
            next_choice = next_round.get('choice')
            won = current_round.get('win', False)

            if current_choice == 1:  # Skip if stopped
                continue

            if won:
                loss_chase_stats[condition]['after_win'].append(next_choice)
                loss_chase_stats[condition]['after_win_count'] += 1
            else:
                loss_chase_stats[condition]['after_loss'].append(next_choice)
                loss_chase_stats[condition]['after_loss_count'] += 1

    return loss_chase_stats

def analyze_multi_round_games(results):
    """Analyze games that lasted 2+ rounds"""
    stats = defaultdict(lambda: {
        'single_round': 0,
        'multi_round': 0,
        'avg_rounds_multi': [],
        'multi_round_bankruptcies': 0,
        'multi_round_voluntary_stops': 0
    })

    for game in results:
        condition = game['bet_type']
        rounds = game.get('rounds_completed', 0)

        if rounds == 1:
            stats[condition]['single_round'] += 1
        else:
            stats[condition]['multi_round'] += 1
            stats[condition]['avg_rounds_multi'].append(rounds)

            if game.get('bankruptcy', False):
                stats[condition]['multi_round_bankruptcies'] += 1
            elif game.get('stopped_voluntarily', False):
                stats[condition]['multi_round_voluntary_stops'] += 1

    return stats

def main():
    data_dir = Path("/scratch/x3415a02/data/llm-addiction/investment_choice")
    llama_file = data_dir / "llama_investment_unlimited_20260213_110822.json"

    with open(llama_file, 'r') as f:
        data = json.load(f)

    results = data['results']

    print(f"\n{'='*70}")
    print(f"DETAILED INVESTMENT CHOICE ANALYSIS: LLaMA-3.1-8B")
    print(f"{'='*70}\n")

    # First round choices
    print(f"{'='*70}")
    print(f"First Round Choice Distribution")
    print(f"{'='*70}\n")

    first_round = analyze_first_round_choices(results)
    for condition in sorted(first_round.keys()):
        total = sum(first_round[condition].values())
        print(f"{condition.upper()} Betting:")
        for choice in [1, 2, 3, 4]:
            count = first_round[condition].get(choice, 0)
            pct = count / total * 100 if total > 0 else 0
            choice_name = {
                1: 'Safe/Stop',
                2: 'Moderate Risk',
                3: 'High Risk',
                4: 'Very High Risk'
            }.get(choice, choice)
            print(f"  Option {choice} ({choice_name}): {count}/{total} ({pct:.1f}%)")
        print()

    # Loss chasing
    print(f"{'='*70}")
    print(f"Loss Chasing Analysis")
    print(f"{'='*70}\n")

    loss_chase = analyze_loss_chasing(results)
    for condition in sorted(loss_chase.keys()):
        stats = loss_chase[condition]
        print(f"{condition.upper()} Betting:")

        if stats['after_loss_count'] > 0:
            avg_after_loss = np.mean(stats['after_loss'])
            print(f"  After LOSS (n={stats['after_loss_count']}):")
            print(f"    Avg next choice: {avg_after_loss:.2f}")

            # Distribution
            for choice in [1, 2, 3, 4]:
                count = stats['after_loss'].count(choice)
                pct = count / stats['after_loss_count'] * 100
                print(f"      Option {choice}: {count} ({pct:.1f}%)")

        if stats['after_win_count'] > 0:
            avg_after_win = np.mean(stats['after_win'])
            print(f"  After WIN (n={stats['after_win_count']}):")
            print(f"    Avg next choice: {avg_after_win:.2f}")

            # Distribution
            for choice in [1, 2, 3, 4]:
                count = stats['after_win'].count(choice)
                pct = count / stats['after_win_count'] * 100
                print(f"      Option {choice}: {count} ({pct:.1f}%)")

        # Compare
        if stats['after_loss_count'] > 0 and stats['after_win_count'] > 0:
            avg_after_loss = np.mean(stats['after_loss'])
            avg_after_win = np.mean(stats['after_win'])
            diff = avg_after_loss - avg_after_win
            print(f"  Loss Chasing Index: {diff:.3f} (positive = more risk after loss)")

        print()

    # Multi-round games
    print(f"{'='*70}")
    print(f"Single vs Multi-Round Games")
    print(f"{'='*70}\n")

    multi_round = analyze_multi_round_games(results)
    for condition in sorted(multi_round.keys()):
        stats = multi_round[condition]
        total = stats['single_round'] + stats['multi_round']

        print(f"{condition.upper()} Betting:")
        print(f"  Single round games: {stats['single_round']}/{total} ({stats['single_round']/total*100:.1f}%)")
        print(f"  Multi-round games: {stats['multi_round']}/{total} ({stats['multi_round']/total*100:.1f}%)")

        if stats['multi_round'] > 0:
            avg_rounds = np.mean(stats['avg_rounds_multi'])
            print(f"  Avg rounds (multi-round only): {avg_rounds:.1f}")
            print(f"  Multi-round bankruptcies: {stats['multi_round_bankruptcies']}/{stats['multi_round']} ({stats['multi_round_bankruptcies']/stats['multi_round']*100:.1f}%)")
            print(f"  Multi-round voluntary stops: {stats['multi_round_voluntary_stops']}/{stats['multi_round']} ({stats['multi_round_voluntary_stops']/stats['multi_round']*100:.1f}%)")

        print()

    print(f"{'='*70}")
    print("Analysis Complete")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
