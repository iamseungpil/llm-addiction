#!/usr/bin/env python3
"""
Analyze completed experiments from LLM addiction research
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def analyze_investment_choice(filepath):
    """Analyze investment choice experiment results"""
    print(f"\n{'='*80}")
    print(f"INVESTMENT CHOICE ANALYSIS: {Path(filepath).name}")
    print(f"{'='*80}\n")

    with open(filepath, 'r') as f:
        full_data = json.load(f)

    # Extract experiment info
    print(f"Model: {full_data.get('model', 'UNKNOWN')}")
    print(f"Timestamp: {full_data.get('timestamp', 'UNKNOWN')}")
    print(f"Constraint: ${full_data.get('config', {}).get('bet_constraint', 'UNKNOWN')}")

    data = full_data.get('results', [])
    total_games = len(data)
    print(f"Total games: {total_games}")

    # Group by conditions
    conditions = defaultdict(list)
    for game in data:
        condition = game.get('prompt_condition', 'UNKNOWN')
        bet_type = game.get('bet_type', 'UNKNOWN')
        key = f"{bet_type}/{condition}"
        conditions[key].append(game)

    print(f"\nConditions analyzed: {len(conditions)}")
    for cond, games in sorted(conditions.items()):
        print(f"  {cond}: {len(games)} games")

    # Overall statistics
    outcomes = [g.get('final_outcome', 'UNKNOWN') for g in data]
    outcome_counts = {}
    for outcome in set(outcomes):
        outcome_counts[outcome] = outcomes.count(outcome)

    print(f"\nOverall Outcomes:")
    for outcome, count in sorted(outcome_counts.items()):
        pct = 100 * count / total_games
        print(f"  {outcome}: {count} ({pct:.1f}%)")

    # Overall bankruptcy rate
    bankruptcies = sum(1 for g in data if g.get('bankruptcy', False))
    print(f"\nOverall bankruptcy rate: {bankruptcies}/{total_games} ({100*bankruptcies/total_games:.1f}%)")

    # Analyze by condition
    print(f"\n{'='*80}")
    print("CONDITION-WISE ANALYSIS")
    print(f"{'='*80}")

    for cond_name, games in sorted(conditions.items()):
        print(f"\n{cond_name} ({len(games)} games):")

        # Outcomes
        voluntary_stops = sum(1 for g in games if g.get('stopped_voluntarily', False))
        bankruptcies = sum(1 for g in games if g.get('bankruptcy', False))
        max_rounds = sum(1 for g in games if g.get('max_rounds_reached', False))

        print(f"  Voluntary stops: {voluntary_stops}/{len(games)} ({100*voluntary_stops/len(games):.1f}%)")
        print(f"  Bankruptcies: {bankruptcies}/{len(games)} ({100*bankruptcies/len(games):.1f}%)")
        if max_rounds > 0:
            print(f"  Max rounds reached: {max_rounds}/{len(games)} ({100*max_rounds/len(games):.1f}%)")

        # Average rounds and final balance
        rounds = [g.get('rounds_completed', 0) for g in games]
        final_balances = [g.get('final_balance', 0) for g in games]
        balance_changes = [g.get('balance_change', 0) for g in games]

        print(f"  Avg rounds: {np.mean(rounds):.1f} ± {np.std(rounds):.1f}")
        print(f"  Avg final balance: ${np.mean(final_balances):.1f} ± ${np.std(final_balances):.1f}")
        print(f"  Avg balance change: ${np.mean(balance_changes):.1f} ± ${np.std(balance_changes):.1f}")

        # Investment analysis
        total_invested = [g.get('total_invested', 0) for g in games]
        total_won = [g.get('total_won', 0) for g in games]
        print(f"  Avg total invested: ${np.mean(total_invested):.1f} ± ${np.std(total_invested):.1f}")
        print(f"  Avg total won: ${np.mean(total_won):.1f} ± ${np.std(total_won):.1f}")

        # Choice patterns (options 1-4)
        all_choices = defaultdict(int)
        for g in games:
            choice_counts = g.get('choice_counts', {})
            for choice, count in choice_counts.items():
                all_choices[choice] += count

        if all_choices:
            total_choices = sum(all_choices.values())
            print(f"  Choice distribution:")
            for choice in sorted(all_choices.keys()):
                count = all_choices[choice]
                pct = 100 * count / total_choices
                print(f"    Option {choice}: {count} ({pct:.1f}%)")

    return data

def analyze_blackjack(filepath):
    """Analyze blackjack experiment results"""
    print(f"\n{'='*80}")
    print(f"BLACKJACK ANALYSIS: {Path(filepath).name}")
    print(f"{'='*80}\n")

    with open(filepath, 'r') as f:
        full_data = json.load(f)

    # Extract experiment info
    if 'results' in full_data and len(full_data['results']) > 0:
        first_game = full_data['results'][0]
        print(f"Model: {first_game.get('model', 'UNKNOWN')}")

    data = full_data.get('results', [])
    total_games = len(data)
    print(f"Total games: {total_games}")
    print(f"Completed: {full_data.get('completed', 'UNKNOWN')}")
    print(f"Total expected: {full_data.get('total', 'UNKNOWN')}")

    # Group by conditions
    conditions = defaultdict(list)
    for game in data:
        components = game.get('components', 'UNKNOWN')
        bet_type = game.get('bet_type', 'UNKNOWN')
        key = f"{bet_type}/{components}"
        conditions[key].append(game)

    print(f"\nConditions analyzed: {len(conditions)}")
    for cond, games in sorted(conditions.items()):
        print(f"  {cond}: {len(games)} games")

    # Overall statistics
    outcomes = [g.get('outcome', 'UNKNOWN') for g in data]
    outcome_counts = {}
    for outcome in set(outcomes):
        outcome_counts[outcome] = outcomes.count(outcome)

    print(f"\nOverall Outcomes:")
    for outcome, count in sorted(outcome_counts.items()):
        pct = 100 * count / total_games
        print(f"  {outcome}: {count} ({pct:.1f}%)")

    # Overall bankruptcy rate
    bankruptcies = sum(1 for g in data if g.get('outcome', '') == 'bankrupt')
    print(f"\nOverall bankruptcy rate: {bankruptcies}/{total_games} ({100*bankruptcies/total_games:.1f}%)")

    # Analyze by condition
    print(f"\n{'='*80}")
    print("CONDITION-WISE ANALYSIS")
    print(f"{'='*80}")

    for cond_name, games in sorted(conditions.items()):
        print(f"\n{cond_name} ({len(games)} games):")

        # Outcomes
        cond_outcomes = [g.get('outcome', 'UNKNOWN') for g in games]
        voluntary_stops = cond_outcomes.count('voluntary_stop')
        bankruptcies = cond_outcomes.count('bankrupt')
        max_rounds = cond_outcomes.count('max_rounds')

        print(f"  Voluntary stops: {voluntary_stops}/{len(games)} ({100*voluntary_stops/len(games):.1f}%)")
        print(f"  Bankruptcies: {bankruptcies}/{len(games)} ({100*bankruptcies/len(games):.1f}%)")
        if max_rounds > 0:
            print(f"  Max rounds reached: {max_rounds}/{len(games)} ({100*max_rounds/len(games):.1f}%)")

        # Average rounds and final balance
        rounds = [g.get('total_rounds', 0) for g in games]
        initial_chips = [g.get('initial_chips', 0) for g in games]
        final_chips = [g.get('final_chips', 0) for g in games]
        balance_changes = [final - initial for final, initial in zip(final_chips, initial_chips)]

        print(f"  Avg rounds: {np.mean(rounds):.1f} ± {np.std(rounds):.1f}")
        print(f"  Avg initial chips: ${np.mean(initial_chips):.1f} ± ${np.std(initial_chips):.1f}")
        print(f"  Avg final chips: ${np.mean(final_chips):.1f} ± ${np.std(final_chips):.1f}")
        print(f"  Avg chip change: ${np.mean(balance_changes):.1f} ± ${np.std(balance_changes):.1f}")

        # Betting patterns (for variable bet games)
        if 'variable' in cond_name:
            all_bets = []
            for g in games:
                rounds_list = g.get('rounds', [])
                bets = [r.get('bet', 0) for r in rounds_list if 'bet' in r]
                all_bets.extend(bets)

            if all_bets:
                print(f"  Avg bet: ${np.mean(all_bets):.1f} ± ${np.std(all_bets):.1f}")
                print(f"  Bet range: ${min(all_bets)} - ${max(all_bets)}")

                # Betting aggressiveness - estimate chips before bet
                bet_chip_pairs = []
                for g in games:
                    rounds_list = g.get('rounds', [])
                    chips_before = g.get('initial_chips', 0)
                    for r in rounds_list:
                        bet = r.get('bet', 0)
                        if chips_before > 0:
                            bet_chip_pairs.append((bet, chips_before))
                        # Update chips for next round
                        payout = r.get('payout', 0)
                        chips_before = r.get('chips', chips_before + payout - bet)

                if bet_chip_pairs:
                    bet_ratios = [bet/chips for bet, chips in bet_chip_pairs]
                    print(f"  Avg bet/chips ratio: {np.mean(bet_ratios):.3f} ± {np.std(bet_ratios):.3f}")

    return data

def main():
    data_dir = Path("/scratch/x3415a02/data/llm-addiction")

    # Find completed experiments
    print("="*80)
    print("COMPLETED EXPERIMENT ANALYSIS")
    print("="*80)

    # Investment choice
    investment_files = list((data_dir / "investment_choice").glob("llama_investment_c10_*.json"))
    if investment_files:
        for f in investment_files:
            analyze_investment_choice(f)

    # Blackjack
    blackjack_files = list((data_dir / "blackjack").glob("llama_blackjack_checkpoint_400.json"))
    if blackjack_files:
        for f in blackjack_files:
            analyze_blackjack(f)

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
