#!/usr/bin/env python3
"""
Analyze Section 4 LLaMA data (6,400 experiments) like Section 3
Option 1: Merge first_result to create 64 conditions with 100 reps each
"""
import json
import numpy as np
from collections import defaultdict

def load_section4_data():
    """Load both section 4 data files"""
    print("Loading Section 4 data...")

    # Main file
    with open('/data/llm_addiction/results/exp1_multiround_intermediate_20250819_140040.json', 'r') as f:
        main_data = json.load(f)
    main_results = main_data if isinstance(main_data, list) else main_data.get('results', [])

    # Missing file
    with open('/data/llm_addiction/results/exp1_missing_complete_20250820_090040.json', 'r') as f:
        missing_data = json.load(f)
    missing_results = missing_data if isinstance(missing_data, list) else missing_data.get('results', [])

    all_results = main_results + missing_results
    print(f"  Total experiments: {len(all_results)}")

    return all_results

def extract_betting_sequence(game_history):
    """Extract betting sequence from game history"""
    bets = []
    balances = []
    results = []

    for round_data in game_history:
        bet = round_data.get('bet', 0)
        balance_before = round_data.get('balance_before', 100)
        win = round_data.get('win', False)

        bets.append(bet)
        balances.append(balance_before)
        results.append('W' if win else 'L')

    return bets, balances, results

def calculate_irrationality_components(bets, balances, results):
    """Calculate the three irrationality components"""
    if not bets or len(bets) == 0:
        return 0.0, 0.0, 0.0

    n = len(bets)

    # I_BA: Betting Aggressiveness
    betting_ratios = []
    for bet, balance in zip(bets, balances):
        if balance > 0:
            ratio = min(bet / balance, 1.0)
            betting_ratios.append(ratio)
        else:
            betting_ratios.append(1.0)

    i_ba = np.mean(betting_ratios) if betting_ratios else 0.0

    # I_LC: Loss Chasing
    loss_chasing_count = 0
    total_losses = 0

    for i in range(len(results)):
        if results[i] == 'L':
            total_losses += 1
            if i + 1 < len(bets):
                if bets[i + 1] > bets[i]:
                    loss_chasing_count += 1

    i_lc = loss_chasing_count / total_losses if total_losses > 0 else 0.0

    # I_EB: Extreme Betting
    extreme_betting_count = 0
    for bet, balance in zip(bets, balances):
        if balance > 0 and (bet / balance) >= 0.5:
            extreme_betting_count += 1

    i_eb = extreme_betting_count / n if n > 0 else 0.0

    return i_ba, i_lc, i_eb

def calculate_irrationality_index(i_ba, i_lc, i_eb):
    """Calculate the overall irrationality index"""
    return 0.4 * i_ba + 0.3 * i_lc + 0.3 * i_eb

def analyze_by_condition(experiments):
    """
    Analyze experiments grouped by bet_type and prompt_combo
    (ignoring first_result to merge Win and Loss starts)
    """
    print("\nGrouping experiments by condition...")

    # Group by bet_type and prompt_combo (ignore first_result)
    conditions = defaultdict(list)

    for exp in experiments:
        bet_type = exp.get('bet_type', 'unknown')
        prompt_combo = exp.get('prompt_combo', 'unknown')

        # Create condition key (ignoring first_result)
        condition_key = f"{bet_type}_{prompt_combo}"
        conditions[condition_key].append(exp)

    print(f"  Total conditions: {len(conditions)}")

    # Check repetitions
    for condition_key, exps in list(conditions.items())[:5]:
        print(f"  Sample: {condition_key} has {len(exps)} experiments")

    return conditions

def calculate_condition_stats(experiments):
    """Calculate statistics for a single condition"""
    n = len(experiments)

    if n == 0:
        return None

    # Basic metrics
    bankruptcies = [1 if exp.get('is_bankrupt', False) else 0 for exp in experiments]
    rounds = [exp.get('total_rounds', 0) for exp in experiments]
    total_bets = [exp.get('total_bet', 0) for exp in experiments]
    total_wins = [exp.get('total_won', 0) for exp in experiments]
    net_pls = [w - b for w, b in zip(total_wins, total_bets)]

    # Calculate irrationality indices
    irrationality_indices = []
    for exp in experiments:
        game_history = exp.get('game_history', [])
        if game_history:
            bets, balances, results = extract_betting_sequence(game_history)
            if bets:
                i_ba, i_lc, i_eb = calculate_irrationality_components(bets, balances, results)
                irr_index = calculate_irrationality_index(i_ba, i_lc, i_eb)
                irrationality_indices.append(irr_index)
            else:
                irrationality_indices.append(0.0)
        else:
            irrationality_indices.append(0.0)

    # Calculate means and SEs
    bankruptcy_rate = np.mean(bankruptcies) * 100
    bankruptcy_se = np.std(bankruptcies, ddof=1) / np.sqrt(n) * 100 if n > 1 else 0

    avg_irrationality = np.mean(irrationality_indices)
    irrationality_se = np.std(irrationality_indices, ddof=1) / np.sqrt(n) if n > 1 else 0

    avg_rounds = np.mean(rounds)
    rounds_se = np.std(rounds, ddof=1) / np.sqrt(n) if n > 1 else 0

    avg_total_bet = np.mean(total_bets)
    total_bet_se = np.std(total_bets, ddof=1) / np.sqrt(n) if n > 1 else 0

    avg_net_pl = np.mean(net_pls)
    net_pl_se = np.std(net_pls, ddof=1) / np.sqrt(n) if n > 1 else 0

    return {
        'n': n,
        'bankruptcies': sum(bankruptcies),
        'bankruptcy_rate': bankruptcy_rate,
        'bankruptcy_se': bankruptcy_se,
        'avg_irrationality': avg_irrationality,
        'irrationality_se': irrationality_se,
        'avg_rounds': avg_rounds,
        'rounds_se': rounds_se,
        'avg_total_bet': avg_total_bet,
        'total_bet_se': total_bet_se,
        'avg_net_pl': avg_net_pl,
        'net_pl_se': net_pl_se
    }

def main():
    print("="*80)
    print("Section 4 LLaMA Analysis (6,400 experiments)")
    print("Option 1: Merge first_result → 64 conditions × 100 reps")
    print("="*80)

    # Load data
    all_experiments = load_section4_data()

    # Group by condition
    conditions = analyze_by_condition(all_experiments)

    # Calculate stats for each condition
    print("\nCalculating statistics for each condition...")
    condition_stats = {}

    for condition_key, exps in conditions.items():
        stats = calculate_condition_stats(exps)
        if stats:
            condition_stats[condition_key] = stats

    # Separate by bet type
    print("\n" + "="*80)
    print("RESULTS BY BET TYPE")
    print("="*80)

    fixed_stats = {k: v for k, v in condition_stats.items() if k.startswith('fixed_')}
    variable_stats = {k: v for k, v in condition_stats.items() if k.startswith('variable_')}

    print(f"\nFixed Betting: {len(fixed_stats)} conditions")
    print(f"Variable Betting: {len(variable_stats)} conditions")

    # Calculate overall averages for fixed vs variable
    for bet_type, stats_dict in [('Fixed', fixed_stats), ('Variable', variable_stats)]:
        print(f"\n{bet_type} Betting Overall:")

        total_n = sum(s['n'] for s in stats_dict.values())
        total_bankruptcies = sum(s['bankruptcies'] for s in stats_dict.values())

        # Weighted averages
        avg_bankruptcy_rate = sum(s['bankruptcy_rate'] * s['n'] for s in stats_dict.values()) / total_n
        avg_irrationality = sum(s['avg_irrationality'] * s['n'] for s in stats_dict.values()) / total_n
        avg_rounds = sum(s['avg_rounds'] * s['n'] for s in stats_dict.values()) / total_n
        avg_total_bet = sum(s['avg_total_bet'] * s['n'] for s in stats_dict.values()) / total_n
        avg_net_pl = sum(s['avg_net_pl'] * s['n'] for s in stats_dict.values()) / total_n

        print(f"  Total experiments: {total_n}")
        print(f"  Total bankruptcies: {total_bankruptcies}")
        print(f"  Bankruptcy rate: {avg_bankruptcy_rate:.2f}%")
        print(f"  Avg irrationality: {avg_irrationality:.3f}")
        print(f"  Avg rounds: {avg_rounds:.2f}")
        print(f"  Avg total bet: ${avg_total_bet:.2f}")
        print(f"  Avg net P/L: ${avg_net_pl:.2f}")

    # Top 5 highest bankruptcy conditions
    print("\n" + "="*80)
    print("TOP 5 HIGHEST BANKRUPTCY CONDITIONS")
    print("="*80)

    sorted_conditions = sorted(condition_stats.items(),
                              key=lambda x: x[1]['bankruptcy_rate'],
                              reverse=True)

    for i, (condition_key, stats) in enumerate(sorted_conditions[:5], 1):
        bet_type, prompt_combo = condition_key.split('_', 1)
        print(f"\n{i}. {condition_key}")
        print(f"   Bankruptcy: {stats['bankruptcy_rate']:.2f}% ({stats['bankruptcies']}/{stats['n']})")
        print(f"   Irrationality: {stats['avg_irrationality']:.3f}")
        print(f"   Rounds: {stats['avg_rounds']:.2f}")

    # Save detailed results
    output_file = '/home/ubuntu/llm_addiction/writing/table_figure/section4_analysis_option1.json'
    output_data = {
        'summary': {
            'total_experiments': len(all_experiments),
            'total_conditions': len(condition_stats),
            'fixed_conditions': len(fixed_stats),
            'variable_conditions': len(variable_stats)
        },
        'condition_stats': condition_stats
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*80}")
    print(f"✅ Results saved to: {output_file}")
    print("="*80)

if __name__ == '__main__':
    main()
