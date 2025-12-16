#!/usr/bin/env python3
"""
Investment Choice Data Analysis for Table 3
NO HALLUCINATION - All statistics computed from actual experiment data
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

# Data paths
RESULTS_DIR = Path('/data/llm_addiction/investment_choice_experiment/results')
OUTPUT_FILE = Path('/home/ubuntu/llm_addiction/rebuttal_analysis/investment_choice_stats.json')

def load_all_results() -> Dict:
    """Load all 8 result files (4 models × 2 bet types)"""
    results = {}

    files = sorted(RESULTS_DIR.glob('*.json'))
    print(f"Found {len(files)} result files:")

    for file_path in files:
        print(f"  Loading: {file_path.name}")
        with open(file_path) as f:
            data = json.load(f)

        config = data.get('experiment_config', {})
        model = config.get('model', 'unknown')
        bet_type = config.get('bet_type', 'unknown')
        games = data.get('results', [])

        key = (model, bet_type)
        results[key] = games
        print(f"    Model: {model}, Bet type: {bet_type}, Games: {len(games)}")

    return results

def calculate_bankruptcy_rate(games: List[Dict]) -> float:
    """Calculate bankruptcy rate (%)"""
    bankruptcies = sum(1 for g in games if g.get('exit_reason') == 'bankruptcy')
    return (bankruptcies / len(games)) * 100 if games else 0.0

def calculate_avg_rounds(games: List[Dict]) -> float:
    """Calculate average number of rounds per game"""
    rounds = [g.get('rounds_played', 0) for g in games]
    return np.mean(rounds) if rounds else 0.0

def calculate_total_bet(games: List[Dict]) -> float:
    """Calculate average total bet amount per game"""
    total_bets = []
    for game in games:
        game_total = sum(decision.get('bet', 0)
                        for decision in game.get('decisions', []))
        total_bets.append(game_total)
    return np.mean(total_bets) if total_bets else 0.0

def calculate_net_pl(games: List[Dict]) -> float:
    """Calculate average net profit/loss per game"""
    net_pls = []
    for game in games:
        initial = 100  # Initial balance
        final = game.get('final_balance', 100)
        net_pls.append(final - initial)
    return np.mean(net_pls) if net_pls else 0.0

def calculate_option4_rate(games: List[Dict]) -> float:
    """Calculate Option 4 selection rate as irrationality proxy"""
    option4_count = 0
    total_choices = 0

    for game in games:
        for decision in game.get('decisions', []):
            choice = decision.get('choice')
            if choice is not None:
                total_choices += 1
                if choice == 4:
                    option4_count += 1

    return (option4_count / total_choices * 100) if total_choices > 0 else 0.0

def calculate_sem(values: List[float]) -> float:
    """Calculate standard error of mean"""
    return np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0

def analyze_model_condition(games: List[Dict]) -> Dict:
    """Analyze single model-bettype condition"""
    n_games = len(games)

    # Calculate metrics
    bankruptcy_rate = calculate_bankruptcy_rate(games)
    avg_rounds = calculate_avg_rounds(games)
    total_bet = calculate_total_bet(games)
    net_pl = calculate_net_pl(games)
    option4_rate = calculate_option4_rate(games)

    # Calculate SEMs for each metric
    bankruptcy_values = [1.0 if g.get('exit_reason') == 'bankruptcy' else 0.0 for g in games]
    rounds_values = [g.get('rounds_played', 0) for g in games]
    bet_values = [sum(d.get('bet', 0) for d in g.get('decisions', [])) for g in games]
    pl_values = [g.get('final_balance', 100) - 100 for g in games]

    return {
        'n_games': n_games,
        'bankruptcy_rate': bankruptcy_rate,
        'bankruptcy_sem': calculate_sem(bankruptcy_values) * 100,  # Convert to percentage
        'avg_rounds': avg_rounds,
        'rounds_sem': calculate_sem(rounds_values),
        'total_bet': total_bet,
        'bet_sem': calculate_sem(bet_values),
        'net_pl': net_pl,
        'pl_sem': calculate_sem(pl_values),
        'option4_rate': option4_rate,  # Irrationality proxy
    }

def main():
    print("="*80)
    print("Investment Choice Data Analysis")
    print("="*80)
    print()

    # Load all results
    all_results = load_all_results()
    print(f"\nTotal conditions loaded: {len(all_results)}")
    print()

    # Analyze each condition
    stats = {}

    print("="*80)
    print("Calculating Statistics")
    print("="*80)

    for (model, bet_type), games in sorted(all_results.items()):
        print(f"\n{model} - {bet_type.upper()}")
        print("-" * 60)

        condition_stats = analyze_model_condition(games)
        stats[f"{model}_{bet_type}"] = condition_stats

        print(f"  N games: {condition_stats['n_games']}")
        print(f"  Bankruptcy rate: {condition_stats['bankruptcy_rate']:.2f}% ± {condition_stats['bankruptcy_sem']:.2f}")
        print(f"  Avg rounds: {condition_stats['avg_rounds']:.2f} ± {condition_stats['rounds_sem']:.2f}")
        print(f"  Total bet: ${condition_stats['total_bet']:.2f} ± ${condition_stats['bet_sem']:.2f}")
        print(f"  Net P/L: ${condition_stats['net_pl']:.2f} ± ${condition_stats['pl_sem']:.2f}")
        print(f"  Option 4 rate: {condition_stats['option4_rate']:.2f}%")

    # Save results
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(stats, f, indent=2)

    print("\n" + "="*80)
    print(f"✅ Statistics saved to: {OUTPUT_FILE}")
    print("="*80)

    # Summary table
    print("\nSUMMARY TABLE (for verification)")
    print("="*80)
    print(f"{'Model':<20} {'Bet Type':<10} {'Bankrupt%':<12} {'Rounds':<10} {'Total Bet':<12} {'Net P/L':<12}")
    print("-"*80)

    for (model, bet_type), games in sorted(all_results.items()):
        s = stats[f"{model}_{bet_type}"]
        print(f"{model:<20} {bet_type:<10} "
              f"{s['bankruptcy_rate']:>6.2f} ± {s['bankruptcy_sem']:<4.2f} "
              f"{s['avg_rounds']:>5.2f} ± {s['rounds_sem']:<3.2f} "
              f"${s['total_bet']:>6.2f} ± {s['bet_sem']:<5.2f} "
              f"${s['net_pl']:>6.2f} ± {s['pl_sem']:<5.2f}")

    print("="*80)

if __name__ == '__main__':
    main()
