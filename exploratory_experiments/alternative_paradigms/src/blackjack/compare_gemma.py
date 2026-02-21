#!/usr/bin/env python3
"""
Compare Gemma: Fixed ($10, $30) vs Variable
"""

import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import setup_logger

logger = setup_logger(__name__)

def load_data(file_path):
    """Load experiment JSON data"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def analyze_outcomes(data, label):
    """Analyze game outcomes"""
    logger.info(f"\n{'='*60}")
    logger.info(f"{label}")
    logger.info(f"{'='*60}")

    games = data['games']
    total_games = len(games)

    # Count outcomes
    bankrupt = [g for g in games if g['outcome'] == 'bankrupt']
    stopped = [g for g in games if g['outcome'] == 'stopped' or g['outcome'] == 'voluntary_stop']
    maxrounds = [g for g in games if g['outcome'] == 'max_rounds']

    bankruptcy_rate = len(bankrupt) / total_games * 100
    stopped_rate = len(stopped) / total_games * 100
    maxrounds_rate = len(maxrounds) / total_games * 100

    logger.info(f"\n📊 Outcomes:")
    logger.info(f"  Total games:    {total_games}")
    logger.info(f"  Bankrupt:       {len(bankrupt):3d} ({bankruptcy_rate:5.1f}%)")
    logger.info(f"  Stopped:        {len(stopped):3d} ({stopped_rate:5.1f}%)")
    logger.info(f"  Max Rounds:     {len(maxrounds):3d} ({maxrounds_rate:5.1f}%)")

    # Betting stats
    all_bets = []
    for game in games:
        for round_data in game['rounds']:
            all_bets.append(round_data['bet'])

    logger.info(f"\n💰 Betting:")
    logger.info(f"  Mean bet:       ${np.mean(all_bets):.2f}")
    logger.info(f"  Median bet:     ${np.median(all_bets):.2f}")
    logger.info(f"  Std dev:        ${np.std(all_bets):.2f}")
    logger.info(f"  Min/Max:        ${min(all_bets)} / ${max(all_bets)}")

    # Bankruptcy patterns
    if bankrupt:
        bankrupt_rounds = [g['total_rounds'] for g in bankrupt]
        logger.info(f"\n💸 Bankruptcy:")
        logger.info(f"  Mean rounds:    {np.mean(bankrupt_rounds):.1f}")
        logger.info(f"  Median rounds:  {np.median(bankrupt_rounds):.1f}")

    # Bet aggressiveness
    bet_ratios = []
    for game in games:
        for round_data in game['rounds']:
            chips = round_data['chips']
            bet = round_data['bet']
            if chips > 0:
                bet_ratios.append(bet / chips)

    logger.info(f"\n📈 Aggressiveness (Bet/Chips):")
    logger.info(f"  Mean:           {np.mean(bet_ratios):.3f}")
    logger.info(f"  Median:         {np.median(bet_ratios):.3f}")
    logger.info(f"  75th %ile:      {np.percentile(bet_ratios, 75):.3f}")
    logger.info(f"  90th %ile:      {np.percentile(bet_ratios, 90):.3f}")

    return {
        'total_games': total_games,
        'bankruptcy_rate': bankruptcy_rate,
        'stopped_rate': stopped_rate,
        'maxrounds_rate': maxrounds_rate,
        'mean_bet': np.mean(all_bets),
        'median_bet': np.median(all_bets),
        'mean_bet_ratio': np.mean(bet_ratios),
    }

def main():
    logger.info("="*60)
    logger.info("GEMMA: Fixed vs Variable Comparison")
    logger.info("="*60)

    # File paths
    gemma_fixed_10 = "/scratch/x3415a02/data/llm-addiction/blackjack/blackjack_gemma_20260220_023838.json"
    gemma_fixed_30 = "/scratch/x3415a02/data/llm-addiction/blackjack/blackjack_gemma_20260220_113119.json"
    gemma_variable = "/scratch/x3415a02/data/llm-addiction/blackjack/blackjack_gemma_20260221_164948.json"

    logger.info("\nLoading Gemma data...")
    fixed_10 = load_data(gemma_fixed_10)
    fixed_30 = load_data(gemma_fixed_30)
    variable = load_data(gemma_variable)

    # Analyze
    stats_10 = analyze_outcomes(fixed_10, "Gemma - FIXED $10")
    stats_30 = analyze_outcomes(fixed_30, "Gemma - FIXED $30")
    stats_var = analyze_outcomes(variable, "Gemma - VARIABLE")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY COMPARISON")
    logger.info(f"{'='*60}")

    logger.info(f"\n{'Metric':<25} {'Fixed $10':>12} {'Fixed $30':>12} {'Variable':>12}")
    logger.info(f"{'-'*60}")
    logger.info(f"{'Bankruptcy Rate':<25} {stats_10['bankruptcy_rate']:>11.1f}% {stats_30['bankruptcy_rate']:>11.1f}% {stats_var['bankruptcy_rate']:>11.1f}%")
    logger.info(f"{'Stopping Rate':<25} {stats_10['stopped_rate']:>11.1f}% {stats_30['stopped_rate']:>11.1f}% {stats_var['stopped_rate']:>11.1f}%")
    logger.info(f"{'Mean Bet':<25} ${stats_10['mean_bet']:>10.2f} ${stats_30['mean_bet']:>10.2f} ${stats_var['mean_bet']:>10.2f}")
    logger.info(f"{'Mean Bet/Chips':<25} {stats_10['mean_bet_ratio']:>11.3f} {stats_30['mean_bet_ratio']:>11.3f} {stats_var['mean_bet_ratio']:>11.3f}")

    logger.info(f"\n{'='*60}")
    logger.info("KEY FINDINGS")
    logger.info(f"{'='*60}")

    logger.info(f"\n📊 Bankruptcy Rate:")
    logger.info(f"   Fixed $10:   {stats_10['bankruptcy_rate']:5.1f}%")
    logger.info(f"   Fixed $30:   {stats_30['bankruptcy_rate']:5.1f}%")
    logger.info(f"   Variable:    {stats_var['bankruptcy_rate']:5.1f}%")

    # Compare with LLaMA results
    logger.info(f"\n🔍 Comparison with LLaMA:")
    logger.info(f"   LLaMA Fixed $10:  0.0% bankruptcy")
    logger.info(f"   LLaMA Fixed $30:  0.0% bankruptcy")
    logger.info(f"   LLaMA Variable:  11.2% bankruptcy")
    logger.info(f"")
    logger.info(f"   Gemma Fixed $10: {stats_10['bankruptcy_rate']:5.1f}% bankruptcy")
    logger.info(f"   Gemma Fixed $30: {stats_30['bankruptcy_rate']:5.1f}% bankruptcy")
    logger.info(f"   Gemma Variable:  {stats_var['bankruptcy_rate']:5.1f}% bankruptcy")

    logger.info(f"\n{'='*60}")

if __name__ == '__main__':
    main()
