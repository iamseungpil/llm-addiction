#!/usr/bin/env python3
"""
Correct analysis: Compare Fixed vs Variable within the SAME model
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
        'bankrupt_games': bankrupt,
        'stopped_games': stopped,
    }

def main():
    logger.info("="*60)
    logger.info("CORRECT COMPARISON: Same Model, Different Bet Types")
    logger.info("="*60)

    # LLaMA comparison
    llama_fixed_file = "/scratch/x3415a02/data/llm-addiction/blackjack/blackjack_llama_20260217_194244.json"
    llama_variable_file = "/scratch/x3415a02/data/llm-addiction/blackjack/blackjack_llama_20260219_005625.json"

    logger.info("\nLoading LLaMA data...")
    llama_fixed = load_data(llama_fixed_file)
    llama_variable = load_data(llama_variable_file)

    # Analyze
    fixed_stats = analyze_outcomes(llama_fixed, "LLaMA - FIXED Betting")
    variable_stats = analyze_outcomes(llama_variable, "LLaMA - VARIABLE Betting")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY: LLaMA Fixed vs Variable")
    logger.info(f"{'='*60}")
    logger.info(f"\nBankruptcy Rate:")
    logger.info(f"  Fixed:          {fixed_stats['bankruptcy_rate']:5.1f}%")
    logger.info(f"  Variable:       {variable_stats['bankruptcy_rate']:5.1f}%")
    logger.info(f"  Difference:     {fixed_stats['bankruptcy_rate'] - variable_stats['bankruptcy_rate']:+5.1f}pp")

    logger.info(f"\nStopping Rate:")
    logger.info(f"  Fixed:          {fixed_stats['stopped_rate']:5.1f}%")
    logger.info(f"  Variable:       {variable_stats['stopped_rate']:5.1f}%")

    logger.info(f"\nMean Bet:")
    logger.info(f"  Fixed:          ${fixed_stats['mean_bet']:.2f}")
    logger.info(f"  Variable:       ${variable_stats['mean_bet']:.2f}")

    # Effect size
    diff = fixed_stats['bankruptcy_rate'] - variable_stats['bankruptcy_rate']
    if diff > 0:
        logger.info(f"\n⚠️  Fixed betting has {diff:.1f}pp HIGHER bankruptcy rate!")
        logger.info(f"   ({fixed_stats['bankruptcy_rate']:.1f}% vs {variable_stats['bankruptcy_rate']:.1f}%)")
    elif diff < 0:
        logger.info(f"\n✅ Variable betting has {-diff:.1f}pp HIGHER bankruptcy rate!")
        logger.info(f"   ({variable_stats['bankruptcy_rate']:.1f}% vs {fixed_stats['bankruptcy_rate']:.1f}%)")
    else:
        logger.info(f"\n🤷 No difference in bankruptcy rate")

    logger.info(f"\n{'='*60}")

if __name__ == '__main__':
    main()
