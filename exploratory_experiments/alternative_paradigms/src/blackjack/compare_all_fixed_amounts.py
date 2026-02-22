#!/usr/bin/env python3
"""
Compare LLaMA Fixed betting with different amounts: $10, $30, $50
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

def analyze_fixed_amount(data, amount):
    """Analyze outcomes for a specific fixed bet amount"""
    games = data['games']
    total_games = len(games)

    # Count outcomes
    bankrupt = [g for g in games if g['outcome'] == 'bankrupt']
    stopped = [g for g in games if g['outcome'] == 'stopped' or g['outcome'] == 'voluntary_stop']
    maxrounds = [g for g in games if g['outcome'] == 'max_rounds']

    bankruptcy_rate = len(bankrupt) / total_games * 100
    stopped_rate = len(stopped) / total_games * 100

    # Bet aggressiveness
    bet_ratios = []
    for game in games:
        for round_data in game['rounds']:
            chips = round_data['chips']
            bet = round_data['bet']
            if chips > 0:
                bet_ratios.append(bet / chips)

    # Bankruptcy stats
    bankrupt_rounds = [g['total_rounds'] for g in bankrupt] if bankrupt else [0]

    return {
        'amount': amount,
        'total_games': total_games,
        'bankrupt': len(bankrupt),
        'bankruptcy_rate': bankruptcy_rate,
        'stopped': len(stopped),
        'stopped_rate': stopped_rate,
        'mean_bet_ratio': np.mean(bet_ratios),
        'median_bet_ratio': np.median(bet_ratios),
        'p75_bet_ratio': np.percentile(bet_ratios, 75),
        'p90_bet_ratio': np.percentile(bet_ratios, 90),
        'mean_bankrupt_rounds': np.mean(bankrupt_rounds),
        'median_bankrupt_rounds': np.median(bankrupt_rounds),
    }

def main():
    logger.info("="*70)
    logger.info("LLaMA Fixed Betting: $10 vs $30 vs $50")
    logger.info("="*70)

    # File paths (use most recent for each amount)
    files = {
        10: "/home/jovyan/beomi/llm-addiction-data/blackjack/blackjack_llama_20260217_182212.json",
        30: "/home/jovyan/beomi/llm-addiction-data/blackjack/blackjack_llama_20260217_190727.json",
        50: "/home/jovyan/beomi/llm-addiction-data/blackjack/blackjack_llama_20260217_194244.json",
    }

    results = {}
    for amount, file in files.items():
        logger.info(f"\nAnalyzing Fixed ${amount}...")
        data = load_data(file)
        results[amount] = analyze_fixed_amount(data, amount)

    # Print comparison table
    logger.info(f"\n{'='*70}")
    logger.info("COMPARISON TABLE")
    logger.info(f"{'='*70}")
    logger.info(f"\n{'Metric':<30} {'$10':>10} {'$30':>10} {'$50':>10}")
    logger.info(f"{'-'*70}")

    # Outcomes
    logger.info(f"{'Total Games':<30} {results[10]['total_games']:>10} {results[30]['total_games']:>10} {results[50]['total_games']:>10}")
    logger.info(f"{'Bankrupt Games':<30} {results[10]['bankrupt']:>10} {results[30]['bankrupt']:>10} {results[50]['bankrupt']:>10}")
    logger.info(f"{'Bankruptcy Rate':<30} {results[10]['bankruptcy_rate']:>9.1f}% {results[30]['bankruptcy_rate']:>9.1f}% {results[50]['bankruptcy_rate']:>9.1f}%")
    logger.info(f"{'Stopped Games':<30} {results[10]['stopped']:>10} {results[30]['stopped']:>10} {results[50]['stopped']:>10}")
    logger.info(f"{'Stopping Rate':<30} {results[10]['stopped_rate']:>9.1f}% {results[30]['stopped_rate']:>9.1f}% {results[50]['stopped_rate']:>9.1f}%")

    logger.info(f"\n{'-'*70}")
    logger.info("Betting Aggressiveness (Bet/Chips Ratio)")
    logger.info(f"{'-'*70}")
    logger.info(f"{'Mean Ratio':<30} {results[10]['mean_bet_ratio']:>10.3f} {results[30]['mean_bet_ratio']:>10.3f} {results[50]['mean_bet_ratio']:>10.3f}")
    logger.info(f"{'Median Ratio':<30} {results[10]['median_bet_ratio']:>10.3f} {results[30]['median_bet_ratio']:>10.3f} {results[50]['median_bet_ratio']:>10.3f}")
    logger.info(f"{'75th Percentile':<30} {results[10]['p75_bet_ratio']:>10.3f} {results[30]['p75_bet_ratio']:>10.3f} {results[50]['p75_bet_ratio']:>10.3f}")
    logger.info(f"{'90th Percentile':<30} {results[10]['p90_bet_ratio']:>10.3f} {results[30]['p90_bet_ratio']:>10.3f} {results[50]['p90_bet_ratio']:>10.3f}")

    logger.info(f"\n{'-'*70}")
    logger.info("Bankruptcy Patterns")
    logger.info(f"{'-'*70}")
    logger.info(f"{'Mean Rounds to Bankrupt':<30} {results[10]['mean_bankrupt_rounds']:>10.1f} {results[30]['mean_bankrupt_rounds']:>10.1f} {results[50]['mean_bankrupt_rounds']:>10.1f}")
    logger.info(f"{'Median Rounds to Bankrupt':<30} {results[10]['median_bankrupt_rounds']:>10.1f} {results[30]['median_bankrupt_rounds']:>10.1f} {results[50]['median_bankrupt_rounds']:>10.1f}")

    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("KEY INSIGHTS")
    logger.info(f"{'='*70}")

    logger.info(f"\n📊 Bankruptcy Rate Trend:")
    logger.info(f"   $10: {results[10]['bankruptcy_rate']:.1f}%")
    logger.info(f"   $30: {results[30]['bankruptcy_rate']:.1f}%")
    logger.info(f"   $50: {results[50]['bankruptcy_rate']:.1f}%")

    # Check if trend is linear
    diff_10_30 = results[30]['bankruptcy_rate'] - results[10]['bankruptcy_rate']
    diff_30_50 = results[50]['bankruptcy_rate'] - results[30]['bankruptcy_rate']

    logger.info(f"\n📈 Rate Increase:")
    logger.info(f"   $10 → $30: +{diff_10_30:.1f}pp")
    logger.info(f"   $30 → $50: +{diff_30_50:.1f}pp")

    logger.info(f"\n🎯 Aggressiveness (Mean Bet/Chips):")
    logger.info(f"   $10: {results[10]['mean_bet_ratio']:.1%} of chips")
    logger.info(f"   $30: {results[30]['mean_bet_ratio']:.1%} of chips")
    logger.info(f"   $50: {results[50]['mean_bet_ratio']:.1%} of chips")

    logger.info(f"\n⏱️  Survival Time (when bankrupt):")
    logger.info(f"   $10: {results[10]['median_bankrupt_rounds']:.0f} rounds (median)")
    logger.info(f"   $30: {results[30]['median_bankrupt_rounds']:.0f} rounds (median)")
    logger.info(f"   $50: {results[50]['median_bankrupt_rounds']:.0f} rounds (median)")

    logger.info(f"\n{'='*70}")

if __name__ == '__main__':
    main()
