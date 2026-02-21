#!/usr/bin/env python3
"""
Analyze why Fixed betting had higher bankruptcy than Variable betting
Examines actual model outputs and decision patterns
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import setup_logger

logger = setup_logger(__name__)

def load_data(file_path):
    """Load experiment JSON data"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def analyze_bankruptcy_patterns(data, label):
    """Analyze bankruptcy and stopping patterns"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Analyzing: {label}")
    logger.info(f"{'='*60}")

    games = data['games']

    # Basic stats
    total_games = len(games)
    bankrupt_games = [g for g in games if g['outcome'] == 'bankrupt']
    stopped_games = [g for g in games if g['outcome'] == 'stopped']
    maxrounds_games = [g for g in games if g['outcome'] == 'max_rounds']

    bankruptcy_rate = len(bankrupt_games) / total_games * 100
    stopped_rate = len(stopped_games) / total_games * 100
    maxrounds_rate = len(maxrounds_games) / total_games * 100

    logger.info(f"\n📊 Overall Statistics:")
    logger.info(f"  Total games: {total_games}")
    logger.info(f"  Bankrupt: {len(bankrupt_games)} ({bankruptcy_rate:.1f}%)")
    logger.info(f"  Stopped: {len(stopped_games)} ({stopped_rate:.1f}%)")
    logger.info(f"  Max Rounds: {len(maxrounds_games)} ({maxrounds_rate:.1f}%)")

    # Betting patterns
    all_bets = []
    for game in games:
        for round_data in game['rounds']:
            all_bets.append(round_data['bet'])

    logger.info(f"\n💰 Betting Patterns:")
    logger.info(f"  Mean bet: ${np.mean(all_bets):.2f}")
    logger.info(f"  Median bet: ${np.median(all_bets):.2f}")
    logger.info(f"  Min bet: ${min(all_bets)}")
    logger.info(f"  Max bet: ${max(all_bets)}")
    logger.info(f"  Std dev: ${np.std(all_bets):.2f}")

    # Analyze stopping behavior
    stop_chips = []
    stop_rounds = []
    for game in stopped_games:
        stop_chips.append(game['final_chips'])
        stop_rounds.append(game['total_rounds'])

    if stop_chips:
        logger.info(f"\n🛑 Stopping Behavior:")
        logger.info(f"  Games that stopped: {len(stop_chips)}")
        logger.info(f"  Mean chips when stopped: ${np.mean(stop_chips):.2f}")
        logger.info(f"  Median chips when stopped: ${np.median(stop_chips):.2f}")
        logger.info(f"  Mean rounds before stop: {np.mean(stop_rounds):.1f}")

    # Analyze bankruptcy behavior
    bankrupt_rounds = []
    last_bets_before_bankrupt = []
    for game in bankrupt_games:
        bankrupt_rounds.append(game['total_rounds'])
        if game['rounds']:
            last_bets_before_bankrupt.append(game['rounds'][-1]['bet'])

    if bankrupt_rounds:
        logger.info(f"\n💸 Bankruptcy Patterns:")
        logger.info(f"  Games that went bankrupt: {len(bankrupt_rounds)}")
        logger.info(f"  Mean rounds to bankruptcy: {np.mean(bankrupt_rounds):.1f}")
        logger.info(f"  Median rounds to bankruptcy: {np.median(bankrupt_rounds):.1f}")
        if last_bets_before_bankrupt:
            logger.info(f"  Mean final bet: ${np.mean(last_bets_before_bankrupt):.2f}")

    return {
        'total_games': total_games,
        'bankruptcy_rate': bankruptcy_rate,
        'stopped_rate': stopped_rate,
        'maxrounds_rate': maxrounds_rate,
        'mean_bet': np.mean(all_bets),
        'median_bet': np.median(all_bets),
        'stop_chips': stop_chips,
        'bankrupt_rounds': bankrupt_rounds,
        'games': games
    }

def examine_model_outputs(games, label, n_examples=3):
    """Examine actual model prompts and responses"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Sample Model Outputs: {label}")
    logger.info(f"{'='*60}")

    # Get bankrupt games
    bankrupt_games = [g for g in games if g['outcome'] == 'bankrupt']

    if not bankrupt_games:
        logger.info("No bankrupt games found")
        return

    # Sample a few bankrupt games
    import random
    sample_games = random.sample(bankrupt_games, min(n_examples, len(bankrupt_games)))

    for i, game in enumerate(sample_games):
        logger.info(f"\n--- Bankrupt Game Example {i+1} ---")
        logger.info(f"Game ID: {game['game_id']}")
        logger.info(f"Total rounds: {game['total_rounds']}")
        logger.info(f"Initial chips: ${game['initial_chips']}")
        logger.info(f"Final chips: ${game['final_chips']}")

        # Show last few rounds
        last_rounds = game['rounds'][-5:] if len(game['rounds']) >= 5 else game['rounds']

        logger.info(f"\nLast {len(last_rounds)} rounds before bankruptcy:")
        for round_data in last_rounds:
            logger.info(f"\n  Round {round_data['round']}:")
            logger.info(f"    Chips before bet: ${round_data['chips']}")
            logger.info(f"    Bet: ${round_data['bet']}")
            logger.info(f"    Outcome: {round_data['outcome']}")
            if 'payout' in round_data:
                logger.info(f"    Payout: ${round_data['payout']}")

            # Check if there's a stop decision
            if 'stop' in round_data:
                logger.info(f"    Stop decision: {round_data['stop']}")

def compare_betting_aggressiveness(data1, data2, label1, label2):
    """Compare betting aggressiveness between two datasets"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Betting Aggressiveness Comparison")
    logger.info(f"{'='*60}")

    def calculate_aggressiveness(games):
        """Calculate bet/chips ratio"""
        ratios = []
        for game in games:
            for round_data in game['rounds']:
                chips = round_data['chips']
                bet = round_data['bet']
                if chips > 0:
                    ratios.append(bet / chips)
        return ratios

    ratios1 = calculate_aggressiveness(data1['games'])
    ratios2 = calculate_aggressiveness(data2['games'])

    logger.info(f"\n{label1}:")
    logger.info(f"  Mean bet/chips ratio: {np.mean(ratios1):.3f}")
    logger.info(f"  Median bet/chips ratio: {np.median(ratios1):.3f}")
    logger.info(f"  75th percentile: {np.percentile(ratios1, 75):.3f}")
    logger.info(f"  90th percentile: {np.percentile(ratios1, 90):.3f}")
    logger.info(f"  Max ratio: {max(ratios1):.3f}")

    logger.info(f"\n{label2}:")
    logger.info(f"  Mean bet/chips ratio: {np.mean(ratios2):.3f}")
    logger.info(f"  Median bet/chips ratio: {np.median(ratios2):.3f}")
    logger.info(f"  75th percentile: {np.percentile(ratios2, 75):.3f}")
    logger.info(f"  90th percentile: {np.percentile(ratios2, 90):.3f}")
    logger.info(f"  Max ratio: {max(ratios2):.3f}")

def main():
    # File paths
    gemma_fixed_file = "/scratch/x3415a02/data/llm-addiction/blackjack/blackjack_gemma_20260220_023838.json"
    llama_variable_file = "/scratch/x3415a02/data/llm-addiction/blackjack/blackjack_llama_20260219_005625.json"

    logger.info("Loading data files...")

    # Load data
    gemma_fixed = load_data(gemma_fixed_file)
    llama_variable = load_data(llama_variable_file)

    # Analyze patterns
    gemma_stats = analyze_bankruptcy_patterns(gemma_fixed, "Gemma - Fixed Betting")
    llama_stats = analyze_bankruptcy_patterns(llama_variable, "LLaMA - Variable Betting")

    # Compare aggressiveness
    compare_betting_aggressiveness(gemma_fixed, llama_variable,
                                   "Gemma Fixed", "LLaMA Variable")

    # Examine model outputs
    examine_model_outputs(gemma_stats['games'], "Gemma Fixed Betting", n_examples=3)
    examine_model_outputs(llama_stats['games'], "LLaMA Variable Betting", n_examples=3)

    # Summary comparison
    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY COMPARISON")
    logger.info(f"{'='*60}")
    logger.info(f"\nBankruptcy Rate:")
    logger.info(f"  Gemma Fixed:     {gemma_stats['bankruptcy_rate']:.1f}%")
    logger.info(f"  LLaMA Variable:  {llama_stats['bankruptcy_rate']:.1f}%")
    logger.info(f"  Difference:      {gemma_stats['bankruptcy_rate'] - llama_stats['bankruptcy_rate']:.1f}pp")

    logger.info(f"\nStopping Rate:")
    logger.info(f"  Gemma Fixed:     {gemma_stats['stopped_rate']:.1f}%")
    logger.info(f"  LLaMA Variable:  {llama_stats['stopped_rate']:.1f}%")

    logger.info(f"\nMean Bet Size:")
    logger.info(f"  Gemma Fixed:     ${gemma_stats['mean_bet']:.2f}")
    logger.info(f"  LLaMA Variable:  ${llama_stats['mean_bet']:.2f}")

    logger.info(f"\n{'='*60}")

if __name__ == '__main__':
    main()
