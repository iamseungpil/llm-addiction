#!/usr/bin/env python3
"""
Quick test script for alternative paradigms experiments.
Runs minimal tests to verify everything works before full experiment.

Usage:
    python test_quick.py --experiment blackjack --model gemma
    python test_quick.py --experiment investment --model gemma
    python test_quick.py --all --model gemma
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from common import ModelLoader, setup_logger, set_random_seed

logger = setup_logger(__name__)


# Lootbox experiment removed


def test_blackjack(model_loader, n_games: int = 2):
    """Test Blackjack experiment with minimal games"""
    from blackjack.game_logic import BlackjackGame
    from blackjack.run_experiment import BlackjackExperiment

    logger.info("=" * 50)
    logger.info("TESTING BLACKJACK EXPERIMENT")
    logger.info("=" * 50)

    set_random_seed(42)

    # Test game logic
    game = BlackjackGame(initial_chips=1000, min_bet=10, max_bet=500)

    # Create experiment instance
    exp = BlackjackExperiment.__new__(BlackjackExperiment)
    exp.model_name = model_loader.model_name
    exp.model_loader = model_loader
    exp.initial_chips = 1000
    exp.min_bet = 10
    exp.max_bet = 500
    exp.max_rounds = 10
    exp.max_retries = 3

    # Test betting phase prompt
    prompt = exp.build_prompt(game, components='GM', is_betting_phase=True)
    logger.info(f"Betting prompt length: {len(prompt)} chars")
    logger.info(f"Prompt preview:\n{prompt[:500]}...")

    # Test model response
    logger.info("\nGenerating model response...")
    response = model_loader.generate(prompt, max_new_tokens=150, temperature=0.7)
    logger.info(f"Response: {response[:200]}...")

    # Test parsing
    bet = exp.parse_bet_decision(response, game)
    logger.info(f"Parsed bet: {bet}")

    # Test play phase
    game.reset_round()
    game.place_bet(50)
    game.deal_initial_cards()

    play_prompt = exp.build_prompt(
        game,
        player_hand=str(game.player_hand),
        dealer_upcard=str(game.dealer_hand.cards[0]),
        components='GM',
        is_betting_phase=False
    )
    logger.info(f"\nPlay prompt preview:\n{play_prompt[:500]}...")

    response = model_loader.generate(play_prompt, max_new_tokens=100, temperature=0.7)
    action = exp.parse_play_decision(response)
    logger.info(f"Parsed action: {action}")

    logger.info("✅ Blackjack test passed")
    return True


def test_investment(model_loader, n_games: int = 2):
    """Test Investment Choice experiment with minimal games"""
    from investment_choice.game_logic import InvestmentChoiceGame
    from investment_choice.run_experiment import InvestmentChoiceExperiment

    logger.info("=" * 50)
    logger.info("TESTING INVESTMENT CHOICE EXPERIMENT")
    logger.info("=" * 50)

    set_random_seed(42)

    # Test game logic
    game = InvestmentChoiceGame(initial_balance=100, bet_type='variable')

    # Create experiment instance
    exp = InvestmentChoiceExperiment.__new__(InvestmentChoiceExperiment)
    exp.model_name = model_loader.model_name
    exp.model_loader = model_loader
    exp.bet_type = 'variable'
    exp.bet_constraint = 'unlimited'
    exp.initial_balance = 100
    exp.max_rounds = 10
    exp.max_retries = 3

    # Test prompt
    prompt = exp.build_prompt(game, prompt_condition='GM', current_goal=None)
    logger.info(f"Prompt length: {len(prompt)} chars")
    logger.info(f"Prompt preview:\n{prompt[:500]}...")

    # Test model response
    logger.info("\nGenerating model response...")
    response = model_loader.generate(prompt, max_new_tokens=200, temperature=0.7)
    logger.info(f"Response: {response[:200]}...")

    # Test parsing
    parsed = exp.parse_choice(response, 'variable')
    logger.info(f"Parsed choice: {parsed}")

    logger.info("✅ Investment Choice test passed")
    return True


def main():
    parser = argparse.ArgumentParser(description='Quick test for alternative paradigms')
    parser.add_argument('--experiment', type=str, choices=['blackjack', 'investment', 'all'],
                        default='all', help='Which experiment to test')
    parser.add_argument('--model', type=str, choices=['llama', 'gemma', 'qwen'],
                        default='gemma', help='Model to use')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("ALTERNATIVE PARADIGMS QUICK TEST")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"GPU: {args.gpu}")
    logger.info(f"Experiment: {args.experiment}")
    logger.info("=" * 60)

    # Load model
    logger.info("\nLoading model...")
    model_loader = ModelLoader(args.model, args.gpu)
    model_loader.load()

    results = {}

    try:
        if args.experiment in ['blackjack', 'all']:
            results['blackjack'] = test_blackjack(model_loader)

        if args.experiment in ['investment', 'all']:
            results['investment'] = test_investment(model_loader)

    finally:
        model_loader.unload()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    for exp, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"  {exp}: {status}")

    all_passed = all(results.values())
    if all_passed:
        logger.info("\n🎉 All tests passed! Ready for full experiment.")
    else:
        logger.info("\n⚠️ Some tests failed. Please check logs above.")

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
