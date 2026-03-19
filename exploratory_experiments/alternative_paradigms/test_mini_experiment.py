#!/usr/bin/env python3
"""
Mini experiment test - runs a small subset to verify full pipeline.

Tests:
1. Full game play (multiple rounds)
2. Result saving (JSON)
3. Statistics calculation
4. Multiple conditions

Usage:
    python test_mini_experiment.py --experiment investment --model gemma --n-games 3
    python test_mini_experiment.py --experiment blackjack --model gemma --n-games 3
    python test_mini_experiment.py --all --model gemma --n-games 2
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from common import ModelLoader, setup_logger, save_json, set_random_seed

logger = setup_logger(__name__)


def test_investment_mini(model_loader, n_games: int = 3):
    """Run mini Investment Choice experiment"""
    from investment_choice.game_logic import InvestmentChoiceGame
    from investment_choice.run_experiment import InvestmentChoiceExperiment

    logger.info("=" * 60)
    logger.info("MINI EXPERIMENT: Investment Choice")
    logger.info(f"Games per condition: {n_games}")
    logger.info("=" * 60)

    # Create experiment
    output_dir = Path('/home/jovyan/beomi/llm-addiction-data/investment_choice/test')
    output_dir.mkdir(parents=True, exist_ok=True)

    exp = InvestmentChoiceExperiment(
        model_name=model_loader.model_name,
        gpu_id=0,
        bet_type='variable',
        bet_constraint='unlimited',
        output_dir=str(output_dir)
    )
    exp.model_loader = model_loader  # Use already loaded model

    # Test conditions
    conditions = ['BASE', 'GM']
    results = []

    for condition in conditions:
        logger.info(f"\n--- Condition: {condition} ---")
        for game_id in range(n_games):
            seed = game_id + 12345
            logger.info(f"  Game {game_id + 1}/{n_games}")

            try:
                result = exp.play_game(condition, game_id, seed)
                results.append(result)
                logger.info(f"    Rounds: {result['rounds_completed']}, "
                           f"Balance: ${result['final_balance']}, "
                           f"Outcome: {result['final_outcome']}")
            except Exception as e:
                logger.error(f"    Error: {e}")
                return False

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"mini_test_{timestamp}.json"

    save_json({
        'experiment': 'investment_choice_mini_test',
        'model': model_loader.model_name,
        'n_games': len(results),
        'conditions': conditions,
        'results': results
    }, output_file)

    logger.info(f"\n✅ Results saved: {output_file}")

    # Print statistics
    logger.info("\n--- Statistics ---")
    voluntary = sum(1 for r in results if r['stopped_voluntarily'])
    bankrupt = sum(1 for r in results if r['bankruptcy'])
    avg_rounds = sum(r['rounds_completed'] for r in results) / len(results)

    logger.info(f"Voluntary stops: {voluntary}/{len(results)}")
    logger.info(f"Bankruptcies: {bankrupt}/{len(results)}")
    logger.info(f"Avg rounds: {avg_rounds:.1f}")

    return True


# Lootbox experiment removed


def test_blackjack_mini(model_loader, n_games: int = 3):
    """Run mini Blackjack experiment"""
    from blackjack.game_logic import BlackjackGame
    from blackjack.run_experiment import BlackjackExperiment

    logger.info("=" * 60)
    logger.info("MINI EXPERIMENT: Blackjack")
    logger.info(f"Games per condition: {n_games}")
    logger.info("=" * 60)

    # Create experiment
    output_dir = Path('/home/jovyan/beomi/llm-addiction-data/blackjack/test')
    output_dir.mkdir(parents=True, exist_ok=True)

    exp = BlackjackExperiment(
        model_name=model_loader.model_name,
        gpu_id=0,
        bet_type='variable',
        output_dir=str(output_dir)
    )
    exp.model_loader = model_loader

    # Test conditions
    conditions = ['BASE', 'GM']
    results = []

    for condition in conditions:
        logger.info(f"\n--- Condition: {condition} ---")
        for game_id in range(n_games):
            seed = game_id + 99999
            logger.info(f"  Game {game_id + 1}/{n_games}")

            try:
                result = exp.play_game(condition, game_id, seed)
                results.append(result)
                logger.info(f"    Rounds: {result['total_rounds']}, "
                           f"Chips: {result['final_chips']}, "
                           f"Outcome: {result['outcome']}")
            except Exception as e:
                logger.error(f"    Error: {e}")
                import traceback
                traceback.print_exc()
                return False

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"mini_test_{timestamp}.json"

    save_json({
        'experiment': 'blackjack_mini_test',
        'model': model_loader.model_name,
        'n_games': len(results),
        'conditions': conditions,
        'results': results
    }, output_file)

    logger.info(f"\n✅ Results saved: {output_file}")

    # Print statistics
    logger.info("\n--- Statistics ---")
    bankrupt = sum(1 for r in results if r['outcome'] == 'bankrupt')
    voluntary = sum(1 for r in results if r['outcome'] == 'voluntary_stop')
    max_rounds = sum(1 for r in results if r['outcome'] == 'max_rounds')
    avg_rounds = sum(r['total_rounds'] for r in results) / len(results)

    logger.info(f"Bankruptcies: {bankrupt}/{len(results)}")
    logger.info(f"Voluntary stops: {voluntary}/{len(results)}")
    logger.info(f"Max rounds reached: {max_rounds}/{len(results)}")
    logger.info(f"Avg rounds: {avg_rounds:.1f}")

    return True


def main():
    parser = argparse.ArgumentParser(description='Mini experiment test')
    parser.add_argument('--experiment', type=str,
                        choices=['investment', 'blackjack', 'all'],
                        default='investment', help='Which experiment to test')
    parser.add_argument('--model', type=str, choices=['llama', 'gemma', 'qwen'],
                        default='gemma', help='Model to use')
    parser.add_argument('--n-games', type=int, default=3,
                        help='Number of games per condition')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("MINI EXPERIMENT PIPELINE TEST")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Experiment: {args.experiment}")
    logger.info(f"Games per condition: {args.n_games}")
    logger.info("=" * 60)

    # Load model
    logger.info("\nLoading model...")
    model_loader = ModelLoader(args.model, args.gpu)
    model_loader.load()

    results = {}

    try:
        if args.experiment in ['investment', 'all']:
            results['investment'] = test_investment_mini(model_loader, args.n_games)

        if args.experiment in ['blackjack', 'all']:
            results['blackjack'] = test_blackjack_mini(model_loader, args.n_games)

    finally:
        model_loader.unload()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("MINI EXPERIMENT SUMMARY")
    logger.info("=" * 60)

    all_passed = True
    for exp, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"  {exp}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        logger.info("\n🎉 All mini experiments passed! Ready for full experiment.")
    else:
        logger.info("\n⚠️ Some experiments failed. Check logs above.")

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
