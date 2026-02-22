#!/usr/bin/env python3
"""
Blackjack Pilot Test with New Prompt Design

Quick test to verify:
1. No biasing labels
2. Option parsing works
3. Voluntary stop rate increases
"""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from blackjack.run_experiment import BlackjackExperiment
from common import setup_logger, save_json

logger = setup_logger(__name__)

def run_mini_pilot(model_name='llama', gpu_id=0, n_games=5):
    """
    Run minimal pilot test (5 games).

    Args:
        model_name: Model to test
        gpu_id: GPU ID
        n_games: Number of games (default 5 for quick test)
    """
    logger.info("=" * 80)
    logger.info("BLACKJACK PILOT TEST - New Prompt Design")
    logger.info("=" * 80)
    logger.info(f"Model: {model_name}")
    logger.info(f"Games: {n_games}")
    logger.info(f"Purpose: Verify Option-based prompt works correctly")
    logger.info("=" * 80)

    # Create experiment
    output_dir = Path('/home/jovyan/beomi/llm-addiction-data/blackjack')
    exp = BlackjackExperiment(
        model_name=model_name,
        gpu_id=gpu_id,
        bet_type='variable',
        output_dir=str(output_dir)
    )

    # Load model
    logger.info("\nLoading model...")
    exp.load_model()

    # Run games
    results = []
    for game_id in range(n_games):
        logger.info(f"\n{'='*60}")
        logger.info(f"Game {game_id + 1}/{n_games}")
        logger.info(f"{'='*60}")

        seed = game_id * 1000
        result = exp.play_game(
            components='BASE',
            game_id=game_id,
            seed=seed
        )
        results.append(result)

        # Log result
        logger.info(f"Outcome: {result['outcome']}")
        logger.info(f"Rounds: {result['total_rounds']}")
        logger.info(f"Final chips: ${result['final_chips']}")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'pilot_test_{model_name}_{timestamp}.json'

    save_json({
        'experiment': 'blackjack_pilot_test',
        'model': model_name,
        'timestamp': timestamp,
        'n_games': len(results),
        'prompt_design': 'option_based_no_bias',
        'results': results
    }, output_file)

    # Analysis
    logger.info("\n" + "=" * 80)
    logger.info("PILOT TEST RESULTS")
    logger.info("=" * 80)

    voluntary_stops = sum(1 for r in results if r['outcome'] == 'voluntary_stop')
    bankruptcies = sum(1 for r in results if r['outcome'] == 'bankrupt')
    max_rounds = sum(1 for r in results if r['outcome'] == 'max_rounds')

    avg_rounds = sum(r['total_rounds'] for r in results) / len(results)
    avg_final = sum(r['final_chips'] for r in results) / len(results)

    logger.info(f"\nTotal games: {len(results)}")
    logger.info(f"Voluntary stops: {voluntary_stops} ({voluntary_stops/len(results)*100:.1f}%)")
    logger.info(f"Bankruptcies: {bankruptcies} ({bankruptcies/len(results)*100:.1f}%)")
    logger.info(f"Max rounds: {max_rounds} ({max_rounds/len(results)*100:.1f}%)")
    logger.info(f"\nAverage rounds: {avg_rounds:.1f}")
    logger.info(f"Average final chips: ${avg_final:.1f}")

    logger.info(f"\nResults saved to: {output_file}")

    # Verdict
    logger.info("\n" + "=" * 80)
    logger.info("VERDICT")
    logger.info("=" * 80)

    if voluntary_stops > 0:
        logger.info("✅ SUCCESS: Voluntary stops detected!")
        logger.info(f"   Old design: 0% voluntary stops")
        logger.info(f"   New design: {voluntary_stops/len(results)*100:.1f}% voluntary stops")
        logger.info("\n   Ready to proceed with full quick mode (160 games)")
    else:
        logger.info("⚠️  WARNING: No voluntary stops yet")
        logger.info("   May need more games or prompt adjustment")
        logger.info("   Recommend running more games before conclusion")

    logger.info("=" * 80)

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Blackjack Pilot Test')
    parser.add_argument('--model', type=str, default='llama',
                        choices=['llama', 'gemma', 'qwen'],
                        help='Model to test')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--n-games', type=int, default=5,
                        help='Number of games to test')

    args = parser.parse_args()

    run_mini_pilot(
        model_name=args.model,
        gpu_id=args.gpu,
        n_games=args.n_games
    )
