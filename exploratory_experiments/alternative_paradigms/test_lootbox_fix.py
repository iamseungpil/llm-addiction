#!/usr/bin/env python3
"""
Quick test of simplified Lootbox prompt.
Run 10 games to verify prompt parsing improvement.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from common import ModelLoader, setup_logger, set_random_seed
from lootbox.game_logic import LootBoxGame
from lootbox.run_experiment import LootBoxExperiment

logger = setup_logger(__name__)

def quick_test():
    """Run 10 games to test prompt fix"""

    logger.info("=" * 80)
    logger.info("LOOTBOX PROMPT FIX TEST")
    logger.info("Testing simplified prompt (matching Blackjack success pattern)")
    logger.info("=" * 80)

    # Create experiment
    experiment = LootBoxExperiment(
        model_name='llama',
        gpu_id=0,
        output_dir='/scratch/x3415a02/data/llm-addiction/lootbox/test'
    )

    # Load model
    logger.info("Loading LLaMA model...")
    experiment.load_model()

    # Run 10 test games (5 variable, 5 fixed)
    results = []

    logger.info("\n" + "=" * 80)
    logger.info("Running 10 test games (5 variable + 5 fixed)")
    logger.info("=" * 80)

    for game_id in range(10):
        bet_type = 'variable' if game_id < 5 else 'fixed'
        components = 'BASE'  # Simple condition
        seed = game_id + 99999

        logger.info(f"\nGame {game_id+1}/10: bet_type={bet_type}, seed={seed}")

        try:
            result = experiment.play_game(bet_type, components, game_id, seed)
            results.append(result)

            # Print summary
            logger.info(f"  ✓ Completed: {result['rounds_completed']} rounds, "
                       f"Final gems: {result['final_gems']}, "
                       f"Legendary: {result['legendary_obtained']}, "
                       f"Bankrupt: {result['bankruptcy']}, "
                       f"Voluntary stop: {result['stopped_voluntarily']}")

            # Show first few responses
            if 'trials' in result and len(result['trials']) > 0:
                for i, trial in enumerate(result['trials'][:3]):
                    resp = trial.get('response', '')[:60]
                    action = trial.get('action', '?')
                    logger.info(f"    Round {i+1} [{action}]: {resp}")

        except Exception as e:
            logger.error(f"  ✗ Game {game_id+1} failed: {e}")
            import traceback
            traceback.print_exc()

    # Summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 80)

    import numpy as np

    if results:
        rounds = [r['rounds_completed'] for r in results]
        bankruptcies = sum(1 for r in results if r['bankruptcy'])
        voluntary_stops = sum(1 for r in results if r['stopped_voluntarily'])

        logger.info(f"\nGames completed: {len(results)}/10")
        logger.info(f"Average rounds: {np.mean(rounds):.2f} (min: {np.min(rounds)}, max: {np.max(rounds)})")
        logger.info(f"Bankruptcy rate: {bankruptcies}/{len(results)} ({bankruptcies/len(results)*100:.1f}%)")
        logger.info(f"Voluntary stops: {voluntary_stops}/{len(results)} ({voluntary_stops/len(results)*100:.1f}%)")

        # Compare to old results
        logger.info("\n" + "-" * 80)
        logger.info("COMPARISON TO OLD PROMPT:")
        logger.info("  OLD: 36% games = 0 rounds, avg ~2 rounds, 0% bankruptcy")
        logger.info(f"  NEW: {sum(1 for r in rounds if r == 0)/len(rounds)*100:.1f}% games = 0 rounds, "
                   f"avg {np.mean(rounds):.1f} rounds, {bankruptcies/len(results)*100:.1f}% bankruptcy")

        if np.mean(rounds) > 5:
            logger.info("  ✓ IMPROVEMENT: Games running much longer!")
        if bankruptcies > 0:
            logger.info("  ✓ IMPROVEMENT: Bankruptcy rate > 0%!")
    else:
        logger.error("No games completed!")

    logger.info("\n" + "=" * 80)

if __name__ == '__main__':
    quick_test()
