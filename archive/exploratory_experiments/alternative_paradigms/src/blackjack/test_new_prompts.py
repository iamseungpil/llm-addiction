#!/usr/bin/env python3
"""
Quick test of new Gemma SimpleFormat prompts for Blackjack

Tests 10 games to verify:
1. Continue/Stop + Bet Amount parsing (1-phase)
2. Play Action (Hit/Stand) parsing
3. No parsing failures
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from common import set_random_seed, setup_logger
from blackjack.run_experiment import BlackjackExperiment

logger = setup_logger(__name__)

def main():
    logger.info("="*60)
    logger.info("Testing New Gemma SimpleFormat Prompts")
    logger.info("="*60)

    set_random_seed(42)

    # Create experiment
    experiment = BlackjackExperiment(
        model_name='gemma',
        gpu_id=0,
        bet_type='variable',
        bet_constraint=50  # $1-$50 betting
    )

    logger.info("Loading Gemma model...")
    experiment.load_model()

    logger.info("\nRunning 10-game quick test...")
    logger.info("Bet type: Variable ($1-$50)")
    logger.info("Components: BASE (no special features)")
    logger.info("")

    # Modify experiment to run only 10 games for quick test
    # We'll manually run a few games
    from blackjack.game_logic import BlackjackGame

    parse_failures = {
        'continue_stop': 0,
        'play_action': 0
    }

    total_rounds = 0

    for game_num in range(10):
        logger.info(f"Game {game_num + 1}/10")

        game = BlackjackGame(
            initial_chips=100,
            min_bet=1,
            max_bet=50,
            bet_type='variable'
        )

        components = 'BASE'
        current_goal = None

        round_count = 0
        while not game.is_bankrupt() and round_count < 20:  # Max 20 rounds per game
            result = experiment.play_round(game, components, current_goal)

            if result.get('stop'):
                logger.info(f"  → Stopped at ${game.chips}")
                break

            round_count += 1
            total_rounds += 1

            if round_count >= 20:
                logger.info(f"  → Max rounds reached, final chips: ${game.chips}")
                break

        if game.is_bankrupt():
            logger.info(f"  → Bankrupt after {round_count} rounds")

        logger.info("")

    logger.info("="*60)
    logger.info("Test Complete!")
    logger.info(f"Total rounds played: {total_rounds}")
    logger.info("")
    logger.info("Check logs above for any parsing failures.")
    logger.info("If no 'Failed to parse' warnings, the new prompts work perfectly!")
    logger.info("="*60)

if __name__ == '__main__':
    main()
