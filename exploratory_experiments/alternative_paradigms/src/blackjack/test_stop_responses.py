#!/usr/bin/env python3
"""
실제 모델이 정지할 때 어떻게 응답하는지 확인
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from common import set_random_seed, setup_logger
from blackjack.run_experiment import BlackjackExperiment
from blackjack.game_logic import BlackjackGame

logger = setup_logger(__name__)

def main():
    logger.info("="*60)
    logger.info("모델 Stop 응답 테스트")
    logger.info("="*60)

    set_random_seed(42)

    # Create experiment
    experiment = BlackjackExperiment(
        model_name='gemma',
        gpu_id=0,
        bet_type='variable',
    )

    logger.info("Loading Gemma model...")
    experiment.load_model()

    # Run a few games and capture stop responses
    logger.info("\n5개 게임 실행하여 Stop 응답 확인...\n")

    for game_num in range(5):
        logger.info(f"{'='*60}")
        logger.info(f"Game {game_num + 1}")
        logger.info(f"{'='*60}")

        game = BlackjackGame(
            initial_chips=100,
            min_bet=1,
            max_bet=100,
            bet_type='variable'
        )

        components = 'BASE'
        current_goal = None

        round_count = 0
        while not game.is_bankrupt() and round_count < 20:
            # Get decision
            result = experiment.play_round(game, components, current_goal)

            # Log the response
            if 'raw_response' in result:
                logger.info(f"\n  Round {round_count + 1}:")
                logger.info(f"    Chips: ${game.chips}")
                logger.info(f"    모델 응답: \"{result['raw_response']}\"")

            if result.get('stop'):
                logger.info(f"\n  ✋ 정지 결정!")
                logger.info(f"  최종 칩: ${game.chips}")
                logger.info(f"  정지 이유: 모델이 'Stop' 응답")
                break

            round_count += 1

        logger.info("")

if __name__ == '__main__':
    main()
