#!/usr/bin/env python3
"""
Quick test for Lootbox bankruptcy fix.
Tests only BASE condition with 20 games.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from common import ModelLoader, setup_logger, save_json, set_random_seed
from lootbox.game_logic import LootBoxGame
from lootbox.run_experiment import LootBoxExperiment

logger = setup_logger(__name__)

# Create experiment instance
experiment = LootBoxExperiment(model_name='llama', gpu_id=0)

# Load model
logger.info("Loading model...")
experiment.load_model()

# Run 20 games (BASE condition only, variable betting)
results = []
bet_type = 'variable'
components = 'BASE'

logger.info(f"\n{'='*70}")
logger.info(f"Testing Lootbox Bankruptcy Fix")
logger.info(f"{'='*70}")
logger.info(f"Condition: {bet_type}/{components}")
logger.info(f"Games: 20")
logger.info(f"Initial gems: {experiment.initial_gems}")
logger.info(f"{'='*70}\n")

for i in range(20):
    game_id = i + 1
    seed = game_id + 54321

    logger.info(f"Game {game_id}/20...")
    result = experiment.play_game(bet_type, components, game_id, seed)
    results.append(result)

    logger.info(f"  → Rounds: {result['rounds_completed']}, "
                f"Gems: {result['final_gems']}, "
                f"Bankrupt: {result['bankruptcy']}, "
                f"Voluntary: {result['stopped_voluntarily']}")

# Analyze results
bankruptcies = sum(1 for r in results if r['bankruptcy'])
voluntary = sum(1 for r in results if r['stopped_voluntarily'])
rounds = [r['rounds_completed'] for r in results]

logger.info(f"\n{'='*70}")
logger.info(f"RESULTS SUMMARY")
logger.info(f"{'='*70}")
logger.info(f"Total games: 20")
logger.info(f"Bankruptcies: {bankruptcies}/20 ({bankruptcies/20*100:.1f}%)")
logger.info(f"Voluntary stops: {voluntary}/20 ({voluntary/20*100:.1f}%)")
logger.info(f"Avg rounds: {sum(rounds)/len(rounds):.1f}")
logger.info(f"Min/Max rounds: {min(rounds)}/{max(rounds)}")
logger.info(f"{'='*70}\n")

# Save results
output_file = Path('/scratch/x3415a02/data/llm-addiction/lootbox/test_bankruptcy_fix.json')
save_json({'results': results, 'summary': {
    'bankruptcies': bankruptcies,
    'voluntary_stops': voluntary,
    'avg_rounds': sum(rounds)/len(rounds)
}}, output_file)
logger.info(f"Results saved: {output_file}")
