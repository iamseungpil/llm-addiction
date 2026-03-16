#!/usr/bin/env python3
"""
Resume Coin Flip Experiment from checkpoint_950.

Checkpoint has 19/32 variable conditions completed (R-naming).
Current code uses H-naming (R→H rename).
This script:
1. Loads existing 950 games from checkpoint
2. Renames R→H in condition names for consistency
3. Runs remaining 13 variable conditions (650 games)
4. Runs all 32 fixed conditions (1600 games)
5. Saves combined final output (3200 games)

Usage:
    python src/coin_flip/resume_experiment.py --model gemma --gpu 0
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from common import ModelLoader, setup_logger, save_json, set_random_seed
from coin_flip.run_experiment import CoinFlipExperiment

logger = setup_logger(__name__)

CHECKPOINT_FILE = Path("/home/jovyan/beomi/llm-addiction-data/coin_flip/gemma_coinflip_checkpoint_950.json")
OUTPUT_DIR = Path("/home/jovyan/beomi/llm-addiction-data/coin_flip")

# R→H mapping for condition names
def rename_r_to_h(condition: str) -> str:
    """Rename R→H in condition name (sorted alphabetically)."""
    if condition == "BASE":
        return condition
    components = list(condition)
    components = [('H' if c == 'R' else c) for c in components]
    return ''.join(sorted(components))


def main():
    parser = argparse.ArgumentParser(description="Resume Coin Flip Experiment")
    parser.add_argument('--model', type=str, default='gemma', choices=['llama', 'gemma'])
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("COIN FLIP EXPERIMENT — RESUME FROM CHECKPOINT")
    logger.info("=" * 70)

    # 1. Load checkpoint
    logger.info(f"Loading checkpoint: {CHECKPOINT_FILE}")
    with open(CHECKPOINT_FILE) as f:
        ckpt = json.load(f)

    existing_results = ckpt["results"]
    logger.info(f"Existing games: {len(existing_results)}")

    # 2. Rename R→H in existing results
    for g in existing_results:
        old_cond = g.get("prompt_condition", "")
        new_cond = rename_r_to_h(old_cond)
        if old_cond != new_cond:
            g["prompt_condition"] = new_cond

    completed_conditions = set(g["prompt_condition"] for g in existing_results)
    logger.info(f"Completed conditions (renamed R→H): {sorted(completed_conditions)}")

    # 3. Create experiment instance
    experiment = CoinFlipExperiment(
        args.model,
        args.gpu,
        bet_type='variable',
        bet_constraint='10',
    )

    # Get full condition list from code
    all_combos = experiment.get_prompt_combinations()
    all_condition_names = [name for name, _ in all_combos]
    logger.info(f"All conditions: {len(all_condition_names)}")

    # Remaining variable conditions
    remaining_variable = [c for c in all_condition_names if c not in completed_conditions]
    logger.info(f"Remaining variable conditions: {len(remaining_variable)} → {remaining_variable}")

    # All fixed conditions (none were done)
    all_fixed = all_condition_names
    logger.info(f"Fixed conditions: {len(all_fixed)} (all)")

    total_remaining = len(remaining_variable) * 50 + len(all_fixed) * 50
    logger.info(f"Total remaining games: {total_remaining}")
    logger.info(f"Final total: {len(existing_results) + total_remaining}")

    # 4. Load model
    experiment.load_model()

    # Track game_id continuing from checkpoint
    # Checkpoint ended at game_id = 950 (conditions) but some games beyond that were lost
    # We use a fresh counter starting from max existing game_id + 1
    max_existing_id = max(g.get("game_id", 0) for g in existing_results)
    game_id = max_existing_id
    repetitions = 50
    results = list(existing_results)  # Start with existing

    # 5. Run remaining VARIABLE conditions
    if remaining_variable:
        experiment.bet_type = 'variable'
        logger.info(f"\n{'='*70}")
        logger.info(f"PHASE 1: REMAINING VARIABLE CONDITIONS ({len(remaining_variable)} × {repetitions} = {len(remaining_variable)*50} games)")
        logger.info(f"{'='*70}")

        for condition in remaining_variable:
            logger.info(f"\nCondition: variable/{condition}")

            for rep in tqdm(range(repetitions), desc=f"  variable/{condition}"):
                game_id += 1
                seed = game_id + 99999

                try:
                    result = experiment.play_game(condition, game_id, seed)
                    results.append(result)
                except Exception as e:
                    logger.error(f"  Game {game_id} failed: {e}")
                    continue

            # Checkpoint after each condition
            ckpt_file = OUTPUT_DIR / f"{args.model}_coinflip_resume_checkpoint_{len(results)}.json"
            save_json({'results': results, 'completed': len(results), 'total': 3200}, ckpt_file)
            logger.info(f"  Checkpoint: {ckpt_file} ({len(results)}/3200)")

    # 6. Run ALL FIXED conditions
    experiment.bet_type = 'fixed'
    logger.info(f"\n{'='*70}")
    logger.info(f"PHASE 2: FIXED BET CONDITIONS ({len(all_fixed)} × {repetitions} = {len(all_fixed)*50} games)")
    logger.info(f"{'='*70}")

    for condition in all_fixed:
        logger.info(f"\nCondition: fixed/{condition}")

        for rep in tqdm(range(repetitions), desc=f"  fixed/{condition}"):
            game_id += 1
            seed = game_id + 99999

            try:
                result = experiment.play_game(condition, game_id, seed)
                results.append(result)
            except Exception as e:
                logger.error(f"  Game {game_id} failed: {e}")
                continue

        # Checkpoint after each condition
        ckpt_file = OUTPUT_DIR / f"{args.model}_coinflip_resume_checkpoint_{len(results)}.json"
        save_json({'results': results, 'completed': len(results), 'total': 3200}, ckpt_file)
        logger.info(f"  Checkpoint: {ckpt_file} ({len(results)}/3200)")

    # 7. Save final output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_file = OUTPUT_DIR / f"{args.model}_coinflip_c10_final_{timestamp}.json"

    final_output = {
        'experiment': 'coin_flip',
        'model': args.model,
        'timestamp': timestamp,
        'resumed_from': str(CHECKPOINT_FILE),
        'config': {
            'initial_balance': 100,
            'max_rounds': 100,
            'bet_types': ['variable', 'fixed'],
            'bet_constraint': '10',
            'total_games': len(results),
            'conditions': 32,
            'condition_names': all_condition_names,
            'repetitions': 50,
            'win_probability': 0.50,
            'payout_multiplier': 1.8,
            'expected_value': 0.90,
            'factorial_design': '2^5 (G, M, H, W, P)',
        },
        'results': results
    }

    save_json(final_output, final_file)

    logger.info("=" * 70)
    logger.info("EXPERIMENT COMPLETED")
    logger.info(f"Total games: {len(results)}")
    logger.info(f"  Variable: {sum(1 for r in results if r.get('bet_type') == 'variable')}")
    logger.info(f"  Fixed: {sum(1 for r in results if r.get('bet_type') == 'fixed')}")
    logger.info(f"Output: {final_file}")
    logger.info("=" * 70)

    # Summary
    experiment.print_summary(results)


if __name__ == '__main__':
    main()
