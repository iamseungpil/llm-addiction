#!/usr/bin/env python3
"""
New format quick test for LLaMA and Gemma
- 수정된 프롬프트 포맷 검증: "Respond using this format: Reasoning / Final Decision"
- Quick: 2 bet_type × 4 conditions × 5 reps = 40 games per model
"""

import sys
import json
import re
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import ModelLoader, setup_logger, save_json, set_random_seed
from investment_choice.game_logic import InvestmentChoiceGame
# 수정된 build_prompt를 run_experiment에서 직접 가져옴
from investment_choice.run_experiment import InvestmentChoiceExperiment

logger = setup_logger(__name__)

OUTPUT_DIR = Path('/scratch/x3415a02/data/llm-addiction/investment_choice')


def run_format_test(model_name, gpu_id):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"{model_name}_new_format_test_{timestamp}.json"

    conditions = ['BASE', 'G', 'M', 'GM']
    bet_types = ['variable', 'fixed']
    reps = 5  # 2 × 4 × 5 = 40 games

    # Experiment 인스턴스 (build_prompt 재사용)
    exp = InvestmentChoiceExperiment(model_name, gpu_id, 'variable', '50')
    exp.load_model()

    results = []
    parse_stats = defaultdict(int)
    game_id = 0

    logger.info("=" * 60)
    logger.info(f"NEW FORMAT TEST ({model_name.upper()}, c50)")
    logger.info(f"Total games: {len(bet_types) * len(conditions) * reps}")
    logger.info("=" * 60)

    for bet_type in bet_types:
        exp.bet_type = bet_type
        for cond in conditions:
            logger.info(f"\n[{bet_type}/{cond}]")
            for rep in range(reps):
                game_id += 1
                seed = game_id + 55555
                set_random_seed(seed)

                game = InvestmentChoiceGame(
                    initial_balance=100,
                    max_rounds=100,
                    bet_type=bet_type,
                    bet_constraint=50
                )

                current_goal = None
                decisions = []

                while not game.is_finished and game.round < game.max_rounds:
                    prompt = exp.build_prompt(game, cond, current_goal)

                    response = exp.model_loader.generate(
                        prompt, max_new_tokens=250, temperature=0.7
                    )

                    # 파싱
                    if bet_type == 'variable':
                        parsed = exp.parse_choice_variable(response, game.balance)
                    else:
                        parsed = exp.parse_choice_fixed(response)

                    # 파싱 분류
                    resp_lower = response.lower()
                    if 'final decision:' in resp_lower and 'reasoning:' in resp_lower:
                        method = 'full_format'       # Reasoning + Final Decision 둘 다
                    elif 'final decision:' in resp_lower:
                        method = 'final_only'        # Final Decision만
                    elif 'reasoning:' in resp_lower:
                        method = 'reasoning_only'    # Reasoning만 (Final Decision 없음)
                    else:
                        method = 'no_format'         # 둘 다 없음
                    parse_stats[method] += 1

                    if not parsed.get('valid'):
                        logger.warning(f"  Parse invalid | method={method} | resp: {response[:80]!r}")

                    # Goal 추출
                    if 'G' in cond and response:
                        extracted = exp.extract_goal_from_response(response)
                        if extracted:
                            current_goal = extracted

                    choice = parsed['choice']
                    bet_amount = parsed.get('bet_amount')
                    outcome = game.play_round(choice, bet_amount)

                    if 'error' in outcome:
                        logger.error(f"  Game error: {outcome['error']}")
                        break

                    decisions.append({
                        'round': game.round,
                        'choice': choice,
                        'bet_amount': bet_amount,
                        'parse_method': method,
                        'parse_valid': parsed.get('valid'),
                        'response': response,
                        'outcome': outcome.get('outcome'),
                    })

                    if outcome.get('is_finished'):
                        break

                result = game.get_game_result()
                result.update({
                    'game_id': game_id,
                    'bet_type': bet_type,
                    'prompt_condition': cond,
                    'seed': seed,
                    'decisions': decisions,
                })
                results.append(result)
                logger.info(f"  Game {game_id}: rounds={result['rounds_completed']} "
                           f"balance=${result['final_balance']} bankrupt={result['bankruptcy']}")

    # 파싱 통계
    total = sum(parse_stats.values())
    logger.info("\n" + "=" * 60)
    logger.info("PARSE STATISTICS")
    logger.info("=" * 60)
    for method, count in sorted(parse_stats.items()):
        logger.info(f"  {method}: {count} ({100*count/total:.1f}%)")

    import numpy as np
    for bt in ['variable', 'fixed']:
        subset = [r for r in results if r['bet_type'] == bt]
        if subset:
            br = np.mean([r['bankruptcy'] for r in subset])
            sr = np.mean([r['stopped_voluntarily'] for r in subset])
            logger.info(f"[{bt}] bankruptcy={br:.1%} stopped={sr:.1%}")

    save_json({'results': results, 'parse_stats': dict(parse_stats)}, output_file)
    logger.info(f"\nSaved: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['llama', 'gemma'])
    parser.add_argument('--gpu', type=int, default=1)
    args = parser.parse_args()
    run_format_test(args.model, args.gpu)
