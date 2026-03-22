#!/usr/bin/env python3
"""
LLaMA Mystery Wheel experiment via vLLM server (async, 64-game parallel).

Uses the ORIGINAL MysteryWheelExperiment class for prompt building, parsing,
and game logic — only replaces model_loader.generate() with vLLM API calls.
This ensures 100% identical prompts, parsing, and data format as Gemma MW.

Prerequisites:
  1. vLLM server running:
     conda activate vllm-server
     python -m vllm.entrypoints.openai.api_server \
       --model meta-llama/Llama-3.1-8B-Instruct --dtype bfloat16 --port 8000

  2. Run this script (llm-addiction env):
     conda activate llm-addiction
     python run_mw_vllm.py
"""
import os, sys, json, asyncio, random, logging, time, copy
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Add experiment source to path
SRC_DIR = Path(__file__).parent / "exploratory_experiments" / "alternative_paradigms" / "src"
sys.path.insert(0, str(SRC_DIR))

from mystery_wheel.game_logic import MysteryWheelGame
from mystery_wheel.run_experiment import (
    MysteryWheelExperiment, PROMPT_TO_GAME, GAME_TO_PROMPT,
    PROMPT_COMPONENTS, MIN_VARIABLE_BET, ROLE_INSTRUCTION,
)
from common.utils import save_json

from openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mw_vllm")

# ── Configuration ──
VLLM_URL = "http://localhost:8000/v1"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
TEMPERATURE = 0.7
MAX_TOKENS = 1024
MAX_RETRIES = 5
MAX_CONCURRENT = 64
BET_CONSTRAINT = 30
REPETITIONS = 50
OUTPUT_DIR = Path("/home/v-seungplee/data/llm-addiction/behavioral/mystery_wheel/llama_v2_role")


class VLLMModelShim:
    """Drop-in replacement for ModelLoader that routes generate() to vLLM API.

    Supports synchronous generate() by running async code in a new event loop,
    BUT in the async game runner we call generate_async() directly.
    """
    def __init__(self, client: AsyncOpenAI):
        self.client = client
        self.config = {'chat_template': True}  # LLaMA-Instruct uses chat template

    async def generate_async(self, prompt: str, max_new_tokens: int = 1024,
                              temperature: float = 0.7, **kwargs) -> str:
        try:
            completion = await self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=temperature,
            )
            return completion.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"vLLM API error: {e}")
            return ""


def make_exp_instance(bet_type: str, client: AsyncOpenAI) -> MysteryWheelExperiment:
    """Create a lightweight per-game experiment instance (no model loading)."""
    exp = MysteryWheelExperiment.__new__(MysteryWheelExperiment)
    exp.model_name = 'llama'
    exp.gpu_id = 0
    exp.bet_constraint = BET_CONSTRAINT
    exp.bet_type = bet_type
    exp.initial_balance = 100
    exp.max_rounds = 100
    exp.max_retries = MAX_RETRIES
    exp.results_dir = OUTPUT_DIR
    exp.model_loader = VLLMModelShim(client)
    return exp


async def play_game_async(
    client: AsyncOpenAI,
    prompt_condition: str,
    bet_type: str,
    game_id: int,
    seed: int,
    semaphore: asyncio.Semaphore,
) -> Dict:
    """Play one MW game using original experiment logic + vLLM API.

    Each game gets its own exp instance (no shared mutable state)
    and its own random.Random instance (no global seed corruption).
    """
    async with semaphore:
        # Per-game experiment instance (avoids bet_type race condition)
        exp = make_exp_instance(bet_type, client)

        # Per-game RNG to avoid global random state corruption
        rng = random.Random(seed)

        game = MysteryWheelGame(
            initial_balance=100,
            max_rounds=100,
            bet_type=bet_type,
            bet_constraint=BET_CONSTRAINT,
        )
        # Patch game's _spin_wheel to use per-game RNG
        def seeded_spin():
            from mystery_wheel.game_logic import ZONES
            roll = rng.random()
            cumulative = 0.0
            for name, prob, mult in ZONES:
                cumulative += prob
                if roll < cumulative:
                    return name, mult
            return ZONES[-1][0], ZONES[-1][2]
        game._spin_wheel = seeded_spin

        decisions = []
        current_goal = None
        consecutive_skips = 0
        total_skips = 0
        max_consecutive_skips = 10
        max_total_skips = 30

        while not game.is_finished and game.round < game.max_rounds:
            # Build prompt using original method (exp has correct bet_type set)
            base_prompt = exp.build_prompt(game, prompt_condition, current_goal)

            parsed_choice = None
            response = None

            for retry in range(MAX_RETRIES):
                prompt = base_prompt

                # Retry hints (identical to original)
                if retry > 0:
                    if bet_type == 'variable':
                        fmt_hint = "\nIMPORTANT: You MUST end with exactly: Final Decision: Option 1, $Y  or  Final Decision: Option 2"
                    else:
                        fmt_hint = "\nIMPORTANT: You MUST end with exactly: Final Decision: Option 1  or  Final Decision: Option 2"
                    prompt = base_prompt.replace(
                        "\nExplain your reasoning",
                        fmt_hint + "\nExplain your reasoning"
                    )

                # Generate via vLLM API
                try:
                    completion = await client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=MAX_TOKENS,
                        temperature=TEMPERATURE,
                    )
                    response = completion.choices[0].message.content or ""
                except Exception as e:
                    logger.error(f"Game {game_id} R{game.round+1}: API error: {e}")
                    response = ""

                # Parse using original methods
                if bet_type == 'fixed':
                    parsed_choice = exp.parse_choice_fixed(response)
                else:
                    parsed_choice = exp.parse_choice_variable(response, game.balance)

                if parsed_choice.get('valid'):
                    break

            # All retries failed
            if not parsed_choice or not parsed_choice.get('valid'):
                consecutive_skips += 1
                total_skips += 1
                decisions.append({
                    'round': game.round + 1,
                    'balance_before': game.balance,
                    'choice': None,
                    'prompt_option': None,
                    'bet_amount': None,
                    'goal_before': None if 'G' not in prompt_condition else current_goal,
                    'goal_after': current_goal if 'G' in prompt_condition else None,
                    'full_prompt': base_prompt,
                    'response': response,
                    'skipped': True,
                    'skip_reason': parsed_choice.get('reason') if parsed_choice else 'no_parse',
                })
                if consecutive_skips >= max_consecutive_skips or total_skips >= max_total_skips:
                    break
                continue

            consecutive_skips = 0

            # Goal extraction using original method
            if 'G' in prompt_condition and response:
                extracted_goal = exp.extract_goal_from_response(response)
                if extracted_goal:
                    current_goal = extracted_goal

            # Save decision (identical to original)
            decision_info = {
                'round': game.round + 1,
                'balance_before': game.balance,
                'choice': parsed_choice['choice'],
                'prompt_option': GAME_TO_PROMPT[parsed_choice['choice']],
                'bet_amount': parsed_choice.get('bet_amount'),
                'goal_before': None if 'G' not in prompt_condition else (current_goal if game.round > 0 else None),
                'goal_after': current_goal if 'G' in prompt_condition else None,
                'full_prompt': base_prompt,
                'actual_prompt': prompt,
                'response': response,
                'parse_reason': parsed_choice.get('reason'),
                'skipped': False,
            }

            choice = parsed_choice['choice']
            bet_amount = parsed_choice.get('bet_amount')
            outcome = game.play_round(choice, bet_amount)

            if 'error' in outcome:
                break

            decision_info['outcome'] = outcome
            decision_info['balance_after'] = game.balance
            decisions.append(decision_info)

            if outcome.get('is_finished'):
                break

        # Final result using original game method
        result = game.get_game_result()
        result['game_id'] = game_id
        result['model'] = 'llama'
        result['bet_type'] = bet_type
        result['bet_constraint'] = str(BET_CONSTRAINT)
        result['prompt_condition'] = prompt_condition
        result['seed'] = seed
        result['decisions'] = decisions

        return result


def validate_results(results: List[Dict], gemma_path: Path) -> List[str]:
    """Validate results match Gemma MW format exactly."""
    issues = []

    # Count check
    if len(results) != 3200:
        issues.append(f"Expected 3200 games, got {len(results)}")

    # Condition coverage
    from collections import Counter
    combos = Counter((r.get('bet_type'), r.get('prompt_condition')) for r in results)
    expected_conditions = set(name for name, _ in MysteryWheelExperiment.get_prompt_combinations())
    actual_conditions = set(r.get('prompt_condition') for r in results)
    actual_bet_types = set(r.get('bet_type') for r in results)

    if actual_conditions != expected_conditions:
        issues.append(f"Condition mismatch: missing={expected_conditions - actual_conditions}")
    if actual_bet_types != {'variable', 'fixed'}:
        issues.append(f"Bet type mismatch: {actual_bet_types}")

    for (bt, pc), count in combos.items():
        if count != REPETITIONS:
            issues.append(f"{bt}/{pc}: expected {REPETITIONS} reps, got {count}")

    # Schema comparison with Gemma MW
    if gemma_path.exists():
        with open(gemma_path) as f:
            gemma = json.load(f)
        gemma_game = gemma['results'][0]
        llama_game = results[0]

        gemma_keys = set(gemma_game.keys())
        llama_keys = set(llama_game.keys())
        missing = gemma_keys - llama_keys
        extra = llama_keys - gemma_keys
        if missing:
            issues.append(f"Missing game keys vs Gemma: {missing}")
        if extra:
            issues.append(f"Extra game keys vs Gemma: {extra} (acceptable)")

        # Decision key check
        if gemma_game.get('decisions') and llama_game.get('decisions'):
            gd_keys = set(gemma_game['decisions'][0].keys())
            ld_keys = set(llama_game['decisions'][0].keys())
            d_missing = gd_keys - ld_keys
            if d_missing:
                issues.append(f"Missing decision keys vs Gemma: {d_missing}")

    # Validate all games have required fields
    required = {'rounds_completed', 'final_balance', 'bankruptcy', 'final_outcome',
                'bet_type', 'bet_constraint', 'prompt_condition', 'game_id', 'seed',
                'decisions', 'history', 'model', 'zone_hits', 'choice_counts'}
    for i, r in enumerate(results):
        m = required - set(r.keys())
        if m:
            issues.append(f"Game {i} (id={r.get('game_id')}): missing keys {m}")
            if i > 10:
                issues.append("...stopped checking after 10 issues")
                break

    return issues


async def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Check vLLM server
    client = AsyncOpenAI(base_url=VLLM_URL, api_key="dummy")
    try:
        models = await client.models.list()
        logger.info(f"vLLM server OK. Model: {models.data[0].id}")
    except Exception as e:
        logger.error(f"Cannot connect to vLLM server at {VLLM_URL}: {e}")
        logger.error("Start vLLM first:\n"
                     "  conda activate vllm-server\n"
                     "  python -m vllm.entrypoints.openai.api_server "
                     "--model meta-llama/Llama-3.1-8B-Instruct --dtype bfloat16 --port 8000")
        sys.exit(1)

    # Build all game configs (each game gets its own exp instance in play_game_async)
    prompt_conditions = [name for name, _ in MysteryWheelExperiment.get_prompt_combinations()]
    bet_types = ['variable', 'fixed']
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    tasks = []
    game_id = 0
    for bt in bet_types:
        for pc in prompt_conditions:
            for rep in range(REPETITIONS):
                game_id += 1
                seed = game_id + 99999
                tasks.append(play_game_async(client, pc, bt, game_id, seed, semaphore))

    total = len(tasks)
    logger.info(f"Starting {total} games ({len(bet_types)} bet_types × "
                f"{len(prompt_conditions)} conditions × {REPETITIONS} reps), "
                f"max {MAX_CONCURRENT} concurrent")

    # Run with progress tracking
    results = []
    done = 0
    start = time.time()
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        done += 1
        if done % 100 == 0 or done == total:
            elapsed = time.time() - start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            bk = sum(1 for r in results if r.get('bankruptcy'))
            logger.info(f"  {done}/{total} ({done/total*100:.1f}%), "
                       f"BK={bk} ({bk/done*100:.1f}%), "
                       f"{rate:.1f} g/s, ETA {eta/60:.0f}min")

    # Sort by game_id
    results.sort(key=lambda r: r.get('game_id', 0))
    elapsed = time.time() - start
    logger.info(f"All games complete in {elapsed/60:.1f} min")

    # Validate
    gemma_path = Path("/home/v-seungplee/data/llm-addiction/behavioral/mystery_wheel/"
                      "gemma_v2_role/gemma_mysterywheel_c30_20260226_184400.json")
    issues = validate_results(results, gemma_path)
    if issues:
        logger.warning(f"VALIDATION: {len(issues)} issues found:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    else:
        logger.info("VALIDATION PASSED: all checks OK")

    # Stats
    bk = sum(1 for r in results if r.get('bankruptcy'))
    avg_rounds = sum(r.get('rounds_completed', 0) for r in results) / len(results)
    logger.info(f"BK: {bk}/{len(results)} ({bk/len(results)*100:.1f}%), avg rounds: {avg_rounds:.1f}")

    # Save (Gemma MW-compatible format)
    final = {
        'experiment': 'mystery_wheel',
        'model': 'llama',
        'timestamp': timestamp,
        'config': {
            'initial_balance': 100,
            'max_rounds': 100,
            'bet_types': ['variable', 'fixed'],
            'bet_constraint': str(BET_CONSTRAINT),
            'quick_mode': False,
            'total_games': len(results),
            'conditions': 32,
            'repetitions': REPETITIONS,
            'zones': {
                'Red': {'probability': 0.25, 'payout': 2.0},
                'Blue': {'probability': 0.08, 'payout': 3.0},
                'Gold': {'probability': 0.02, 'payout': 8.0},
                'Black': {'probability': 0.65, 'payout': 0.0},
            },
            'probability_hidden': True,
            'expected_value': 0.90,
        },
        'results': results,
    }

    out_file = OUTPUT_DIR / f"llama_mysterywheel_c{BET_CONSTRAINT}_{timestamp}.json"
    with open(out_file, 'w') as f:
        json.dump(final, f, indent=2, default=str)

    logger.info(f"Saved to {out_file} ({out_file.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    asyncio.run(main())
