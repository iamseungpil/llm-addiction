"""D1: Gemma baseline inflation test — disentangle "floor effect" from "axis absence".

Intent:
  Exp B (Gemma SM/MW) produced only "Floor" verdicts: direction specificity failed
  because baseline bankruptcy rate is too low (SM 2.7%, MW 1.7%). This leaves the
  question open — is the aligned BK axis absent in Gemma SM/MW, or is behavior just
  too flat for α<0 to lower BK further and for α>0 to increase it?

Hypothesis:
  If the axis exists but baseline behavior is floored, then steering on a subset
  of games selected to have a higher baseline BK rate (variable betting + G-prompts)
  should recover direction specificity (permutation p < 0.05).

Design:
  Three subset-steering runs, all using the canonical per-task BK direction computed
  from the full hidden_states_dp.npz (no change to axis):
    1. D1-test: Gemma SM, subset = variable bet + G-containing prompt  (baseline ~10.4%)
    2. D1-neg:  Gemma SM, subset = variable bet + no-G prompt          (baseline ~0.5%)
    3. D1-pos:  LLaMA SM, subset = variable bet + G-containing prompt  (baseline ~80.4%)

  Each run performs Phase 1 main sweep over α ∈ {-2, -1, -0.5, 0, +0.5, +1, +2}
  with n=200 games per α, and Phase 2 permutation against 100 random unit directions.

Verification:
  D1-pos: perm_p < 0.05 expected (replicates Exp A LLaMA SM pattern; sanity check
    that the subset filter does not break the pipeline).
  D1-test: perm_p < 0.05 would confirm floor-effect hypothesis — same axis works
    once baseline variation is restored.
  D1-neg: perm_p > 0.05 expected — low baseline subset should still be floored.

This script is standalone: it imports from run_aligned_factor_steering without
modifying that file's experiment functions. The only upstream change is the new
`bet_filter` / `prompt_contains` / `prompt_excludes` keyword arguments added to
`play_exact_behavioral_game` in exact_behavioral_replay.py (default None = no filter).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Repo paths
REPO_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_SRC))

from run_aligned_factor_steering import (  # noqa: E402
    LAYERS,
    _NumpyEncoder,
    compute_experiment_stats,
    compute_per_task_direction,
    generate_random_directions,
    get_layer_module,
    load_model,
    make_hook,
    permutation_p_value,
    validate_behavioral_replay,
)
from exact_behavioral_replay import (  # noqa: E402
    get_filtered_catalog,
    play_exact_behavioral_game,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("d1_gemma_inflation")


ANALYSIS_ROOT = Path(os.environ.get("LLM_ADDICTION_ANALYSIS_ROOT", "/scratch/llm_addiction/sae_v3_analysis"))
RESULTS_DIR = ANALYSIS_ROOT / "results"
JSON_DIR = RESULTS_DIR / "json"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
LOG_DIR = RESULTS_DIR / "logs"
for d in (JSON_DIR, CHECKPOINT_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Subset steering loop
# -----------------------------------------------------------------------------

def _subset_layer_for(model_name: str, task: str) -> int:
    """Use the same per-model layer as Exp A (LLaMA L25) and Exp B (Gemma L12)."""
    if model_name == "llama":
        return 25
    if model_name == "gemma":
        return 12
    raise ValueError(f"Unknown model: {model_name}")


def _summarize_games(games: list[dict]) -> dict:
    n = len(games)
    if n == 0:
        return {"n_games": 0, "bk_count": 0, "stop_count": 0,
                "bk_rate": 0.0, "stop_rate": 0.0,
                "mean_terminal_wealth": 0.0, "mean_iba": 0.0,
                "parse_failures": 0}
    bk = sum(1 for g in games if g.get("bk"))
    stop = sum(1 for g in games if g.get("stopped"))
    return {
        "n_games": n,
        "bk_count": bk,
        "stop_count": stop,
        "bk_rate": bk / n,
        "stop_rate": stop / n,
        "mean_terminal_wealth": float(np.mean([g.get("terminal_wealth", 0) for g in games])),
        "mean_iba": float(np.mean([g.get("mean_iba", 0.0) for g in games])),
        "parse_failures": sum(g.get("parse_failures", 0) for g in games),
    }


def run_subset_sweep(
    model,
    tokenizer,
    device: str,
    layer_module,
    model_name: str,
    direction: np.ndarray,
    task: str,
    alpha_values: list[float],
    n_games: int,
    bet_filter: str | None,
    prompt_contains: str | None,
    prompt_excludes: str | None,
    seed_offset: int = 0,
) -> list[dict]:
    """Run α-sweep on filtered subset for one (model, task) pair.

    Returns list of per-α summaries with (alpha, n_games, bk_count, bk_rate, ...).
    """
    logger.info(
        "Subset sweep: model=%s task=%s filters=(bet=%s, contains=%s, excludes=%s), "
        "n_games=%d, alphas=%s",
        model_name, task, bet_filter, prompt_contains, prompt_excludes, n_games, alpha_values,
    )
    cat = get_filtered_catalog(task, model_name,
                               bet_filter=bet_filter,
                               prompt_contains=prompt_contains,
                               prompt_excludes=prompt_excludes)
    logger.info("  filtered catalog size: %d unique conditions", len(cat))

    direction_tensor = torch.as_tensor(direction, dtype=torch.float32, device=device)

    results: list[dict] = []
    # Deterministic filter-key seed (Python `hash` is non-deterministic without PYTHONHASHSEED)
    filter_key = f"{bet_filter or ''}|{prompt_contains or ''}|{prompt_excludes or ''}"
    filter_seed = sum(ord(ch) * (i + 1) for i, ch in enumerate(filter_key)) & 0xFFFF

    for alpha in alpha_values:
        hook_fn = make_hook(alpha, direction_tensor) if alpha != 0.0 else None
        games: list[dict] = []
        for g in range(n_games):
            seed = g + seed_offset + filter_seed
            try:
                r = play_exact_behavioral_game(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    hook_fn=hook_fn,
                    layer_module=layer_module,
                    model_name=model_name,
                    task=task,
                    game_index=g,
                    seed=seed,
                    bet_filter=bet_filter,
                    prompt_contains=prompt_contains,
                    prompt_excludes=prompt_excludes,
                )
                games.append(r)
            except Exception as e:
                logger.warning("  game %d failed: %s", g, str(e)[:120])
                games.append({"skipped": True, "parse_failures": 1})
        summary = _summarize_games(games)
        summary["alpha"] = alpha
        summary["task"] = task
        results.append(summary)
        logger.info(
            "    alpha=%+.1f task=%s: BK=%d/%d (%.1f%%), Stop=%d/%d (%.1f%%), "
            "Wealth=%.0f, IBA=%.3f",
            alpha, task, summary["bk_count"], summary["n_games"], 100 * summary["bk_rate"],
            summary["stop_count"], summary["n_games"], 100 * summary["stop_rate"],
            summary["mean_terminal_wealth"], summary["mean_iba"],
        )
    return results


# -----------------------------------------------------------------------------
# Phase 1 + Phase 2 with filter
# -----------------------------------------------------------------------------

def run_condition(
    condition_id: str,
    model,
    tokenizer,
    device: str,
    model_name: str,
    task: str,
    bet_filter: str | None,
    prompt_contains: str | None,
    prompt_excludes: str | None,
    n_main: int,
    n_null_games: int,
    n_null_dirs: int,
    smoke: bool,
) -> dict:
    """Run one D1 condition end to end: direction, main sweep, null permutation."""
    layer = _subset_layer_for(model_name, task)
    layer_idx = LAYERS.index(layer)
    layer_module = get_layer_module(model, model_name, layer)

    direction, dir_meta = compute_per_task_direction(model_name, task, layer_idx)
    dim = direction.shape[0]

    alpha_values = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    if smoke:
        alpha_values = [-2.0, 0.0, 2.0]
        n_main = min(n_main, 20)
        n_null_games = min(n_null_games, 10)
        n_null_dirs = min(n_null_dirs, 4)

    # Phase 1: main sweep on BK direction under filter
    logger.info("=" * 60)
    logger.info("[%s] Phase 1: Main sweep", condition_id)
    logger.info("=" * 60)
    main_sweep = run_subset_sweep(
        model, tokenizer, device, layer_module, model_name, direction, task,
        alpha_values=alpha_values, n_games=n_main,
        bet_filter=bet_filter, prompt_contains=prompt_contains, prompt_excludes=prompt_excludes,
        seed_offset=10000,
    )
    main_stats = compute_experiment_stats(main_sweep, alpha_values)

    # Phase 2: null distribution — random unit directions, same filter
    logger.info("=" * 60)
    logger.info("[%s] Phase 2: Null distribution (%d random directions)", condition_id, n_null_dirs)
    logger.info("=" * 60)
    random_dirs = generate_random_directions(dim, n_null_dirs, norm_val=1.0, seed=42)
    null_summaries: list[dict] = []
    for i, rd in enumerate(random_dirs):
        rd = rd / (np.linalg.norm(rd) + 1e-12)
        rd_results = run_subset_sweep(
            model, tokenizer, device, layer_module, model_name, rd, task,
            alpha_values=alpha_values, n_games=n_null_games,
            bet_filter=bet_filter, prompt_contains=prompt_contains, prompt_excludes=prompt_excludes,
            seed_offset=20000 + i * 1000,
        )
        rd_stats = compute_experiment_stats(rd_results, alpha_values)
        null_summaries.append({
            "dir_idx": i,
            "ca_z": rd_stats["cochran_armitage_bk"]["Z"],
            "ols_slope": rd_stats["ols_bk_rate"]["slope"],
            "rho": rd_stats["spearman"]["bk_rate"]["rho"],
        })

    # Permutation p values (permutation_p_value takes abs() internally)
    null_ca = [r["ca_z"] for r in null_summaries if r["ca_z"] is not None and not np.isnan(r["ca_z"])]
    null_slope = [r["ols_slope"] for r in null_summaries if r["ols_slope"] is not None and not np.isnan(r["ols_slope"])]
    null_rho = [r["rho"] for r in null_summaries if r["rho"] is not None and not np.isnan(r["rho"])]

    real_ca = main_stats["cochran_armitage_bk"]["Z"]
    real_slope = main_stats["ols_bk_rate"]["slope"]
    real_rho = main_stats["spearman"]["bk_rate"]["rho"]

    perm_tests = {
        "cochran_armitage_Z": {"real": real_ca,
                                "perm_p": permutation_p_value(real_ca, null_ca)},
        "ols_slope": {"real": real_slope,
                       "perm_p": permutation_p_value(real_slope, null_slope)},
        "spearman_rho": {"real": real_rho,
                          "perm_p": permutation_p_value(real_rho, null_rho)},
    }

    return {
        "condition_id": condition_id,
        "model": model_name,
        "task": task,
        "layer": layer,
        "filters": {
            "bet_filter": bet_filter,
            "prompt_contains": prompt_contains,
            "prompt_excludes": prompt_excludes,
        },
        "filtered_catalog_size": len(get_filtered_catalog(
            task, model_name, bet_filter, prompt_contains, prompt_excludes)),
        "direction_meta": dir_meta,
        "alpha_values": alpha_values,
        "counts": {"n_main": n_main, "n_null_games": n_null_games, "n_null_dirs": n_null_dirs},
        "main_sweep": main_sweep,
        "main_stats": main_stats,
        "null_distribution": {
            "n_dirs": n_null_dirs,
            "n_games_per_dir": n_null_games,
            "summaries": null_summaries,
        },
        "permutation_tests": perm_tests,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

CONDITIONS = [
    {"id": "D1-test",
     "model": "gemma", "task": "sm",
     "bet_filter": "variable", "prompt_contains": "G", "prompt_excludes": None},
    {"id": "D1-neg",
     "model": "gemma", "task": "sm",
     "bet_filter": "variable", "prompt_contains": None, "prompt_excludes": "G"},
    {"id": "D1-pos",
     "model": "llama", "task": "sm",
     "bet_filter": "variable", "prompt_contains": "G", "prompt_excludes": None},
]


def main():
    parser = argparse.ArgumentParser(description="D1 Gemma baseline inflation test")
    parser.add_argument("--condition", choices=[c["id"] for c in CONDITIONS], required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny test: 3 alphas, n=20 main, n=10 null, 4 null dirs")
    parser.add_argument("--n-main", type=int, default=200)
    parser.add_argument("--n-null-games", type=int, default=50)
    parser.add_argument("--n-null-dirs", type=int, default=100)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    cond = next(c for c in CONDITIONS if c["id"] == args.condition)
    logger.info("Condition: %s", cond)

    model, tokenizer = load_model(cond["model"], device)
    if not validate_behavioral_replay(cond["model"]):
        logger.error("Behavioral replay validation failed. Abort.")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = run_condition(
        condition_id=cond["id"],
        model=model, tokenizer=tokenizer, device=device,
        model_name=cond["model"], task=cond["task"],
        bet_filter=cond["bet_filter"],
        prompt_contains=cond["prompt_contains"],
        prompt_excludes=cond["prompt_excludes"],
        n_main=args.n_main, n_null_games=args.n_null_games, n_null_dirs=args.n_null_dirs,
        smoke=args.smoke,
    )

    out = {
        "script": "run_d1_gemma_inflation.py",
        "args": vars(args),
        "timestamp": ts,
        "result": result,
    }
    outpath = JSON_DIR / f"d1_{cond['id']}_{cond['model']}_{cond['task']}_{ts}.json"
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2, cls=_NumpyEncoder)
    logger.info("Saved: %s", outpath)

    # Short verdict line
    pt = result["permutation_tests"]
    ms = result["main_stats"]
    logger.info(
        "[%s] VERDICT: Spearman ρ=%.3f, CA Z=%.2f, perm_p(ρ)=%.4f, perm_p(CA)=%.4f",
        cond["id"],
        ms["spearman"]["bk_rate"]["rho"],
        ms["cochran_armitage_bk"]["Z"],
        pt["spearman_rho"]["perm_p"],
        pt["cochran_armitage_Z"]["perm_p"],
    )


if __name__ == "__main__":
    main()
