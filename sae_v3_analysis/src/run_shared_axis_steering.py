"""RQ2 Shared-Axis Causal Validation.

Intent
------
  Paper's RQ2 claims a rank-1 common risk axis exists across 3 tasks, extracted
  via between-class scatter of (centroid(BK) - centroid(VS)) per task. This is
  currently observational evidence only.

  This experiment tests the causal reality of the axis: add alpha * shared_axis
  to hidden state during inference, see if bankruptcy behavior moves in all 3
  tasks consistently.

Hypothesis
----------
  Strong: all 3 tasks show monotonic dose-response with same sign under shared
    axis steering (real shared causal axis).
  Medium: 2/3 tasks pass permutation test, at least some sign consistency.
  Null: no task distinguishable from random directions.

Verification
------------
  Per (model, task): n=200 games x 7 alphas x Phase 1 main sweep, plus Phase 2
  permutation against 100 random unit directions (n=50 games per direction,
  7 alphas each). Report Cochran-Armitage Z, Spearman rho, perm_p, verdict.

  Pass criterion: perm_p < 0.05. Sign interpretation done after all 3 tasks done.

Implementation: reuses run_aligned_factor_steering.compute_shared_axis and
run_direction_sweep. Only difference vs Exp A/B/C: direction is shared axis
instead of task-specific BK direction.
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

REPO_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_SRC))

from run_aligned_factor_steering import (  # noqa: E402
    ALPHA_VALUES,
    LAYERS,
    _NumpyEncoder,
    compute_experiment_stats,
    compute_shared_axis,
    generate_random_directions,
    get_layer_module,
    load_model,
    permutation_p_value,
    run_direction_sweep,
    validate_behavioral_replay,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("shared_axis_steering")


ANALYSIS_ROOT = Path(os.environ.get("LLM_ADDICTION_ANALYSIS_ROOT", "/scratch/llm_addiction/sae_v3_analysis"))
RESULTS_DIR = ANALYSIS_ROOT / "results"
JSON_DIR = RESULTS_DIR / "json"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
LOG_DIR = RESULTS_DIR / "logs"
for d in (JSON_DIR, CHECKPOINT_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Per-model dominant shared-axis layer (from paper Table)
SHARED_LAYER = {"gemma": 12, "llama": 25}


def run_shared_axis_condition(
    model, tokenizer, device: str, model_name: str, task: str,
    n_main: int, n_null_games: int, n_null_dirs: int, smoke: bool,
    concurrent_games: int = 1,
) -> dict:
    """Run shared-axis steering for one (model, task) cell."""
    tasks = ["sm", "ic", "mw"]
    layer = SHARED_LAYER[model_name]
    layer_idx = LAYERS.index(layer)
    layer_module = get_layer_module(model, model_name, layer)

    # Extract shared axis from ALL 3 tasks
    shared_axis, axis_meta = compute_shared_axis(model_name, tasks, layer_idx)
    dim = shared_axis.shape[0]

    alpha_values = list(ALPHA_VALUES)
    if smoke:
        alpha_values = [-2.0, 0.0, 2.0]
        n_main = min(n_main, 20)
        n_null_games = min(n_null_games, 10)
        n_null_dirs = min(n_null_dirs, 4)

    # Phase 1: main sweep on shared axis for this task
    # Use run_direction_sweep's built-in checkpoint mechanism for resume support.
    logger.info("=" * 60)
    logger.info(f"[shared_axis/{model_name}/{task}] Phase 1: Main sweep")
    logger.info("=" * 60)

    ckpt_key = f"shared_axis_{model_name}"
    main_sweep = run_direction_sweep(
        model, tokenizer, device, layer_module, model_name, shared_axis, task,
        n_games=n_main, alpha_values=alpha_values,
        task_seed_offset=30000,
        concurrent_games=concurrent_games,
        checkpoint_key=ckpt_key,
    )
    main_sweep = sorted(main_sweep, key=lambda x: x.get("alpha", 0))
    main_stats = compute_experiment_stats(main_sweep, alpha_values)

    # Phase 2: null distribution — 100 random unit directions, same task
    logger.info("=" * 60)
    logger.info(f"[shared_axis/{model_name}/{task}] Phase 2: Null distribution")
    logger.info("=" * 60)
    random_dirs = generate_random_directions(dim, n_null_dirs, norm_val=1.0, seed=42)
    null_summaries = []
    for i, rd in enumerate(random_dirs):
        rd = rd / (np.linalg.norm(rd) + 1e-12)
        rd_results = run_direction_sweep(
            model, tokenizer, device, layer_module, model_name, rd, task,
            n_games=n_null_games, alpha_values=alpha_values,
            task_seed_offset=40000 + i * 1000,
            concurrent_games=concurrent_games,
        )
        rd_stats = compute_experiment_stats(rd_results, alpha_values)
        null_summaries.append({
            "dir_idx": i,
            "ca_z": rd_stats["cochran_armitage_bk"]["Z"],
            "ols_slope": rd_stats["ols_bk_rate"]["slope"],
            "rho": rd_stats["spearman"]["bk_rate"]["rho"],
        })

    null_ca = [r["ca_z"] for r in null_summaries
               if r["ca_z"] is not None and not np.isnan(r["ca_z"])]
    null_slope = [r["ols_slope"] for r in null_summaries
                  if r["ols_slope"] is not None and not np.isnan(r["ols_slope"])]
    null_rho = [r["rho"] for r in null_summaries
                if r["rho"] is not None and not np.isnan(r["rho"])]

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
        "experiment": "shared_axis",
        "model": model_name,
        "task": task,
        "layer": layer,
        "axis_meta": axis_meta,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gemma", "llama"], required=True)
    parser.add_argument("--task", choices=["sm", "ic", "mw"], required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--n-main", type=int, default=200)
    parser.add_argument("--n-null-games", type=int, default=50)
    parser.add_argument("--n-null-dirs", type=int, default=100)
    parser.add_argument("--concurrent-games", type=int, default=2)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"device: {device} (GPU {args.gpu})")

    model, tokenizer = load_model(args.model, device)
    if not validate_behavioral_replay(args.model):
        logger.error("Behavioral replay validation failed.")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = run_shared_axis_condition(
        model, tokenizer, device, args.model, args.task,
        n_main=args.n_main, n_null_games=args.n_null_games, n_null_dirs=args.n_null_dirs,
        smoke=args.smoke, concurrent_games=args.concurrent_games,
    )

    out = {
        "script": "run_shared_axis_steering.py",
        "args": vars(args),
        "timestamp": ts,
        "result": result,
    }
    outpath = JSON_DIR / f"shared_axis_{args.model}_{args.task}_{ts}.json"
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2, cls=_NumpyEncoder)
    logger.info(f"Saved: {outpath}")

    pt = result["permutation_tests"]
    ms = result["main_stats"]
    logger.info(
        f"[shared_axis/{args.model}/{args.task}] "
        f"rho={ms['spearman']['bk_rate']['rho']:.3f}, "
        f"CA Z={ms['cochran_armitage_bk']['Z']:.2f}, "
        f"perm_p(rho)={pt['spearman_rho']['perm_p']:.4f}, "
        f"perm_p(CA)={pt['cochran_armitage_Z']['perm_p']:.4f}"
    )


if __name__ == "__main__":
    main()
