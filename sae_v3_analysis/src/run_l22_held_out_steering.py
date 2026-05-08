"""L22 held-out steering with paired-seed evaluation (Plan v5).

This script runs LLaMA / Gemma BK-direction steering at frozen layers and
held-out seed offsets (g >= 1000) to address codex agentic-reviewer concern
#1 ("n=50, single task, single model, single layer").

Pre-registered design (Plan v5):
- Direction = mean(h_bk) - mean(h_stop) on existing discovery NPZ (game_ids
  even, g < 200). +α applied to L22 residual moves toward bk (matches body
  §4.4 dose-response: bk-rate increases with α). No re-selection here.
- Evaluation seeds: g >= 1000 (cell-specific offsets H1/H2/H3/H4/H5).
- Alpha grid: {-2, -1, -0.5, 0, +0.5, +1, +2} (immutable).
- Primary statistical test computed offline from per-game outputs.

Outputs: one JSON per cell to RESULTS_DIR with full per-game data.

Example invocations (deployed via SSH on running AMLT nodes):
    # H1 partial sweep on GPU 1 (alphas -2, -1, -0.5)
    python run_l22_held_out_steering.py \
        --cell H1 --task sm --model llama --layer 22 \
        --n-games 200 --g-offset 1000 \
        --alphas -2.0 -1.0 -0.5 \
        --gpu 1 --output /scratch/l22_runs/h1_g1.json

    # H5 axis null on GPU 1 (one random direction index per process; loop indices 0..29 across processes)
    python run_l22_held_out_steering.py \
        --cell H5 --task sm --model llama --layer 22 \
        --n-games 20 --g-offset 2000 \
        --alphas 1.0 \
        --random-axis --random-axis-index 0 \
        --gpu 1 --output /scratch/l22_runs/h5_rand0_g1.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Use the existing pipeline's components. We stay in the same package directory
# so internal imports work.
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from run_aligned_factor_steering import (  # type: ignore
    DATA_ROOT,
    LAYERS,
    TASK_DIR_MAP,
    TASK_SEED_OFFSET,
    GAME_TIMEOUT_SECONDS,
    load_model,
    get_layer_module,
    make_hook,
    play_game_rich,
    _disable_dynamo,
)

logger = logging.getLogger("run_l22_held_out_steering")
logger.setLevel(logging.INFO)
_h = logging.StreamHandler()
_h.setFormatter(logging.Formatter("%(asctime)s [l22] %(message)s", "%H:%M:%S"))
logger.addHandler(_h)


# ============================================================
# Frozen plan v5 constants
# ============================================================

ALPHA_GRID_DEFAULT = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
RANDOM_AXIS_RNG_SEED = 2026042700  # PCG64 seed for H5 directions


# ============================================================
# Discovery direction (no re-selection: load from existing NPZ)
# ============================================================


def load_discovery_states(model_name: str, task: str, layer: int):
    """Load even-game-id discovery hidden states at the requested layer."""
    if layer not in LAYERS:
        raise ValueError(
            f"Layer {layer} not in NPZ-stored LAYERS={LAYERS}. "
            f"Add to discovery extraction first."
        )
    layer_idx_in_npz = LAYERS.index(layer)
    dp_path = DATA_ROOT / TASK_DIR_MAP[task] / model_name / "hidden_states_dp.npz"
    if not dp_path.exists():
        raise FileNotFoundError(f"Discovery NPZ not found at {dp_path}")
    d = np.load(dp_path, allow_pickle=True)
    layers_in_npz = list(d["layers"])
    li = layers_in_npz.index(layer)
    if li != layer_idx_in_npz:
        raise RuntimeError(
            f"Layer ordering mismatch: requested L{layer} at NPZ idx {layer_idx_in_npz}, "
            f"actually at NPZ idx {li}"
        )
    game_ids = d["game_ids"]
    outcomes = d["game_outcomes"]
    discovery_mask = game_ids % 2 == 0
    hs = d["hidden_states"][discovery_mask, li, :].astype(np.float32)
    bk_labels = outcomes[discovery_mask] == "bankruptcy"
    return hs, bk_labels, {
        "model": model_name,
        "task": task,
        "layer": int(layer),
        "n_discovery_total": int(discovery_mask.sum()),
        "n_bk_discovery": int(bk_labels.sum()),
        "n_vs_discovery": int((~bk_labels).sum()),
        "data_path": str(dp_path),
    }


def compute_bk_direction(
    model_name: str, task: str, layer: int
) -> tuple[np.ndarray, dict]:
    """ĥ_BK = mean(h_bk) - mean(h_stop).

    Sign convention matches the existing pipeline (run_aligned_factor_steering.
    compute_per_task_direction, line 179). Body §4.4 reports +α giving higher
    bk rate (34→52% across α=-2→+2, OLS slope +0.042 on bk_rate vs α), which
    is consistent with the bk - stop direction since the hook adds α·d̂ to
    the residual.
    """
    hs, bk_labels, meta = load_discovery_states(model_name, task, layer)
    centroid_bk = hs[bk_labels].mean(axis=0)
    centroid_vs = hs[~bk_labels].mean(axis=0)
    d = centroid_bk - centroid_vs  # +α moves toward bk (increase bk rate)
    norm = float(np.linalg.norm(d))
    return d / norm, {**meta, "direction_norm": norm, "sign_convention": "bk - stop"}


def make_random_unit_directions(
    n: int, dim: int, rng_seed: int = RANDOM_AXIS_RNG_SEED
) -> np.ndarray:
    """Isotropic random unit directions in `dim` dimensions (frozen RNG)."""
    g = np.random.Generator(np.random.PCG64(rng_seed))
    raw = g.standard_normal(size=(n, dim)).astype(np.float32)
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    return raw


# ============================================================
# Held-out evaluation sweep
# ============================================================


def play_one_game_with_timeout(
    model, tokenizer, device, hook_fn, layer_module,
    model_name, seed, task, game_index, timeout_seconds: int
):
    old_handler = signal.getsignal(signal.SIGALRM)
    def timeout_handler(signum, frame):
        raise TimeoutError("Game timed out")
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        return play_game_rich(
            model, tokenizer, device, hook_fn, layer_module,
            model_name, seed, task, game_index,
        )
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def run_alpha_sweep_for_direction(
    *,
    model,
    tokenizer,
    device: str,
    layer_module,
    direction: np.ndarray,
    direction_label: str,
    task: str,
    model_name: str,
    n_games: int,
    g_offset: int,
    alphas: list[float],
    timeout_seconds: int = GAME_TIMEOUT_SECONDS,
    checkpoint_path=None,
    cell_meta=None,
) -> list[dict]:
    """Run paired-seed alpha sweep for one direction.

    Game index = g_offset + i for i in [0, n_games). Same indices reused across
    every alpha to keep paired design.

    Per-alpha checkpoint: if checkpoint_path is provided, the result list is
    written to disk after EACH alpha completes. On a fresh start, any
    already-done alphas are skipped. cell_meta is the top-level metadata dict
    that wraps results_by_alpha; it must include schema_version, cell, etc.
    """
    direction_tensor = torch.tensor(
        direction.astype(np.float32), dtype=torch.bfloat16, device=device
    )
    task_seed_offset = TASK_SEED_OFFSET[task]
    # Resume from checkpoint if exists
    out = []
    done_alphas = set()
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            prev = json.load(open(checkpoint_path))
            for r in prev.get("results_by_alpha", []):
                if r.get("n_valid", 0) > 0:
                    out.append(r)
                    done_alphas.add(float(r["alpha"]))
            if done_alphas:
                logger.info(f"Resume: skipping already-done alphas {sorted(done_alphas)}")
        except Exception as e:
            logger.warning(f"Failed to read checkpoint at {checkpoint_path}: {e}")

    def _flush():
        if not checkpoint_path:
            return
        if cell_meta is None:
            return
        snap = dict(cell_meta)
        snap["results_by_alpha"] = out
        tmp = checkpoint_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(snap, f, indent=2)
        os.replace(tmp, checkpoint_path)

    for alpha in alphas:
        if alpha in done_alphas:
            continue
        hook_fn = make_hook(alpha, direction_tensor) if alpha != 0.0 else None
        bk_count = 0
        stop_count = 0
        total_w = 0.0
        total_iba = 0.0
        total_rounds = 0
        total_parse_fail = 0
        n_errors = 0
        outcomes = []
        t0 = time.time()
        for i in range(n_games):
            g = g_offset + i
            seed = g + task_seed_offset
            try:
                r = play_one_game_with_timeout(
                    model, tokenizer, device, hook_fn, layer_module,
                    model_name, seed, task, g, timeout_seconds,
                )
                if r["bk"]:
                    bk_count += 1
                if r["stopped"]:
                    stop_count += 1
                total_w += r["terminal_wealth"]
                total_iba += r["mean_iba"]
                total_rounds += r["n_rounds"]
                total_parse_fail += r["parse_failures"]
                outcomes.append({
                    "g": g,
                    "seed": seed,
                    "ok": True,
                    "bk": bool(r["bk"]),
                    "stopped": bool(r["stopped"]),
                    "terminal_wealth": float(r["terminal_wealth"]),
                    "mean_iba": round(float(r["mean_iba"]), 4),
                    "n_rounds": int(r["n_rounds"]),
                })
            except Exception as e:
                logger.warning(
                    f"  game error (alpha={alpha}, g={g}, seed={seed}): {e}"
                )
                n_errors += 1
                outcomes.append({
                    "g": g,
                    "seed": seed,
                    "ok": False,
                    "error": str(e),
                })
        # Denominator excludes errors — error games are NOT counted as bk=False;
        # they are reported separately (n_valid + n_errors = n_games).
        n_valid = n_games - n_errors
        denom = max(n_valid, 1)
        wall = time.time() - t0
        out.append({
            "direction_label": direction_label,
            "alpha": alpha,
            "task": task,
            "n_games": n_games,
            "n_valid": n_valid,
            "n_errors": n_errors,
            "g_offset": g_offset,
            "bk_count": bk_count,
            "stop_count": stop_count,
            "bk_rate": round(bk_count / denom, 4),
            "stop_rate": round(stop_count / denom, 4),
            "mean_terminal_wealth": round(total_w / denom, 2),
            "mean_iba": round(total_iba / denom, 4),
            "mean_rounds": round(total_rounds / denom, 2),
            "parse_failures": total_parse_fail,
            "wall_seconds": round(wall, 1),
            "game_outcomes": outcomes,
        })
        logger.info(
            f"  alpha={alpha:+5.1f} {task} {direction_label}: "
            f"BK={bk_count}/{n_games} ({out[-1]['bk_rate']*100:.1f}%), "
            f"Stop={stop_count}/{n_games}, wall={wall:.0f}s"
        )
        # Per-alpha checkpoint flush (preemption-resilient).
        _flush()
    return out


# ============================================================
# CLI
# ============================================================


CELL_CONSTRAINTS = {
    # cell -> frozen (task, model, layer, n_games, g_offset (relative), random_axis, allowed_alphas)
    # NPZ inspection (LLaMA): SM gid in [1,3200], MW gid in [1,3200], IC gid in [1,1600].
    # Held-out g_offset is set above max(gid) per task to guarantee no overlap.
    "H1":    dict(task="sm", model="llama", layer=22, n_games=200, g_offset=4000,
                  random_axis=False, allowed_alphas=set(ALPHA_GRID_DEFAULT)),
    "H2":    dict(task="mw", model="llama", layer=22, n_games=100, g_offset=4000,
                  random_axis=False, allowed_alphas=set(ALPHA_GRID_DEFAULT)),
    # H3 (Gemma SM L22) DROPPED: Gemma discovery NPZ not present on running
    # nodes (sae_features_v3/*/gemma/hidden_states_dp.npz absent). Cross-model
    # extension is future work.
    # H4 (LLaMA SM L23) DROPPED: L23 not in pre-extracted NPZ layers.
    # H5: two modes on the same seed window (g_offset=5000):
    "H5":    dict(task="sm", model="llama", layer=22, n_games=20, g_offset=5000,
                  random_axis=None, allowed_alphas={1.0}),
    # Plan v6 RQ4 strengthening (Apr 27 fanout): multi-layer SM + within-task IC/MW.
    # Same g_offset families as H1/H2 (held-out), n_games preserves test power.
    "H6":    dict(task="sm", model="llama", layer=25, n_games=100, g_offset=4000,
                  random_axis=False, allowed_alphas=set(ALPHA_GRID_DEFAULT)),
    "H7":    dict(task="sm", model="llama", layer=30, n_games=100, g_offset=4000,
                  random_axis=False, allowed_alphas=set(ALPHA_GRID_DEFAULT)),
    "H8":    dict(task="sm", model="llama", layer=12, n_games=100, g_offset=4000,
                  random_axis=False, allowed_alphas=set(ALPHA_GRID_DEFAULT)),
    "H9":    dict(task="sm", model="llama", layer=8,  n_games=100, g_offset=4000,
                  random_axis=False, allowed_alphas=set(ALPHA_GRID_DEFAULT)),
    "H10":   dict(task="ic", model="llama", layer=22, n_games=100, g_offset=2000,
                  random_axis=False, allowed_alphas=set(ALPHA_GRID_DEFAULT)),
    "H11":   dict(task="ic", model="llama", layer=25, n_games=100, g_offset=2000,
                  random_axis=False, allowed_alphas=set(ALPHA_GRID_DEFAULT)),
    "H12":   dict(task="ic", model="llama", layer=30, n_games=100, g_offset=2000,
                  random_axis=False, allowed_alphas=set(ALPHA_GRID_DEFAULT)),
    "H13":   dict(task="mw", model="llama", layer=25, n_games=100, g_offset=4000,
                  random_axis=False, allowed_alphas=set(ALPHA_GRID_DEFAULT)),
    "H14":   dict(task="mw", model="llama", layer=30, n_games=100, g_offset=4000,
                  random_axis=False, allowed_alphas=set(ALPHA_GRID_DEFAULT)),
}


def enforce_cell_constraints(args) -> None:
    """Enforce frozen Plan v5. Refuses any deviation that would compromise pre-registration.

    --g-offset is interpreted as the relative (intra-task) index; the absolute
    random seed adds TASK_SEED_OFFSET[task] inside run_alpha_sweep_for_direction.
    """
    cons = CELL_CONSTRAINTS[args.cell]
    mismatches = []
    if args.task != cons["task"]:
        mismatches.append(f"task {args.task} != frozen {cons['task']}")
    if args.model != cons["model"]:
        mismatches.append(f"model {args.model} != frozen {cons['model']}")
    if args.layer != cons["layer"]:
        mismatches.append(f"layer {args.layer} != frozen L{cons['layer']}")
    if args.n_games != cons["n_games"]:
        mismatches.append(f"n_games {args.n_games} != frozen {cons['n_games']}")
    if args.g_offset != cons["g_offset"]:
        mismatches.append(
            f"g_offset {args.g_offset} != frozen {cons['g_offset']} "
            f"(--g-offset is RELATIVE; absolute seeds are g_offset + i + TASK_SEED_OFFSET[{args.task}])"
        )
    if cons["random_axis"] is not None and args.random_axis != cons["random_axis"]:
        mismatches.append(
            f"random_axis {args.random_axis} != frozen {cons['random_axis']}"
        )
    forbidden = [a for a in args.alphas if a not in cons["allowed_alphas"]]
    if forbidden:
        mismatches.append(f"alphas {forbidden} not in frozen set {sorted(cons['allowed_alphas'])}")
    # Alpha SUBSET allowed for GPU partitioning.
    if mismatches:
        raise ValueError(
            "Plan v5 cell constraint violation:\n  - " + "\n  - ".join(mismatches)
        )


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--cell", required=True, choices=list(CELL_CONSTRAINTS.keys()),
                   help="frozen Plan v5 cell (H4 was dropped: L23 not in extracted NPZ)")
    p.add_argument("--task", required=True, choices=["sm", "ic", "mw"])
    p.add_argument("--model", required=True, choices=["llama", "gemma"])
    p.add_argument("--layer", type=int, required=True,
                   help="hidden-state layer to hook (must be in pre-extracted NPZ layers)")
    p.add_argument("--n-games", type=int, required=True)
    p.add_argument("--g-offset", type=int, required=True,
                   help="held-out game index start; must be >= 1000 to avoid discovery overlap")
    p.add_argument("--alphas", type=float, nargs="+",
                   default=list(ALPHA_GRID_DEFAULT),
                   help="alpha grid (default: full pre-registered grid)")
    p.add_argument("--gpu", type=int, default=0,
                   help="logical GPU index inside CUDA_VISIBLE_DEVICES (always cuda:0 after masking)")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--random-axis", action="store_true",
                   help="H5 mode: use a random unit direction (matched norm) instead of BK direction")
    p.add_argument("--random-axis-index", type=int, default=None,
                   help="when --random-axis is set, which random direction index in [0, 30) to use")
    p.add_argument("--timeout-seconds", type=int, default=GAME_TIMEOUT_SECONDS)
    p.add_argument("--smoke", action="store_true",
                   help="bypass n_games and alpha-grid constraints for a 1-game smoke test; "
                        "results from --smoke runs are NOT used in the paper")
    args = p.parse_args()

    if not args.smoke:
        enforce_cell_constraints(args)
    else:
        logger.warning("SMOKE MODE: cell constraints bypassed; output is for validation only")

    if args.g_offset < 1000:
        raise ValueError("--g-offset < 1000 violates Plan v5 held-out frozen seeds")
    if args.layer not in LAYERS:
        raise ValueError(f"--layer must be in {LAYERS}; got {args.layer}")

    # GPU pinning: CUDA_VISIBLE_DEVICES masks to a single physical device, so
    # the script always uses cuda:0 inside the process. Set BEFORE importing
    # any torch CUDA state (we already imported torch above, but no CUDA call
    # has been made since). Caller must invoke with the desired physical GPU
    # via env var or set the env var before launching.
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda:0"

    logger.info(
        f"START cell={args.cell} task={args.task} model={args.model} "
        f"L{args.layer} n={args.n_games} g_offset={args.g_offset} "
        f"alphas={args.alphas} gpu={args.gpu}"
    )

    # 1) compute BK direction (or random unit direction for H5)
    bk_dir, dir_meta = compute_bk_direction(args.model, args.task, args.layer)
    direction_norm = dir_meta["direction_norm"]

    if args.random_axis:
        if args.random_axis_index is None:
            raise ValueError("--random-axis requires --random-axis-index in [0, 30)")
        n_random = 30
        dim = bk_dir.shape[0]
        random_dirs = make_random_unit_directions(n_random, dim, RANDOM_AXIS_RNG_SEED)
        idx = int(args.random_axis_index)
        if not 0 <= idx < n_random:
            raise ValueError(f"--random-axis-index must be in [0, {n_random})")
        # The hook adds α * direction (assumed unit length, matching the BK
        # reference which is already unit). Random axes are also unit so that
        # `α` has the same per-step magnitude meaning across BK and random.
        direction_eff = random_dirs[idx]
        direction_label = f"random_{idx}"
    else:
        direction_eff = bk_dir  # already unit length
        direction_label = "bk_minus_stop"

    # 2) load model
    logger.info(f"Loading model {args.model} on {device} ...")
    model, tokenizer = load_model(args.model, device)
    layer_module = get_layer_module(model, args.model, args.layer)

    # 3) sweep alphas (per-alpha checkpoint to args.output for preemption resume)
    cell_meta = {
        "schema_version": "l22_held_out_v2",
        "plan_version": "v5",
        "cell": args.cell,
        "task": args.task,
        "model": args.model,
        "layer": args.layer,
        "n_games": args.n_games,
        "g_offset": args.g_offset,
        "alphas": list(args.alphas),
        "direction_label": direction_label,
        "direction_meta": {
            **dir_meta,
            "random_axis": bool(args.random_axis),
            "random_axis_index": (int(args.random_axis_index)
                                  if args.random_axis else None),
            "random_axis_rng_seed": (RANDOM_AXIS_RNG_SEED
                                     if args.random_axis else None),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = run_alpha_sweep_for_direction(
        model=model,
        tokenizer=tokenizer,
        device=device,
        layer_module=layer_module,
        direction=direction_eff,
        direction_label=direction_label,
        task=args.task,
        model_name=args.model,
        n_games=args.n_games,
        g_offset=args.g_offset,
        alphas=list(args.alphas),
        timeout_seconds=args.timeout_seconds,
        checkpoint_path=str(args.output),
        cell_meta=cell_meta,
    )

    # 4) write final output (idempotent with last per-alpha flush)
    out = {
        **cell_meta,
        "direction_meta": {
            **dir_meta,
            "random_axis": bool(args.random_axis),
            "random_axis_index": (int(args.random_axis_index)
                                  if args.random_axis else None),
            "random_axis_rng_seed": (RANDOM_AXIS_RNG_SEED
                                     if args.random_axis else None),
            # When random_axis: dir_meta still describes the BK reference for
            # the rank-test denominator. The actual direction added per step is
            # a *unit* random axis from the frozen RNG; α has the same per-step
            # magnitude meaning across BK and random axes.
        },
        "results_by_alpha": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
