#!/usr/bin/env python3
"""
RQ2 Aligned-Factor Causal Steering Experiments
===============================================

Computes a shared BK axis across multiple gambling tasks via between-class
scatter matrix (Fisher-style), then steers LLM hidden states along that axis
to causally validate cross-task gambling risk representations.

Experiments:
  C  - Single-task BK direction on SM (replication/sanity)
  A  - Shared axis steering on LLaMA (3 tasks)
  B  - Shared axis steering on Gemma (3 tasks, IC-anchored)

Usage:
  python run_aligned_factor_steering.py --experiment c --model llama --gpu 0
  python run_aligned_factor_steering.py --experiment all --model llama --gpu 0
  python run_aligned_factor_steering.py --experiment all --model llama --gpu 0 --smoke
  python run_aligned_factor_steering.py --experiment a --model llama --gpu 0 --resume path/to/partial.json
"""

import os
import sys
import json
import time
import signal
import random
import logging
import argparse
import gc
from copy import deepcopy
from pathlib import Path
from datetime import datetime
from typing import Any, Optional

import numpy as np
import torch
from scipy.sparse.linalg import eigsh
from scipy.stats import norm, spearmanr
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm

# ---------------------------------------------------------------------------
# Path setup and imports from existing V12 code
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

from exact_behavioral_replay import (
    play_exact_behavioral_game,
    validate_behavioral_catalog,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("aligned_factor_steering")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ANALYSIS_ROOT = Path(
    os.environ.get(
        "LLM_ADDICTION_ANALYSIS_ROOT",
        "/home/v-seungplee/llm-addiction/sae_v3_analysis",
    )
)
DATA_ROOT = Path(
    os.environ.get(
        "LLM_ADDICTION_DATA_ROOT",
        "/home/v-seungplee/data/llm-addiction/sae_features_v3",
    )
)
RESULTS_DIR = ANALYSIS_ROOT / "results" / "json"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "llama": {"mid": "meta-llama/Llama-3.1-8B-Instruct", "hidden_dim": 4096},
    "gemma": {"mid": "google/gemma-2-9b-it", "hidden_dim": 3584},
}

TASK_DIR_MAP = {"sm": "slot_machine", "ic": "investment_choice", "mw": "mystery_wheel"}
LAYERS = [8, 12, 22, 25, 30]  # matches hidden_states_dp.npz layer ordering
EXP_LAYER = {"c": 25, "a": 25, "b": 12}
EXP_LAYER_IDX = {"c": 3, "a": 3, "b": 1}

ALPHA_VALUES = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]

TASK_SEED_OFFSET = {"sm": 0, "ic": 100000, "mw": 200000}

GAME_TIMEOUT_SECONDS = 600

# Game counts per experiment (full / smoke)
EXP_C_COUNTS = {
    "n_main": 200,
    "n_null_dirs": 100,
    "n_null_games": 50,
    "tasks": ["sm"],
}
EXP_A_COUNTS = {
    "n_main": 200,
    "n_null_dirs": 100,
    "n_null_games": 50,
    "tasks": ["sm", "ic", "mw"],
}
EXP_B_COUNTS = {
    "n_main": {"ic": 200, "sm": 500, "mw": 500},
    "n_null_dirs": {"ic": 50},
    "n_null_games": 50,
    "tasks": ["ic", "sm", "mw"],
}


# ============================================================
# Data Loading
# ============================================================


def load_discovery_states(
    model_name: str, task: str, layer_idx: int
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load hidden states for the discovery split (even game_ids).

    Returns:
        hs: (n_discovery, hidden_dim) float32 array
        bk_labels: (n_discovery,) bool array, True = bankruptcy
        meta: dict with counts and metadata
    """
    dp_path = DATA_ROOT / TASK_DIR_MAP[task] / model_name / "hidden_states_dp.npz"
    d = np.load(dp_path, allow_pickle=True)

    layers = list(d["layers"])
    target_layer = LAYERS[layer_idx]
    li = layers.index(target_layer)
    assert li == layer_idx, f"Layer ordering mismatch: LAYERS[{layer_idx}]={target_layer} at NPZ index {li}"

    game_ids = d["game_ids"]
    outcomes = d["game_outcomes"]

    discovery_mask = game_ids % 2 == 0
    hs = d["hidden_states"][discovery_mask, li, :].astype(np.float32)
    bk_labels = outcomes[discovery_mask] == "bankruptcy"

    meta = {
        "model": model_name,
        "task": task,
        "layer": int(LAYERS[layer_idx]),
        "layer_idx": layer_idx,
        "n_discovery": int(discovery_mask.sum()),
        "n_bk": int(bk_labels.sum()),
        "n_vs": int((~bk_labels).sum()),
        "data_path": str(dp_path),
    }
    logger.info(
        f"  Loaded {task}/{model_name} L{LAYERS[layer_idx]} discovery: "
        f"{meta['n_bk']} BK, {meta['n_vs']} VS (total {meta['n_discovery']})"
    )
    return hs, bk_labels, meta


def compute_per_task_direction(
    model_name: str, task: str, layer_idx: int
) -> tuple[np.ndarray, dict]:
    """Compute BK-VS centroid difference on discovery split.

    Returns:
        direction: (hidden_dim,) unit vector
        meta: dict with norm, counts
    """
    hs, bk_labels, load_meta = load_discovery_states(model_name, task, layer_idx)

    centroid_bk = hs[bk_labels].mean(axis=0)
    centroid_vs = hs[~bk_labels].mean(axis=0)
    d_t = centroid_bk - centroid_vs

    direction_norm = float(np.linalg.norm(d_t))
    d_unit = d_t / direction_norm

    meta = {
        **load_meta,
        "direction_norm": direction_norm,
    }
    logger.info(
        f"  Direction {task}/{model_name}: ||d||={direction_norm:.4f}, "
        f"n_BK={meta['n_bk']}, n_VS={meta['n_vs']}"
    )
    return d_unit, meta


def compute_shared_axis(
    model_name: str, tasks: list[str], layer_idx: int
) -> tuple[np.ndarray, dict]:
    """Compute shared BK axis across tasks via between-class scatter.

    For each task t:
      d_t = centroid(BK) - centroid(VS) on discovery split
      w_t = n_BK * n_VS / (n_BK + n_VS)
      S_B = sum_t  w_t * outer(d_t, d_t)
    u = top eigenvector of S_B.

    Sign convention: u dot (sum_t w_t * d_t) > 0, else negate.

    Returns:
        axis: (hidden_dim,) unit vector
        meta: dict with per-task info, eigenvalue, alignment cosines
    """
    per_task_info = []
    d_list = []
    w_list = []

    for task in tasks:
        hs, bk_labels, load_meta = load_discovery_states(model_name, task, layer_idx)
        centroid_bk = hs[bk_labels].mean(axis=0)
        centroid_vs = hs[~bk_labels].mean(axis=0)
        d_t = centroid_bk - centroid_vs

        n_bk = int(bk_labels.sum())
        n_vs = int((~bk_labels).sum())
        w_t = n_bk * n_vs / (n_bk + n_vs)

        d_list.append(d_t)
        w_list.append(w_t)
        per_task_info.append({
            "task": task,
            "n_bk": n_bk,
            "n_vs": n_vs,
            "weight": float(w_t),
            "direction_norm": float(np.linalg.norm(d_t)),
        })

    # Build between-class scatter matrix
    hidden_dim = d_list[0].shape[0]
    S_B = np.zeros((hidden_dim, hidden_dim), dtype=np.float64)
    for d_t, w_t in zip(d_list, w_list):
        d64 = d_t.astype(np.float64)
        S_B += w_t * np.outer(d64, d64)

    # Top eigenvector
    eigenvalues, eigenvectors = eigsh(S_B, k=1)
    u = eigenvectors[:, 0].astype(np.float32)

    # Sign convention: u dot weighted_mean_direction > 0
    weighted_d = np.zeros(hidden_dim, dtype=np.float64)
    for d_t, w_t in zip(d_list, w_list):
        weighted_d += w_t * d_t.astype(np.float64)
    if np.dot(u.astype(np.float64), weighted_d) < 0:
        u = -u

    # Normalize
    u = u / np.linalg.norm(u)

    # Compute alignment cosines
    for i, d_t in enumerate(d_list):
        d_unit = d_t / np.linalg.norm(d_t)
        cos_sim = float(np.dot(u, d_unit))
        per_task_info[i]["cos_with_shared"] = cos_sim

    meta = {
        "model": model_name,
        "tasks": tasks,
        "layer": int(LAYERS[layer_idx]),
        "layer_idx": layer_idx,
        "eigenvalue": float(eigenvalues[0]),
        "per_task": per_task_info,
        "shared_axis_norm": float(np.linalg.norm(u)),
    }
    cos_str = ", ".join(
        f"{t['cos_with_shared']:.3f}" for t in per_task_info
    )
    logger.info(
        f"  Shared axis ({model_name}, L{LAYERS[layer_idx]}): "
        f"eigenvalue={eigenvalues[0]:.4f}, cos=[{cos_str}]"
    )
    return u, meta


def generate_random_directions(
    dim: int, n_dirs: int, norm_val: float = 1.0, seed: int = 42
) -> np.ndarray:
    """Generate unit-norm random directions.

    Returns:
        (n_dirs, dim) float32 array, each row has norm = norm_val
    """
    rng = np.random.RandomState(seed)
    dirs = rng.randn(n_dirs, dim).astype(np.float32)
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs / norms * norm_val


# ============================================================
# Hook
# ============================================================


def make_hook(alpha: float, direction_tensor: torch.Tensor):
    """Create a forward hook that adds alpha * direction to the hidden state.

    Compatible with both plain tensor and tuple outputs.
    """
    def hook_fn(module, input, output):
        perturbation = alpha * direction_tensor.unsqueeze(0).unsqueeze(0)
        if isinstance(output, tuple):
            modified = output[0] + perturbation
            return (modified,) + output[1:]
        return output + perturbation

    return hook_fn


# ============================================================
# Model Loading
# ============================================================

def _disable_dynamo(model):
    try:
        import torch._dynamo as _dynamo
        _dynamo.config.suppress_errors = True
        _dynamo.disable(model)
        logger.info("torch._dynamo disabled for steering compatibility")
    except Exception:
        pass


def load_model(model_name: str, device: str) -> tuple:
    """Load model and tokenizer.

    Returns:
        (model, tokenizer) tuple
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mid = MODELS[model_name]["mid"]
    logger.info(f"Loading model: {mid}")

    tokenizer = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForCausalLM.from_pretrained(
        mid, torch_dtype=torch.bfloat16, device_map={"": device}
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    _disable_dynamo(model)
    logger.info(f"Model loaded: {mid}")
    return model, tokenizer


def get_layer_module(model, model_name: str, layer_num: int):
    """Get the transformer layer module for hooking."""
    return model.model.layers[layer_num]


def play_game_rich(
    model,
    tokenizer,
    device: str,
    hook_fn,
    layer_module,
    model_name: str,
    seed: int,
    task: str,
    game_index: int,
) -> dict:
    """Replay one behavioral-profile game under steering."""
    return play_exact_behavioral_game(
        model=model,
        tokenizer=tokenizer,
        device=device,
        hook_fn=hook_fn,
        layer_module=layer_module,
        model_name=model_name,
        task=task,
        game_index=game_index,
        seed=seed,
    )


def validate_behavioral_replay(model_name: str) -> bool:
    """Validate that behavioral condition catalogs are available and sane."""
    ok = True
    for task in ("sm", "ic", "mw"):
        summary = validate_behavioral_catalog(task, model_name)
        logger.info(
            "Behavioral catalog %s/%s: n=%d, prompt_conditions=%d, bet_types=%s, bet_constraints=%s",
            model_name,
            task,
            summary["n_games"],
            len(summary["prompt_conditions"]),
            summary["bet_types"],
            summary["bet_constraints"],
        )
        if summary["n_games"] <= 0:
            ok = False
    return ok


# ============================================================
# Direction Sweep
# ============================================================


def _play_one_game_safe(
    model, tokenizer, device, hook_fn, layer_module,
    model_name, task, game_index, seed,
):
    """Play one game with timeout. Thread-safe (no signal.alarm)."""
    return play_game_rich(
        model, tokenizer, device, hook_fn, layer_module,
        model_name, seed, task, game_index,
    )


def run_direction_sweep(
    model,
    tokenizer,
    device: str,
    layer_module,
    model_name: str,
    direction: np.ndarray,
    task: str,
    n_games: int,
    alpha_values: list[float],
    task_seed_offset: int,
    concurrent_games: int = 1,
) -> list[dict]:
    """Run games across all alpha values for a single direction+task.

    Seeding: seed = g + task_seed_offset  (alpha-independent!)
    concurrent_games > 1 uses ThreadPool for CPU-GPU overlap.

    Returns list of dicts, one per alpha value.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    direction_tensor = torch.tensor(
        direction, dtype=torch.bfloat16, device=device
    )
    results = []

    for alpha in alpha_values:
        hook_fn = make_hook(alpha, direction_tensor) if alpha != 0.0 else None
        bk_count = 0
        stop_count = 0
        total_wealth = 0.0
        total_iba = 0.0
        total_rounds = 0
        total_parse_fail = 0
        game_outcomes = []

        if concurrent_games <= 1:
            # Sequential mode (original)
            for g in range(n_games):
                seed = g + task_seed_offset
                try:
                    old_handler = signal.getsignal(signal.SIGALRM)
                    def timeout_handler(signum, frame):
                        raise TimeoutError("Game timed out")
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(GAME_TIMEOUT_SECONDS)

                    r = play_game_rich(
                        model, tokenizer, device, hook_fn, layer_module,
                        model_name, seed, task, g,
                    )
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)

                    if r["bk"]:
                        bk_count += 1
                    if r["stopped"]:
                        stop_count += 1
                    total_wealth += r["terminal_wealth"]
                    total_iba += r["mean_iba"]
                    total_rounds += r["n_rounds"]
                    total_parse_fail += r["parse_failures"]
                    game_outcomes.append({
                        "seed": seed,
                        "bk": r["bk"],
                        "stopped": r["stopped"],
                        "terminal_wealth": r["terminal_wealth"],
                        "mean_iba": round(r["mean_iba"], 4),
                        "n_rounds": r["n_rounds"],
                        "behavioral_condition": r.get("behavioral_condition"),
                    })
                except (TimeoutError, Exception) as e:
                    signal.alarm(0)
                    logger.warning(f"  Game error (alpha={alpha}, seed={seed}): {e}")
                    game_outcomes.append({
                        "seed": seed,
                        "bk": False,
                        "stopped": False,
                        "terminal_wealth": 0,
                        "mean_iba": 0.0,
                        "n_rounds": 0,
                        "error": str(e),
                    })
        else:
            # Concurrent mode: run games in thread pool for CPU-GPU overlap
            futures_map = {}
            with ThreadPoolExecutor(max_workers=concurrent_games) as pool:
                for g in range(n_games):
                    seed = g + task_seed_offset
                    future = pool.submit(
                        _play_one_game_safe,
                        model, tokenizer, device, hook_fn, layer_module,
                        model_name, task, g, seed,
                    )
                    futures_map[future] = (g, seed)

                for future in as_completed(futures_map):
                    g, seed = futures_map[future]
                    try:
                        r = future.result(timeout=GAME_TIMEOUT_SECONDS)
                        if r["bk"]:
                            bk_count += 1
                        if r["stopped"]:
                            stop_count += 1
                        total_wealth += r["terminal_wealth"]
                        total_iba += r["mean_iba"]
                        total_rounds += r["n_rounds"]
                        total_parse_fail += r["parse_failures"]
                        game_outcomes.append({
                            "seed": seed,
                            "bk": r["bk"],
                            "stopped": r["stopped"],
                            "terminal_wealth": r["terminal_wealth"],
                            "mean_iba": round(r["mean_iba"], 4),
                            "n_rounds": r["n_rounds"],
                            "behavioral_condition": r.get("behavioral_condition"),
                        })
                    except Exception as e:
                        logger.warning(f"  Game error (alpha={alpha}, seed={seed}): {e}")
                        game_outcomes.append({
                            "seed": seed,
                            "bk": False,
                            "stopped": False,
                            "terminal_wealth": 0,
                            "mean_iba": 0.0,
                            "n_rounds": 0,
                            "error": str(e),
                        })

        n_valid = max(len(game_outcomes), 1)
        result = {
            "alpha": alpha,
            "task": task,
            "n_games": n_games,
            "bk_count": bk_count,
            "stop_count": stop_count,
            "bk_rate": round(bk_count / n_valid, 4),
            "stop_rate": round(stop_count / n_valid, 4),
            "mean_terminal_wealth": round(total_wealth / n_valid, 2),
            "mean_iba": round(total_iba / n_valid, 4),
            "mean_rounds": round(total_rounds / n_valid, 2),
            "parse_failures": total_parse_fail,
            "game_outcomes": game_outcomes,
        }
        results.append(result)
        logger.info(
            f"    alpha={alpha:+5.1f} task={task}: "
            f"BK={bk_count}/{n_games} ({result['bk_rate']*100:.1f}%), "
            f"Stop={stop_count}/{n_games} ({result['stop_rate']*100:.1f}%), "
            f"Wealth={result['mean_terminal_wealth']:.0f}, "
            f"IBA={result['mean_iba']:.3f}"
        )

    return results


# ============================================================
# Statistical Tests
# ============================================================


def cochran_armitage_test(
    results_by_alpha: list[dict], alpha_values: list[float]
) -> dict:
    """Cochran-Armitage trend test for BK proportions across alpha doses.

    Manual implementation since statsmodels does not expose
    proportions_trend_test in v0.14.

    H0: no linear trend in proportion of BK across ordered alpha groups.
    Test statistic: Z = sum(w_i * (p_i - p_bar)) / SE
    """
    # Extract counts
    n_list = []
    x_list = []
    scores = []
    for res in results_by_alpha:
        n_i = res["n_games"]
        x_i = res["bk_count"]
        n_list.append(n_i)
        x_list.append(x_i)
        scores.append(res["alpha"])

    n_arr = np.array(n_list, dtype=np.float64)
    x_arr = np.array(x_list, dtype=np.float64)
    t_arr = np.array(scores, dtype=np.float64)

    N = n_arr.sum()
    X = x_arr.sum()
    p_bar = X / N

    # Weighted scores
    t_bar = np.dot(n_arr, t_arr) / N

    # Numerator
    numerator = np.dot(x_arr, t_arr) - X * t_bar

    # Denominator
    variance_t = np.dot(n_arr, (t_arr - t_bar) ** 2)
    denominator = np.sqrt(p_bar * (1 - p_bar) * variance_t)

    if denominator < 1e-12:
        return {"Z": 0.0, "p": 1.0, "trend_direction": "none"}

    Z = numerator / denominator
    p_value = 2 * norm.sf(abs(Z))  # two-sided

    return {
        "Z": round(float(Z), 4),
        "p": float(p_value),
        "trend_direction": "increasing" if Z > 0 else "decreasing",
    }


def ols_trend_test(values: list[float], alphas: list[float]) -> dict:
    """OLS regression of values on alpha with intercept.

    Returns slope, intercept, t-stat, p-value, R-squared.
    """
    X = sm.add_constant(np.array(alphas, dtype=np.float64))
    y = np.array(values, dtype=np.float64)

    model = OLS(y, X).fit()
    slope = float(model.params[1])
    intercept = float(model.params[0])
    t_stat = float(model.tvalues[1])
    p_value = float(model.pvalues[1])
    r_squared = float(model.rsquared)

    return {
        "slope": round(slope, 6),
        "intercept": round(intercept, 4),
        "t_stat": round(t_stat, 4),
        "p_value": p_value,
        "r_squared": round(r_squared, 4),
    }


def permutation_p_value(real_stat: float, null_stats: list[float]) -> float:
    """Two-sided permutation p-value with finite-sample correction."""
    null_arr = np.array(null_stats)
    if len(null_arr) == 0:
        return 1.0
    n_exceed = int((np.abs(null_arr) >= abs(real_stat)).sum())
    return float((n_exceed + 1) / (len(null_arr) + 1))


def compute_experiment_stats(
    sweep_results: list[dict], alpha_values: list[float]
) -> dict:
    """Compute comprehensive statistics from a direction sweep.

    Returns dict with Cochran-Armitage, OLS trend tests for BK rate,
    stop rate, wealth, IBA, and Spearman correlation.
    """
    stats_out = {}

    # Cochran-Armitage for BK
    stats_out["cochran_armitage_bk"] = cochran_armitage_test(
        sweep_results, alpha_values
    )

    # Extract rates for each metric
    bk_rates = [r["bk_rate"] for r in sweep_results]
    stop_rates = [r["stop_rate"] for r in sweep_results]
    wealths = [r["mean_terminal_wealth"] for r in sweep_results]
    ibas = [r["mean_iba"] for r in sweep_results]
    alphas = [r["alpha"] for r in sweep_results]

    # OLS trend tests
    stats_out["ols_bk_rate"] = ols_trend_test(bk_rates, alphas)
    stats_out["ols_stop_rate"] = ols_trend_test(stop_rates, alphas)
    stats_out["ols_wealth"] = ols_trend_test(wealths, alphas)
    stats_out["ols_iba"] = ols_trend_test(ibas, alphas)

    # Spearman correlations
    rho_bk, p_bk = spearmanr(alphas, bk_rates)
    rho_stop, p_stop = spearmanr(alphas, stop_rates)
    rho_wealth, p_wealth = spearmanr(alphas, wealths)
    rho_iba, p_iba = spearmanr(alphas, ibas)

    stats_out["spearman"] = {
        "bk_rate": {"rho": round(float(rho_bk), 4), "p": float(p_bk)},
        "stop_rate": {"rho": round(float(rho_stop), 4), "p": float(p_stop)},
        "wealth": {"rho": round(float(rho_wealth), 4), "p": float(p_wealth)},
        "iba": {"rho": round(float(rho_iba), 4), "p": float(p_iba)},
    }

    return stats_out


# ============================================================
# Phase 0: Sanity Check
# ============================================================


def phase0_sanity_check(
    model,
    tokenizer,
    device: str,
    model_name: str,
    layer_num: int,
    direction: np.ndarray,
    alpha_max: float = 2.0,
) -> dict:
    """Quick sanity check: run 20 games at alpha=0, +max, -max on SM.

    Verifies that the direction causes divergent BK rates.
    """
    logger.info("=" * 60)
    logger.info("Phase 0: Sanity Check")
    logger.info("=" * 60)

    layer_module = get_layer_module(model, model_name, layer_num)
    direction_tensor = torch.tensor(direction, dtype=torch.bfloat16, device=device)

    check_alphas = [0.0, -alpha_max, alpha_max]
    results = {}

    for alpha in check_alphas:
        hook_fn = make_hook(alpha, direction_tensor) if alpha != 0.0 else None
        bk_count = 0
        n_check = 20
        for g in range(n_check):
            seed = g + TASK_SEED_OFFSET["sm"]
            try:
                old_handler = signal.getsignal(signal.SIGALRM)
                def timeout_handler(signum, frame):
                    raise TimeoutError("Game timed out")
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(GAME_TIMEOUT_SECONDS)
                r = play_game_rich(
                    model,
                    tokenizer,
                    device,
                    hook_fn,
                    layer_module,
                    model_name,
                    seed,
                    "sm",
                    g,
                )
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                if r.get("bk", False):
                    bk_count += 1
            except Exception as e:
                signal.alarm(0)
                logger.warning(f"  Sanity game error: {e}")

        rate = bk_count / n_check
        results[str(alpha)] = {"bk": bk_count, "n": n_check, "bk_rate": round(rate, 4)}
        logger.info(f"  alpha={alpha:+5.1f}: BK={bk_count}/{n_check} ({rate*100:.1f}%)")

    # Check divergence
    rate_pos = results[str(alpha_max)]["bk_rate"]
    rate_neg = results[str(-alpha_max)]["bk_rate"]
    rate_base = results["0.0"]["bk_rate"]
    divergence = rate_pos - rate_neg

    passed = abs(divergence) > 0.0  # any divergence at all
    logger.info(
        f"  Sanity check {'PASSED' if passed else 'WARNING: no divergence'}: "
        f"BK(+{alpha_max})={rate_pos:.2f}, BK(-{alpha_max})={rate_neg:.2f}, "
        f"divergence={divergence:+.2f}"
    )

    return {
        "phase": 0,
        "check": "sanity",
        "results": results,
        "divergence": round(divergence, 4),
        "passed": passed,
    }


# ============================================================
# Experiment C: Single-task BK direction (SM only)
# ============================================================


def run_experiment_c(
    model,
    tokenizer,
    device: str,
    model_name: str,
    smoke: bool = False,
) -> dict:
    """Experiment C: Steer along single-task SM BK direction.

    Phase 1: Main sweep (7 alphas x n_main games)
    Phase 2: Null distribution (n_null_dirs random directions x n_null_games)
    """
    logger.info("=" * 60)
    logger.info("Experiment C: Single-Task BK Direction (SM)")
    logger.info("=" * 60)

    counts = deepcopy(EXP_C_COUNTS)
    if smoke:
        counts["n_main"] = max(counts["n_main"] // 10, 10)
        counts["n_null_dirs"] = max(counts["n_null_dirs"] // 10, 5)
        counts["n_null_games"] = max(counts["n_null_games"] // 10, 5)

    layer_idx = LAYERS.index(EXP_LAYER["c"])

    # Compute BK direction for SM
    direction, dir_meta = compute_per_task_direction(model_name, "sm", layer_idx)
    layer_module = get_layer_module(model, model_name, EXP_LAYER["c"])

    # Phase 1: Main sweep
    logger.info(f"\n--- Exp C Phase 1: Main sweep (n={counts['n_main']}) ---")
    main_results = run_direction_sweep(
        model, tokenizer, device, layer_module, model_name, direction,
        "sm", counts["n_main"], ALPHA_VALUES,
        TASK_SEED_OFFSET["sm"],
    )
    main_stats = compute_experiment_stats(main_results, ALPHA_VALUES)

    # Phase 2: Null distribution
    logger.info(
        f"\n--- Exp C Phase 2: Null distribution "
        f"(n_dirs={counts['n_null_dirs']}, n_games={counts['n_null_games']}) ---"
    )
    hidden_dim = MODELS[model_name]["hidden_dim"]
    random_dirs = generate_random_directions(
        hidden_dim, counts["n_null_dirs"], norm_val=1.0, seed=42
    )

    null_z_stats = []
    null_ols_slopes = []
    null_rho_stats = []
    null_summaries = []

    for d_idx in range(counts["n_null_dirs"]):
        rand_dir = random_dirs[d_idx]
        null_sweep = run_direction_sweep(
            model, tokenizer, device, layer_module, model_name, rand_dir,
            "sm", counts["n_null_games"], ALPHA_VALUES,
            TASK_SEED_OFFSET["sm"],
        )
        null_stats = compute_experiment_stats(null_sweep, ALPHA_VALUES)
        null_z_stats.append(null_stats["cochran_armitage_bk"]["Z"])
        null_ols_slopes.append(null_stats["ols_bk_rate"]["slope"])
        null_rho_stats.append(null_stats["spearman"]["bk_rate"]["rho"])
        null_summaries.append({
            "dir_idx": d_idx,
            "ca_z": null_stats["cochran_armitage_bk"]["Z"],
            "ols_slope": null_stats["ols_bk_rate"]["slope"],
            "rho": null_stats["spearman"]["bk_rate"]["rho"],
        })
        if (d_idx + 1) % 10 == 0:
            logger.info(f"  Null directions: {d_idx + 1}/{counts['n_null_dirs']}")

    # Permutation p-values
    real_z = main_stats["cochran_armitage_bk"]["Z"]
    real_slope = main_stats["ols_bk_rate"]["slope"]
    real_rho = main_stats["spearman"]["bk_rate"]["rho"]

    perm_p_z = permutation_p_value(real_z, null_z_stats)
    perm_p_slope = permutation_p_value(real_slope, null_ols_slopes)
    perm_p_rho = permutation_p_value(real_rho, null_rho_stats)

    result = {
        "experiment": "C",
        "model": model_name,
        "task": "sm",
        "layer": EXP_LAYER["c"],
        "direction_meta": dir_meta,
        "counts": counts,
        "alpha_values": ALPHA_VALUES,
        "main_sweep": _strip_game_outcomes(main_results),
        "main_stats": main_stats,
        "null_distribution": {
            "n_dirs": counts["n_null_dirs"],
            "n_games_per_dir": counts["n_null_games"],
            "summaries": null_summaries,
        },
        "permutation_tests": {
            "cochran_armitage_Z": {
                "real": real_z,
                "perm_p": perm_p_z,
            },
            "ols_slope": {
                "real": real_slope,
                "perm_p": perm_p_slope,
            },
            "spearman_rho": {
                "real": real_rho,
                "perm_p": perm_p_rho,
            },
        },
        "verdict": _verdict(main_stats, perm_p_z),
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"Exp C Result: {result['verdict']}")
    logger.info(
        f"  CA Z={real_z:.3f} (perm_p={perm_p_z:.4f}), "
        f"OLS slope={real_slope:.4f} (perm_p={perm_p_slope:.4f}), "
        f"rho={real_rho:.3f} (perm_p={perm_p_rho:.4f})"
    )
    logger.info("=" * 60)
    return result


# ============================================================
# Experiment A: Shared axis steering (LLaMA, 3 tasks)
# ============================================================


def run_experiment_a(
    model,
    tokenizer,
    device: str,
    model_name: str,
    shared_axis: np.ndarray,
    shared_meta: dict,
    smoke: bool = False,
    task_filter: str | None = None,
    n_games_override: int | None = None,
    concurrent_games: int = 1,
) -> dict:
    """Experiment A: Steer along shared axis across 3 tasks."""
    logger.info("=" * 60)
    logger.info("Experiment A: Shared Axis Steering (3 tasks)")
    logger.info("=" * 60)

    counts = deepcopy(EXP_A_COUNTS)
    if smoke:
        counts["n_main"] = max(counts["n_main"] // 10, 10)
        counts["n_null_dirs"] = max(counts["n_null_dirs"] // 10, 5)
        counts["n_null_games"] = max(counts["n_null_games"] // 10, 5)
    if n_games_override is not None:
        counts["n_main"] = n_games_override
    if task_filter:
        counts["tasks"] = [task_filter]
        logger.info(f"  task-filter active: running only [{task_filter}]")

    layer_module = get_layer_module(model, model_name, EXP_LAYER["a"])
    hidden_dim = MODELS[model_name]["hidden_dim"]

    # Phase 1: Main sweep for each task
    task_main_results = {}
    task_main_stats = {}
    for task in counts["tasks"]:
        logger.info(f"\n--- Exp A Phase 1: {task} (n={counts['n_main']}) ---")
        main_results = run_direction_sweep(
            model, tokenizer, device, layer_module, model_name, shared_axis,
            task, counts["n_main"], ALPHA_VALUES,
            TASK_SEED_OFFSET[task],
            concurrent_games=concurrent_games,
        )
        task_main_results[task] = main_results
        task_main_stats[task] = compute_experiment_stats(main_results, ALPHA_VALUES)

    # Phase 2: Null distribution
    logger.info(
        f"\n--- Exp A Phase 2: Null distribution "
        f"(n_dirs={counts['n_null_dirs']}, n_games={counts['n_null_games']}) ---"
    )
    random_dirs = generate_random_directions(
        hidden_dim, counts["n_null_dirs"], norm_val=1.0, seed=7777
    )

    null_per_task = {task: {"z": [], "slope": [], "rho": []} for task in counts["tasks"]}
    null_summaries = []

    for d_idx in range(counts["n_null_dirs"]):
        rand_dir = random_dirs[d_idx]
        dir_summary = {"dir_idx": d_idx}
        for task in counts["tasks"]:
            null_sweep = run_direction_sweep(
                model, tokenizer, device, layer_module, model_name, rand_dir,
                task, counts["n_null_games"], ALPHA_VALUES,
                TASK_SEED_OFFSET[task],
            )
            null_stats = compute_experiment_stats(null_sweep, ALPHA_VALUES)
            null_per_task[task]["z"].append(null_stats["cochran_armitage_bk"]["Z"])
            null_per_task[task]["slope"].append(null_stats["ols_bk_rate"]["slope"])
            null_per_task[task]["rho"].append(null_stats["spearman"]["bk_rate"]["rho"])
            dir_summary[f"{task}_z"] = null_stats["cochran_armitage_bk"]["Z"]
            dir_summary[f"{task}_slope"] = null_stats["ols_bk_rate"]["slope"]
        null_summaries.append(dir_summary)
        if (d_idx + 1) % 10 == 0:
            logger.info(f"  Null directions: {d_idx + 1}/{counts['n_null_dirs']}")

    # Permutation p-values per task
    perm_results = {}
    all_p_values = []
    for task in counts["tasks"]:
        real_z = task_main_stats[task]["cochran_armitage_bk"]["Z"]
        real_slope = task_main_stats[task]["ols_bk_rate"]["slope"]
        real_rho = task_main_stats[task]["spearman"]["bk_rate"]["rho"]

        perm_p_z = permutation_p_value(real_z, null_per_task[task]["z"])
        perm_p_slope = permutation_p_value(real_slope, null_per_task[task]["slope"])
        perm_p_rho = permutation_p_value(real_rho, null_per_task[task]["rho"])

        perm_results[task] = {
            "cochran_armitage_Z": {"real": real_z, "perm_p": perm_p_z},
            "ols_slope": {"real": real_slope, "perm_p": perm_p_slope},
            "spearman_rho": {"real": real_rho, "perm_p": perm_p_rho},
        }
        all_p_values.append(perm_p_z)

    # Holm correction across tasks
    if len(all_p_values) > 1:
        reject, corrected_p, _, _ = multipletests(all_p_values, method="holm")
        holm_results = {
            task: {"raw_p": all_p_values[i], "corrected_p": float(corrected_p[i]), "reject": bool(reject[i])}
            for i, task in enumerate(counts["tasks"])
        }
    else:
        holm_results = {
            counts["tasks"][0]: {"raw_p": all_p_values[0], "corrected_p": all_p_values[0], "reject": all_p_values[0] < 0.05}
        }

    result = {
        "experiment": "A",
        "model": model_name,
        "tasks": counts["tasks"],
        "layer": EXP_LAYER["a"],
        "shared_axis_meta": shared_meta,
        "counts": counts,
        "alpha_values": ALPHA_VALUES,
        "task_results": {
            task: {
                "main_sweep": _strip_game_outcomes(task_main_results[task]),
                "main_stats": task_main_stats[task],
            }
            for task in counts["tasks"]
        },
        "null_distribution": {
            "n_dirs": counts["n_null_dirs"],
            "n_games_per_dir": counts["n_null_games"],
            "summaries": null_summaries,
        },
        "permutation_tests": perm_results,
        "holm_correction": holm_results,
        "verdict": _verdict_multi(task_main_stats, perm_results, holm_results, counts["tasks"]),
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"Exp A Result: {result['verdict']}")
    for task in counts["tasks"]:
        pr = perm_results[task]
        logger.info(
            f"  {task}: Z={pr['cochran_armitage_Z']['real']:.3f} "
            f"(perm_p={pr['cochran_armitage_Z']['perm_p']:.4f}), "
            f"Holm={holm_results[task]['corrected_p']:.4f}"
        )
    logger.info("=" * 60)
    return result


# ============================================================
# Experiment B: Shared axis steering (Gemma, IC-anchored)
# ============================================================


def run_experiment_b(
    model,
    tokenizer,
    device: str,
    model_name: str,
    shared_axis: np.ndarray,
    shared_meta: dict,
    smoke: bool = False,
    task_filter: str | None = None,
    n_games_override: int | None = None,
    concurrent_games: int = 1,
) -> dict:
    """Experiment B: Steer along shared axis on Gemma.

    IC is the anchor task (more BK), SM/MW are supplementary (high n).
    """
    logger.info("=" * 60)
    logger.info("Experiment B: Shared Axis Steering (Gemma, IC-anchored)")
    logger.info("=" * 60)

    counts = deepcopy(EXP_B_COUNTS)
    if smoke:
        for k in counts["n_main"]:
            counts["n_main"][k] = max(counts["n_main"][k] // 10, 10)
        for k in counts["n_null_dirs"]:
            counts["n_null_dirs"][k] = max(counts["n_null_dirs"][k] // 10, 5)
        counts["n_null_games"] = max(counts["n_null_games"] // 10, 5)
    if n_games_override is not None:
        counts["n_main"] = {k: n_games_override for k in counts["n_main"]}
    if task_filter:
        counts["tasks"] = [task_filter]
        logger.info(f"  task-filter active: running only [{task_filter}]")

    layer_module = get_layer_module(model, model_name, EXP_LAYER["b"])
    hidden_dim = MODELS[model_name]["hidden_dim"]

    # Phase 1: Main sweep for each task
    task_main_results = {}
    task_main_stats = {}
    for task in counts["tasks"]:
        n_main = counts["n_main"][task] if isinstance(counts["n_main"], dict) else counts["n_main"]
        logger.info(f"\n--- Exp B Phase 1: {task} (n={n_main}) ---")
        main_results = run_direction_sweep(
            model, tokenizer, device, layer_module, model_name, shared_axis,
            task, n_main, ALPHA_VALUES,
            TASK_SEED_OFFSET[task],
            concurrent_games=concurrent_games,
        )
        task_main_results[task] = main_results
        task_main_stats[task] = compute_experiment_stats(main_results, ALPHA_VALUES)

    # Phase 2: Null distribution (IC only, since it's the anchor)
    null_tasks = list(counts["n_null_dirs"].keys())  # ["ic"]
    logger.info(
        f"\n--- Exp B Phase 2: Null distribution for {null_tasks} ---"
    )

    null_per_task = {task: {"z": [], "slope": [], "rho": []} for task in null_tasks}
    null_summaries = []

    for null_task in null_tasks:
        n_null = counts["n_null_dirs"][null_task]
        random_dirs = generate_random_directions(
            hidden_dim, n_null, norm_val=1.0, seed=9999
        )
        for d_idx in range(n_null):
            rand_dir = random_dirs[d_idx]
            null_sweep = run_direction_sweep(
                model, tokenizer, device, layer_module, model_name, rand_dir,
                null_task, counts["n_null_games"], ALPHA_VALUES,
                TASK_SEED_OFFSET[null_task],
            )
            null_stats = compute_experiment_stats(null_sweep, ALPHA_VALUES)
            null_per_task[null_task]["z"].append(null_stats["cochran_armitage_bk"]["Z"])
            null_per_task[null_task]["slope"].append(null_stats["ols_bk_rate"]["slope"])
            null_per_task[null_task]["rho"].append(null_stats["spearman"]["bk_rate"]["rho"])
            null_summaries.append({
                "task": null_task,
                "dir_idx": d_idx,
                "z": null_stats["cochran_armitage_bk"]["Z"],
                "slope": null_stats["ols_bk_rate"]["slope"],
            })
            if (d_idx + 1) % 10 == 0:
                logger.info(f"  Null dirs ({null_task}): {d_idx + 1}/{n_null}")

    # Permutation p-values
    perm_results = {}
    all_p_values = []
    tasks_with_perm = []
    for task in counts["tasks"]:
        real_z = task_main_stats[task]["cochran_armitage_bk"]["Z"]
        real_slope = task_main_stats[task]["ols_bk_rate"]["slope"]
        real_rho = task_main_stats[task]["spearman"]["bk_rate"]["rho"]

        if task in null_per_task and len(null_per_task[task]["z"]) > 0:
            perm_p_z = permutation_p_value(real_z, null_per_task[task]["z"])
            perm_p_slope = permutation_p_value(real_slope, null_per_task[task]["slope"])
            perm_p_rho = permutation_p_value(real_rho, null_per_task[task]["rho"])
        else:
            # No null distribution for this task; use parametric p only
            perm_p_z = task_main_stats[task]["cochran_armitage_bk"]["p"]
            perm_p_slope = task_main_stats[task]["ols_bk_rate"]["p_value"]
            perm_p_rho = task_main_stats[task]["spearman"]["bk_rate"]["p"]

        perm_results[task] = {
            "cochran_armitage_Z": {"real": real_z, "perm_p": perm_p_z},
            "ols_slope": {"real": real_slope, "perm_p": perm_p_slope},
            "spearman_rho": {"real": real_rho, "perm_p": perm_p_rho},
        }
        all_p_values.append(perm_p_z)
        tasks_with_perm.append(task)

    # Holm correction across all tasks
    if len(all_p_values) > 1:
        reject, corrected_p, _, _ = multipletests(all_p_values, method="holm")
        holm_results = {
            tasks_with_perm[i]: {
                "raw_p": all_p_values[i],
                "corrected_p": float(corrected_p[i]),
                "reject": bool(reject[i]),
            }
            for i in range(len(tasks_with_perm))
        }
    else:
        holm_results = {
            tasks_with_perm[0]: {
                "raw_p": all_p_values[0],
                "corrected_p": all_p_values[0],
                "reject": all_p_values[0] < 0.05,
            }
        }

    result = {
        "experiment": "B",
        "model": model_name,
        "tasks": counts["tasks"],
        "layer": EXP_LAYER["b"],
        "shared_axis_meta": shared_meta,
        "counts": _serialize_counts(counts),
        "alpha_values": ALPHA_VALUES,
        "task_results": {
            task: {
                "main_sweep": _strip_game_outcomes(task_main_results[task]),
                "main_stats": task_main_stats[task],
            }
            for task in counts["tasks"]
        },
        "null_distribution": {
            "null_tasks": null_tasks,
            "n_games_per_dir": counts["n_null_games"],
            "summaries": null_summaries,
        },
        "permutation_tests": perm_results,
        "holm_correction": holm_results,
        "verdict": _verdict_multi(task_main_stats, perm_results, holm_results, counts["tasks"]),
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"Exp B Result: {result['verdict']}")
    for task in counts["tasks"]:
        pr = perm_results[task]
        logger.info(
            f"  {task}: Z={pr['cochran_armitage_Z']['real']:.3f} "
            f"(perm_p={pr['cochran_armitage_Z']['perm_p']:.4f}), "
            f"Holm={holm_results.get(task, {}).get('corrected_p', 'N/A')}"
        )
    logger.info("=" * 60)
    return result


# ============================================================
# Verdict Helpers
# ============================================================


def _verdict(main_stats: dict, perm_p_z: float) -> str:
    """Determine verdict for single-task experiment."""
    ca_p = main_stats["cochran_armitage_bk"]["p"]
    rho = main_stats["spearman"]["bk_rate"]["rho"]

    if perm_p_z < 0.05 and ca_p < 0.05 and rho > 0.5:
        return "BK_SPECIFIC_DOSE_RESPONSE"
    elif ca_p < 0.05 and rho > 0.3:
        return "DOSE_RESPONSE_ONLY"
    elif ca_p < 0.05:
        return "TREND_SIGNIFICANT"
    else:
        return "NOT_SIGNIFICANT"


def _verdict_multi(
    task_stats: dict,
    perm_results: dict,
    holm_results: dict,
    tasks: list[str],
) -> str:
    """Determine verdict for multi-task experiment."""
    n_sig = sum(1 for t in tasks if holm_results.get(t, {}).get("reject", False))
    n_trend = sum(
        1
        for t in tasks
        if task_stats[t]["cochran_armitage_bk"]["p"] < 0.05
    )

    if n_sig == len(tasks):
        return f"ALL_{len(tasks)}_TASKS_SIGNIFICANT_HOLM"
    elif n_sig > 0:
        return f"{n_sig}/{len(tasks)}_TASKS_SIGNIFICANT_HOLM"
    elif n_trend == len(tasks):
        return f"ALL_{len(tasks)}_TASKS_TREND_UNCORRECTED"
    elif n_trend > 0:
        return f"{n_trend}/{len(tasks)}_TASKS_TREND_UNCORRECTED"
    else:
        return "NOT_SIGNIFICANT"


# ============================================================
# Utility Helpers
# ============================================================


def _strip_game_outcomes(sweep_results: list[dict]) -> list[dict]:
    """Remove per-game outcome details to keep JSON manageable."""
    stripped = []
    for r in sweep_results:
        r_copy = {k: v for k, v in r.items() if k != "game_outcomes"}
        stripped.append(r_copy)
    return stripped


def _serialize_counts(counts: dict) -> dict:
    """Ensure counts dict is JSON-serializable (convert any non-standard types)."""
    out = {}
    for k, v in counts.items():
        if isinstance(v, dict):
            out[k] = {str(kk): vv for kk, vv in v.items()}
        elif isinstance(v, list):
            out[k] = v
        else:
            out[k] = v
    return out


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _save_results(result: dict, experiment: str, model_name: str) -> Path:
    """Save result dict to JSON with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"aligned_steering_{experiment}_{model_name}_{timestamp}.json"
    out_path = RESULTS_DIR / filename
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, cls=_NumpyEncoder)
    logger.info(f"Saved results to {out_path}")
    return out_path


def _load_resume(resume_path: str) -> dict:
    """Load partial results from a previous run."""
    with open(resume_path) as f:
        data = json.load(f)
    logger.info(f"Resumed from {resume_path}")
    return data


# ============================================================
# Main Orchestrator
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="RQ2 Aligned-Factor Causal Steering Experiments"
    )
    parser.add_argument(
        "--experiment",
        choices=["a", "b", "c", "all"],
        required=True,
        help="Which experiment to run: a, b, c, or all",
    )
    parser.add_argument(
        "--model",
        choices=["gemma", "llama"],
        required=True,
        help="Model to use: gemma or llama",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device index (sets CUDA_VISIBLE_DEVICES)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test: reduce all game counts by 10x",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to partial results JSON to resume from",
    )
    parser.add_argument(
        "--phase",
        choices=["0", "1", "2", "all"],
        default="all",
        help="Which phase to run (0=sanity, 1=main, 2=null). Default: all",
    )
    parser.add_argument(
        "--task-filter",
        type=str,
        default=None,
        help="Run only this task in multi-task experiments (sm, ic, mw). "
             "Enables splitting Exp A/B across GPUs.",
    )
    parser.add_argument(
        "--n-games",
        type=int,
        default=None,
        help="Override n_main games (default: use experiment-specific counts)",
    )
    parser.add_argument(
        "--concurrent-games",
        type=int,
        default=1,
        help="Run N games in parallel via ThreadPool (2-3 recommended). "
             "Overlaps CPU prep with GPU inference for ~30%% speedup.",
    )
    args = parser.parse_args()

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device} (GPU {args.gpu})")

    # Load model
    model, tokenizer = load_model(args.model, device)

    all_results = {}
    timestamp_start = datetime.now().isoformat()

    if not validate_behavioral_replay(args.model):
        logger.error(
            "FATAL: behavioral replay validation failed. "
            "Check raw behavioral catalogs before running experiments."
        )
        return
    logger.info("Behavioral replay validation passed.")

    try:
        if args.experiment in ("c", "all"):
            # Phase 0: Sanity check
            layer_idx = LAYERS.index(EXP_LAYER["c"])
            direction_c, dir_meta_c = compute_per_task_direction(
                args.model, "sm", layer_idx
            )

            if args.phase in ("0", "all"):
                sanity = phase0_sanity_check(
                    model, tokenizer, device, args.model,
                    EXP_LAYER["c"], direction_c, alpha_max=2.0,
                )
                all_results["phase0_sanity"] = sanity

            # ── Experiment C (Phase 1, LLaMA L25 SM positive control) ──
            if args.phase in ("1", "2", "all"):
                result_c = run_experiment_c(
                    model, tokenizer, device, args.model,
                    smoke=args.smoke,
                )
                all_results["experiment_c"] = result_c
                _save_results(result_c, "C", args.model)

            # ── GATE: C must pass before A proceeds ──
            if args.experiment == "all":
                c_perm_p = (
                    all_results.get("experiment_c", {})
                    .get("permutation_tests", {})
                    .get("cochran_armitage_Z", {})
                    .get("perm_p", 1.0)
                )
                if c_perm_p >= 0.05:
                    logger.error(
                        f"GATE FAILED: Exp C perm_p={c_perm_p:.3f} >= 0.05. "
                        f"L25 is not intervention-sensitive. Stopping."
                    )
                    all_results["gate_decision"] = "STOP"
                    _save_results(all_results, "gate_stop", args.model)
                    return
                logger.info(f"GATE PASSED: Exp C perm_p={c_perm_p:.3f} < 0.05")
                all_results["gate_decision"] = "PASS"

        # ── Experiment A: Shared-axis steering, LLaMA L25 ──
        if args.experiment in ("a", "all") and args.model == "llama":
            layer_a = EXP_LAYER["a"]
            layer_idx_a = EXP_LAYER_IDX["a"]
            shared_axis, shared_meta = compute_shared_axis(
                "llama", ["sm", "ic", "mw"], layer_idx_a
            )
            if args.phase in ("0", "all") and "phase0_sanity_a" not in all_results:
                sanity = phase0_sanity_check(
                    model, tokenizer, device, "llama",
                    layer_a, shared_axis, alpha_max=2.0,
                )
                all_results["phase0_sanity_a"] = sanity

            if args.phase in ("1", "2", "all"):
                result_a = run_experiment_a(
                    model, tokenizer, device, "llama",
                    shared_axis, shared_meta,
                    smoke=args.smoke,
                    task_filter=args.task_filter,
                    n_games_override=args.n_games,
                    concurrent_games=args.concurrent_games,
                )
                all_results["experiment_a"] = result_a
                _save_results(result_a, "A", "llama")

        # ── Experiment B: Shared-axis steering, Gemma L12 ──
        if args.experiment in ("b", "all") and args.model == "gemma":
            layer_b = EXP_LAYER["b"]
            layer_idx_b = EXP_LAYER_IDX["b"]
            shared_axis_b, shared_meta_b = compute_shared_axis(
                "gemma", ["ic", "sm", "mw"], layer_idx_b
            )
            if args.phase in ("0", "all") and "phase0_sanity_b" not in all_results:
                sanity = phase0_sanity_check(
                    model, tokenizer, device, "gemma",
                    layer_b, shared_axis_b, alpha_max=2.0,
                )
                all_results["phase0_sanity_b"] = sanity

            if args.phase in ("1", "2", "all"):
                result_b = run_experiment_b(
                    model, tokenizer, device, "gemma",
                    shared_axis_b, shared_meta_b,
                    smoke=args.smoke,
                    task_filter=args.task_filter,
                    n_games_override=args.n_games,
                    concurrent_games=args.concurrent_games,
                )
                all_results["experiment_b"] = result_b
                _save_results(result_b, "B", "gemma")

    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Saving partial results...")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        # Save combined results
        combined = {
            "script": "run_aligned_factor_steering.py",
            "args": vars(args),
            "timestamp_start": timestamp_start,
            "timestamp_end": datetime.now().isoformat(),
            "results": all_results,
        }
        _save_results(combined, f"combined_{args.experiment}", args.model)

        # Cleanup
        del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Done.")


if __name__ == "__main__":
    main()
