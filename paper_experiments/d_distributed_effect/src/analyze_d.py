"""Track D analysis — paired bootstrap of (Δ_G_random_K − Δ_G_top_K).

Pre-registered primary verdict (Plan v4 §5; frozen in d_config.yaml):
  At K=primary_K (50): lower bound of 95% paired-bootstrap CI on
  Δ_G_random_K − Δ_G_top_K  > 0  → distributed-effect branch passes.

Bootstrap design
================
Each top-K run yields one Δ_G_top scalar. The N_random_baselines_per_K random
runs yield {Δ_G_random_r}_{r=1..N}. We resample with replacement from the
*per-fold* R² lists that each run wrote out, paired across (random_r, top)
within fold index. This pairs the bootstrap by GroupKFold fold (which is a
deterministic, game_id-grouped partition; resampling fold indices is therefore
resampling by game_id at fold granularity — `bootstrap_resample_unit: game_id`
in the config).

For each bootstrap iter b:
  1. Resample fold indices i_1, …, i_F with replacement from {0..F-1}.
  2. For each random replicate r, recompute
        Δ_G_random_r,b = mean_+G_r[i_1..i_F] − mean_-G_r[i_1..i_F]
     and analogously Δ_G_top,b from the top-K run's per-fold R²s.
  3. Δb = mean_r(Δ_G_random_r,b) − Δ_G_top,b.
The 95% CI is the (2.5, 97.5) quantiles of {Δb}_{b=1..1000}.

Outputs (per model):
  results/d_analysis_{paradigm_dir}_{indicator}_L{layer}.json
    {
      "model": ...,
      "K_values": [10, 50, 100],
      "by_K": {
         "10": {"delta_top": ..., "delta_random_mean": ..., "ci_low": ..., "ci_high": ..., "primary_pass": null},
         "50": {... "primary_pass": true|false},
         "100": {...}
      },
      "primary_K": 50,
      "primary_pass": true|false,
      "outcome_branch": "D-passes" | "D-mixed" | "D-fails"
    }

Outcome classification (cf. claim_surgery_D_outcome_branches.md):
  D-passes:  K=50 primary CI lower bound > 0 AND K=10 + K=100 also positive
  D-mixed:   K=50 passes but at least one of K=10 / K=100 fails
  D-fails:   K=50 primary CI includes 0 or is below
"""
from __future__ import annotations
import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
DEFAULT_CONFIG = ROOT / "configs" / "d_config.yaml"

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger("d.analyze")


# ----- per-run JSON loader -------------------------------------------------


_RUN_PATTERN = re.compile(
    r"^(?P<model>[a-z0-9]+)_(?P<paradigm>[a-z_]+)_L(?P<layer>\d+)_"
    r"(?P<indicator>i_[a-z]+)_K(?P<K>\d+)_(?P<rtype>top|random_seed_-?\d+)\.json$"
)


def _per_fold_arrays(run: Dict) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    plus = run.get("plus_G", {}).get("per_fold_r2")
    minus = run.get("minus_G", {}).get("per_fold_r2")
    if not plus or not minus:
        return None
    n = min(len(plus), len(minus))
    if n == 0:
        return None
    return np.asarray(plus[:n], dtype=float), np.asarray(minus[:n], dtype=float)


def load_runs(per_run_dir: Path, model: str, paradigm_dir: str, layer: int,
              indicator: str, K: int) -> Dict:
    """Load top + all random per-run JSONs for one (model, K) cell."""
    prefix = f"{model}_{paradigm_dir}_L{layer}_{indicator}_K{K}_"
    files = sorted(per_run_dir.glob(f"{prefix}*.json"))
    top_run, random_runs = None, []
    for fp in files:
        with open(fp) as f:
            run = json.load(f)
        if fp.name.endswith("_top.json"):
            top_run = run
        elif "_random_seed_" in fp.name:
            random_runs.append(run)
    return {"top": top_run, "random": random_runs}


# ----- paired bootstrap -----------------------------------------------------


def paired_bootstrap_random_minus_top(
    top_plus: np.ndarray,
    top_minus: np.ndarray,
    random_runs: List[Tuple[np.ndarray, np.ndarray]],
    n_resamples: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float, np.ndarray]:
    """Returns (mean_diff, ci_low, ci_high, distribution)."""
    n_folds = min(
        len(top_plus), len(top_minus),
        *[len(p) for (p, _) in random_runs] or [0],
        *[len(m) for (_, m) in random_runs] or [0],
    )
    if n_folds < 2 or len(random_runs) == 0:
        return float("nan"), float("nan"), float("nan"), np.array([])

    top_plus = top_plus[:n_folds]
    top_minus = top_minus[:n_folds]
    rand_plus = np.stack([p[:n_folds] for (p, _) in random_runs])   # (R, n_folds)
    rand_minus = np.stack([m[:n_folds] for (_, m) in random_runs])  # (R, n_folds)

    diffs = np.empty(n_resamples, dtype=float)
    for b in range(n_resamples):
        idx = rng.integers(0, n_folds, size=n_folds)
        delta_top_b = float(top_plus[idx].mean() - top_minus[idx].mean())
        delta_rand_b = float(rand_plus[:, idx].mean(axis=1).mean()
                             - rand_minus[:, idx].mean(axis=1).mean())
        diffs[b] = delta_rand_b - delta_top_b

    return (
        float(np.mean(diffs)),
        float(np.quantile(diffs, 0.025)),
        float(np.quantile(diffs, 0.975)),
        diffs,
    )


# ----- outcome classification ----------------------------------------------


def classify_outcome(by_K: Dict[str, Dict], primary_K: int) -> str:
    """Three outcome branches per claim_surgery_D_outcome_branches.md."""
    primary = by_K.get(str(primary_K), {})
    primary_pass = bool(primary.get("primary_pass"))
    if not primary_pass:
        return "D-fails"
    others_pass = [
        bool(v.get("ci_low", float("-inf")) > 0)
        for k, v in by_K.items() if k != str(primary_K) and "ci_low" in v
    ]
    if others_pass and all(others_pass):
        return "D-passes"
    return "D-mixed"


# ----- driver --------------------------------------------------------------


def analyze_one_model(
    model: str,
    paradigm_dir: str,
    layer: int,
    indicator: str,
    K_values: List[int],
    per_run_dir: Path,
    bootstrap_n: int,
    primary_K: int,
    seed: int,
) -> Dict:
    rng = np.random.default_rng(seed)
    by_K: Dict[str, Dict] = {}
    for K in K_values:
        runs = load_runs(per_run_dir, model, paradigm_dir, layer, indicator, K)
        top_run = runs["top"]
        random_runs = runs["random"]
        if top_run is None:
            log.warning("missing top-K run for K=%d (%s)", K, model)
            by_K[str(K)] = {"reason": "missing top-K run"}
            continue
        top_pf = _per_fold_arrays(top_run)
        if top_pf is None:
            by_K[str(K)] = {"reason": "no per-fold R² in top run"}
            continue
        rand_pfs = []
        for r in random_runs:
            arr = _per_fold_arrays(r)
            if arr is not None:
                rand_pfs.append(arr)
        if len(rand_pfs) == 0:
            by_K[str(K)] = {"reason": "no random replicates"}
            continue

        delta_top = float(top_pf[0].mean() - top_pf[1].mean())
        delta_rand_per = np.array([float(p.mean() - m.mean()) for (p, m) in rand_pfs])
        mean_diff, ci_low, ci_high, _ = paired_bootstrap_random_minus_top(
            top_pf[0], top_pf[1], rand_pfs, bootstrap_n, rng,
        )
        primary_pass = bool(ci_low > 0) if K == primary_K else None
        by_K[str(K)] = {
            "K": int(K),
            "n_random_replicates": int(len(rand_pfs)),
            "delta_top": delta_top,
            "delta_random_mean": float(delta_rand_per.mean()),
            "delta_random_std": float(delta_rand_per.std(ddof=1) if len(delta_rand_per) > 1 else 0.0),
            "diff_mean": mean_diff,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "primary_pass": primary_pass,
        }
        log.info("[%s K=%d] Δ_top=%.4f Δ_rand=%.4f diff=%.4f CI=[%.4f, %.4f]%s",
                 model, K, delta_top, float(delta_rand_per.mean()),
                 mean_diff, ci_low, ci_high,
                 "  PRIMARY PASS" if primary_pass else (
                     "  PRIMARY FAIL" if primary_pass is False else ""))

    primary_block = by_K.get(str(primary_K), {})
    return {
        "model": model,
        "paradigm_dir": paradigm_dir,
        "indicator": indicator,
        "layer": layer,
        "primary_K": primary_K,
        "primary_pass": bool(primary_block.get("primary_pass")) if primary_block.get("primary_pass") is not None else None,
        "by_K": by_K,
        "outcome_branch": classify_outcome(by_K, primary_K),
    }


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Track D — paired bootstrap analysis")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--paradigm-dir", default="slot_machine")
    p.add_argument("--per_run_dir", type=Path, default=None)
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--model", choices=["gemma", "llama", "all"], default="all")
    args = p.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    layer = int(cfg["layer"])
    indicator = cfg["indicator"]
    K_values = list(cfg["K_values"])
    primary_K = int(cfg["preregistration"]["primary_K"])
    bootstrap_n = int(cfg["preregistration"]["bootstrap_n_resamples"])
    seed = int(cfg.get("seeds", {}).get("numpy", 42))

    out_root = Path(args.output_dir or cfg["output"]["root"])
    per_run_dir = args.per_run_dir or (out_root / cfg["output"].get("per_run_subdir", "per_run"))
    results_dir = out_root / cfg["output"].get("results_subdir", "results")
    results_dir.mkdir(parents=True, exist_ok=True)

    models = ["gemma", "llama"] if args.model == "all" else [args.model]
    final: Dict = {
        "paradigm_dir": args.paradigm_dir,
        "layer": layer,
        "indicator": indicator,
        "K_values": K_values,
        "primary_K": primary_K,
        "bootstrap_n_resamples": bootstrap_n,
        "models": {},
    }
    for mk in models:
        final["models"][mk] = analyze_one_model(
            mk, args.paradigm_dir, layer, indicator,
            K_values, per_run_dir, bootstrap_n, primary_K, seed,
        )

    out_path = results_dir / f"d_analysis_{args.paradigm_dir}_{indicator}_L{layer}.json"
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2)
    log.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
