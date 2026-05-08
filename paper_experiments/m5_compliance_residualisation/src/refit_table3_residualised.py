"""Re-fit the §4.3 condition-modulation Ridge readout on residualised features.

Pipeline parity (must match `sae_v3_analysis/src/run_groupkfold_recompute.py`):
  - 5-fold GroupKFold by game_id (no shuffle; deterministic group→fold map)
  - within-fold RandomForest deconfound on [bal, rn, bal², log1p(bal), bal·rn]
  - top-K=200 features by |Spearman ρ| with deconfounded target on TRAIN fold
  - StandardScaler + Ridge(α=100) on top-K
  - Active-feature filter: nnz > 10 across rows (matches reference)

For each residualised feature cache produced by `residualise_sae_features.py`,
this script:
  1. Loads the cache (sparse COO + meta).
  2. Computes I_BA target per round (bet/balance), via the same `compute_iba`
     we reuse from `sae_v3_analysis/src/run_comprehensive_robustness.py`.
  3. Filters to plus_G subset → fit GroupKFold → R²_+G' (residualised).
     Filters to minus_G subset → fit → R²_-G'.
  4. Computes Δ_G' = R²_+G' − R²_-G'.
  5. Saves per-mode results to JSON.

We import the canonical helpers from sae_v3_analysis to guarantee numerical
parity — there is exactly one fit_groupkfold/nl_deconfound_split implementation
in the repo and we reuse it instead of reimplementing.
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
DEFAULT_CONFIG = ROOT / "configs" / "m5_config.yaml"

# Reuse existing GroupKFold protocol — single source of truth.
SAE_V3_SRC = Path(__file__).resolve().parents[3] / "sae_v3_analysis" / "src"
if str(SAE_V3_SRC) not in sys.path:
    sys.path.insert(0, str(SAE_V3_SRC))

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger("m5.refit")


def _load_sparse_features(npz_path: Path):
    """Mirror of run_perm_null_ilc.load_sae_and_meta but for our residualised
    NPZ files. Returns (csr_matrix, meta_dict)."""
    from scipy import sparse
    z = np.load(npz_path, allow_pickle=False)
    shape = tuple(z["shape"])
    sp = sparse.csr_matrix(
        (z["values"], (z["row_indices"], z["col_indices"])),
        shape=shape,
        dtype=np.float32,
    )
    meta = {k: z[k] for k in z.keys() if k not in ("row_indices", "col_indices", "values", "shape")}
    return sp, meta


def _filter_subset(meta: Dict, subset: str, target: np.ndarray, balances: np.ndarray, indicator: str):
    """Mirror of fit_one_subset.valid_filter logic for plus_G / minus_G / fixed_all.

    Returns boolean mask of valid rows.
    """
    bt = meta["bet_types"]
    valid = (bt == "variable") & ~np.isnan(target) & ~np.isnan(balances) & (balances > 0)
    if indicator == "i_ba":
        valid = valid & (target > 0)

    pc = None
    for k in ("prompt_combo", "prompt_combos", "prompt_conditions"):
        if k in meta and meta[k] is not None:
            pc = meta[k]
            break

    if subset == "all_variable":
        return valid
    if subset == "fixed_all":
        valid = (~np.isnan(target)) & ~np.isnan(balances) & (balances > 0) & (bt == "fixed")
        if indicator == "i_ba":
            valid = valid & (target > 0)
        return valid
    if pc is None:
        log.warning("no prompt_condition field in meta — cannot filter %s", subset)
        return np.zeros_like(valid)
    if subset == "plus_G":
        return valid & np.array(["G" in str(p) for p in pc])
    if subset == "minus_G":
        return valid & np.array(["G" not in str(p) for p in pc])
    if subset == "plus_M":
        return valid & np.array(["M" in str(p) for p in pc])
    if subset == "minus_M":
        return valid & np.array(["M" not in str(p) for p in pc])
    raise ValueError(f"unknown subset {subset}")


def fit_one_residualised_subset(
    sp,
    meta: Dict,
    model: str,
    task: str,
    indicator: str,
    subset: str,
):
    """Apply the canonical GroupKFold readout to a single subset.

    Lazily imports from sae_v3_analysis so pytest collect-only does not pull
    in heavyweight dependencies.
    """
    from run_groupkfold_recompute import fit_groupkfold, compute_loss_chasing_continuous
    from run_comprehensive_robustness import compute_iba

    if indicator == "i_ba":
        result = compute_iba(meta, model, task)
        if result is None:
            return {"reason": "compute_iba returned None"}
        target, balances = result
    elif indicator == "i_lc":
        target, balances = compute_loss_chasing_continuous(meta, model, task)
    elif indicator == "i_ec":
        result = compute_iba(meta, model, task)
        if result is None:
            return {"reason": "compute_iba returned None"}
        bet_ratios, balances = result
        target = np.where(np.isnan(bet_ratios), np.nan, (bet_ratios >= 0.5).astype(float))
    else:
        raise ValueError(f"unsupported indicator {indicator}")

    valid = _filter_subset(meta, subset, target, balances, indicator)
    n = int(valid.sum())
    if n < 100:
        return {"reason": f"n<100 ({n})", "n": n}

    X_sparse = sp[valid]
    t = target[valid]
    bal = balances[valid]
    rn = meta["round_nums"][valid].astype(float)
    gids = np.asarray(meta["game_ids"])[valid]

    nnz = np.diff(X_sparse.tocsc().indptr)
    active = np.where(nnz > 10)[0]
    X = X_sparse[:, active].toarray()
    if X.shape[1] == 0:
        return {"reason": "no active features", "n": n}

    n_groups = int(len(np.unique(gids)))
    r2_mean, r2_std = fit_groupkfold(X, t, bal, rn, gids)
    return {
        "n": n,
        "n_groups": n_groups,
        "r2_mean": r2_mean,
        "r2_std": r2_std,
        "mean_target": float(np.nanmean(t)) if r2_mean is not None else None,
    }


def refit_for_mode(
    npz_path: Path,
    model: str,
    task: str,
    indicator: str,
    subsets: List[str],
) -> Dict:
    """Re-fit Ridge readout on a residualised cache. Returns subset → R² dict."""
    sp, meta = _load_sparse_features(npz_path)
    out: Dict[str, Dict] = {}
    for subset in subsets:
        log.info("  subset %s ...", subset)
        out[subset] = fit_one_residualised_subset(sp, meta, model, task, indicator, subset)
    return out


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="M5 — re-fit Table 3 readout on residualised features")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--model", choices=["gemma", "llama", "all"], default="all")
    p.add_argument("--paradigm-dir", default="slot_machine")
    p.add_argument(
        "--subsets", nargs="+",
        default=["plus_G", "minus_G"],
        help="condition subsets to fit (matches §4.3 keys)",
    )
    p.add_argument("--input", type=Path, default=None,
                   help="override residualised cache directory")
    p.add_argument("--output", type=Path, default=None,
                   help="override results subdir")
    args = p.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    layer = int(cfg["layer"])
    indicator = cfg["indicator"]
    task_short = cfg["task"]  # "sm"
    paradigm_dir = args.paradigm_dir

    out_root = Path(args.output or cfg["output"]["root"])
    resid_dir = args.input or (out_root / cfg["output"].get("residualised_subdir", "residualised"))
    results_dir = out_root / cfg["output"].get("results_subdir", "results")
    results_dir.mkdir(parents=True, exist_ok=True)

    models = ["gemma", "llama"] if args.model == "all" else [args.model]
    all_results: Dict = {}
    for mk in models:
        # Find every residualised cache for this model+paradigm.
        prefix = f"{mk}_{paradigm_dir}_L{layer}_"
        caches = sorted(Path(resid_dir).glob(f"{prefix}*.npz"))
        if not caches:
            log.warning("no residualised caches found for %s/%s in %s", mk, paradigm_dir, resid_dir)
            continue
        per_mode: Dict[str, Dict] = {}
        for cache in caches:
            mode_tag = cache.stem[len(prefix):]
            log.info("[%s] mode=%s file=%s", mk, mode_tag, cache.name)
            per_mode[mode_tag] = refit_for_mode(cache, mk, task_short, indicator, args.subsets)
        all_results[mk] = per_mode

    out_path = results_dir / f"refit_results_{paradigm_dir}_{indicator}_L{layer}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
