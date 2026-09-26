"""Track D CLI — run paired top-K vs random-K SAE feature removal at one (model, K).

For one (model, layer, task, indicator, K) cell this writes:
  per_run/{model}_{paradigm_dir}_L{layer}_{indicator}_K{K}_top.json
  per_run/{model}_{paradigm_dir}_L{layer}_{indicator}_K{K}_random_seed_{s}.json   (×N_random_baselines_per_K)

Each per-run JSON contains plus_G + minus_G R² per fold, Δ_G, and the
(model, K, removal_type, replicate_seed) tag for downstream paired bootstrap.

We deliberately do not aggregate the random replicates here — `analyze_d.py`
loads them and runs the paired bootstrap. This keeps each launch atomic and
restartable.

Reuse:
  load_sae_and_meta  ← run_perm_null_ilc.py
  compute_iba         ← run_comprehensive_robustness.py
  nl_deconfound_split + TOP_K + RIDGE_ALPHA via topk_removal._import_canonical
  prompt-condition filter logic mirrors run_groupkfold_recompute.fit_one_subset
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
DEFAULT_CONFIG = ROOT / "configs" / "d_config.yaml"

SAE_V3_SRC = Path(__file__).resolve().parents[3] / "sae_v3_analysis" / "src"
if str(SAE_V3_SRC) not in sys.path:
    sys.path.insert(0, str(SAE_V3_SRC))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger("d.topk")


def _build_target_and_filters(meta: Dict, model: str, task: str, indicator: str):
    """Mirror of sae_v3_analysis run_groupkfold_recompute.fit_one_subset semantics
    up to the point of producing (target, balances, valid_plus_g, valid_minus_g).
    """
    from run_comprehensive_robustness import compute_iba
    from run_groupkfold_recompute import compute_loss_chasing_continuous, get_meta_field

    if indicator == "i_ba":
        result = compute_iba(meta, model, task)
        if result is None:
            return None
        target, balances = result
    elif indicator == "i_lc":
        target, balances = compute_loss_chasing_continuous(meta, model, task)
    elif indicator == "i_ec":
        result = compute_iba(meta, model, task)
        if result is None:
            return None
        bet_ratios, balances = result
        target = np.where(np.isnan(bet_ratios), np.nan, (bet_ratios >= 0.5).astype(float))
    else:
        raise ValueError(f"unsupported indicator {indicator}")

    bt = meta["bet_types"]
    valid_base = (bt == "variable") & ~np.isnan(target) & ~np.isnan(balances) & (balances > 0)
    if indicator == "i_ba":
        valid_base = valid_base & (target > 0)

    pc = get_meta_field(meta, "prompt_combo")
    if pc is None:
        return None
    plus_g_mask = valid_base & np.array(["G" in str(p) for p in pc])
    minus_g_mask = valid_base & np.array(["G" not in str(p) for p in pc])
    return target, balances, plus_g_mask, minus_g_mask


def run_one_K_one_removal(
    sp,
    meta: Dict,
    model: str,
    task: str,
    indicator: str,
    K: int,
    removal_type: str,
    seed: int = 0,
) -> Dict:
    """One CRT — fits Δ_G under one (K, removal_type, seed) ablation."""
    from topk_removal import compute_delta_g_with_removal

    built = _build_target_and_filters(meta, model, task, indicator)
    if built is None:
        return {"reason": "target/filter build failed", "K": K, "removal_type": removal_type}
    target, balances, plus_g_mask, minus_g_mask = built

    full_valid = plus_g_mask | minus_g_mask
    if int(full_valid.sum()) < 100:
        return {"reason": f"valid n<100 ({int(full_valid.sum())})", "K": K, "removal_type": removal_type}

    X_sparse = sp[full_valid]
    nnz = np.diff(X_sparse.tocsc().indptr)
    active = np.where(nnz > 10)[0]
    if active.size == 0:
        return {"reason": "no active features", "K": K, "removal_type": removal_type}
    X = X_sparse[:, active].toarray()
    rn = meta["round_nums"][full_valid].astype(float)
    gids = np.asarray(meta["game_ids"])[full_valid]
    bal = balances[full_valid]
    t = target[full_valid]

    plus_local = plus_g_mask[full_valid]
    minus_local = minus_g_mask[full_valid]

    rng = np.random.default_rng(seed) if removal_type == "random" else None
    res = compute_delta_g_with_removal(
        X_dense=X, target=t, balances=bal, rounds=rn, groups=gids,
        valid_plus_g=plus_local, valid_minus_g=minus_local,
        K=K, removal_type=removal_type, rng=rng,
    )
    res.update({
        "model": model, "task": task, "indicator": indicator,
        "n_active_features": int(active.size),
        "seed": int(seed) if removal_type == "random" else None,
    })
    return res


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Track D — top-K vs random-K SAE feature removal")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--model", choices=["gemma", "llama"], required=True)
    p.add_argument("--layer", type=int, default=None)
    p.add_argument("--task", default=None, help="paradigm short tag, e.g. sm")
    p.add_argument("--indicator", default=None, choices=["i_ba", "i_lc", "i_ec"])
    p.add_argument("--K", type=int, required=True)
    p.add_argument("--paradigm-dir", default="slot_machine")
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--n_random", type=int, default=None,
                   help="override N_random_baselines_per_K from config")
    p.add_argument("--only", choices=["top", "random", "both"], default="both")
    args = p.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    layer = int(args.layer if args.layer is not None else cfg["layer"])
    task = args.task or cfg["task"]
    indicator = args.indicator or cfg["indicator"]
    n_random = int(args.n_random if args.n_random is not None else cfg["N_random_baselines_per_K"])
    seed_base = int(cfg.get("random_seed_base", 1000))

    out_root = Path(args.output_dir or cfg["output"]["root"])
    per_run = out_root / cfg["output"].get("per_run_subdir", "per_run")
    per_run.mkdir(parents=True, exist_ok=True)

    from run_perm_null_ilc import load_sae_and_meta
    sp, meta = load_sae_and_meta(args.model, task, layer)
    if sp is None:
        log.error("SAE features missing for %s/%s L%d", args.model, task, layer)
        return 1

    tag = f"{args.model}_{args.paradigm_dir}_L{layer}_{indicator}_K{args.K}"

    if args.only in ("top", "both"):
        log.info("[%s] removal=top  K=%d  ...", tag, args.K)
        t0 = time.time()
        res = run_one_K_one_removal(sp, meta, args.model, task, indicator, args.K, "top", seed=0)
        log.info("  Δ_G=%s  (%.0fs)", res.get("delta_g"), time.time() - t0)
        with open(per_run / f"{tag}_top.json", "w") as f:
            json.dump(res, f, indent=2)

    if args.only in ("random", "both"):
        for i in range(n_random):
            seed = seed_base + i
            log.info("[%s] removal=random  seed=%d  K=%d  (%d/%d)",
                     tag, seed, args.K, i + 1, n_random)
            t0 = time.time()
            res = run_one_K_one_removal(
                sp, meta, args.model, task, indicator, args.K, "random", seed=seed,
            )
            log.info("  Δ_G=%s  (%.0fs)", res.get("delta_g"), time.time() - t0)
            with open(per_run / f"{tag}_random_seed_{seed}.json", "w") as f:
                json.dump(res, f, indent=2)

    log.info("done — outputs under %s", per_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
