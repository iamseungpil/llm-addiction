#!/usr/bin/env python3
"""RQ1 Level 2 Selectivity: prompt-group-held-out cross validation.

Intent
------
  Existing Level 1 (`run_probe_selectivity_controls.py`) splits games into folds
  with GroupKFold(groups=game_ids). Two games with the same prompt can still
  appear in different folds, so the probe can memorize prompt-specific fingerprint.

  Level 2 swaps the grouping: GroupKFold(groups=prompt_conditions). Every
  prompt is held out entirely from training in some fold, so the probe must
  generalize across prompt conditions to score.

Hypothesis
----------
  The SAE-feature readout of I_LC (and I_BA where applicable) generalizes to
  held-out prompt conditions with R^2 above a label-permuted null. This
  directly answers "your readout is just memorizing prompt fingerprints".

Design
------
  * Scope: 6 cells (Gemma/LLaMA x SM/IC/MW), using same layer/metric as
    paper_neural_audit.json.
  * GroupKFold(k=min(5, n_prompts)) with groups=prompt_conditions.
    - SM/MW have 32 prompts -> k=5
    - IC has 4 prompts -> k=4
  * Within each fold (nested):
      1. balance + round RF residualize (nl_deconfound_split, train-only fit)
      2. top-200 SAE feature selection on train residuals only
      3. Ridge(alpha=100) fit on train, predict on held-out fold
      4. R^2 on held-out prompt games
  * Null: shuffle target across all games (not within prompt), apply same
    GroupKFold pipeline, n=100 controls. Report perm_p.

Verification
------------
  * Real R^2 > 95th percentile null -> pass (perm_p < 0.05)
  * Output JSON with real_r2, null_r2s, perm_p for each cell.
  * Paper: add Table 2 subsection "Level 2 Prompt-group held-out".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))

from run_comprehensive_robustness import (
    RIDGE_ALPHA,
    TOP_K,
    compute_iba,
    load_sae_and_meta,
    nl_deconfound_split,
)
from run_perm_null_ilc import compute_loss_chasing
from run_probe_selectivity_controls import (
    active_columns_from_train,
    build_valid_subset,
    select_top_features,
)


RESULTS_DIR = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis/results/robustness")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CONFIGS = [
    {"model": "gemma", "paradigm": "sm", "layer": 24, "metric": "i_lc"},
    {"model": "gemma", "paradigm": "ic", "layer": 24, "metric": "i_lc"},
    {"model": "gemma", "paradigm": "mw", "layer": 24, "metric": "i_lc"},
    {"model": "llama", "paradigm": "sm", "layer": 16, "metric": "i_lc"},
    {"model": "llama", "paradigm": "ic", "layer": 16, "metric": "i_lc"},
    {"model": "llama", "paradigm": "mw", "layer": 16, "metric": "i_lc"},
]


def eval_prompt_group_pipeline(
    X_sparse,
    target,
    balances,
    round_nums,
    prompt_conditions,
    min_active_nnz,
    permuted_target=None,
):
    """5-fold GroupKFold with groups=prompt_conditions.

    If permuted_target is given, use it instead of target (for null distribution).
    Features selected and balance-residualized within each fold only.
    """
    y = permuted_target if permuted_target is not None else target
    unique_prompts = np.unique(prompt_conditions)
    n_splits = min(5, len(unique_prompts))
    if n_splits < 3:
        raise ValueError(f"Need at least 3 prompt groups for GroupKFold, got {len(unique_prompts)}")

    cv = GroupKFold(n_splits=n_splits)
    folds = list(cv.split(np.zeros(len(y)), y, groups=prompt_conditions))

    fold_r2s = []
    for tr, te in folds:
        Xtr_sparse = X_sparse[tr]
        Xte_sparse = X_sparse[te]
        ytr = y[tr]
        yte = y[te]

        active_cols = active_columns_from_train(Xtr_sparse, min_active_nnz)
        if len(active_cols) == 0:
            continue

        Xtr = Xtr_sparse[:, active_cols].toarray()
        Xte = Xte_sparse[:, active_cols].toarray()

        res_tr, res_te = nl_deconfound_split(
            ytr, balances[tr], round_nums[tr],
            yte, balances[te], round_nums[te],
        )
        top_idx = select_top_features(Xtr, res_tr, TOP_K)

        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr[:, top_idx])
        Xte_s = scaler.transform(Xte[:, top_idx])

        pred = Ridge(RIDGE_ALPHA).fit(Xtr_s, res_tr).predict(Xte_s)
        fold_r2s.append(r2_score(res_te, pred))

    if not fold_r2s:
        return np.nan, []
    return float(np.mean(fold_r2s)), [float(x) for x in fold_r2s]


def run_one_config(cfg, n_perms, min_active_nnz, smoke, seed=42):
    model = cfg["model"]
    paradigm = cfg["paradigm"]
    layer = cfg["layer"]
    metric = cfg["metric"]
    tag = f"{model}_{paradigm}_L{layer}_{metric}"
    print(f"\n{'=' * 80}\n{tag} [Level 2: prompt-group held-out]\n{'=' * 80}")

    sp, meta = load_sae_and_meta(model, paradigm, layer)
    if sp is None:
        print("  SKIP: SAE data missing")
        return None

    subset = build_valid_subset(sp, meta, model, paradigm, metric)
    if subset is None:
        print("  SKIP: valid subset missing")
        return None

    X_sparse = subset["X_sparse"]
    target = subset["target"]
    balances = subset["balances"]
    round_nums = subset["round_nums"]
    prompt_conditions = subset["prompt_conditions"]

    n_prompts = len(np.unique(prompt_conditions))
    if n_prompts < 3:
        print(f"  SKIP: only {n_prompts} prompt groups (need >= 3 for GroupKFold)")
        return None

    print(f"  n={len(target)}, prompt_groups={n_prompts}, "
          f"metric_mean={target.mean():.4f}, active_input_dim={X_sparse.shape[1]}")

    real_r2, real_folds = eval_prompt_group_pipeline(
        X_sparse, target, balances, round_nums, prompt_conditions, min_active_nnz,
    )
    print(f"  Real prompt-held-out R²: {real_r2:.4f} (folds: {[f'{x:.3f}' for x in real_folds]})")

    n_perms = 5 if smoke else n_perms
    null_r2s = []
    rng = np.random.RandomState(seed)
    for pi in range(n_perms):
        perm_idx = rng.permutation(len(target))
        ptarget = target[perm_idx]
        null_r2, _ = eval_prompt_group_pipeline(
            X_sparse, target, balances, round_nums, prompt_conditions,
            min_active_nnz, permuted_target=ptarget,
        )
        null_r2s.append(null_r2)
        if (pi + 1) % 10 == 0 or smoke:
            print(f"  Permutation {pi + 1}/{n_perms}: null R²={null_r2:.4f}")

    null_r2s = np.array(null_r2s, dtype=float)
    null_valid = null_r2s[~np.isnan(null_r2s)]
    perm_p = (1 + np.sum(null_valid >= real_r2)) / (1 + len(null_valid))

    result = {
        "config": cfg,
        "level": 2,
        "grouping": "prompt_conditions",
        "n_samples": int(len(target)),
        "n_prompt_groups": int(n_prompts),
        "real_r2": float(real_r2),
        "real_fold_r2s": real_folds,
        "n_perms": int(len(null_valid)),
        "null_r2_mean": float(np.nanmean(null_valid)) if len(null_valid) else float("nan"),
        "null_r2_std": float(np.nanstd(null_valid)) if len(null_valid) else float("nan"),
        "null_r2_95pct": float(np.nanpercentile(null_valid, 95)) if len(null_valid) else float("nan"),
        "null_r2s": [float(x) for x in null_r2s],
        "perm_p": float(perm_p),
    }
    print(f"  Null R² mean±std: {result['null_r2_mean']:.4f} ± {result['null_r2_std']:.4f}")
    print(f"  Null R² 95th pct: {result['null_r2_95pct']:.4f}")
    print(f"  Permutation p:    {result['perm_p']:.4f} "
          f"{'PASS' if result['perm_p'] < 0.05 else 'FAIL'}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-perms", type=int, default=100)
    parser.add_argument("--min-active-nnz", type=int, default=10)
    parser.add_argument("--smoke", action="store_true",
                        help="Small test: 5 perms, 1 cell")
    parser.add_argument("--config", action="append",
                        help="Optional config(s) as model:paradigm:layer:metric")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    configs = DEFAULT_CONFIGS
    if args.config:
        configs = []
        for item in args.config:
            m, p, l, met = item.split(":")
            configs.append({"model": m, "paradigm": p, "layer": int(l), "metric": met})
    elif args.smoke:
        configs = [DEFAULT_CONFIGS[0]]

    results = {}
    for cfg in configs:
        res = run_one_config(cfg, args.n_perms, args.min_active_nnz, args.smoke)
        if res is not None:
            tag = f"{cfg['model']}_{cfg['paradigm']}_L{cfg['layer']}_{cfg['metric']}"
            results[tag] = res

    out_name = "rq1_l2_selectivity_smoke.json" if args.smoke else "rq1_l2_selectivity.json"
    out_path = RESULTS_DIR / out_name
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")
    print(f"\nSummary:")
    for tag, r in results.items():
        verdict = "PASS" if r["perm_p"] < 0.05 else "FAIL"
        print(f"  {tag}: R²={r['real_r2']:.4f} null95={r['null_r2_95pct']:.4f} "
              f"perm_p={r['perm_p']:.4f} {verdict}")


if __name__ == "__main__":
    main()
