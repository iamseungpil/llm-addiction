#!/usr/bin/env python3
"""
Probe selectivity with nuisance-matched control targets.

Purpose
-------
Strengthen RQ1 without changing the core pipeline:
  1. hold out whole games with GroupKFold
  2. keep RF deconfounding, train-fold feature selection, and Ridge readout
  3. compare the real target against nuisance-matched control targets

This is meant to answer a stricter probing question than the current
random-feature and label-permutation checks:
does the SAE signal still beat a target that preserves easy nuisance
structure (round / balance / prompt condition) but breaks the real
behavioral assignment?
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold
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


RESULTS_DIR = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis/results/robustness")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CONFIGS = [
    {"model": "gemma", "paradigm": "sm", "layer": 24, "metric": "i_lc"},
    {"model": "llama", "paradigm": "sm", "layer": 16, "metric": "i_lc"},
    {"model": "gemma", "paradigm": "mw", "layer": 24, "metric": "i_ba"},
    {"model": "llama", "paradigm": "mw", "layer": 16, "metric": "i_ba"},
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-controls", type=int, default=20)
    parser.add_argument("--min-active-nnz", type=int, default=10)
    parser.add_argument("--balance-bins", type=int, default=10)
    parser.add_argument(
        "--config",
        action="append",
        help="Optional config(s) in model:paradigm:layer:metric format",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def load_target(metric, model, paradigm, meta):
    if metric == "i_ba":
        result = compute_iba(meta, model, paradigm)
    elif metric == "i_lc":
        result = compute_loss_chasing(meta, model, paradigm)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    if result is None:
        return None
    target, balances = result
    return target.astype(float), balances.astype(float)


def build_valid_subset(sp, meta, model, paradigm, metric):
    target_result = load_target(metric, model, paradigm, meta)
    if target_result is None:
        return None

    target, balances = target_result
    round_nums = meta["round_nums"].astype(float)
    game_ids = meta["game_ids"]
    prompt_conditions = meta.get("prompt_conditions")
    if prompt_conditions is None:
        prompt_conditions = np.array(["UNK"] * len(game_ids))
    bet_types = meta["bet_types"]

    valid = (
        (bet_types == "variable")
        & ~np.isnan(target)
        & ~np.isnan(balances)
        & (balances > 0)
    )
    if metric == "i_ba":
        valid &= target > 0

    if valid.sum() < 200:
        return None

    return {
        "X_sparse": sp[valid],
        "target": target[valid],
        "balances": balances[valid],
        "round_nums": round_nums[valid],
        "game_ids": game_ids[valid],
        "prompt_conditions": prompt_conditions[valid],
    }


def fit_balance_bin_edges(values, n_bins):
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(values, qs)
    return np.unique(edges)


def apply_balance_bins(values, edges):
    if len(edges) <= 2:
        return np.zeros(len(values), dtype=int)
    bins = np.digitize(values, edges[1:-1], right=True)
    return bins.astype(int)


def make_strata(balance_bins, round_nums, prompt_conditions):
    rounded_rounds = round_nums.astype(int)
    return np.array(
        [f"{rb}|{rn}|{pc}" for rb, rn, pc in zip(balance_bins, rounded_rounds, prompt_conditions)],
        dtype=object,
    )


def nuisance_matched_game_shuffle(
    train_target,
    train_balances,
    train_round_nums,
    train_prompt_conditions,
    train_game_ids,
    test_balances,
    test_round_nums,
    test_prompt_conditions,
    test_game_ids,
    n_bins,
    rng,
):
    """Build a fold-local control by reassigning whole train games within nuisance strata."""
    edges = fit_balance_bin_edges(train_balances, n_bins)
    train_bins = apply_balance_bins(train_balances, edges)
    test_bins = apply_balance_bins(test_balances, edges)

    train_strata = make_strata(train_bins, train_round_nums, train_prompt_conditions)
    test_strata = make_strata(test_bins, test_round_nums, test_prompt_conditions)

    game_to_rows = {}
    for gid in np.unique(train_game_ids):
        row_idx = np.where(train_game_ids == gid)[0]
        if len(row_idx) == 0:
            continue
        strata_seq = tuple(train_strata[row_idx].tolist())
        game_to_rows[gid] = {
            "rows": row_idx,
            "strata_seq": strata_seq,
            "target": train_target[row_idx].copy(),
        }

    buckets = {}
    for gid, payload in game_to_rows.items():
        buckets.setdefault(payload["strata_seq"], []).append(gid)

    gid_map = {}
    for _, gids in buckets.items():
        if len(gids) < 2:
            continue
        perm = np.array(gids, dtype=object)[rng.permutation(len(gids))]
        if np.array_equal(perm, np.array(gids, dtype=object)):
            perm = np.roll(perm, 1)
        gid_map.update({src: dst for src, dst in zip(gids, perm)})

    ctrl_train = train_target.copy()
    for gid, payload in game_to_rows.items():
        donor_gid = gid_map.get(gid)
        if donor_gid is None:
            continue
        donor = game_to_rows[donor_gid]["target"]
        if len(donor) != len(payload["rows"]):
            continue
        ctrl_train[payload["rows"]] = donor

    donor_by_stratum = {}
    for key in np.unique(train_strata):
        donor_by_stratum[key] = train_target[train_strata == key]

    ctrl_test = test_balances.astype(float).copy()
    ctrl_test[:] = np.nan
    for i, key in enumerate(test_strata):
        donors = donor_by_stratum.get(key)
        if donors is None or len(donors) == 0:
            ctrl_test[i] = test_balances[i]
        else:
            ctrl_test[i] = donors[rng.randint(len(donors))]

    return ctrl_train, ctrl_test


def active_columns_from_train(X_train_sparse, min_active_nnz):
    nnz_per_col = np.diff(X_train_sparse.tocsc().indptr)
    active_cols = np.where(nnz_per_col > min_active_nnz)[0]
    return active_cols


def select_top_features(X_train, residual_train, top_k):
    corrs = np.array(
        [
            abs(spearmanr(X_train[:, j], residual_train)[0]) if X_train[:, j].std() > 0 else 0.0
            for j in range(X_train.shape[1])
        ]
    )
    corrs = np.nan_to_num(corrs, nan=0.0)
    k = min(top_k, len(corrs))
    return np.argsort(corrs)[-k:]


def eval_group_pipeline(
    X_sparse,
    target,
    balances,
    round_nums,
    game_ids,
    prompt_conditions,
    min_active_nnz,
    control_mode=False,
    balance_bins=10,
    control_seed=42,
):
    unique_games = np.unique(game_ids)
    n_splits = min(5, len(unique_games))
    if n_splits < 3:
        raise ValueError(f"Need at least 3 games for GroupKFold, got {len(unique_games)}")

    cv = GroupKFold(n_splits=n_splits)
    folds = list(cv.split(np.zeros(len(target)), target, groups=game_ids))

    fold_r2s = []
    for fold_idx, (tr, te) in enumerate(folds):
        Xtr_sparse = X_sparse[tr]
        Xte_sparse = X_sparse[te]
        ytr = target[tr]
        yte = target[te]

        if control_mode:
            ytr, yte = nuisance_matched_game_shuffle(
                train_target=target[tr],
                train_balances=balances[tr],
                train_round_nums=round_nums[tr],
                train_prompt_conditions=prompt_conditions[tr],
                train_game_ids=game_ids[tr],
                test_balances=balances[te],
                test_round_nums=round_nums[te],
                test_prompt_conditions=prompt_conditions[te],
                test_game_ids=game_ids[te],
                n_bins=balance_bins,
                rng=np.random.RandomState(control_seed + fold_idx),
            )

        active_cols = active_columns_from_train(Xtr_sparse, min_active_nnz)
        if len(active_cols) == 0:
            continue

        Xtr = Xtr_sparse[:, active_cols].toarray()
        Xte = Xte_sparse[:, active_cols].toarray()

        res_tr, res_te = nl_deconfound_split(
            ytr, balances[tr], round_nums[tr], yte, balances[te], round_nums[te]
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


def eval_standard_pipeline(X_sparse, target, balances, round_nums, min_active_nnz):
    X_dense = X_sparse.toarray()
    folds = list(KFold(n_splits=5, shuffle=True, random_state=42).split(X_dense))

    fold_r2s = []
    for tr, te in folds:
        active_cols = np.where((X_dense[tr] != 0).sum(axis=0) > min_active_nnz)[0]
        if len(active_cols) == 0:
            continue

        Xtr = X_dense[tr][:, active_cols]
        Xte = X_dense[te][:, active_cols]
        res_tr, res_te = nl_deconfound_split(
            target[tr], balances[tr], round_nums[tr], target[te], balances[te], round_nums[te]
        )
        top_idx = select_top_features(Xtr, res_tr, TOP_K)
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr[:, top_idx])
        Xte_s = scaler.transform(Xte[:, top_idx])
        pred = Ridge(RIDGE_ALPHA).fit(Xtr_s, res_tr).predict(Xte_s)
        fold_r2s.append(r2_score(res_te, pred))

    if not fold_r2s:
        return np.nan
    return float(np.mean(fold_r2s))


def run_one_config(cfg, n_controls, min_active_nnz, balance_bins, smoke):
    model = cfg["model"]
    paradigm = cfg["paradigm"]
    layer = cfg["layer"]
    metric = cfg["metric"]
    tag = f"{model}_{paradigm}_L{layer}_{metric}"
    print(f"\n{'=' * 80}")
    print(tag)
    print(f"{'=' * 80}")

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
    game_ids = subset["game_ids"]
    prompt_conditions = subset["prompt_conditions"]

    print(
        f"  n={len(target)}, games={len(np.unique(game_ids))}, "
        f"metric_mean={target.mean():.4f}, active_input_dim={X_sparse.shape[1]}, "
        f"prompt_strata={len(np.unique(prompt_conditions))}"
    )

    real_r2, real_folds = eval_group_pipeline(
        X_sparse,
        target,
        balances,
        round_nums,
        game_ids,
        prompt_conditions,
        min_active_nnz,
    )
    standard_r2 = eval_standard_pipeline(
        X_sparse, target, balances, round_nums, min_active_nnz
    )
    print(f"  Real GroupKFold R²: {real_r2:.4f}")
    print(f"  Standard KFold R²: {standard_r2:.4f}")

    control_r2s = []
    n_controls = 3 if smoke else n_controls
    for ci in range(n_controls):
        ctrl_r2, _ = eval_group_pipeline(
            X_sparse,
            target,
            balances,
            round_nums,
            game_ids,
            prompt_conditions,
            min_active_nnz,
            control_mode=True,
            balance_bins=balance_bins,
            control_seed=42 + 1000 * ci,
        )
        control_r2s.append(ctrl_r2)
        print(f"  Control {ci + 1}/{n_controls}: R²={ctrl_r2:.4f}")

    control_r2s = np.array(control_r2s, dtype=float)
    p_value = (1 + np.sum(control_r2s >= real_r2)) / (1 + len(control_r2s))
    result = {
        "config": cfg,
        "n_samples": int(len(target)),
        "n_games": int(len(np.unique(game_ids))),
        "real_group_r2": float(real_r2),
        "standard_kfold_r2": float(standard_r2),
        "group_minus_standard": float(real_r2 - standard_r2),
        "real_fold_r2s": real_folds,
        "control_r2_mean": float(np.nanmean(control_r2s)),
        "control_r2_std": float(np.nanstd(control_r2s)),
        "control_r2s": [float(x) for x in control_r2s],
        "selectivity_gap": float(real_r2 - np.nanmean(control_r2s)),
        "p_selectivity": float(p_value),
        "n_controls": int(len(control_r2s)),
        "balance_bins": int(balance_bins),
        "control_construction": "fold_local_game_strata_shuffle",
    }
    print(
        f"  Control mean±std: {result['control_r2_mean']:.4f} ± {result['control_r2_std']:.4f}\n"
        f"  Selectivity gap:  {result['selectivity_gap']:.4f}\n"
        f"  p_selectivity:    {result['p_selectivity']:.4f}"
    )
    return result


def main():
    args = parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    results = {}

    configs = DEFAULT_CONFIGS
    if args.config:
        configs = []
        for item in args.config:
            model, paradigm, layer, metric = item.split(":")
            configs.append(
                {
                    "model": model,
                    "paradigm": paradigm,
                    "layer": int(layer),
                    "metric": metric,
                }
            )
    elif args.smoke:
        configs = [DEFAULT_CONFIGS[0]]

    for cfg in configs:
        res = run_one_config(
            cfg,
            n_controls=args.n_controls,
            min_active_nnz=args.min_active_nnz,
            balance_bins=args.balance_bins,
            smoke=args.smoke,
        )
        if res is not None:
            tag = f"{cfg['model']}_{cfg['paradigm']}_L{cfg['layer']}_{cfg['metric']}"
            results[tag] = res

    out_name = "probe_selectivity_controls_smoke.json" if args.smoke else "probe_selectivity_controls.json"
    out_path = RESULTS_DIR / out_name
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
