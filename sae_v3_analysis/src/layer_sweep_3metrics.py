#!/usr/bin/env python3
"""
Layer sweep for I_LC, I_BA, I_EC across all (model, task, layer).

This version matches the canonical V17 / paper_neural_audit.json methodology:
  - I_BA  = bet/balance per round                         (continuous, [0, 1])
  - I_LC  = binary: 1 if (prev round was a loss AND       (binary)
            current bet_ratio > previous bet_ratio) else 0
  - I_EC  = binary: 1 if bet_ratio >= 0.5 else 0          (binary)
  - Filter to bet_types == "variable" rounds only
  - Within-fold RandomForest deconfound on
    [bal, rn, bal^2, log1p(bal), bal*rn]  (no leakage across folds)
  - Top-K=200 SAE features by |Pearson r| with target
  - Ridge alpha=100, 5-fold CV
  - Game-block permutation null (default n_perm=20 for the sweep, 200 in
    paper-grade single cells)

Output one JSONL line per (model, task, layer) cell with all three metrics.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

DATA_ROOT = Path("/home/v-seungplee/data/llm-addiction")
SAE_ROOT = DATA_ROOT / "sae_features_v3"
BEHAV_ROOT = DATA_ROOT / "behavioral"
OUT_DIR = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis/results/sweep_3metrics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

GEMMA_LAYERS = list(range(0, 42))
LLAMA_LAYERS = list(range(0, 32))

TASK_DIRS = {"sm": "slot_machine", "ic": "investment_choice", "mw": "mystery_wheel"}

# Canonical V17 hyperparameters (run_perm_null_ilc.py)
TOP_K = 200
RF_TREES = 50
RF_DEPTH = 8
RIDGE_ALPHA = 100.0


# -------------------- IO --------------------

def load_features(model: str, task: str, layer: int):
    npz_path = SAE_ROOT / TASK_DIRS[task] / model / f"sae_features_L{layer}.npz"
    if not npz_path.exists():
        return None
    d = np.load(npz_path, allow_pickle=False)
    n_rounds, n_feat = tuple(d["shape"])
    X = csr_matrix(
        (d["values"], (d["row_indices"], d["col_indices"])),
        shape=(n_rounds, n_feat), dtype=np.float32,
    )
    meta = {k: d[k] for k in d.keys()
            if k not in ("row_indices", "col_indices", "values", "shape")}
    return X, meta


def load_games(model: str, task: str):
    if task == "sm":
        if model == "gemma":
            paths = [BEHAV_ROOT / "slot_machine/gemma_v4_role/final_gemma_20260227_002507.json"]
        else:
            paths = [BEHAV_ROOT / "slot_machine/llama_v4_role/final_llama_20260315_062428.json"]
    elif task == "mw":
        sub = f"mystery_wheel/{'gemma_v2_role' if model == 'gemma' else 'llama_v2_role'}"
        paths = sorted((BEHAV_ROOT / sub).glob(f"{model}_mysterywheel_*.json"))
    elif task == "ic":
        sub = f"investment_choice/{'v2_role_gemma' if model == 'gemma' else 'v2_role_llama'}"
        paths = sorted((BEHAV_ROOT / sub).glob("*.json"))
    else:
        return []

    games = []
    for p in paths:
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        results = d.get("results", d.get("games", []))
        if isinstance(results, dict):
            games.extend(results.values())
        else:
            games.extend(results)
    return games


# -------------------- Round-level labels (V17 canonical) --------------------

def compute_round_labels(meta, games):
    """Compute per-round labels matching V17.

    I_BA: bet/balance (continuous in [0,1])
    I_LC: 1 if (prev round was loss AND bet_ratio > prev_bet_ratio) else 0
    I_EC: 1 if bet_ratio >= 0.5 else 0
    """
    game_map = {i + 1: g for i, g in enumerate(games)}
    n = len(meta["game_ids"])
    i_ba = np.full(n, np.nan)
    i_lc = np.full(n, np.nan)
    i_ec = np.full(n, np.nan)
    balances_out = (
        meta["balances"].astype(float).copy()
        if meta.get("balances") is not None else np.full(n, np.nan)
    )

    for i in range(n):
        gid = int(meta["game_ids"][i])
        rn = int(meta["round_nums"][i]) - 1  # 0-indexed list position
        g = game_map.get(gid)
        if g is None:
            continue
        raw_decs = g.get("decisions", g.get("history", g.get("rounds", [])))
        decs = [d for d in raw_decs
                if d.get("action") != "skip" and not d.get("skipped", False)]
        hist = g.get("history", decs)
        if rn < 0 or rn >= len(decs):
            continue

        dec = decs[rn]
        bet_val = dec.get("parsed_bet") or dec.get("bet") or dec.get("bet_amount")
        bal_val = dec.get("balance_before") or dec.get("balance")
        if bet_val is None:
            continue
        try:
            bet = float(bet_val)
            bal = float(bal_val) if bal_val is not None else float(balances_out[i])
        except (ValueError, TypeError):
            continue
        if not (bal > 0 and bet > 0):
            continue
        balances_out[i] = bal
        br = min(bet / bal, 1.0)
        i_ba[i] = br
        i_ec[i] = 1.0 if br >= 0.5 else 0.0

        # I_LC: binary, requires previous round + previous-round-was-loss
        if rn >= 1:
            prev_dec = decs[rn - 1]
            p_bet = prev_dec.get("parsed_bet") or prev_dec.get("bet") or prev_dec.get("bet_amount")
            p_bal = prev_dec.get("balance_before") or prev_dec.get("balance")
            if p_bet is not None and p_bal is not None:
                try:
                    p_bet, p_bal = float(p_bet), float(p_bal)
                except (ValueError, TypeError):
                    p_bet = p_bal = None
                if p_bet and p_bal and p_bet > 0 and p_bal > 0:
                    p_br = min(p_bet / p_bal, 1.0)
                    prev_loss = False
                    if rn - 1 < len(hist):
                        h = hist[rn - 1]
                        # win flag, default True (so missing flag -> not loss)
                        prev_loss = not h.get("win", str(h.get("result", "")) == "W")
                    i_lc[i] = 1.0 if (prev_loss and br > p_br) else 0.0

    return {"i_ba": i_ba, "i_lc": i_lc, "i_ec": i_ec, "balances": balances_out}


# -------------------- Methodology helpers --------------------

def _make_cov(bal, rn):
    """V17 nonlinear control covariates."""
    return np.column_stack([bal, rn, bal ** 2, np.log1p(bal), bal * rn])


def nl_deconfound_split(y_tr, bal_tr, rn_tr, y_te, bal_te, rn_te,
                        n_estimators=RF_TREES, max_depth=RF_DEPTH):
    """Within-fold RF deconfound. Train RF on train; residualize both."""
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                               random_state=42, n_jobs=-1)
    rf.fit(_make_cov(bal_tr, rn_tr), y_tr)
    return (
        y_tr - rf.predict(_make_cov(bal_tr, rn_tr)),
        y_te - rf.predict(_make_cov(bal_te, rn_te)),
    )


def cv_r2_with_deconfound(X_v, y, bal, rn, top_k=TOP_K, alpha=RIDGE_ALPHA,
                          n_splits=5, n_perm=0, gids=None):
    """5-fold CV R² with within-fold RF deconfound and Top-K screening.

    Returns dict with r2, n, null_mean, null_95, p (if n_perm > 0).
    """
    n = X_v.shape[0]
    if n < 100:
        return {"r2": None, "n": int(n), "reason": "n_too_small"}

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_r2 = []
    rng = np.random.default_rng(42)

    null_pool = [] if n_perm > 0 else None

    for tr, te in kf.split(X_v):
        # Deconfound y within fold
        y_tr_r, y_te_r = nl_deconfound_split(
            y[tr], bal[tr], rn[tr], y[te], bal[te], rn[te]
        )
        # Top-K screening on training residual
        y_tr_c = y_tr_r - y_tr_r.mean()
        # X_v is sparse; X_v[tr] is also sparse
        Xt_tr = X_v[tr]
        Xt_te = X_v[te]
        num = Xt_tr.T.dot(y_tr_c)  # numerator of correlation
        # standard deviation per feature (training only) -- centred
        col_sum = np.asarray(Xt_tr.sum(axis=0)).ravel()
        col_sq = np.asarray(Xt_tr.power(2).sum(axis=0)).ravel()
        n_tr = Xt_tr.shape[0]
        x_var = np.maximum(col_sq / n_tr - (col_sum / n_tr) ** 2, 1e-12)
        y_var = max(float(np.var(y_tr_r)), 1e-12)
        denom = np.sqrt(x_var * y_var) * n_tr
        corrs = num / np.maximum(denom, 1e-12)
        k = min(top_k, X_v.shape[1])
        top_idx = np.argsort(-np.abs(corrs))[:k]
        X_tr = Xt_tr[:, top_idx].toarray()
        X_te = Xt_te[:, top_idx].toarray()

        ridge = Ridge(alpha=alpha, solver="lsqr").fit(X_tr, y_tr_r)
        pred = ridge.predict(X_te)
        ss_res = float(np.sum((y_te_r - pred) ** 2))
        y_te_c = y_te_r - y_te_r.mean()
        ss_tot = float(np.sum(y_te_c ** 2))
        if ss_tot < 1e-12:
            continue
        fold_r2.append(1.0 - ss_res / ss_tot)

        # Permutation null (game-block shuffle within this fold)
        if n_perm > 0 and gids is not None:
            for _ in range(n_perm):
                # Block shuffle: permute game ids in train
                tr_gids = gids[tr]
                unique = np.unique(tr_gids)
                perm_unique = rng.permutation(unique)
                gid_map = dict(zip(unique, perm_unique))
                # Construct permuted target vector by remapping each row's game id
                y_perm = np.empty_like(y_tr_r)
                for g in unique:
                    mask = tr_gids == g
                    src_g = gid_map[g]
                    src_mask = tr_gids == src_g
                    y_perm[mask] = rng.choice(y_tr_r[src_mask], size=mask.sum(), replace=True)
                ridge_p = Ridge(alpha=alpha, solver="lsqr").fit(X_tr, y_perm)
                pred_p = ridge_p.predict(X_te)
                ss_res_p = float(np.sum((y_te_r - pred_p) ** 2))
                null_pool.append(1.0 - ss_res_p / ss_tot)

    if not fold_r2:
        return {"r2": None, "n": int(n), "reason": "all_folds_degenerate"}
    r2_mean = float(np.mean(fold_r2))
    out = {"r2": r2_mean, "n": int(n), "n_folds": len(fold_r2)}
    if null_pool:
        arr = np.asarray(null_pool)
        out.update({
            "null_mean": float(arr.mean()),
            "null_95": float(np.percentile(arr, 95)),
            "p": float(np.mean(arr >= r2_mean)),
        })
    return out


# -------------------- Per-cell driver --------------------

def run_one_cell(model: str, task: str, layer: int, games,
                 metrics=("i_lc", "i_ba", "i_ec"), n_perm=0, top_k=TOP_K,
                 alpha=RIDGE_ALPHA):
    out = {"model": model, "task": task, "layer": layer}
    feat = load_features(model, task, layer)
    if feat is None:
        out["error"] = "missing_features"
        return out
    X, meta = feat
    labels = compute_round_labels(meta, games)
    bt = meta.get("bet_types")
    rn_arr = meta["round_nums"].astype(float)
    bal_arr = labels["balances"].astype(float)
    gids = meta["game_ids"]

    for metric in metrics:
        y = labels[metric]
        valid = ~np.isnan(y) & ~np.isnan(bal_arr) & (bal_arr > 0)
        # Filter to variable betting (canonical V17 filter)
        if bt is not None:
            valid = valid & (bt == "variable")
        if metric == "i_ba":
            valid = valid & (y > 0)
        if valid.sum() < 100:
            out[metric] = {"r2": None, "n": int(valid.sum()), "reason": "filtered_n_too_small"}
            continue
        X_v = X[valid]
        y_v = y[valid]
        bal_v = bal_arr[valid]
        rn_v = rn_arr[valid]
        gids_v = gids[valid] if gids is not None else None
        out[metric] = cv_r2_with_deconfound(
            X_v, y_v, bal_v, rn_v, top_k=top_k, alpha=alpha,
            n_perm=n_perm, gids=gids_v,
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gemma,llama")
    ap.add_argument("--tasks", default="sm,ic,mw")
    ap.add_argument("--layers", default="all", help="all|comma-sep|range:start:end:step")
    ap.add_argument("--n_perm", type=int, default=0)
    ap.add_argument("--top_k", type=int, default=TOP_K)
    ap.add_argument("--alpha", type=float, default=RIDGE_ALPHA)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    models = args.models.split(",")
    tasks = args.tasks.split(",")
    out_path = (Path(args.out) if args.out else
                OUT_DIR / f"sweep_{datetime.now():%Y%m%d_%H%M%S}.jsonl")

    print(f"[sweep] V17-matched methodology: top_k={args.top_k} alpha={args.alpha} n_perm={args.n_perm}", flush=True)
    print(f"[sweep] writing to {out_path}", flush=True)
    print(f"[sweep] models={models} tasks={tasks}", flush=True)

    with open(out_path, "w") as f_out:
        for model in models:
            base_layers = GEMMA_LAYERS if model == "gemma" else LLAMA_LAYERS
            if args.layers == "all":
                layers = base_layers
            elif args.layers.startswith("range:"):
                parts = args.layers.split(":")
                step = int(parts[3]) if len(parts) > 3 else 1
                layers = list(range(int(parts[1]), int(parts[2]), step))
            else:
                layers = [int(x) for x in args.layers.split(",")]

            for task in tasks:
                print(f"[sweep] loading {model}/{task} games ...", flush=True)
                games = load_games(model, task)
                print(f"  loaded {len(games)} games", flush=True)
                for layer in layers:
                    t0 = time.time()
                    res = run_one_cell(
                        model, task, layer, games,
                        n_perm=args.n_perm, top_k=args.top_k, alpha=args.alpha,
                    )
                    elapsed = time.time() - t0
                    f_out.write(json.dumps(res) + "\n")
                    f_out.flush()

                    def _r2(r):
                        if isinstance(r, dict):
                            v = r.get("r2")
                            if isinstance(v, float):
                                return f"{v:+.3f}"
                            return r.get("reason", "n/a")[:8]
                        return "None"
                    print(
                        f"  {model}/{task}/L{layer:>2}: "
                        f"LC={_r2(res.get('i_lc'))} "
                        f"BA={_r2(res.get('i_ba'))} "
                        f"EC={_r2(res.get('i_ec'))} ({elapsed:.0f}s)",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
