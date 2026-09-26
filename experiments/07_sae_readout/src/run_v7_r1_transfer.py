#!/usr/bin/env python3
"""
V7 R1 Cross-Domain Transfer Analysis

Key question: Does BK prediction transfer across paradigms when balance is controlled?
At R1 (Round 1), all games have balance=$100, eliminating balance as confound.

Tests:
  1. Full LR transfer at R1 (hidden states): train on paradigm A R1, test on paradigm B R1
  2. Full LR transfer at DP (hidden states): comparison (known balance-confounded)
  3. Direction transfer at R1: 1D projection of LR weight vector
  4. Centroid transfer at R1: project onto BK-nonBK centroid direction
  5. Full LR transfer at R1 (SAE features): same as 1 but with SAE
  6. Multi-layer R1 transfer sweep

Usage:
    cd /home/jovyan/llm-addiction/sae_v3_analysis/src
    python run_v7_r1_transfer.py
"""

import gc
import json
import logging
import sys
import time
import numpy as np
from datetime import datetime
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")

from config import (
    PARADIGMS, N_LAYERS, HIDDEN_DIM,
    FIGURE_DIR, JSON_DIR, LOG_DIR,
    RANDOM_SEED,
)
from data_loader import (
    load_layer_features, load_sparse_npz, get_metadata, get_labels,
)

# ===================================================================
# Setup
# ===================================================================
LOG_DIR.mkdir(parents=True, exist_ok=True)
JSON_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

log_file = LOG_DIR / f"v7_r1_transfer_{datetime.now():%Y%m%d_%H%M%S}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="w"),
    ],
)
log = logging.getLogger(__name__)

PARADIGM_KEYS = ["ic", "sm", "mw"]
PAIRS = [("ic", "sm"), ("ic", "mw"), ("sm", "mw")]


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ===================================================================
# Data loading
# ===================================================================
def load_hidden_r1_dp(paradigm, layer):
    """Load hidden states for R1 and DP from checkpoint.
    Returns (r1_features, r1_labels, dp_features, dp_labels).
    """
    sae_dir = PARADIGMS[paradigm]["sae_dir"]
    ckpt_path = sae_dir / "checkpoint" / "phase_a_hidden_states.npz"
    if not ckpt_path.exists():
        log.error(f"Checkpoint not found: {ckpt_path}")
        return None, None, None, None

    t0 = time.time()
    ckpt = np.load(str(ckpt_path), allow_pickle=False, mmap_mode="r")
    hidden_layer = np.array(ckpt["hidden_states"][:, layer, :], dtype=np.float32)

    valid_mask = None
    if "valid_mask" in ckpt.files:
        valid_mask = np.array(ckpt["valid_mask"]).astype(bool)

    any_sae = sorted(sae_dir.glob("sae_features_L*.npz"))
    raw = load_sparse_npz(any_sae[0])
    meta = get_metadata(raw)
    del raw

    if valid_mask is not None:
        hidden_layer = hidden_layer[valid_mask]
        meta = {k: v[valid_mask] for k, v in meta.items()}

    # R1 mask
    round_nums = np.array(meta["round_nums"])
    r1_mask = round_nums == 1
    r1_features = hidden_layer[r1_mask]
    r1_outcomes = np.array(meta["game_outcomes"])[r1_mask]
    r1_labels = (r1_outcomes == "bankruptcy").astype(np.int32)

    # DP mask
    dp_mask = meta["is_last_round"]
    dp_features = hidden_layer[dp_mask]
    dp_meta = {k: v[dp_mask] for k, v in meta.items()}
    dp_labels = get_labels(dp_meta)

    del hidden_layer, ckpt
    gc.collect()

    log.info(f"  {paradigm} L{layer}: R1={r1_features.shape[0]}({r1_labels.sum()} BK), "
             f"DP={dp_features.shape[0]}({dp_labels.sum()} BK), {time.time()-t0:.1f}s")
    return r1_features, r1_labels, dp_features, dp_labels


def load_sae_r1(paradigm, layer):
    """Load SAE features for R1 rounds.
    Returns (r1_features_dense, r1_labels).
    """
    result = load_layer_features(paradigm, layer, mode="all_rounds", dense=True)
    if result is None:
        return None, None
    features, meta = result
    round_nums = np.array(meta["round_nums"])
    r1_mask = round_nums == 1
    r1_features = features[r1_mask]
    r1_outcomes = np.array(meta["game_outcomes"])[r1_mask]
    r1_labels = (r1_outcomes == "bankruptcy").astype(np.int32)
    del features
    gc.collect()
    return r1_features, r1_labels


# ===================================================================
# Transfer functions
# ===================================================================
def transfer_auc(X_train, y_train, X_test, y_test, use_pca=None):
    """Train LR on (X_train, y_train), test on (X_test, y_test). Returns AUC."""
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return 0.5
    if y_train.sum() < 2 or y_test.sum() < 2:
        return 0.5

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    if use_pca is not None and X_tr.shape[1] > use_pca:
        pca = PCA(n_components=use_pca, random_state=RANDOM_SEED)
        X_tr = pca.fit_transform(X_tr)
        X_te = pca.transform(X_te)

    clf = LogisticRegression(
        C=1.0, solver="liblinear", class_weight="balanced",
        random_state=RANDOM_SEED, max_iter=1000,
    )
    clf.fit(X_tr, y_train)
    proba = clf.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_test, proba)
    return float(auc)


def within_auc(X, y, n_folds=5):
    """5-fold CV AUC."""
    if len(np.unique(y)) < 2 or y.sum() < 2:
        return 0.5
    skf = StratifiedKFold(n_splits=min(n_folds, int(y.sum()), int(len(y)-y.sum())),
                          shuffle=True, random_state=RANDOM_SEED)
    aucs = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler()
        clf = LogisticRegression(C=1.0, solver="liblinear", class_weight="balanced",
                                  random_state=RANDOM_SEED, max_iter=1000)
        clf.fit(sc.fit_transform(X[tr]), y[tr])
        if len(np.unique(y[te])) < 2:
            continue
        aucs.append(roc_auc_score(y[te], clf.predict_proba(sc.transform(X[te]))[:, 1]))
    return float(np.mean(aucs)) if aucs else 0.5


def direction_transfer_auc(X_src, y_src, X_tgt, y_tgt):
    """1D projection: train LR on src, project tgt onto weight vector, compute AUC."""
    if len(np.unique(y_tgt)) < 2:
        return 0.5
    sc = StandardScaler()
    X_s = sc.fit_transform(X_src)
    clf = LogisticRegression(C=1.0, solver="liblinear", class_weight="balanced",
                              random_state=RANDOM_SEED, max_iter=1000)
    clf.fit(X_s, y_src)
    # Weight vector in original space
    w = clf.coef_[0] / sc.scale_
    w = w / np.linalg.norm(w)
    # Project target
    scores = X_tgt @ w
    auc = roc_auc_score(y_tgt, scores)
    return float(max(auc, 1 - auc))


# ===================================================================
# Main analyses
# ===================================================================
def main():
    t_start = time.time()
    log.info("V7 R1 Cross-Domain Transfer Analysis")
    log.info(f"Start: {datetime.now().isoformat()}")
    log.info(f"Question: Does BK prediction transfer across paradigms at R1 (balance-controlled)?")
    sys.stdout.flush()

    results = {}

    # ---------------------------------------------------------------
    # 1. Load hidden states at best layers: R1 + DP
    # ---------------------------------------------------------------
    best_layers = {"ic": 22, "sm": 16, "mw": 22}
    log.info("=" * 70)
    log.info(f"Loading hidden states at best layers: {best_layers}")
    log.info("=" * 70)

    hs = {}  # paradigm -> {r1: (X, y), dp: (X, y)}
    for p in PARADIGM_KEYS:
        r1_f, r1_l, dp_f, dp_l = load_hidden_r1_dp(p, best_layers[p])
        if r1_f is None:
            log.error(f"Failed to load {p}")
            return
        hs[p] = {"r1": (r1_f, r1_l), "dp": (dp_f, dp_l)}

    # ---------------------------------------------------------------
    # 2. Within-domain AUC (sanity check)
    # ---------------------------------------------------------------
    log.info("\n" + "=" * 70)
    log.info("WITHIN-DOMAIN AUC (hidden states)")
    log.info("=" * 70)
    within_results = {}
    for p in PARADIGM_KEYS:
        for mode in ["r1", "dp"]:
            X, y = hs[p][mode]
            auc = within_auc(X, y)
            within_results[f"{p}_{mode}"] = round(auc, 4)
            log.info(f"  {p} {mode.upper()}: AUC={auc:.4f} (n={len(y)}, BK={y.sum()})")
    results["within_domain_hidden"] = within_results

    # ---------------------------------------------------------------
    # 3. Cross-domain FULL LR transfer: R1 vs DP
    # ---------------------------------------------------------------
    log.info("\n" + "=" * 70)
    log.info("CROSS-DOMAIN FULL LR TRANSFER (hidden states)")
    log.info("=" * 70)

    transfer_results = {"r1": {}, "dp": {}}
    for mode in ["r1", "dp"]:
        log.info(f"\n  --- {mode.upper()} transfer ---")
        for p_src in PARADIGM_KEYS:
            for p_tgt in PARADIGM_KEYS:
                if p_src == p_tgt:
                    continue
                X_tr, y_tr = hs[p_src][mode]
                X_te, y_te = hs[p_tgt][mode]
                auc = transfer_auc(X_tr, y_tr, X_te, y_te)
                key = f"{p_src}_to_{p_tgt}"
                transfer_results[mode][key] = round(auc, 4)
                log.info(f"    {p_src.upper()} → {p_tgt.upper()}: AUC={auc:.4f}")
    results["cross_domain_lr_transfer"] = transfer_results

    # ---------------------------------------------------------------
    # 4. Direction transfer: R1 vs DP
    # ---------------------------------------------------------------
    log.info("\n" + "=" * 70)
    log.info("DIRECTION TRANSFER (1D projection, hidden states)")
    log.info("=" * 70)

    dir_results = {"r1": {}, "dp": {}}
    for mode in ["r1", "dp"]:
        log.info(f"\n  --- {mode.upper()} direction transfer ---")
        for p_src in PARADIGM_KEYS:
            for p_tgt in PARADIGM_KEYS:
                if p_src == p_tgt:
                    continue
                X_src, y_src = hs[p_src][mode]
                X_tgt, y_tgt = hs[p_tgt][mode]
                auc = direction_transfer_auc(X_src, y_src, X_tgt, y_tgt)
                key = f"{p_src}_to_{p_tgt}"
                dir_results[mode][key] = round(auc, 4)
                log.info(f"    {p_src.upper()} → {p_tgt.upper()}: AUC={auc:.4f}")
    results["direction_transfer_hidden"] = dir_results

    # ---------------------------------------------------------------
    # 5. Cosine similarity of weight vectors: R1 vs DP
    # ---------------------------------------------------------------
    log.info("\n" + "=" * 70)
    log.info("WEIGHT VECTOR COSINE SIMILARITY (R1 vs DP)")
    log.info("=" * 70)

    cosine_results = {"r1": {}, "dp": {}}
    for mode in ["r1", "dp"]:
        weights = {}
        for p in PARADIGM_KEYS:
            X, y = hs[p][mode]
            sc = StandardScaler()
            Xs = sc.fit_transform(X)
            clf = LogisticRegression(C=1.0, solver="liblinear", class_weight="balanced",
                                      random_state=RANDOM_SEED, max_iter=1000)
            clf.fit(Xs, y)
            w = clf.coef_[0] / sc.scale_
            w = w / np.linalg.norm(w)
            weights[p] = w

        for (p1, p2) in PAIRS:
            cos = float(np.dot(weights[p1], weights[p2]))
            cosine_results[mode][f"{p1}_{p2}"] = round(cos, 4)
            log.info(f"  {mode.upper()} cos(w_{p1}, w_{p2}) = {cos:.4f}")
    results["weight_cosine"] = cosine_results

    # Free hidden states
    del hs
    gc.collect()

    # ---------------------------------------------------------------
    # 6. SAE-based R1 cross-domain transfer
    # ---------------------------------------------------------------
    log.info("\n" + "=" * 70)
    log.info("SAE R1 CROSS-DOMAIN TRANSFER (L22)")
    log.info("=" * 70)

    sae_layer = 22
    sae_r1 = {}
    for p in PARADIGM_KEYS:
        t0 = time.time()
        X, y = load_sae_r1(p, sae_layer)
        if X is None:
            log.error(f"Failed to load SAE R1 for {p}")
            continue
        # Filter to active features
        active_mask = (X != 0).mean(axis=0) >= 0.01
        X_active = X[:, active_mask]
        sae_r1[p] = (X_active, y)
        log.info(f"  {p}: R1 SAE L{sae_layer} shape={X_active.shape}, BK={y.sum()}/{len(y)}, "
                 f"{time.time()-t0:.1f}s")
        del X
        gc.collect()

    sae_within = {}
    sae_transfer = {}
    if len(sae_r1) == 3:
        # Within-domain
        for p in PARADIGM_KEYS:
            X, y = sae_r1[p]
            auc = within_auc(X, y)
            sae_within[p] = round(auc, 4)
            log.info(f"  SAE R1 within {p}: AUC={auc:.4f}")

        # Cross-domain
        for p_src in PARADIGM_KEYS:
            for p_tgt in PARADIGM_KEYS:
                if p_src == p_tgt:
                    continue
                X_tr, y_tr = sae_r1[p_src]
                X_te, y_te = sae_r1[p_tgt]
                # Need common feature space — use all features
                # Reload with shared indexing
                pass

        # SAE features are already in same space (131K features, same model)
        # But active features differ. Use union of active features.
        log.info("  Reloading SAE with full feature space for transfer...")
        sae_r1_full = {}
        for p in PARADIGM_KEYS:
            t0 = time.time()
            X, y = load_sae_r1(p, sae_layer)
            if X is None:
                continue
            sae_r1_full[p] = (X, y)
            log.info(f"  {p}: R1 SAE full shape={X.shape}, {time.time()-t0:.1f}s")

        if len(sae_r1_full) == 3:
            # Use PCA to make transfer feasible with 131K features
            for p_src in PARADIGM_KEYS:
                for p_tgt in PARADIGM_KEYS:
                    if p_src == p_tgt:
                        continue
                    X_tr, y_tr = sae_r1_full[p_src]
                    X_te, y_te = sae_r1_full[p_tgt]
                    auc = transfer_auc(X_tr, y_tr, X_te, y_te, use_pca=100)
                    key = f"{p_src}_to_{p_tgt}"
                    sae_transfer[key] = round(auc, 4)
                    log.info(f"  SAE R1 transfer {p_src.upper()} → {p_tgt.upper()}: AUC={auc:.4f}")

            del sae_r1_full
            gc.collect()

    results["sae_r1_transfer"] = {
        "layer": sae_layer,
        "within": sae_within,
        "cross_domain": sae_transfer,
    }

    del sae_r1
    gc.collect()

    # ---------------------------------------------------------------
    # 7. Multi-layer R1 transfer sweep (hidden states)
    # ---------------------------------------------------------------
    log.info("\n" + "=" * 70)
    log.info("MULTI-LAYER R1 HIDDEN STATE TRANSFER")
    log.info("=" * 70)

    sweep_layers = [0, 6, 10, 14, 18, 22, 26, 30, 34, 38, 41]
    layer_sweep = []

    for layer in sweep_layers:
        t0 = time.time()
        hs_l = {}
        for p in PARADIGM_KEYS:
            r1_f, r1_l, _, _ = load_hidden_r1_dp(p, layer)
            if r1_f is not None:
                hs_l[p] = (r1_f, r1_l)

        if len(hs_l) < 3:
            continue

        entry = {"layer": layer}
        # Within
        for p in PARADIGM_KEYS:
            X, y = hs_l[p]
            entry[f"within_{p}"] = round(within_auc(X, y), 4)

        # Transfer (all 6 directions)
        for p_src in PARADIGM_KEYS:
            for p_tgt in PARADIGM_KEYS:
                if p_src == p_tgt:
                    continue
                X_tr, y_tr = hs_l[p_src]
                X_te, y_te = hs_l[p_tgt]
                auc = transfer_auc(X_tr, y_tr, X_te, y_te)
                entry[f"{p_src}_to_{p_tgt}"] = round(auc, 4)

        layer_sweep.append(entry)
        log.info(f"  L{layer:>2d}: within IC={entry.get('within_ic', '?')}, "
                 f"IC→SM={entry.get('ic_to_sm', '?')}, IC→MW={entry.get('ic_to_mw', '?')}, "
                 f"SM→IC={entry.get('sm_to_ic', '?')} [{time.time()-t0:.1f}s]")

        del hs_l
        gc.collect()

    results["layer_sweep_r1_transfer"] = layer_sweep

    # ---------------------------------------------------------------
    # Save & Summary
    # ---------------------------------------------------------------
    output_path = JSON_DIR / "v7_r1_transfer.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    log.info(f"\nResults saved to {output_path}")

    elapsed = time.time() - t_start
    log.info("\n" + "=" * 70)
    log.info(f"COMPLETE in {elapsed/60:.1f} minutes")
    log.info("=" * 70)

    # Summary
    log.info("\n=== KEY FINDINGS ===")
    log.info("\nWithin-domain AUC (hidden states):")
    for k, v in sorted(results["within_domain_hidden"].items()):
        log.info(f"  {k}: {v}")

    log.info("\nCross-domain FULL LR transfer:")
    for mode in ["r1", "dp"]:
        log.info(f"  {mode.upper()}:")
        for k, v in sorted(results["cross_domain_lr_transfer"][mode].items()):
            log.info(f"    {k}: {v}")

    log.info("\nWeight vector cosine similarity:")
    for mode in ["r1", "dp"]:
        log.info(f"  {mode.upper()}:")
        for k, v in sorted(results["weight_cosine"][mode].items()):
            log.info(f"    {k}: {v}")

    log.info("\nSAE R1 transfer:")
    if results["sae_r1_transfer"]["cross_domain"]:
        for k, v in sorted(results["sae_r1_transfer"]["cross_domain"].items()):
            log.info(f"  {k}: {v}")


if __name__ == "__main__":
    main()
