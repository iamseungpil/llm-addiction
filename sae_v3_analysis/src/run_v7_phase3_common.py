#!/usr/bin/env python3
"""
V7 Phase 3: Deep analysis of common features and hidden state directions
across paradigms (IC, SM, MW) for Gemma-2-9B-IT.

Analyses:
  3a: Shared SAE feature classification — top-k sweep, Jaccard, shared-only AUC
  3b: Hidden state direction transfer — LR weight vectors, 1D projection, cosine
  3c: Layer-wise overlap evolution — SAE Jaccard + hidden cosine across 42 layers
  3d: BK centroid analysis — centroid difference vectors, cosine, transfer AUC

Data: Gemma-2-9B-IT, 3 paradigms, 131K SAE features, 3584-dim hidden states, 42 layers.

Usage:
    cd /home/jovyan/llm-addiction/sae_v3_analysis/src
    python run_v7_phase3_common.py
"""

import gc
import json
import logging
import sys
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import rankdata

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from config import (
    PARADIGMS, N_LAYERS, N_SAE_FEATURES, HIDDEN_DIM,
    FIGURE_DIR, JSON_DIR, LOG_DIR,
    MIN_ACTIVATION_RATE, RANDOM_SEED,
    PARADIGM_COLORS, PARADIGM_LABELS,
)
from data_loader import (
    load_layer_features, filter_active_features, get_labels,
    load_sparse_npz, sparse_to_dense, get_metadata,
)

# ===================================================================
# Setup
# ===================================================================
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "v7_phase3_common.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)

PAIRS = [("ic", "sm"), ("ic", "mw"), ("sm", "mw")]
PAIR_NAMES = ["ic_sm", "ic_mw", "sm_mw"]
PARADIGM_KEYS = ["ic", "sm", "mw"]


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ===================================================================
# Utility: fast vectorized Spearman correlation
# ===================================================================
def fast_spearman_abs(features, labels, min_activation_rate=0.01):
    """Vectorized |Spearman rho| for all features vs binary labels.

    Optimized: only computes correlations for features with activation rate
    >= min_activation_rate (others are all-zero and have rho=0).
    For 131K SAE features where only ~400-800 are active, this is ~100x faster.

    Returns array of shape (n_features,) with |rho| values.
    """
    n_features = features.shape[1]
    abs_rho = np.zeros(n_features, dtype=np.float64)

    # Filter to active features only
    active_rate = (features != 0).mean(axis=0)
    active_mask = active_rate >= min_activation_rate
    n_active = active_mask.sum()

    if n_active == 0:
        return abs_rho

    active_feats = features[:, active_mask]

    rank_y = rankdata(labels)
    rank_y_centered = rank_y - rank_y.mean()
    ss_y = np.dot(rank_y_centered, rank_y_centered)

    if ss_y == 0:
        return abs_rho

    abs_rho_active = np.zeros(n_active, dtype=np.float64)
    chunk_size = 5000
    for start in range(0, n_active, chunk_size):
        end = min(start + chunk_size, n_active)
        chunk = active_feats[:, start:end]
        rank_x = np.apply_along_axis(rankdata, 0, chunk)
        rank_x_centered = rank_x - rank_x.mean(axis=0, keepdims=True)
        ss_x = (rank_x_centered ** 2).sum(axis=0)
        numerator = rank_x_centered.T @ rank_y_centered
        denom = np.sqrt(ss_x * ss_y)
        denom = np.where(denom == 0, 1.0, denom)
        rho = numerator / denom
        abs_rho_active[start:end] = np.abs(rho)

    abs_rho[active_mask] = abs_rho_active
    return abs_rho


def get_topk_features(abs_rho, k):
    """Return indices of top-k features by |rho|, sorted descending."""
    return np.argsort(abs_rho)[-k:][::-1]


def classify_cv(features, labels, n_folds=5, seed=RANDOM_SEED):
    """5-fold stratified CV with balanced class weights. Returns AUC."""
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    effective_folds = min(n_folds, n_pos, n_neg)
    if effective_folds < 2:
        return 0.5

    skf = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=seed)
    aucs = []
    for train_idx, test_idx in skf.split(features, labels):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(features[train_idx])
        X_test = scaler.transform(features[test_idx])
        clf = LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=1000,
            class_weight="balanced", random_state=seed,
        )
        clf.fit(X_train, labels[train_idx])
        if len(np.unique(labels[test_idx])) < 2:
            continue
        proba = clf.predict_proba(X_test)[:, 1]
        aucs.append(roc_auc_score(labels[test_idx], proba))

    return float(np.mean(aucs)) if aucs else 0.5


def train_lr_full(features, labels, seed=RANDOM_SEED):
    """Train LR on full dataset and return clf + scaler."""
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    clf = LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=1000,
        class_weight="balanced", random_state=seed,
    )
    clf.fit(X, labels)
    return clf, scaler


def get_lr_direction(features, labels, seed=RANDOM_SEED):
    """Train LR and return normalized weight vector in original space."""
    clf, scaler = train_lr_full(features, labels, seed)
    w = clf.coef_[0] / scaler.scale_  # transform to original space
    w = w / np.linalg.norm(w)
    return w


# ===================================================================
# Hidden state loading: paradigm-at-a-time for layer sweeps
# ===================================================================
def load_all_dp_hidden_states(paradigm):
    """Load hidden states checkpoint and return decision-point data for ALL layers.

    Returns (dp_hidden_all_layers, dp_labels) where:
        dp_hidden_all_layers: ndarray (n_dp_games, n_layers, hidden_dim)
        dp_labels: ndarray (n_dp_games,) binary BK labels
    """
    sae_dir = PARADIGMS[paradigm]["sae_dir"]
    ckpt_path = sae_dir / "checkpoint" / "phase_a_hidden_states.npz"

    if not ckpt_path.exists():
        log.error(f"Checkpoint not found: {ckpt_path}")
        return None, None

    t0 = time.time()
    ckpt = np.load(str(ckpt_path), allow_pickle=False)
    hidden_all = ckpt["hidden_states"]  # (n_rounds, n_layers, hidden_dim)

    # Valid mask
    valid_mask = None
    if "valid_mask" in ckpt.files:
        valid_mask = ckpt["valid_mask"].astype(bool)

    # Get metadata from any SAE feature file
    any_sae = sorted(sae_dir.glob("sae_features_L*.npz"))
    raw = load_sparse_npz(any_sae[0])
    meta = get_metadata(raw)
    del raw

    # Apply valid mask
    if valid_mask is not None:
        hidden_all = hidden_all[valid_mask]
        meta = {k: v[valid_mask] for k, v in meta.items()}

    # Decision point mask
    dp_mask = meta["is_last_round"]
    dp_hidden = hidden_all[dp_mask]  # (n_dp, n_layers, hidden_dim)
    dp_meta = {k: v[dp_mask] for k, v in meta.items()}
    dp_labels = get_labels(dp_meta)

    del hidden_all, ckpt, meta
    gc.collect()

    log.info(f"  {paradigm}: loaded all layers DP hidden states "
             f"shape={dp_hidden.shape}, BK={dp_labels.sum()}/{len(dp_labels)}, "
             f"{time.time()-t0:.1f}s")
    return dp_hidden, dp_labels


def load_hidden_states_single_layer(paradigm, layer):
    """Load hidden states for a single paradigm/layer from mmap. Returns (dp_features, dp_labels)."""
    sae_dir = PARADIGMS[paradigm]["sae_dir"]
    ckpt_path = sae_dir / "checkpoint" / "phase_a_hidden_states.npz"

    if not ckpt_path.exists():
        log.error(f"Checkpoint not found: {ckpt_path}")
        return None, None

    ckpt = np.load(str(ckpt_path), allow_pickle=False, mmap_mode="r")
    hidden_all = ckpt["hidden_states"]
    features = np.array(hidden_all[:, layer, :], dtype=np.float32)

    valid_mask = None
    if "valid_mask" in ckpt.files:
        valid_mask = np.array(ckpt["valid_mask"]).astype(bool)

    any_sae = sorted(sae_dir.glob("sae_features_L*.npz"))
    raw = load_sparse_npz(any_sae[0])
    meta = get_metadata(raw)
    del raw

    if valid_mask is not None:
        features = features[valid_mask]
        meta = {k: v[valid_mask] for k, v in meta.items()}

    dp_mask = meta["is_last_round"]
    features_dp = features[dp_mask]
    labels = get_labels({k: v[dp_mask] for k, v in meta.items()})

    del features, ckpt
    gc.collect()

    return features_dp, labels


# ===================================================================
# Analysis 3a: Shared SAE feature classification
# ===================================================================
def analysis_3a(target_layer=22):
    """Shared SAE feature classification at target layer with top-k sweep."""
    log.info("=" * 70)
    log.info(f"ANALYSIS 3a: Shared SAE feature classification at L{target_layer}")
    log.info("=" * 70)

    # Load decision-point features for each paradigm
    data = {}
    for p in PARADIGM_KEYS:
        t0 = time.time()
        result = load_layer_features(p, target_layer, mode="decision_point", dense=True)
        if result is None:
            log.error(f"Failed to load L{target_layer} for {p}")
            return None
        features, meta = result
        labels = get_labels(meta)
        log.info(f"  {p}: features={features.shape}, BK={labels.sum()}/{len(labels)}, "
                 f"loaded in {time.time()-t0:.1f}s")
        data[p] = {"features": features, "labels": labels}

    # Compute |Spearman rho| for all features — vectorized
    log.info("Computing Spearman correlations (vectorized)...")
    abs_rho = {}
    for p in PARADIGM_KEYS:
        t0 = time.time()
        abs_rho[p] = fast_spearman_abs(data[p]["features"], data[p]["labels"])
        top1_idx = np.argmax(abs_rho[p])
        log.info(f"  {p}: max |rho|={abs_rho[p][top1_idx]:.4f} (feature #{top1_idx}), "
                 f"computed in {time.time()-t0:.1f}s")

    # Top-k sweep
    k_values = [10, 20, 50, 100, 200, 500, 1000, 2000]
    sweep_results = []

    for k in k_values:
        t0 = time.time()
        topk = {p: set(get_topk_features(abs_rho[p], k)) for p in PARADIGM_KEYS}

        # Intersections
        ic_sm = topk["ic"] & topk["sm"]
        ic_mw = topk["ic"] & topk["mw"]
        sm_mw = topk["sm"] & topk["mw"]
        all3 = topk["ic"] & topk["sm"] & topk["mw"]

        # Jaccard indices
        jaccard_ic_sm = len(ic_sm) / len(topk["ic"] | topk["sm"]) if len(topk["ic"] | topk["sm"]) > 0 else 0
        jaccard_ic_mw = len(ic_mw) / len(topk["ic"] | topk["mw"]) if len(topk["ic"] | topk["mw"]) > 0 else 0
        jaccard_sm_mw = len(sm_mw) / len(topk["sm"] | topk["mw"]) if len(topk["sm"] | topk["mw"]) > 0 else 0

        # Union of all shared features (appear in >= 2 paradigms)
        shared_any2 = ic_sm | ic_mw | sm_mw

        # Classify using ONLY shared (any-2) features
        auc_shared = {}
        for p in PARADIGM_KEYS:
            if len(shared_any2) >= 2:
                shared_idx = sorted(shared_any2)
                X_shared = data[p]["features"][:, shared_idx]
                auc_shared[p] = classify_cv(X_shared, data[p]["labels"])
            else:
                auc_shared[p] = 0.5

        entry = {
            "k": k,
            "n_ic_sm": len(ic_sm), "n_ic_mw": len(ic_mw),
            "n_sm_mw": len(sm_mw), "n_all3": len(all3),
            "n_shared_any2": len(shared_any2),
            "jaccard_ic_sm": round(jaccard_ic_sm, 4),
            "jaccard_ic_mw": round(jaccard_ic_mw, 4),
            "jaccard_sm_mw": round(jaccard_sm_mw, 4),
            "auc_shared_ic": round(auc_shared["ic"], 4),
            "auc_shared_sm": round(auc_shared["sm"], 4),
            "auc_shared_mw": round(auc_shared["mw"], 4),
        }
        sweep_results.append(entry)
        log.info(f"  k={k:>5d}: IC∩SM={len(ic_sm):>4d}, IC∩MW={len(ic_mw):>4d}, "
                 f"SM∩MW={len(sm_mw):>4d}, all3={len(all3):>3d}, "
                 f"J(IC,SM)={jaccard_ic_sm:.3f}, "
                 f"AUC(shared): IC={auc_shared['ic']:.3f} SM={auc_shared['sm']:.3f} "
                 f"MW={auc_shared['mw']:.3f} [{time.time()-t0:.1f}s]")

    # Full-feature AUC for reference
    log.info("Computing full-feature AUC for comparison...")
    full_auc = {}
    for p in PARADIGM_KEYS:
        filt, _ = filter_active_features(data[p]["features"], min_rate=MIN_ACTIVATION_RATE)
        full_auc[p] = classify_cv(filt, data[p]["labels"])
        log.info(f"  {p} full AUC: {full_auc[p]:.4f}")

    # All-3 shared features at k=100
    topk100 = {p: set(get_topk_features(abs_rho[p], 100)) for p in PARADIGM_KEYS}
    all3_at_100 = sorted(topk100["ic"] & topk100["sm"] & topk100["mw"])

    all3_auc = {}
    if len(all3_at_100) >= 2:
        for p in PARADIGM_KEYS:
            X_all3 = data[p]["features"][:, all3_at_100]
            all3_auc[p] = classify_cv(X_all3, data[p]["labels"])
    elif len(all3_at_100) == 1:
        # 1 feature: use AUC directly from that feature
        for p in PARADIGM_KEYS:
            scores = data[p]["features"][:, all3_at_100[0]]
            auc = roc_auc_score(data[p]["labels"], scores)
            all3_auc[p] = max(auc, 1 - auc)
    else:
        all3_auc = {p: 0.5 for p in PARADIGM_KEYS}

    log.info(f"All-3 shared features at k=100: {all3_at_100}")
    log.info(f"All-3 shared AUC: {all3_auc}")

    result = {
        f"L{target_layer}": {
            "top_k_sweep": sweep_results,
            "full_feature_auc": {p: round(full_auc[p], 4) for p in PARADIGM_KEYS},
            "all3_shared_features": all3_at_100,
            "all3_shared_auc": {p: round(all3_auc.get(p, 0.5), 4) for p in PARADIGM_KEYS},
        }
    }

    del data, abs_rho
    gc.collect()
    return result


# ===================================================================
# Analysis 3b: Hidden state direction transfer
# ===================================================================
def analysis_3b(best_layers=None):
    """Hidden state direction transfer analysis."""
    log.info("=" * 70)
    log.info("ANALYSIS 3b: Hidden state direction transfer")
    log.info("=" * 70)

    if best_layers is None:
        best_layers = {"ic": 22, "sm": 12, "mw": 33}

    weights = {}   # paradigm -> normalized weight vector (3584,)
    hs_data = {}   # paradigm -> (features_dp, labels)

    for p in PARADIGM_KEYS:
        layer = best_layers[p]
        log.info(f"Loading hidden states for {p} at L{layer}...")
        t0 = time.time()
        features, labels = load_hidden_states_single_layer(p, layer)
        if features is None:
            log.error(f"Failed to load hidden states for {p}")
            return None
        log.info(f"  {p} L{layer}: shape={features.shape}, BK={labels.sum()}/{len(labels)}, "
                 f"loaded in {time.time()-t0:.1f}s")

        w = get_lr_direction(features, labels)
        weights[p] = w
        hs_data[p] = (features, labels)

        cv_auc = classify_cv(features, labels)
        log.info(f"  LR CV AUC: {cv_auc:.4f}")

    # Cosine similarity between weight vectors
    cosine_sim = {}
    for (p1, p2), name in zip(PAIRS, PAIR_NAMES):
        cos = float(np.dot(weights[p1], weights[p2]))
        cosine_sim[name] = round(cos, 4)
        log.info(f"  cos(w_{p1}, w_{p2}) = {cos:.4f}")

    # 1D projection transfer
    projection_auc = {}
    for p_src in PARADIGM_KEYS:
        w_src = weights[p_src]
        for p_tgt in PARADIGM_KEYS:
            if p_src == p_tgt:
                continue
            X_tgt, y_tgt = hs_data[p_tgt]
            scores = X_tgt @ w_src
            if len(np.unique(y_tgt)) >= 2:
                auc = roc_auc_score(y_tgt, scores)
                auc = max(auc, 1 - auc)
            else:
                auc = 0.5
            key = f"{p_src}_direction_on_{p_tgt}"
            projection_auc[key] = round(float(auc), 4)
            log.info(f"  1D projection: {key} = AUC {auc:.4f}")

    # Multi-layer direction analysis — load each paradigm's full hidden states ONCE
    sweep_layers = [0, 3, 6, 10, 12, 15, 18, 22, 26, 30, 33, 37, 40]
    log.info(f"Layer-wise direction analysis at {len(sweep_layers)} layers...")
    log.info("Loading full hidden state checkpoints (one per paradigm)...")

    # Pre-load all paradigm hidden states
    all_dp_hidden = {}  # paradigm -> (dp_hidden_all_layers, dp_labels)
    for p in PARADIGM_KEYS:
        dp_hidden, dp_labels = load_all_dp_hidden_states(p)
        if dp_hidden is None:
            log.error(f"Failed to load full hidden states for {p}")
            continue
        all_dp_hidden[p] = (dp_hidden, dp_labels)

    layer_wise_directions = []
    for layer in sweep_layers:
        t0 = time.time()
        layer_weights = {}
        for p in PARADIGM_KEYS:
            if p not in all_dp_hidden:
                continue
            dp_hidden, dp_labels = all_dp_hidden[p]
            features = dp_hidden[:, layer, :]
            w = get_lr_direction(features, dp_labels)
            layer_weights[p] = w

        entry = {"layer": layer}
        for (p1, p2), name in zip(PAIRS, PAIR_NAMES):
            if p1 in layer_weights and p2 in layer_weights:
                cos = float(np.dot(layer_weights[p1], layer_weights[p2]))
                entry[f"cosine_{name}"] = round(cos, 4)
            else:
                entry[f"cosine_{name}"] = None

        layer_wise_directions.append(entry)
        log.info(f"  L{layer:>2d}: cos(IC,SM)={entry.get('cosine_ic_sm', '?')}, "
                 f"cos(IC,MW)={entry.get('cosine_ic_mw', '?')}, "
                 f"cos(SM,MW)={entry.get('cosine_sm_mw', '?')} [{time.time()-t0:.1f}s]")

    del weights, hs_data, all_dp_hidden
    gc.collect()

    return {
        "best_layers": best_layers,
        "weight_cosine_similarity": cosine_sim,
        "1d_projection_auc": projection_auc,
        "layer_wise": layer_wise_directions,
    }


# ===================================================================
# Analysis 3c: Layer-wise overlap evolution (SAE Jaccard + hidden cosine)
# ===================================================================
def analysis_3c():
    """Layer-wise SAE Jaccard and hidden state cosine similarity for ALL 42 layers."""
    log.info("=" * 70)
    log.info("ANALYSIS 3c: Layer-wise overlap evolution (42 layers)")
    log.info("=" * 70)

    TOP_K = 100

    # Step 1: Pre-load all hidden states (one load per paradigm)
    log.info("Step 1: Loading hidden states for all paradigms...")
    all_dp_hidden = {}
    for p in PARADIGM_KEYS:
        dp_hidden, dp_labels = load_all_dp_hidden_states(p)
        if dp_hidden is None:
            log.error(f"Failed to load hidden states for {p}")
            continue
        all_dp_hidden[p] = (dp_hidden, dp_labels)

    # Step 2: Iterate over layers
    results = []
    for layer in range(N_LAYERS):
        t0 = time.time()
        entry = {"layer": layer}

        # --- SAE Jaccard ---
        topk_sets = {}
        for p in PARADIGM_KEYS:
            result = load_layer_features(p, layer, mode="decision_point", dense=True)
            if result is None:
                continue
            features, meta = result
            labels = get_labels(meta)
            rho = fast_spearman_abs(features, labels)
            top_idx = get_topk_features(rho, TOP_K)
            topk_sets[p] = set(top_idx)
            del features, meta, rho
            gc.collect()

        for (p1, p2), name in zip(PAIRS, PAIR_NAMES):
            if p1 in topk_sets and p2 in topk_sets:
                inter = len(topk_sets[p1] & topk_sets[p2])
                union = len(topk_sets[p1] | topk_sets[p2])
                entry[f"jaccard_{name}"] = round(inter / union if union > 0 else 0, 4)
                entry[f"shared_{name}"] = inter
            else:
                entry[f"jaccard_{name}"] = None
                entry[f"shared_{name}"] = None

        # --- Hidden state cosine similarity ---
        layer_weights = {}
        for p in PARADIGM_KEYS:
            if p not in all_dp_hidden:
                continue
            dp_hidden, dp_labels = all_dp_hidden[p]
            features = dp_hidden[:, layer, :]
            if dp_labels.sum() < 2 or (len(dp_labels) - dp_labels.sum()) < 2:
                continue
            w = get_lr_direction(features, dp_labels)
            layer_weights[p] = w

        for (p1, p2), name in zip(PAIRS, PAIR_NAMES):
            if p1 in layer_weights and p2 in layer_weights:
                cos = float(np.dot(layer_weights[p1], layer_weights[p2]))
                entry[f"cosine_{name}"] = round(cos, 4)
            else:
                entry[f"cosine_{name}"] = None

        results.append(entry)
        j_ic_sm = entry.get("jaccard_ic_sm", "?")
        c_ic_sm = entry.get("cosine_ic_sm", "?")
        log.info(f"  L{layer:>2d}: J(IC,SM)={j_ic_sm}, cos(IC,SM)={c_ic_sm} [{time.time()-t0:.1f}s]")

        del topk_sets, layer_weights
        gc.collect()

    del all_dp_hidden
    gc.collect()

    return results


# ===================================================================
# Analysis 3d: BK centroid analysis
# ===================================================================
def analysis_3d(best_layers=None):
    """BK centroid analysis: centroid difference vectors and transfer."""
    log.info("=" * 70)
    log.info("ANALYSIS 3d: BK centroid analysis")
    log.info("=" * 70)

    if best_layers is None:
        best_layers = {"ic": 22, "sm": 12, "mw": 33}

    centroids = {}
    hs_data = {}

    for p in PARADIGM_KEYS:
        layer = best_layers[p]
        log.info(f"Loading hidden states for {p} at L{layer}...")
        t0 = time.time()
        features, labels = load_hidden_states_single_layer(p, layer)
        if features is None:
            log.error(f"Failed to load hidden states for {p}")
            return None
        log.info(f"  {p} L{layer}: shape={features.shape}, BK={labels.sum()}/{len(labels)}, "
                 f"loaded in {time.time()-t0:.1f}s")

        mean_bk = features[labels == 1].mean(axis=0)
        mean_nbk = features[labels == 0].mean(axis=0)
        direction = mean_bk - mean_nbk
        direction_norm = direction / np.linalg.norm(direction)

        centroids[p] = direction_norm
        hs_data[p] = (features, labels)
        log.info(f"  ||centroid_diff|| = {np.linalg.norm(direction):.4f}")

    # Cosine similarity of centroid directions
    centroid_cosine = {}
    for (p1, p2), name in zip(PAIRS, PAIR_NAMES):
        cos = float(np.dot(centroids[p1], centroids[p2]))
        centroid_cosine[name] = round(cos, 4)
        log.info(f"  centroid cos({p1}, {p2}) = {cos:.4f}")

    # Transfer AUC
    centroid_transfer_auc = {}
    for p_src in PARADIGM_KEYS:
        for p_tgt in PARADIGM_KEYS:
            if p_src == p_tgt:
                continue
            X_tgt, y_tgt = hs_data[p_tgt]
            scores = X_tgt @ centroids[p_src]
            if len(np.unique(y_tgt)) >= 2:
                auc = roc_auc_score(y_tgt, scores)
                auc = max(auc, 1 - auc)
            else:
                auc = 0.5
            key = f"{p_src}_on_{p_tgt}"
            centroid_transfer_auc[key] = round(float(auc), 4)
            log.info(f"  centroid transfer: {key} = AUC {auc:.4f}")

    # Self AUC (sanity check)
    centroid_self_auc = {}
    for p in PARADIGM_KEYS:
        X, y = hs_data[p]
        scores = X @ centroids[p]
        auc = roc_auc_score(y, scores) if len(np.unique(y)) >= 2 else 0.5
        auc = max(auc, 1 - auc)
        centroid_self_auc[p] = round(float(auc), 4)
        log.info(f"  centroid self AUC ({p}): {auc:.4f}")

    # Multi-layer centroid cosine sweep — load full hidden states once
    log.info("Centroid cosine layer sweep...")
    sweep_layers = [0, 6, 10, 12, 18, 22, 26, 33, 40]

    all_dp_hidden = {}
    for p in PARADIGM_KEYS:
        dp_hidden, dp_labels = load_all_dp_hidden_states(p)
        if dp_hidden is not None:
            all_dp_hidden[p] = (dp_hidden, dp_labels)

    centroid_layer_sweep = []
    for layer in sweep_layers:
        t0 = time.time()
        layer_centroids = {}
        for p in PARADIGM_KEYS:
            if p not in all_dp_hidden:
                continue
            dp_hidden, dp_labels = all_dp_hidden[p]
            features = dp_hidden[:, layer, :]
            if dp_labels.sum() < 2:
                continue
            mean_bk = features[dp_labels == 1].mean(axis=0)
            mean_nbk = features[dp_labels == 0].mean(axis=0)
            d = mean_bk - mean_nbk
            d_norm = d / np.linalg.norm(d)
            layer_centroids[p] = d_norm

        entry = {"layer": layer}
        for (p1, p2), name in zip(PAIRS, PAIR_NAMES):
            if p1 in layer_centroids and p2 in layer_centroids:
                cos = float(np.dot(layer_centroids[p1], layer_centroids[p2]))
                entry[f"centroid_cosine_{name}"] = round(cos, 4)
            else:
                entry[f"centroid_cosine_{name}"] = None
        centroid_layer_sweep.append(entry)
        log.info(f"  L{layer}: centroid cos(IC,SM)={entry.get('centroid_cosine_ic_sm', '?')} "
                 f"[{time.time()-t0:.1f}s]")

    del hs_data, all_dp_hidden
    gc.collect()

    return {
        "best_layers": best_layers,
        "centroid_cosine": centroid_cosine,
        "centroid_transfer_auc": centroid_transfer_auc,
        "centroid_self_auc": centroid_self_auc,
        "centroid_layer_sweep": centroid_layer_sweep,
    }


# ===================================================================
# Visualization
# ===================================================================
def plot_3a_sweep(sweep_data, save_path):
    """Plot top-k sweep: Jaccard and shared-feature AUC."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ks = [d["k"] for d in sweep_data]

    ax = axes[0]
    ax.plot(ks, [d["jaccard_ic_sm"] for d in sweep_data], "o-", label="IC-SM", color="#e74c3c")
    ax.plot(ks, [d["jaccard_ic_mw"] for d in sweep_data], "s-", label="IC-MW", color="#3498db")
    ax.plot(ks, [d["jaccard_sm_mw"] for d in sweep_data], "^-", label="SM-MW", color="#2ecc71")
    ax.set_xlabel("Top-k features per paradigm")
    ax.set_ylabel("Jaccard Index")
    ax.set_title("Pairwise Feature Overlap vs Top-k")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(ks, [d["auc_shared_ic"] for d in sweep_data], "o-", label="IC", color=PARADIGM_COLORS["ic"])
    ax.plot(ks, [d["auc_shared_sm"] for d in sweep_data], "s-", label="SM", color=PARADIGM_COLORS["sm"])
    ax.plot(ks, [d["auc_shared_mw"] for d in sweep_data], "^-", label="MW", color=PARADIGM_COLORS["mw"])
    ax.axhline(0.5, ls="--", color="gray", alpha=0.5, label="Chance")
    ax.set_xlabel("Top-k features per paradigm")
    ax.set_ylabel("AUC (shared features only)")
    ax.set_title("Classification AUC Using Only Shared Features")
    ax.set_xscale("log")
    ax.set_ylim(0.4, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {save_path}")


def plot_3c_evolution(layer_data, save_path):
    """Plot layer-wise SAE Jaccard and hidden cosine similarity."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    layers = [d["layer"] for d in layer_data]

    ax = axes[0]
    for name, color, label in [("ic_sm", "#e74c3c", "IC-SM"),
                                ("ic_mw", "#3498db", "IC-MW"),
                                ("sm_mw", "#2ecc71", "SM-MW")]:
        vals = [d.get(f"jaccard_{name}") for d in layer_data]
        valid = [(l, v) for l, v in zip(layers, vals) if v is not None]
        if valid:
            ax.plot([x[0] for x in valid], [x[1] for x in valid], "o-",
                    label=label, color=color, markersize=3)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Jaccard Index (top-100)")
    ax.set_title("SAE Feature Overlap Across Layers")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for name, color, label in [("ic_sm", "#e74c3c", "IC-SM"),
                                ("ic_mw", "#3498db", "IC-MW"),
                                ("sm_mw", "#2ecc71", "SM-MW")]:
        vals = [d.get(f"cosine_{name}") for d in layer_data]
        valid = [(l, v) for l, v in zip(layers, vals) if v is not None]
        if valid:
            ax.plot([x[0] for x in valid], [x[1] for x in valid], "o-",
                    label=label, color=color, markersize=3)
    ax.axhline(0, ls="--", color="gray", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity (LR weight vectors)")
    ax.set_title("Hidden State Direction Alignment Across Layers")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {save_path}")


def plot_3b_transfer_matrix(projection_auc, save_path):
    """Plot direction transfer AUC as heatmap."""
    paradigms = ["ic", "sm", "mw"]
    labels = ["IC", "SM", "MW"]
    matrix = np.zeros((3, 3))
    for i, p_src in enumerate(paradigms):
        for j, p_tgt in enumerate(paradigms):
            if p_src == p_tgt:
                matrix[i, j] = np.nan
            else:
                key = f"{p_src}_direction_on_{p_tgt}"
                matrix[i, j] = projection_auc.get(key, 0.5)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
    for i in range(3):
        for j in range(3):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=12,
                        fontweight="bold" if matrix[i, j] > 0.7 else "normal")
            else:
                ax.text(j, i, "self", ha="center", va="center", fontsize=10, color="gray")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Target paradigm data")
    ax.set_ylabel("Source direction (LR weight vector)")
    ax.set_title("1D Projection Transfer AUC")
    plt.colorbar(im, ax=ax, label="AUC")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {save_path}")


def plot_combined_layer_sweep(layer_3c_data, direction_3b_data, centroid_3d_data, save_path):
    """Combined plot: SAE Jaccard, LR direction cosine, centroid cosine across layers."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: SAE Jaccard
    ax = axes[0]
    layers_3c = [d["layer"] for d in layer_3c_data]
    for name, color, label in [("ic_sm", "#e74c3c", "IC-SM"),
                                ("ic_mw", "#3498db", "IC-MW"),
                                ("sm_mw", "#2ecc71", "SM-MW")]:
        vals = [d.get(f"jaccard_{name}") for d in layer_3c_data]
        valid = [(l, v) for l, v in zip(layers_3c, vals) if v is not None]
        if valid:
            ax.plot([x[0] for x in valid], [x[1] for x in valid], "o-",
                    label=label, color=color, markersize=3)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Jaccard (top-100)")
    ax.set_title("(a) SAE Feature Overlap")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: LR direction cosine
    ax = axes[1]
    layers_3b = [d["layer"] for d in direction_3b_data]
    for name, color, label in [("ic_sm", "#e74c3c", "IC-SM"),
                                ("ic_mw", "#3498db", "IC-MW"),
                                ("sm_mw", "#2ecc71", "SM-MW")]:
        vals = [d.get(f"cosine_{name}") for d in direction_3b_data]
        valid = [(l, v) for l, v in zip(layers_3b, vals) if v is not None]
        if valid:
            ax.plot([x[0] for x in valid], [x[1] for x in valid], "o-",
                    label=label, color=color, markersize=3)
    ax.axhline(0, ls="--", color="gray", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("(b) LR Direction Alignment")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: Centroid direction cosine
    ax = axes[2]
    layers_3d = [d["layer"] for d in centroid_3d_data]
    for name, color, label in [("ic_sm", "#e74c3c", "IC-SM"),
                                ("ic_mw", "#3498db", "IC-MW"),
                                ("sm_mw", "#2ecc71", "SM-MW")]:
        vals = [d.get(f"centroid_cosine_{name}") for d in centroid_3d_data]
        valid = [(l, v) for l, v in zip(layers_3d, vals) if v is not None]
        if valid:
            ax.plot([x[0] for x in valid], [x[1] for x in valid], "o-",
                    label=label, color=color, markersize=3)
    ax.axhline(0, ls="--", color="gray", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("(c) Centroid Direction Alignment")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved: {save_path}")


# ===================================================================
# Main
# ===================================================================
def save_results(all_results, tag=""):
    """Save current results incrementally."""
    output_path = JSON_DIR / "v7_phase3_common.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    log.info(f"Results saved to {output_path} {tag}")
    sys.stdout.flush()


def main():
    t_start = time.time()
    log.info("V7 Phase 3: Common features and hidden state directions")
    log.info(f"Start time: {datetime.now().isoformat()}")
    sys.stdout.flush()

    JSON_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # ---------------------------------------------------------------
    # Analysis 3a
    # ---------------------------------------------------------------
    result_3a = analysis_3a(target_layer=22)
    if result_3a is not None:
        all_results["shared_feature_classification"] = result_3a
        plot_3a_sweep(result_3a["L22"]["top_k_sweep"], FIGURE_DIR / "v7_3a_topk_sweep.png")
        save_results(all_results, "(after 3a)")
    gc.collect()

    # ---------------------------------------------------------------
    # Analysis 3b
    # ---------------------------------------------------------------
    result_3b = analysis_3b(best_layers={"ic": 22, "sm": 12, "mw": 33})
    if result_3b is not None:
        all_results["direction_transfer"] = result_3b
        plot_3b_transfer_matrix(result_3b["1d_projection_auc"], FIGURE_DIR / "v7_3b_direction_transfer.png")
        save_results(all_results, "(after 3b)")
    gc.collect()

    # ---------------------------------------------------------------
    # Analysis 3c
    # ---------------------------------------------------------------
    result_3c = analysis_3c()
    if result_3c is not None:
        all_results["layer_wise_overlap"] = result_3c
        plot_3c_evolution(result_3c, FIGURE_DIR / "v7_3c_layer_evolution.png")
        save_results(all_results, "(after 3c)")
    gc.collect()

    # ---------------------------------------------------------------
    # Analysis 3d
    # ---------------------------------------------------------------
    result_3d = analysis_3d(best_layers={"ic": 22, "sm": 12, "mw": 33})
    if result_3d is not None:
        all_results["centroid_analysis"] = result_3d
        save_results(all_results, "(after 3d)")
    gc.collect()

    # ---------------------------------------------------------------
    # Combined figure
    # ---------------------------------------------------------------
    if result_3b is not None and result_3c is not None and result_3d is not None:
        plot_combined_layer_sweep(
            result_3c,
            result_3b.get("layer_wise", []),
            result_3d.get("centroid_layer_sweep", []),
            FIGURE_DIR / "v7_combined_layer_sweep.png",
        )

    # ---------------------------------------------------------------
    # Final save
    # ---------------------------------------------------------------
    save_results(all_results, "(FINAL)")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    elapsed = time.time() - t_start
    log.info("=" * 70)
    log.info(f"COMPLETE in {elapsed/60:.1f} minutes")
    log.info("=" * 70)

    if "shared_feature_classification" in all_results:
        r = all_results["shared_feature_classification"]["L22"]
        log.info("3a Summary (L22):")
        log.info(f"  All-3 shared features (k=100): {r['all3_shared_features']}")
        log.info(f"  All-3 shared AUC: {r['all3_shared_auc']}")
        log.info(f"  Full-feature AUC: {r['full_feature_auc']}")
        for entry in r["top_k_sweep"]:
            if entry["k"] in [100, 500, 2000]:
                log.info(f"  k={entry['k']}: J(IC,SM)={entry['jaccard_ic_sm']}, "
                         f"shared_AUC: IC={entry['auc_shared_ic']} SM={entry['auc_shared_sm']} "
                         f"MW={entry['auc_shared_mw']}")

    if "direction_transfer" in all_results:
        r = all_results["direction_transfer"]
        log.info("3b Summary:")
        log.info(f"  Weight cosine: {r['weight_cosine_similarity']}")
        log.info(f"  1D projection AUC: {r['1d_projection_auc']}")

    if "centroid_analysis" in all_results:
        r = all_results["centroid_analysis"]
        log.info("3d Summary:")
        log.info(f"  Centroid cosine: {r['centroid_cosine']}")
        log.info(f"  Centroid transfer AUC: {r['centroid_transfer_auc']}")
        log.info(f"  Centroid self AUC: {r['centroid_self_auc']}")


if __name__ == "__main__":
    main()
