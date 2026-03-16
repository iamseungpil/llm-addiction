#!/usr/bin/env python3
"""
Comprehensive LLaMA IC analysis — symmetric with Gemma analyses.

LLaMA-3.1-8B × IC V2role (1600 games, 142 BK, 32 layers, 32K SAE features, 4096 hidden dim)

Analyses:
  1. SAE BK classification (DP + R1, all 32 layers)
  2. Hidden BK classification (DP + R1, all 32 layers)
  3. SAE Risk choice classification (all rounds)
  4. Hidden Risk choice classification (all rounds)
  5. Per-condition SAE BK (bet_type, prompt, constraint)
  6. Per-condition Hidden BK (bet_type, prompt, constraint)
  7. Per-condition SAE Risk
  8. Per-condition Hidden Risk
  9. Balance-matched BK (SAE + Hidden)
  10. Cross-model transfer (Gemma IC ↔ LLaMA IC)

Usage:
    cd /home/jovyan/llm-addiction/sae_v3_analysis/src
    python run_llama_ic.py [--analyses 1,2,3,4,5,6,7,8,9,10] [--layers all|best]
"""

import argparse
import gc
import json
import logging
import sys
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from config import (DATA_ROOT, RANDOM_SEED, JSON_DIR, LOG_DIR, FIGURE_DIR,
                    MIN_ACTIVATION_RATE)
from data_loader import (load_sparse_npz, sparse_to_dense, get_metadata,
                         filter_active_features)

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_SEED)


# ===================================================================
# LLaMA IC config
# ===================================================================
LLAMA_SAE_DIR = DATA_ROOT / "sae_features_v3" / "investment_choice" / "llama"
GEMMA_SAE_DIR = DATA_ROOT / "sae_features_v3" / "investment_choice" / "gemma"

LLAMA_IC_JSON_DIR = DATA_ROOT / "investment_choice_v2_role_llama"
GEMMA_IC_JSON_DIR = DATA_ROOT / "investment_choice_v2_role"

N_LLAMA_LAYERS = 32
N_LLAMA_SAE_FEATURES = 32768
LLAMA_HIDDEN_DIM = 4096

BEST_LAYERS = [0, 4, 8, 11, 14, 16, 18, 20, 22, 24, 28, 31]

PROMPT_COMPONENTS = ["G", "M"]
COMPONENT_DATA_MAP = {"H": "R"}  # not used for IC but keep for consistency


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
# Data loading (LLaMA-specific)
# ===================================================================

def load_llama_sae(layer, mode="decision_point"):
    """Load LLaMA SAE features for IC."""
    npz_path = LLAMA_SAE_DIR / f"sae_features_L{layer}.npz"
    if not npz_path.exists():
        return None
    raw = load_sparse_npz(npz_path)
    meta = get_metadata(raw)
    features = sparse_to_dense(raw)

    if mode == "decision_point":
        mask = meta["is_last_round"]
        return features[mask], {k: v[mask] for k, v in meta.items()}
    elif mode == "all_rounds":
        return features, meta
    return features, meta


def load_llama_hidden(layer, mode="decision_point"):
    """Load LLaMA hidden states for IC."""
    ckpt_path = LLAMA_SAE_DIR / "checkpoints" / "phase_a_hidden_states.npz"
    if not ckpt_path.exists():
        return None

    ckpt = np.load(ckpt_path, allow_pickle=False)
    hidden_all = ckpt["hidden_states"]  # (n_rounds, n_layers, hidden_dim)

    # Get metadata from SAE feature file
    any_sae = list(LLAMA_SAE_DIR.glob("sae_features_L*.npz"))
    if not any_sae:
        return None
    raw = load_sparse_npz(any_sae[0])
    meta = get_metadata(raw)

    features = hidden_all[:, layer, :]

    if "valid_mask" in ckpt:
        valid = ckpt["valid_mask"].astype(bool)
        features = features[valid]
        meta = {k: v[valid] for k, v in meta.items()}

    if mode == "decision_point":
        mask = meta["is_last_round"]
        return features[mask], {k: v[mask] for k, v in meta.items()}
    elif mode == "all_rounds":
        return features, meta
    return features, meta


def load_gemma_hidden(layer, mode="decision_point"):
    """Load Gemma hidden states for IC (for cross-model transfer)."""
    ckpt_path = GEMMA_SAE_DIR / "checkpoint" / "phase_a_hidden_states.npz"
    if not ckpt_path.exists():
        return None

    ckpt = np.load(ckpt_path, allow_pickle=False)
    hidden_all = ckpt["hidden_states"]

    any_sae = list(GEMMA_SAE_DIR.glob("sae_features_L*.npz"))
    if not any_sae:
        return None
    raw = load_sparse_npz(any_sae[0])
    meta = get_metadata(raw)

    features = hidden_all[:, layer, :]

    if "valid_mask" in ckpt:
        valid = ckpt["valid_mask"].astype(bool)
        features = features[valid]
        meta = {k: v[valid] for k, v in meta.items()}

    if mode == "decision_point":
        mask = meta["is_last_round"]
        return features[mask], {k: v[mask] for k, v in meta.items()}
    elif mode == "all_rounds":
        return features, meta
    return features, meta


def get_labels(meta):
    """Binary labels: 1=bankruptcy, 0=voluntary_stop."""
    return (meta["game_outcomes"] == "bankruptcy").astype(np.int32)


def load_llama_ic_choices():
    """Load LLaMA IC per-round choices."""
    choices = {}
    files = sorted(LLAMA_IC_JSON_DIR.glob("llama_investment_c[0-9][0-9]_*.json"))
    offset = 0
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        for i, game in enumerate(data["results"]):
            npz_gid = offset + i + 1
            choices[npz_gid] = {}
            for dec in game.get("decisions", []):
                choices[npz_gid][dec["round"]] = dec["choice"]
        offset += len(data["results"])
    return choices


def get_risk_labels(game_ids, round_nums):
    """IC risk: choice 3,4 = risky (1), choice 1,2 = safe (0)."""
    choices = load_llama_ic_choices()
    labels = np.full(len(game_ids), -1, dtype=np.int32)
    for i, (gid, rnd) in enumerate(zip(game_ids, round_nums)):
        ch = choices.get(int(gid), {}).get(int(rnd))
        if ch is not None:
            labels[i] = 1 if ch >= 3 else 0
    valid = labels >= 0
    return labels, valid


# ===================================================================
# Shared classification
# ===================================================================

def classify_cv(features, labels, n_folds=5):
    """5-fold stratified CV with balanced class weights."""
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    effective_folds = min(n_folds, n_pos, n_neg)

    if effective_folds < 2:
        return {"auc": 0.5, "f1": 0.0, "n_pos": n_pos, "n_neg": n_neg, "skipped": True}

    skf = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=RANDOM_SEED)
    aucs, f1s = [], []

    for train_idx, test_idx in skf.split(features, labels):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(features[train_idx])
        X_test = scaler.transform(features[test_idx])

        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000,
                                 class_weight="balanced", random_state=RANDOM_SEED)
        clf.fit(X_train, labels[train_idx])
        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred = clf.predict(X_test)

        aucs.append(roc_auc_score(labels[test_idx], y_prob))
        f1s.append(f1_score(labels[test_idx], y_pred, zero_division=0))

    return {
        "auc": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
        "f1": float(np.mean(f1s)),
        "n_pos": n_pos, "n_neg": n_neg, "skipped": False,
    }


def classify_transfer(train_feat, train_labels, test_feat, test_labels):
    """Train on one set, test on another."""
    n_train_pos = int(train_labels.sum())
    n_test_pos = int(test_labels.sum())

    if n_train_pos < 2 or n_test_pos < 2:
        return {"auc": 0.5, "n_train": len(train_labels), "n_test": len(test_labels), "skipped": True}

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_feat)
    X_test = scaler.transform(test_feat)

    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000,
                             class_weight="balanced", random_state=RANDOM_SEED)
    clf.fit(X_train, train_labels)
    y_prob = clf.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(test_labels, y_prob)
    return {
        "auc": float(auc),
        "n_train": len(train_labels), "n_test": len(test_labels),
        "n_train_pos": n_train_pos, "n_test_pos": n_test_pos,
        "skipped": False,
    }


# ===================================================================
# Analysis 1: SAE BK Classification (DP + R1)
# ===================================================================

def run_sae_bk(logger, layers):
    logger.info("=" * 60)
    logger.info("Analysis 1: LLaMA SAE BK Classification (DP + R1)")
    logger.info("=" * 60)

    results = {"dp": [], "r1": []}

    for layer in layers:
        # Decision-point
        loaded = load_llama_sae(layer, mode="decision_point")
        if loaded is None:
            continue
        features, meta = loaded
        active_feat, active_idx = filter_active_features(features, MIN_ACTIVATION_RATE)
        labels = get_labels(meta)
        res = classify_cv(active_feat, labels)
        res["layer"] = layer
        res["n_active_features"] = len(active_idx)
        results["dp"].append(res)
        logger.info(f"  L{layer} DP: AUC={res['auc']:.3f} (BK={res['n_pos']}, feat={len(active_idx)})")

        # R1
        r1_loaded = load_llama_sae(layer, mode="all_rounds")
        if r1_loaded is not None:
            r1_feat, r1_meta = r1_loaded
            r1_mask = r1_meta["round_nums"] == 1
            if r1_mask.sum() > 0:
                r1_active = r1_feat[r1_mask][:, active_idx]
                r1_labels = get_labels({k: v[r1_mask] for k, v in r1_meta.items()})
                r1_res = classify_cv(r1_active, r1_labels)
                r1_res["layer"] = layer
                results["r1"].append(r1_res)
                logger.info(f"  L{layer} R1: AUC={r1_res['auc']:.3f}")
            del r1_feat
            gc.collect()

        del features, active_feat
        gc.collect()

    if results["dp"]:
        best = max(results["dp"], key=lambda r: r["auc"])
        logger.info(f"  Best DP: L{best['layer']} AUC={best['auc']:.3f}")
    if results["r1"]:
        best = max(results["r1"], key=lambda r: r["auc"])
        logger.info(f"  Best R1: L{best['layer']} AUC={best['auc']:.3f}")

    return results


# ===================================================================
# Analysis 2: Hidden BK Classification (DP + R1)
# ===================================================================

def run_hidden_bk(logger, layers):
    logger.info("=" * 60)
    logger.info("Analysis 2: LLaMA Hidden BK Classification (DP + R1)")
    logger.info("=" * 60)

    results = {"dp": [], "r1": []}

    for layer in layers:
        dp_loaded = load_llama_hidden(layer, mode="decision_point")
        if dp_loaded is None:
            continue
        features, meta = dp_loaded
        labels = get_labels(meta)
        res = classify_cv(features, labels)
        res["layer"] = layer
        results["dp"].append(res)
        logger.info(f"  L{layer} DP: AUC={res['auc']:.3f} (BK={res['n_pos']})")

        r1_loaded = load_llama_hidden(layer, mode="all_rounds")
        if r1_loaded is not None:
            r1_feat, r1_meta = r1_loaded
            r1_mask = r1_meta["round_nums"] == 1
            if r1_mask.sum() > 0:
                r1_labels = get_labels({k: v[r1_mask] for k, v in r1_meta.items()})
                r1_res = classify_cv(r1_feat[r1_mask], r1_labels)
                r1_res["layer"] = layer
                results["r1"].append(r1_res)
                logger.info(f"  L{layer} R1: AUC={r1_res['auc']:.3f}")

        gc.collect()

    if results["dp"]:
        best = max(results["dp"], key=lambda r: r["auc"])
        logger.info(f"  Best DP: L{best['layer']} AUC={best['auc']:.3f}")
    if results["r1"]:
        best = max(results["r1"], key=lambda r: r["auc"])
        logger.info(f"  Best R1: L{best['layer']} AUC={best['auc']:.3f}")

    return results


# ===================================================================
# Analysis 3: SAE Risk Classification
# ===================================================================

def run_sae_risk(logger, layers):
    logger.info("=" * 60)
    logger.info("Analysis 3: LLaMA SAE Risk Choice Classification")
    logger.info("=" * 60)

    results = []
    risk_labels_cache = None

    for layer in layers:
        loaded = load_llama_sae(layer, mode="all_rounds")
        if loaded is None:
            continue
        features, meta = loaded
        active_feat, active_idx = filter_active_features(features, MIN_ACTIVATION_RATE)

        if risk_labels_cache is None:
            risk_labels_cache, valid = get_risk_labels(meta["game_ids"], meta["round_nums"])
        else:
            valid = risk_labels_cache >= 0

        if valid.sum() < 20:
            continue

        res = classify_cv(active_feat[valid], risk_labels_cache[valid])
        res["layer"] = layer
        res["n_active_features"] = len(active_idx)
        results.append(res)
        logger.info(f"  L{layer}: AUC={res['auc']:.3f} (risky={res['n_pos']}, safe={res['n_neg']})")

        del features, active_feat
        gc.collect()

    if results:
        best = max(results, key=lambda r: r["auc"])
        logger.info(f"  Best: L{best['layer']} AUC={best['auc']:.3f}")

    return results


# ===================================================================
# Analysis 4: Hidden Risk Classification
# ===================================================================

def run_hidden_risk(logger, layers):
    logger.info("=" * 60)
    logger.info("Analysis 4: LLaMA Hidden Risk Choice Classification")
    logger.info("=" * 60)

    results = []
    risk_labels_cache = None

    for layer in layers:
        loaded = load_llama_hidden(layer, mode="all_rounds")
        if loaded is None:
            continue
        features, meta = loaded

        if risk_labels_cache is None:
            risk_labels_cache, valid = get_risk_labels(meta["game_ids"], meta["round_nums"])
        else:
            valid = risk_labels_cache >= 0

        if valid.sum() < 20:
            continue

        res = classify_cv(features[valid], risk_labels_cache[valid])
        res["layer"] = layer
        results.append(res)
        logger.info(f"  L{layer}: AUC={res['auc']:.3f} (risky={res['n_pos']}, safe={res['n_neg']})")

        gc.collect()

    if results:
        best = max(results, key=lambda r: r["auc"])
        logger.info(f"  Best: L{best['layer']} AUC={best['auc']:.3f}")

    return results


# ===================================================================
# Analysis 5: Per-condition SAE BK
# ===================================================================

def run_percondition_sae_bk(logger, layers):
    logger.info("=" * 60)
    logger.info("Analysis 5: LLaMA Per-Condition SAE BK (DP + R1)")
    logger.info("=" * 60)

    results = {}

    for layer in layers:
        loaded = load_llama_sae(layer, mode="decision_point")
        if loaded is None:
            continue
        features, meta = loaded
        active_feat, active_idx = filter_active_features(features, MIN_ACTIVATION_RATE)
        labels = get_labels(meta)

        # R1
        r1_loaded = load_llama_sae(layer, mode="all_rounds")
        r1_features, r1_meta, r1_labels = None, None, None
        if r1_loaded is not None:
            r1_feat_full, r1_meta = r1_loaded
            r1_mask = r1_meta["round_nums"] == 1
            r1_features = r1_feat_full[r1_mask][:, active_idx]
            r1_meta = {k: v[r1_mask] for k, v in r1_meta.items()}
            r1_labels = get_labels(r1_meta)
            del r1_feat_full
            gc.collect()

        # bet_type
        for bt in ["fixed", "variable"]:
            bt_mask = meta["bet_types"] == bt
            n_bk = int((labels[bt_mask] == 1).sum()) if bt_mask.sum() > 0 else 0
            if bt_mask.sum() >= 10 and n_bk >= 2:
                res = classify_cv(active_feat[bt_mask], labels[bt_mask])
                res["layer"] = layer
                res["condition"] = f"bet_type={bt}"
                results[f"L{layer}_bt_{bt}"] = res
                logger.info(f"  L{layer} {bt}: AUC={res['auc']:.3f} (BK={n_bk})")

                if r1_meta is not None:
                    r1_bt = r1_meta["bet_types"] == bt
                    r1_nbk = int((r1_labels[r1_bt] == 1).sum()) if r1_bt.sum() > 0 else 0
                    if r1_bt.sum() >= 10 and r1_nbk >= 2:
                        r1_res = classify_cv(r1_features[r1_bt], r1_labels[r1_bt])
                        r1_res["layer"] = layer
                        results[f"L{layer}_bt_{bt}_R1"] = r1_res

        # prompt components
        if "prompt_conditions" in meta:
            for comp in PROMPT_COMPONENTS:
                data_comp = COMPONENT_DATA_MAP.get(comp, comp)
                has_comp = np.array([data_comp in str(pc) for pc in meta["prompt_conditions"]])
                for name, mask in [("with_" + comp, has_comp), ("without_" + comp, ~has_comp)]:
                    n_bk = int((labels[mask] == 1).sum()) if mask.sum() > 0 else 0
                    if mask.sum() >= 10 and n_bk >= 2:
                        res = classify_cv(active_feat[mask], labels[mask])
                        res["layer"] = layer
                        res["condition"] = name
                        results[f"L{layer}_{name}"] = res
                        logger.info(f"  L{layer} {name}: AUC={res['auc']:.3f} (BK={n_bk})")

        # bet_constraint
        if "bet_constraints" in meta:
            for bc in ["30", "50", "70"]:
                bc_mask = meta["bet_constraints"] == bc
                n_bk = int((labels[bc_mask] == 1).sum()) if bc_mask.sum() > 0 else 0
                if bc_mask.sum() >= 10 and n_bk >= 2:
                    res = classify_cv(active_feat[bc_mask], labels[bc_mask])
                    res["layer"] = layer
                    results[f"L{layer}_bc_{bc}"] = res
                    logger.info(f"  L{layer} bc_{bc}: AUC={res['auc']:.3f} (BK={n_bk})")

        del features, active_feat
        gc.collect()

    return results


# ===================================================================
# Analysis 6: Per-condition Hidden BK
# ===================================================================

def run_percondition_hidden_bk(logger, layers):
    logger.info("=" * 60)
    logger.info("Analysis 6: LLaMA Per-Condition Hidden BK (DP + R1)")
    logger.info("=" * 60)

    results = {}

    for layer in layers:
        dp_loaded = load_llama_hidden(layer, mode="decision_point")
        if dp_loaded is None:
            continue
        features, meta = dp_loaded
        labels = get_labels(meta)

        # R1
        r1_loaded = load_llama_hidden(layer, mode="all_rounds")
        r1_feat, r1_meta, r1_labels = None, None, None
        if r1_loaded is not None:
            r1_f, r1_m = r1_loaded
            r1_mask = r1_m["round_nums"] == 1
            r1_feat = r1_f[r1_mask]
            r1_meta = {k: v[r1_mask] for k, v in r1_m.items()}
            r1_labels = get_labels(r1_meta)

        # bet_type
        for bt in ["fixed", "variable"]:
            bt_mask = meta["bet_types"] == bt
            n_bk = int((labels[bt_mask] == 1).sum()) if bt_mask.sum() > 0 else 0
            if bt_mask.sum() >= 10 and n_bk >= 2:
                res = classify_cv(features[bt_mask], labels[bt_mask])
                res["layer"] = layer
                res["condition"] = f"bet_type={bt}"
                results[f"L{layer}_bt_{bt}"] = res
                logger.info(f"  L{layer} hidden {bt}: AUC={res['auc']:.3f} (BK={n_bk})")

                if r1_meta is not None:
                    r1_bt = r1_meta["bet_types"] == bt
                    r1_nbk = int((r1_labels[r1_bt] == 1).sum()) if r1_bt.sum() > 0 else 0
                    if r1_bt.sum() >= 10 and r1_nbk >= 2:
                        r1_res = classify_cv(r1_feat[r1_bt], r1_labels[r1_bt])
                        r1_res["layer"] = layer
                        results[f"L{layer}_bt_{bt}_R1"] = r1_res

        # prompt components
        if "prompt_conditions" in meta:
            for comp in PROMPT_COMPONENTS:
                data_comp = COMPONENT_DATA_MAP.get(comp, comp)
                has_comp = np.array([data_comp in str(pc) for pc in meta["prompt_conditions"]])
                for name, mask in [("with_" + comp, has_comp), ("without_" + comp, ~has_comp)]:
                    n_bk = int((labels[mask] == 1).sum()) if mask.sum() > 0 else 0
                    if mask.sum() >= 10 and n_bk >= 2:
                        res = classify_cv(features[mask], labels[mask])
                        res["layer"] = layer
                        res["condition"] = name
                        results[f"L{layer}_hidden_{name}"] = res
                        logger.info(f"  L{layer} hidden {name}: AUC={res['auc']:.3f} (BK={n_bk})")

        # bet_constraint
        if "bet_constraints" in meta:
            for bc in ["30", "50", "70"]:
                bc_mask = meta["bet_constraints"] == bc
                n_bk = int((labels[bc_mask] == 1).sum()) if bc_mask.sum() > 0 else 0
                if bc_mask.sum() >= 10 and n_bk >= 2:
                    res = classify_cv(features[bc_mask], labels[bc_mask])
                    res["layer"] = layer
                    results[f"L{layer}_hidden_bc_{bc}"] = res
                    logger.info(f"  L{layer} hidden bc_{bc}: AUC={res['auc']:.3f} (BK={n_bk})")

        gc.collect()

    return results


# ===================================================================
# Analysis 7: Per-condition SAE Risk
# ===================================================================

def run_percondition_sae_risk(logger, layers):
    logger.info("=" * 60)
    logger.info("Analysis 7: LLaMA Per-Condition SAE Risk")
    logger.info("=" * 60)

    results = {}
    risk_labels_cache = None

    for layer in layers:
        loaded = load_llama_sae(layer, mode="all_rounds")
        if loaded is None:
            continue
        features, meta = loaded
        active_feat, active_idx = filter_active_features(features, MIN_ACTIVATION_RATE)

        if risk_labels_cache is None:
            risk_labels_cache, valid = get_risk_labels(meta["game_ids"], meta["round_nums"])
        else:
            valid = risk_labels_cache >= 0

        if valid.sum() < 20:
            continue

        risk_labels = risk_labels_cache[valid]
        feat_valid = active_feat[valid]
        meta_valid = {k: v[valid] for k, v in meta.items()}

        # bet_type
        for bt in ["fixed", "variable"]:
            bt_mask = meta_valid["bet_types"] == bt
            if bt_mask.sum() >= 20:
                n_pos = int(risk_labels[bt_mask].sum())
                n_neg = int((~risk_labels[bt_mask].astype(bool)).sum())
                if n_pos >= 5 and n_neg >= 5:
                    res = classify_cv(feat_valid[bt_mask], risk_labels[bt_mask])
                    res["layer"] = layer
                    results[f"L{layer}_bt_{bt}"] = res
                    logger.info(f"  L{layer} risk {bt}: AUC={res['auc']:.3f}")

        # prompt components
        if "prompt_conditions" in meta_valid:
            for comp in PROMPT_COMPONENTS:
                data_comp = COMPONENT_DATA_MAP.get(comp, comp)
                has_comp = np.array([data_comp in str(pc) for pc in meta_valid["prompt_conditions"]])
                for name, mask in [("with_" + comp, has_comp), ("without_" + comp, ~has_comp)]:
                    if mask.sum() >= 20:
                        n_pos = int(risk_labels[mask].sum())
                        n_neg = int((~risk_labels[mask].astype(bool)).sum())
                        if n_pos >= 5 and n_neg >= 5:
                            res = classify_cv(feat_valid[mask], risk_labels[mask])
                            res["layer"] = layer
                            results[f"L{layer}_{name}"] = res
                            logger.info(f"  L{layer} risk {name}: AUC={res['auc']:.3f}")

        # bet_constraint
        if "bet_constraints" in meta_valid:
            for bc in ["30", "50", "70"]:
                bc_mask = meta_valid["bet_constraints"] == bc
                if bc_mask.sum() >= 20:
                    n_pos = int(risk_labels[bc_mask].sum())
                    n_neg = int((~risk_labels[bc_mask].astype(bool)).sum())
                    if n_pos >= 5 and n_neg >= 5:
                        res = classify_cv(feat_valid[bc_mask], risk_labels[bc_mask])
                        res["layer"] = layer
                        results[f"L{layer}_bc_{bc}"] = res
                        logger.info(f"  L{layer} risk bc_{bc}: AUC={res['auc']:.3f}")

        del features, active_feat
        gc.collect()

    return results


# ===================================================================
# Analysis 8: Per-condition Hidden Risk
# ===================================================================

def run_percondition_hidden_risk(logger, layers):
    logger.info("=" * 60)
    logger.info("Analysis 8: LLaMA Per-Condition Hidden Risk")
    logger.info("=" * 60)

    results = {}
    risk_labels_cache = None

    for layer in layers:
        loaded = load_llama_hidden(layer, mode="all_rounds")
        if loaded is None:
            continue
        features, meta = loaded

        if risk_labels_cache is None:
            risk_labels_cache, valid = get_risk_labels(meta["game_ids"], meta["round_nums"])
        else:
            valid = risk_labels_cache >= 0

        if valid.sum() < 20:
            continue

        risk_labels = risk_labels_cache[valid]
        feat_valid = features[valid]
        meta_valid = {k: v[valid] for k, v in meta.items()}

        # bet_type
        for bt in ["fixed", "variable"]:
            bt_mask = meta_valid["bet_types"] == bt
            if bt_mask.sum() >= 20:
                n_pos = int(risk_labels[bt_mask].sum())
                n_neg = int((~risk_labels[bt_mask].astype(bool)).sum())
                if n_pos >= 5 and n_neg >= 5:
                    res = classify_cv(feat_valid[bt_mask], risk_labels[bt_mask])
                    res["layer"] = layer
                    results[f"L{layer}_bt_{bt}"] = res
                    logger.info(f"  L{layer} risk {bt}: AUC={res['auc']:.3f}")

        # prompt components
        if "prompt_conditions" in meta_valid:
            for comp in PROMPT_COMPONENTS:
                data_comp = COMPONENT_DATA_MAP.get(comp, comp)
                has_comp = np.array([data_comp in str(pc) for pc in meta_valid["prompt_conditions"]])
                for name, mask in [("with_" + comp, has_comp), ("without_" + comp, ~has_comp)]:
                    if mask.sum() >= 20:
                        n_pos = int(risk_labels[mask].sum())
                        n_neg = int((~risk_labels[mask].astype(bool)).sum())
                        if n_pos >= 5 and n_neg >= 5:
                            res = classify_cv(feat_valid[mask], risk_labels[mask])
                            res["layer"] = layer
                            results[f"L{layer}_{name}"] = res
                            logger.info(f"  L{layer} risk {name}: AUC={res['auc']:.3f}")

        # bet_constraint
        if "bet_constraints" in meta_valid:
            for bc in ["30", "50", "70"]:
                bc_mask = meta_valid["bet_constraints"] == bc
                if bc_mask.sum() >= 20:
                    n_pos = int(risk_labels[bc_mask].sum())
                    n_neg = int((~risk_labels[bc_mask].astype(bool)).sum())
                    if n_pos >= 5 and n_neg >= 5:
                        res = classify_cv(feat_valid[bc_mask], risk_labels[bc_mask])
                        res["layer"] = layer
                        results[f"L{layer}_bc_{bc}"] = res
                        logger.info(f"  L{layer} risk bc_{bc}: AUC={res['auc']:.3f}")

        gc.collect()

    return results


# ===================================================================
# Analysis 9: Balance-Matched BK (SAE + Hidden)
# ===================================================================

def run_balance_matched(logger, layers):
    logger.info("=" * 60)
    logger.info("Analysis 9: LLaMA Balance-Matched BK Classification")
    logger.info("=" * 60)

    results = {"sae_dp": [], "sae_r1": [], "hidden_dp": [], "hidden_r1": []}

    for layer in layers:
        # Decision-point
        for rep_type, loader in [("sae", load_llama_sae), ("hidden", load_llama_hidden)]:
            dp_loaded = loader(layer, mode="decision_point")
            if dp_loaded is None:
                continue
            features, meta = dp_loaded
            labels = get_labels(meta)

            if rep_type == "sae":
                features, active_idx = filter_active_features(features, MIN_ACTIVATION_RATE)

            # Balance-match: for each BK game, find closest-balance safe game
            if "balances" not in meta:
                continue

            bk_idx = np.where(labels == 1)[0]
            safe_idx = np.where(labels == 0)[0]

            if len(bk_idx) < 5 or len(safe_idx) < 5:
                continue

            bk_balances = meta["balances"][bk_idx].astype(float)
            safe_balances = meta["balances"][safe_idx].astype(float)

            matched_safe = []
            used = set()
            for bb in bk_balances:
                dists = np.abs(safe_balances - bb)
                for j in np.argsort(dists):
                    if j not in used:
                        matched_safe.append(safe_idx[j])
                        used.add(j)
                        break

            if len(matched_safe) < 5:
                continue

            matched_idx = np.concatenate([bk_idx, np.array(matched_safe)])
            matched_labels = labels[matched_idx]
            matched_feat = features[matched_idx]

            res = classify_cv(matched_feat, matched_labels)
            res["layer"] = layer
            res["n_matched"] = len(matched_idx)
            results[f"{rep_type}_dp"].append(res)
            logger.info(f"  L{layer} {rep_type} DP balanced: AUC={res['auc']:.3f} (n={len(matched_idx)})")

        # R1
        for rep_type, loader in [("sae", load_llama_sae), ("hidden", load_llama_hidden)]:
            r1_loaded = loader(layer, mode="all_rounds")
            if r1_loaded is None:
                continue
            r1_feat, r1_meta = r1_loaded
            r1_mask = r1_meta["round_nums"] == 1
            r1_feat = r1_feat[r1_mask]
            r1_meta_f = {k: v[r1_mask] for k, v in r1_meta.items()}
            r1_labels = get_labels(r1_meta_f)

            if rep_type == "sae":
                r1_feat, _ = filter_active_features(r1_feat, MIN_ACTIVATION_RATE)

            if "balances" not in r1_meta_f:
                continue

            bk_idx = np.where(r1_labels == 1)[0]
            safe_idx = np.where(r1_labels == 0)[0]

            if len(bk_idx) < 5 or len(safe_idx) < 5:
                continue

            bk_bal = r1_meta_f["balances"][bk_idx].astype(float)
            safe_bal = r1_meta_f["balances"][safe_idx].astype(float)

            matched_safe = []
            used = set()
            for bb in bk_bal:
                dists = np.abs(safe_bal - bb)
                for j in np.argsort(dists):
                    if j not in used:
                        matched_safe.append(safe_idx[j])
                        used.add(j)
                        break

            if len(matched_safe) < 5:
                continue

            matched_idx = np.concatenate([bk_idx, np.array(matched_safe)])
            res = classify_cv(r1_feat[matched_idx], r1_labels[matched_idx])
            res["layer"] = layer
            results[f"{rep_type}_r1"].append(res)
            logger.info(f"  L{layer} {rep_type} R1 balanced: AUC={res['auc']:.3f}")

        gc.collect()

    for key in results:
        if results[key]:
            best = max(results[key], key=lambda r: r["auc"])
            logger.info(f"  Best {key}: L{best['layer']} AUC={best['auc']:.3f}")

    return results


# ===================================================================
# Analysis 10: Cross-Model Transfer (Gemma IC ↔ LLaMA IC)
# ===================================================================

def run_cross_model_transfer(logger, layers):
    """Train on one model's hidden states, test on another.

    Since Gemma (3584) and LLaMA (4096) have different hidden dims,
    we use PCA to project both to a common dimensionality.
    """
    logger.info("=" * 60)
    logger.info("Analysis 10: Cross-Model Transfer (Gemma IC ↔ LLaMA IC)")
    logger.info("=" * 60)

    from sklearn.decomposition import PCA

    # Map LLaMA layers to approximate Gemma layers (proportional mapping)
    # LLaMA has 32 layers, Gemma has 42 layers
    # Mapping: llama_layer / 31 ≈ gemma_layer / 41
    def llama_to_gemma_layer(ll):
        return min(41, round(ll * 41 / 31))

    results = {"gemma_to_llama": [], "llama_to_gemma": []}
    common_dim = 256  # PCA target dimension

    for llama_layer in layers:
        gemma_layer = llama_to_gemma_layer(llama_layer)

        llama_loaded = load_llama_hidden(llama_layer, mode="decision_point")
        gemma_loaded = load_gemma_hidden(gemma_layer, mode="decision_point")

        if llama_loaded is None or gemma_loaded is None:
            continue

        llama_feat, llama_meta = llama_loaded
        gemma_feat, gemma_meta = gemma_loaded
        llama_labels = get_labels(llama_meta)
        gemma_labels = get_labels(gemma_meta)

        # PCA to common dimension
        pca_llama = PCA(n_components=common_dim, random_state=RANDOM_SEED)
        pca_gemma = PCA(n_components=common_dim, random_state=RANDOM_SEED)

        llama_pca = pca_llama.fit_transform(llama_feat)
        gemma_pca = pca_gemma.fit_transform(gemma_feat)

        # Gemma → LLaMA
        res_g2l = classify_transfer(gemma_pca, gemma_labels, llama_pca, llama_labels)
        res_g2l["llama_layer"] = llama_layer
        res_g2l["gemma_layer"] = gemma_layer
        results["gemma_to_llama"].append(res_g2l)
        logger.info(f"  L{gemma_layer}(G)→L{llama_layer}(L): AUC={res_g2l['auc']:.3f}")

        # LLaMA → Gemma
        res_l2g = classify_transfer(llama_pca, llama_labels, gemma_pca, gemma_labels)
        res_l2g["llama_layer"] = llama_layer
        res_l2g["gemma_layer"] = gemma_layer
        results["llama_to_gemma"].append(res_l2g)
        logger.info(f"  L{llama_layer}(L)→L{gemma_layer}(G): AUC={res_l2g['auc']:.3f}")

        gc.collect()

    for direction in results:
        if results[direction]:
            best = max(results[direction], key=lambda r: r["auc"])
            logger.info(f"  Best {direction}: AUC={best['auc']:.3f}")

    return results


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyses", type=str, default="1,2,3,4,5,6,7,8,9,10",
                        help="Comma-separated analysis numbers (1-10)")
    parser.add_argument("--layers", type=str, default="all",
                        help="'best' for 12 layers, 'all' for all 32")
    args = parser.parse_args()

    analyses_to_run = set(int(x) for x in args.analyses.split(","))

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    JSON_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"llama_ic_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"LLaMA IC Comprehensive Analysis — analyses: {sorted(analyses_to_run)}")

    if args.layers == "best":
        analysis_layers = BEST_LAYERS
    else:
        analysis_layers = list(range(N_LLAMA_LAYERS))

    logger.info(f"Layers: {len(analysis_layers)} — {analysis_layers}")

    all_results = {}

    analysis_map = {
        1: ("sae_bk", run_sae_bk),
        2: ("hidden_bk", run_hidden_bk),
        3: ("sae_risk", run_sae_risk),
        4: ("hidden_risk", run_hidden_risk),
        5: ("percondition_sae_bk", run_percondition_sae_bk),
        6: ("percondition_hidden_bk", run_percondition_hidden_bk),
        7: ("percondition_sae_risk", run_percondition_sae_risk),
        8: ("percondition_hidden_risk", run_percondition_hidden_risk),
        9: ("balance_matched", run_balance_matched),
        10: ("cross_model_transfer", run_cross_model_transfer),
    }

    for num in sorted(analyses_to_run):
        if num in analysis_map:
            key, func = analysis_map[num]
            all_results[key] = func(logger, analysis_layers)

    # Save results
    json_file = JSON_DIR / f"llama_ic_{timestamp}.json"
    with open(json_file, "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    logger.info(f"\nResults saved: {json_file}")

    logger.info(f"\n{'=' * 60}")
    logger.info("LLAMA IC COMPREHENSIVE ANALYSIS COMPLETE")
    logger.info(f"{'=' * 60}")
    logger.info(f"Results: {json_file}")
    logger.info(f"Log: {log_file}")


if __name__ == "__main__":
    main()
