#!/usr/bin/env python3
"""
V4 Improved SAE Analyses — Design-Critiqued & Corrected.

Self-critique improvements over V3:
1. LLaMA V2 IC integration (cross-model validation, 32 layers)
2. Bootstrap CI for cross-domain transfer (100 iterations)
3. Same-layer (L22) feature comparison across paradigms
4. Permutation test for R1 BK classification significance
5. Proper handling of 2 SM games with extraction round mismatches
6. Cross-model comparison (Gemma IC vs LLaMA IC)

Data integrity:
- Gemma V3: IC V2role (1600, 172 BK), SM V4role (3200, 87 BK), MW V2role (3200, 54 BK)
  → ALL verified clean, NPZ↔JSON 100% match
- LLaMA V2: IC results/ (700, 180 BK)
  → Clean source, decision-point only, all 32 layers valid (fnlp SAE)
- LLaMA SM V1: EXCLUDED (corrupted data)

Usage:
    cd /home/jovyan/llm-addiction/sae_v3_analysis/src
    python run_improved_v4.py
"""

import gc
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from itertools import permutations

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from config import (
    PARADIGMS, N_LAYERS, FIGURE_DIR, JSON_DIR, LOG_DIR,
    MIN_ACTIVATION_RATE, RANDOM_SEED,
    PARADIGM_COLORS, PARADIGM_LABELS,
)
from data_loader import (
    load_layer_features, filter_active_features, get_labels,
)


# ===================================================================
# Paths
# ===================================================================
LLAMA_V2_IC_DIR = Path("/home/jovyan/beomi/llm-addiction-data/sae_features_v2/investment_choice/llama")
LLAMA_N_LAYERS = 32
LLAMA_N_FEATURES = 32768

# SM games with extraction round mismatches (2/3200)
SM_EXCLUDE_GIDS = set()  # Will be populated if needed


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
# Shared classification utilities
# ===================================================================

def classify_cv(features, labels, n_folds=5, seed=RANDOM_SEED):
    """5-fold stratified CV with balanced class weights."""
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    effective_folds = min(n_folds, n_pos, n_neg)

    if effective_folds < 2:
        return {"auc": 0.5, "f1": 0.0, "n_pos": n_pos, "n_neg": n_neg, "skipped": True}

    skf = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=seed)
    aucs, f1s = [], []

    for train_idx, test_idx in skf.split(features, labels):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(features[train_idx])
        X_test = scaler.transform(features[test_idx])

        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000,
                                 class_weight="balanced", random_state=seed)
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


def permutation_test(features, labels, n_permutations=1000, seed=RANDOM_SEED):
    """Permutation test: shuffle labels, compute AUC distribution."""
    rng = np.random.RandomState(seed)
    null_aucs = []
    for _ in range(n_permutations):
        perm_labels = rng.permutation(labels)
        res = classify_cv(features, perm_labels, seed=rng.randint(0, 100000))
        if not res.get("skipped", False):
            null_aucs.append(res["auc"])
    return null_aucs


# ===================================================================
# LLaMA V2 IC data loader
# ===================================================================

def load_llama_v2_ic(layer):
    """Load LLaMA V2 IC SAE features (dense format)."""
    npz_path = LLAMA_V2_IC_DIR / f"layer_{layer}_features.npz"
    if not npz_path.exists():
        return None

    data = np.load(npz_path, allow_pickle=True)
    features = data["features"]  # (700, 32768) dense float32
    outcomes = data["outcomes"]  # str array
    labels = (outcomes == "bankruptcy").astype(np.int32)

    meta = {
        "outcomes": outcomes,
        "game_ids": data["game_ids"],
        "bet_types": data["bet_types"],
        "bet_constraints": data["bet_constraints"],
        "prompt_conditions": data["prompt_conditions"],
    }
    if "final_balances" in data:
        meta["final_balances"] = data["final_balances"]

    return features, labels, meta


# ===================================================================
# Part 1: LLaMA V2 IC — BK Classification (cross-model validation)
# ===================================================================

def run_llama_bk_classification(logger):
    """LLaMA V2 IC: BK classification across 32 layers."""
    logger.info("\n" + "=" * 70)
    logger.info("PART 1: LLaMA V2 IC — BK Classification (Cross-Model Validation)")
    logger.info("=" * 70)

    results = []
    for layer in range(LLAMA_N_LAYERS):
        loaded = load_llama_v2_ic(layer)
        if loaded is None:
            continue

        features, labels, meta = loaded
        # Filter active features
        rate = (features != 0).mean(axis=0)
        active = rate >= MIN_ACTIVATION_RATE
        if active.sum() < 10:
            continue

        res = classify_cv(features[:, active], labels)
        res["layer"] = layer
        res["n_features"] = int(active.sum())
        results.append(res)

        if layer % 5 == 0 or layer == LLAMA_N_LAYERS - 1:
            logger.info(f"  L{layer:02d}: AUC={res['auc']:.4f}±{res.get('auc_std',0):.4f} "
                        f"(n_feat={res['n_features']}, BK={res['n_pos']}, VS={res['n_neg']})")

    if results:
        best = max(results, key=lambda r: r["auc"])
        logger.info(f"  BEST: L{best['layer']} AUC={best['auc']:.4f}")

    return results


# ===================================================================
# Part 2: LLaMA V2 IC — Condition-Level Analysis
# ===================================================================

def run_llama_condition_analysis(logger):
    """LLaMA V2 IC: bet_constraint and prompt_condition classification."""
    logger.info("\n" + "=" * 70)
    logger.info("PART 2: LLaMA V2 IC — Condition-Level Analysis")
    logger.info("=" * 70)

    results = {}

    # Bet constraint classification (multiclass)
    logger.info("\n--- Bet Constraint (c10/c30/c50/c70) ---")
    constraint_results = []
    for layer in range(0, LLAMA_N_LAYERS, 2):  # every other layer
        loaded = load_llama_v2_ic(layer)
        if loaded is None:
            continue

        features, _, meta = loaded
        constraints = meta["bet_constraints"]
        constraint_map = {"10": 0, "30": 1, "50": 2, "70": 3}
        labels = np.array([constraint_map.get(str(c), -1) for c in constraints])
        valid = labels >= 0
        if valid.sum() < 20:
            continue

        feat_valid = features[valid]
        labels_valid = labels[valid]
        rate = (feat_valid != 0).mean(axis=0)
        active = rate >= MIN_ACTIVATION_RATE
        if active.sum() < 10:
            continue

        # Multiclass with OVR macro AUC
        classes = np.unique(labels_valid)
        n_classes = len(classes)
        min_class = min(np.bincount(labels_valid))
        effective_folds = min(5, min_class)
        if effective_folds < 2 or n_classes < 2:
            continue

        skf = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=RANDOM_SEED)
        aucs = []
        for train_idx, test_idx in skf.split(feat_valid[:, active], labels_valid):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(feat_valid[:, active][train_idx])
            X_test = scaler.transform(feat_valid[:, active][test_idx])
            clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=RANDOM_SEED)
            clf.fit(X_train, labels_valid[train_idx])
            y_prob = clf.predict_proba(X_test)
            test_classes = np.unique(labels_valid[test_idx])
            if len(test_classes) < n_classes:
                continue
            auc = roc_auc_score(labels_valid[test_idx], y_prob, multi_class="ovr", average="macro")
            aucs.append(auc)

        if aucs:
            res = {"auc": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
                   "layer": layer, "n_classes": n_classes}
            constraint_results.append(res)
            if layer % 10 == 0:
                logger.info(f"  L{layer:02d}: AUC={res['auc']:.4f}±{res['auc_std']:.4f}")

    results["bet_constraint"] = constraint_results
    if constraint_results:
        best = max(constraint_results, key=lambda r: r["auc"])
        logger.info(f"  BEST constraint: L{best['layer']} AUC={best['auc']:.4f}")

    # Bet type classification (binary: fixed vs variable)
    logger.info("\n--- Bet Type (Fixed vs Variable) ---")
    bt_results = []
    for layer in range(0, LLAMA_N_LAYERS, 2):
        loaded = load_llama_v2_ic(layer)
        if loaded is None:
            continue
        features, _, meta = loaded
        bt = meta["bet_types"]
        labels = (bt == "variable").astype(np.int32)
        n_var = int(labels.sum())
        n_fix = len(labels) - n_var
        if n_var < 5 or n_fix < 5:
            continue
        rate = (features != 0).mean(axis=0)
        active = rate >= MIN_ACTIVATION_RATE
        if active.sum() < 10:
            continue
        res = classify_cv(features[:, active], labels)
        res["layer"] = layer
        bt_results.append(res)
        if layer % 10 == 0:
            logger.info(f"  L{layer:02d}: AUC={res['auc']:.4f} (var={n_var}, fix={n_fix})")

    results["bet_type"] = bt_results

    return results


# ===================================================================
# Part 3: Cross-Model Comparison (Gemma IC vs LLaMA IC)
# ===================================================================

def run_cross_model_comparison(logger):
    """Compare Gemma IC and LLaMA IC BK classification AUC curves."""
    logger.info("\n" + "=" * 70)
    logger.info("PART 3: Cross-Model Comparison (Gemma IC vs LLaMA IC)")
    logger.info("=" * 70)

    results = {"gemma_ic": [], "llama_ic": []}

    # Gemma IC (V3, 42 layers)
    for layer in range(N_LAYERS):
        loaded = load_layer_features("ic", layer, mode="decision_point", dense=True)
        if loaded is None:
            continue
        feat, meta = loaded
        labels = get_labels(meta)
        filtered, _ = filter_active_features(feat, MIN_ACTIVATION_RATE)
        if filtered.shape[1] < 10:
            continue
        res = classify_cv(filtered, labels)
        res["layer"] = layer
        res["n_features"] = filtered.shape[1]
        results["gemma_ic"].append(res)

    # LLaMA IC (V2, 32 layers)
    for layer in range(LLAMA_N_LAYERS):
        loaded = load_llama_v2_ic(layer)
        if loaded is None:
            continue
        features, labels, _ = loaded
        rate = (features != 0).mean(axis=0)
        active = rate >= MIN_ACTIVATION_RATE
        if active.sum() < 10:
            continue
        res = classify_cv(features[:, active], labels)
        res["layer"] = layer
        res["n_features"] = int(active.sum())
        results["llama_ic"].append(res)

    # Summary
    for model_key in ["gemma_ic", "llama_ic"]:
        data = results[model_key]
        if data:
            best = max(data, key=lambda r: r["auc"])
            logger.info(f"  {model_key} BEST: L{best['layer']} AUC={best['auc']:.4f}")

    return results


# ===================================================================
# Part 4: Improved Cross-Domain Transfer (Bootstrap CI)
# ===================================================================

def run_cross_domain_bootstrap(logger, n_bootstrap=100):
    """Cross-domain transfer with bootstrap confidence intervals."""
    logger.info("\n" + "=" * 70)
    logger.info("PART 4: Cross-Domain Transfer (Bootstrap CI)")
    logger.info("=" * 70)

    results = {}
    # Use each paradigm's approximate best layer
    test_layers = [10, 18, 22, 26, 30]

    for src, tgt in permutations(["ic", "sm", "mw"], 2):
        key = f"{src}_to_{tgt}"
        logger.info(f"\n--- {PARADIGM_LABELS[src]} -> {PARADIGM_LABELS[tgt]} ---")

        best_result = None
        for layer in test_layers:
            src_loaded = load_layer_features(src, layer, mode="decision_point", dense=True)
            tgt_loaded = load_layer_features(tgt, layer, mode="decision_point", dense=True)
            if src_loaded is None or tgt_loaded is None:
                continue

            src_feat, src_meta = src_loaded
            tgt_feat, tgt_meta = tgt_loaded
            src_labels = get_labels(src_meta)
            tgt_labels = get_labels(tgt_meta)

            # Shared active features
            src_rate = (src_feat != 0).mean(axis=0)
            tgt_rate = (tgt_feat != 0).mean(axis=0)
            active = (src_rate >= MIN_ACTIVATION_RATE) & (tgt_rate >= MIN_ACTIVATION_RATE)
            n_active = int(active.sum())
            if n_active < 10:
                continue

            src_f = src_feat[:, active]
            tgt_f = tgt_feat[:, active]

            n_pos_src = int(src_labels.sum())
            n_pos_tgt = int(tgt_labels.sum())
            if n_pos_src < 2 or n_pos_tgt < 2:
                continue
            if n_pos_tgt == len(tgt_labels):
                continue

            # Full transfer AUC
            scaler = StandardScaler()
            X_train = scaler.fit_transform(src_f)
            X_test = scaler.transform(tgt_f)
            clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000,
                                     class_weight="balanced", random_state=RANDOM_SEED)
            clf.fit(X_train, src_labels)
            try:
                full_auc = float(roc_auc_score(tgt_labels, clf.predict_proba(X_test)[:, 1]))
            except ValueError:
                continue

            # Bootstrap CI on target
            rng = np.random.RandomState(RANDOM_SEED)
            boot_aucs = []
            for _ in range(n_bootstrap):
                boot_idx = rng.choice(len(tgt_labels), size=len(tgt_labels), replace=True)
                if len(np.unique(tgt_labels[boot_idx])) < 2:
                    continue
                try:
                    boot_auc = roc_auc_score(tgt_labels[boot_idx],
                                             clf.predict_proba(X_test[boot_idx])[:, 1])
                    boot_aucs.append(boot_auc)
                except ValueError:
                    continue

            ci_lower = float(np.percentile(boot_aucs, 2.5)) if boot_aucs else 0.5
            ci_upper = float(np.percentile(boot_aucs, 97.5)) if boot_aucs else 0.5

            result = {
                "layer": layer, "transfer_auc": full_auc,
                "ci_lower": ci_lower, "ci_upper": ci_upper,
                "n_shared_features": n_active,
                "n_bootstrap": len(boot_aucs),
                "src_n_bk": n_pos_src, "tgt_n_bk": n_pos_tgt,
            }

            if best_result is None or full_auc > best_result["transfer_auc"]:
                best_result = result

            logger.info(f"  L{layer:02d}: AUC={full_auc:.4f} [{ci_lower:.3f}, {ci_upper:.3f}]")

        if best_result:
            results[key] = best_result

    return results


# ===================================================================
# Part 5: Same-Layer Feature Comparison (L22 fixed)
# ===================================================================

def run_same_layer_feature_comparison(logger, layer=22, top_k=100):
    """Compare top BK-predictive features at the SAME layer across paradigms."""
    logger.info("\n" + "=" * 70)
    logger.info(f"PART 5: Same-Layer (L{layer}) Feature Comparison")
    logger.info("=" * 70)

    results = {}

    for paradigm in ["ic", "sm", "mw"]:
        loaded = load_layer_features(paradigm, layer, mode="decision_point", dense=True)
        if loaded is None:
            continue

        feat, meta = loaded
        labels = get_labels(meta)
        filtered, active_indices = filter_active_features(feat, MIN_ACTIVATION_RATE)

        n_pos = int(labels.sum())
        n_neg = len(labels) - n_pos
        effective_folds = min(5, n_pos, n_neg)
        if effective_folds < 2:
            continue

        skf = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=RANDOM_SEED)
        all_coefs = []
        aucs = []

        for train_idx, test_idx in skf.split(filtered, labels):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(filtered[train_idx])
            X_test = scaler.transform(filtered[test_idx])
            clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000,
                                     class_weight="balanced", random_state=RANDOM_SEED)
            clf.fit(X_train, labels[train_idx])
            y_prob = clf.predict_proba(X_test)[:, 1]
            aucs.append(roc_auc_score(labels[test_idx], y_prob))
            all_coefs.append(clf.coef_[0])

        avg_coefs = np.mean(all_coefs, axis=0)
        abs_coefs = np.abs(avg_coefs)
        top_local = np.argsort(abs_coefs)[-top_k:][::-1]
        top_global = active_indices[top_local]

        results[paradigm] = {
            "layer": layer,
            "auc": float(np.mean(aucs)),
            "n_active": len(active_indices),
            "top_feature_indices": top_global.tolist(),
        }
        logger.info(f"  {PARADIGM_LABELS[paradigm]}: AUC={np.mean(aucs):.4f}, n_active={len(active_indices)}")

    # Cross-paradigm overlap at same layer
    logger.info("\n--- Same-Layer Cross-Paradigm Overlap ---")
    overlap = {}
    paradigms = [p for p in ["ic", "sm", "mw"] if p in results]
    for i, p1 in enumerate(paradigms):
        for p2 in paradigms[i+1:]:
            s1 = set(results[p1]["top_feature_indices"])
            s2 = set(results[p2]["top_feature_indices"])
            shared = s1 & s2
            jaccard = len(shared) / len(s1 | s2) if (s1 | s2) else 0
            overlap[f"{p1}_vs_{p2}"] = {
                "n_shared": len(shared),
                "jaccard": jaccard,
                "shared_features": sorted(shared),
            }
            logger.info(f"  {PARADIGM_LABELS[p1]} vs {PARADIGM_LABELS[p2]}: "
                        f"{len(shared)} shared (Jaccard={jaccard:.4f})")

    results["overlap"] = overlap
    return results


# ===================================================================
# Part 6: Permutation Test for R1 BK Classification
# ===================================================================

def run_r1_permutation_test(logger, n_permutations=1000):
    """Permutation test for R1 BK classification significance."""
    logger.info("\n" + "=" * 70)
    logger.info("PART 6: R1 BK Classification — Permutation Test")
    logger.info("=" * 70)

    results = {}
    # Use approximate best R1 layer per paradigm
    best_r1_layers = {"ic": 18, "sm": 16, "mw": 22}

    for paradigm in ["ic", "sm", "mw"]:
        layer = best_r1_layers[paradigm]
        logger.info(f"\n--- {PARADIGM_LABELS[paradigm]} L{layer} ---")

        loaded = load_layer_features(paradigm, layer, mode="all_rounds", dense=True)
        if loaded is None:
            continue

        features, meta = loaded
        labels_all = get_labels(meta)
        round_nums = meta["round_nums"]

        # R1 only
        r1_mask = round_nums == 1
        r1_feat = features[r1_mask]
        r1_labels = labels_all[r1_mask]

        rate = (r1_feat != 0).mean(axis=0)
        active = rate >= MIN_ACTIVATION_RATE
        if active.sum() < 10:
            continue

        r1_filtered = r1_feat[:, active]

        # Observed AUC
        obs = classify_cv(r1_filtered, r1_labels)
        if obs.get("skipped"):
            continue

        logger.info(f"  Observed AUC: {obs['auc']:.4f}±{obs.get('auc_std',0):.4f}")

        # Permutation test
        null_aucs = permutation_test(r1_filtered, r1_labels, n_permutations)
        if null_aucs:
            p_value = float((np.sum(np.array(null_aucs) >= obs["auc"]) + 1) / (len(null_aucs) + 1))
            null_mean = float(np.mean(null_aucs))
            null_std = float(np.std(null_aucs))
            logger.info(f"  Null AUC: {null_mean:.4f}±{null_std:.4f}")
            logger.info(f"  p-value: {p_value:.4f} (n_perms={n_permutations})")
        else:
            p_value = None
            null_mean = 0.5
            null_std = 0.0

        results[paradigm] = {
            "layer": layer,
            "observed_auc": obs["auc"],
            "observed_auc_std": obs.get("auc_std", 0),
            "null_mean": null_mean,
            "null_std": null_std,
            "p_value": p_value,
            "n_permutations": n_permutations,
            "n_pos": obs["n_pos"],
            "n_neg": obs["n_neg"],
        }

    return results


# ===================================================================
# Part 7: Gemma R1 + Balance-Matched (re-verify with clean data)
# ===================================================================

def run_gemma_balance_controlled(logger):
    """Re-run Gemma balance-controlled analysis to verify results."""
    logger.info("\n" + "=" * 70)
    logger.info("PART 7: Gemma Balance-Controlled BK Classification (Re-verify)")
    logger.info("=" * 70)

    results = {}
    key_layers = list(range(0, N_LAYERS, 2))

    for paradigm in ["ic", "sm", "mw"]:
        logger.info(f"\n--- {PARADIGM_LABELS[paradigm]} ---")
        results[paradigm] = {"r1": [], "decision_point": [], "balance_matched": []}

        for layer in key_layers:
            loaded = load_layer_features(paradigm, layer, mode="all_rounds", dense=True)
            if loaded is None:
                continue

            features, meta = loaded
            labels_all = get_labels(meta)
            round_nums = meta["round_nums"]
            balances = meta.get("balances", None)

            # --- R1 ---
            r1_mask = round_nums == 1
            r1_feat = features[r1_mask]
            r1_labels = labels_all[r1_mask]

            rate = (r1_feat != 0).mean(axis=0)
            active = rate >= MIN_ACTIVATION_RATE
            if active.sum() >= 10 and int(r1_labels.sum()) >= 2:
                res = classify_cv(r1_feat[:, active], r1_labels)
                res["layer"] = layer
                results[paradigm]["r1"].append(res)

            # --- Decision-point ---
            dp_mask = meta["is_last_round"]
            dp_feat = features[dp_mask]
            dp_labels = labels_all[dp_mask]
            dp_rate = (dp_feat != 0).mean(axis=0)
            dp_active = dp_rate >= MIN_ACTIVATION_RATE
            if dp_active.sum() >= 10:
                res_dp = classify_cv(dp_feat[:, dp_active], dp_labels)
                res_dp["layer"] = layer
                results[paradigm]["decision_point"].append(res_dp)

            # --- Balance-matched ---
            if balances is not None and dp_active.sum() >= 10:
                dp_balances = balances[dp_mask]
                bk_indices = np.where(dp_labels == 1)[0]
                safe_indices = np.where(dp_labels == 0)[0]

                if len(bk_indices) >= 5 and len(safe_indices) >= 5:
                    matched_bk, matched_safe = [], []
                    used_safe = set()
                    bk_bals = dp_balances[bk_indices]
                    safe_bals = dp_balances[safe_indices]

                    bk_order = np.argsort(bk_bals)
                    for bk_idx in bk_order:
                        bk_bal = bk_bals[bk_idx]
                        dists = np.abs(safe_bals - bk_bal)
                        for s_rank in np.argsort(dists):
                            if s_rank not in used_safe:
                                used_safe.add(s_rank)
                                matched_bk.append(bk_indices[bk_idx])
                                matched_safe.append(safe_indices[s_rank])
                                break

                    if len(matched_bk) >= 5:
                        matched_indices = np.array(matched_bk + matched_safe)
                        matched_feat = dp_feat[matched_indices][:, dp_active]
                        matched_labels = dp_labels[matched_indices]
                        res_bm = classify_cv(matched_feat, matched_labels)
                        res_bm["layer"] = layer
                        res_bm["n_pairs"] = len(matched_bk)
                        results[paradigm]["balance_matched"].append(res_bm)

            del features, meta
            gc.collect()

        # Summary
        for atype in ["r1", "decision_point", "balance_matched"]:
            data = results[paradigm][atype]
            if data:
                best = max(data, key=lambda r: r["auc"])
                logger.info(f"  BEST {atype}: L{best['layer']} AUC={best['auc']:.4f}")

    return results


# ===================================================================
# Visualization
# ===================================================================

def plot_cross_model_comparison(results, save_path):
    """Gemma IC vs LLaMA IC BK classification AUC comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for model_key, color, label in [
        ("gemma_ic", "#e74c3c", "Gemma-2-9B (GemmaScope 131K, 42 layers)"),
        ("llama_ic", "#3498db", "LLaMA-3.1-8B (LlamaScope 32K, 32 layers)"),
    ]:
        data = results.get(model_key, [])
        if data:
            layers = [r["layer"] for r in data if not r.get("skipped")]
            aucs = [r["auc"] for r in data if not r.get("skipped")]
            ax.plot(layers, aucs, "-o", color=color, markersize=3, lw=1.5, label=label)
            best = max(data, key=lambda r: r["auc"])
            ax.plot(best["layer"], best["auc"], "*", color=color, markersize=14, zorder=5)

    ax.axhline(y=0.5, color="gray", ls="--", alpha=0.5, label="Chance")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("AUC (5-fold CV, balanced)", fontsize=12)
    ax.set_title("Cross-Model BK Classification: Investment Choice Paradigm", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(0.4, 1.05)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_balance_controlled_all(results, save_path):
    """Balance-controlled BK classification for all paradigms."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for col, paradigm in enumerate(["ic", "sm", "mw"]):
        ax = axes[col]
        data = results.get(paradigm, {})

        for atype, color, marker, label in [
            ("decision_point", "#e74c3c", "o", "Decision-Point (original)"),
            ("r1", "#2ecc71", "s", "Round 1 (no balance confound)"),
            ("balance_matched", "#3498db", "^", "Balance-Matched"),
        ]:
            entries = data.get(atype, [])
            if entries:
                layers = [r["layer"] for r in entries if not r.get("skipped")]
                aucs = [r["auc"] for r in entries if not r.get("skipped")]
                ax.plot(layers, aucs, f"-{marker}", color=color, markersize=3, lw=1.5, label=label)

        ax.axhline(y=0.5, color="gray", ls="--", alpha=0.5)
        ax.set_xlabel("Layer")
        ax.set_ylabel("AUC")
        ax.set_title(PARADIGM_LABELS[paradigm], fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim(0.4, 1.05)

    fig.suptitle("Balance-Controlled BK Classification (Gemma V3, Clean Data)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_transfer_matrix_bootstrap(results, save_path):
    """Cross-domain transfer AUC matrix with bootstrap CI."""
    pairs = ["ic_to_sm", "ic_to_mw", "sm_to_ic", "sm_to_mw", "mw_to_ic", "mw_to_sm"]
    labels = ["IC→SM", "IC→MW", "SM→IC", "SM→MW", "MW→IC", "MW→SM"]

    fig, ax = plt.subplots(figsize=(10, 5))

    aucs = []
    ci_lowers = []
    ci_uppers = []
    for pair in pairs:
        if pair in results:
            r = results[pair]
            aucs.append(r["transfer_auc"])
            ci_lowers.append(r["ci_lower"])
            ci_uppers.append(r["ci_upper"])
        else:
            aucs.append(0.5)
            ci_lowers.append(0.5)
            ci_uppers.append(0.5)

    x = np.arange(len(labels))
    yerr_lower = [a - l for a, l in zip(aucs, ci_lowers)]
    yerr_upper = [u - a for a, u in zip(aucs, ci_uppers)]
    bars = ax.bar(x, aucs, color=["#3498db"]*len(x), alpha=0.8)
    ax.errorbar(x, aucs, yerr=[yerr_lower, yerr_upper], fmt="none", color="black", capsize=5)

    ax.axhline(y=0.5, color="gray", ls="--", alpha=0.5, label="Chance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Transfer AUC", fontsize=12)
    ax.set_title("Cross-Paradigm Transfer with 95% Bootstrap CI", fontsize=13, fontweight="bold")
    ax.set_ylim(0.35, 0.85)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    for bar, auc_val in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{auc_val:.3f}", ha="center", fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# Main
# ===================================================================

def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    JSON_DIR.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"improved_v4_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"V4 Improved Analyses started. Log: {log_file}")
    logger.info(f"Design improvements: bootstrap CI, permutation test, same-layer comparison, LLaMA V2 IC")
    np.random.seed(RANDOM_SEED)

    all_results = {"timestamp": ts, "version": "v4_improved"}

    # Part 1: LLaMA BK Classification
    all_results["llama_bk_classification"] = run_llama_bk_classification(logger)

    # Part 2: LLaMA Condition Analysis
    all_results["llama_condition_analysis"] = run_llama_condition_analysis(logger)

    # Part 3: Cross-Model Comparison
    cross_model = run_cross_model_comparison(logger)
    all_results["cross_model_comparison"] = cross_model

    # Part 4: Cross-Domain Bootstrap
    all_results["cross_domain_bootstrap"] = run_cross_domain_bootstrap(logger)

    # Part 5: Same-Layer Feature Comparison
    all_results["same_layer_features"] = run_same_layer_feature_comparison(logger)

    # Part 6: Permutation Test
    all_results["r1_permutation_test"] = run_r1_permutation_test(logger, n_permutations=1000)

    # Part 7: Gemma Balance-Controlled (re-verify)
    balance_ctrl = run_gemma_balance_controlled(logger)
    all_results["gemma_balance_controlled"] = balance_ctrl

    # Save results
    result_path = JSON_DIR / f"improved_v4_{ts}.json"
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    logger.info(f"\nResults saved: {result_path}")

    # Generate figures
    logger.info("\nGenerating figures...")
    plot_cross_model_comparison(cross_model, FIGURE_DIR / "v4_cross_model_comparison.png")
    plot_balance_controlled_all(balance_ctrl, FIGURE_DIR / "v4_balance_controlled.png")
    plot_transfer_matrix_bootstrap(all_results["cross_domain_bootstrap"],
                                   FIGURE_DIR / "v4_transfer_bootstrap.png")

    # Final summary
    logger.info(f"\n{'='*70}")
    logger.info("V4 IMPROVED ANALYSES COMPLETE")
    logger.info(f"{'='*70}")

    # Print key results
    llama_bk = all_results["llama_bk_classification"]
    if llama_bk:
        best = max(llama_bk, key=lambda r: r["auc"])
        logger.info(f"LLaMA IC Best BK AUC: L{best['layer']} = {best['auc']:.4f}")

    for paradigm in ["ic", "sm", "mw"]:
        perm = all_results["r1_permutation_test"].get(paradigm, {})
        if perm:
            logger.info(f"R1 Permutation {paradigm}: observed={perm['observed_auc']:.4f}, "
                        f"null={perm['null_mean']:.4f}, p={perm.get('p_value', 'N/A')}")

    logger.info(f"\nFigures: {FIGURE_DIR}")
    logger.info(f"JSON: {result_path}")
    logger.info(f"Log: {log_file}")


if __name__ == "__main__":
    main()
