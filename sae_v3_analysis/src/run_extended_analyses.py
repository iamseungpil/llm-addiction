#!/usr/bin/env python3
"""
Extended V3 SAE analyses — 4 new experiment categories.

Exp 2a: Balance-controlled BK classification (R1 + balance-matched)
Exp 2b: Feature importance (top features, cross-paradigm overlap)
Exp 3:  Round-level risk choice classification (invest/non-invest)
Exp 4:  Condition-level activation analysis (bet_type, constraint, prompt)

Usage:
    cd /home/jovyan/llm-addiction/sae_v3_analysis/src
    python run_extended_analyses.py
"""

import gc
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from config import (
    PARADIGMS, N_LAYERS, FIGURE_DIR, JSON_DIR, LOG_DIR,
    MIN_ACTIVATION_RATE, RANDOM_SEED, DATA_ROOT,
    PARADIGM_COLORS, PARADIGM_LABELS,
)
from data_loader import (
    load_layer_features, load_hidden_states,
    filter_active_features, get_labels,
)


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
# Shared utilities
# ===================================================================

def classify_cv(features, labels, n_folds=5):
    """5-fold stratified CV with balanced class weights."""
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    effective_folds = min(n_folds, n_pos, n_neg)

    if effective_folds < 2:
        return {"auc": 0.5, "f1": 0.0, "n_pos": n_pos, "n_neg": n_neg, "skipped": True}

    skf = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=RANDOM_SEED)
    aucs, f1s, precs, recs = [], [], [], []

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
        precs.append(precision_score(labels[test_idx], y_pred, zero_division=0))
        recs.append(recall_score(labels[test_idx], y_pred, zero_division=0))

    return {
        "auc": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
        "f1": float(np.mean(f1s)), "precision": float(np.mean(precs)),
        "recall": float(np.mean(recs)),
        "n_pos": n_pos, "n_neg": n_neg, "skipped": False,
    }


def classify_cv_with_coefs(features, labels, n_folds=5):
    """Like classify_cv but also returns averaged LR coefficients."""
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    effective_folds = min(n_folds, n_pos, n_neg)

    if effective_folds < 2:
        return {"auc": 0.5, "skipped": True, "coefs": None, "n_pos": n_pos, "n_neg": n_neg}

    skf = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=RANDOM_SEED)
    aucs = []
    all_coefs = []

    for train_idx, test_idx in skf.split(features, labels):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(features[train_idx])
        X_test = scaler.transform(features[test_idx])

        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000,
                                 class_weight="balanced", random_state=RANDOM_SEED)
        clf.fit(X_train, labels[train_idx])
        y_prob = clf.predict_proba(X_test)[:, 1]
        aucs.append(roc_auc_score(labels[test_idx], y_prob))
        all_coefs.append(clf.coef_[0])

    return {
        "auc": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
        "coefs": np.mean(all_coefs, axis=0),
        "n_pos": n_pos, "n_neg": n_neg, "skipped": False,
    }


def classify_multiclass_cv(features, labels, n_folds=5):
    """Multiclass stratified CV with macro OVR AUC."""
    classes, counts = np.unique(labels, return_counts=True)
    n_classes = len(classes)
    min_class = int(counts.min())

    if n_classes < 2 or min_class < 2:
        return {"auc": 0.5, "auc_std": 0.0, "n_classes": n_classes, "skipped": True}

    effective_folds = min(n_folds, min_class)
    if effective_folds < 2:
        return {"auc": 0.5, "auc_std": 0.0, "n_classes": n_classes, "skipped": True}

    skf = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=RANDOM_SEED)
    aucs = []

    for train_idx, test_idx in skf.split(features, labels):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(features[train_idx])
        X_test = scaler.transform(features[test_idx])

        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000,
                                 random_state=RANDOM_SEED)
        clf.fit(X_train, labels[train_idx])
        y_prob = clf.predict_proba(X_test)

        # Ensure all classes present in test fold
        test_classes = np.unique(labels[test_idx])
        if len(test_classes) < n_classes:
            continue

        auc = roc_auc_score(labels[test_idx], y_prob, multi_class="ovr", average="macro")
        aucs.append(auc)

    if not aucs:
        return {"auc": 0.5, "auc_std": 0.0, "n_classes": n_classes, "skipped": True}

    return {
        "auc": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
        "n_classes": n_classes,
        "class_counts": {str(c): int(cnt) for c, cnt in zip(classes, counts)},
        "skipped": False,
    }


# ===================================================================
# Game JSON loaders (for per-round choice data)
# ===================================================================

IC_JSON_FILES = [
    (DATA_ROOT / "investment_choice_v2_role" / "gemma_investment_c10_20260225_122319.json", 0),
    (DATA_ROOT / "investment_choice_v2_role" / "gemma_investment_c30_20260225_184458.json", 400),
    (DATA_ROOT / "investment_choice_v2_role" / "gemma_investment_c50_20260226_020821.json", 800),
    (DATA_ROOT / "investment_choice_v2_role" / "gemma_investment_c70_20260226_082029.json", 1200),
]
SM_JSON_FILE = DATA_ROOT / "slot_machine" / "experiment_0_gemma_v4_role" / "final_gemma_20260227_002507.json"
MW_JSON_FILE = DATA_ROOT / "mystery_wheel_v2_role" / "gemma_mysterywheel_checkpoint_3200.json"


def load_ic_choices():
    """Load IC per-round choices. Returns {npz_game_id: {round_num: choice(1-4)}}."""
    choices = {}
    for json_path, offset in IC_JSON_FILES:
        with open(json_path) as f:
            data = json.load(f)
        for i, game in enumerate(data["results"]):
            npz_gid = offset + i + 1
            choices[npz_gid] = {}
            for dec in game.get("decisions", []):
                choices[npz_gid][dec["round"]] = dec["choice"]
    return choices


def load_sm_choices():
    """Load SM per-round actions. Returns {npz_game_id: {round_num: {'action', 'bet'}}}."""
    with open(SM_JSON_FILE) as f:
        data = json.load(f)
    choices = {}
    for i, game in enumerate(data["results"]):
        npz_gid = i + 1
        choices[npz_gid] = {}
        for dec in game.get("decisions", []):
            choices[npz_gid][dec["round"]] = {
                "action": dec["action"],
                "bet": dec.get("bet", 0),
            }
    return choices


def load_mw_choices():
    """Load MW per-round choices. Returns {npz_game_id: {round_num: {'choice', 'bet_amount'}}}."""
    with open(MW_JSON_FILE) as f:
        data = json.load(f)
    choices = {}
    for i, game in enumerate(data["results"]):
        npz_gid = i + 1
        choices[npz_gid] = {}
        for dec in game.get("decisions", []):
            choices[npz_gid][dec["round"]] = {
                "choice": dec["choice"],
                "bet_amount": dec.get("bet_amount", 0),
            }
    return choices


# ===================================================================
# Exp 2a: Balance-Controlled BK Classification
# ===================================================================

def run_exp2a(logger):
    """Balance-controlled BK classification: R1 analysis + balance-matched."""
    logger.info("\n" + "=" * 70)
    logger.info("EXP 2a: Balance-Controlled BK Classification")
    logger.info("=" * 70)

    results = {}
    key_layers = list(range(0, N_LAYERS, 2))  # every other layer

    for paradigm in ["ic", "sm", "mw"]:
        cfg = PARADIGMS[paradigm]
        logger.info(f"\n--- {cfg['name']} ---")
        results[paradigm] = {"r1": [], "balanced_matched": [], "decision_point_original": []}

        for layer in key_layers:
            loaded = load_layer_features(paradigm, layer, mode="all_rounds", dense=True)
            if loaded is None:
                continue

            features, meta = loaded
            labels_all = get_labels(meta)
            game_ids = meta["game_ids"]
            round_nums = meta["round_nums"]
            balances = meta.get("balances", None)

            # --- R1 Analysis (no balance confound) ---
            r1_mask = round_nums == 1
            r1_feat = features[r1_mask]
            r1_labels = labels_all[r1_mask]

            # Filter active features
            rate = (r1_feat != 0).mean(axis=0)
            active = rate >= MIN_ACTIVATION_RATE
            if active.sum() >= 10 and int(r1_labels.sum()) >= 2:
                r1_filtered = r1_feat[:, active]
                res_r1 = classify_cv(r1_filtered, r1_labels)
                res_r1["layer"] = layer
                res_r1["n_features"] = int(active.sum())
                results[paradigm]["r1"].append(res_r1)

                if layer % 10 == 0 or layer == 22:
                    logger.info(f"  L{layer:02d} R1:      AUC={res_r1['auc']:.4f}±{res_r1.get('auc_std',0):.4f} "
                                f"(BK={res_r1['n_pos']}, safe={res_r1['n_neg']})")

            # --- Decision-Point Original (for comparison) ---
            dp_mask = meta["is_last_round"]
            dp_feat = features[dp_mask]
            dp_labels = labels_all[dp_mask]
            dp_rate = (dp_feat != 0).mean(axis=0)
            dp_active = dp_rate >= MIN_ACTIVATION_RATE
            if dp_active.sum() >= 10:
                dp_filtered = dp_feat[:, dp_active]
                res_dp = classify_cv(dp_filtered, dp_labels)
                res_dp["layer"] = layer
                results[paradigm]["decision_point_original"].append(res_dp)

            # --- Balance-Matched Analysis ---
            if balances is not None and dp_active.sum() >= 10:
                dp_balances = balances[dp_mask] if hasattr(balances, '__getitem__') else meta["balances"][dp_mask]
                bk_indices = np.where(dp_labels == 1)[0]
                safe_indices = np.where(dp_labels == 0)[0]

                if len(bk_indices) >= 5 and len(safe_indices) >= 5:
                    # For each BK game, find closest-balance safe game (without replacement)
                    matched_bk = []
                    matched_safe = []
                    used_safe = set()
                    bk_bals = dp_balances[bk_indices]
                    safe_bals = dp_balances[safe_indices]

                    # Sort BK by balance for stable matching
                    bk_order = np.argsort(bk_bals)
                    for bk_idx in bk_order:
                        bk_bal = bk_bals[bk_idx]
                        # Find closest unused safe game
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
                        res_bm["n_matched_pairs"] = len(matched_bk)

                        # Report balance overlap
                        bk_matched_bals = dp_balances[matched_bk]
                        safe_matched_bals = dp_balances[matched_safe]
                        res_bm["bk_bal_mean"] = float(np.mean(bk_matched_bals))
                        res_bm["safe_bal_mean"] = float(np.mean(safe_matched_bals))
                        res_bm["bal_diff_mean"] = float(np.mean(np.abs(bk_matched_bals - safe_matched_bals)))

                        results[paradigm]["balanced_matched"].append(res_bm)

                        if layer % 10 == 0 or layer == 22:
                            logger.info(f"  L{layer:02d} Matched: AUC={res_bm['auc']:.4f} "
                                        f"(n_pairs={len(matched_bk)}, bal_diff={res_bm['bal_diff_mean']:.1f})")

        # Summary
        for analysis_type in ["r1", "balanced_matched", "decision_point_original"]:
            data = results[paradigm][analysis_type]
            if data:
                best = max(data, key=lambda r: r["auc"])
                logger.info(f"  BEST {analysis_type}: L{best['layer']} AUC={best['auc']:.4f}")

    return results


# ===================================================================
# Exp 2b: Feature Importance Analysis
# ===================================================================

def run_exp2b(logger):
    """Feature importance: top LR coefficients at best layer, cross-paradigm overlap."""
    logger.info("\n" + "=" * 70)
    logger.info("EXP 2b: Feature Importance Analysis")
    logger.info("=" * 70)

    results = {}
    best_layers = {"ic": 22, "sm": 12, "mw": 33}
    top_k = 100

    for paradigm in ["ic", "sm", "mw"]:
        layer = best_layers[paradigm]
        logger.info(f"\n--- {PARADIGM_LABELS[paradigm]} L{layer} ---")

        loaded = load_layer_features(paradigm, layer, mode="decision_point", dense=True)
        if loaded is None:
            continue

        feat, meta = loaded
        labels = get_labels(meta)
        filtered, active_indices = filter_active_features(feat, MIN_ACTIVATION_RATE)

        res = classify_cv_with_coefs(filtered, labels)
        if res["skipped"] or res["coefs"] is None:
            continue

        coefs = res["coefs"]
        # Top features by absolute coefficient
        abs_coefs = np.abs(coefs)
        top_local = np.argsort(abs_coefs)[-top_k:][::-1]
        top_global_indices = active_indices[top_local]

        top_features = []
        for rank, (local_idx, global_idx) in enumerate(zip(top_local, top_global_indices)):
            top_features.append({
                "rank": rank + 1,
                "feature_idx": int(global_idx),
                "coef": float(coefs[local_idx]),
                "abs_coef": float(abs_coefs[local_idx]),
                "direction": "BK+" if coefs[local_idx] > 0 else "BK-",
            })

        results[paradigm] = {
            "layer": layer,
            "auc": res["auc"],
            "n_active_features": len(active_indices),
            "top_features": top_features,
            "top_feature_indices": top_global_indices.tolist(),
        }

        logger.info(f"  AUC={res['auc']:.4f}, n_active={len(active_indices)}")
        logger.info(f"  Top 10 BK+ features: {[f['feature_idx'] for f in top_features[:10] if f['direction']=='BK+'][:10]}")
        logger.info(f"  Top 10 BK- features: {[f['feature_idx'] for f in top_features if f['direction']=='BK-'][:10]}")

    # Cross-paradigm overlap
    logger.info("\n--- Cross-Paradigm Feature Overlap ---")
    overlap = {}
    paradigms = [p for p in ["ic", "sm", "mw"] if p in results]
    for i, p1 in enumerate(paradigms):
        for p2 in paradigms[i+1:]:
            s1 = set(results[p1]["top_feature_indices"])
            s2 = set(results[p2]["top_feature_indices"])
            shared = s1 & s2
            overlap[f"{p1}_vs_{p2}"] = {
                "n_shared": len(shared),
                "shared_features": sorted(shared),
                "jaccard": len(shared) / len(s1 | s2) if s1 | s2 else 0,
            }
            logger.info(f"  {PARADIGM_LABELS[p1]} vs {PARADIGM_LABELS[p2]}: "
                        f"{len(shared)} shared features (Jaccard={overlap[f'{p1}_vs_{p2}']['jaccard']:.3f})")

    results["cross_paradigm_overlap"] = overlap
    return results


# ===================================================================
# Exp 3: Round-Level Risk Choice Classification
# ===================================================================

def run_exp3(logger):
    """Round-level risk choice classification using per-round choice data."""
    logger.info("\n" + "=" * 70)
    logger.info("EXP 3: Round-Level Risk Choice Classification")
    logger.info("=" * 70)

    results = {}
    key_layers = list(range(0, N_LAYERS, 2))

    # --- IC: risky (choice 3,4) vs safe (choice 1,2) ---
    logger.info("\n--- IC: Risky (choice 3,4) vs Safe (choice 1,2) ---")
    ic_choices = load_ic_choices()
    results["ic_risk_choice"] = []

    for layer in key_layers:
        loaded = load_layer_features("ic", layer, mode="all_rounds", dense=True)
        if loaded is None:
            continue

        features, meta = loaded
        game_ids = meta["game_ids"]
        round_nums = meta["round_nums"]

        # Build per-round risk labels
        risk_labels = np.full(len(game_ids), -1, dtype=np.int32)
        for i in range(len(game_ids)):
            gid = int(game_ids[i])
            rnd = int(round_nums[i])
            if gid in ic_choices and rnd in ic_choices[gid]:
                choice = ic_choices[gid][rnd]
                if choice in (3, 4):
                    risk_labels[i] = 1  # risky
                elif choice in (1, 2):
                    risk_labels[i] = 0  # safe

        valid = risk_labels >= 0
        if valid.sum() < 20:
            continue

        feat_valid = features[valid]
        labels_valid = risk_labels[valid]

        rate = (feat_valid != 0).mean(axis=0)
        active = rate >= MIN_ACTIVATION_RATE
        if active.sum() < 10:
            continue

        res = classify_cv(feat_valid[:, active], labels_valid)
        res["layer"] = layer
        res["n_features"] = int(active.sum())
        results["ic_risk_choice"].append(res)

        if layer % 10 == 0 or layer == 22:
            logger.info(f"  L{layer:02d}: AUC={res['auc']:.4f}±{res.get('auc_std',0):.4f} "
                        f"(risky={res['n_pos']}, safe={res['n_neg']})")

    # IC: risky choice in BK vs non-BK games separately
    logger.info("\n--- IC: Risk Choice by Game Outcome ---")
    results["ic_risk_by_outcome"] = {}
    layer = 22
    # Reuse last loaded if it was L22, otherwise reload
    loaded = load_layer_features("ic", layer, mode="all_rounds", dense=True)
    if loaded is not None:
        features, meta = loaded
        game_ids = meta["game_ids"]
        round_nums = meta["round_nums"]
        outcomes = meta["game_outcomes"]

        for outcome_label, outcome_val in [("bk_games", "bankruptcy"), ("safe_games", "voluntary_stop")]:
            outcome_mask = outcomes == outcome_val
            risk_labels = np.full(len(game_ids), -1, dtype=np.int32)
            for i in range(len(game_ids)):
                if not outcome_mask[i]:
                    continue
                gid = int(game_ids[i])
                rnd = int(round_nums[i])
                if gid in ic_choices and rnd in ic_choices[gid]:
                    choice = ic_choices[gid][rnd]
                    if choice in (3, 4):
                        risk_labels[i] = 1
                    elif choice in (1, 2):
                        risk_labels[i] = 0

            valid = risk_labels >= 0
            if valid.sum() < 20 or int((risk_labels[valid] == 1).sum()) < 2:
                continue

            feat_valid = features[valid]
            labels_valid = risk_labels[valid]
            rate = (feat_valid != 0).mean(axis=0)
            active = rate >= MIN_ACTIVATION_RATE
            if active.sum() < 10:
                continue

            res = classify_cv(feat_valid[:, active], labels_valid)
            res["layer"] = layer
            results["ic_risk_by_outcome"][outcome_label] = res
            logger.info(f"  {outcome_label} L{layer}: AUC={res['auc']:.4f} "
                        f"(risky={res['n_pos']}, safe={res['n_neg']})")

    # --- SM variable: high bet vs low bet ---
    logger.info("\n--- SM Variable: High Bet vs Low Bet ---")
    sm_choices = load_sm_choices()
    results["sm_bet_magnitude"] = []

    for layer in key_layers:
        loaded = load_layer_features("sm", layer, mode="all_rounds", dense=True)
        if loaded is None:
            continue

        features, meta = loaded
        game_ids = meta["game_ids"]
        round_nums = meta["round_nums"]
        bet_types = meta["bet_types"]

        # Only variable-bet rounds where action is 'bet'
        bet_amounts = []
        valid_indices = []
        for i in range(len(game_ids)):
            if bet_types[i] != "variable":
                continue
            gid = int(game_ids[i])
            rnd = int(round_nums[i])
            if gid in sm_choices and rnd in sm_choices[gid]:
                info = sm_choices[gid][rnd]
                if info["action"] == "bet" and info["bet"] > 0:
                    bet_amounts.append(info["bet"])
                    valid_indices.append(i)

        if len(bet_amounts) < 50:
            continue

        bet_amounts = np.array(bet_amounts)
        valid_indices = np.array(valid_indices)
        median_bet = np.median(bet_amounts)

        # Binary: high (> median) vs low (<= median)
        high_mask = bet_amounts > median_bet
        labels = high_mask.astype(np.int32)

        feat_valid = features[valid_indices]
        rate = (feat_valid != 0).mean(axis=0)
        active = rate >= MIN_ACTIVATION_RATE
        if active.sum() < 10:
            continue

        res = classify_cv(feat_valid[:, active], labels)
        res["layer"] = layer
        res["median_bet"] = float(median_bet)
        results["sm_bet_magnitude"].append(res)

        if layer % 10 == 0 or layer == 22:
            logger.info(f"  L{layer:02d}: AUC={res['auc']:.4f} "
                        f"(high={res['n_pos']}, low={res['n_neg']}, median=${median_bet:.0f})")

        del features, meta
        gc.collect()

    # --- MW variable: bet amount classification ---
    logger.info("\n--- MW Variable: High Bet vs Low Bet ---")
    mw_choices = load_mw_choices()
    results["mw_bet_magnitude"] = []

    for layer in key_layers:
        loaded = load_layer_features("mw", layer, mode="all_rounds", dense=True)
        if loaded is None:
            continue

        features, meta = loaded
        game_ids = meta["game_ids"]
        round_nums = meta["round_nums"]
        bet_types = meta["bet_types"]

        bet_amounts = []
        valid_indices = []
        for i in range(len(game_ids)):
            if bet_types[i] != "variable":
                continue
            gid = int(game_ids[i])
            rnd = int(round_nums[i])
            if gid in mw_choices and rnd in mw_choices[gid]:
                info = mw_choices[gid][rnd]
                if info["bet_amount"] > 0:
                    bet_amounts.append(info["bet_amount"])
                    valid_indices.append(i)

        if len(bet_amounts) < 50:
            continue

        bet_amounts = np.array(bet_amounts)
        valid_indices = np.array(valid_indices)
        median_bet = np.median(bet_amounts)

        high_mask = bet_amounts > median_bet
        labels = high_mask.astype(np.int32)

        feat_valid = features[valid_indices]
        rate = (feat_valid != 0).mean(axis=0)
        active = rate >= MIN_ACTIVATION_RATE
        if active.sum() < 10:
            continue

        res = classify_cv(feat_valid[:, active], labels)
        res["layer"] = layer
        res["median_bet"] = float(median_bet)
        results["mw_bet_magnitude"].append(res)

        if layer % 10 == 0 or layer == 22:
            logger.info(f"  L{layer:02d}: AUC={res['auc']:.4f} "
                        f"(high={res['n_pos']}, low={res['n_neg']}, median=${median_bet:.0f})")

    # Summary
    for key in ["ic_risk_choice", "sm_bet_magnitude", "mw_bet_magnitude"]:
        data = results.get(key, [])
        if data:
            best = max(data, key=lambda r: r["auc"])
            logger.info(f"  BEST {key}: L{best['layer']} AUC={best['auc']:.4f}")

    return results


# ===================================================================
# Exp 4: Condition-Level Analysis
# ===================================================================

def run_exp4(logger):
    """Condition-level activation analysis: bet_type, constraint, prompt."""
    logger.info("\n" + "=" * 70)
    logger.info("EXP 4: Condition-Level Activation Analysis")
    logger.info("=" * 70)

    results = {}
    key_layers = list(range(0, N_LAYERS, 2))

    # --- 4a: IC bet_constraint classification (4-class) ---
    logger.info("\n--- 4a: IC Bet Constraint (c10/c30/c50/c70) ---")
    results["ic_bet_constraint"] = []

    for layer in key_layers:
        loaded = load_layer_features("ic", layer, mode="decision_point", dense=True)
        if loaded is None:
            continue

        feat, meta = loaded
        constraints = meta["bet_constraints"]
        # Encode as 0-3
        constraint_map = {"10": 0, "30": 1, "50": 2, "70": 3}
        labels = np.array([constraint_map.get(str(c), -1) for c in constraints])
        valid = labels >= 0
        if valid.sum() < 20:
            continue

        feat_valid = feat[valid]
        labels_valid = labels[valid]

        rate = (feat_valid != 0).mean(axis=0)
        active = rate >= MIN_ACTIVATION_RATE
        if active.sum() < 10:
            continue

        res = classify_multiclass_cv(feat_valid[:, active], labels_valid)
        if res["skipped"]:
            continue
        res["layer"] = layer
        results["ic_bet_constraint"].append(res)

        if layer % 10 == 0 or layer == 22:
            logger.info(f"  L{layer:02d}: AUC={res['auc']:.4f}±{res['auc_std']:.4f} (4-class)")

    # --- 4b: IC prompt condition (BASE/G/GM/M) ---
    logger.info("\n--- 4b: IC Prompt Condition (BASE/G/GM/M) ---")
    results["ic_prompt_condition"] = []

    for layer in key_layers:
        loaded = load_layer_features("ic", layer, mode="decision_point", dense=True)
        if loaded is None:
            continue

        feat, meta = loaded
        prompts = meta["prompt_conditions"]
        prompt_map = {"BASE": 0, "G": 1, "GM": 2, "M": 3}
        labels = np.array([prompt_map.get(str(p), -1) for p in prompts])
        valid = labels >= 0
        if valid.sum() < 20:
            continue

        feat_valid = feat[valid]
        labels_valid = labels[valid]

        rate = (feat_valid != 0).mean(axis=0)
        active = rate >= MIN_ACTIVATION_RATE
        if active.sum() < 10:
            continue

        res = classify_multiclass_cv(feat_valid[:, active], labels_valid)
        if res["skipped"]:
            continue
        res["layer"] = layer
        results["ic_prompt_condition"].append(res)

        if layer % 10 == 0 or layer == 22:
            logger.info(f"  L{layer:02d}: AUC={res['auc']:.4f}±{res['auc_std']:.4f} (4-class)")

    # --- 4c: bet_type (fixed vs variable) for all paradigms ---
    logger.info("\n--- 4c: Bet Type (Fixed vs Variable) ---")
    results["bet_type"] = {}

    for paradigm in ["ic", "sm", "mw"]:
        logger.info(f"\n  {PARADIGM_LABELS[paradigm]}:")
        results["bet_type"][paradigm] = []

        for layer in key_layers:
            loaded = load_layer_features(paradigm, layer, mode="decision_point", dense=True)
            if loaded is None:
                continue

            feat, meta = loaded
            bt = meta["bet_types"]
            labels = (bt == "variable").astype(np.int32)

            n_var = int(labels.sum())
            n_fix = len(labels) - n_var
            if n_var < 5 or n_fix < 5:
                continue

            rate = (feat != 0).mean(axis=0)
            active = rate >= MIN_ACTIVATION_RATE
            if active.sum() < 10:
                continue

            res = classify_cv(feat[:, active], labels)
            res["layer"] = layer
            results["bet_type"][paradigm].append(res)

            if layer % 10 == 0 or layer == 22:
                logger.info(f"    L{layer:02d}: AUC={res['auc']:.4f} "
                            f"(var={n_var}, fix={n_fix})")

    # --- 4d: SM/MW prompt component effect (binary: has G vs not) ---
    logger.info("\n--- 4d: Prompt Component Effects (SM) ---")
    results["prompt_components"] = {}

    for paradigm in ["sm", "mw"]:
        results["prompt_components"][paradigm] = {}
        logger.info(f"\n  {PARADIGM_LABELS[paradigm]}:")

        layer = 22
        loaded = load_layer_features(paradigm, layer, mode="decision_point", dense=True)
        if loaded is None:
            continue

        feat, meta = loaded
        prompts = meta["prompt_conditions"]

        rate = (feat != 0).mean(axis=0)
        active = rate >= MIN_ACTIVATION_RATE
        if active.sum() < 10:
            continue

        # Test each component
        for component in ["G", "M", "R", "W", "P"]:
            has_comp = np.array([component in str(p) for p in prompts]).astype(np.int32)
            n_has = int(has_comp.sum())
            n_not = len(has_comp) - n_has
            if n_has < 10 or n_not < 10:
                continue

            res = classify_cv(feat[:, active], has_comp)
            res["component"] = component
            results["prompt_components"][paradigm][component] = res
            logger.info(f"    {component}: AUC={res['auc']:.4f} (has={n_has}, not={n_not})")

    # Summary
    for key in ["ic_bet_constraint", "ic_prompt_condition"]:
        data = results.get(key, [])
        if data:
            best = max(data, key=lambda r: r["auc"])
            logger.info(f"  BEST {key}: L{best['layer']} AUC={best['auc']:.4f}")

    for paradigm in ["ic", "sm", "mw"]:
        data = results.get("bet_type", {}).get(paradigm, [])
        if data:
            best = max(data, key=lambda r: r["auc"])
            logger.info(f"  BEST bet_type/{paradigm}: L{best['layer']} AUC={best['auc']:.4f}")

    return results


# ===================================================================
# Visualization
# ===================================================================

def plot_exp2a(results, save_path):
    """R1 vs Decision-Point vs Balance-Matched AUC comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for col, paradigm in enumerate(["ic", "sm", "mw"]):
        ax = axes[col]
        data = results.get(paradigm, {})

        for analysis_type, color, marker, label in [
            ("decision_point_original", "#e74c3c", "o", "Decision-Point (original)"),
            ("r1", "#2ecc71", "s", "Round 1 Only (no balance confound)"),
            ("balanced_matched", "#3498db", "^", "Balance-Matched"),
        ]:
            entries = data.get(analysis_type, [])
            if entries:
                layers = [r["layer"] for r in entries if not r.get("skipped")]
                aucs = [r["auc"] for r in entries if not r.get("skipped")]
                ax.plot(layers, aucs, f"-{marker}", color=color, markersize=3, lw=1.5, label=label)

        ax.axhline(y=0.5, color="gray", ls="--", alpha=0.5)
        ax.set_xlabel("Layer")
        ax.set_ylabel("AUC (5-fold CV)")
        ax.set_title(f"{PARADIGM_LABELS[paradigm]}", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim(0.4, 1.05)

    fig.suptitle("Exp 2a: Balance-Controlled BK Classification", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_exp3(results, save_path):
    """Round-level risk choice classification curves."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    plot_data = [
        ("ic_risk_choice", "IC: Risky vs Safe Choice", "#2ecc71"),
        ("sm_bet_magnitude", "SM: High vs Low Bet", "#e74c3c"),
        ("mw_bet_magnitude", "MW: High vs Low Bet", "#3498db"),
    ]

    for col, (key, title, color) in enumerate(plot_data):
        ax = axes[col]
        data = results.get(key, [])
        if data:
            layers = [r["layer"] for r in data if not r.get("skipped")]
            aucs = [r["auc"] for r in data if not r.get("skipped")]
            ax.plot(layers, aucs, "-o", color=color, markersize=3, lw=1.5)
            if data:
                best = max(data, key=lambda r: r["auc"])
                ax.plot(best["layer"], best["auc"], "*", color=color, markersize=12, zorder=5)

        ax.axhline(y=0.5, color="gray", ls="--", alpha=0.5)
        ax.set_xlabel("Layer")
        ax.set_ylabel("AUC")
        ax.set_title(title, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.set_ylim(0.4, 1.05)

    fig.suptitle("Exp 3: Round-Level Risk Choice Classification", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_exp4(results, save_path):
    """Condition-level classification results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 4a: IC bet constraint
    ax = axes[0, 0]
    data = results.get("ic_bet_constraint", [])
    if data:
        layers = [r["layer"] for r in data]
        aucs = [r["auc"] for r in data]
        ax.plot(layers, aucs, "-o", color="#2ecc71", markersize=3, lw=1.5)
    ax.axhline(y=0.5, color="gray", ls="--", alpha=0.5, label="Chance")
    ax.set_title("IC: Bet Constraint (c10/c30/c50/c70)", fontweight="bold")
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUC (macro, OVR)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 4b: IC prompt condition
    ax = axes[0, 1]
    data = results.get("ic_prompt_condition", [])
    if data:
        layers = [r["layer"] for r in data]
        aucs = [r["auc"] for r in data]
        ax.plot(layers, aucs, "-o", color="#9b59b6", markersize=3, lw=1.5)
    ax.axhline(y=0.5, color="gray", ls="--", alpha=0.5, label="Chance")
    ax.set_title("IC: Prompt Condition (BASE/G/GM/M)", fontweight="bold")
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUC (macro, OVR)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 4c: bet type
    ax = axes[1, 0]
    for paradigm, color in PARADIGM_COLORS.items():
        data = results.get("bet_type", {}).get(paradigm, [])
        if data:
            layers = [r["layer"] for r in data if not r.get("skipped")]
            aucs = [r["auc"] for r in data if not r.get("skipped")]
            ax.plot(layers, aucs, "-o", color=color, markersize=3, lw=1.5,
                    label=PARADIGM_LABELS[paradigm])
    ax.axhline(y=0.5, color="gray", ls="--", alpha=0.5)
    ax.set_title("Bet Type (Fixed vs Variable)", fontweight="bold")
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUC")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 4d: prompt components
    ax = axes[1, 1]
    comp_colors = {"G": "#e74c3c", "M": "#3498db", "R": "#2ecc71", "W": "#f39c12", "P": "#9b59b6"}
    for paradigm in ["sm"]:
        comp_data = results.get("prompt_components", {}).get(paradigm, {})
        if comp_data:
            components = sorted(comp_data.keys())
            aucs = [comp_data[c]["auc"] for c in components]
            bars = ax.bar(components, aucs, color=[comp_colors.get(c, "gray") for c in components])
            for bar, auc_val in zip(bars, aucs):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{auc_val:.3f}", ha="center", fontsize=9)
    ax.axhline(y=0.5, color="gray", ls="--", alpha=0.5)
    ax.set_title("SM: Prompt Component Classification (L22)", fontweight="bold")
    ax.set_ylabel("AUC")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Exp 4: Condition-Level Activation Analysis", fontsize=14, fontweight="bold")
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
    log_file = LOG_DIR / f"extended_analyses_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Extended analyses started. Log: {log_file}")
    np.random.seed(RANDOM_SEED)

    # Run all experiments
    exp2a = run_exp2a(logger)
    exp2b = run_exp2b(logger)
    exp3 = run_exp3(logger)
    exp4 = run_exp4(logger)

    # Save results
    all_results = {
        "exp2a_balance_controlled": exp2a,
        "exp2b_feature_importance": exp2b,
        "exp3_round_level_risk": exp3,
        "exp4_condition_level": exp4,
        "timestamp": ts,
    }
    result_path = JSON_DIR / f"extended_analyses_{ts}.json"
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    logger.info(f"\nResults saved: {result_path}")

    # Generate figures
    logger.info("\nGenerating figures...")
    plot_exp2a(exp2a, FIGURE_DIR / "exp2a_balance_controlled.png")
    plot_exp3(exp3, FIGURE_DIR / "exp3_round_level_risk.png")
    plot_exp4(exp4, FIGURE_DIR / "exp4_condition_level.png")

    # Final summary
    logger.info(f"\n{'='*70}")
    logger.info("EXTENDED ANALYSES COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Figures: {FIGURE_DIR}")
    logger.info(f"JSON:    {result_path}")


if __name__ == "__main__":
    main()
