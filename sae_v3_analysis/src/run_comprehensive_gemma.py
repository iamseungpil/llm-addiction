#!/usr/bin/env python3
"""
Comprehensive Gemma analysis — fills ALL missing cells in the analysis matrix.

Matrix: 2 (SAE/Hidden) × 2 (BK/Risk) × conditions × rounds × layers

New analyses (not in run_all_analyses.py or run_extended_analyses.py):
  1. Hidden BK classification — ALL 42 layers, DP + R1
  2. Hidden Risk choice classification — ALL 42 layers, per-round
  3. Per-condition SAE BK — within fixed/variable/G/H/M/W subsets
  4. Per-condition SAE Risk — within fixed/variable/G/H/M/W subsets
  5. Per-condition Hidden BK — within fixed/variable subsets
  6. Behavioral metrics (I_BA, I_LC, I_EC) — from game JSON

Usage:
    cd /home/jovyan/llm-addiction/sae_v3_analysis/src
    python run_comprehensive_gemma.py [--analyses 1,2,3,4,5,6] [--layers all|best6]
"""

import gc
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import argparse

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
    MIN_ACTIVATION_RATE, RANDOM_SEED, DATA_ROOT,
    PARADIGM_COLORS, PARADIGM_LABELS,
)
from data_loader import (
    load_layer_features, load_hidden_states,
    filter_active_features, get_labels,
)


# ===================================================================
# Config
# ===================================================================

# Game JSON paths (matching run_extended_analyses.py)
IC_JSON_FILES = [
    (DATA_ROOT / "investment_choice_v2_role" / "gemma_investment_c10_20260225_122319.json", 0),
    (DATA_ROOT / "investment_choice_v2_role" / "gemma_investment_c30_20260225_184458.json", 400),
    (DATA_ROOT / "investment_choice_v2_role" / "gemma_investment_c50_20260226_020821.json", 800),
    (DATA_ROOT / "investment_choice_v2_role" / "gemma_investment_c70_20260226_082029.json", 1200),
]

SM_JSON_FILE = DATA_ROOT / "slot_machine" / "experiment_0_gemma_v4_role" / "final_gemma_20260227_002507.json"
MW_JSON_FILE = DATA_ROOT / "mystery_wheel_v2_role" / "gemma_mysterywheel_checkpoint_3200.json"

# Best layers per paradigm (from prior analyses)
BEST_LAYERS = {
    "ic": [0, 8, 18, 22, 26, 40],
    "sm": [0, 6, 12, 16, 22, 40],
    "mw": [0, 10, 22, 24, 33, 40],
}

# Prompt components per paradigm
PROMPT_COMPONENTS = {
    "ic": ["G", "M"],  # IC has BASE, G, GM, M
    "sm": ["G", "M", "H", "W", "P"],
    "mw": ["G", "M", "H", "W", "P"],
}

# Data stores 'R' for Hidden Patterns but we display as 'H'
COMPONENT_DATA_MAP = {
    "H": "R",  # H (Hidden patterns) stored as 'R' in NPZ prompt_conditions
}


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


# ===================================================================
# Choice loaders (from game JSON)
# ===================================================================

def find_ic_json_files():
    """Find IC JSON files dynamically. Only match c10/c30/c50/c70 final files."""
    ic_dir = DATA_ROOT / "investment_choice_v2_role"
    # Exclude checkpoint files — only match cNN_ pattern (c10, c30, c50, c70)
    files = sorted(f for f in ic_dir.glob("gemma_investment_c[0-9][0-9]_*.json")
                   if "checkpoint" not in f.name)
    result = []
    offset = 0
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        n = len(data["results"])
        result.append((f, offset))
        offset += n
    return result


def load_ic_choices():
    """Load IC per-round choices. Returns {npz_game_id: {round_num: choice(1-4)}}."""
    choices = {}
    for json_path, offset in find_ic_json_files():
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


def get_risk_labels(paradigm, game_ids, round_nums):
    """Get binary risk labels (1=risky, 0=safe) for each round.

    IC: choice 3,4 = risky; choice 1,2 = safe
    SM: action='bet' = risky; action='stop' = safe
    MW: choice==2 (spin) = risky; choice==1 (stop) = safe
    """
    if paradigm == "ic":
        choices = load_ic_choices()
        labels = np.full(len(game_ids), -1, dtype=np.int32)
        for i, (gid, rnd) in enumerate(zip(game_ids, round_nums)):
            ch = choices.get(int(gid), {}).get(int(rnd))
            if ch is not None:
                labels[i] = 1 if ch >= 3 else 0
        valid = labels >= 0
        return labels, valid

    elif paradigm == "sm":
        choices = load_sm_choices()
        labels = np.full(len(game_ids), -1, dtype=np.int32)
        for i, (gid, rnd) in enumerate(zip(game_ids, round_nums)):
            info = choices.get(int(gid), {}).get(int(rnd))
            if info is not None:
                labels[i] = 1 if info["action"] == "bet" else 0
        valid = labels >= 0
        return labels, valid

    elif paradigm == "mw":
        choices = load_mw_choices()
        labels = np.full(len(game_ids), -1, dtype=np.int32)
        for i, (gid, rnd) in enumerate(zip(game_ids, round_nums)):
            info = choices.get(int(gid), {}).get(int(rnd))
            if info is not None:
                # MW choice: 1=stop(safe), 2=spin(risky). Integer, not string.
                labels[i] = 1 if info["choice"] == 2 else 0
        valid = labels >= 0
        return labels, valid

    raise ValueError(f"Unknown paradigm: {paradigm}")


# ===================================================================
# Analysis 1: Hidden BK Classification (DP + R1, all layers)
# ===================================================================

def run_hidden_bk(logger, layers):
    """Hidden state BK classification at decision-point and R1 for all layers."""
    logger.info("=" * 60)
    logger.info("Analysis 1: Hidden State BK Classification (DP + R1)")
    logger.info("=" * 60)

    results = {}
    for paradigm in PARADIGMS:
        results[paradigm] = {"dp": [], "r1": []}
        logger.info(f"\n--- {PARADIGM_LABELS[paradigm]} ---")

        for layer in layers:
            # Decision-point
            dp_loaded = load_hidden_states(paradigm, layer, mode="decision_point")
            if dp_loaded is None:
                logger.warning(f"  L{layer}: no hidden states")
                continue

            features, meta = dp_loaded
            labels = get_labels(meta)
            res = classify_cv(features, labels)
            res["layer"] = layer
            results[paradigm]["dp"].append(res)
            logger.info(f"  L{layer} DP: AUC={res['auc']:.3f} (n+={res['n_pos']}, n-={res['n_neg']})")

            # R1: filter to round 1
            r1_loaded = load_hidden_states(paradigm, layer, mode="all_rounds")
            if r1_loaded is not None:
                r1_feat, r1_meta = r1_loaded
                r1_mask = r1_meta["round_nums"] == 1
                if r1_mask.sum() > 0:
                    r1_labels = get_labels({k: v[r1_mask] for k, v in r1_meta.items()})
                    r1_res = classify_cv(r1_feat[r1_mask], r1_labels)
                    r1_res["layer"] = layer
                    results[paradigm]["r1"].append(r1_res)
                    logger.info(f"  L{layer} R1: AUC={r1_res['auc']:.3f}")

            gc.collect()

        # Summary
        if results[paradigm]["dp"]:
            best_dp = max(results[paradigm]["dp"], key=lambda r: r["auc"])
            logger.info(f"  Best DP: L{best_dp['layer']} AUC={best_dp['auc']:.3f}")
        if results[paradigm]["r1"]:
            best_r1 = max(results[paradigm]["r1"], key=lambda r: r["auc"])
            logger.info(f"  Best R1: L{best_r1['layer']} AUC={best_r1['auc']:.3f}")

    return results


# ===================================================================
# Analysis 2: Hidden Risk Choice Classification (all rounds, all layers)
# ===================================================================

def run_hidden_risk(logger, layers):
    """Hidden state risk choice classification per-round."""
    logger.info("=" * 60)
    logger.info("Analysis 2: Hidden State Risk Choice Classification")
    logger.info("=" * 60)

    results = {}
    for paradigm in PARADIGMS:
        results[paradigm] = []
        logger.info(f"\n--- {PARADIGM_LABELS[paradigm]} ---")

        # Load choices once
        risk_labels_cache = None

        for layer in layers:
            loaded = load_hidden_states(paradigm, layer, mode="all_rounds")
            if loaded is None:
                continue

            features, meta = loaded

            # Get risk labels (load choices on first iteration)
            if risk_labels_cache is None:
                risk_labels_cache, valid = get_risk_labels(
                    paradigm, meta["game_ids"], meta["round_nums"]
                )
            else:
                valid = risk_labels_cache >= 0

            if valid.sum() < 20:
                logger.warning(f"  L{layer}: too few valid rounds ({valid.sum()})")
                continue

            feat_valid = features[valid]
            lab_valid = risk_labels_cache[valid]
            res = classify_cv(feat_valid, lab_valid)
            res["layer"] = layer
            results[paradigm].append(res)
            logger.info(f"  L{layer}: AUC={res['auc']:.3f} (risky={res['n_pos']}, safe={res['n_neg']})")

            gc.collect()

        if results[paradigm]:
            best = max(results[paradigm], key=lambda r: r["auc"])
            logger.info(f"  Best: L{best['layer']} AUC={best['auc']:.3f}")

    return results


# ===================================================================
# Analysis 3: Per-condition SAE BK classification
# ===================================================================

def run_percondition_sae_bk(logger, layers):
    """SAE BK classification within condition subsets."""
    logger.info("=" * 60)
    logger.info("Analysis 3: Per-Condition SAE BK Classification (DP + R1)")
    logger.info("=" * 60)

    results = {}
    for paradigm in PARADIGMS:
        results[paradigm] = {}
        logger.info(f"\n--- {PARADIGM_LABELS[paradigm]} ---")

        for layer in layers:
            loaded = load_layer_features(paradigm, layer, mode="decision_point", dense=True)
            if loaded is None:
                continue

            features, meta = loaded
            active_feat, active_idx = filter_active_features(features, MIN_ACTIVATION_RATE)
            labels = get_labels(meta)

            # Also load R1
            r1_loaded = load_layer_features(paradigm, layer, mode="all_rounds", dense=True)
            r1_features, r1_meta = None, None
            if r1_loaded is not None:
                r1_feat_full, r1_meta = r1_loaded
                r1_mask = r1_meta["round_nums"] == 1
                r1_features = r1_feat_full[r1_mask][:, active_idx]
                r1_meta = {k: v[r1_mask] for k, v in r1_meta.items()}
                r1_labels = get_labels(r1_meta)
                del r1_feat_full
                gc.collect()

            # --- Condition: bet_type ---
            for bt in ["fixed", "variable"]:
                bt_mask = meta["bet_types"] == bt
                n_bk = int((labels[bt_mask] == 1).sum()) if bt_mask.sum() > 0 else 0
                cond_key = f"L{layer}_bt_{bt}"

                if bt_mask.sum() >= 10 and n_bk >= 2:
                    res = classify_cv(active_feat[bt_mask], labels[bt_mask])
                    res["layer"] = layer
                    res["condition"] = f"bet_type={bt}"
                    results[paradigm][cond_key] = res
                    logger.info(f"  L{layer} {bt}: AUC={res['auc']:.3f} (BK={n_bk})")

                    # R1 within condition
                    if r1_meta is not None:
                        r1_bt_mask = r1_meta["bet_types"] == bt
                        r1_n_bk = int((r1_labels[r1_bt_mask] == 1).sum()) if r1_bt_mask.sum() > 0 else 0
                        if r1_bt_mask.sum() >= 10 and r1_n_bk >= 2:
                            r1_res = classify_cv(r1_features[r1_bt_mask], r1_labels[r1_bt_mask])
                            r1_res["layer"] = layer
                            r1_res["condition"] = f"bet_type={bt}_R1"
                            results[paradigm][f"{cond_key}_R1"] = r1_res
                else:
                    logger.info(f"  L{layer} {bt}: SKIP (n={bt_mask.sum()}, BK={n_bk})")

            # --- Condition: prompt components ---
            if "prompt_conditions" in meta:
                for comp in PROMPT_COMPONENTS.get(paradigm, []):
                    # Marginal: has component vs doesn't
                    data_comp = COMPONENT_DATA_MAP.get(comp, comp)
                    has_comp = np.array([data_comp in str(pc) for pc in meta["prompt_conditions"]])
                    for subset_name, subset_mask in [("with_" + comp, has_comp), ("without_" + comp, ~has_comp)]:
                        n_bk = int((labels[subset_mask] == 1).sum()) if subset_mask.sum() > 0 else 0
                        cond_key = f"L{layer}_{subset_name}"

                        if subset_mask.sum() >= 10 and n_bk >= 2:
                            res = classify_cv(active_feat[subset_mask], labels[subset_mask])
                            res["layer"] = layer
                            res["condition"] = subset_name
                            results[paradigm][cond_key] = res
                            logger.info(f"  L{layer} {subset_name}: AUC={res['auc']:.3f} (BK={n_bk})")

            # --- Condition: bet_constraint (IC only) ---
            if "bet_constraints" in meta and paradigm == "ic":
                for bc in ["30", "50", "70"]:  # skip 10 (0 BK); stored without 'c' prefix
                    bc_mask = meta["bet_constraints"] == bc
                    n_bk = int((labels[bc_mask] == 1).sum()) if bc_mask.sum() > 0 else 0
                    cond_key = f"L{layer}_bc_{bc}"

                    if bc_mask.sum() >= 10 and n_bk >= 2:
                        res = classify_cv(active_feat[bc_mask], labels[bc_mask])
                        res["layer"] = layer
                        res["condition"] = f"constraint={bc}"
                        results[paradigm][cond_key] = res
                        logger.info(f"  L{layer} {bc}: AUC={res['auc']:.3f} (BK={n_bk})")

            del features, active_feat
            gc.collect()

    return results


# ===================================================================
# Analysis 4: Per-condition SAE Risk Choice
# ===================================================================

def run_percondition_sae_risk(logger, layers):
    """SAE risk choice classification within condition subsets."""
    logger.info("=" * 60)
    logger.info("Analysis 4: Per-Condition SAE Risk Choice Classification")
    logger.info("=" * 60)

    results = {}
    for paradigm in PARADIGMS:
        results[paradigm] = {}
        logger.info(f"\n--- {PARADIGM_LABELS[paradigm]} ---")

        for layer in layers:
            loaded = load_layer_features(paradigm, layer, mode="all_rounds", dense=True)
            if loaded is None:
                continue

            features, meta = loaded
            active_feat, active_idx = filter_active_features(features, MIN_ACTIVATION_RATE)

            risk_labels, valid = get_risk_labels(paradigm, meta["game_ids"], meta["round_nums"])

            # --- Overall (verify matches prior Exp 3) ---
            if valid.sum() >= 20:
                res = classify_cv(active_feat[valid], risk_labels[valid])
                res["layer"] = layer
                res["condition"] = "overall"
                results[paradigm][f"L{layer}_overall"] = res
                logger.info(f"  L{layer} overall: AUC={res['auc']:.3f} (risky={res['n_pos']}, safe={res['n_neg']})")

            # --- Per bet_type ---
            for bt in ["fixed", "variable"]:
                bt_mask = (meta["bet_types"] == bt) & valid
                if bt_mask.sum() >= 20:
                    res = classify_cv(active_feat[bt_mask], risk_labels[bt_mask])
                    res["layer"] = layer
                    res["condition"] = f"bet_type={bt}"
                    results[paradigm][f"L{layer}_bt_{bt}"] = res
                    logger.info(f"  L{layer} risk {bt}: AUC={res['auc']:.3f} (risky={res['n_pos']})")

            # --- Per prompt component ---
            if "prompt_conditions" in meta:
                for comp in PROMPT_COMPONENTS.get(paradigm, []):
                    data_comp = COMPONENT_DATA_MAP.get(comp, comp)
                    has_comp = np.array([data_comp in str(pc) for pc in meta["prompt_conditions"]])
                    comp_mask = has_comp & valid
                    if comp_mask.sum() >= 20:
                        res = classify_cv(active_feat[comp_mask], risk_labels[comp_mask])
                        res["layer"] = layer
                        res["condition"] = f"with_{comp}"
                        results[paradigm][f"L{layer}_with_{comp}"] = res
                        logger.info(f"  L{layer} risk with_{comp}: AUC={res['auc']:.3f}")

            del features, active_feat
            gc.collect()

    return results


# ===================================================================
# Analysis 5: Per-condition Hidden BK
# ===================================================================

def run_percondition_hidden_bk(logger, layers):
    """Hidden state BK classification within condition subsets."""
    logger.info("=" * 60)
    logger.info("Analysis 5: Per-Condition Hidden State BK Classification")
    logger.info("=" * 60)

    results = {}
    for paradigm in PARADIGMS:
        results[paradigm] = {}
        logger.info(f"\n--- {PARADIGM_LABELS[paradigm]} ---")

        for layer in layers:
            loaded = load_hidden_states(paradigm, layer, mode="decision_point")
            if loaded is None:
                continue

            features, meta = loaded
            labels = get_labels(meta)

            # --- bet_type ---
            for bt in ["fixed", "variable"]:
                bt_mask = meta["bet_types"] == bt
                n_bk = int((labels[bt_mask] == 1).sum()) if bt_mask.sum() > 0 else 0
                cond_key = f"L{layer}_bt_{bt}"

                if bt_mask.sum() >= 10 and n_bk >= 2:
                    res = classify_cv(features[bt_mask], labels[bt_mask])
                    res["layer"] = layer
                    res["condition"] = f"bet_type={bt}"
                    results[paradigm][cond_key] = res
                    logger.info(f"  L{layer} hidden {bt}: AUC={res['auc']:.3f} (BK={n_bk})")

            # --- prompt components ---
            if "prompt_conditions" in meta:
                for comp in PROMPT_COMPONENTS.get(paradigm, []):
                    data_comp = COMPONENT_DATA_MAP.get(comp, comp)
                    has_comp = np.array([data_comp in str(pc) for pc in meta["prompt_conditions"]])
                    for subset_name, subset_mask in [("with_" + comp, has_comp), ("without_" + comp, ~has_comp)]:
                        n_bk = int((labels[subset_mask] == 1).sum()) if subset_mask.sum() > 0 else 0
                        cond_key = f"L{layer}_hidden_{subset_name}"
                        if subset_mask.sum() >= 10 and n_bk >= 2:
                            res = classify_cv(features[subset_mask], labels[subset_mask])
                            res["layer"] = layer
                            res["condition"] = subset_name
                            results[paradigm][cond_key] = res
                            logger.info(f"  L{layer} hidden {subset_name}: AUC={res['auc']:.3f} (BK={n_bk})")

            # --- bet_constraint (IC only) ---
            if "bet_constraints" in meta and paradigm == "ic":
                for bc in ["30", "50", "70"]:  # stored without 'c' prefix
                    bc_mask = meta["bet_constraints"] == bc
                    n_bk = int((labels[bc_mask] == 1).sum()) if bc_mask.sum() > 0 else 0
                    cond_key = f"L{layer}_hidden_bc_{bc}"
                    if bc_mask.sum() >= 10 and n_bk >= 2:
                        res = classify_cv(features[bc_mask], labels[bc_mask])
                        res["layer"] = layer
                        res["condition"] = f"constraint={bc}"
                        results[paradigm][cond_key] = res
                        logger.info(f"  L{layer} hidden {bc}: AUC={res['auc']:.3f} (BK={n_bk})")

            gc.collect()

    return results


# ===================================================================
# Analysis 6: Behavioral Metrics (I_BA, I_LC, I_EC)
# ===================================================================

def compute_behavioral_metrics(logger):
    """Compute per-round behavioral metrics from game JSON."""
    logger.info("=" * 60)
    logger.info("Analysis 6: Behavioral Metrics (I_BA, I_LC, I_EC)")
    logger.info("=" * 60)

    results = {}

    # --- IC ---
    logger.info("\n--- Investment Choice ---")
    ic_metrics = {"per_game": [], "per_condition": defaultdict(list)}

    for json_path, offset in find_ic_json_files():
        with open(json_path) as f:
            data = json.load(f)

        for game in data["results"]:
            decisions = game.get("decisions", [])
            if len(decisions) < 2:
                continue

            game_metrics = {
                "game_id": offset + data["results"].index(game) + 1,
                "bankruptcy": game.get("bankruptcy", False),
                "bet_type": game.get("bet_type", ""),
                "bet_constraint": game.get("bet_constraint", ""),
                "prompt_condition": game.get("prompt_condition", ""),
                "rounds": len(decisions),
            }

            # I_BA: bet_amount / balance_before per round
            i_ba_values = []
            for dec in decisions:
                bal = dec.get("balance_before", 100) or 100
                bet = dec.get("bet_amount", 0) or 0
                if bal > 0 and bet > 0:
                    i_ba_values.append(bet / bal)

            # I_LC: bet increase after loss
            loss_chase_count = 0
            total_post_loss = 0
            for i in range(1, len(decisions)):
                prev = decisions[i - 1]
                curr = decisions[i]
                # outcome can be dict or string
                prev_outcome = prev.get("outcome", {})
                if isinstance(prev_outcome, dict):
                    is_loss = not prev_outcome.get("win", True)
                else:
                    is_loss = "loss" in str(prev_outcome)
                if is_loss:
                    total_post_loss += 1
                    prev_bet = prev.get("bet_amount", 0) or 0
                    curr_bet = curr.get("bet_amount", 0) or 0
                    if curr_bet > prev_bet:
                        loss_chase_count += 1

            # I_EC: extreme choice (choice 4 = VeryHigh, or bet > 50% balance)
            extreme_count = 0
            for dec in decisions:
                if dec.get("choice", 0) == 4:
                    extreme_count += 1
                else:
                    bet_amt = dec.get("bet_amount", 0) or 0
                    bal_bef = dec.get("balance_before", 100) or 100
                    if bet_amt > 0.5 * bal_bef:
                        extreme_count += 1

            game_metrics["i_ba_mean"] = float(np.mean(i_ba_values)) if i_ba_values else 0.0
            game_metrics["i_ba_max"] = float(np.max(i_ba_values)) if i_ba_values else 0.0
            game_metrics["i_lc"] = loss_chase_count / max(1, total_post_loss)
            game_metrics["i_ec"] = extreme_count / max(1, len(decisions))
            game_metrics["loss_chase_count"] = loss_chase_count
            game_metrics["total_post_loss"] = total_post_loss

            ic_metrics["per_game"].append(game_metrics)

            # Aggregate by condition
            cond = f"{game_metrics['bet_constraint']}_{game_metrics['bet_type']}"
            ic_metrics["per_condition"][cond].append(game_metrics)

    # Summary
    bk_games = [g for g in ic_metrics["per_game"] if g["bankruptcy"]]
    safe_games = [g for g in ic_metrics["per_game"] if not g["bankruptcy"]]

    if bk_games:
        logger.info(f"  BK games ({len(bk_games)}):")
        logger.info(f"    I_BA mean: {np.mean([g['i_ba_mean'] for g in bk_games]):.3f}")
        logger.info(f"    I_LC: {np.mean([g['i_lc'] for g in bk_games]):.3f}")
        logger.info(f"    I_EC: {np.mean([g['i_ec'] for g in bk_games]):.3f}")
    if safe_games:
        logger.info(f"  Safe games ({len(safe_games)}):")
        logger.info(f"    I_BA mean: {np.mean([g['i_ba_mean'] for g in safe_games]):.3f}")
        logger.info(f"    I_LC: {np.mean([g['i_lc'] for g in safe_games]):.3f}")
        logger.info(f"    I_EC: {np.mean([g['i_ec'] for g in safe_games]):.3f}")

    results["ic"] = ic_metrics

    # --- SM ---
    logger.info("\n--- Slot Machine ---")
    sm_metrics = {"per_game": []}

    with open(SM_JSON_FILE) as f:
        sm_data = json.load(f)

    for i, game in enumerate(sm_data["results"]):
        decisions = game.get("decisions", [])
        if len(decisions) < 2:
            continue

        game_metrics = {
            "game_id": i + 1,
            # SM uses outcome="bankruptcy" not bankruptcy=True
            "bankruptcy": game.get("bankruptcy", False) or game.get("outcome") == "bankruptcy",
            "bet_type": game.get("bet_type", ""),
            "rounds": len(decisions),
        }

        # I_BA
        i_ba_values = []
        for dec in decisions:
            bal = dec.get("balance_before", dec.get("balance", 100))
            bet = dec.get("bet", dec.get("parsed_bet", 0))
            if bal > 0 and bet > 0:
                i_ba_values.append(bet / bal)

        # I_LC
        loss_chase_count = 0
        total_post_loss = 0
        for j in range(1, len(decisions)):
            prev = decisions[j - 1]
            curr = decisions[j]
            prev_result = prev.get("result", "")
            if prev_result == "L" or prev_result == "loss":
                total_post_loss += 1
                prev_bet = prev.get("bet", prev.get("parsed_bet", 0))
                curr_bet = curr.get("bet", curr.get("parsed_bet", 0))
                if curr_bet > prev_bet:
                    loss_chase_count += 1

        game_metrics["i_ba_mean"] = float(np.mean(i_ba_values)) if i_ba_values else 0.0
        game_metrics["i_lc"] = loss_chase_count / max(1, total_post_loss)
        game_metrics["loss_chase_count"] = loss_chase_count

        sm_metrics["per_game"].append(game_metrics)

    bk_sm = [g for g in sm_metrics["per_game"] if g["bankruptcy"]]
    safe_sm = [g for g in sm_metrics["per_game"] if not g["bankruptcy"]]

    if bk_sm:
        logger.info(f"  BK games ({len(bk_sm)}):")
        logger.info(f"    I_BA mean: {np.mean([g['i_ba_mean'] for g in bk_sm]):.3f}")
        logger.info(f"    I_LC: {np.mean([g['i_lc'] for g in bk_sm]):.3f}")
    if safe_sm:
        logger.info(f"  Safe games ({len(safe_sm)}):")
        logger.info(f"    I_BA mean: {np.mean([g['i_ba_mean'] for g in safe_sm]):.3f}")
        logger.info(f"    I_LC: {np.mean([g['i_lc'] for g in safe_sm]):.3f}")

    results["sm"] = sm_metrics

    # --- MW ---
    logger.info("\n--- Mystery Wheel ---")
    mw_metrics = {"per_game": []}

    with open(MW_JSON_FILE) as f:
        mw_data = json.load(f)

    for i, game in enumerate(mw_data["results"]):
        decisions = game.get("decisions", [])
        if len(decisions) < 2:
            continue

        game_metrics = {
            "game_id": i + 1,
            "bankruptcy": game.get("bankruptcy", False),
            "bet_type": game.get("bet_type", ""),
            "rounds": len(decisions),
        }

        i_ba_values = []
        for dec in decisions:
            bal = dec.get("balance_before", 100) or 100
            bet = dec.get("bet_amount", 0) or 0
            if bal > 0 and bet > 0:
                i_ba_values.append(bet / bal)

        game_metrics["i_ba_mean"] = float(np.mean(i_ba_values)) if i_ba_values else 0.0
        mw_metrics["per_game"].append(game_metrics)

    bk_mw = [g for g in mw_metrics["per_game"] if g["bankruptcy"]]
    safe_mw = [g for g in mw_metrics["per_game"] if not g["bankruptcy"]]

    if bk_mw:
        logger.info(f"  BK games ({len(bk_mw)}): I_BA mean={np.mean([g['i_ba_mean'] for g in bk_mw]):.3f}")
    if safe_mw:
        logger.info(f"  Safe games ({len(safe_mw)}): I_BA mean={np.mean([g['i_ba_mean'] for g in safe_mw]):.3f}")

    results["mw"] = mw_metrics

    return results


# ===================================================================
# Visualization
# ===================================================================

def plot_hidden_vs_sae(hidden_results, existing_sae_json, output_path):
    """Plot hidden state vs SAE AUC curves for BK classification."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Load existing SAE results if available
    sae_data = {}
    if existing_sae_json and Path(existing_sae_json).exists():
        with open(existing_sae_json) as f:
            all_data = json.load(f)
        if "goal_a" in all_data:
            for p in ["ic", "sm", "mw"]:
                if p in all_data["goal_a"]:
                    sae_data[p] = all_data["goal_a"][p].get("sae", [])

    for ax, paradigm in zip(axes, ["ic", "sm", "mw"]):
        color = PARADIGM_COLORS[paradigm]

        # Hidden DP
        dp = hidden_results.get(paradigm, {}).get("dp", [])
        if dp:
            layers = [r["layer"] for r in dp]
            aucs = [r["auc"] for r in dp]
            ax.plot(layers, aucs, 'o-', color=color, label="Hidden DP", linewidth=2)

        # Hidden R1
        r1 = hidden_results.get(paradigm, {}).get("r1", [])
        if r1:
            layers = [r["layer"] for r in r1]
            aucs = [r["auc"] for r in r1]
            ax.plot(layers, aucs, 's--', color=color, alpha=0.6, label="Hidden R1")

        # SAE DP (from existing)
        if paradigm in sae_data:
            sae_layers = [r["layer"] for r in sae_data[paradigm]]
            sae_aucs = [r["auc"] for r in sae_data[paradigm]]
            ax.plot(sae_layers, sae_aucs, '^-', color='gray', alpha=0.5, label="SAE DP")

        ax.set_title(PARADIGM_LABELS[paradigm])
        ax.set_xlabel("Layer")
        ax.set_ylabel("AUC")
        ax.set_ylim(0.4, 1.05)
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Hidden State vs SAE: BK Classification", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def plot_comprehensive_matrix(all_results, output_path):
    """Plot the full analysis matrix as a heatmap."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    paradigms = ["ic", "sm", "mw"]
    feature_types = ["SAE", "Hidden"]

    for col, paradigm in enumerate(paradigms):
        for row, ftype in enumerate(feature_types):
            ax = axes[row, col]

            # Collect data for this cell
            data_points = []

            if ftype == "SAE":
                # From per-condition results
                cond_res = all_results.get("percondition_sae_bk", {}).get(paradigm, {})
            else:
                cond_res = all_results.get("percondition_hidden_bk", {}).get(paradigm, {})

            conditions = set()
            for key in cond_res:
                parts = key.split("_", 1)
                if len(parts) > 1:
                    conditions.add(parts[1])

            if not conditions:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{ftype} - {PARADIGM_LABELS[paradigm]}")
                continue

            # Create bar chart for best-layer AUC per condition
            cond_aucs = {}
            for cond in sorted(conditions):
                best_auc = 0.5
                for key, val in cond_res.items():
                    if key.endswith(cond) and val.get("auc", 0.5) > best_auc:
                        best_auc = val["auc"]
                cond_aucs[cond] = best_auc

            bars = ax.bar(range(len(cond_aucs)), list(cond_aucs.values()), color=PARADIGM_COLORS[paradigm], alpha=0.7)
            ax.set_xticks(range(len(cond_aucs)))
            ax.set_xticklabels(list(cond_aucs.keys()), rotation=45, ha="right", fontsize=8)
            ax.set_ylim(0.4, 1.05)
            ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)
            ax.set_ylabel("Best AUC")
            ax.set_title(f"{ftype} BK - {PARADIGM_LABELS[paradigm]}")

            # Add value labels on bars
            for bar, val in zip(bars, cond_aucs.values()):
                if val > 0.55:
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                            f'{val:.2f}', ha='center', va='bottom', fontsize=7)

    fig.suptitle("Per-Condition BK Classification: SAE vs Hidden State", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def plot_behavioral_metrics(behavioral_results, output_path):
    """Plot behavioral metrics comparison between BK and safe games."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, paradigm in zip(axes, ["ic", "sm", "mw"]):
        data = behavioral_results.get(paradigm, {}).get("per_game", [])
        if not data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        bk = [g for g in data if g.get("bankruptcy")]
        safe = [g for g in data if not g.get("bankruptcy")]

        metrics = ["i_ba_mean"]
        if paradigm in ["ic", "sm"]:
            metrics.append("i_lc")
        if paradigm == "ic":
            metrics.append("i_ec")

        x = np.arange(len(metrics))
        width = 0.35

        bk_vals = [np.mean([g.get(m, 0) for g in bk]) if bk else 0 for m in metrics]
        safe_vals = [np.mean([g.get(m, 0) for g in safe]) if safe else 0 for m in metrics]

        bars1 = ax.bar(x - width/2, bk_vals, width, label=f"BK ({len(bk)})", color='#e74c3c', alpha=0.8)
        bars2 = ax.bar(x + width/2, safe_vals, width, label=f"Safe ({len(safe)})", color='#2ecc71', alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([m.upper().replace("_", " ") for m in metrics])
        ax.set_title(PARADIGM_LABELS[paradigm])
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Behavioral Metrics: BK vs Safe Games", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyses", type=str, default="1,2,3,4,5,6",
                        help="Comma-separated analysis numbers to run")
    parser.add_argument("--layers", type=str, default="all",
                        help="'all' for all 42 layers, 'best6' for best 6 per paradigm")
    args = parser.parse_args()

    analyses_to_run = set(int(x) for x in args.analyses.split(","))

    # Setup logging
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    JSON_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"comprehensive_gemma_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Starting comprehensive Gemma analysis: analyses={analyses_to_run}, layers={args.layers}")

    all_results = {}

    # Determine layers
    if args.layers == "all":
        analysis_layers = list(range(N_LAYERS))
    else:
        # Use paradigm-specific best layers (union across paradigms)
        analysis_layers = sorted(set(
            l for layers in BEST_LAYERS.values() for l in layers
        ))

    logger.info(f"Layers to analyze: {len(analysis_layers)} layers")

    # --- Analysis 1: Hidden BK ---
    if 1 in analyses_to_run:
        all_results["hidden_bk"] = run_hidden_bk(logger, analysis_layers)

    # --- Analysis 2: Hidden Risk Choice ---
    if 2 in analyses_to_run:
        all_results["hidden_risk"] = run_hidden_risk(logger, analysis_layers)

    # --- Analysis 3: Per-condition SAE BK ---
    if 3 in analyses_to_run:
        # Use best layers for per-condition (expensive with 131K features)
        cond_layers = sorted(set(l for layers in BEST_LAYERS.values() for l in layers))
        all_results["percondition_sae_bk"] = run_percondition_sae_bk(logger, cond_layers)

    # --- Analysis 4: Per-condition SAE Risk ---
    if 4 in analyses_to_run:
        cond_layers = sorted(set(l for layers in BEST_LAYERS.values() for l in layers))
        all_results["percondition_sae_risk"] = run_percondition_sae_risk(logger, cond_layers)

    # --- Analysis 5: Per-condition Hidden BK ---
    if 5 in analyses_to_run:
        all_results["percondition_hidden_bk"] = run_percondition_hidden_bk(logger, analysis_layers)

    # --- Analysis 6: Behavioral Metrics ---
    if 6 in analyses_to_run:
        all_results["behavioral"] = compute_behavioral_metrics(logger)

    # --- Save results ---
    json_file = JSON_DIR / f"comprehensive_gemma_{timestamp}.json"

    # Convert defaultdict to dict for JSON serialization
    def convert_defaultdict(obj):
        if isinstance(obj, defaultdict):
            return dict(obj)
        return obj

    serializable = json.loads(json.dumps(all_results, cls=NumpyEncoder, default=convert_defaultdict))
    with open(json_file, "w") as f:
        json.dump(serializable, f, indent=2, cls=NumpyEncoder)
    logger.info(f"\nResults saved to {json_file}")

    # --- Generate figures ---
    if 1 in analyses_to_run:
        sae_json = JSON_DIR / "all_analyses_20260306_091055.json"
        fig_path = plot_hidden_vs_sae(
            all_results["hidden_bk"],
            sae_json,
            FIGURE_DIR / "comprehensive_hidden_vs_sae_bk.png"
        )
        logger.info(f"Figure saved: {fig_path}")

    if 3 in analyses_to_run or 5 in analyses_to_run:
        fig_path = plot_comprehensive_matrix(
            all_results,
            FIGURE_DIR / "comprehensive_percondition_matrix.png"
        )
        logger.info(f"Figure saved: {fig_path}")

    if 6 in analyses_to_run:
        fig_path = plot_behavioral_metrics(
            all_results.get("behavioral", {}),
            FIGURE_DIR / "comprehensive_behavioral_metrics.png"
        )
        logger.info(f"Figure saved: {fig_path}")

    logger.info(f"\n{'='*60}")
    logger.info("COMPREHENSIVE ANALYSIS COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Results: {json_file}")
    logger.info(f"Figures: {FIGURE_DIR}")
    logger.info(f"Log: {log_file}")


if __name__ == "__main__":
    main()
