#!/usr/bin/env python3
"""
Fill 4 hidden state analysis gaps:
  7. Balance-matched Hidden BK (DP)
  8. Hidden BK per-condition R1
  9. Hidden Risk per-condition
  10. Hidden cross-domain transfer

Usage:
  python run_hidden_gaps.py [--analyses 7,8,9,10] [--layers best|all]
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
from config import (PARADIGMS, N_LAYERS, HIDDEN_DIM, RANDOM_SEED,
                    PARADIGM_LABELS, PARADIGM_COLORS,
                    JSON_DIR, LOG_DIR, FIGURE_DIR)
from data_loader import load_hidden_states, get_labels

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_SEED)

# Same best layers as comprehensive_gemma
BEST_LAYERS = {
    "ic": [0, 6, 8, 10, 12, 16, 18, 22, 24, 26, 33, 40],
    "sm": [0, 6, 8, 10, 12, 16, 18, 22, 24, 26, 33, 40],
    "mw": [0, 6, 8, 10, 12, 16, 18, 22, 24, 26, 33, 40],
}

PROMPT_COMPONENTS = {
    "ic": ["G", "M"],
    "sm": ["G", "M", "H", "W", "P"],
    "mw": ["G", "M", "H", "W", "P"],
}

COMPONENT_DATA_MAP = {"H": "R"}

DATA_ROOT = Path("/home/jovyan/beomi/llm-addiction-data")
SM_JSON_FILE = DATA_ROOT / "slot_machine" / "experiment_0_gemma_v4_role" / "final_gemma_20260227_002507.json"
MW_JSON_FILE = DATA_ROOT / "mystery_wheel_v2_role" / "gemma_mysterywheel_checkpoint_3200.json"


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


def classify_transfer(X_train, y_train, X_test, y_test):
    """Train on one dataset, test on another."""
    n_pos_train = int(y_train.sum())
    n_neg_train = len(y_train) - n_pos_train
    n_pos_test = int(y_test.sum())
    n_neg_test = len(y_test) - n_pos_test

    if n_pos_train < 2 or n_neg_train < 2 or n_pos_test < 2 or n_neg_test < 2:
        return {"auc": 0.5, "skipped": True, "n_train": len(y_train), "n_test": len(y_test)}

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000,
                             class_weight="balanced", random_state=RANDOM_SEED)
    clf.fit(X_tr, y_train)
    y_prob = clf.predict_proba(X_te)[:, 1]

    return {
        "auc": float(roc_auc_score(y_test, y_prob)),
        "n_train": len(y_train), "n_test": len(y_test),
        "n_pos_train": n_pos_train, "n_pos_test": n_pos_test,
        "skipped": False,
    }


# ===================================================================
# Choice loaders (for risk classification)
# ===================================================================

def find_ic_json_files():
    ic_dir = DATA_ROOT / "investment_choice_v2_role"
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


def get_risk_labels(paradigm, game_ids, round_nums):
    """Binary risk labels: 1=risky, 0=safe."""
    if paradigm == "ic":
        choices = {}
        for json_path, offset in find_ic_json_files():
            with open(json_path) as f:
                data = json.load(f)
            for i, game in enumerate(data["results"]):
                npz_gid = offset + i + 1
                choices[npz_gid] = {}
                for dec in game.get("decisions", []):
                    choices[npz_gid][dec["round"]] = dec["choice"]
        labels = np.full(len(game_ids), -1, dtype=np.int32)
        for i, (gid, rnd) in enumerate(zip(game_ids, round_nums)):
            ch = choices.get(int(gid), {}).get(int(rnd))
            if ch is not None:
                labels[i] = 1 if ch >= 3 else 0
        valid = labels >= 0
        return labels, valid

    elif paradigm == "sm":
        with open(SM_JSON_FILE) as f:
            data = json.load(f)
        choices = {}
        for i, game in enumerate(data["results"]):
            npz_gid = i + 1
            choices[npz_gid] = {}
            for dec in game.get("decisions", []):
                choices[npz_gid][dec["round"]] = {"action": dec["action"], "bet": dec.get("bet", 0)}
        labels = np.full(len(game_ids), -1, dtype=np.int32)
        for i, (gid, rnd) in enumerate(zip(game_ids, round_nums)):
            info = choices.get(int(gid), {}).get(int(rnd))
            if info is not None:
                labels[i] = 1 if info["action"] == "bet" else 0
        valid = labels >= 0
        return labels, valid

    elif paradigm == "mw":
        with open(MW_JSON_FILE) as f:
            data = json.load(f)
        choices = {}
        for i, game in enumerate(data["results"]):
            npz_gid = i + 1
            choices[npz_gid] = {}
            for dec in game.get("decisions", []):
                # MW choice: 1=stop(safe), 2=spin(risky). Integer, not string.
                choices[npz_gid][dec["round"]] = dec["choice"]
        labels = np.full(len(game_ids), -1, dtype=np.int32)
        for i, (gid, rnd) in enumerate(zip(game_ids, round_nums)):
            ch = choices.get(int(gid), {}).get(int(rnd))
            if ch is not None:
                labels[i] = 1 if ch == 2 else 0  # 2=spin(risky), 1=stop(safe)
        valid = labels >= 0
        return labels, valid

    raise ValueError(f"Unknown paradigm: {paradigm}")


# ===================================================================
# Analysis 7: Balance-Matched Hidden BK
# ===================================================================

def run_balance_matched_hidden(logger, layers):
    """Hidden state BK classification with balance-matched controls."""
    logger.info("=" * 60)
    logger.info("Analysis 7: Balance-Matched Hidden BK Classification")
    logger.info("=" * 60)

    results = {}
    for paradigm in PARADIGMS:
        results[paradigm] = {"dp": [], "r1": []}
        logger.info(f"\n--- {PARADIGM_LABELS[paradigm]} ---")

        for layer in layers:
            # Decision-point
            dp_loaded = load_hidden_states(paradigm, layer, mode="decision_point")
            if dp_loaded is None:
                continue

            features, meta = dp_loaded
            labels = get_labels(meta)
            n_bk = int(labels.sum())

            if n_bk < 5:
                continue

            # Balance matching: for each BK game, find safe game with closest balance
            if "balances" in meta:
                balances = meta["balances"].astype(float)
            else:
                # Fallback: skip balance matching
                logger.info(f"  L{layer}: no balance data, skipping")
                continue

            bk_idx = np.where(labels == 1)[0]
            safe_idx = np.where(labels == 0)[0]

            # Match each BK to nearest-balance safe game (without replacement)
            matched_safe = []
            available_safe = set(range(len(safe_idx)))
            for bi in bk_idx:
                bk_bal = balances[bi]
                best_j = None
                best_diff = float("inf")
                for j in available_safe:
                    diff = abs(balances[safe_idx[j]] - bk_bal)
                    if diff < best_diff:
                        best_diff = diff
                        best_j = j
                if best_j is not None:
                    matched_safe.append(safe_idx[best_j])
                    available_safe.remove(best_j)

            if len(matched_safe) < 5:
                continue

            matched_idx = np.concatenate([bk_idx, np.array(matched_safe)])
            matched_labels = labels[matched_idx]
            matched_features = features[matched_idx]

            res = classify_cv(matched_features, matched_labels)
            res["layer"] = layer
            res["n_pairs"] = len(matched_safe)
            results[paradigm]["dp"].append(res)
            logger.info(f"  L{layer} DP balanced: AUC={res['auc']:.3f} "
                        f"(n_pairs={len(matched_safe)}, n_pos={res['n_pos']})")

            # R1: balance matching makes less sense (all start at $100)
            # but include for completeness — at R1 all balances are $100
            r1_loaded = load_hidden_states(paradigm, layer, mode="all_rounds")
            if r1_loaded is not None:
                r1_feat, r1_meta = r1_loaded
                r1_mask = r1_meta["round_nums"] == 1
                if r1_mask.sum() > 0:
                    r1_labels = get_labels({k: v[r1_mask] for k, v in r1_meta.items()})
                    r1_n_bk = int(r1_labels.sum())
                    if r1_n_bk >= 5:
                        # At R1, all balances are equal ($100), so no matching needed
                        # Just use balanced class weights (already in classify_cv)
                        r1_res = classify_cv(r1_feat[r1_mask], r1_labels)
                        r1_res["layer"] = layer
                        results[paradigm]["r1"].append(r1_res)
                        logger.info(f"  L{layer} R1 (all $100): AUC={r1_res['auc']:.3f}")

            gc.collect()

        for mode in ["dp", "r1"]:
            if results[paradigm][mode]:
                best = max(results[paradigm][mode], key=lambda r: r["auc"])
                logger.info(f"  Best {mode.upper()}: L{best['layer']} AUC={best['auc']:.3f}")

    return results


# ===================================================================
# Analysis 8: Hidden BK Per-Condition R1
# ===================================================================

def run_percondition_hidden_bk_r1(logger, layers):
    """Hidden state BK classification at R1 within condition subsets."""
    logger.info("=" * 60)
    logger.info("Analysis 8: Per-Condition Hidden BK R1")
    logger.info("=" * 60)

    results = {}
    for paradigm in PARADIGMS:
        results[paradigm] = {}
        logger.info(f"\n--- {PARADIGM_LABELS[paradigm]} ---")

        for layer in layers:
            loaded = load_hidden_states(paradigm, layer, mode="all_rounds")
            if loaded is None:
                continue

            features, meta = loaded
            r1_mask = meta["round_nums"] == 1
            if r1_mask.sum() == 0:
                continue

            r1_feat = features[r1_mask]
            r1_meta = {k: v[r1_mask] for k, v in meta.items()}
            r1_labels = get_labels(r1_meta)

            # --- bet_type ---
            for bt in ["fixed", "variable"]:
                bt_mask = r1_meta["bet_types"] == bt
                n_bk = int((r1_labels[bt_mask] == 1).sum()) if bt_mask.sum() > 0 else 0
                cond_key = f"L{layer}_bt_{bt}_R1"

                if bt_mask.sum() >= 10 and n_bk >= 2:
                    res = classify_cv(r1_feat[bt_mask], r1_labels[bt_mask])
                    res["layer"] = layer
                    res["condition"] = f"bet_type={bt}"
                    results[paradigm][cond_key] = res
                    logger.info(f"  L{layer} R1 {bt}: AUC={res['auc']:.3f} (BK={n_bk})")

            # --- prompt components ---
            if "prompt_conditions" in r1_meta:
                for comp in PROMPT_COMPONENTS.get(paradigm, []):
                    data_comp = COMPONENT_DATA_MAP.get(comp, comp)
                    has_comp = np.array([data_comp in str(pc) for pc in r1_meta["prompt_conditions"]])
                    for subset_name, subset_mask in [("with_" + comp, has_comp), ("without_" + comp, ~has_comp)]:
                        n_bk = int((r1_labels[subset_mask] == 1).sum()) if subset_mask.sum() > 0 else 0
                        cond_key = f"L{layer}_{subset_name}_R1"
                        if subset_mask.sum() >= 10 and n_bk >= 2:
                            res = classify_cv(r1_feat[subset_mask], r1_labels[subset_mask])
                            res["layer"] = layer
                            res["condition"] = subset_name
                            results[paradigm][cond_key] = res
                            logger.info(f"  L{layer} R1 {subset_name}: AUC={res['auc']:.3f} (BK={n_bk})")

            # --- bet_constraint (IC only) ---
            if "bet_constraints" in r1_meta and paradigm == "ic":
                for bc in ["30", "50", "70"]:
                    bc_mask = r1_meta["bet_constraints"] == bc
                    n_bk = int((r1_labels[bc_mask] == 1).sum()) if bc_mask.sum() > 0 else 0
                    cond_key = f"L{layer}_bc_{bc}_R1"
                    if bc_mask.sum() >= 10 and n_bk >= 2:
                        res = classify_cv(r1_feat[bc_mask], r1_labels[bc_mask])
                        res["layer"] = layer
                        res["condition"] = f"constraint={bc}"
                        results[paradigm][cond_key] = res
                        logger.info(f"  L{layer} R1 bc_{bc}: AUC={res['auc']:.3f} (BK={n_bk})")

            gc.collect()

    return results


# ===================================================================
# Analysis 9: Hidden Risk Per-Condition
# ===================================================================

def run_percondition_hidden_risk(logger, layers):
    """Hidden state risk classification within condition subsets."""
    logger.info("=" * 60)
    logger.info("Analysis 9: Per-Condition Hidden Risk Classification")
    logger.info("=" * 60)

    results = {}
    for paradigm in PARADIGMS:
        results[paradigm] = {}
        logger.info(f"\n--- {PARADIGM_LABELS[paradigm]} ---")

        risk_labels_cache = None

        for layer in layers:
            loaded = load_hidden_states(paradigm, layer, mode="all_rounds")
            if loaded is None:
                continue

            features, meta = loaded

            # Get risk labels (once)
            if risk_labels_cache is None:
                risk_labels_cache, valid_mask = get_risk_labels(
                    paradigm, meta["game_ids"], meta["round_nums"]
                )
            else:
                valid_mask = risk_labels_cache >= 0

            if valid_mask.sum() < 20:
                continue

            risk_labels = risk_labels_cache[valid_mask]
            feat_valid = features[valid_mask]
            meta_valid = {k: v[valid_mask] for k, v in meta.items()}

            # --- bet_type ---
            for bt in ["fixed", "variable"]:
                bt_mask = meta_valid["bet_types"] == bt
                if bt_mask.sum() >= 20:
                    n_pos = int(risk_labels[bt_mask].sum())
                    n_neg = int((~risk_labels[bt_mask].astype(bool)).sum())
                    if n_pos >= 5 and n_neg >= 5:
                        res = classify_cv(feat_valid[bt_mask], risk_labels[bt_mask])
                        res["layer"] = layer
                        results[paradigm][f"L{layer}_bt_{bt}"] = res
                        logger.info(f"  L{layer} risk {bt}: AUC={res['auc']:.3f} "
                                    f"(risky={res['n_pos']}, safe={res['n_neg']})")

            # --- prompt components ---
            if "prompt_conditions" in meta_valid:
                for comp in PROMPT_COMPONENTS.get(paradigm, []):
                    data_comp = COMPONENT_DATA_MAP.get(comp, comp)
                    has_comp = np.array([data_comp in str(pc) for pc in meta_valid["prompt_conditions"]])
                    for subset_name, subset_mask in [("with_" + comp, has_comp), ("without_" + comp, ~has_comp)]:
                        if subset_mask.sum() >= 20:
                            n_pos = int(risk_labels[subset_mask].sum())
                            n_neg = int((~risk_labels[subset_mask].astype(bool)).sum())
                            if n_pos >= 5 and n_neg >= 5:
                                res = classify_cv(feat_valid[subset_mask], risk_labels[subset_mask])
                                res["layer"] = layer
                                results[paradigm][f"L{layer}_{subset_name}"] = res
                                logger.info(f"  L{layer} risk {subset_name}: AUC={res['auc']:.3f}")

            # --- bet_constraint (IC only) ---
            if "bet_constraints" in meta_valid and paradigm == "ic":
                for bc in ["30", "50", "70"]:
                    bc_mask = meta_valid["bet_constraints"] == bc
                    if bc_mask.sum() >= 20:
                        n_pos = int(risk_labels[bc_mask].sum())
                        n_neg = int((~risk_labels[bc_mask].astype(bool)).sum())
                        if n_pos >= 5 and n_neg >= 5:
                            res = classify_cv(feat_valid[bc_mask], risk_labels[bc_mask])
                            res["layer"] = layer
                            results[paradigm][f"L{layer}_bc_{bc}"] = res
                            logger.info(f"  L{layer} risk bc_{bc}: AUC={res['auc']:.3f}")

            gc.collect()

    return results


# ===================================================================
# Analysis 10: Hidden Cross-Domain Transfer
# ===================================================================

def run_hidden_cross_domain(logger, layers):
    """Train hidden-state classifier on one paradigm, test on another."""
    logger.info("=" * 60)
    logger.info("Analysis 10: Hidden Cross-Domain Transfer")
    logger.info("=" * 60)

    pairs = [
        ("ic", "sm"), ("ic", "mw"),
        ("sm", "ic"), ("sm", "mw"),
        ("mw", "ic"), ("mw", "sm"),
    ]

    results = {}
    for src, tgt in pairs:
        pair_key = f"{src}_to_{tgt}"
        results[pair_key] = []
        logger.info(f"\n--- {PARADIGM_LABELS[src]} → {PARADIGM_LABELS[tgt]} ---")

        for layer in layers:
            src_loaded = load_hidden_states(src, layer, mode="decision_point")
            tgt_loaded = load_hidden_states(tgt, layer, mode="decision_point")

            if src_loaded is None or tgt_loaded is None:
                continue

            src_feat, src_meta = src_loaded
            tgt_feat, tgt_meta = tgt_loaded
            src_labels = get_labels(src_meta)
            tgt_labels = get_labels(tgt_meta)

            res = classify_transfer(src_feat, src_labels, tgt_feat, tgt_labels)
            res["layer"] = layer
            results[pair_key].append(res)
            logger.info(f"  L{layer}: AUC={res['auc']:.3f} "
                        f"(train={res['n_train']}, test={res['n_test']})")

            gc.collect()

        if results[pair_key]:
            best = max(results[pair_key], key=lambda r: r["auc"])
            logger.info(f"  Best: L{best['layer']} AUC={best['auc']:.3f}")

    return results


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyses", type=str, default="7,8,9,10",
                        help="Comma-separated analysis numbers (7-10)")
    parser.add_argument("--layers", type=str, default="best",
                        help="'best' for 12 best layers, 'all' for all 42")
    args = parser.parse_args()

    analyses_to_run = set(int(x) for x in args.analyses.split(","))

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    JSON_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"hidden_gaps_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Starting hidden gap analyses: {analyses_to_run}")

    if args.layers == "all":
        analysis_layers = list(range(N_LAYERS))
    else:
        analysis_layers = sorted(set(l for layers in BEST_LAYERS.values() for l in layers))

    logger.info(f"Layers: {len(analysis_layers)}")

    all_results = {}

    if 7 in analyses_to_run:
        all_results["balance_matched_hidden"] = run_balance_matched_hidden(logger, analysis_layers)

    if 8 in analyses_to_run:
        all_results["percondition_hidden_bk_r1"] = run_percondition_hidden_bk_r1(logger, analysis_layers)

    if 9 in analyses_to_run:
        all_results["percondition_hidden_risk"] = run_percondition_hidden_risk(logger, analysis_layers)

    if 10 in analyses_to_run:
        all_results["hidden_cross_domain"] = run_hidden_cross_domain(logger, analysis_layers)

    # Save results
    json_file = JSON_DIR / f"hidden_gaps_{timestamp}.json"
    with open(json_file, "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    logger.info(f"\nResults saved: {json_file}")

    logger.info(f"\n{'=' * 60}")
    logger.info("HIDDEN GAP ANALYSES COMPLETE")
    logger.info(f"{'=' * 60}")
    logger.info(f"Results: {json_file}")
    logger.info(f"Log: {log_file}")


if __name__ == "__main__":
    main()
