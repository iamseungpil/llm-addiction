#!/usr/bin/env python3
"""
V7 Phase 1: Class-imbalance-robust metrics for BK classification.

Adds PR-AUC, bootstrap CIs, and downsampling validation to existing results.

Paradigms (Gemma-2-9B-IT):
  IC: 1600 games, 172 BK (10.8%)
  SM: 3200 games, 87 BK (2.7%)
  MW: 3200 games, 54 BK (1.7%)

Performance notes:
  - Main CV uses lbfgs (matches existing analyses)
  - Bootstrap/downsample uses liblinear (~30x faster)
  - Hidden states (3584 dims) use PCA(100) for bootstrap/downsample
    (retains >98% variance, ~60x faster)

Usage:
    cd /home/jovyan/llm-addiction/sae_v3_analysis/src
    python run_v7_phase1_robust.py
"""

import gc
import json
import logging
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    PARADIGMS, N_LAYERS, RANDOM_SEED,
    MIN_ACTIVATION_RATE, JSON_DIR, LOG_DIR,
)
from data_loader import (
    load_layer_features, load_hidden_states,
    filter_active_features, get_labels,
)

warnings.filterwarnings("ignore")

# PCA components for high-dimensional data in bootstrap/downsample
PCA_N_COMPONENTS = 100


# ===================================================================
# Best layers from existing analyses
# ===================================================================
BEST_LAYERS = {
    "ic": {"sae_dp": 22, "hidden_dp": 26, "sae_r1": 18, "hidden_r1": 33},
    "sm": {"sae_dp": 12, "hidden_dp": 10, "sae_r1": 16, "hidden_r1": 26},
    "mw": {"sae_dp": 33, "hidden_dp": 12, "sae_r1": 22, "hidden_r1": 0},
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


def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"v7_phase1_robust_{datetime.now():%Y%m%d_%H%M%S}.log"

    logger = logging.getLogger("v7_phase1")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.info(f"Log file: {log_file}")
    return logger


# ===================================================================
# Core classification: returns both ROC-AUC and PR-AUC per fold
# ===================================================================
def classify_cv_robust(features, labels, n_folds=5, seed=RANDOM_SEED):
    """5-fold stratified CV returning ROC-AUC and PR-AUC (lbfgs solver)."""
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    effective_folds = min(n_folds, n_pos, n_neg)

    if effective_folds < 2:
        return {
            "roc_auc": 0.5, "pr_auc": n_pos / (n_pos + n_neg),
            "n_pos": n_pos, "n_neg": n_neg, "skipped": True,
        }

    skf = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=seed)
    roc_aucs, pr_aucs = [], []

    for train_idx, test_idx in skf.split(features, labels):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(features[train_idx])
        X_test = scaler.transform(features[test_idx])

        clf = LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=1000,
            class_weight="balanced", random_state=seed,
        )
        clf.fit(X_train, labels[train_idx])
        y_prob = clf.predict_proba(X_test)[:, 1]

        roc_aucs.append(roc_auc_score(labels[test_idx], y_prob))
        pr_aucs.append(average_precision_score(labels[test_idx], y_prob))

    return {
        "roc_auc": float(np.mean(roc_aucs)),
        "pr_auc": float(np.mean(pr_aucs)),
        "roc_auc_std": float(np.std(roc_aucs)),
        "pr_auc_std": float(np.std(pr_aucs)),
        "n_pos": n_pos, "n_neg": n_neg, "skipped": False,
    }


def _maybe_pca(features, seed=RANDOM_SEED):
    """Apply PCA if features are high-dimensional (>500 dims)."""
    if features.shape[1] <= 500:
        return features, None, None
    n_comp = min(PCA_N_COMPONENTS, features.shape[0], features.shape[1])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    pca = PCA(n_components=n_comp, random_state=seed)
    X_pca = pca.fit_transform(X_scaled)
    var_explained = float(pca.explained_variance_ratio_.sum())
    return X_pca, pca, var_explained


# ===================================================================
# Bootstrap 95% CI
# ===================================================================
def bootstrap_ci(features, labels, n_bootstrap=1000, seed=RANDOM_SEED, logger=None):
    """Stratified bootstrap CIs for ROC-AUC and PR-AUC.

    Uses OOB evaluation with liblinear solver. For high-dim data (>500
    features), applies PCA(100) first.
    """
    # PCA for high-dimensional data
    feat_work, pca_obj, var_expl = _maybe_pca(features, seed)
    if pca_obj is not None and logger:
        logger.info(f"    PCA({feat_work.shape[1]}) applied for bootstrap, "
                    f"variance retained: {var_expl:.3f}")

    rng = np.random.RandomState(seed)
    n = len(labels)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    roc_scores = []
    pr_scores = []
    all_idx = np.arange(n)

    t0 = time.time()
    for i in range(n_bootstrap):
        # Stratified resample with replacement
        boot_pos = rng.choice(pos_idx, size=n_pos, replace=True)
        boot_neg = rng.choice(neg_idx, size=n_neg, replace=True)
        boot_idx = np.concatenate([boot_pos, boot_neg])

        # Out-of-bag indices
        oob_mask = np.ones(n, dtype=bool)
        oob_mask[np.unique(boot_idx)] = False
        oob_idx = all_idx[oob_mask]

        oob_labels = labels[oob_idx]
        oob_pos = int(oob_labels.sum())
        oob_neg = len(oob_labels) - oob_pos
        if oob_pos < 2 or oob_neg < 2:
            continue

        scaler = StandardScaler()
        X_train = scaler.fit_transform(feat_work[boot_idx])
        X_test = scaler.transform(feat_work[oob_idx])

        clf = LogisticRegression(
            C=1.0, solver="liblinear", max_iter=1000,
            class_weight="balanced", random_state=seed,
        )
        clf.fit(X_train, labels[boot_idx])
        y_prob = clf.predict_proba(X_test)[:, 1]

        roc_scores.append(roc_auc_score(oob_labels, y_prob))
        pr_scores.append(average_precision_score(oob_labels, y_prob))

        # Progress at 25%, 50%, 75%
        if logger and (i + 1) in (250, 500, 750):
            elapsed = time.time() - t0
            logger.info(f"    ... bootstrap {i+1}/{n_bootstrap} ({elapsed:.0f}s elapsed)")

    elapsed = time.time() - t0
    if logger:
        logger.info(f"    Bootstrap completed in {elapsed:.0f}s")

    if len(roc_scores) < 100:
        return {
            "roc_ci": [None, None], "pr_ci": [None, None],
            "n_valid_bootstraps": len(roc_scores),
        }

    roc_arr = np.array(roc_scores)
    pr_arr = np.array(pr_scores)

    result = {
        "roc_ci": [float(np.percentile(roc_arr, 2.5)), float(np.percentile(roc_arr, 97.5))],
        "pr_ci": [float(np.percentile(pr_arr, 2.5)), float(np.percentile(pr_arr, 97.5))],
        "roc_bootstrap_mean": float(np.mean(roc_arr)),
        "pr_bootstrap_mean": float(np.mean(pr_arr)),
        "n_valid_bootstraps": len(roc_scores),
    }
    if var_expl is not None:
        result["pca_variance_retained"] = var_expl
    return result


# ===================================================================
# Downsampling validation
# ===================================================================
def downsample_validation(features, labels, n_repeats=100, seed=RANDOM_SEED, logger=None):
    """Downsample majority to match minority, 5-fold CV per repeat."""
    rng = np.random.RandomState(seed)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos

    if n_pos < 5 or n_neg < 5:
        return {"downsample_auc_mean": None, "downsample_auc_std": None, "skipped": True}

    # PCA for high-dimensional data
    feat_work, pca_obj, var_expl = _maybe_pca(features, seed)
    if pca_obj is not None and logger:
        logger.info(f"    PCA({feat_work.shape[1]}) applied for downsampling, "
                    f"variance retained: {var_expl:.3f}")

    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    minority_size = min(n_pos, n_neg)

    roc_aucs = []
    pr_aucs = []

    t0 = time.time()
    for i in range(n_repeats):
        if n_pos <= n_neg:
            sub_neg = rng.choice(neg_idx, size=minority_size, replace=False)
            sub_idx = np.concatenate([pos_idx, sub_neg])
        else:
            sub_pos = rng.choice(pos_idx, size=minority_size, replace=False)
            sub_idx = np.concatenate([sub_pos, neg_idx])

        sub_feat = feat_work[sub_idx]
        sub_labels = labels[sub_idx]

        if sub_labels.sum() < 2 or (len(sub_labels) - sub_labels.sum()) < 2:
            continue

        # Fast 5-fold CV with liblinear
        n_sub_pos = int(sub_labels.sum())
        n_sub_neg = len(sub_labels) - n_sub_pos
        effective_folds = min(5, n_sub_pos, n_sub_neg)
        if effective_folds < 2:
            continue

        skf = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=seed + i)
        fold_rocs, fold_prs = [], []

        for train_idx, test_idx in skf.split(sub_feat, sub_labels):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(sub_feat[train_idx])
            X_test = scaler.transform(sub_feat[test_idx])

            clf = LogisticRegression(
                C=1.0, solver="liblinear", max_iter=1000,
                class_weight="balanced", random_state=seed,
            )
            clf.fit(X_train, sub_labels[train_idx])
            y_prob = clf.predict_proba(X_test)[:, 1]

            fold_rocs.append(roc_auc_score(sub_labels[test_idx], y_prob))
            fold_prs.append(average_precision_score(sub_labels[test_idx], y_prob))

        roc_aucs.append(float(np.mean(fold_rocs)))
        pr_aucs.append(float(np.mean(fold_prs)))

    elapsed = time.time() - t0
    if logger:
        logger.info(f"    Downsampling completed in {elapsed:.0f}s")

    if len(roc_aucs) == 0:
        return {"downsample_auc_mean": None, "downsample_auc_std": None, "skipped": True}

    result = {
        "downsample_roc_mean": float(np.mean(roc_aucs)),
        "downsample_roc_std": float(np.std(roc_aucs)),
        "downsample_pr_mean": float(np.mean(pr_aucs)),
        "downsample_pr_std": float(np.std(pr_aucs)),
        "n_valid_repeats": len(roc_aucs),
        "downsample_size_per_class": minority_size,
        "skipped": False,
    }
    if var_expl is not None:
        result["pca_variance_retained"] = var_expl
    return result


# ===================================================================
# Data loading helpers
# ===================================================================
def load_sae_data(paradigm, layer, mode):
    """Load SAE features + labels for a paradigm/layer/mode."""
    if mode == "dp":
        loaded = load_layer_features(paradigm, layer, mode="decision_point", dense=True)
    elif mode == "r1":
        loaded = load_layer_features(paradigm, layer, mode="all_rounds", dense=True)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if loaded is None:
        return None, None

    features, meta = loaded
    labels = get_labels(meta)

    if mode == "r1":
        r1_mask = meta["round_nums"] == 1
        features = features[r1_mask]
        labels = labels[r1_mask]

    # Filter to active features
    filtered, _ = filter_active_features(features, MIN_ACTIVATION_RATE)
    return filtered, labels


def load_hidden_data(paradigm, layer, mode):
    """Load hidden states + labels for a paradigm/layer/mode."""
    if mode == "dp":
        loaded = load_hidden_states(paradigm, layer, mode="decision_point")
    elif mode == "r1":
        loaded = load_hidden_states(paradigm, layer, mode="all_rounds")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if loaded is None:
        return None, None

    features, meta = loaded
    labels = get_labels(meta)

    if mode == "r1":
        r1_mask = meta["round_nums"] == 1
        features = features[r1_mask]
        labels = labels[r1_mask]

    return features, labels


# ===================================================================
# Main analysis
# ===================================================================
def run_analysis(logger):
    """Run all robust metric analyses."""
    results = {}
    total_start = time.time()
    analysis_count = 0

    for paradigm in ["ic", "sm", "mw"]:
        cfg = PARADIGMS[paradigm]
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"PARADIGM: {cfg['name']} ({cfg['n_games']} games, {cfg['n_bk']} BK)")
        logger.info("=" * 70)

        paradigm_results = {}
        best = BEST_LAYERS[paradigm]

        for analysis_key in ["sae_dp", "hidden_dp", "sae_r1", "hidden_r1"]:
            analysis_count += 1
            layer = best[analysis_key]
            source_type, eval_mode = analysis_key.split("_")
            mode_label = "Decision Point" if eval_mode == "dp" else "Round 1"
            source_label = "SAE" if source_type == "sae" else "Hidden"

            logger.info(f"\n--- [{analysis_count}/12] {source_label} {mode_label} (L{layer}) ---")

            # Load data
            t_load = time.time()
            if source_type == "sae":
                features, labels = load_sae_data(paradigm, layer, eval_mode)
            else:
                features, labels = load_hidden_data(paradigm, layer, eval_mode)
            logger.info(f"  Data loaded in {time.time()-t_load:.1f}s")

            if features is None or labels is None:
                logger.warning(f"  Data not available for {analysis_key}")
                paradigm_results[analysis_key] = {"error": "data_not_available"}
                continue

            n_pos = int(labels.sum())
            n_neg = len(labels) - n_pos
            n_feat = features.shape[1]
            logger.info(f"  Samples: {len(labels)} ({n_pos} BK, {n_neg} non-BK)")
            logger.info(f"  Features: {n_feat}")

            # 1. Baseline classification with both metrics
            logger.info("  [1/3] Classification (ROC-AUC + PR-AUC, lbfgs)...")
            t0 = time.time()
            cv_res = classify_cv_robust(features, labels)
            logger.info(f"    ROC-AUC: {cv_res['roc_auc']:.3f} (std={cv_res.get('roc_auc_std', 0):.3f})")
            logger.info(f"    PR-AUC:  {cv_res['pr_auc']:.3f} (std={cv_res.get('pr_auc_std', 0):.3f})")
            logger.info(f"    Completed in {time.time()-t0:.0f}s")

            # 2. Bootstrap CI
            logger.info("  [2/3] Bootstrap 95% CI (1000 iterations, liblinear)...")
            boot_res = bootstrap_ci(features, labels, n_bootstrap=1000, logger=logger)
            if boot_res["roc_ci"][0] is not None:
                logger.info(f"    ROC-AUC CI: [{boot_res['roc_ci'][0]:.3f}, {boot_res['roc_ci'][1]:.3f}]")
                logger.info(f"    PR-AUC  CI: [{boot_res['pr_ci'][0]:.3f}, {boot_res['pr_ci'][1]:.3f}]")
                logger.info(f"    Valid bootstraps: {boot_res['n_valid_bootstraps']}")
            else:
                logger.warning("    Bootstrap failed (insufficient valid samples)")

            # 3. Downsampling validation
            logger.info("  [3/3] Downsampling validation (100 repeats, liblinear)...")
            ds_res = downsample_validation(features, labels, n_repeats=100, logger=logger)
            if not ds_res.get("skipped", True):
                logger.info(f"    Downsample ROC: {ds_res['downsample_roc_mean']:.3f} +/- {ds_res['downsample_roc_std']:.3f}")
                logger.info(f"    Downsample PR:  {ds_res['downsample_pr_mean']:.3f} +/- {ds_res['downsample_pr_std']:.3f}")
                logger.info(f"    Balanced size: {ds_res['downsample_size_per_class']} per class")
            else:
                logger.warning("    Downsampling skipped (insufficient samples)")

            # Combine results
            entry = {
                "layer": layer,
                "roc_auc": cv_res["roc_auc"],
                "pr_auc": cv_res["pr_auc"],
                "roc_auc_std": cv_res.get("roc_auc_std", 0),
                "pr_auc_std": cv_res.get("pr_auc_std", 0),
                "roc_ci": boot_res.get("roc_ci", [None, None]),
                "pr_ci": boot_res.get("pr_ci", [None, None]),
                "n_pos": n_pos,
                "n_neg": n_neg,
                "n_features": n_feat,
            }

            # Add downsampling fields
            if not ds_res.get("skipped", True):
                entry["downsample_roc_mean"] = ds_res["downsample_roc_mean"]
                entry["downsample_roc_std"] = ds_res["downsample_roc_std"]
                entry["downsample_pr_mean"] = ds_res["downsample_pr_mean"]
                entry["downsample_pr_std"] = ds_res["downsample_pr_std"]
                entry["downsample_size_per_class"] = ds_res["downsample_size_per_class"]

            paradigm_results[analysis_key] = entry

            # Free memory
            del features, labels
            gc.collect()

        results[paradigm] = paradigm_results

    total_elapsed = time.time() - total_start
    logger.info(f"\nTotal analysis time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    return results


def print_summary(results, logger):
    """Print a compact summary table."""
    logger.info("")
    logger.info("=" * 110)
    logger.info("SUMMARY TABLE")
    logger.info("=" * 110)
    logger.info(f"{'Para':<4} {'Analysis':<12} {'L':>3} {'ROC-AUC':>8} {'ROC CI':>16} "
                f"{'PR-AUC':>8} {'PR CI':>16} {'DS-ROC':>14} {'DS-PR':>14}")
    logger.info("-" * 110)

    for paradigm in ["ic", "sm", "mw"]:
        for analysis_key in ["sae_dp", "hidden_dp", "sae_r1", "hidden_r1"]:
            entry = results.get(paradigm, {}).get(analysis_key, {})
            if "error" in entry:
                logger.info(f"{paradigm.upper():<4} {analysis_key:<12} {'N/A':>3} {'ERROR':>8}")
                continue

            layer = entry.get("layer", -1)
            roc = entry.get("roc_auc", 0)
            pr = entry.get("pr_auc", 0)

            roc_ci = entry.get("roc_ci", [None, None])
            pr_ci = entry.get("pr_ci", [None, None])
            roc_ci_str = f"[{roc_ci[0]:.3f},{roc_ci[1]:.3f}]" if roc_ci[0] is not None else "N/A"
            pr_ci_str = f"[{pr_ci[0]:.3f},{pr_ci[1]:.3f}]" if pr_ci[0] is not None else "N/A"

            ds_roc = entry.get("downsample_roc_mean")
            ds_roc_std = entry.get("downsample_roc_std")
            ds_pr = entry.get("downsample_pr_mean")
            ds_pr_std = entry.get("downsample_pr_std")
            ds_roc_str = f"{ds_roc:.3f}+/-{ds_roc_std:.3f}" if ds_roc is not None else "N/A"
            ds_pr_str = f"{ds_pr:.3f}+/-{ds_pr_std:.3f}" if ds_pr is not None else "N/A"

            logger.info(f"{paradigm.upper():<4} {analysis_key:<12} {layer:>3} {roc:>8.3f} {roc_ci_str:>16} "
                        f"{pr:>8.3f} {pr_ci_str:>16} {ds_roc_str:>14} {ds_pr_str:>14}")
        logger.info("-" * 110)


def main():
    logger = setup_logging()
    logger.info("V7 Phase 1: Class-Imbalance-Robust Metrics")
    logger.info(f"Timestamp: {datetime.now():%Y-%m-%d %H:%M:%S}")
    logger.info(f"Random seed: {RANDOM_SEED}")
    logger.info(f"Solvers: lbfgs (main CV), liblinear (bootstrap/downsample)")
    logger.info(f"PCA: {PCA_N_COMPONENTS} components for features > 500 dims")

    results = run_analysis(logger)

    print_summary(results, logger)

    # Save results
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "phase1_robust_metrics": results,
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "random_seed": RANDOM_SEED,
            "n_bootstrap": 1000,
            "n_downsample_repeats": 100,
            "cv_folds": 5,
            "class_weight": "balanced",
            "best_layers": BEST_LAYERS,
            "pca_n_components_for_bootstrap": PCA_N_COMPONENTS,
            "main_cv_solver": "lbfgs",
            "bootstrap_downsample_solver": "liblinear",
        },
    }

    out_path = JSON_DIR / "v7_phase1_robust.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    logger.info(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    main()
