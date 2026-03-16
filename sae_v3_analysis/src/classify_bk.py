#!/usr/bin/env python3
"""
BK Classification Analysis (V3 All-Rounds SAE Features).

Three classification modes:
  1. Decision-point: last-round features only (comparable to V2)
  2. Game-mean: mean activation across all rounds per game
  3. Game-max: max activation across all rounds per game

For each mode:
  - 5-fold stratified CV with balanced logistic regression
  - Layer-wise AUC curve
  - Permutation test on best layer
  - Top discriminative features

Usage:
    cd /home/jovyan/llm-addiction/sae_v3_analysis/src
    python classify_bk.py --paradigm ic
    python classify_bk.py --paradigm ic sm mw  # all available
    python classify_bk.py --paradigm ic --layers 18,26,30  # specific layers
"""

import sys
import json
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from config import (
    PARADIGMS, N_LAYERS, FIGURE_DIR, JSON_DIR, LOG_DIR,
    MIN_ACTIVATION_RATE, CLASSIFICATION_CV_FOLDS, CLASSIFICATION_C,
    N_PERMUTATIONS, RANDOM_SEED, PARADIGM_COLORS, PARADIGM_LABELS,
)
from data_loader import (
    load_layer_features, filter_active_features, get_labels,
    check_paradigm_ready,
)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def setup_logging(paradigms: list) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "_".join(paradigms)
    log_file = LOG_DIR / f"classify_bk_{tag}_{ts}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Log: {log_file}")
    return logger


# ===================================================================
# Classification core
# ===================================================================
def classify_layer(features, labels, n_folds=5, C=1.0):
    """5-fold stratified CV classification. Returns dict with metrics."""
    n_samples, n_features = features.shape
    n_pos = int(labels.sum())
    n_neg = n_samples - n_pos

    if n_pos < n_folds or n_neg < n_folds:
        return {
            "auc_mean": 0.5, "auc_std": 0.0, "accuracy_mean": 0.5,
            "f1_mean": 0.0, "n_features": n_features,
            "n_pos": n_pos, "n_neg": n_neg, "skipped": True,
        }

    np.random.seed(RANDOM_SEED)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    aucs, accs, f1s = [], [], []
    all_coefs = np.zeros(n_features)

    for train_idx, test_idx in skf.split(features, labels):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(features[train_idx])
        X_test = scaler.transform(features[test_idx])
        y_train, y_test = labels[train_idx], labels[test_idx]

        clf = LogisticRegression(
            C=C, solver="lbfgs", max_iter=1000,
            class_weight="balanced", random_state=RANDOM_SEED,
        )
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred = clf.predict(X_test)

        aucs.append(roc_auc_score(y_test, y_prob))
        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred, zero_division=0))
        all_coefs += clf.coef_[0]

    all_coefs /= n_folds
    top_k = min(50, n_features)
    top_idx = np.argsort(np.abs(all_coefs))[-top_k:][::-1]

    return {
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "auc_folds": [float(a) for a in aucs],
        "accuracy_mean": float(np.mean(accs)),
        "f1_mean": float(np.mean(f1s)),
        "n_features": n_features,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "skipped": False,
        "top_features": [
            {"feature_idx": int(top_idx[i]), "coef": float(all_coefs[top_idx[i]])}
            for i in range(len(top_idx))
        ],
    }


def permutation_test(features, labels, observed_auc, n_perm=1000, n_folds=5, C=1.0):
    """Permutation test: shuffle labels, measure AUC distribution."""
    perm_aucs = []
    for i in range(n_perm):
        perm_labels = np.random.RandomState(i).permutation(labels)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=i)
        fold_aucs = []
        for train_idx, test_idx in skf.split(features, perm_labels):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(features[train_idx])
            X_test = scaler.transform(features[test_idx])
            clf = LogisticRegression(
                C=C, solver="lbfgs", max_iter=500,
                class_weight="balanced", random_state=RANDOM_SEED,
            )
            clf.fit(X_train, perm_labels[train_idx])
            try:
                fold_aucs.append(roc_auc_score(
                    perm_labels[test_idx], clf.predict_proba(X_test)[:, 1]
                ))
            except ValueError:
                fold_aucs.append(0.5)
        perm_aucs.append(np.mean(fold_aucs))

    perm_aucs = np.array(perm_aucs)
    return {
        "p_value": float((perm_aucs >= observed_auc).mean()),
        "observed_auc": float(observed_auc),
        "perm_mean": float(perm_aucs.mean()),
        "perm_std": float(perm_aucs.std()),
        "perm_95": float(np.percentile(perm_aucs, 95)),
    }


# ===================================================================
# Main analysis per paradigm
# ===================================================================
def analyze_paradigm(paradigm: str, layers: list, modes: list, logger, skip_perm=False):
    """Run classification for one paradigm across layers and modes."""
    cfg = PARADIGMS[paradigm]
    logger.info(f"\n{'='*70}")
    logger.info(f"CLASSIFY: {cfg['name']} ({cfg['short']})")
    logger.info(f"{'='*70}")

    results = {}

    for mode in modes:
        logger.info(f"\n--- Mode: {mode} ---")
        layer_results = []
        best_layer, best_auc = None, 0.0
        best_features, best_labels = None, None

        for layer in layers:
            loaded = load_layer_features(paradigm, layer, mode=mode, dense=True)
            if loaded is None:
                logger.warning(f"  Layer {layer}: not found")
                continue

            features, meta = loaded
            labels = get_labels(meta)
            filtered, active_idx = filter_active_features(features, MIN_ACTIVATION_RATE)

            n_bk = int(labels.sum())
            logger.info(f"  L{layer:02d}: {features.shape[0]} samples, "
                        f"{filtered.shape[1]} active features, {n_bk} BK")

            result = classify_layer(filtered, labels, CLASSIFICATION_CV_FOLDS, CLASSIFICATION_C)
            result["layer"] = layer
            result["active_feature_indices"] = active_idx.tolist()[:20]  # save first 20 for reference
            layer_results.append(result)

            if not result["skipped"] and result["auc_mean"] > best_auc:
                best_auc = result["auc_mean"]
                best_layer = layer
                best_features = filtered
                best_labels = labels

            logger.info(f"         AUC={result['auc_mean']:.4f} +/- {result['auc_std']:.4f}")

        # Permutation test on best layer
        perm_result = None
        if best_layer is not None and best_features is not None and not skip_perm:
            logger.info(f"\n  Permutation test: L{best_layer} (AUC={best_auc:.4f}), "
                        f"{N_PERMUTATIONS} permutations...")
            perm_result = permutation_test(
                best_features, best_labels, best_auc, N_PERMUTATIONS,
            )
            logger.info(f"    p={perm_result['p_value']:.4f}, "
                        f"null AUC={perm_result['perm_mean']:.4f}+/-{perm_result['perm_std']:.4f}")

        results[mode] = {
            "layer_results": layer_results,
            "best_layer": best_layer,
            "best_auc": float(best_auc) if best_auc > 0 else None,
            "permutation_test": perm_result,
        }

    return results


# ===================================================================
# Visualization
# ===================================================================
def plot_auc_curves(all_results: dict, save_path: Path):
    """Plot AUC vs layer for all paradigms and modes."""
    modes = list(next(iter(all_results.values())).keys())
    paradigms = list(all_results.keys())
    n_modes = len(modes)

    fig, axes = plt.subplots(1, n_modes, figsize=(6 * n_modes, 5), squeeze=False)

    for col, mode in enumerate(modes):
        ax = axes[0, col]
        for paradigm in paradigms:
            res = all_results[paradigm].get(mode, {})
            lr = res.get("layer_results", [])
            if not lr:
                continue
            layers = [r["layer"] for r in lr if not r.get("skipped")]
            aucs = [r["auc_mean"] for r in lr if not r.get("skipped")]
            stds = [r["auc_std"] for r in lr if not r.get("skipped")]
            if not layers:
                continue
            color = PARADIGM_COLORS[paradigm]
            label = PARADIGM_LABELS[paradigm]
            ax.plot(layers, aucs, "-o", color=color, markersize=3, lw=1.5, label=label)
            ax.fill_between(layers,
                            np.array(aucs) - np.array(stds),
                            np.array(aucs) + np.array(stds),
                            alpha=0.15, color=color)

            # Mark best
            best = res.get("best_layer")
            best_auc = res.get("best_auc")
            if best is not None and best_auc is not None:
                ax.plot(best, best_auc, "*", color=color, markersize=12, zorder=5)
                perm = res.get("permutation_test", {})
                p_val = perm.get("p_value", 1.0) if perm else 1.0
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                ax.annotate(f"L{best}: {best_auc:.3f} ({sig})",
                            xy=(best, best_auc), xytext=(5, 10),
                            textcoords="offset points", fontsize=8, color=color,
                            fontweight="bold")

        ax.axhline(y=0.5, color="gray", ls="--", alpha=0.5, lw=0.8)
        ax.set_xlabel("Layer")
        ax.set_ylabel("AUC (5-fold CV)")
        mode_label = mode.replace("_", " ").title()
        ax.set_title(f"BK Classification — {mode_label}", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim(0.4, 1.05)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_mode_comparison(all_results: dict, save_path: Path):
    """Compare decision_point vs game_mean vs game_max for each paradigm."""
    paradigms = list(all_results.keys())
    n_para = len(paradigms)
    if n_para == 0:
        return

    fig, axes = plt.subplots(1, n_para, figsize=(6 * n_para, 5), squeeze=False)

    for col, paradigm in enumerate(paradigms):
        ax = axes[0, col]
        modes = all_results[paradigm]
        mode_colors = {
            "decision_point": "#e74c3c",
            "game_mean": "#3498db",
            "game_max": "#2ecc71",
        }
        for mode, mode_res in modes.items():
            lr = mode_res.get("layer_results", [])
            if not lr:
                continue
            layers = [r["layer"] for r in lr if not r.get("skipped")]
            aucs = [r["auc_mean"] for r in lr if not r.get("skipped")]
            if not layers:
                continue
            color = mode_colors.get(mode, "gray")
            label = mode.replace("_", " ").title()
            ax.plot(layers, aucs, "-o", color=color, markersize=3, lw=1.5, label=label)

        ax.axhline(y=0.5, color="gray", ls="--", alpha=0.5, lw=0.8)
        ax.set_xlabel("Layer")
        ax.set_ylabel("AUC")
        ax.set_title(f"{PARADIGM_LABELS[paradigm]}", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim(0.4, 1.05)

    fig.suptitle("Aggregation Mode Comparison: Decision-Point vs All-Rounds",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ===================================================================
# Entry point
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="BK Classification (V3 SAE)")
    parser.add_argument("--paradigm", nargs="+", required=True, choices=["ic", "sm", "mw"])
    parser.add_argument("--layers", type=str, default="all",
                        help="'all' or comma-separated: '18,26,30'")
    parser.add_argument("--modes", nargs="+",
                        default=["decision_point", "game_mean", "game_max"],
                        choices=["decision_point", "game_mean", "game_max"])
    parser.add_argument("--skip-permutation", action="store_true",
                        help="Skip permutation test (faster)")
    args = parser.parse_args()

    if args.layers == "all":
        layers = list(range(N_LAYERS))
    else:
        layers = [int(x.strip()) for x in args.layers.split(",")]

    logger = setup_logging(args.paradigm)

    # Check readiness
    for p in args.paradigm:
        status = check_paradigm_ready(p)
        logger.info(f"  {p}: {status['n_layers']}/42 layers {'READY' if status['complete'] else 'INCOMPLETE'}")
        if not status["complete"]:
            logger.warning(f"  {p} extraction incomplete — running with available layers")

    skip_perm = args.skip_permutation

    # Run analysis
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    JSON_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for p in args.paradigm:
        all_results[p] = analyze_paradigm(p, layers, args.modes, logger, skip_perm=skip_perm)

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "_".join(args.paradigm)
    result_path = JSON_DIR / f"classify_bk_{tag}_{ts}.json"
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    logger.info(f"\nResults saved: {result_path}")

    # Generate figures
    logger.info("\nGenerating figures...")
    plot_auc_curves(all_results, FIGURE_DIR / f"bk_auc_curves_{tag}.png")
    if len(args.modes) > 1:
        plot_mode_comparison(all_results, FIGURE_DIR / f"bk_mode_comparison_{tag}.png")

    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("SUMMARY")
    logger.info(f"{'='*70}")
    for p in args.paradigm:
        for mode, mode_res in all_results[p].items():
            best = mode_res.get("best_layer")
            auc = mode_res.get("best_auc")
            perm = mode_res.get("permutation_test", {})
            p_val = perm.get("p_value", "N/A") if perm else "N/A"
            logger.info(f"  {PARADIGM_LABELS[p]} [{mode}]: "
                        f"Best L{best}, AUC={auc:.4f}, p={p_val}")


if __name__ == "__main__":
    main()
