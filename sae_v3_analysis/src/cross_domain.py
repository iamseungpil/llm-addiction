#!/usr/bin/env python3
"""
Cross-Domain Transfer Analysis (V3 SAE Features).

Tests whether BK-discriminative features generalize across paradigms:
  - Train on IC -> Test on SM, MW
  - Train on SM -> Test on IC, MW
  - Train on MW -> Test on IC, SM

For each transfer pair:
  - Train logistic regression on source paradigm
  - Evaluate AUC on target paradigm (zero-shot transfer)
  - Compare with within-domain AUC (upper bound)
  - Layer-wise transfer AUC curves

Usage:
    cd /home/jovyan/llm-addiction/sae_v3_analysis/src
    python cross_domain.py                          # all pairs
    python cross_domain.py --source ic --target sm   # specific pair
    python cross_domain.py --mode game_mean          # aggregation mode
"""

import sys
import json
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from itertools import permutations

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    PARADIGMS, N_LAYERS, FIGURE_DIR, JSON_DIR, LOG_DIR,
    MIN_ACTIVATION_RATE, CLASSIFICATION_C, RANDOM_SEED,
    PARADIGM_COLORS, PARADIGM_LABELS,
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


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"cross_domain_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Log: {log_file}")
    return logger


def within_domain_auc(features, labels, n_folds=5):
    """Within-domain 5-fold CV AUC (baseline upper bound)."""
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos < n_folds or n_neg < n_folds:
        return 0.5

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    aucs = []
    for train_idx, test_idx in skf.split(features, labels):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(features[train_idx])
        X_test = scaler.transform(features[test_idx])
        clf = LogisticRegression(
            C=CLASSIFICATION_C, solver="lbfgs", max_iter=1000,
            class_weight="balanced", random_state=RANDOM_SEED,
        )
        clf.fit(X_train, labels[train_idx])
        try:
            aucs.append(roc_auc_score(labels[test_idx], clf.predict_proba(X_test)[:, 1]))
        except ValueError:
            aucs.append(0.5)
    return float(np.mean(aucs))


def transfer_auc(src_features, src_labels, tgt_features, tgt_labels):
    """Train on source, test on target. Returns AUC."""
    n_pos = int(src_labels.sum())
    n_neg = len(src_labels) - n_pos
    if n_pos < 2 or n_neg < 2:
        return 0.5
    if int(tgt_labels.sum()) == 0 or int(tgt_labels.sum()) == len(tgt_labels):
        return 0.5

    scaler = StandardScaler()
    X_train = scaler.fit_transform(src_features)
    X_test = scaler.transform(tgt_features)

    clf = LogisticRegression(
        C=CLASSIFICATION_C, solver="lbfgs", max_iter=1000,
        class_weight="balanced", random_state=RANDOM_SEED,
    )
    clf.fit(X_train, src_labels)
    try:
        return float(roc_auc_score(tgt_labels, clf.predict_proba(X_test)[:, 1]))
    except ValueError:
        return 0.5


def analyze_transfer_pair(source: str, target: str, layers: list, mode: str, logger):
    """Run cross-domain transfer for one source->target pair."""
    logger.info(f"\n  {PARADIGM_LABELS[source]} -> {PARADIGM_LABELS[target]}")

    results = []
    for layer in layers:
        src_loaded = load_layer_features(source, layer, mode=mode, dense=True)
        tgt_loaded = load_layer_features(target, layer, mode=mode, dense=True)

        if src_loaded is None or tgt_loaded is None:
            continue

        src_feat, src_meta = src_loaded
        tgt_feat, tgt_meta = tgt_loaded
        src_labels = get_labels(src_meta)
        tgt_labels = get_labels(tgt_meta)

        # Use intersection of active features from both domains
        src_rate = (src_feat != 0).mean(axis=0)
        tgt_rate = (tgt_feat != 0).mean(axis=0)
        active_mask = (src_rate >= MIN_ACTIVATION_RATE) & (tgt_rate >= MIN_ACTIVATION_RATE)
        n_active = int(active_mask.sum())

        if n_active < 10:
            logger.warning(f"    L{layer}: only {n_active} shared active features, skipping")
            continue

        src_filtered = src_feat[:, active_mask]
        tgt_filtered = tgt_feat[:, active_mask]

        # Within-domain AUCs
        src_within = within_domain_auc(src_filtered, src_labels)
        tgt_within = within_domain_auc(tgt_filtered, tgt_labels)

        # Cross-domain AUC
        xfer_auc = transfer_auc(src_filtered, src_labels, tgt_filtered, tgt_labels)

        results.append({
            "layer": layer,
            "n_shared_features": n_active,
            "source_within_auc": src_within,
            "target_within_auc": tgt_within,
            "transfer_auc": xfer_auc,
            "transfer_gap": tgt_within - xfer_auc,
            "source_n_bk": int(src_labels.sum()),
            "target_n_bk": int(tgt_labels.sum()),
        })

        logger.info(f"    L{layer:02d}: transfer={xfer_auc:.4f}, "
                    f"src_within={src_within:.4f}, tgt_within={tgt_within:.4f}, "
                    f"gap={tgt_within - xfer_auc:.4f}, n_feat={n_active}")

    best = max(results, key=lambda r: r["transfer_auc"]) if results else None
    return {
        "source": source,
        "target": target,
        "mode": mode,
        "layer_results": results,
        "best_transfer_layer": best["layer"] if best else None,
        "best_transfer_auc": best["transfer_auc"] if best else None,
    }


def plot_transfer_matrix(all_results: dict, save_path: Path):
    """Heatmap of best transfer AUC between paradigm pairs."""
    paradigms = sorted(set(
        [r["source"] for r in all_results.values()]
        + [r["target"] for r in all_results.values()]
    ))
    n = len(paradigms)
    matrix = np.full((n, n), np.nan)

    for key, res in all_results.items():
        src_idx = paradigms.index(res["source"])
        tgt_idx = paradigms.index(res["target"])
        if res["best_transfer_auc"] is not None:
            matrix[src_idx, tgt_idx] = res["best_transfer_auc"]

    # Diagonal = within-domain best AUC
    for key, res in all_results.items():
        src_idx = paradigms.index(res["source"])
        best_within = max(
            (r["source_within_auc"] for r in res["layer_results"]),
            default=0.5
        )
        matrix[src_idx, src_idx] = best_within

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")

    labels = [PARADIGM_LABELS.get(p, p) for p in paradigms]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Target (Test)", fontsize=11)
    ax.set_ylabel("Source (Train)", fontsize=11)

    for i in range(n):
        for j in range(n):
            if not np.isnan(matrix[i, j]):
                color = "white" if matrix[i, j] < 0.6 else "black"
                style = "bold" if i == j else "normal"
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                        fontsize=11, color=color, fontweight=style)

    plt.colorbar(im, ax=ax, label="AUC", shrink=0.8)
    ax.set_title("Cross-Domain BK Prediction Transfer", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_transfer_curves(all_results: dict, save_path: Path):
    """Layer-wise transfer AUC curves for all pairs."""
    pairs = list(all_results.values())
    n_pairs = len(pairs)
    if n_pairs == 0:
        return

    ncols = min(3, n_pairs)
    nrows = (n_pairs + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for idx, pair_res in enumerate(pairs):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        lr = pair_res["layer_results"]
        if not lr:
            continue

        layers = [r["layer"] for r in lr]
        transfer = [r["transfer_auc"] for r in lr]
        src_within = [r["source_within_auc"] for r in lr]
        tgt_within = [r["target_within_auc"] for r in lr]

        src = pair_res["source"]
        tgt = pair_res["target"]
        ax.plot(layers, transfer, "-o", color="#e74c3c", markersize=3, lw=1.5, label="Transfer")
        ax.plot(layers, tgt_within, "--", color="#3498db", lw=1.0, alpha=0.7, label="Target within")
        ax.plot(layers, src_within, ":", color="#2ecc71", lw=1.0, alpha=0.7, label="Source within")
        ax.axhline(y=0.5, color="gray", ls="--", alpha=0.4)
        ax.set_xlabel("Layer")
        ax.set_ylabel("AUC")
        ax.set_title(f"{PARADIGM_LABELS[src]} -> {PARADIGM_LABELS[tgt]}", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax.set_ylim(0.35, 1.05)

    # Hide unused axes
    for idx in range(n_pairs, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle("Cross-Domain Transfer: Layer-wise AUC", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Cross-Domain Transfer (V3 SAE)")
    parser.add_argument("--source", nargs="*", default=None, choices=["ic", "sm", "mw"])
    parser.add_argument("--target", nargs="*", default=None, choices=["ic", "sm", "mw"])
    parser.add_argument("--mode", type=str, default="decision_point",
                        choices=["decision_point", "game_mean", "game_max"])
    parser.add_argument("--layers", type=str, default="all")
    args = parser.parse_args()

    if args.layers == "all":
        layers = list(range(N_LAYERS))
    else:
        layers = [int(x.strip()) for x in args.layers.split(",")]

    logger = setup_logging()

    # Determine pairs
    available = []
    for p in ["ic", "sm", "mw"]:
        status = check_paradigm_ready(p)
        if status["n_layers"] > 0:
            available.append(p)
            logger.info(f"  {p}: {status['n_layers']}/42 layers")

    if args.source and args.target:
        pairs = [(s, t) for s in args.source for t in args.target if s != t]
    else:
        pairs = [(s, t) for s, t in permutations(available, 2)]

    logger.info(f"\nTransfer pairs: {pairs}")
    logger.info(f"Mode: {args.mode}")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    JSON_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for src, tgt in pairs:
        key = f"{src}_to_{tgt}"
        all_results[key] = analyze_transfer_pair(src, tgt, layers, args.mode, logger)

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = JSON_DIR / f"cross_domain_{args.mode}_{ts}.json"
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    logger.info(f"\nResults saved: {result_path}")

    # Figures
    plot_transfer_matrix(all_results, FIGURE_DIR / f"cross_domain_matrix_{args.mode}.png")
    plot_transfer_curves(all_results, FIGURE_DIR / f"cross_domain_curves_{args.mode}.png")

    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("CROSS-DOMAIN SUMMARY")
    logger.info(f"{'='*70}")
    for key, res in all_results.items():
        best = res.get("best_transfer_auc")
        layer = res.get("best_transfer_layer")
        logger.info(f"  {key}: Best L{layer}, Transfer AUC={best:.4f}" if best else f"  {key}: No results")


if __name__ == "__main__":
    main()
