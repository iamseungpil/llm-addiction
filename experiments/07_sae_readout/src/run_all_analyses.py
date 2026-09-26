#!/usr/bin/env python3
"""
Run ALL V3 SAE analyses (14 experiments).

Goal A: BK classification (SAE feature + hidden state, 3 paradigms)
Goal B: Early prediction + PCA trajectory
Goal C: Cross-domain transfer (6 pairs)
Goal D: Feature vs Hidden State comparison

Usage:
    cd /home/jovyan/llm-addiction/sae_v3_analysis/src
    python run_all_analyses.py
"""

import sys
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from itertools import permutations

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

from config import (
    PARADIGMS, N_LAYERS, FIGURE_DIR, JSON_DIR, LOG_DIR,
    MIN_ACTIVATION_RATE, RANDOM_SEED,
    PARADIGM_COLORS, PARADIGM_LABELS,
)
from data_loader import (
    load_layer_features, load_hidden_states,
    filter_active_features, get_labels, check_paradigm_ready,
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


# ===================================================================
# Shared classification function
# ===================================================================
def classify_cv(features, labels, n_folds=5):
    """5-fold stratified CV. Returns detailed metrics dict."""
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos

    if n_pos < n_folds or n_neg < n_folds:
        return {"auc": 0.5, "f1": 0.0, "n_pos": n_pos, "n_neg": n_neg, "skipped": True}

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
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


# ===================================================================
# Goal A: BK Classification (SAE + Hidden State)
# ===================================================================
def run_goal_a(logger):
    """A1-A6: BK classification for all paradigms, SAE + Hidden State."""
    logger.info("\n" + "=" * 70)
    logger.info("GOAL A: BK Classification (SAE Feature + Hidden State)")
    logger.info("=" * 70)

    results = {}
    key_layers = [0, 5, 10, 15, 18, 20, 22, 25, 28, 30, 33, 35, 38, 40]

    for paradigm in ["ic", "sm", "mw"]:
        results[paradigm] = {"sae": [], "hidden": []}
        cfg = PARADIGMS[paradigm]
        logger.info(f"\n--- {cfg['name']} ---")

        for layer in range(N_LAYERS):
            # SAE Feature
            loaded = load_layer_features(paradigm, layer, mode="decision_point", dense=True)
            if loaded is not None:
                feat, meta = loaded
                labels = get_labels(meta)
                filtered, _ = filter_active_features(feat, MIN_ACTIVATION_RATE)
                res = classify_cv(filtered, labels)
                res["layer"] = layer
                res["n_features"] = filtered.shape[1]
                results[paradigm]["sae"].append(res)

                if layer in key_layers:
                    logger.info(f"  L{layer:02d} SAE:    AUC={res['auc']:.4f}, F1={res['f1']:.3f}, "
                                f"n_feat={filtered.shape[1]}, BK={res['n_pos']}")

            # Hidden State (only key layers to save time — 3584 dim is expensive)
            if layer in key_layers:
                h_loaded = load_hidden_states(paradigm, layer, mode="decision_point")
                if h_loaded is not None:
                    h_feat, h_meta = h_loaded
                    h_labels = get_labels(h_meta)
                    h_res = classify_cv(h_feat, h_labels)
                    h_res["layer"] = layer
                    h_res["n_features"] = h_feat.shape[1]
                    results[paradigm]["hidden"].append(h_res)
                    logger.info(f"  L{layer:02d} Hidden: AUC={h_res['auc']:.4f}, F1={h_res['f1']:.3f}, "
                                f"dim={h_feat.shape[1]}")

        # Summary
        sae_best = max(results[paradigm]["sae"], key=lambda r: r["auc"]) if results[paradigm]["sae"] else None
        hid_best = max(results[paradigm]["hidden"], key=lambda r: r["auc"]) if results[paradigm]["hidden"] else None
        if sae_best:
            logger.info(f"  BEST SAE:    L{sae_best['layer']} AUC={sae_best['auc']:.4f}")
        if hid_best:
            logger.info(f"  BEST Hidden: L{hid_best['layer']} AUC={hid_best['auc']:.4f}")

    return results


# ===================================================================
# Goal B: Early Prediction
# ===================================================================
def run_goal_b(logger):
    """B1-B3: Early prediction for all paradigms."""
    logger.info("\n" + "=" * 70)
    logger.info("GOAL B: Early Prediction (Round-by-Round)")
    logger.info("=" * 70)

    results = {}
    # Use best layer per paradigm from Goal A (approximate: L22 for IC, try L22 for all)
    best_layers = {"ic": 22, "sm": 22, "mw": 22}

    for paradigm in ["ic", "sm", "mw"]:
        layer = best_layers[paradigm]
        logger.info(f"\n--- {PARADIGMS[paradigm]['name']} L{layer} ---")

        loaded = load_layer_features(paradigm, layer, mode="all_rounds", dense=True)
        if loaded is None:
            logger.warning(f"  {paradigm} L{layer} not found")
            continue

        features, meta = loaded
        labels = get_labels(meta)
        game_ids = meta["game_ids"]
        round_nums = meta["round_nums"]

        unique_games = np.unique(game_ids)
        game_outcome = {}
        for gid in unique_games:
            gmask = game_ids == gid
            game_outcome[gid] = int(labels[gmask][0])

        max_round = int(round_nums.max())
        round_results = []

        for cutoff in range(1, min(max_round + 1, 51)):
            game_features, game_labels = [], []
            for gid in unique_games:
                gmask = (game_ids == gid) & (round_nums == cutoff)
                if gmask.sum() == 0:
                    continue
                idx = np.where(gmask)[0][0]
                game_features.append(features[idx])
                game_labels.append(game_outcome[gid])

            if len(game_features) < 20:
                continue

            X = np.array(game_features)
            y = np.array(game_labels)
            n_bk = int(y.sum())
            if n_bk < 3 or (len(y) - n_bk) < 3:
                continue

            rate = (X != 0).mean(axis=0)
            active = rate >= MIN_ACTIVATION_RATE
            if active.sum() < 10:
                continue

            res = classify_cv(X[:, active], y, n_folds=min(5, n_bk, len(y) - n_bk))
            if not res["skipped"]:
                round_results.append({
                    "round": cutoff, "auc": res["auc"],
                    "n_games": len(y), "n_bk": n_bk,
                })
                if cutoff <= 5 or cutoff % 5 == 0:
                    logger.info(f"  Round {cutoff:2d}: AUC={res['auc']:.4f}, "
                                f"n={len(y)} ({n_bk} BK)")

        results[paradigm] = {"layer": layer, "rounds": round_results}

    return results


# ===================================================================
# Goal C: Cross-Domain Transfer
# ===================================================================
def run_goal_c(logger):
    """C1-C6: Cross-domain transfer between all paradigm pairs."""
    logger.info("\n" + "=" * 70)
    logger.info("GOAL C: Cross-Domain Transfer")
    logger.info("=" * 70)

    results = {}
    key_layers = list(range(0, N_LAYERS, 2))  # every other layer for speed

    for src, tgt in permutations(["ic", "sm", "mw"], 2):
        key = f"{src}_to_{tgt}"
        logger.info(f"\n--- {PARADIGM_LABELS[src]} -> {PARADIGM_LABELS[tgt]} ---")

        pair_results = []
        for layer in key_layers:
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

            # Train on source
            n_pos = int(src_labels.sum())
            n_neg = len(src_labels) - n_pos
            if n_pos < 2 or n_neg < 2:
                continue
            if int(tgt_labels.sum()) == 0 or int(tgt_labels.sum()) == len(tgt_labels):
                continue

            scaler = StandardScaler()
            X_train = scaler.fit_transform(src_f)
            X_test = scaler.transform(tgt_f)

            clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000,
                                     class_weight="balanced", random_state=RANDOM_SEED)
            clf.fit(X_train, src_labels)
            try:
                xfer_auc = float(roc_auc_score(tgt_labels, clf.predict_proba(X_test)[:, 1]))
            except ValueError:
                xfer_auc = 0.5

            pair_results.append({
                "layer": layer, "transfer_auc": xfer_auc,
                "n_shared_features": n_active,
                "src_n_bk": n_pos, "tgt_n_bk": int(tgt_labels.sum()),
            })

        if pair_results:
            best = max(pair_results, key=lambda r: r["transfer_auc"])
            logger.info(f"  Best: L{best['layer']} Transfer AUC={best['transfer_auc']:.4f}")

        results[key] = pair_results

    return results


# ===================================================================
# Visualization
# ===================================================================
def plot_goal_a(results, save_path):
    """AUC curves: SAE vs Hidden State, all paradigms."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for col, paradigm in enumerate(["ic", "sm", "mw"]):
        ax = axes[col]
        data = results.get(paradigm, {})

        # SAE
        sae = data.get("sae", [])
        if sae:
            layers = [r["layer"] for r in sae if not r.get("skipped")]
            aucs = [r["auc"] for r in sae if not r.get("skipped")]
            ax.plot(layers, aucs, "-o", color="#e74c3c", markersize=2, lw=1.5, label="SAE Feature")
            best_sae = max(sae, key=lambda r: r["auc"])
            ax.plot(best_sae["layer"], best_sae["auc"], "*", color="#e74c3c", markersize=12, zorder=5)

        # Hidden
        hid = data.get("hidden", [])
        if hid:
            layers_h = [r["layer"] for r in hid if not r.get("skipped")]
            aucs_h = [r["auc"] for r in hid if not r.get("skipped")]
            ax.plot(layers_h, aucs_h, "-s", color="#3498db", markersize=5, lw=1.5, label="Hidden State")
            best_hid = max(hid, key=lambda r: r["auc"])
            ax.plot(best_hid["layer"], best_hid["auc"], "*", color="#3498db", markersize=12, zorder=5)

        ax.axhline(y=0.5, color="gray", ls="--", alpha=0.5)
        ax.set_xlabel("Layer")
        ax.set_ylabel("AUC (5-fold CV)")
        n_bk = PARADIGMS[paradigm]["n_bk"]
        n_games = PARADIGMS[paradigm]["n_games"]
        ax.set_title(f"{PARADIGM_LABELS[paradigm]}\n({n_games} games, {n_bk} BK)", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(0.4, 1.05)

    fig.suptitle("Goal A: BK Classification — SAE Feature vs Hidden State",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_goal_b(results, save_path):
    """Early prediction curves for all paradigms."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for paradigm in ["ic", "sm", "mw"]:
        data = results.get(paradigm, {})
        rounds_data = data.get("rounds", [])
        if not rounds_data:
            continue
        x = [r["round"] for r in rounds_data]
        y = [r["auc"] for r in rounds_data]
        color = PARADIGM_COLORS[paradigm]
        ax.plot(x, y, "-o", color=color, markersize=3, lw=1.5,
                label=f"{PARADIGM_LABELS[paradigm]} (L{data['layer']})")

    ax.axhline(y=0.5, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Round Number", fontsize=11)
    ax.set_ylabel("BK Prediction AUC", fontsize=11)
    ax.set_title("Goal B: Early BK Prediction — How Soon Can We Tell?",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0.35, 1.05)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_goal_c(results, save_path):
    """Cross-domain transfer heatmap."""
    paradigms = ["ic", "sm", "mw"]
    n = len(paradigms)
    matrix = np.full((n, n), np.nan)

    for i, src in enumerate(paradigms):
        for j, tgt in enumerate(paradigms):
            if i == j:
                continue
            key = f"{src}_to_{tgt}"
            pair = results.get(key, [])
            if pair:
                best = max(pair, key=lambda r: r["transfer_auc"])
                matrix[i, j] = best["transfer_auc"]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")

    labels = [PARADIGM_LABELS[p] for p in paradigms]
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
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                        fontsize=14, color=color, fontweight="bold")
            elif i == j:
                ax.text(j, i, "—", ha="center", va="center", fontsize=14, color="gray")

    plt.colorbar(im, ax=ax, label="Transfer AUC", shrink=0.8)
    ax.set_title("Goal C: Cross-Domain BK Prediction Transfer",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_goal_c_curves(results, save_path):
    """Layer-wise transfer AUC curves."""
    pairs = [(s, t) for s, t in permutations(["ic", "sm", "mw"], 2)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for idx, (src, tgt) in enumerate(pairs):
        row, col = divmod(idx, 3)
        ax = axes[row, col]
        key = f"{src}_to_{tgt}"
        pair_data = results.get(key, [])
        if pair_data:
            layers = [r["layer"] for r in pair_data]
            aucs = [r["transfer_auc"] for r in pair_data]
            ax.plot(layers, aucs, "-o", color="#e74c3c", markersize=3, lw=1.5)
        ax.axhline(y=0.5, color="gray", ls="--", alpha=0.5)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Transfer AUC")
        ax.set_title(f"{PARADIGM_LABELS[src]} -> {PARADIGM_LABELS[tgt]}", fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_ylim(0.3, 1.05)

    fig.suptitle("Goal C: Cross-Domain Transfer — Layer-wise AUC",
                 fontsize=13, fontweight="bold")
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
    log_file = LOG_DIR / f"run_all_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Log: {log_file}")

    # Check readiness
    for p in ["ic", "sm", "mw"]:
        status = check_paradigm_ready(p)
        logger.info(f"  {p}: {status['n_layers']}/42 layers {'READY' if status['complete'] else 'INCOMPLETE'}")

    # Run all goals
    goal_a = run_goal_a(logger)
    goal_b = run_goal_b(logger)
    goal_c = run_goal_c(logger)

    # Save results
    all_results = {
        "goal_a_classification": goal_a,
        "goal_b_early_prediction": goal_b,
        "goal_c_cross_domain": goal_c,
        "timestamp": ts,
    }
    result_path = JSON_DIR / f"all_analyses_{ts}.json"
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    logger.info(f"\nAll results saved: {result_path}")

    # Generate figures
    logger.info("\nGenerating figures...")
    plot_goal_a(goal_a, FIGURE_DIR / "goal_a_sae_vs_hidden.png")
    plot_goal_b(goal_b, FIGURE_DIR / "goal_b_early_prediction.png")
    plot_goal_c(goal_c, FIGURE_DIR / "goal_c_transfer_matrix.png")
    plot_goal_c_curves(goal_c, FIGURE_DIR / "goal_c_transfer_curves.png")

    # Final summary
    logger.info(f"\n{'='*70}")
    logger.info("FINAL SUMMARY")
    logger.info(f"{'='*70}")

    for p in ["ic", "sm", "mw"]:
        sae_data = goal_a.get(p, {}).get("sae", [])
        hid_data = goal_a.get(p, {}).get("hidden", [])
        sae_best = max(sae_data, key=lambda r: r["auc"]) if sae_data else None
        hid_best = max(hid_data, key=lambda r: r["auc"]) if hid_data else None

        logger.info(f"\n  {PARADIGM_LABELS[p]}:")
        if sae_best:
            logger.info(f"    SAE Best:    L{sae_best['layer']} AUC={sae_best['auc']:.4f} "
                        f"F1={sae_best['f1']:.3f}")
        if hid_best:
            logger.info(f"    Hidden Best: L{hid_best['layer']} AUC={hid_best['auc']:.4f} "
                        f"F1={hid_best['f1']:.3f}")

        early = goal_b.get(p, {}).get("rounds", [])
        if early:
            r1 = early[0]
            logger.info(f"    Early R1:    AUC={r1['auc']:.4f} ({r1['n_games']} games)")

    logger.info(f"\n  Cross-Domain Transfer (best per pair):")
    for key, pair_data in goal_c.items():
        if pair_data:
            best = max(pair_data, key=lambda r: r["transfer_auc"])
            logger.info(f"    {key}: L{best['layer']} AUC={best['transfer_auc']:.4f}")

    logger.info(f"\n{'='*70}")
    logger.info(f"Figures: {FIGURE_DIR}")
    logger.info(f"JSON:    {result_path}")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()
