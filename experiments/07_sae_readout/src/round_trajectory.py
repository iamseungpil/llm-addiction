#!/usr/bin/env python3
"""
Round-Level Trajectory Analysis (V3 SAE Features).

Leverages ALL rounds (not just decision-point) to analyze:
  1. Early prediction: At round N, can we predict eventual BK?
  2. PCA trajectory: How do BK vs Safe games diverge over rounds?
  3. Feature dynamics: Which features change most between early/late rounds?

Usage:
    cd /home/jovyan/llm-addiction/sae_v3_analysis/src
    python round_trajectory.py --paradigm ic --layer 18
    python round_trajectory.py --paradigm ic sm --layer 18 26 30
"""

import sys
import json
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    PARADIGMS, N_LAYERS, FIGURE_DIR, JSON_DIR, LOG_DIR,
    MIN_ACTIVATION_RATE, RANDOM_SEED,
    PARADIGM_COLORS, PARADIGM_LABELS,
)
from data_loader import load_layer_features, filter_active_features, get_labels


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def setup_logging(tag: str) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"round_trajectory_{tag}_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Log: {log_file}")
    return logger


# ===================================================================
# 1. Early Prediction
# ===================================================================
def early_prediction(paradigm: str, layer: int, logger) -> dict:
    """At round N, using features up to round N, predict eventual BK.

    For each cutoff round (1, 2, 3, ..., max_round):
      - Include only games that have at least this many rounds
      - Use the feature vector AT that round
      - Predict whether the game eventually goes bankrupt
    """
    logger.info(f"  Early prediction: {paradigm} L{layer}")

    loaded = load_layer_features(paradigm, layer, mode="all_rounds", dense=True)
    if loaded is None:
        return {"error": "layer not found"}

    features, meta = loaded
    labels_all = get_labels(meta)
    game_ids = meta["game_ids"]
    round_nums = meta["round_nums"]

    # Build per-game outcome lookup
    unique_games = np.unique(game_ids)
    game_outcome = {}
    for gid in unique_games:
        gmask = game_ids == gid
        game_outcome[gid] = int(labels_all[gmask][0])  # same for all rounds

    max_round = int(round_nums.max())

    results_by_round = []
    for cutoff in range(1, min(max_round + 1, 51)):  # up to round 50
        # For each game, get feature at round=cutoff (if it exists)
        game_features = []
        game_labels = []

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
        n_safe = len(y) - n_bk

        if n_bk < 3 or n_safe < 3:
            continue

        # Filter active features
        rate = (X != 0).mean(axis=0)
        active = rate >= MIN_ACTIVATION_RATE
        if active.sum() < 10:
            continue
        X_filtered = X[:, active]

        # Quick 3-fold CV (less data per round)
        n_folds = min(5, n_bk, n_safe)
        if n_folds < 2:
            continue

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
        aucs = []
        for train_idx, test_idx in skf.split(X_filtered, y):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_filtered[train_idx])
            X_test = scaler.transform(X_filtered[test_idx])
            clf = LogisticRegression(
                C=1.0, solver="lbfgs", max_iter=500,
                class_weight="balanced", random_state=RANDOM_SEED,
            )
            clf.fit(X_train, y[train_idx])
            try:
                aucs.append(roc_auc_score(y[test_idx], clf.predict_proba(X_test)[:, 1]))
            except ValueError:
                aucs.append(0.5)

        results_by_round.append({
            "round": cutoff,
            "n_games": len(y),
            "n_bk": n_bk,
            "n_safe": n_safe,
            "auc_mean": float(np.mean(aucs)),
            "auc_std": float(np.std(aucs)),
            "n_active_features": int(active.sum()),
        })

        logger.info(f"    Round {cutoff:2d}: {len(y)} games ({n_bk} BK), "
                    f"AUC={np.mean(aucs):.4f}")

    return {
        "paradigm": paradigm,
        "layer": layer,
        "rounds": results_by_round,
    }


# ===================================================================
# 2. PCA Trajectory
# ===================================================================
def pca_trajectory(paradigm: str, layer: int, logger) -> dict:
    """PCA on all-round features, colored by BK/Safe and round number."""
    logger.info(f"  PCA trajectory: {paradigm} L{layer}")

    loaded = load_layer_features(paradigm, layer, mode="all_rounds", dense=True)
    if loaded is None:
        return {"error": "layer not found"}

    features, meta = loaded
    labels = get_labels(meta)
    game_ids = meta["game_ids"]
    round_nums = meta["round_nums"]

    # Filter active features
    filtered, active_idx = filter_active_features(features, MIN_ACTIVATION_RATE)

    # Subsample if too many rounds (for PCA speed)
    n_rounds = filtered.shape[0]
    if n_rounds > 10000:
        np.random.seed(RANDOM_SEED)
        idx = np.random.choice(n_rounds, 10000, replace=False)
        idx.sort()
        filtered = filtered[idx]
        labels = labels[idx]
        game_ids = game_ids[idx]
        round_nums = round_nums[idx]

    # PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(filtered)
    pca = PCA(n_components=3, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_scaled)

    # Compute mean trajectory per outcome
    max_round = min(int(round_nums.max()), 30)
    bk_traj = []
    safe_traj = []
    for r in range(1, max_round + 1):
        bk_mask = (labels == 1) & (round_nums == r)
        safe_mask = (labels == 0) & (round_nums == r)
        if bk_mask.sum() > 0:
            bk_traj.append(X_pca[bk_mask].mean(axis=0).tolist())
        if safe_mask.sum() > 0:
            safe_traj.append(X_pca[safe_mask].mean(axis=0).tolist())

    return {
        "paradigm": paradigm,
        "layer": layer,
        "explained_variance": pca.explained_variance_ratio_.tolist(),
        "bk_trajectory": bk_traj,
        "safe_trajectory": safe_traj,
        "max_round": max_round,
        "n_rounds_used": len(labels),
    }


# ===================================================================
# 3. Feature Dynamics
# ===================================================================
def feature_dynamics(paradigm: str, layer: int, logger) -> dict:
    """Find features whose activation changes most between early and late rounds."""
    logger.info(f"  Feature dynamics: {paradigm} L{layer}")

    loaded = load_layer_features(paradigm, layer, mode="all_rounds", dense=True)
    if loaded is None:
        return {"error": "layer not found"}

    features, meta = loaded
    labels = get_labels(meta)
    round_nums = meta["round_nums"]

    max_round = int(round_nums.max())
    early_cutoff = max(1, max_round // 4)  # first quarter
    late_cutoff = max_round - early_cutoff  # last quarter

    early_mask = round_nums <= early_cutoff
    late_mask = round_nums >= late_cutoff

    # Separate by outcome
    results = {}
    for outcome, outcome_label in [(1, "bk"), (0, "safe")]:
        omask = labels == outcome
        early_feat = features[early_mask & omask]
        late_feat = features[late_mask & omask]

        if early_feat.shape[0] < 10 or late_feat.shape[0] < 10:
            results[outcome_label] = {"error": "insufficient data"}
            continue

        # Mean activation difference (late - early) per feature
        early_mean = early_feat.mean(axis=0)
        late_mean = late_feat.mean(axis=0)
        diff = late_mean - early_mean

        # Find features with largest absolute change
        top_k = 50
        top_idx = np.argsort(np.abs(diff))[-top_k:][::-1]

        results[outcome_label] = {
            "n_early": int(early_feat.shape[0]),
            "n_late": int(late_feat.shape[0]),
            "early_cutoff": int(early_cutoff),
            "late_cutoff": int(late_cutoff),
            "top_changing_features": [
                {
                    "feature_idx": int(idx),
                    "early_mean": float(early_mean[idx]),
                    "late_mean": float(late_mean[idx]),
                    "diff": float(diff[idx]),
                }
                for idx in top_idx
            ],
        }

        logger.info(f"    {outcome_label}: {early_feat.shape[0]} early, "
                    f"{late_feat.shape[0]} late rounds")

    return {
        "paradigm": paradigm,
        "layer": layer,
        "dynamics": results,
    }


# ===================================================================
# Visualization
# ===================================================================
def plot_early_prediction(results: list, save_path: Path):
    """Plot AUC vs round number for early prediction."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for res in results:
        if "error" in res:
            continue
        rounds_data = res["rounds"]
        if not rounds_data:
            continue
        p = res["paradigm"]
        layer = res["layer"]
        x = [r["round"] for r in rounds_data]
        y = [r["auc_mean"] for r in rounds_data]
        yerr = [r["auc_std"] for r in rounds_data]
        color = PARADIGM_COLORS.get(p, "gray")
        ax.plot(x, y, "-o", color=color, markersize=3, lw=1.5,
                label=f"{PARADIGM_LABELS[p]} L{layer}")
        ax.fill_between(x,
                        np.array(y) - np.array(yerr),
                        np.array(y) + np.array(yerr),
                        alpha=0.1, color=color)

    ax.axhline(y=0.5, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Round Number", fontsize=11)
    ax.set_ylabel("BK Prediction AUC", fontsize=11)
    ax.set_title("Early BK Prediction: How Soon Can We Tell?", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(0.35, 1.05)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_pca_trajectory(results: list, save_path: Path):
    """Plot PCA trajectories for BK vs Safe games."""
    n = len([r for r in results if "error" not in r])
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)

    plot_idx = 0
    for res in results:
        if "error" in res:
            continue
        ax = axes[0, plot_idx]
        p = res["paradigm"]
        layer = res["layer"]

        bk_traj = np.array(res["bk_trajectory"])
        safe_traj = np.array(res["safe_trajectory"])

        if bk_traj.shape[0] > 0:
            ax.plot(bk_traj[:, 0], bk_traj[:, 1], "o-", color="#e74c3c",
                    markersize=3, lw=1.5, label="BK games", alpha=0.8)
            ax.plot(bk_traj[0, 0], bk_traj[0, 1], "s", color="#e74c3c",
                    markersize=10, zorder=5)
            ax.plot(bk_traj[-1, 0], bk_traj[-1, 1], "*", color="#e74c3c",
                    markersize=15, zorder=5)

        if safe_traj.shape[0] > 0:
            ax.plot(safe_traj[:, 0], safe_traj[:, 1], "o-", color="#3498db",
                    markersize=3, lw=1.5, label="Safe games", alpha=0.8)
            ax.plot(safe_traj[0, 0], safe_traj[0, 1], "s", color="#3498db",
                    markersize=10, zorder=5)
            ax.plot(safe_traj[-1, 0], safe_traj[-1, 1], "*", color="#3498db",
                    markersize=15, zorder=5)

        ev = res["explained_variance"]
        ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)", fontsize=10)
        ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)", fontsize=10)
        ax.set_title(f"{PARADIGM_LABELS[p]} L{layer}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        plot_idx += 1

    fig.suptitle("PCA Trajectory: BK vs Safe Games Over Rounds",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ===================================================================
# Entry point
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="Round-Level Trajectory Analysis (V3 SAE)")
    parser.add_argument("--paradigm", nargs="+", required=True, choices=["ic", "sm", "mw"])
    parser.add_argument("--layer", nargs="+", type=int, required=True,
                        help="Layer(s) to analyze, e.g. 18 26 30")
    args = parser.parse_args()

    tag = "_".join(args.paradigm)
    logger = setup_logging(tag)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    JSON_DIR.mkdir(parents=True, exist_ok=True)

    all_early = []
    all_pca = []
    all_dynamics = []

    for p in args.paradigm:
        for layer in args.layer:
            early = early_prediction(p, layer, logger)
            pca = pca_trajectory(p, layer, logger)
            dynamics = feature_dynamics(p, layer, logger)

            all_early.append(early)
            all_pca.append(pca)
            all_dynamics.append(dynamics)

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    layers_tag = "_".join(str(l) for l in args.layer)
    result_path = JSON_DIR / f"round_trajectory_{tag}_L{layers_tag}_{ts}.json"
    with open(result_path, "w") as f:
        json.dump({
            "early_prediction": all_early,
            "pca_trajectory": all_pca,
            "feature_dynamics": all_dynamics,
        }, f, indent=2, cls=NumpyEncoder)
    logger.info(f"\nResults saved: {result_path}")

    # Figures
    plot_early_prediction(all_early, FIGURE_DIR / f"early_prediction_{tag}_L{layers_tag}.png")
    plot_pca_trajectory(all_pca, FIGURE_DIR / f"pca_trajectory_{tag}_L{layers_tag}.png")

    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("ROUND TRAJECTORY SUMMARY")
    logger.info(f"{'='*70}")
    for ep in all_early:
        if "error" in ep:
            continue
        rounds_data = ep["rounds"]
        if rounds_data:
            r1 = rounds_data[0]
            r_best = max(rounds_data, key=lambda r: r["auc_mean"])
            logger.info(f"  {PARADIGM_LABELS[ep['paradigm']]} L{ep['layer']}: "
                        f"Round 1 AUC={r1['auc_mean']:.4f}, "
                        f"Best Round {r_best['round']} AUC={r_best['auc_mean']:.4f}")


if __name__ == "__main__":
    main()
