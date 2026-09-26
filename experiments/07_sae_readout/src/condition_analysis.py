#!/usr/bin/env python3
"""
Condition-Level SAE Analysis (NEW).

Analyses:
  1. BK rate by prompt condition (behavioral baseline)
  2. BK prediction AUC stratified by condition
  3. Prompt component marginal effect on BK rate and AUC
  4. Condition-specific feature importance differences

Usage:
    cd /home/jovyan/llm-addiction/sae_v3_analysis/src
    python condition_analysis.py
"""

import sys
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from itertools import combinations

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    PARADIGMS, FIGURE_DIR, JSON_DIR, LOG_DIR,
    MIN_ACTIVATION_RATE, RANDOM_SEED,
    PARADIGM_COLORS, PARADIGM_LABELS,
)
from data_loader import load_layer_features, filter_active_features, get_labels

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

np.random.seed(RANDOM_SEED)


# ─── Best layers from V3/V4 analysis ───
BEST_LAYERS = {
    "ic": {"dp": 22, "r1": 18},
    "sm": {"dp": 12, "r1": 16},
    "mw": {"dp": 33, "r1": 22},
}

# Known prompt components (CLAUDE.md: G=Goal, M=Mood, H=Hidden patterns (was R), W=Win celebration, P=combined)
ALL_COMPONENTS = ["G", "M", "R", "W", "P"]  # R in data = H in paper


def classify_cv(X, y, n_splits=5):
    """5-fold stratified CV LogisticRegression, returns AUC mean/std."""
    if len(np.unique(y)) < 2 or np.sum(y) < n_splits:
        return {"auc": float("nan"), "auc_std": float("nan"), "n": len(y)}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    aucs = []
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        clf = LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=2000,
            class_weight="balanced", random_state=RANDOM_SEED
        )
        clf.fit(X_train, y[train_idx])
        prob = clf.predict_proba(X_test)[:, 1]
        if len(np.unique(y[test_idx])) == 2:
            aucs.append(roc_auc_score(y[test_idx], prob))
    if not aucs:
        return {"auc": float("nan"), "auc_std": float("nan"), "n": len(y)}
    return {"auc": float(np.mean(aucs)), "auc_std": float(np.std(aucs)), "n": len(y)}


# ═══════════════════════════════════════════════════════════════════
# Analysis 1: BK Rate by Prompt Condition (behavioral)
# ═══════════════════════════════════════════════════════════════════
def analyze_bk_rate_by_condition(paradigm, layer=0):
    """Compute BK rate per prompt condition from metadata."""
    loaded = load_layer_features(paradigm, layer, mode="decision_point", dense=False)
    if loaded is None:
        logger.warning(f"No data for {paradigm} L{layer}")
        return {}

    _, meta = loaded
    labels = get_labels(meta)

    results = {}
    if "prompt_conditions" not in meta:
        logger.warning(f"{paradigm}: no prompt_conditions in metadata")
        return {}

    conditions = meta["prompt_conditions"]
    unique_conds = np.unique(conditions)

    for cond in unique_conds:
        mask = conditions == cond
        n_total = mask.sum()
        n_bk = labels[mask].sum()
        results[str(cond)] = {
            "n_total": int(n_total),
            "n_bk": int(n_bk),
            "bk_rate": float(n_bk / n_total) if n_total > 0 else 0.0,
        }

    # Also by bet_type
    if "bet_types" in meta:
        bet_types = meta["bet_types"]
        for bt in np.unique(bet_types):
            bt_mask = bet_types == bt
            n_total = bt_mask.sum()
            n_bk = labels[bt_mask].sum()
            results[f"bet_{bt}"] = {
                "n_total": int(n_total),
                "n_bk": int(n_bk),
                "bk_rate": float(n_bk / n_total) if n_total > 0 else 0.0,
            }

    return results


# ═══════════════════════════════════════════════════════════════════
# Analysis 2: BK Prediction AUC Stratified by Condition
# ═══════════════════════════════════════════════════════════════════
def analyze_bk_auc_by_condition(paradigm, layer):
    """BK classification AUC separately for each prompt condition."""
    loaded = load_layer_features(paradigm, layer, mode="decision_point", dense=True)
    if loaded is None:
        return {}

    features, meta = loaded
    labels = get_labels(meta)

    # Filter active features
    rate = (features != 0).mean(axis=0)
    active = rate >= MIN_ACTIVATION_RATE
    features = features[:, active]

    results = {"overall": classify_cv(features, labels)}

    if "prompt_conditions" in meta:
        conditions = meta["prompt_conditions"]
        for cond in np.unique(conditions):
            mask = conditions == cond
            n_bk = labels[mask].sum()
            if n_bk < 3:
                results[str(cond)] = {"auc": float("nan"), "n": int(mask.sum()), "n_bk": int(n_bk), "reason": "too_few_bk"}
                continue
            res = classify_cv(features[mask], labels[mask], n_splits=min(5, int(n_bk)))
            res["n_bk"] = int(n_bk)
            results[str(cond)] = res

    if "bet_types" in meta:
        bet_types = meta["bet_types"]
        for bt in np.unique(bet_types):
            mask = bet_types == bt
            n_bk = labels[mask].sum()
            if n_bk < 3:
                results[f"bet_{bt}"] = {"auc": float("nan"), "n": int(mask.sum()), "n_bk": int(n_bk), "reason": "too_few_bk"}
                continue
            res = classify_cv(features[mask], labels[mask], n_splits=min(5, int(n_bk)))
            res["n_bk"] = int(n_bk)
            results[f"bet_{bt}"] = res

    return results


# ═══════════════════════════════════════════════════════════════════
# Analysis 3: Prompt Component Marginal Effect
# ═══════════════════════════════════════════════════════════════════
def analyze_component_marginal(paradigm, layer):
    """For SM/MW with 32 prompt combinations: marginal effect of each component."""
    loaded = load_layer_features(paradigm, layer, mode="decision_point", dense=True)
    if loaded is None:
        return {}

    features, meta = loaded
    labels = get_labels(meta)

    if "prompt_conditions" not in meta:
        return {}

    conditions = meta["prompt_conditions"]

    # Filter active features
    rate = (features != 0).mean(axis=0)
    active = rate >= MIN_ACTIVATION_RATE
    features_filtered = features[:, active]

    results = {}

    for comp in ALL_COMPONENTS:
        # Games WITH this component vs WITHOUT
        has_comp = np.array([comp in str(c) for c in conditions])
        no_comp = ~has_comp

        if has_comp.sum() < 10 or no_comp.sum() < 10:
            continue

        # BK rate comparison
        bk_with = labels[has_comp].mean()
        bk_without = labels[no_comp].mean()

        # AUC comparison (BK prediction within each group)
        auc_with = classify_cv(features_filtered[has_comp], labels[has_comp])
        auc_without = classify_cv(features_filtered[no_comp], labels[no_comp])

        results[comp] = {
            "n_with": int(has_comp.sum()),
            "n_without": int(no_comp.sum()),
            "bk_rate_with": float(bk_with),
            "bk_rate_without": float(bk_without),
            "bk_rate_diff": float(bk_with - bk_without),
            "auc_with": auc_with,
            "auc_without": auc_without,
        }

    return results


# ═══════════════════════════════════════════════════════════════════
# Analysis 4: Condition-Specific Feature Importance
# ═══════════════════════════════════════════════════════════════════
def analyze_condition_features(paradigm, layer, top_k=100):
    """Extract top-K BK-predictive features per condition and compare overlap."""
    loaded = load_layer_features(paradigm, layer, mode="decision_point", dense=True)
    if loaded is None:
        return {}

    features, meta = loaded
    labels = get_labels(meta)

    if "prompt_conditions" not in meta:
        return {}

    conditions = meta["prompt_conditions"]
    unique_conds = [c for c in np.unique(conditions) if c != "BASE"]

    # Filter active features
    rate = (features != 0).mean(axis=0)
    active_mask = rate >= MIN_ACTIVATION_RATE
    active_indices = np.where(active_mask)[0]
    features_filtered = features[:, active_mask]

    # Get top features for overall
    clf_all = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000,
                                 class_weight="balanced", random_state=RANDOM_SEED)
    clf_all.fit(features_filtered, labels)
    importance_all = np.abs(clf_all.coef_[0])
    top_all = set(active_indices[np.argsort(importance_all)[-top_k:]])

    # Get top features per condition
    condition_tops = {"overall": top_all}
    for cond in unique_conds:
        mask = conditions == cond
        n_bk = labels[mask].sum()
        if n_bk < 5:
            continue
        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000,
                                 class_weight="balanced", random_state=RANDOM_SEED)
        clf.fit(features_filtered[mask], labels[mask])
        importance = np.abs(clf.coef_[0])
        top_cond = set(active_indices[np.argsort(importance)[-top_k:]])
        condition_tops[str(cond)] = top_cond

    # Compute pairwise Jaccard
    overlaps = {}
    cond_names = list(condition_tops.keys())
    for i, c1 in enumerate(cond_names):
        for c2 in cond_names[i+1:]:
            s1, s2 = condition_tops[c1], condition_tops[c2]
            jaccard = len(s1 & s2) / len(s1 | s2) if (s1 | s2) else 0
            overlaps[f"{c1}_vs_{c2}"] = {
                "shared": len(s1 & s2),
                "jaccard": float(jaccard),
            }

    return {"top_k": top_k, "overlaps": overlaps}


# ═══════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════
def plot_bk_rate_by_condition(all_results):
    """Bar chart of BK rate by prompt condition for each paradigm."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (paradigm, results) in enumerate(all_results.items()):
        ax = axes[idx]
        # Filter to prompt conditions only (not bet_* entries)
        conds = {k: v for k, v in results.items() if not k.startswith("bet_")}
        if not conds:
            ax.set_title(f"{PARADIGM_LABELS.get(paradigm, paradigm)}: No data")
            continue

        names = sorted(conds.keys())
        rates = [conds[n]["bk_rate"] * 100 for n in names]
        counts = [conds[n]["n_total"] for n in names]

        bars = ax.bar(range(len(names)), rates, color=PARADIGM_COLORS.get(paradigm, "#666"))
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Bankruptcy Rate (%)")
        ax.set_title(f"{PARADIGM_LABELS.get(paradigm, paradigm)}")

        # Add count labels
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"n={count}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Bankruptcy Rate by Prompt Condition", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "v5_bk_rate_by_condition.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {FIGURE_DIR / 'v5_bk_rate_by_condition.png'}")


def plot_auc_by_condition(all_results):
    """Bar chart of BK prediction AUC by condition."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (paradigm, results) in enumerate(all_results.items()):
        ax = axes[idx]
        overall_auc = results.get("overall", {}).get("auc", 0)

        conds = {k: v for k, v in results.items()
                 if k not in ("overall",) and not k.startswith("bet_")
                 and isinstance(v, dict) and not np.isnan(v.get("auc", float("nan")))}

        if not conds:
            ax.set_title(f"{PARADIGM_LABELS.get(paradigm, paradigm)}: No data")
            continue

        names = sorted(conds.keys())
        aucs = [conds[n]["auc"] for n in names]
        stds = [conds[n].get("auc_std", 0) for n in names]

        bars = ax.bar(range(len(names)), aucs, yerr=stds,
                      color=PARADIGM_COLORS.get(paradigm, "#666"), capsize=3)
        ax.axhline(y=overall_auc, color="gray", linestyle="--", label=f"Overall: {overall_auc:.3f}")
        ax.axhline(y=0.5, color="red", linestyle=":", alpha=0.5, label="Chance")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("AUC")
        ax.set_ylim(0.4, 1.05)
        ax.set_title(f"{PARADIGM_LABELS.get(paradigm, paradigm)}")
        ax.legend(fontsize=8)

    fig.suptitle("BK Prediction AUC by Prompt Condition (Decision-Point)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "v5_auc_by_condition.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {FIGURE_DIR / 'v5_auc_by_condition.png'}")


def plot_component_marginal(all_results):
    """Marginal effect of each prompt component on BK rate."""
    paradigms_with_data = {p: r for p, r in all_results.items() if r}
    if not paradigms_with_data:
        logger.warning("No component marginal data to plot")
        return

    fig, axes = plt.subplots(1, len(paradigms_with_data), figsize=(6*len(paradigms_with_data), 5))
    if len(paradigms_with_data) == 1:
        axes = [axes]

    component_labels = {"G": "Goal", "M": "Mood", "R": "Hidden(H)", "W": "Win", "P": "Combined"}

    for idx, (paradigm, results) in enumerate(paradigms_with_data.items()):
        ax = axes[idx]
        comps = sorted(results.keys())
        diffs = [results[c]["bk_rate_diff"] * 100 for c in comps]
        labels = [component_labels.get(c, c) for c in comps]

        colors = ["#e74c3c" if d > 0 else "#3498db" for d in diffs]
        bars = ax.barh(range(len(comps)), diffs, color=colors)
        ax.set_yticks(range(len(comps)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("BK Rate Difference (%p)\n(with component - without)")
        ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.8)
        ax.set_title(f"{PARADIGM_LABELS.get(paradigm, paradigm)}")

        # Add values
        for bar, diff in zip(bars, diffs):
            ax.text(bar.get_width() + 0.1 * np.sign(bar.get_width()),
                    bar.get_y() + bar.get_height()/2,
                    f"{diff:+.1f}%p", va="center", fontsize=9)

    fig.suptitle("Prompt Component Marginal Effect on Bankruptcy Rate", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "v5_component_marginal.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {FIGURE_DIR / 'v5_component_marginal.png'}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Setup logging
    log_file = LOG_DIR / f"condition_analysis_{timestamp}.log"
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    logger.info("=" * 70)
    logger.info("CONDITION-LEVEL SAE ANALYSIS (NEW)")
    logger.info("=" * 70)

    all_results = {}

    for paradigm in ["ic", "sm", "mw"]:
        logger.info(f"\n{'='*70}")
        logger.info(f"PARADIGM: {PARADIGM_LABELS[paradigm]}")
        logger.info(f"{'='*70}")

        best_dp = BEST_LAYERS[paradigm]["dp"]
        results = {"paradigm": paradigm, "best_layer": best_dp}

        # ── Analysis 1: BK rate by condition ──
        logger.info(f"\n--- Analysis 1: BK Rate by Condition ---")
        bk_rates = analyze_bk_rate_by_condition(paradigm, layer=best_dp)
        results["bk_rate_by_condition"] = bk_rates
        for cond, data in sorted(bk_rates.items()):
            logger.info(f"  {cond}: {data['n_bk']}/{data['n_total']} ({data['bk_rate']*100:.1f}%)")

        # ── Analysis 2: BK AUC by condition ──
        logger.info(f"\n--- Analysis 2: BK AUC Stratified by Condition (L{best_dp}) ---")
        auc_by_cond = analyze_bk_auc_by_condition(paradigm, best_dp)
        results["auc_by_condition"] = auc_by_cond
        for cond, data in sorted(auc_by_cond.items()):
            if isinstance(data, dict) and "auc" in data:
                auc_val = data["auc"]
                if np.isnan(auc_val):
                    logger.info(f"  {cond}: AUC=NaN (n_bk={data.get('n_bk', '?')})")
                else:
                    logger.info(f"  {cond}: AUC={auc_val:.4f} (n={data.get('n', '?')}, n_bk={data.get('n_bk', '?')})")

        # ── Analysis 3: Component marginal effect (SM/MW only) ──
        if paradigm in ("sm", "mw"):
            logger.info(f"\n--- Analysis 3: Component Marginal Effect (L{best_dp}) ---")
            marginal = analyze_component_marginal(paradigm, best_dp)
            results["component_marginal"] = marginal
            for comp, data in sorted(marginal.items()):
                logger.info(f"  {comp}: BK rate diff = {data['bk_rate_diff']*100:+.2f}%p "
                          f"(with={data['bk_rate_with']*100:.1f}%, without={data['bk_rate_without']*100:.1f}%)")
        else:
            # IC only has BASE/G/M/GM — treat G, M as components
            logger.info(f"\n--- Analysis 3: IC Prompt Condition Effect ---")
            marginal = {}
            loaded = load_layer_features(paradigm, best_dp, mode="decision_point", dense=True)
            if loaded:
                features, meta = loaded
                labels = get_labels(meta)
                rate = (features != 0).mean(axis=0)
                active = rate >= MIN_ACTIVATION_RATE
                features_f = features[:, active]

                if "prompt_conditions" in meta:
                    conditions = meta["prompt_conditions"]
                    for comp in ["G", "M"]:
                        has = np.array([comp in str(c) for c in conditions])
                        no = ~has
                        if has.sum() >= 10 and no.sum() >= 10:
                            bk_with = labels[has].mean()
                            bk_without = labels[no].mean()
                            auc_with = classify_cv(features_f[has], labels[has])
                            auc_without = classify_cv(features_f[no], labels[no])
                            marginal[comp] = {
                                "n_with": int(has.sum()), "n_without": int(no.sum()),
                                "bk_rate_with": float(bk_with), "bk_rate_without": float(bk_without),
                                "bk_rate_diff": float(bk_with - bk_without),
                                "auc_with": auc_with, "auc_without": auc_without,
                            }
                            logger.info(f"  {comp}: BK rate diff = {(bk_with-bk_without)*100:+.2f}%p "
                                      f"(with={bk_with*100:.1f}%, without={bk_without*100:.1f}%)")
            results["component_marginal"] = marginal

        # ── Analysis 4: Condition-specific features ──
        logger.info(f"\n--- Analysis 4: Condition-Specific Feature Overlap (L{best_dp}) ---")
        feat_overlap = analyze_condition_features(paradigm, best_dp)
        results["feature_overlap"] = feat_overlap
        if "overlaps" in feat_overlap:
            for pair, data in sorted(feat_overlap["overlaps"].items()):
                logger.info(f"  {pair}: {data['shared']} shared (Jaccard={data['jaccard']:.3f})")

        all_results[paradigm] = results

    # ── Save results ──
    json_file = JSON_DIR / f"condition_analysis_{timestamp}.json"

    # Convert sets to lists for JSON serialization
    def make_serializable(obj):
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    import copy
    serializable = json.loads(json.dumps(all_results, default=make_serializable))
    with open(json_file, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"\nResults saved: {json_file}")

    # ── Generate figures ──
    logger.info("\nGenerating figures...")

    bk_rate_data = {p: r["bk_rate_by_condition"] for p, r in all_results.items()}
    plot_bk_rate_by_condition(bk_rate_data)

    auc_data = {p: r["auc_by_condition"] for p, r in all_results.items()}
    plot_auc_by_condition(auc_data)

    marginal_data = {p: r.get("component_marginal", {}) for p, r in all_results.items()}
    plot_component_marginal(marginal_data)

    logger.info(f"\n{'='*70}")
    logger.info("CONDITION ANALYSIS COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"JSON: {json_file}")
    logger.info(f"Log: {log_file}")
    logger.info(f"Figures: {FIGURE_DIR}/v5_*.png")


if __name__ == "__main__":
    main()
