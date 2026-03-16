#!/usr/bin/env python3
"""
Condition Analysis V2 — Statistical rigor improvements.

Adds:
  1. Chi-square / Fisher's exact test for BK rate differences
  2. Permutation test for component marginal effects
  3. Bootstrap CI for component AUC differences
  4. Summary table for paper

Usage:
    cd /home/jovyan/llm-addiction/sae_v3_analysis/src
    python condition_analysis_v2.py
"""

import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats

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

BEST_LAYERS = {"ic": {"dp": 22}, "sm": {"dp": 12}, "mw": {"dp": 33}}
ALL_COMPONENTS = ["G", "M", "R", "W", "P"]
COMPONENT_LABELS = {"G": "Goal", "M": "Mood", "R": "Hidden(H)", "W": "Win", "P": "Combined"}
N_PERMUTATIONS = 1000
N_BOOTSTRAP = 200


def classify_cv(X, y, n_splits=5):
    """5-fold stratified CV, returns AUC."""
    if len(np.unique(y)) < 2 or np.sum(y) < n_splits:
        return float("nan")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    aucs = []
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000,
                                 class_weight="balanced", random_state=RANDOM_SEED)
        clf.fit(X_train, y[train_idx])
        prob = clf.predict_proba(X_test)[:, 1]
        if len(np.unique(y[test_idx])) == 2:
            aucs.append(roc_auc_score(y[test_idx], prob))
    return float(np.mean(aucs)) if aucs else float("nan")


def permutation_test_bk_rate(labels, has_component, n_perm=N_PERMUTATIONS):
    """Permutation test: is BK rate difference significant?"""
    rng = np.random.RandomState(RANDOM_SEED)
    observed_diff = labels[has_component].mean() - labels[~has_component].mean()
    n_extreme = 0
    for _ in range(n_perm):
        perm = rng.permutation(has_component)
        perm_diff = labels[perm].mean() - labels[~perm].mean()
        if abs(perm_diff) >= abs(observed_diff):
            n_extreme += 1
    return float((n_extreme + 1) / (n_perm + 1))


def fisher_exact_bk(labels, group_mask):
    """Fisher's exact test for 2x2 table (BK/NoBK × With/Without component)."""
    a = int(labels[group_mask].sum())         # BK with
    b = int(group_mask.sum() - a)             # NoBK with
    c = int(labels[~group_mask].sum())        # BK without
    d = int((~group_mask).sum() - c)          # NoBK without
    _, p = stats.fisher_exact([[a, b], [c, d]])
    return float(p)


def bootstrap_auc_diff(features, labels, group_mask, n_bootstrap=N_BOOTSTRAP):
    """Bootstrap CI for AUC difference (with - without component) using OOB evaluation."""
    idx_with = np.where(group_mask)[0]
    idx_without = np.where(~group_mask)[0]
    rng = np.random.RandomState(RANDOM_SEED)
    diffs = []
    for _ in range(n_bootstrap):
        # Resample game indices with replacement
        boot_with = rng.choice(idx_with, len(idx_with), replace=True)
        boot_without = rng.choice(idx_without, len(idx_without), replace=True)

        # OOB evaluation for "with" group
        oob_with = np.setdiff1d(idx_with, boot_with)
        if len(oob_with) < 3 or np.sum(labels[boot_with]) < 2:
            continue
        scaler = StandardScaler()
        X_train = scaler.fit_transform(features[boot_with])
        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000,
                                 class_weight="balanced", random_state=RANDOM_SEED)
        clf.fit(X_train, labels[boot_with])
        if len(oob_with) > 0 and len(np.unique(labels[oob_with])) == 2:
            X_test = scaler.transform(features[oob_with])
            auc_with = roc_auc_score(labels[oob_with], clf.predict_proba(X_test)[:, 1])
        else:
            continue

        # OOB evaluation for "without" group
        oob_without = np.setdiff1d(idx_without, boot_without)
        if len(oob_without) < 3 or np.sum(labels[boot_without]) < 2:
            continue
        scaler2 = StandardScaler()
        X_train2 = scaler2.fit_transform(features[boot_without])
        clf2 = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000,
                                  class_weight="balanced", random_state=RANDOM_SEED)
        clf2.fit(X_train2, labels[boot_without])
        if len(oob_without) > 0 and len(np.unique(labels[oob_without])) == 2:
            X_test2 = scaler2.transform(features[oob_without])
            auc_without = roc_auc_score(labels[oob_without], clf2.predict_proba(X_test2)[:, 1])
        else:
            continue

        diffs.append(auc_with - auc_without)

    if len(diffs) < 10:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n_valid": len(diffs)}

    return {
        "mean": float(np.mean(diffs)),
        "ci_lo": float(np.percentile(diffs, 2.5)),
        "ci_hi": float(np.percentile(diffs, 97.5)),
        "n_valid": len(diffs),
    }


def run_analysis():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"condition_v2_{timestamp}.log"
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    logger.info("=" * 70)
    logger.info("CONDITION ANALYSIS V2 — Statistical Rigor")
    logger.info("=" * 70)

    all_results = {}

    for paradigm in ["ic", "sm", "mw"]:
        best_dp = BEST_LAYERS[paradigm]["dp"]
        logger.info(f"\n{'='*70}")
        logger.info(f"{PARADIGM_LABELS[paradigm]} (L{best_dp})")
        logger.info(f"{'='*70}")

        loaded = load_layer_features(paradigm, best_dp, mode="decision_point", dense=True)
        if loaded is None:
            continue

        features, meta = loaded
        labels = get_labels(meta)

        rate = (features != 0).mean(axis=0)
        active = rate >= MIN_ACTIVATION_RATE
        features_f = features[:, active]
        n_active = int(active.sum())

        logger.info(f"  Samples: {len(labels)}, BK: {labels.sum()}, Active features: {n_active}")

        results = {"paradigm": paradigm, "layer": best_dp, "n": len(labels),
                    "n_bk": int(labels.sum()), "n_features": n_active}

        # ── Bet Type Analysis ──
        if "bet_types" in meta:
            bt = meta["bet_types"]
            for btype in np.unique(bt):
                mask = bt == btype
                n_bk = int(labels[mask].sum())
                bk_rate = float(labels[mask].mean())
                auc = classify_cv(features_f[mask], labels[mask])
                results[f"bet_{btype}"] = {"n": int(mask.sum()), "n_bk": n_bk,
                                            "bk_rate": bk_rate, "auc": auc}
                logger.info(f"  Bet {btype}: n={mask.sum()}, BK={n_bk} ({bk_rate*100:.1f}%), AUC={auc:.4f}")

            # Fisher's exact for fixed vs variable
            fixed_mask = bt == "fixed"
            p_fisher = fisher_exact_bk(labels, fixed_mask)
            results["bet_type_fisher_p"] = p_fisher
            logger.info(f"  Fixed vs Variable BK rate: Fisher p={p_fisher:.4f}")

        # ── Component Marginal Analysis ──
        if "prompt_conditions" not in meta:
            all_results[paradigm] = results
            continue

        conditions = meta["prompt_conditions"]
        components = ALL_COMPONENTS if paradigm in ("sm", "mw") else ["G", "M"]

        logger.info(f"\n  --- Component Marginal Effects ---")
        comp_results = {}

        for comp in components:
            has = np.array([comp in str(c) for c in conditions])
            if has.sum() < 10 or (~has).sum() < 10:
                continue

            bk_with = float(labels[has].mean())
            bk_without = float(labels[~has].mean())
            diff = bk_with - bk_without

            # Fisher's exact test
            p_fisher = fisher_exact_bk(labels, has)

            # Permutation test (faster, 500 perms)
            p_perm = permutation_test_bk_rate(labels, has, n_perm=500)

            # AUC with vs without
            auc_with = classify_cv(features_f[has], labels[has])
            auc_without = classify_cv(features_f[~has], labels[~has])

            comp_results[comp] = {
                "n_with": int(has.sum()), "n_without": int((~has).sum()),
                "bk_with": int(labels[has].sum()), "bk_without": int(labels[~has].sum()),
                "bk_rate_with": bk_with, "bk_rate_without": bk_without,
                "bk_rate_diff": diff,
                "fisher_p": p_fisher,
                "perm_p": p_perm,
                "auc_with": auc_with,
                "auc_without": auc_without,
                "auc_diff": (auc_with - auc_without) if not (np.isnan(auc_with) or np.isnan(auc_without)) else float("nan"),
            }

            sig = "***" if p_fisher < 0.001 else "**" if p_fisher < 0.01 else "*" if p_fisher < 0.05 else "ns"
            logger.info(f"  {COMPONENT_LABELS.get(comp, comp):12s}: "
                       f"BK diff={diff*100:+.2f}%p (Fisher p={p_fisher:.4f} {sig}) "
                       f"AUC with={auc_with:.3f}, without={auc_without:.3f}")

        results["component_marginal"] = comp_results

        # ── IC: Prompt condition BK rates (all 4 conditions) ──
        if paradigm == "ic":
            logger.info(f"\n  --- IC Prompt Conditions ---")
            for cond in ["BASE", "G", "M", "GM"]:
                mask = conditions == cond
                n_bk = int(labels[mask].sum())
                bk_rate = float(labels[mask].mean())
                auc = classify_cv(features_f[mask], labels[mask])
                results[f"cond_{cond}"] = {"n": int(mask.sum()), "n_bk": n_bk,
                                            "bk_rate": bk_rate, "auc": auc}
                logger.info(f"  {cond}: n={mask.sum()}, BK={n_bk} ({bk_rate*100:.1f}%), AUC={auc:.4f}")

            # Chi-square test across 4 conditions
            obs_bk = [int(labels[conditions == c].sum()) for c in ["BASE", "G", "M", "GM"]]
            obs_nobk = [int((conditions == c).sum() - labels[conditions == c].sum()) for c in ["BASE", "G", "M", "GM"]]
            chi2, p_chi2 = stats.chi2_contingency(np.array([obs_bk, obs_nobk]))[:2]
            results["ic_4cond_chi2_p"] = float(p_chi2)
            logger.info(f"  4-condition chi-square: chi2={chi2:.2f}, p={p_chi2:.4f}")

        all_results[paradigm] = results

    # ── Save ──
    json_file = JSON_DIR / f"condition_v2_{timestamp}.json"
    with open(json_file, "w") as f:
        json.dump(all_results, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else list(o) if isinstance(o, np.ndarray) else o)
    logger.info(f"\nResults: {json_file}")

    # ── Summary Figure ──
    plot_summary(all_results)

    logger.info(f"\n{'='*70}")
    logger.info("CONDITION ANALYSIS V2 COMPLETE")
    logger.info(f"{'='*70}")


def plot_summary(all_results):
    """Summary figure: component effects across paradigms."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, paradigm in enumerate(["ic", "sm", "mw"]):
        ax = axes[idx]
        results = all_results.get(paradigm, {})
        comp = results.get("component_marginal", {})

        if not comp:
            ax.set_title(f"{PARADIGM_LABELS.get(paradigm, paradigm)}")
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        names = sorted(comp.keys())
        diffs = [comp[n]["bk_rate_diff"] * 100 for n in names]
        p_values = [comp[n]["fisher_p"] for n in names]
        labels = [COMPONENT_LABELS.get(n, n) for n in names]

        colors = []
        for d, p in zip(diffs, p_values):
            if p < 0.05:
                colors.append("#e74c3c" if d > 0 else "#3498db")
            else:
                colors.append("#bdc3c7")

        bars = ax.barh(range(len(names)), diffs, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("BK Rate Difference (%p)")
        ax.axvline(x=0, color="gray", linewidth=0.8)
        ax.set_title(f"{PARADIGM_LABELS.get(paradigm, paradigm)}")

        for bar, d, p in zip(bars, diffs, p_values):
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            ax.text(bar.get_width() + 0.15 * np.sign(bar.get_width()),
                    bar.get_y() + bar.get_height()/2,
                    f"{d:+.1f}%p {sig}", va="center", fontsize=9)

    fig.suptitle("Prompt Component Marginal Effect on Bankruptcy Rate\n(colored = p<0.05, gray = n.s.)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "v5_component_marginal_v2.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {FIGURE_DIR / 'v5_component_marginal_v2.png'}")


if __name__ == "__main__":
    run_analysis()
