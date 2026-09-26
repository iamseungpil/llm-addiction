#!/usr/bin/env python3
"""
Stage 2 + 3c: Feature-level BK correlation analysis (Gemma).

Analyses:
  2a. DP correlation (balance confound included)
  2b. R1 overall correlation (prompt confound included)
  2b'. R1 within-condition correlation (cleanest — KEY analysis)
  2d. Cross-paradigm feature overlap (hypergeometric test)
  2e. Per-condition correlation (IC fixed only, exploratory)
  2f. Behavioral ↔ SAE linkage (Spearman)
  3c. Top feature temporal dynamics

Usage:
  python run_correlation_analysis.py [--analyses 2a,2b,2bp,2d,2e,2f,3c] [--layers best|all]
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
from scipy import stats
from scipy.stats import mannwhitneyu, fisher_exact, hypergeom, spearmanr
from statsmodels.stats.multitest import fdrcorrection

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    PARADIGMS, PARADIGM_LABELS, PARADIGM_COLORS, BET_COLORS,
    FDR_ALPHA, RANDOM_SEED, RESULTS_DIR, FIGURE_DIR, JSON_DIR, LOG_DIR,
    MIN_ACTIVATION_RATE,
)
from data_loader import load_layer_features, get_labels, filter_active_features, get_metadata, load_sparse_npz

warnings.filterwarnings("ignore", category=RuntimeWarning)
np.random.seed(RANDOM_SEED)

# ===================================================================
# Constants
# ===================================================================

# Best layers per paradigm (from prior classification analyses)
BEST_LAYERS = {
    "ic": [0, 8, 18, 22, 26, 40],
    "sm": [0, 6, 12, 16, 22, 40],
    "mw": [0, 10, 22, 24, 33, 40],
}
ALL_BEST_LAYERS = sorted(set(l for layers in BEST_LAYERS.values() for l in layers))

# 2b' within-condition subsets (cleanest analysis)
WITHIN_CONDITION_SUBSETS = {
    "ic": [
        {"name": "c50_fixed", "bet_type": "fixed", "constraint": "50", "label": "IC c50 fixed"},
        {"name": "c70_fixed", "bet_type": "fixed", "constraint": "70", "label": "IC c70 fixed"},
        {"name": "c50c70_fixed_pooled", "bet_type": "fixed", "constraint": ["50", "70"],
         "label": "IC c50+c70 fixed (pooled)"},
    ],
    "sm": [
        {"name": "var_hasG", "bet_type": "variable", "has_component": "G",
         "label": "SM variable+G"},
    ],
    "mw": [
        {"name": "fixed_hasG", "bet_type": "fixed", "has_component": "G",
         "label": "MW fixed+G"},
    ],
}

# Game JSON paths (for behavioral metrics)
DATA_ROOT = Path("/home/jovyan/beomi/llm-addiction-data")
IC_JSON_FILES = [
    (DATA_ROOT / "investment_choice_v2_role" / "gemma_investment_c10_20260225_122319.json", 0),
    (DATA_ROOT / "investment_choice_v2_role" / "gemma_investment_c30_20260225_184458.json", 400),
    (DATA_ROOT / "investment_choice_v2_role" / "gemma_investment_c50_20260226_020821.json", 800),
    (DATA_ROOT / "investment_choice_v2_role" / "gemma_investment_c70_20260226_082029.json", 1200),
]
SM_JSON_FILE = DATA_ROOT / "slot_machine" / "experiment_0_gemma_v4_role" / "final_gemma_20260227_002507.json"
MW_JSON_FILE = DATA_ROOT / "mystery_wheel_v2_role" / "gemma_mysterywheel_checkpoint_3200.json"

MIN_COHENS_D = 0.3


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def setup_logger(name: str) -> logging.Logger:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(LOG_DIR / f"correlation_{timestamp}.log")
    sh = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger, timestamp


# ===================================================================
# Core statistical function: two-stage correlation test
# ===================================================================

def two_stage_test(feature_values, labels):
    """Two-stage test for sparse SAE feature vs binary label.

    Stage 1: Fisher's exact test on activation rate (fire/not-fire).
    Stage 2: Mann-Whitney U on nonzero values only (magnitude).

    Returns dict with r, p_fisher, p_mw, p_combined, cohen_d, direction, etc.
    """
    bk_vals = feature_values[labels == 1]
    safe_vals = feature_values[labels == 0]
    n_bk = len(bk_vals)
    n_safe = len(safe_vals)

    if n_bk < 5 or n_safe < 5:
        return None

    # Stage 1: Fisher's exact on activation rate
    bk_active = (bk_vals != 0).sum()
    bk_inactive = n_bk - bk_active
    safe_active = (safe_vals != 0).sum()
    safe_inactive = n_safe - safe_active

    # Skip if too few active in either group
    if bk_active + safe_active < 5:
        return None

    _, p_fisher = fisher_exact([[bk_active, bk_inactive], [safe_active, safe_inactive]])
    rate_bk = bk_active / n_bk
    rate_safe = safe_active / n_safe

    # Stage 2: Mann-Whitney U on nonzero values only
    bk_nonzero = bk_vals[bk_vals != 0]
    safe_nonzero = safe_vals[safe_vals != 0]
    p_mw = 1.0
    if len(bk_nonzero) >= 3 and len(safe_nonzero) >= 3:
        try:
            _, p_mw = mannwhitneyu(bk_nonzero, safe_nonzero, alternative="two-sided")
        except ValueError:
            p_mw = 1.0

    # Combined p-value: min of the two, with Bonferroni-like correction (×2)
    p_combined = min(1.0, 2 * min(p_fisher, p_mw))

    # Cohen's d on all values (including zeros)
    mean_bk = bk_vals.mean()
    mean_safe = safe_vals.mean()
    pooled_std = np.sqrt(((n_bk - 1) * bk_vals.std(ddof=1) ** 2 +
                          (n_safe - 1) * safe_vals.std(ddof=1) ** 2) /
                         max(1, n_bk + n_safe - 2))
    d = (mean_bk - mean_safe) / pooled_std if pooled_std > 0 else 0.0

    return {
        "p_fisher": float(p_fisher),
        "p_mw": float(p_mw),
        "p_combined": float(p_combined),
        "cohen_d": float(d),
        "direction": "bk_higher" if d > 0 else "safe_higher",
        "rate_bk": float(rate_bk),
        "rate_safe": float(rate_safe),
        "mean_bk": float(mean_bk),
        "mean_safe": float(mean_safe),
        "n_bk": int(n_bk),
        "n_safe": int(n_safe),
    }


def run_correlation_on_subset(features, labels, active_idx, logger, prefix=""):
    """Run two-stage test on all active features, apply FDR."""
    results = []
    p_values = []

    for i, feat_idx in enumerate(active_idx):
        res = two_stage_test(features[:, i], labels)
        if res is not None:
            res["feature_idx"] = int(feat_idx)
            results.append(res)
            p_values.append(res["p_combined"])

    if not results:
        logger.info(f"  {prefix}No testable features")
        return []

    # FDR correction
    p_arr = np.array(p_values)
    rejected, p_corrected = fdrcorrection(p_arr, alpha=FDR_ALPHA)

    for i, res in enumerate(results):
        res["p_fdr"] = float(p_corrected[i])
        res["fdr_significant"] = bool(rejected[i])

    # Filter significant with |d| > threshold
    sig = [r for r in results if r["fdr_significant"] and abs(r["cohen_d"]) >= MIN_COHENS_D]
    sig_any = [r for r in results if r["fdr_significant"]]

    n_bk_higher = len([r for r in sig if r["direction"] == "bk_higher"])
    n_safe_higher = len([r for r in sig if r["direction"] == "safe_higher"])

    logger.info(f"  {prefix}Tested: {len(results)}, FDR sig: {len(sig_any)}, "
                f"|d|>{MIN_COHENS_D}: {len(sig)} (BK↑:{n_bk_higher}, Safe↑:{n_safe_higher})")

    return results


# ===================================================================
# Analysis 2a: DP correlation
# ===================================================================

def analysis_2a_dp_correlation(logger, layers):
    """Decision-point correlation (balance confound included)."""
    logger.info("=" * 60)
    logger.info("Analysis 2a: DP Feature-BK Correlation (balance confound)")
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

            layer_results = run_correlation_on_subset(
                active_feat, labels, active_idx, logger, f"L{layer} DP: ")

            results[paradigm][f"L{layer}"] = {
                "n_features_tested": len(layer_results),
                "n_significant": len([r for r in layer_results if r["fdr_significant"]]),
                "n_sig_strong": len([r for r in layer_results
                                     if r["fdr_significant"] and abs(r["cohen_d"]) >= MIN_COHENS_D]),
                "top_features": sorted(
                    [r for r in layer_results if r["fdr_significant"]],
                    key=lambda x: abs(x["cohen_d"]), reverse=True
                )[:20],
            }

            del features, active_feat
            gc.collect()

    return results


# ===================================================================
# Analysis 2b: R1 overall correlation
# ===================================================================

def analysis_2b_r1_correlation(logger, layers):
    """R1-only correlation (prompt confound included, no balance confound)."""
    logger.info("=" * 60)
    logger.info("Analysis 2b: R1 Feature-BK Correlation (prompt confound)")
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
            r1_mask = meta["round_nums"] == 1
            features_r1 = features[r1_mask]
            meta_r1 = {k: v[r1_mask] for k, v in meta.items()}

            active_feat, active_idx = filter_active_features(features_r1, MIN_ACTIVATION_RATE)
            labels = get_labels(meta_r1)

            layer_results = run_correlation_on_subset(
                active_feat, labels, active_idx, logger, f"L{layer} R1: ")

            results[paradigm][f"L{layer}"] = {
                "n_features_tested": len(layer_results),
                "n_significant": len([r for r in layer_results if r["fdr_significant"]]),
                "n_sig_strong": len([r for r in layer_results
                                     if r["fdr_significant"] and abs(r["cohen_d"]) >= MIN_COHENS_D]),
                "top_features": sorted(
                    [r for r in layer_results if r["fdr_significant"]],
                    key=lambda x: abs(x["cohen_d"]), reverse=True
                )[:20],
            }

            del features, features_r1, active_feat
            gc.collect()

    return results


# ===================================================================
# Analysis 2b': R1 within-condition correlation (KEY)
# ===================================================================

def _filter_subset(meta, subset_def):
    """Create boolean mask for a within-condition subset."""
    mask = np.ones(len(meta["game_ids"]), dtype=bool)

    if "bet_type" in subset_def:
        mask &= meta["bet_types"] == subset_def["bet_type"]

    if "constraint" in subset_def:
        c = subset_def["constraint"]
        if isinstance(c, list):
            mask &= np.isin(meta["bet_constraints"], c)
        else:
            mask &= meta["bet_constraints"] == c

    if "has_component" in subset_def:
        comp = subset_def["has_component"]
        # Map display label to data label (H→R for Hidden patterns)
        data_comp = {"H": "R"}.get(comp, comp)
        mask &= np.array([data_comp in str(pc) for pc in meta["prompt_conditions"]])

    return mask


def analysis_2bp_within_condition_r1(logger, layers):
    """R1 within-condition correlation — cleanest analysis."""
    logger.info("=" * 60)
    logger.info("Analysis 2b': R1 Within-Condition Correlation (CLEANEST)")
    logger.info("=" * 60)

    results = {}
    for paradigm in PARADIGMS:
        results[paradigm] = {}
        logger.info(f"\n--- {PARADIGM_LABELS[paradigm]} ---")

        subsets = WITHIN_CONDITION_SUBSETS.get(paradigm, [])
        if not subsets:
            continue

        for layer in layers:
            loaded = load_layer_features(paradigm, layer, mode="all_rounds", dense=True)
            if loaded is None:
                continue

            features, meta = loaded
            r1_mask = meta["round_nums"] == 1
            features_r1 = features[r1_mask]
            meta_r1 = {k: v[r1_mask] for k, v in meta.items()}

            for subset_def in subsets:
                subset_mask = _filter_subset(meta_r1, subset_def)
                n_total = subset_mask.sum()
                if n_total < 20:
                    continue

                sub_features = features_r1[subset_mask]
                sub_labels = get_labels({k: v[subset_mask] for k, v in meta_r1.items()})
                n_bk = int(sub_labels.sum())

                if n_bk < 5:
                    logger.info(f"  L{layer} {subset_def['name']}: SKIP (n={n_total}, BK={n_bk})")
                    continue

                active_feat, active_idx = filter_active_features(sub_features, MIN_ACTIVATION_RATE)
                layer_results = run_correlation_on_subset(
                    active_feat, sub_labels, active_idx, logger,
                    f"L{layer} {subset_def['name']} (n={n_total},BK={n_bk}): ")

                key = f"L{layer}_{subset_def['name']}"
                results[paradigm][key] = {
                    "subset": subset_def["label"],
                    "n_total": int(n_total),
                    "n_bk": n_bk,
                    "n_features_tested": len(layer_results),
                    "n_significant": len([r for r in layer_results if r["fdr_significant"]]),
                    "n_sig_strong": len([r for r in layer_results
                                         if r["fdr_significant"] and abs(r["cohen_d"]) >= MIN_COHENS_D]),
                    "top_features": sorted(
                        [r for r in layer_results if r["fdr_significant"]],
                        key=lambda x: abs(x["cohen_d"]), reverse=True
                    )[:20],
                    "all_results": layer_results,
                }

            del features, features_r1
            gc.collect()

    return results


# ===================================================================
# Analysis 2d: Cross-paradigm overlap
# ===================================================================

def analysis_2d_cross_paradigm_overlap(logger, results_2a):
    """Cross-paradigm top-K feature overlap with hypergeometric test."""
    logger.info("=" * 60)
    logger.info("Analysis 2d: Cross-Paradigm Feature Overlap")
    logger.info("=" * 60)

    K = 50
    results = {}

    # Find common layers across paradigms
    common_layers = set()
    for paradigm in PARADIGMS:
        if paradigm in results_2a:
            common_layers.update(results_2a[paradigm].keys())
    common_layers = sorted(common_layers, key=lambda x: int(x.replace("L", "")))

    for layer_key in common_layers:
        # Get active feature sets per paradigm
        paradigm_features = {}
        for paradigm in PARADIGMS:
            if paradigm in results_2a and layer_key in results_2a[paradigm]:
                layer_data = results_2a[paradigm][layer_key]
                # All tested features (not just significant)
                if "top_features" in layer_data:
                    # Get all results sorted by |d|
                    # We need full results — load from top_features + reconstruct
                    all_feats = layer_data.get("top_features", [])
                    paradigm_features[paradigm] = all_feats

        if len(paradigm_features) < 2:
            continue

        # For overlap, we need the full sorted feature list from the layer
        # Use top_features (up to 20) — if we want top-K=50, we need all_results
        # For now, use what's available
        layer_num = int(layer_key.replace("L", ""))
        overlap_results = {}

        paradigms_list = list(paradigm_features.keys())
        for i in range(len(paradigms_list)):
            for j in range(i + 1, len(paradigms_list)):
                p1, p2 = paradigms_list[i], paradigms_list[j]

                # Load active features for both paradigms at this layer
                loaded1 = load_layer_features(p1, layer_num, mode="decision_point", dense=True)
                loaded2 = load_layer_features(p2, layer_num, mode="decision_point", dense=True)
                if loaded1 is None or loaded2 is None:
                    continue

                feat1, meta1 = loaded1
                feat2, meta2 = loaded2
                _, active_idx1 = filter_active_features(feat1, MIN_ACTIVATION_RATE)
                _, active_idx2 = filter_active_features(feat2, MIN_ACTIVATION_RATE)

                # Intersection of active features
                shared_active = np.intersect1d(active_idx1, active_idx2)
                N_pool = len(shared_active)

                if N_pool < K:
                    overlap_results[f"{p1}_vs_{p2}"] = {
                        "skipped": True, "reason": f"shared_active={N_pool} < K={K}"}
                    continue

                # Run correlation on both paradigms using shared features only
                labels1 = get_labels(meta1)
                labels2 = get_labels(meta2)

                # Get Cohen's d for each shared feature in both paradigms
                d_scores_1 = []
                d_scores_2 = []

                def _cohens_d(vals, labs):
                    bk = vals[labs == 1]
                    safe = vals[labs == 0]
                    if len(bk) < 2 or len(safe) < 2:
                        return 0.0
                    pooled = np.sqrt(((len(bk)-1)*bk.std(ddof=1)**2 +
                                      (len(safe)-1)*safe.std(ddof=1)**2) /
                                     max(1, len(bk)+len(safe)-2))
                    return abs((bk.mean() - safe.mean()) / pooled) if pooled > 0 else 0.0

                for feat_global_idx in shared_active:
                    v1 = feat1[:, feat_global_idx]
                    v2 = feat2[:, feat_global_idx]
                    d_scores_1.append(_cohens_d(v1, labels1))
                    d_scores_2.append(_cohens_d(v2, labels2))

                d1 = np.array(d_scores_1)
                d2 = np.array(d_scores_2)

                # Top-K by |d| in each paradigm
                top_k1 = set(shared_active[np.argsort(d1)[-K:]])
                top_k2 = set(shared_active[np.argsort(d2)[-K:]])

                overlap = top_k1 & top_k2
                jaccard = len(overlap) / len(top_k1 | top_k2) if len(top_k1 | top_k2) > 0 else 0

                # Hypergeometric test
                # P(X >= observed_overlap | N=pool, K1=K, K2=K)
                hyper_p = hypergeom.sf(len(overlap) - 1, N_pool, K, K)

                overlap_results[f"{p1}_vs_{p2}"] = {
                    "shared_active_pool": int(N_pool),
                    "K": K,
                    "overlap_count": len(overlap),
                    "jaccard": float(jaccard),
                    "hypergeom_p": float(hyper_p),
                    "significant": bool(hyper_p < 0.05),
                    "overlap_features": sorted(overlap),
                }

                logger.info(f"  {layer_key} {p1} vs {p2}: overlap={len(overlap)}/{K}, "
                            f"Jaccard={jaccard:.3f}, p={hyper_p:.4f}")

                del feat1, feat2
                gc.collect()

        results[layer_key] = overlap_results

    return results


# ===================================================================
# Analysis 2e: Per-condition (IC fixed only)
# ===================================================================

def analysis_2e_percondition_ic(logger, layers):
    """Per-constraint correlation within IC fixed bet type."""
    logger.info("=" * 60)
    logger.info("Analysis 2e: Per-Condition IC Fixed (exploratory)")
    logger.info("=" * 60)

    results = {}
    # Use best 2 layers for IC (by prior AUC: L18, L22)
    ic_layers = [18, 22] if len(layers) > 2 else layers

    for layer in ic_layers:
        loaded = load_layer_features("ic", layer, mode="all_rounds", dense=True)
        if loaded is None:
            continue

        features, meta = loaded
        r1_mask = meta["round_nums"] == 1
        features_r1 = features[r1_mask]
        meta_r1 = {k: v[r1_mask] for k, v in meta.items()}

        for constraint in ["30", "50", "70"]:
            mask = (meta_r1["bet_types"] == "fixed") & (meta_r1["bet_constraints"] == constraint)
            n_total = mask.sum()
            if n_total < 20:
                continue

            sub_features = features_r1[mask]
            sub_labels = get_labels({k: v[mask] for k, v in meta_r1.items()})
            n_bk = int(sub_labels.sum())

            if n_bk < 5:
                logger.info(f"  L{layer} c{constraint}_fixed: SKIP (n={n_total}, BK={n_bk})")
                continue

            active_feat, active_idx = filter_active_features(sub_features, MIN_ACTIVATION_RATE)
            layer_results = run_correlation_on_subset(
                active_feat, sub_labels, active_idx, logger,
                f"L{layer} c{constraint}_fixed (n={n_total},BK={n_bk}): ")

            key = f"L{layer}_c{constraint}_fixed"
            results[key] = {
                "constraint": constraint,
                "n_total": int(n_total),
                "n_bk": n_bk,
                "n_features_tested": len(layer_results),
                "n_significant": len([r for r in layer_results if r["fdr_significant"]]),
                "top_features": sorted(
                    [r for r in layer_results if r["fdr_significant"]],
                    key=lambda x: abs(x["cohen_d"]), reverse=True
                )[:20],
            }

        del features, features_r1
        gc.collect()

    return results


# ===================================================================
# Analysis 2f: Behavioral ↔ SAE linkage
# ===================================================================

def _load_behavioral_metrics():
    """Load per-game behavioral metrics from latest comprehensive analysis JSON."""
    candidates = sorted(JSON_DIR.glob("comprehensive_gemma_*.json"), reverse=True)
    for json_path in candidates:
        with open(json_path) as f:
            data = json.load(f)
        if "behavioral" in data:
            return data["behavioral"]
    return None


def analysis_2f_behavioral_sae(logger, results_2bp):
    """Correlate top BK features with behavioral metrics (Spearman)."""
    logger.info("=" * 60)
    logger.info("Analysis 2f: Behavioral ↔ SAE Feature Linkage")
    logger.info("=" * 60)

    behavioral = _load_behavioral_metrics()
    if behavioral is None:
        logger.warning("  Behavioral metrics not found, skipping")
        return {}

    results = {}
    for paradigm in PARADIGMS:
        if paradigm not in results_2bp or paradigm not in behavioral:
            continue

        results[paradigm] = {}
        logger.info(f"\n--- {PARADIGM_LABELS[paradigm]} ---")

        # Get per-game behavioral metrics
        per_game = behavioral[paradigm].get("per_game", [])
        if not per_game:
            continue

        # Detect which metrics are available (SM/MW may lack i_ec)
        sample = per_game[0] if per_game else {}
        available_metrics = []
        for m in ["i_ba", "i_lc", "i_ec"]:
            key_name = "i_ba_mean" if m == "i_ba" else m
            if key_name in sample and sample[key_name] is not None:
                available_metrics.append(m)
        logger.info(f"  Available behavioral metrics: {available_metrics}")

        # Build game_id → metrics lookup
        metrics_by_game = {}
        for g in per_game:
            gid = g["game_id"]
            entry = {}
            if "i_ba" in available_metrics:
                entry["i_ba"] = g.get("i_ba_mean", 0) or 0
            if "i_lc" in available_metrics:
                entry["i_lc"] = g.get("i_lc", 0) or 0
            if "i_ec" in available_metrics:
                entry["i_ec"] = g.get("i_ec", 0) or 0
            metrics_by_game[gid] = entry

        # For each subset in 2b' results, find top features and correlate
        for key, subset_result in results_2bp[paradigm].items():
            sig_features = [f for f in subset_result.get("top_features", [])
                           if f.get("fdr_significant")]
            if not sig_features:
                continue

            # Load features for this layer
            layer_num = int(key.split("_")[0].replace("L", ""))
            loaded = load_layer_features(paradigm, layer_num, mode="decision_point", dense=True)
            if loaded is None:
                continue

            features, meta = loaded
            game_ids = meta["game_ids"]

            correlations = []
            for feat_info in sig_features[:10]:  # Top 10
                feat_idx = feat_info["feature_idx"]
                feat_vals = features[:, feat_idx]

                # Align with behavioral metrics
                feat_aligned = []
                metric_vals_dict = {m: [] for m in available_metrics}
                for i, gid in enumerate(game_ids):
                    if int(gid) in metrics_by_game:
                        m = metrics_by_game[int(gid)]
                        feat_aligned.append(feat_vals[i])
                        for mn in available_metrics:
                            metric_vals_dict[mn].append(m.get(mn, 0))

                if len(feat_aligned) < 20:
                    continue

                feat_arr = np.array(feat_aligned)
                corr_results = {"feature_idx": feat_idx, "cohen_d": feat_info["cohen_d"]}

                for metric_name in available_metrics:
                    rho, p = spearmanr(feat_arr, metric_vals_dict[metric_name])
                    corr_results[f"spearman_{metric_name}"] = float(rho) if not np.isnan(rho) else 0.0
                    corr_results[f"p_{metric_name}"] = float(p) if not np.isnan(p) else 1.0

                correlations.append(corr_results)

            if correlations:
                results[paradigm][key] = correlations
                for c in correlations[:3]:
                    logger.info(f"  {key} F{c['feature_idx']}: "
                                f"I_BA rho={c['spearman_i_ba']:.3f}, "
                                f"I_LC rho={c['spearman_i_lc']:.3f}")

            del features
            gc.collect()

    return results


# ===================================================================
# Analysis 3c: Temporal dynamics of top features
# ===================================================================

def analysis_3c_temporal_dynamics(logger, results_2bp, n_top=20):
    """Track top BK features' activation across rounds (BK vs safe)."""
    logger.info("=" * 60)
    logger.info("Analysis 3c: Top Feature Temporal Dynamics")
    logger.info("=" * 60)

    results = {}
    for paradigm in PARADIGMS:
        if paradigm not in results_2bp:
            continue

        results[paradigm] = {}
        logger.info(f"\n--- {PARADIGM_LABELS[paradigm]} ---")

        # Collect top features across all subsets
        top_features_by_layer = defaultdict(set)
        for key, subset_result in results_2bp[paradigm].items():
            layer_num = int(key.split("_")[0].replace("L", ""))
            for f in subset_result.get("top_features", [])[:n_top]:
                if f.get("fdr_significant"):
                    top_features_by_layer[layer_num].add(f["feature_idx"])

        for layer, feat_set in top_features_by_layer.items():
            if not feat_set:
                continue

            loaded = load_layer_features(paradigm, layer, mode="all_rounds", dense=True)
            if loaded is None:
                continue

            features, meta = loaded
            labels_game = get_labels(meta)  # per-round BK label
            round_nums = meta["round_nums"]

            unique_rounds = sorted(np.unique(round_nums))
            feat_list = sorted(feat_set)

            temporal_data = {}
            for feat_idx in feat_list:
                feat_vals = features[:, feat_idx]
                bk_trajectory = []
                safe_trajectory = []

                for rnd in unique_rounds:
                    rnd_mask = round_nums == rnd
                    bk_mask = rnd_mask & (labels_game == 1)
                    safe_mask = rnd_mask & (labels_game == 0)

                    bk_mean = float(feat_vals[bk_mask].mean()) if bk_mask.sum() > 0 else 0.0
                    safe_mean = float(feat_vals[safe_mask].mean()) if safe_mask.sum() > 0 else 0.0
                    bk_trajectory.append(bk_mean)
                    safe_trajectory.append(safe_mean)

                temporal_data[feat_idx] = {
                    "rounds": [int(r) for r in unique_rounds],
                    "bk_mean": bk_trajectory,
                    "safe_mean": safe_trajectory,
                }

            results[paradigm][f"L{layer}"] = {
                "n_features": len(feat_list),
                "features": temporal_data,
            }

            logger.info(f"  L{layer}: tracked {len(feat_list)} features across {len(unique_rounds)} rounds")

            del features
            gc.collect()

    return results


# ===================================================================
# Visualization
# ===================================================================

def plot_correlation_summary(results_2a, results_2b, results_2bp, timestamp):
    """Summary figure: significant features per layer."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (analysis_name, analysis_results, marker) in zip(axes, [
        ("DP", results_2a, "o"),
        ("R1 overall", results_2b, "s"),
        ("R1 within-cond", results_2bp, "^"),
    ]):
        for paradigm, color in PARADIGM_COLORS.items():
            if paradigm not in analysis_results:
                continue
            layers_data = []
            for key, data in analysis_results[paradigm].items():
                layer_num = int(key.split("_")[0].replace("L", ""))
                n_sig = data.get("n_sig_strong", data.get("n_significant", 0))
                layers_data.append((layer_num, n_sig))

            if layers_data:
                layers_data.sort()
                x, y = zip(*layers_data)
                ax.plot(x, y, marker=marker, color=color, label=PARADIGM_LABELS[paradigm],
                        linewidth=2, markersize=8)

        ax.set_xlabel("Layer")
        ax.set_ylabel("# Significant Features (|d|>0.3)")
        ax.set_title(f"{analysis_name} Correlation")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Feature-BK Correlation: Significant Features per Layer", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = FIGURE_DIR / f"correlation_summary_{timestamp}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(path)


def plot_temporal_dynamics(results_3c, timestamp):
    """Plot top feature activation trajectories."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for paradigm, paradigm_data in results_3c.items():
        for layer_key, layer_data in paradigm_data.items():
            feat_data = layer_data.get("features", {})
            if not feat_data:
                continue

            n_feats = min(6, len(feat_data))
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()

            for ax_idx, (feat_idx, traj) in enumerate(list(feat_data.items())[:n_feats]):
                ax = axes[ax_idx]
                rounds = traj["rounds"]
                max_round = min(30, max(rounds))
                mask = [i for i, r in enumerate(rounds) if r <= max_round]

                ax.plot([rounds[i] for i in mask], [traj["bk_mean"][i] for i in mask],
                        color="#e74c3c", linewidth=2, label="BK games")
                ax.plot([rounds[i] for i in mask], [traj["safe_mean"][i] for i in mask],
                        color="#2ecc71", linewidth=2, label="Safe games")
                ax.set_title(f"Feature {feat_idx}")
                ax.set_xlabel("Round")
                ax.set_ylabel("Mean Activation")
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

            for ax_idx in range(n_feats, len(axes)):
                axes[ax_idx].set_visible(False)

            plt.suptitle(f"{PARADIGM_LABELS[paradigm]} {layer_key}: Top Feature Temporal Dynamics",
                         fontsize=14, fontweight="bold")
            plt.tight_layout()
            path = FIGURE_DIR / f"temporal_{paradigm}_{layer_key}_{timestamp}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyses", type=str, default="2a,2b,2bp,2d,2e,2f,3c",
                        help="Comma-separated analysis IDs")
    parser.add_argument("--layers", type=str, default="best",
                        help="'best' for paradigm-specific best layers, 'all' for 0-41")
    args = parser.parse_args()

    analyses = set(args.analyses.split(","))
    logger, timestamp = setup_logger("correlation")

    if args.layers == "all":
        layers = list(range(42))
    else:
        layers = ALL_BEST_LAYERS

    logger.info(f"Starting correlation analysis: analyses={analyses}, layers={len(layers)}")
    logger.info(f"Layers: {layers}")

    all_results = {}

    # 2a: DP correlation
    if "2a" in analyses:
        all_results["dp_correlation"] = analysis_2a_dp_correlation(logger, layers)

    # 2b: R1 overall
    if "2b" in analyses:
        all_results["r1_correlation"] = analysis_2b_r1_correlation(logger, layers)

    # 2b': R1 within-condition (KEY)
    if "2bp" in analyses:
        all_results["r1_within_condition"] = analysis_2bp_within_condition_r1(logger, layers)

    # 2d: Cross-paradigm overlap (requires 2a results)
    if "2d" in analyses:
        dp_results = all_results.get("dp_correlation")
        if dp_results is None and "2a" not in analyses:
            logger.info("Running 2a first for 2d overlap...")
            dp_results = analysis_2a_dp_correlation(logger, layers)
            all_results["dp_correlation"] = dp_results
        if dp_results:
            all_results["cross_paradigm_overlap"] = analysis_2d_cross_paradigm_overlap(logger, dp_results)

    # 2e: Per-condition IC fixed
    if "2e" in analyses:
        all_results["percondition_ic_fixed"] = analysis_2e_percondition_ic(logger, layers)

    # 2f: Behavioral linkage (requires 2b' results)
    if "2f" in analyses:
        bp_results = all_results.get("r1_within_condition")
        if bp_results is None and "2bp" not in analyses:
            logger.info("Running 2b' first for 2f...")
            bp_results = analysis_2bp_within_condition_r1(logger, layers)
            all_results["r1_within_condition"] = bp_results
        if bp_results:
            all_results["behavioral_sae_linkage"] = analysis_2f_behavioral_sae(logger, bp_results)

    # 3c: Temporal dynamics (requires 2b' results)
    if "3c" in analyses:
        bp_results = all_results.get("r1_within_condition")
        if bp_results is None and "2bp" not in analyses:
            bp_results = analysis_2bp_within_condition_r1(logger, layers)
            all_results["r1_within_condition"] = bp_results
        if bp_results:
            all_results["temporal_dynamics"] = analysis_3c_temporal_dynamics(logger, bp_results)

    # Save results
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    json_path = JSON_DIR / f"correlation_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, cls=NumpyEncoder, indent=2)
    logger.info(f"\nResults saved: {json_path}")

    # Generate figures
    try:
        fig_path = plot_correlation_summary(
            all_results.get("dp_correlation", {}),
            all_results.get("r1_correlation", {}),
            all_results.get("r1_within_condition", {}),
            timestamp,
        )
        logger.info(f"Summary figure: {fig_path}")

        if "temporal_dynamics" in all_results:
            plot_temporal_dynamics(all_results["temporal_dynamics"], timestamp)
            logger.info("Temporal dynamics figures saved")
    except Exception as e:
        logger.error(f"Figure generation error: {e}")

    logger.info("\nCORRELATION ANALYSIS COMPLETE")
    logger.info(f"Results: {json_path}")


if __name__ == "__main__":
    main()
