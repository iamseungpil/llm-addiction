#!/usr/bin/env python3
"""
V7 Cross-Domain Feature Analysis: Deep investigation of SAE features and hidden
states shared across IC, SM, and MW paradigms for Gemma-2-9B.

Parts:
  A. Cross-domain important SAE features (L22 + multi-layer)
  B. Feature characterization (effect sizes, balance control, R1 analysis)
  C. Hidden state cross-domain analysis (BK direction transfer, shared subspace)
  D. Hidden state neuron-level analysis (universal BK neurons)
"""

import sys
import json
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# Add src dir to path
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_layer_features, get_labels, load_hidden_states
from config import PARADIGMS, RESULTS_DIR, JSON_DIR, RANDOM_SEED

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_SEED)

PARADIGM_KEYS = ["ic", "sm", "mw"]
PARADIGM_NAMES = {"ic": "Investment Choice", "sm": "Slot Machine", "mw": "Mystery Wheel"}


# ============================================================================
# Utility helpers
# ============================================================================

def safe_auc(labels, scores):
    """Compute AUC, returning NaN if degenerate."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    try:
        return roc_auc_score(labels, scores)
    except Exception:
        return float("nan")


def cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return (m1 - m2) / pooled


def fdr_correction(pvalues, alpha=0.05):
    """Benjamini-Hochberg FDR correction. Returns mask of significant tests."""
    pvals = np.asarray(pvalues)
    n = len(pvals)
    if n == 0:
        return np.array([], dtype=bool)
    sorted_idx = np.argsort(pvals)
    sorted_pvals = pvals[sorted_idx]
    thresholds = alpha * np.arange(1, n + 1) / n
    # Find largest k where p <= threshold
    below = sorted_pvals <= thresholds
    if not np.any(below):
        return np.zeros(n, dtype=bool)
    max_k = np.max(np.where(below)[0])
    sig_mask = np.zeros(n, dtype=bool)
    sig_mask[sorted_idx[:max_k + 1]] = True
    return sig_mask


# ============================================================================
# Part A: Cross-Domain Important SAE Features
# ============================================================================

def part_a_cross_domain_features():
    print("=" * 80)
    print("PART A: Cross-Domain Important SAE Features")
    print("=" * 80)

    results = {}

    # --- A1: L22 analysis ---
    print("\n--- A1: L22 Feature Importance ---")
    layer = 22

    paradigm_data = {}
    paradigm_coefs = {}
    paradigm_models = {}

    for pk in PARADIGM_KEYS:
        print(f"  Loading {PARADIGM_NAMES[pk]} L{layer}...")
        feat, meta = load_layer_features(pk, layer, mode="decision_point")
        labels = get_labels(meta)
        print(f"    Samples: {len(labels)}, BK: {labels.sum()}, "
              f"non-BK: {(1 - labels).sum()}")

        # Train LR
        scaler = StandardScaler()
        X = scaler.fit_transform(feat)
        lr = LogisticRegression(C=1.0, solver="lbfgs", class_weight="balanced",
                                max_iter=2000, random_state=RANDOM_SEED)
        lr.fit(X, labels)
        train_auc = safe_auc(labels, lr.decision_function(X))
        print(f"    Train AUC (full features): {train_auc:.4f}")

        paradigm_data[pk] = (feat, labels)
        paradigm_coefs[pk] = lr.coef_[0]  # (131072,) — in scaled space
        paradigm_models[pk] = (lr, scaler)

    # Compute cross-domain importance = geometric mean of |coef|
    n_features = paradigm_coefs["ic"].shape[0]
    abs_coefs = np.stack([np.abs(paradigm_coefs[pk]) for pk in PARADIGM_KEYS])  # (3, n_features)

    # Geometric mean of |coef| across 3 paradigms
    # Use log-space to avoid underflow
    log_abs = np.log(abs_coefs + 1e-30)
    log_geo_mean = log_abs.mean(axis=0)
    geo_mean = np.exp(log_geo_mean)

    # Rank features
    sorted_idx = np.argsort(-geo_mean)  # descending
    top50_idx = sorted_idx[:50]

    # Individual paradigm ranks
    paradigm_ranks = {}
    for pk in PARADIGM_KEYS:
        abs_c = np.abs(paradigm_coefs[pk])
        # rank (0 = highest)
        rank = np.argsort(-abs_c)
        rank_of = np.empty_like(rank)
        rank_of[rank] = np.arange(len(rank))
        paradigm_ranks[pk] = rank_of

    # Check sign consistency
    signs = np.stack([np.sign(paradigm_coefs[pk]) for pk in PARADIGM_KEYS])  # (3, n_features)
    consistent_sign = np.all(signs == signs[0:1, :], axis=0)

    print(f"\n  Top-20 Cross-Domain Features (L{layer}):")
    print(f"  {'Rank':<6}{'Feature':<12}{'GeoMean':<12}"
          f"{'IC_rank':<10}{'SM_rank':<10}{'MW_rank':<10}"
          f"{'IC_coef':<12}{'SM_coef':<12}{'MW_coef':<12}{'SignOK'}")
    print("  " + "-" * 110)

    top50_info = []
    for rank, fidx in enumerate(top50_idx):
        info = {
            "feature_idx": int(fidx),
            "rank": rank,
            "geo_mean_importance": float(geo_mean[fidx]),
            "paradigm_ranks": {pk: int(paradigm_ranks[pk][fidx]) for pk in PARADIGM_KEYS},
            "paradigm_coefs": {pk: float(paradigm_coefs[pk][fidx]) for pk in PARADIGM_KEYS},
            "consistent_sign": bool(consistent_sign[fidx]),
        }
        top50_info.append(info)

        if rank < 20:
            print(f"  {rank:<6}{fidx:<12}{geo_mean[fidx]:<12.6f}"
                  f"{paradigm_ranks['ic'][fidx]:<10}{paradigm_ranks['sm'][fidx]:<10}"
                  f"{paradigm_ranks['mw'][fidx]:<10}"
                  f"{paradigm_coefs['ic'][fidx]:<12.6f}"
                  f"{paradigm_coefs['sm'][fidx]:<12.6f}"
                  f"{paradigm_coefs['mw'][fidx]:<12.6f}"
                  f"{'YES' if consistent_sign[fidx] else 'NO'}")

    n_consistent = sum(1 for x in top50_info if x["consistent_sign"])
    print(f"\n  Sign consistency in top-50: {n_consistent}/50 "
          f"({100 * n_consistent / 50:.0f}%)")

    # --- AUC using only top-50 features ---
    print(f"\n  AUC using only top-50 cross-domain features:")
    top50_aucs = {}
    for pk in PARADIGM_KEYS:
        feat, labels = paradigm_data[pk]
        X_sub = feat[:, top50_idx]
        scaler_sub = StandardScaler()
        X_sub_s = scaler_sub.fit_transform(X_sub)
        lr_sub = LogisticRegression(C=1.0, solver="lbfgs", class_weight="balanced",
                                    max_iter=2000, random_state=RANDOM_SEED)
        lr_sub.fit(X_sub_s, labels)
        auc = safe_auc(labels, lr_sub.decision_function(X_sub_s))
        top50_aucs[pk] = auc
        print(f"    {PARADIGM_NAMES[pk]}: AUC = {auc:.4f}")

    results["L22_top50"] = {
        "features": top50_info,
        "sign_consistency": f"{n_consistent}/50",
        "top50_aucs": {pk: float(top50_aucs[pk]) for pk in PARADIGM_KEYS},
    }

    # --- A2: Multi-layer analysis (top-20 at L12, L18, L26, L33) ---
    print(f"\n--- A2: Multi-Layer Cross-Domain Analysis ---")
    other_layers = [12, 18, 26, 33]
    multi_layer_results = {}

    for lyr in other_layers:
        print(f"\n  Layer {lyr}:")
        lyr_coefs = {}
        for pk in PARADIGM_KEYS:
            feat, meta = load_layer_features(pk, lyr, mode="decision_point")
            labels = get_labels(meta)
            scaler = StandardScaler()
            X = scaler.fit_transform(feat)
            lr = LogisticRegression(C=1.0, solver="lbfgs", class_weight="balanced",
                                    max_iter=2000, random_state=RANDOM_SEED)
            lr.fit(X, labels)
            lyr_coefs[pk] = lr.coef_[0]

        # Geometric mean
        abs_c = np.stack([np.abs(lyr_coefs[pk]) for pk in PARADIGM_KEYS])
        log_gm = np.log(abs_c + 1e-30).mean(axis=0)
        gm = np.exp(log_gm)
        top20 = np.argsort(-gm)[:20]

        signs_lyr = np.stack([np.sign(lyr_coefs[pk]) for pk in PARADIGM_KEYS])
        cons_lyr = np.all(signs_lyr == signs_lyr[0:1, :], axis=0)

        top20_info = []
        print(f"  {'Rank':<6}{'Feature':<12}{'GeoMean':<12}{'SignOK'}")
        for rank, fidx in enumerate(top20):
            info = {
                "feature_idx": int(fidx),
                "rank": rank,
                "geo_mean_importance": float(gm[fidx]),
                "consistent_sign": bool(cons_lyr[fidx]),
                "paradigm_coefs": {pk: float(lyr_coefs[pk][fidx]) for pk in PARADIGM_KEYS},
            }
            top20_info.append(info)
            if rank < 10:
                print(f"  {rank:<6}{fidx:<12}{gm[fidx]:<12.6f}"
                      f"{'YES' if cons_lyr[fidx] else 'NO'}")

        n_cons = sum(1 for x in top20_info if x["consistent_sign"])
        print(f"  Sign consistency: {n_cons}/20")

        # Check overlap with L22 top-50
        l22_set = set(int(x["feature_idx"]) for x in top50_info)
        lyr_set = set(int(x["feature_idx"]) for x in top20_info)
        overlap = l22_set & lyr_set
        print(f"  Overlap with L22 top-50: {len(overlap)} features")

        multi_layer_results[f"L{lyr}"] = {
            "top20_features": top20_info,
            "sign_consistency": f"{n_cons}/20",
            "overlap_with_L22_top50": list(overlap),
        }

    results["multi_layer"] = multi_layer_results
    return results, paradigm_data


# ============================================================================
# Part B: Feature Characterization
# ============================================================================

def part_b_characterization(paradigm_data, part_a_results):
    print("\n" + "=" * 80)
    print("PART B: Feature Characterization (Top-20 L22)")
    print("=" * 80)

    top20_features = part_a_results["L22_top50"]["features"][:20]
    feature_indices = [f["feature_idx"] for f in top20_features]

    results = {}

    # --- B1: BK vs non-BK activation + Cohen's d ---
    print("\n--- B1: BK vs non-BK Mean Activation (L22 Decision Point) ---")
    print(f"  {'Feature':<12}{'IC_d':<10}{'SM_d':<10}{'MW_d':<10}"
          f"{'IC_BK':<12}{'IC_nBK':<12}{'SM_BK':<12}{'SM_nBK':<12}"
          f"{'MW_BK':<12}{'MW_nBK':<12}")
    print("  " + "-" * 110)

    char_data = []
    for fidx in feature_indices:
        row = {"feature_idx": fidx, "cohens_d": {}, "mean_bk": {}, "mean_nbk": {}}
        for pk in PARADIGM_KEYS:
            feat, labels = paradigm_data[pk]
            bk_vals = feat[labels == 1, fidx]
            nbk_vals = feat[labels == 0, fidx]
            d = cohens_d(bk_vals, nbk_vals)
            row["cohens_d"][pk] = float(d)
            row["mean_bk"][pk] = float(np.mean(bk_vals)) if len(bk_vals) > 0 else 0.0
            row["mean_nbk"][pk] = float(np.mean(nbk_vals)) if len(nbk_vals) > 0 else 0.0
        char_data.append(row)

        print(f"  {fidx:<12}"
              f"{row['cohens_d']['ic']:<10.3f}{row['cohens_d']['sm']:<10.3f}"
              f"{row['cohens_d']['mw']:<10.3f}"
              f"{row['mean_bk']['ic']:<12.4f}{row['mean_nbk']['ic']:<12.4f}"
              f"{row['mean_bk']['sm']:<12.4f}{row['mean_nbk']['sm']:<12.4f}"
              f"{row['mean_bk']['mw']:<12.4f}{row['mean_nbk']['mw']:<12.4f}")

    results["bk_vs_nbk"] = char_data

    # --- B2: Correlation with balance ---
    print("\n--- B2: Correlation with Balance at Decision Point ---")

    balance_corr = []
    for pk in PARADIGM_KEYS:
        # Load with metadata (need balances)
        feat_full, meta = load_layer_features(pk, 22, mode="decision_point")
        labels = get_labels(meta)
        balances = meta.get("balances", None)
        if balances is None:
            print(f"  {PARADIGM_NAMES[pk]}: No balance data available")
            for fidx in feature_indices:
                balance_corr.append({
                    "feature_idx": fidx, "paradigm": pk,
                    "balance_corr": None, "balance_p": None
                })
            continue

        balances = balances.astype(np.float64)
        print(f"\n  {PARADIGM_NAMES[pk]}:")
        print(f"    {'Feature':<12}{'Balance_r':<12}{'Balance_p':<14}{'Significant'}")
        for fidx in feature_indices:
            vals = feat_full[:, fidx].astype(np.float64)
            if np.std(vals) < 1e-12 or np.std(balances) < 1e-12:
                r, p = 0.0, 1.0
            else:
                r, p = stats.pearsonr(vals, balances)
            balance_corr.append({
                "feature_idx": fidx, "paradigm": pk,
                "balance_corr": float(r), "balance_p": float(p),
            })
            sig = "YES" if p < 0.01 else ""
            print(f"    {fidx:<12}{r:<12.4f}{p:<14.6f}{sig}")

    results["balance_correlation"] = balance_corr

    # --- B3: R1 analysis (first round only) ---
    print("\n--- B3: R1-Only Analysis (Future-BK vs Future-non-BK) ---")

    r1_results = []
    for pk in PARADIGM_KEYS:
        feat_all, meta_all = load_layer_features(pk, 22, mode="all_rounds")
        game_outcomes = meta_all["game_outcomes"]
        round_nums = meta_all["round_nums"]

        # R1 mask
        r1_mask = round_nums == 1
        feat_r1 = feat_all[r1_mask]
        outcomes_r1 = game_outcomes[r1_mask]
        labels_r1 = (outcomes_r1 == "bankruptcy").astype(np.int32)

        n_bk_r1 = labels_r1.sum()
        n_nbk_r1 = (1 - labels_r1).sum()
        print(f"\n  {PARADIGM_NAMES[pk]} R1: {len(labels_r1)} games "
              f"(BK={n_bk_r1}, non-BK={n_nbk_r1})")

        if n_bk_r1 < 5:
            print(f"    Too few BK for R1 analysis, skipping")
            r1_results.append({"paradigm": pk, "skipped": True, "n_bk": int(n_bk_r1)})
            continue

        print(f"    {'Feature':<12}{'R1_d':<10}{'R1_BK_mean':<14}{'R1_nBK_mean':<14}")
        r1_info = {"paradigm": pk, "skipped": False, "n_bk": int(n_bk_r1), "features": []}
        for fidx in feature_indices:
            bk_vals = feat_r1[labels_r1 == 1, fidx]
            nbk_vals = feat_r1[labels_r1 == 0, fidx]
            d = cohens_d(bk_vals, nbk_vals)
            r1_info["features"].append({
                "feature_idx": fidx,
                "cohens_d_r1": float(d),
                "mean_bk_r1": float(np.mean(bk_vals)),
                "mean_nbk_r1": float(np.mean(nbk_vals)),
            })
            print(f"    {fidx:<12}{d:<10.3f}{np.mean(bk_vals):<14.4f}{np.mean(nbk_vals):<14.4f}")

        # Also compute AUC with top-20 features at R1
        X_r1_sub = feat_r1[:, feature_indices]
        scaler = StandardScaler()
        X_r1_s = scaler.fit_transform(X_r1_sub)
        lr = LogisticRegression(C=1.0, solver="lbfgs", class_weight="balanced",
                                max_iter=2000, random_state=RANDOM_SEED)
        lr.fit(X_r1_s, labels_r1)
        auc_r1 = safe_auc(labels_r1, lr.decision_function(X_r1_s))
        r1_info["auc_top20_r1"] = float(auc_r1)
        print(f"    R1 AUC (top-20 features): {auc_r1:.4f}")

        r1_results.append(r1_info)

    results["r1_analysis"] = r1_results
    return results


# ============================================================================
# Part C: Hidden State Cross-Domain Analysis
# ============================================================================

def part_c_hidden_state_analysis():
    print("\n" + "=" * 80)
    print("PART C: Hidden State Cross-Domain Analysis (L22)")
    print("=" * 80)

    layer = 22
    results = {}

    # C1: Load hidden states ONCE per paradigm, cache raw arrays
    print("\n--- C1: Train LR on PCA-50 Hidden States ---")
    hs_raw_cache = {}  # {pk: (raw_hs, labels)} — loaded once
    hs_data = {}       # {pk: (pca_transformed, labels, scaled)}
    bk_directions = {} # in PCA space
    pca_models = {}
    scaler_models = {}

    for pk in PARADIGM_KEYS:
        print(f"  Loading {PARADIGM_NAMES[pk]} hidden states L{layer}...")
        hs, meta = load_hidden_states(pk, layer, mode="decision_point")
        labels = get_labels(meta)
        print(f"    Shape: {hs.shape}, BK: {labels.sum()}, non-BK: {(1 - labels).sum()}")

        # Cache raw data
        hs_raw_cache[pk] = (hs, labels)

        # PCA 50
        scaler = StandardScaler()
        hs_s = scaler.fit_transform(hs)
        pca = PCA(n_components=50, random_state=RANDOM_SEED)
        hs_pca = pca.fit_transform(hs_s)
        var_explained = pca.explained_variance_ratio_.sum()
        print(f"    PCA-50 variance explained: {var_explained:.4f}")

        # Train LR
        lr = LogisticRegression(C=1.0, solver="lbfgs", class_weight="balanced",
                                max_iter=2000, random_state=RANDOM_SEED)
        lr.fit(hs_pca, labels)
        auc = safe_auc(labels, lr.decision_function(hs_pca))
        print(f"    PCA-50 LR AUC: {auc:.4f}")

        hs_data[pk] = (hs_pca, labels, hs_s)
        bk_directions[pk] = lr.coef_[0]  # (50,) weight vector = BK direction in PCA space
        pca_models[pk] = pca
        scaler_models[pk] = scaler

        results[f"{pk}_pca50"] = {
            "auc": float(auc),
            "variance_explained": float(var_explained),
            "n_samples": int(len(labels)),
            "n_bk": int(labels.sum()),
        }

    # C2: Cross-projection — project each paradigm's data onto others' BK directions
    # Uses cached raw hidden states instead of reloading from disk
    print("\n--- C2: Cross-Domain BK Direction Transfer ---")
    print(f"  {'Source->Target':<25}{'AUC_self':<12}{'AUC_transfer':<14}{'Ratio'}")
    print("  " + "-" * 65)

    transfer_results = {}
    for src in PARADIGM_KEYS:
        bk_dir = bk_directions[src]
        bk_dir_norm = bk_dir / (np.linalg.norm(bk_dir) + 1e-10)

        for tgt in PARADIGM_KEYS:
            # Use cached raw data
            hs_tgt_raw, labels_tgt = hs_raw_cache[tgt]

            # Scale with source scaler, project with source PCA
            hs_tgt_s = scaler_models[src].transform(hs_tgt_raw)
            hs_tgt_pca = pca_models[src].transform(hs_tgt_s)

            # Project onto BK direction
            projections = hs_tgt_pca @ bk_dir_norm
            auc_transfer = safe_auc(labels_tgt, projections)

            # Self AUC (source data projected onto own direction)
            hs_self_pca = hs_data[src][0]
            labels_self = hs_data[src][1]
            proj_self = hs_self_pca @ bk_dir_norm
            auc_self = safe_auc(labels_self, proj_self)

            ratio = auc_transfer / auc_self if auc_self > 0 else 0.0
            key = f"{src}->{tgt}"
            transfer_results[key] = {
                "auc_self": float(auc_self),
                "auc_transfer": float(auc_transfer),
                "ratio": float(ratio),
            }
            print(f"  {key:<25}{auc_self:<12.4f}{auc_transfer:<14.4f}{ratio:<.4f}")

    results["cross_projection"] = transfer_results

    # C3: Shared BK subspace analysis
    print("\n--- C3: Shared BK Subspace ---")

    # Transform BK directions back to original (scaled) space
    bk_dirs_original = {}
    for pk in PARADIGM_KEYS:
        bk_dir_pca = bk_directions[pk]  # (50,)
        bk_dir_orig = pca_models[pk].inverse_transform(bk_dir_pca.reshape(1, -1)) - \
                       pca_models[pk].inverse_transform(np.zeros((1, 50)))
        bk_dir_orig = bk_dir_orig.flatten()
        bk_dir_orig = bk_dir_orig / (np.linalg.norm(bk_dir_orig) + 1e-10)
        bk_dirs_original[pk] = bk_dir_orig

    # Stack BK directions and compute their PCA
    bk_stack = np.stack([bk_dirs_original[pk] for pk in PARADIGM_KEYS])  # (3, 3584)
    print(f"  BK direction cosine similarities:")
    for i, pk1 in enumerate(PARADIGM_KEYS):
        for j, pk2 in enumerate(PARADIGM_KEYS):
            if j > i:
                cos = np.dot(bk_dirs_original[pk1], bk_dirs_original[pk2])
                print(f"    {PARADIGM_NAMES[pk1]} vs {PARADIGM_NAMES[pk2]}: {cos:.4f}")

    # Shared subspace = PCA of the 3 BK directions (max rank 3)
    pca_bk = PCA(n_components=3, random_state=RANDOM_SEED)
    pca_bk.fit(bk_stack)
    shared_components = pca_bk.components_  # (3, 3584)
    var_ratio = pca_bk.explained_variance_ratio_
    print(f"\n  BK subspace PCA variance: {var_ratio}")
    print(f"  Top-1 component captures {var_ratio[0]:.4f} of BK direction variance")
    print(f"  Top-2 components capture {var_ratio[:2].sum():.4f} of BK direction variance")

    # Compute AUC using shared subspace vs full PCA-50 (use cached data)
    print(f"\n  AUC comparison: Shared BK subspace vs Full PCA-50")
    print(f"  {'Paradigm':<25}{'Full_PCA50':<14}{'Shared_1D':<14}{'Shared_2D':<14}{'Shared_3D':<14}")
    subspace_aucs = {}
    for pk in PARADIGM_KEYS:
        hs_raw, labels = hs_raw_cache[pk]
        hs_s = scaler_models[pk].transform(hs_raw)

        aucs_by_dim = {}
        for n_dim in [1, 2, 3]:
            proj = hs_s @ shared_components[:n_dim].T
            lr_sh = LogisticRegression(C=1.0, solver="lbfgs", class_weight="balanced",
                                       max_iter=2000, random_state=RANDOM_SEED)
            lr_sh.fit(proj, labels)
            auc_sh = safe_auc(labels, lr_sh.decision_function(proj))
            aucs_by_dim[f"{n_dim}D"] = float(auc_sh)

        full_auc = results[f"{pk}_pca50"]["auc"]
        subspace_aucs[pk] = aucs_by_dim
        print(f"  {PARADIGM_NAMES[pk]:<25}{full_auc:<14.4f}"
              f"{aucs_by_dim['1D']:<14.4f}{aucs_by_dim['2D']:<14.4f}"
              f"{aucs_by_dim['3D']:<14.4f}")

    results["shared_subspace"] = {
        "variance_ratio": var_ratio.tolist(),
        "cosine_similarities": {
            f"{pk1}_vs_{pk2}": float(np.dot(bk_dirs_original[pk1], bk_dirs_original[pk2]))
            for i, pk1 in enumerate(PARADIGM_KEYS)
            for j, pk2 in enumerate(PARADIGM_KEYS) if j > i
        },
        "subspace_aucs": subspace_aucs,
    }

    # C4: Decompose variance — shared vs paradigm-specific (use cached data)
    print(f"\n--- C4: Shared vs Paradigm-Specific Variance ---")
    for pk in PARADIGM_KEYS:
        hs_raw, labels = hs_raw_cache[pk]
        hs_s = scaler_models[pk].transform(hs_raw)

        # Project onto shared subspace (3D)
        hs_shared = hs_s @ shared_components.T @ shared_components  # (n, 3584)
        hs_residual = hs_s - hs_shared

        # AUC from residual only
        pca_res = PCA(n_components=50, random_state=RANDOM_SEED)
        hs_res_pca = pca_res.fit_transform(hs_residual)
        lr_res = LogisticRegression(C=1.0, solver="lbfgs", class_weight="balanced",
                                     max_iter=2000, random_state=RANDOM_SEED)
        lr_res.fit(hs_res_pca, labels)
        auc_res = safe_auc(labels, lr_res.decision_function(hs_res_pca))

        full_auc = results[f"{pk}_pca50"]["auc"]
        shared_3d = subspace_aucs[pk]["3D"]
        print(f"  {PARADIGM_NAMES[pk]}: Full={full_auc:.4f}, "
              f"Shared(3D)={shared_3d:.4f}, Residual(PCA50)={auc_res:.4f}")

        results[f"{pk}_decomposition"] = {
            "full_auc": float(full_auc),
            "shared_3d_auc": float(shared_3d),
            "residual_auc": float(auc_res),
        }

    # Free cached raw data
    del hs_raw_cache

    return results


# ============================================================================
# Part D: Hidden State Neuron-Level Analysis
# ============================================================================

def part_d_neuron_analysis():
    print("\n" + "=" * 80)
    print("PART D: Hidden State Neuron-Level Analysis (L22)")
    print("=" * 80)

    layer = 22
    results = {}

    # D1: Point-biserial correlation for each of 3584 neurons
    print("\n--- D1: Per-Neuron BK Correlation ---")

    corr_data = {}
    pvals_data = {}

    for pk in PARADIGM_KEYS:
        print(f"  Computing correlations for {PARADIGM_NAMES[pk]}...")
        hs, meta = load_hidden_states(pk, layer, mode="decision_point")
        labels = get_labels(meta)
        n_dim = hs.shape[1]

        corrs = np.zeros(n_dim)
        pvals = np.zeros(n_dim)

        for d in range(n_dim):
            r, p = stats.pointbiserialr(labels, hs[:, d].astype(np.float64))
            corrs[d] = r
            pvals[d] = p

        # FDR correction
        sig_mask = fdr_correction(pvals, alpha=0.01)
        n_sig = sig_mask.sum()
        print(f"    Significant (FDR p<0.01): {n_sig}/{n_dim} neurons")

        corr_data[pk] = corrs
        pvals_data[pk] = pvals

    # D2: Find neurons significant in ALL 3 paradigms with consistent sign
    print("\n--- D2: Universal BK Neurons ---")

    n_dim = len(corr_data["ic"])
    all_sig = np.ones(n_dim, dtype=bool)
    for pk in PARADIGM_KEYS:
        sig = fdr_correction(pvals_data[pk], alpha=0.01)
        all_sig &= sig

    n_universal = all_sig.sum()
    print(f"  Neurons significant in ALL 3 paradigms: {n_universal}/{n_dim}")

    # Check sign consistency among universal neurons
    if n_universal > 0:
        universal_idx = np.where(all_sig)[0]
        signs = np.stack([np.sign(corr_data[pk][universal_idx]) for pk in PARADIGM_KEYS])
        sign_consistent = np.all(signs == signs[0:1, :], axis=0)
        n_sign_ok = sign_consistent.sum()
        print(f"  Sign-consistent universal neurons: {n_sign_ok}/{n_universal}")

        # Rank by minimum |correlation| across 3 paradigms
        min_abs_corr = np.stack([np.abs(corr_data[pk][universal_idx])
                                  for pk in PARADIGM_KEYS]).min(axis=0)
        rank_order = np.argsort(-min_abs_corr)

        print(f"\n  Top-30 Universal BK Neurons:")
        print(f"  {'Rank':<6}{'Neuron':<10}{'MinAbsR':<10}"
              f"{'IC_r':<10}{'SM_r':<10}{'MW_r':<10}"
              f"{'IC_p':<12}{'SM_p':<12}{'MW_p':<12}{'SignOK'}")
        print("  " + "-" * 100)

        top30_neurons = []
        for rank in range(min(30, len(rank_order))):
            idx_in_universal = rank_order[rank]
            neuron_idx = int(universal_idx[idx_in_universal])
            min_r = float(min_abs_corr[idx_in_universal])
            info = {
                "neuron_idx": neuron_idx,
                "rank": rank,
                "min_abs_corr": min_r,
                "correlations": {pk: float(corr_data[pk][neuron_idx]) for pk in PARADIGM_KEYS},
                "pvalues": {pk: float(pvals_data[pk][neuron_idx]) for pk in PARADIGM_KEYS},
                "consistent_sign": bool(sign_consistent[idx_in_universal]),
            }
            top30_neurons.append(info)

            print(f"  {rank:<6}{neuron_idx:<10}{min_r:<10.4f}"
                  f"{corr_data['ic'][neuron_idx]:<10.4f}"
                  f"{corr_data['sm'][neuron_idx]:<10.4f}"
                  f"{corr_data['mw'][neuron_idx]:<10.4f}"
                  f"{pvals_data['ic'][neuron_idx]:<12.6e}"
                  f"{pvals_data['sm'][neuron_idx]:<12.6e}"
                  f"{pvals_data['mw'][neuron_idx]:<12.6e}"
                  f"{'YES' if sign_consistent[idx_in_universal] else 'NO'}")

        results["universal_neurons"] = {
            "n_total_neurons": n_dim,
            "n_universal": int(n_universal),
            "n_sign_consistent": int(n_sign_ok),
            "top30": top30_neurons,
        }
    else:
        print("  No neurons significant in all 3 paradigms!")
        results["universal_neurons"] = {
            "n_total_neurons": n_dim,
            "n_universal": 0,
            "n_sign_consistent": 0,
            "top30": [],
        }

    # D3: Summary statistics across paradigms
    print("\n--- D3: Per-Paradigm Neuron Statistics ---")
    for pk in PARADIGM_KEYS:
        sig = fdr_correction(pvals_data[pk], alpha=0.01)
        n_pos = (sig & (corr_data[pk] > 0)).sum()
        n_neg = (sig & (corr_data[pk] < 0)).sum()
        max_r = np.max(np.abs(corr_data[pk]))
        median_sig = np.median(np.abs(corr_data[pk][sig])) if sig.any() else 0
        print(f"  {PARADIGM_NAMES[pk]}: {sig.sum()} sig neurons "
              f"(+{n_pos}/-{n_neg}), max|r|={max_r:.4f}, "
              f"median|r| of sig={median_sig:.4f}")

        results[f"{pk}_stats"] = {
            "n_significant": int(sig.sum()),
            "n_positive": int(n_pos),
            "n_negative": int(n_neg),
            "max_abs_r": float(max_r),
            "median_abs_r_sig": float(median_sig),
        }

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    start = datetime.now()
    print(f"V7 Cross-Domain Feature Analysis")
    print(f"Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Paradigms: {', '.join(PARADIGM_NAMES[pk] for pk in PARADIGM_KEYS)}")

    all_results = {"timestamp": start.isoformat(), "random_seed": RANDOM_SEED}

    # Part A
    part_a, paradigm_data = part_a_cross_domain_features()
    all_results["part_a"] = part_a

    # Part B
    part_b = part_b_characterization(paradigm_data, part_a)
    all_results["part_b"] = part_b

    # Part C
    part_c = part_c_hidden_state_analysis()
    all_results["part_c"] = part_c

    # Part D
    part_d = part_d_neuron_analysis()
    all_results["part_d"] = part_d

    # Save results
    output_path = JSON_DIR / "v7_cross_domain_features.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    # Final summary
    elapsed = (datetime.now() - start).total_seconds()
    print(f"\n{'=' * 80}")
    print(f"SUMMARY")
    print(f"{'=' * 80}")

    # Part A summary
    top5 = part_a["L22_top50"]["features"][:5]
    n_consistent = sum(1 for f in part_a["L22_top50"]["features"] if f["consistent_sign"])
    print(f"\nPart A: Top-5 cross-domain features at L22:")
    for f in top5:
        print(f"  Feature {f['feature_idx']}: geo_mean={f['geo_mean_importance']:.6f}, "
              f"ranks=[IC:{f['paradigm_ranks']['ic']}, SM:{f['paradigm_ranks']['sm']}, "
              f"MW:{f['paradigm_ranks']['mw']}], sign={'consistent' if f['consistent_sign'] else 'MIXED'}")
    print(f"  Sign consistency: {n_consistent}/50")
    print(f"  Top-50 AUCs: " + ", ".join(
        f"{PARADIGM_NAMES[pk]}={part_a['L22_top50']['top50_aucs'][pk]:.4f}"
        for pk in PARADIGM_KEYS))

    # Part B summary
    d_vals = part_b["bk_vs_nbk"][:5]
    print(f"\nPart B: Effect sizes (Cohen's d) for top-5 features:")
    for d in d_vals:
        print(f"  Feature {d['feature_idx']}: "
              f"IC={d['cohens_d']['ic']:.3f}, SM={d['cohens_d']['sm']:.3f}, "
              f"MW={d['cohens_d']['mw']:.3f}")

    # Part C summary
    print(f"\nPart C: Hidden state BK direction analysis:")
    cs = part_c.get("shared_subspace", {}).get("cosine_similarities", {})
    for key, val in cs.items():
        print(f"  BK direction cosine ({key}): {val:.4f}")
    for pk in PARADIGM_KEYS:
        decomp = part_c.get(f"{pk}_decomposition", {})
        if decomp:
            print(f"  {PARADIGM_NAMES[pk]}: Full={decomp['full_auc']:.4f}, "
                  f"Shared3D={decomp['shared_3d_auc']:.4f}, "
                  f"Residual={decomp['residual_auc']:.4f}")

    # Part D summary
    univ = part_d.get("universal_neurons", {})
    print(f"\nPart D: Universal BK neurons: {univ.get('n_universal', 0)}/{univ.get('n_total_neurons', 0)}")
    print(f"  Sign-consistent: {univ.get('n_sign_consistent', 0)}")
    if univ.get("top30"):
        top3 = univ["top30"][:3]
        for n in top3:
            print(f"  Neuron {n['neuron_idx']}: min|r|={n['min_abs_corr']:.4f}, "
                  f"IC={n['correlations']['ic']:.4f}, "
                  f"SM={n['correlations']['sm']:.4f}, "
                  f"MW={n['correlations']['mw']:.4f}")

    print(f"\nElapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
