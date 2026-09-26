#!/usr/bin/env python3
"""
Final verification analyses for V9 report (hallucination-free).

Replaces unverified claims in V9 with actual computed numbers:
1. Gemma cross-domain SAE transfer (IC→SM, IC→MW, SM→MW) with permutation test
2. Llama SM per-layer BK-differential statistics (compare with IC)
3. Factor decomposition via SAE features: outcome + bet_type + paradigm regression
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from config import *
from data_loader import load_layer_features, get_labels, filter_active_features

np.random.seed(RANDOM_SEED)

LLAMA_SM_DIR = DATA_ROOT / "sae_features_v3" / "slot_machine" / "llama"
LLAMA_IC_DIR = DATA_ROOT / "sae_features_v3" / "investment_choice" / "llama"


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ============================================================
# Analysis A: Gemma Cross-Domain SAE Transfer
# ============================================================

def gemma_crossdomain_transfer():
    """Compute Gemma IC/SM/MW cross-domain transfer AUC with permutation test."""
    log("=" * 70)
    log("ANALYSIS A: Gemma Cross-Domain SAE Transfer")
    log("=" * 70)

    N_PERM = 200
    LAYER = 22  # Primary layer; also check others
    layers_to_check = [18, 22, 26, 30]

    results = {}

    for layer in layers_to_check:
        log(f"\n  Layer {layer}:")
        paradigm_data = {}

        for p in ["ic", "sm", "mw"]:
            res = load_layer_features(p, layer, mode="decision_point")
            if res is None:
                log(f"    {p}: not found")
                continue
            feats, meta = res
            labels = get_labels(meta)
            n_bk = labels.sum()
            log(f"    {p}: {len(labels)} games, {n_bk} BK ({n_bk/len(labels)*100:.1f}%)")

            if n_bk < 10:
                log(f"    {p}: too few BK, skipping")
                continue

            # Filter active features
            active_rate = (feats != 0).mean(axis=0)
            active = active_rate >= MIN_ACTIVATION_RATE
            paradigm_data[p] = {
                "feats": feats[:, active],
                "labels": labels,
                "active_mask": active,
                "n_bk": int(n_bk),
            }

        # Determine common active features across paradigms
        available = list(paradigm_data.keys())
        if len(available) < 2:
            log(f"    L{layer}: insufficient paradigms")
            continue

        # Find common active mask
        common_active = np.ones(N_SAE_FEATURES_GEMMA, dtype=bool)
        for p in available:
            common_active &= paradigm_data[p]["active_mask"]

        n_common = common_active.sum()
        log(f"    Common active features: {n_common}")

        if n_common < 10:
            log(f"    Too few common features, skipping")
            continue

        # Refilter with common mask
        for p in available:
            res = load_layer_features(p, layer, mode="decision_point")
            feats, meta = res
            paradigm_data[p]["feats_common"] = feats[:, common_active]

        # Cross-domain transfer
        pairs = []
        if "ic" in available and "sm" in available:
            pairs.append(("ic", "sm"))
        if "ic" in available and "mw" in available:
            pairs.append(("ic", "mw"))
        if "sm" in available and "mw" in available:
            pairs.append(("sm", "mw"))
        # Also reverse
        if "sm" in available and "ic" in available:
            pairs.append(("sm", "ic"))
        if "mw" in available and "ic" in available:
            pairs.append(("mw", "ic"))

        for train_p, test_p in pairs:
            if train_p not in paradigm_data or test_p not in paradigm_data:
                continue

            X_train = paradigm_data[train_p]["feats_common"]
            y_train = paradigm_data[train_p]["labels"]
            X_test = paradigm_data[test_p]["feats_common"]
            y_test = paradigm_data[test_p]["labels"]

            n_bk_train = y_train.sum()
            n_bk_test = y_test.sum()

            if n_bk_train < 10 or n_bk_test < 5:
                log(f"    {train_p}→{test_p}: insufficient BK (train={n_bk_train}, test={n_bk_test})")
                continue

            try:
                # Standardize on train, apply to test
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)

                # PCA on train
                n_comp = min(50, X_train_s.shape[0] - 1, X_train_s.shape[1])
                pca = PCA(n_components=n_comp, random_state=RANDOM_SEED)
                X_train_pca = pca.fit_transform(X_train_s)
                X_test_pca = pca.transform(X_test_s)

                # Train classifier
                clf = LogisticRegression(C=1.0, class_weight='balanced',
                                         max_iter=1000, random_state=RANDOM_SEED)
                clf.fit(X_train_pca, y_train)

                # Predict on test
                proba = clf.predict_proba(X_test_pca)[:, 1]
                auc_obs = roc_auc_score(y_test, proba)

                # Permutation test
                null_aucs = []
                for _ in range(N_PERM):
                    y_perm = np.random.permutation(y_test)
                    null_auc = roc_auc_score(y_perm, proba)
                    null_aucs.append(null_auc)
                null_aucs = np.array(null_aucs)
                perm_p = (null_aucs >= auc_obs).mean()

                key = f"L{layer}_{train_p.upper()}_{test_p.upper()}"
                results[key] = {
                    "layer": layer,
                    "train": train_p,
                    "test": test_p,
                    "auc": float(auc_obs),
                    "perm_p": float(perm_p),
                    "null_mean": float(null_aucs.mean()),
                    "n_train": len(y_train),
                    "n_test": len(y_test),
                    "bk_train": int(n_bk_train),
                    "bk_test": int(n_bk_test),
                    "n_common_features": int(n_common),
                }
                log(f"    {train_p}→{test_p}: AUC={auc_obs:.3f}, perm_p={perm_p:.3f}, null_mean={null_aucs.mean():.3f}")

            except Exception as e:
                log(f"    {train_p}→{test_p}: error — {e}")

    return results


# ============================================================
# Analysis B: Llama SM Per-Layer BK-Differential Stats
# ============================================================

def llama_sm_bk_differential():
    """Compute per-layer BK-differential for Llama SM; compare with Llama IC."""
    log("\n" + "=" * 70)
    log("ANALYSIS B: Llama SM Per-Layer BK-Differential")
    log("=" * 70)

    results = {}
    n_layers = len(list(LLAMA_SM_DIR.glob("sae_features_L*.npz")))
    log(f"  Llama SM: {n_layers} SAE layer files")

    for layer in range(32):
        npz_path = LLAMA_SM_DIR / f"sae_features_L{layer}.npz"
        if not npz_path.exists():
            continue

        raw = np.load(npz_path, allow_pickle=False)
        data = {k: raw[k] for k in raw.files}
        shape = tuple(data["shape"])
        dense = np.zeros(shape, dtype=np.float32)
        dense[data["row_indices"], data["col_indices"]] = data["values"]

        is_last = data["is_last_round"].astype(bool)
        feats = dense[is_last]
        outcomes = data["game_outcomes"][is_last]
        labels = (outcomes == "bankruptcy").astype(int)

        n_bk = labels.sum()
        if n_bk < 5:
            continue

        bk_feats = feats[labels == 1]
        safe_feats = feats[labels == 0]
        bk_mean = bk_feats.mean(axis=0)
        safe_mean = safe_feats.mean(axis=0)
        pooled_std = np.sqrt((bk_feats.std(axis=0)**2 + safe_feats.std(axis=0)**2) / 2) + 1e-10
        cohens_d = (bk_mean - safe_mean) / pooled_std

        active = (feats != 0).mean(axis=0) >= 0.01
        n_active = active.sum()

        strong = active & (np.abs(cohens_d) >= 0.3)
        bk_promoting = active & (cohens_d >= 0.3)
        bk_inhibiting = active & (cohens_d <= -0.3)

        results[layer] = {
            "n_active": int(n_active),
            "n_strong_d03": int(strong.sum()),
            "n_bk_promoting": int(bk_promoting.sum()),
            "n_bk_inhibiting": int(bk_inhibiting.sum()),
            "n_bk": int(n_bk),
        }

        if layer % 5 == 0:
            log(f"  L{layer}: active={n_active}, strong(d≥0.3)={strong.sum()} (BK+:{bk_promoting.sum()}, BK-:{bk_inhibiting.sum()}), n_bk={n_bk}")

    if results:
        peak = max(results.items(), key=lambda x: x[1]["n_strong_d03"])
        total_strong = sum(r["n_strong_d03"] for r in results.values())
        log(f"\n  Peak layer: L{peak[0]} ({peak[1]['n_strong_d03']} strong), Total: {total_strong}")

    return results


# ============================================================
# Analysis C: Factor Decomposition (SAE features, combined paradigms)
# ============================================================

def factor_decomposition_sae():
    """
    Regression: SAE feature ~ outcome + bet_type + paradigm
    Test: what % of active SAE features are significantly predicted by outcome
    after controlling for bet_type and paradigm.
    Uses L22 (primary layer).
    """
    log("\n" + "=" * 70)
    log("ANALYSIS C: Factor Decomposition via SAE Features (L22)")
    log("=" * 70)

    layer = 22
    results = {}

    # Load per-paradigm data
    paradigm_data = {}
    for p in ["ic", "sm", "mw"]:
        res = load_layer_features(p, layer, mode="decision_point")
        if res is None:
            log(f"  {p}: not found")
            continue
        feats, meta = res
        labels = get_labels(meta)
        bet_types = (meta["bet_types"] == "variable").astype(float)

        paradigm_data[p] = {
            "feats": feats,
            "labels": labels,
            "bet_types": bet_types,
            "meta": meta,
        }
        log(f"  {p}: {len(labels)} games, {labels.sum()} BK")

    if len(paradigm_data) < 2:
        log("  Insufficient data for factor decomposition")
        return {}

    # Build combined dataset
    available = list(paradigm_data.keys())

    # Find common active features
    active_per_paradigm = {}
    for p in available:
        rate = (paradigm_data[p]["feats"] != 0).mean(axis=0)
        active_per_paradigm[p] = rate >= MIN_ACTIVATION_RATE

    # Union of active features (active in ANY paradigm)
    union_active = np.zeros(N_SAE_FEATURES_GEMMA, dtype=bool)
    for p in available:
        union_active |= active_per_paradigm[p]
    n_active = union_active.sum()
    log(f"  Union active features: {n_active}")

    # Combine data
    feats_list = []
    labels_list = []
    bettype_list = []
    paradigm_list = []

    paradigm_code = {"ic": 0, "sm": 1, "mw": 2}
    for p in available:
        feats_sub = paradigm_data[p]["feats"][:, union_active]
        feats_list.append(feats_sub)
        labels_list.append(paradigm_data[p]["labels"])
        bettype_list.append(paradigm_data[p]["bet_types"])
        paradigm_list.append(np.full(len(paradigm_data[p]["labels"]), paradigm_code[p]))

    X_feats = np.vstack(feats_list)  # (N, K)
    y_outcome = np.concatenate(labels_list)
    y_bettype = np.concatenate(bettype_list)
    y_paradigm = np.concatenate(paradigm_list)
    N = len(y_outcome)

    log(f"  Combined: {N} games, {y_outcome.sum()} BK")

    # Design matrix: [intercept, outcome, bet_type, paradigm_dummy_sm, paradigm_dummy_mw]
    # Paradigm: IC=baseline, SM=dummy1, MW=dummy2
    para_sm = (y_paradigm == 1).astype(float)
    para_mw = (y_paradigm == 2).astype(float)

    X_design = np.column_stack([
        np.ones(N),      # intercept
        y_outcome,       # outcome (BK=1, safe=0)
        y_bettype,       # bet_type (variable=1, fixed=0)
        para_sm,         # paradigm SM dummy
        para_mw,         # paradigm MW dummy
    ])
    # Shape: (N, 5)

    # Vectorized OLS: Beta = (X'X)^{-1} X' Y, residuals, F-stats
    # Y is (N, K) — all active features
    Y = X_feats.astype(np.float64)
    X = X_design.astype(np.float64)

    # OLS: Beta = pinv(X) @ Y
    XtX_inv = np.linalg.pinv(X.T @ X)
    Beta = XtX_inv @ X.T @ Y  # (5, K)

    # Residuals
    Y_hat = X @ Beta  # (N, K)
    Resid = Y - Y_hat  # (N, K)

    # Standard errors: se(beta_j) = sqrt(MSE * (XtX_inv)_{jj})
    df_resid = N - X.shape[1]
    MSE = (Resid**2).sum(axis=0) / df_resid  # (K,)
    # Var(beta_j) = MSE * (XtX_inv)[j, j]
    diag_XtXinv = np.diag(XtX_inv)  # (5,)

    # t-stats for each coefficient
    # Beta shape: (5, K), MSE shape: (K,)
    # SE[j, k] = sqrt(MSE[k] * diag_XtXinv[j])
    t_stats = np.zeros((5, n_active))
    for j in range(5):
        se = np.sqrt(MSE * diag_XtXinv[j] + 1e-30)
        t_stats[j] = Beta[j] / se

    # p-values (two-tailed t-test)
    from scipy.stats import t as t_dist
    p_values = 2 * t_dist.sf(np.abs(t_stats), df=df_resid)  # (5, K)

    # Significant at p < 0.01
    p_thresh = 0.01

    outcome_sig = (p_values[1] < p_thresh).sum()
    bettype_sig = (p_values[2] < p_thresh).sum()
    para_sm_sig = (p_values[3] < p_thresh).sum()
    para_mw_sig = (p_values[4] < p_thresh).sum()
    # Paradigm-significant: sig in either SM or MW dummy
    paradigm_sig = ((p_values[3] < p_thresh) | (p_values[4] < p_thresh)).sum()

    log(f"\n  Combined N={N}, K={n_active} active SAE features")
    log(f"  Outcome-significant (after controlling bet_type+paradigm): {outcome_sig} ({outcome_sig/n_active*100:.1f}%)")
    log(f"  Bet-type-significant (after controlling outcome+paradigm): {bettype_sig} ({bettype_sig/n_active*100:.1f}%)")
    log(f"  Paradigm-significant (SM or MW dummy, after controlling others): {paradigm_sig} ({paradigm_sig/n_active*100:.1f}%)")

    results["L22"] = {
        "n_active": int(n_active),
        "n_combined": N,
        "n_bk": int(y_outcome.sum()),
        "outcome_sig_p01": int(outcome_sig),
        "outcome_sig_pct": float(outcome_sig / n_active * 100),
        "bettype_sig_p01": int(bettype_sig),
        "bettype_sig_pct": float(bettype_sig / n_active * 100),
        "paradigm_sig_p01": int(paradigm_sig),
        "paradigm_sig_pct": float(paradigm_sig / n_active * 100),
        "paradigms_available": available,
    }

    return results


# ============================================================
# Analysis D: Llama SM Classification AUC
# ============================================================

def llama_sm_classification():
    """BK classification AUC for Llama SM, key layers."""
    log("\n" + "=" * 70)
    log("ANALYSIS D: Llama SM Classification AUC")
    log("=" * 70)

    results = {}
    key_layers = [0, 4, 8, 12, 15, 20, 25, 28, 30, 31]

    for layer in key_layers:
        npz_path = LLAMA_SM_DIR / f"sae_features_L{layer}.npz"
        if not npz_path.exists():
            continue

        raw = np.load(npz_path, allow_pickle=False)
        data = {k: raw[k] for k in raw.files}
        shape = tuple(data["shape"])
        dense = np.zeros(shape, dtype=np.float32)
        dense[data["row_indices"], data["col_indices"]] = data["values"]

        is_last = data["is_last_round"].astype(bool)
        feats = dense[is_last]
        outcomes = data["game_outcomes"][is_last]
        labels = (outcomes == "bankruptcy").astype(int)

        n_bk = labels.sum()
        if n_bk < 10:
            continue

        active = (feats != 0).mean(axis=0) >= 0.01
        feats_active = feats[:, active]
        if feats_active.shape[1] < 10:
            continue

        try:
            scaler = StandardScaler()
            X = scaler.fit_transform(feats_active)
            n_comp = min(50, X.shape[0] - 1, X.shape[1])
            pca = PCA(n_components=n_comp, random_state=RANDOM_SEED)
            X_pca = pca.fit_transform(X)

            clf = LogisticRegression(C=1.0, class_weight='balanced',
                                      max_iter=1000, random_state=RANDOM_SEED)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
            scores = cross_val_score(clf, X_pca, labels, cv=cv, scoring='roc_auc')
            auc = scores.mean()

            results[layer] = {
                "auc": float(auc),
                "n_features": int(feats_active.shape[1]),
                "n_bk": int(n_bk),
            }
            log(f"  L{layer}: AUC={auc:.3f} ({feats_active.shape[1]} features, {n_bk} BK)")
        except Exception as e:
            log(f"  L{layer}: error — {e}")

    if results:
        best = max(results.items(), key=lambda x: x[1]["auc"])
        log(f"\n  Best: L{best[0]} AUC={best[1]['auc']:.3f}")

    return results


# ============================================================
# Main
# ============================================================

def main():
    log("=" * 70)
    log("FINAL VERIFICATION ANALYSES FOR V9 REPORT")
    log("=" * 70)

    all_results = {"timestamp": datetime.now().isoformat()}

    # Check data availability
    log("\nData availability:")
    for p_name, p_cfg in PARADIGMS.items():
        sae_dir = p_cfg["sae_dir"]
        n_layers = len(list(sae_dir.glob("sae_features_L*.npz")))
        log(f"  Gemma {p_name}: {n_layers} SAE layers")

    llama_sm_n = len(list(LLAMA_SM_DIR.glob("sae_features_L*.npz")))
    llama_ic_n = len(list(LLAMA_IC_DIR.glob("sae_features_L*.npz")))
    log(f"  Llama SM: {llama_sm_n} SAE layers")
    log(f"  Llama IC: {llama_ic_n} SAE layers")

    all_results["gemma_crossdomain_transfer"] = gemma_crossdomain_transfer()
    all_results["llama_sm_bk_differential"] = llama_sm_bk_differential()
    all_results["factor_decomposition_sae"] = factor_decomposition_sae()
    all_results["llama_sm_classification"] = llama_sm_classification()

    # Save
    out_file = JSON_DIR / f"final_verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, 'w') as f:
        def convert(obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return obj
        json.dump(all_results, f, indent=2, default=convert)

    log(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
