#!/usr/bin/env python3
"""
B1+B2 Brute-Force Analysis: Improve v8 weaknesses.

B1: SAE cross-domain signal strengthening (v8 found only 1 feature)
B2: Variable/Fixed activation-level common risk pattern (v8 found "dissociation")

Uses existing Gemma IC/SM/MW SAE features + hidden states. CPU only.
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from scipy import stats
from scipy.stats import pearsonr, pointbiserialr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, permutation_test_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from config import *
from data_loader import load_layer_features, load_hidden_states, get_labels, filter_active_features, sparse_to_dense, load_sparse_npz, get_metadata

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
JSON_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(RANDOM_SEED)

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ============================================================
# B1: SAE Cross-Domain Signal Strengthening
# ============================================================

def b1a_multilayer_sae_crossdomain():
    """B1-a: Multi-layer aggregation instead of L22 only."""
    log("=== B1-a: Multi-layer SAE cross-domain ===")

    # Top layers from v8 classification AUC (DP): L22, L12, L33, L10, L26
    candidate_layers = [10, 12, 18, 22, 26, 30, 33]
    paradigms = ["ic", "sm", "mw"]

    results = {}

    for layer in candidate_layers:
        log(f"  Layer {layer}...")
        per_paradigm = {}

        for p in paradigms:
            result = load_layer_features(p, layer, mode="decision_point")
            if result is None:
                log(f"    {p} L{layer}: not found")
                continue
            feats, meta = result
            labels = get_labels(meta)
            n_bk = labels.sum()
            if n_bk < 10:
                log(f"    {p} L{layer}: too few BK ({n_bk})")
                continue

            # Per-feature Cohen's d (BK vs safe)
            bk_mask = labels == 1
            safe_mask = labels == 0
            bk_feats = feats[bk_mask]
            safe_feats = feats[safe_mask]

            bk_mean = bk_feats.mean(axis=0)
            safe_mean = safe_feats.mean(axis=0)
            bk_std = bk_feats.std(axis=0) + 1e-10
            safe_std = safe_feats.std(axis=0) + 1e-10
            pooled_std = np.sqrt((bk_std**2 + safe_std**2) / 2)
            cohens_d = (bk_mean - safe_mean) / pooled_std

            # Filter active features
            active_rate = (feats != 0).mean(axis=0)
            active_mask = active_rate >= MIN_ACTIVATION_RATE

            per_paradigm[p] = {
                "cohens_d": cohens_d,
                "active_mask": active_mask,
                "n_bk": int(n_bk),
                "n_safe": int(safe_mask.sum()),
            }

        if len(per_paradigm) < 3:
            continue

        # Find sign-consistent features across all 3 paradigms
        all_active = per_paradigm["ic"]["active_mask"] & per_paradigm["sm"]["active_mask"] & per_paradigm["mw"]["active_mask"]

        d_ic = per_paradigm["ic"]["cohens_d"]
        d_sm = per_paradigm["sm"]["cohens_d"]
        d_mw = per_paradigm["mw"]["cohens_d"]

        # Sign consistent: same sign in all 3, and all active
        sign_consistent = all_active & (
            ((d_ic > 0) & (d_sm > 0) & (d_mw > 0)) |
            ((d_ic < 0) & (d_sm < 0) & (d_mw < 0))
        )

        # Magnitude: geometric mean of |d|
        geo_mean_d = np.cbrt(np.abs(d_ic) * np.abs(d_sm) * np.abs(d_mw))

        # Filter: sign-consistent AND geo_mean >= 0.2
        strong = sign_consistent & (geo_mean_d >= 0.2)
        medium = sign_consistent & (geo_mean_d >= 0.1)

        n_active = all_active.sum()
        n_sign = sign_consistent.sum()
        n_strong = strong.sum()
        n_medium = medium.sum()

        results[layer] = {
            "n_active": int(n_active),
            "n_sign_consistent": int(n_sign),
            "n_strong_d02": int(n_strong),
            "n_medium_d01": int(n_medium),
            "sign_consistent_pct": f"{n_sign/max(n_active,1)*100:.1f}%",
        }

        log(f"    L{layer}: active={n_active}, sign_consistent={n_sign} ({n_sign/max(n_active,1)*100:.1f}%), strong(d≥0.2)={n_strong}, medium(d≥0.1)={n_medium}")

        # Report top features for this layer
        if n_medium > 0:
            medium_idx = np.where(medium)[0]
            top_idx = medium_idx[np.argsort(geo_mean_d[medium_idx])[-5:]]
            for fi in top_idx:
                direction = "BK+" if d_ic[fi] > 0 else "BK-"
                log(f"      Feature #{fi}: d(IC)={d_ic[fi]:.3f}, d(SM)={d_sm[fi]:.3f}, d(MW)={d_mw[fi]:.3f} [{direction}]")

    # Aggregate: union of sign-consistent features across layers
    log("\n  === B1-a SUMMARY ===")
    total_sign = sum(r["n_sign_consistent"] for r in results.values())
    total_medium = sum(r["n_medium_d01"] for r in results.values())
    total_strong = sum(r["n_strong_d02"] for r in results.values())
    log(f"  Total sign-consistent (all layers): {total_sign}")
    log(f"  Total medium (d≥0.1): {total_medium}")
    log(f"  Total strong (d≥0.2): {total_strong}")
    log(f"  v8 baseline: 1 feature (L22 only)")

    return results


def b1d_sparse_logreg_overlap():
    """B1-d: L1-LogReg per paradigm → coefficient overlap."""
    log("\n=== B1-d: Sparse LogReg coefficient overlap ===")

    paradigms = ["ic", "sm", "mw"]
    layer = 22  # Start with L22, the v8 best layer

    coef_sets = {}

    for p in paradigms:
        result = load_layer_features(p, layer, mode="decision_point")
        if result is None:
            log(f"  {p}: data not found")
            continue
        feats, meta = result
        labels = get_labels(meta)
        n_bk = labels.sum()

        if n_bk < 10:
            log(f"  {p}: too few BK ({n_bk})")
            continue

        # Filter active features
        feats_filt, active_idx = filter_active_features(feats, MIN_ACTIVATION_RATE)
        log(f"  {p}: {feats_filt.shape[1]} active features, {n_bk} BK / {len(labels)-n_bk} safe")

        # Standardize
        scaler = StandardScaler()
        X = scaler.fit_transform(feats_filt)

        # L1-LogReg
        clf = LogisticRegression(C=0.1, penalty='l1', solver='saga',
                                 class_weight='balanced', max_iter=2000, random_state=RANDOM_SEED)
        clf.fit(X, labels)

        coefs = clf.coef_[0]
        nonzero = np.abs(coefs) > 1e-6
        n_nonzero = nonzero.sum()

        # Map back to original feature indices
        selected_features = active_idx[nonzero]
        coef_values = coefs[nonzero]

        coef_sets[p] = {
            "features": set(selected_features.tolist()),
            "feature_coefs": dict(zip(selected_features.tolist(), coef_values.tolist())),
            "n_selected": int(n_nonzero),
        }

        log(f"    L1-LogReg selected {n_nonzero} features")

    if len(coef_sets) < 3:
        log("  Insufficient paradigms")
        return {}

    # Overlap analysis
    ic_feats = coef_sets["ic"]["features"]
    sm_feats = coef_sets["sm"]["features"]
    mw_feats = coef_sets["mw"]["features"]

    ic_sm = ic_feats & sm_feats
    ic_mw = ic_feats & mw_feats
    sm_mw = sm_feats & mw_feats
    all_three = ic_feats & sm_feats & mw_feats

    log(f"\n  Overlap: IC∩SM={len(ic_sm)}, IC∩MW={len(ic_mw)}, SM∩MW={len(sm_mw)}, ALL={len(all_three)}")

    # Check sign consistency for overlapping features
    sign_consistent = []
    for fid in all_three:
        ic_sign = np.sign(coef_sets["ic"]["feature_coefs"][fid])
        sm_sign = np.sign(coef_sets["sm"]["feature_coefs"][fid])
        mw_sign = np.sign(coef_sets["mw"]["feature_coefs"][fid])
        if ic_sign == sm_sign == mw_sign:
            sign_consistent.append({
                "feature_id": fid,
                "direction": "BK+" if ic_sign > 0 else "BK-",
                "coef_ic": coef_sets["ic"]["feature_coefs"][fid],
                "coef_sm": coef_sets["sm"]["feature_coefs"][fid],
                "coef_mw": coef_sets["mw"]["feature_coefs"][fid],
            })

    log(f"  Sign-consistent in ALL 3: {len(sign_consistent)}")
    for sc in sorted(sign_consistent, key=lambda x: abs(x["coef_ic"]), reverse=True)[:10]:
        log(f"    Feature #{sc['feature_id']}: IC={sc['coef_ic']:.4f}, SM={sc['coef_sm']:.4f}, MW={sc['coef_mw']:.4f} [{sc['direction']}]")

    return {
        "layer": layer,
        "per_paradigm": {p: v["n_selected"] for p, v in coef_sets.items()},
        "overlap": {"ic_sm": len(ic_sm), "ic_mw": len(ic_mw), "sm_mw": len(sm_mw), "all_three": len(all_three)},
        "sign_consistent": sign_consistent,
    }


def b1e_within_bettype_sae():
    """B1-e: Within-bet-type SAE analysis (remove confound)."""
    log("\n=== B1-e: Within-bet-type SAE cross-domain ===")

    paradigms = ["ic", "sm", "mw"]
    layer = 22

    results = {}

    for bet_type in ["fixed", "variable"]:
        log(f"\n  --- Bet type: {bet_type} ---")
        per_paradigm = {}

        for p in paradigms:
            result = load_layer_features(p, layer, mode="decision_point")
            if result is None:
                continue
            feats, meta = result
            labels = get_labels(meta)

            # Filter by bet type
            bt_mask = meta["bet_types"] == bet_type
            if bt_mask.sum() < 20:
                log(f"    {p}/{bet_type}: too few samples ({bt_mask.sum()})")
                continue

            feats_bt = feats[bt_mask]
            labels_bt = labels[bt_mask]
            n_bk = labels_bt.sum()

            if n_bk < 5:
                log(f"    {p}/{bet_type}: too few BK ({n_bk})")
                continue

            bk_mask = labels_bt == 1
            safe_mask = labels_bt == 0
            bk_mean = feats_bt[bk_mask].mean(axis=0)
            safe_mean = feats_bt[safe_mask].mean(axis=0)
            bk_std = feats_bt[bk_mask].std(axis=0) + 1e-10
            safe_std = feats_bt[safe_mask].std(axis=0) + 1e-10
            pooled_std = np.sqrt((bk_std**2 + safe_std**2) / 2)
            cohens_d = (bk_mean - safe_mean) / pooled_std

            active = (feats_bt != 0).mean(axis=0) >= MIN_ACTIVATION_RATE
            per_paradigm[p] = {"cohens_d": cohens_d, "active": active, "n_bk": int(n_bk)}
            log(f"    {p}/{bet_type}: {n_bk} BK, {safe_mask.sum()} safe")

        if len(per_paradigm) < 2:
            continue

        # Cross-domain consistency for available paradigms
        available = list(per_paradigm.keys())
        all_active = per_paradigm[available[0]]["active"].copy()
        for p in available[1:]:
            all_active &= per_paradigm[p]["active"]

        ds = [per_paradigm[p]["cohens_d"] for p in available]

        # Sign consistent across available paradigms
        if len(available) == 3:
            sign_pos = (ds[0] > 0) & (ds[1] > 0) & (ds[2] > 0)
            sign_neg = (ds[0] < 0) & (ds[1] < 0) & (ds[2] < 0)
        else:
            sign_pos = (ds[0] > 0) & (ds[1] > 0)
            sign_neg = (ds[0] < 0) & (ds[1] < 0)

        sign_consistent = all_active & (sign_pos | sign_neg)
        n_sign = sign_consistent.sum()

        results[bet_type] = {
            "paradigms": available,
            "n_active": int(all_active.sum()),
            "n_sign_consistent": int(n_sign),
        }
        log(f"    Within-{bet_type} sign-consistent: {n_sign} / {all_active.sum()}")

    return results


# ============================================================
# B2: Variable/Fixed Activation-Level Common Risk Pattern
# ============================================================

def b2a_bk_only_comparison():
    """B2-a: Compare Variable BK vs Fixed BK neuron patterns."""
    log("\n=== B2-a: BK-only neuron comparison (Var BK vs Fix BK) ===")

    paradigms = ["ic", "sm", "mw"]
    # Use hidden states for neuron-level analysis
    layers_to_check = [10, 18, 22, 26, 30, 33]

    results = {}

    for layer in layers_to_check:
        log(f"\n  Layer {layer}:")

        for p in paradigms:
            hs_result = load_hidden_states(p, layer, mode="decision_point")
            if hs_result is None:
                log(f"    {p}: hidden states not found")
                continue

            hidden, meta = hs_result
            labels = get_labels(meta)
            bet_types = meta["bet_types"]

            # Split into 4 groups
            var_bk = (labels == 1) & (bet_types == "variable")
            fix_bk = (labels == 1) & (bet_types == "fixed")
            var_safe = (labels == 0) & (bet_types == "variable")
            fix_safe = (labels == 0) & (bet_types == "fixed")

            n_var_bk = var_bk.sum()
            n_fix_bk = fix_bk.sum()

            if n_var_bk < 5 or n_fix_bk < 5:
                log(f"    {p}: insufficient BK samples (var_bk={n_var_bk}, fix_bk={n_fix_bk})")
                continue

            # Key analysis: Are Variable BK and Fixed BK neuron patterns similar?
            var_bk_mean = hidden[var_bk].mean(axis=0)
            fix_bk_mean = hidden[fix_bk].mean(axis=0)
            var_safe_mean = hidden[var_safe].mean(axis=0)
            fix_safe_mean = hidden[fix_safe].mean(axis=0)

            # 1. Cosine similarity between Variable BK and Fixed BK directions
            # "BK direction" = BK_mean - Safe_mean, computed separately per bet type
            var_bk_dir = var_bk_mean - var_safe_mean
            fix_bk_dir = fix_bk_mean - fix_safe_mean

            cos_sim = np.dot(var_bk_dir, fix_bk_dir) / (np.linalg.norm(var_bk_dir) * np.linalg.norm(fix_bk_dir) + 1e-10)

            # 2. Common BK neurons: significant in BOTH bet types
            # Per-neuron t-test within each bet type
            from scipy.stats import ttest_ind

            common_bk_neurons = 0
            n_neurons = hidden.shape[1]

            var_bk_neurons = set()
            fix_bk_neurons = set()

            for ni in range(n_neurons):
                # Variable: BK vs Safe
                if n_var_bk >= 3 and var_safe.sum() >= 3:
                    t_var, p_var = ttest_ind(hidden[var_bk, ni], hidden[var_safe, ni])
                    if p_var < 0.05:
                        var_bk_neurons.add(ni)

                # Fixed: BK vs Safe
                if n_fix_bk >= 3 and fix_safe.sum() >= 3:
                    t_fix, p_fix = ttest_ind(hidden[fix_bk, ni], hidden[fix_safe, ni])
                    if p_fix < 0.05:
                        fix_bk_neurons.add(ni)

            common = var_bk_neurons & fix_bk_neurons

            # Check sign consistency among common neurons
            sign_consistent_common = 0
            for ni in common:
                var_d = hidden[var_bk, ni].mean() - hidden[var_safe, ni].mean()
                fix_d = hidden[fix_bk, ni].mean() - hidden[fix_safe, ni].mean()
                if np.sign(var_d) == np.sign(fix_d):
                    sign_consistent_common += 1

            key = f"{p}_L{layer}"
            results[key] = {
                "paradigm": p,
                "layer": layer,
                "n_var_bk": int(n_var_bk),
                "n_fix_bk": int(n_fix_bk),
                "cos_bk_direction": float(cos_sim),
                "var_sig_neurons": len(var_bk_neurons),
                "fix_sig_neurons": len(fix_bk_neurons),
                "common_neurons": len(common),
                "sign_consistent_common": sign_consistent_common,
            }

            log(f"    {p} L{layer}: var_bk={n_var_bk}, fix_bk={n_fix_bk}")
            log(f"      cos(var_bk_dir, fix_bk_dir) = {cos_sim:.3f}")
            log(f"      sig neurons: var={len(var_bk_neurons)}, fix={len(fix_bk_neurons)}, common={len(common)}, sign-consistent={sign_consistent_common}")

    return results


def b2b_interaction_regression():
    """B2-b: Interaction regression — neuron ~ outcome * bet_type."""
    log("\n=== B2-b: Interaction regression (outcome × bet_type) ===")

    import statsmodels.api as sm

    paradigms = ["ic", "sm", "mw"]
    layer = 22

    results = {}

    for p in paradigms:
        hs_result = load_hidden_states(p, layer, mode="decision_point")
        if hs_result is None:
            continue

        hidden, meta = hs_result
        labels = get_labels(meta)
        bet_types = (meta["bet_types"] == "variable").astype(float)

        n_bk = labels.sum()
        if n_bk < 10:
            continue

        # For each neuron: activation ~ outcome + bet_type + outcome:bet_type
        n_neurons = hidden.shape[1]
        interaction_sig = 0
        outcome_sig = 0
        bettype_sig = 0
        shared_bk_neurons = []  # outcome sig but NOT interaction sig

        for ni in range(n_neurons):
            y = hidden[:, ni]
            X = np.column_stack([labels, bet_types, labels * bet_types])
            X = sm.add_constant(X)

            try:
                model = sm.OLS(y, X).fit()
                p_outcome = model.pvalues[1]
                p_bettype = model.pvalues[2]
                p_interact = model.pvalues[3]

                if p_outcome < 0.01:
                    outcome_sig += 1
                if p_bettype < 0.01:
                    bettype_sig += 1
                if p_interact < 0.01:
                    interaction_sig += 1

                # "Shared BK neuron": outcome-sensitive but NOT bet_type-dependent
                if p_outcome < 0.01 and p_interact > 0.05:
                    shared_bk_neurons.append({
                        "neuron": ni,
                        "outcome_p": float(p_outcome),
                        "interaction_p": float(p_interact),
                        "outcome_coef": float(model.params[1]),
                    })
            except:
                continue

        results[p] = {
            "n_neurons": n_neurons,
            "outcome_sig_p01": outcome_sig,
            "bettype_sig_p01": bettype_sig,
            "interaction_sig_p01": interaction_sig,
            "shared_bk_neurons": len(shared_bk_neurons),
        }

        log(f"  {p} L{layer}: outcome_sig={outcome_sig}, bettype_sig={bettype_sig}, interaction_sig={interaction_sig}")
        log(f"    Shared BK neurons (outcome sig, interaction NOT sig): {len(shared_bk_neurons)}")

    # Cross-paradigm: neurons that are "shared BK" in all paradigms
    if len(results) == 3:
        log(f"\n  Summary: Shared BK neurons per paradigm: IC={results.get('ic',{}).get('shared_bk_neurons',0)}, SM={results.get('sm',{}).get('shared_bk_neurons',0)}, MW={results.get('mw',{}).get('shared_bk_neurons',0)}")

    return results


def b2d_partial_correlation():
    """B2-d: Partial correlation — regress out balance from neurons, check BK prediction."""
    log("\n=== B2-d: Partial correlation (balance regress-out) ===")

    paradigms = ["ic", "sm", "mw"]
    layer = 22

    results = {}

    for p in paradigms:
        hs_result = load_hidden_states(p, layer, mode="decision_point")
        if hs_result is None:
            continue

        hidden, meta = hs_result
        labels = get_labels(meta)

        # Get balances if available
        if "balances" not in meta:
            log(f"  {p}: no balance data, skipping")
            continue

        balances = meta["balances"].astype(float)
        n_bk = labels.sum()

        if n_bk < 10:
            continue

        # For each neuron: compute partial correlation with BK, controlling for balance
        n_neurons = hidden.shape[1]

        # Method: regress neuron_i on balance, take residual, correlate with BK
        partial_sig = 0
        raw_sig = 0

        for ni in range(n_neurons):
            y = hidden[:, ni]

            # Raw correlation with BK
            r_raw, p_raw = pointbiserialr(labels, y)
            if p_raw < 0.01:
                raw_sig += 1

            # Partial: regress out balance
            # residual_neuron = neuron - β·balance
            beta = np.polyfit(balances, y, 1)
            residual = y - np.polyval(beta, balances)

            r_partial, p_partial = pointbiserialr(labels, residual)
            if p_partial < 0.01:
                partial_sig += 1

        results[p] = {
            "n_neurons": n_neurons,
            "raw_bk_sig": raw_sig,
            "partial_bk_sig_balance_controlled": partial_sig,
            "retained_pct": f"{partial_sig/max(raw_sig,1)*100:.1f}%",
        }

        log(f"  {p} L{layer}: raw_sig={raw_sig}, partial(balance-controlled)={partial_sig} ({partial_sig/max(raw_sig,1)*100:.1f}% retained)")

    return results


def b2e_round1_analysis():
    """B2-e: Round 1 analysis — all games start at $100, no balance confound."""
    log("\n=== B2-e: Round 1 analysis (confound-free) ===")

    paradigms = ["ic", "sm", "mw"]
    layer = 22

    results = {}

    for p in paradigms:
        # Load all rounds
        result = load_layer_features(p, layer, mode="all_rounds")
        if result is None:
            hs_result = load_hidden_states(p, layer, mode="all_rounds")
            if hs_result is None:
                log(f"  {p}: no data")
                continue
            hidden, meta = hs_result
        else:
            hidden, meta = result

        # Filter to round 1 only
        r1_mask = meta["round_nums"] == 1
        if r1_mask.sum() < 20:
            # Try round_nums == 0 (zero-indexed?)
            r1_mask = meta["round_nums"] == 0

        if r1_mask.sum() < 20:
            log(f"  {p}: insufficient R1 data ({r1_mask.sum()} samples)")
            continue

        hidden_r1 = hidden[r1_mask]
        labels_r1 = get_labels({k: v[r1_mask] for k, v in meta.items()})
        bet_types_r1 = meta["bet_types"][r1_mask]

        n_bk = labels_r1.sum()
        log(f"  {p} R1: {len(labels_r1)} games, {n_bk} BK")

        if n_bk < 5:
            continue

        # Within-bet-type R1 classification
        for bt in ["fixed", "variable"]:
            bt_mask = bet_types_r1 == bt
            if bt_mask.sum() < 10:
                continue

            X_bt = hidden_r1[bt_mask]
            y_bt = labels_r1[bt_mask]
            n_bk_bt = y_bt.sum()

            if n_bk_bt < 3 or n_bk_bt == len(y_bt):
                continue

            # PCA → LogReg
            try:
                n_comp = min(50, X_bt.shape[0] - 1, X_bt.shape[1])
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_bt)
                pca = PCA(n_components=n_comp)
                X_pca = pca.fit_transform(X_scaled)

                clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=RANDOM_SEED)
                cv = StratifiedKFold(n_splits=min(5, n_bk_bt), shuffle=True, random_state=RANDOM_SEED)
                scores = cross_val_score(clf, X_pca, y_bt, cv=cv, scoring='roc_auc')
                mean_auc = scores.mean()

                key = f"{p}_{bt}_R1_L{layer}"
                results[key] = {
                    "paradigm": p,
                    "bet_type": bt,
                    "n_samples": int(bt_mask.sum()),
                    "n_bk": int(n_bk_bt),
                    "auc": float(mean_auc),
                }
                log(f"    {p}/{bt} R1: AUC={mean_auc:.3f} (n={bt_mask.sum()}, BK={n_bk_bt})")
            except Exception as e:
                log(f"    {p}/{bt} R1: error - {e}")

    return results


# ============================================================
# Main
# ============================================================

def main():
    log("=" * 70)
    log("B1+B2 BRUTE-FORCE ANALYSIS")
    log("Improving v8 weaknesses: SAE cross-domain + Variable/Fixed risk")
    log("=" * 70)

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "description": "B1+B2 brute-force analysis to improve v8 weaknesses",
    }

    # Check data availability first
    log("\nChecking data availability...")
    for p_name, p_cfg in PARADIGMS.items():
        sae_dir = p_cfg["sae_dir"]
        n_layers = len(list(sae_dir.glob("sae_features_L*.npz")))
        ckpt = sae_dir / "checkpoint" / "phase_a_hidden_states.npz"
        has_hs = ckpt.exists()
        log(f"  {p_name}: {n_layers} SAE layers, hidden_states={'YES' if has_hs else 'NO'}")

    # === B1: SAE cross-domain ===
    log("\n" + "=" * 70)
    log("TRACK B1: SAE CROSS-DOMAIN SIGNAL STRENGTHENING")
    log("=" * 70)

    all_results["b1a_multilayer"] = b1a_multilayer_sae_crossdomain()
    all_results["b1d_logreg_overlap"] = b1d_sparse_logreg_overlap()
    all_results["b1e_within_bettype"] = b1e_within_bettype_sae()

    # === B2: Variable/Fixed risk pattern ===
    log("\n" + "=" * 70)
    log("TRACK B2: VARIABLE/FIXED ACTIVATION-LEVEL RISK PATTERN")
    log("=" * 70)

    all_results["b2a_bk_only"] = b2a_bk_only_comparison()
    all_results["b2b_interaction"] = b2b_interaction_regression()
    all_results["b2d_partial_corr"] = b2d_partial_correlation()
    all_results["b2e_round1"] = b2e_round1_analysis()

    # Save results
    out_file = JSON_DIR / f"b1_b2_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, set): return list(obj)
        return obj

    import json
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=convert)

    log(f"\n=== Results saved to {out_file} ===")

    # === FINAL SUMMARY ===
    log("\n" + "=" * 70)
    log("EXECUTIVE SUMMARY")
    log("=" * 70)
    log("B1 (SAE cross-domain):")
    if "b1a_multilayer" in all_results:
        total = sum(r.get("n_medium_d01", 0) for r in all_results["b1a_multilayer"].values())
        log(f"  B1-a multi-layer: {total} medium-strength cross-domain features (v8 baseline: 1)")
    if "b1d_logreg_overlap" in all_results and all_results["b1d_logreg_overlap"]:
        r = all_results["b1d_logreg_overlap"]
        log(f"  B1-d LogReg: {r.get('overlap',{}).get('all_three',0)} overlapping features, {len(r.get('sign_consistent',[]))} sign-consistent")

    log("B2 (Variable/Fixed risk):")
    if "b2a_bk_only" in all_results:
        cos_vals = [v["cos_bk_direction"] for v in all_results["b2a_bk_only"].values() if "cos_bk_direction" in v]
        if cos_vals:
            log(f"  B2-a BK direction cosine: mean={np.mean(cos_vals):.3f} (>0 = shared BK mechanism)")
    if "b2b_interaction" in all_results:
        shared = [v.get("shared_bk_neurons", 0) for v in all_results["b2b_interaction"].values()]
        if shared:
            log(f"  B2-b Shared BK neurons (outcome sig, interaction NOT sig): {shared}")


if __name__ == "__main__":
    main()
