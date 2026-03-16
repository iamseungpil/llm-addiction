"""
V8 Condition × Cross-Domain Analysis
=====================================
Addresses:
1. R1 validity: Is R1 AUC 0.77-0.90 an overclaim? Data sufficiency check.
2. Bet type: Variable vs Fixed behavioral differences beyond bankruptcy.
3. 3D shared BK subspace × conditions: Do conditions map onto the shared space?
4. Cross-domain condition effects: Are condition effects consistent across paradigms?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import json
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, permutation_test_score
from sklearn.metrics import roc_auc_score
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

from data_loader import load_hidden_states, get_labels

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "json")
LAYER = 22  # Shared subspace layer


def load_paradigm_data(paradigm, mode="decision_point"):
    """Load hidden states + metadata for a paradigm.

    mode: 'decision_point', 'round1', or 'all_rounds'
    For 'round1', we load all_rounds and filter to round_nums == 1.
    """
    if mode == "round1":
        # Load all rounds, then filter to round 1 only
        hs_all, meta_all = load_hidden_states(paradigm, layer=LAYER, mode="all_rounds")
        r1_mask = meta_all["round_nums"] == 1
        hs = hs_all[r1_mask]
        meta = {k: v[r1_mask] for k, v in meta_all.items()}
    else:
        hs, meta = load_hidden_states(paradigm, layer=LAYER, mode=mode)
    labels = get_labels(meta)
    return hs, meta, labels


def analyze_r1_validity(hs_r1, meta_r1, labels_r1, paradigm):
    """Check R1 prediction validity: data sufficiency, permutation test, class balance."""
    n_bk = labels_r1.sum()
    n_total = len(labels_r1)
    n_nbk = n_total - n_bk

    result = {
        "n_total": int(n_total),
        "n_bk": int(n_bk),
        "n_nbk": int(n_nbk),
        "bk_rate": float(n_bk / n_total),
        "class_ratio": float(n_nbk / max(n_bk, 1)),
    }

    # R1 classification with proper CV
    scaler = StandardScaler()
    X = scaler.fit_transform(hs_r1)
    pca = PCA(n_components=min(50, X.shape[1], X.shape[0] - 1), random_state=42)
    X_pca = pca.fit_transform(X)

    clf = LogisticRegression(C=1.0, solver="lbfgs", class_weight="balanced", max_iter=1000, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # CV AUC
    aucs = []
    for train_idx, test_idx in cv.split(X_pca, labels_r1):
        clf.fit(X_pca[train_idx], labels_r1[train_idx])
        proba = clf.predict_proba(X_pca[test_idx])[:, 1]
        aucs.append(roc_auc_score(labels_r1[test_idx], proba))
    result["cv_auc_mean"] = float(np.mean(aucs))
    result["cv_auc_std"] = float(np.std(aucs))

    # Permutation test (100 permutations for speed)
    print(f"  Running permutation test for {paradigm} R1 (100 perms)...")
    score, perm_scores, pvalue = permutation_test_score(
        clf, X_pca, labels_r1, scoring="roc_auc", cv=cv, n_permutations=100, random_state=42
    )
    result["permutation_p"] = float(pvalue)
    result["permutation_null_mean"] = float(np.mean(perm_scores))
    result["permutation_null_std"] = float(np.std(perm_scores))
    result["permutation_observed"] = float(score)

    # Effect size: how far is observed from null distribution?
    result["z_score"] = float((score - np.mean(perm_scores)) / max(np.std(perm_scores), 1e-6))

    # Minimum sample size check (rule of thumb: 10 events per predictor)
    n_predictors = X_pca.shape[1]
    min_events_needed = 10 * n_predictors
    result["n_predictors_pca"] = int(n_predictors)
    result["min_events_rule_of_10"] = int(min_events_needed)
    result["sample_sufficient"] = bool(n_bk >= min_events_needed)

    print(f"  {paradigm} R1: AUC={result['cv_auc_mean']:.3f}+/-{result['cv_auc_std']:.3f}, "
          f"perm_p={pvalue:.4f}, BK={n_bk}/{n_total}, sufficient={result['sample_sufficient']}")

    return result


def analyze_bet_type_behavior(meta_dp, labels_dp, paradigm):
    """Analyze variable vs fixed behavioral differences beyond bankruptcy."""
    bet_types = meta_dp["bet_types"]
    balances = meta_dp["balances"]

    result = {}
    for bt in ["fixed", "variable"]:
        mask = bet_types == bt
        if mask.sum() == 0:
            continue
        bt_labels = labels_dp[mask]
        bt_balances = balances[mask]

        n = int(mask.sum())
        n_bk = int(bt_labels.sum())

        # Balance at decision point (proxy for risk-taking behavior)
        bk_balances = bt_balances[bt_labels == 1] if n_bk > 0 else np.array([])
        nbk_balances = bt_balances[bt_labels == 0]

        result[bt] = {
            "n_games": n,
            "n_bk": n_bk,
            "bk_rate": float(n_bk / n) if n > 0 else 0,
            "mean_balance_all": float(bt_balances.mean()),
            "mean_balance_nbk": float(nbk_balances.mean()) if len(nbk_balances) > 0 else None,
            "mean_balance_bk": float(bk_balances.mean()) if len(bk_balances) > 0 else None,
            "median_balance_all": float(np.median(bt_balances)),
        }

    # Statistical test: balance difference between fixed vs variable (non-BK only)
    fixed_mask = (bet_types == "fixed") & (labels_dp == 0)
    var_mask = (bet_types == "variable") & (labels_dp == 0)
    if fixed_mask.sum() > 0 and var_mask.sum() > 0:
        t, p = stats.ttest_ind(balances[fixed_mask], balances[var_mask])
        result["balance_ttest_nonbk"] = {"t": float(t), "p": float(p)}
        result["fixed_nonbk_mean_balance"] = float(balances[fixed_mask].mean())
        result["variable_nonbk_mean_balance"] = float(balances[var_mask].mean())
        d = (balances[var_mask].mean() - balances[fixed_mask].mean()) / np.sqrt(
            (balances[fixed_mask].var() + balances[var_mask].var()) / 2)
        result["balance_cohens_d"] = float(d)

    return result


def compute_shared_subspace(paradigm_data):
    """Compute 3D shared BK subspace from LR weight vectors."""
    bk_directions = {}

    for paradigm, (hs, meta, labels) in paradigm_data.items():
        scaler = StandardScaler()
        X = scaler.fit_transform(hs)
        pca = PCA(n_components=50, random_state=42)
        X_pca = pca.fit_transform(X)

        clf = LogisticRegression(C=1.0, solver="lbfgs", class_weight="balanced", max_iter=1000, random_state=42)
        clf.fit(X_pca, labels)

        # Map back to hidden state space
        coef_hidden = clf.coef_[0] @ pca.components_  # (3584,)
        coef_hidden = coef_hidden / np.linalg.norm(coef_hidden)
        bk_directions[paradigm] = coef_hidden

    # PCA on 3 BK directions
    bk_stack = np.stack([bk_directions[p] for p in ["ic", "sm", "mw"]])  # (3, 3584)
    pca_shared = PCA(n_components=3, random_state=42)
    pca_shared.fit(bk_stack)

    return pca_shared, bk_directions


def analyze_3d_conditions(pca_shared, paradigm_data_dp, paradigm_data_r1=None):
    """Analyze how conditions map onto the 3D shared BK subspace."""
    results = {}

    for mode_name, pdata in [("dp", paradigm_data_dp), ("r1", paradigm_data_r1)]:
        if pdata is None:
            continue
        results[mode_name] = {}

        for paradigm, (hs, meta, labels) in pdata.items():
            # Project into 3D shared space
            scaler = StandardScaler()
            X = scaler.fit_transform(hs)
            coords_3d = X @ pca_shared.components_.T  # (N, 3)

            pres = {
                "n_games": int(len(labels)),
                "n_bk": int(labels.sum()),
            }

            # 1. BK vs non-BK separation in 3D
            bk_mask = labels == 1
            if bk_mask.sum() > 2 and (~bk_mask).sum() > 2:
                bk_coords = coords_3d[bk_mask]
                nbk_coords = coords_3d[~bk_mask]

                pres["bk_centroid"] = bk_coords.mean(axis=0).tolist()
                pres["nbk_centroid"] = nbk_coords.mean(axis=0).tolist()

                # Per-dimension t-tests
                dim_effects = []
                for d in range(3):
                    t, p = stats.ttest_ind(bk_coords[:, d], nbk_coords[:, d])
                    cohens_d = (bk_coords[:, d].mean() - nbk_coords[:, d].mean()) / np.sqrt(
                        (bk_coords[:, d].var() + nbk_coords[:, d].var()) / 2)
                    dim_effects.append({
                        "dim": d, "t": float(t), "p": float(p), "cohens_d": float(cohens_d),
                        "bk_mean": float(bk_coords[:, d].mean()),
                        "nbk_mean": float(nbk_coords[:, d].mean()),
                    })
                pres["bk_vs_nbk_per_dim"] = dim_effects

                # 3D classification AUC
                clf_3d = LogisticRegression(C=1.0, solver="lbfgs", class_weight="balanced", max_iter=1000, random_state=42)
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                aucs_3d = []
                for train_idx, test_idx in cv.split(coords_3d, labels):
                    clf_3d.fit(coords_3d[train_idx], labels[train_idx])
                    proba = clf_3d.predict_proba(coords_3d[test_idx])[:, 1]
                    if len(np.unique(labels[test_idx])) > 1:
                        aucs_3d.append(roc_auc_score(labels[test_idx], proba))
                pres["auc_3d"] = float(np.mean(aucs_3d)) if aucs_3d else None

            # 2. Bet type effect in 3D
            bet_types = meta["bet_types"]
            for bt in ["fixed", "variable"]:
                bt_mask = bet_types == bt
                if bt_mask.sum() > 0:
                    pres[f"mean_3d_{bt}"] = coords_3d[bt_mask].mean(axis=0).tolist()
                    pres[f"n_{bt}"] = int(bt_mask.sum())
                    # BK rate within bet type
                    pres[f"bk_rate_{bt}"] = float(labels[bt_mask].mean())

            # Bet type effect: t-test per dimension
            fixed_mask = bet_types == "fixed"
            var_mask = bet_types == "variable"
            if fixed_mask.sum() > 2 and var_mask.sum() > 2:
                bt_effects = []
                for d in range(3):
                    t, p = stats.ttest_ind(coords_3d[fixed_mask, d], coords_3d[var_mask, d])
                    cohens_d_val = (coords_3d[var_mask, d].mean() - coords_3d[fixed_mask, d].mean()) / np.sqrt(
                        (coords_3d[fixed_mask, d].var() + coords_3d[var_mask, d].var()) / 2)
                    bt_effects.append({
                        "dim": d, "t": float(t), "p": float(p), "cohens_d": float(cohens_d_val),
                        "fixed_mean": float(coords_3d[fixed_mask, d].mean()),
                        "variable_mean": float(coords_3d[var_mask, d].mean()),
                    })
                pres["bet_type_per_dim"] = bt_effects

            # 3. Prompt condition effect in 3D
            prompt_conds = meta["prompt_conditions"]
            unique_prompts = np.unique(prompt_conds)
            if len(unique_prompts) > 1:
                prompt_effects = {}
                for pc in unique_prompts:
                    pc_mask = prompt_conds == pc
                    pc_str = str(pc)
                    prompt_effects[pc_str] = {
                        "n": int(pc_mask.sum()),
                        "mean_3d": coords_3d[pc_mask].mean(axis=0).tolist(),
                        "bk_rate": float(labels[pc_mask].mean()),
                        "n_bk": int(labels[pc_mask].sum()),
                    }
                pres["prompt_conditions"] = prompt_effects

                # G prompt vs non-G (has_G flag)
                has_g = np.array(["G" in str(pc) and str(pc) != "BASE" for pc in prompt_conds])
                if has_g.sum() > 2 and (~has_g).sum() > 2:
                    g_effects = []
                    for d in range(3):
                        t, p = stats.ttest_ind(coords_3d[has_g, d], coords_3d[~has_g, d])
                        cd = (coords_3d[has_g, d].mean() - coords_3d[~has_g, d].mean()) / np.sqrt(
                            (coords_3d[has_g, d].var() + coords_3d[~has_g, d].var()) / 2)
                        g_effects.append({
                            "dim": d, "t": float(t), "p": float(p), "cohens_d": float(cd),
                            "g_mean": float(coords_3d[has_g, d].mean()),
                            "no_g_mean": float(coords_3d[~has_g, d].mean()),
                        })
                    pres["g_prompt_per_dim"] = g_effects
                    pres["g_bk_rate"] = float(labels[has_g].mean())
                    pres["no_g_bk_rate"] = float(labels[~has_g].mean())

            # 4. Bet constraint effect (IC only)
            bet_constraints = meta.get("bet_constraints", meta.get("bet_constraints", None))
            if bet_constraints is not None:
                unique_bc = np.unique(bet_constraints)
                if len(unique_bc) > 1:
                    bc_effects = {}
                    for bc in unique_bc:
                        bc_mask = bet_constraints == bc
                        bc_str = str(bc)
                        bc_effects[bc_str] = {
                            "n": int(bc_mask.sum()),
                            "mean_3d": coords_3d[bc_mask].mean(axis=0).tolist(),
                            "bk_rate": float(labels[bc_mask].mean()),
                            "n_bk": int(labels[bc_mask].sum()),
                        }
                    pres["bet_constraints"] = bc_effects

            # 5. Correlation: 3D coordinates vs balance (confound check)
            balances = meta["balances"]
            balance_corr = []
            for d in range(3):
                r, p = stats.pearsonr(coords_3d[:, d], balances)
                balance_corr.append({"dim": d, "r": float(r), "p": float(p)})
            pres["balance_correlation"] = balance_corr

            results[mode_name][paradigm] = pres

    return results


def analyze_cross_domain_condition_consistency(pca_shared, paradigm_data_dp):
    """Check if condition effects are consistent across paradigms in 3D space."""
    # For each condition type, compute the "direction" in 3D space and check cross-domain consistency
    result = {}

    # Bet type direction per paradigm
    bet_directions = {}
    for paradigm, (hs, meta, labels) in paradigm_data_dp.items():
        scaler = StandardScaler()
        X = scaler.fit_transform(hs)
        coords_3d = X @ pca_shared.components_.T

        bt = meta["bet_types"]
        fixed_mask = bt == "fixed"
        var_mask = bt == "variable"
        if fixed_mask.sum() > 0 and var_mask.sum() > 0:
            direction = coords_3d[var_mask].mean(axis=0) - coords_3d[fixed_mask].mean(axis=0)
            bet_directions[paradigm] = direction

    # Cosine similarity between bet-type directions
    if len(bet_directions) >= 2:
        pairs = [("ic", "sm"), ("ic", "mw"), ("sm", "mw")]
        bt_cosines = {}
        for p1, p2 in pairs:
            if p1 in bet_directions and p2 in bet_directions:
                d1, d2 = bet_directions[p1], bet_directions[p2]
                cos = np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-10)
                bt_cosines[f"{p1}_{p2}"] = float(cos)
        result["bet_type_direction_cosines"] = bt_cosines
        result["bet_type_directions"] = {p: d.tolist() for p, d in bet_directions.items()}

    # G prompt direction per paradigm
    g_directions = {}
    for paradigm, (hs, meta, labels) in paradigm_data_dp.items():
        scaler = StandardScaler()
        X = scaler.fit_transform(hs)
        coords_3d = X @ pca_shared.components_.T

        pc = meta["prompt_conditions"]
        has_g = np.array(["G" in str(p) and str(p) != "BASE" for p in pc])
        if has_g.sum() > 0 and (~has_g).sum() > 0:
            direction = coords_3d[has_g].mean(axis=0) - coords_3d[~has_g].mean(axis=0)
            g_directions[paradigm] = direction

    if len(g_directions) >= 2:
        g_cosines = {}
        for p1, p2 in pairs:
            if p1 in g_directions and p2 in g_directions:
                d1, d2 = g_directions[p1], g_directions[p2]
                cos = np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-10)
                g_cosines[f"{p1}_{p2}"] = float(cos)
        result["g_prompt_direction_cosines"] = g_cosines
        result["g_prompt_directions"] = {p: d.tolist() for p, d in g_directions.items()}

    # BK direction per paradigm in 3D
    bk_directions_3d = {}
    for paradigm, (hs, meta, labels) in paradigm_data_dp.items():
        scaler = StandardScaler()
        X = scaler.fit_transform(hs)
        coords_3d = X @ pca_shared.components_.T

        bk_mask = labels == 1
        if bk_mask.sum() > 0:
            direction = coords_3d[bk_mask].mean(axis=0) - coords_3d[~bk_mask].mean(axis=0)
            bk_directions_3d[paradigm] = direction

    if len(bk_directions_3d) >= 2:
        bk_cosines = {}
        for p1, p2 in pairs:
            if p1 in bk_directions_3d and p2 in bk_directions_3d:
                d1, d2 = bk_directions_3d[p1], bk_directions_3d[p2]
                cos = np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-10)
                bk_cosines[f"{p1}_{p2}"] = float(cos)
        result["bk_direction_3d_cosines"] = bk_cosines
        result["bk_directions_3d"] = {p: d.tolist() for p, d in bk_directions_3d.items()}

    # Alignment: how well does bet-type direction align with BK direction?
    alignment = {}
    for paradigm in ["ic", "sm", "mw"]:
        if paradigm in bet_directions and paradigm in bk_directions_3d:
            bt_d = bet_directions[paradigm]
            bk_d = bk_directions_3d[paradigm]
            cos = np.dot(bt_d, bk_d) / (np.linalg.norm(bt_d) * np.linalg.norm(bk_d) + 1e-10)
            alignment[f"{paradigm}_bettype_bk"] = float(cos)
        if paradigm in g_directions and paradigm in bk_directions_3d:
            g_d = g_directions[paradigm]
            bk_d = bk_directions_3d[paradigm]
            cos = np.dot(g_d, bk_d) / (np.linalg.norm(g_d) * np.linalg.norm(bk_d) + 1e-10)
            alignment[f"{paradigm}_gprompt_bk"] = float(cos)
    result["condition_bk_alignment"] = alignment

    return result


def analyze_r1_bet_type_behavioral(paradigm_data_r1):
    """Analyze bet type effects at R1 (no balance confound)."""
    results = {}
    for paradigm, (hs, meta, labels) in paradigm_data_r1.items():
        bt = meta["bet_types"]
        balances = meta["balances"]

        pres = {}
        for btype in ["fixed", "variable"]:
            mask = bt == btype
            if mask.sum() == 0:
                continue
            bl = labels[mask]
            pres[btype] = {
                "n": int(mask.sum()),
                "n_bk": int(bl.sum()),
                "bk_rate": float(bl.mean()),
                "mean_balance": float(balances[mask].mean()),
            }

        # Test: does bet type predict BK at R1?
        if "fixed" in pres and "variable" in pres:
            fixed_bk = labels[bt == "fixed"]
            var_bk = labels[bt == "variable"]
            # Chi-squared test for BK rate difference
            contingency = np.array([
                [int((bt == "fixed").sum() - fixed_bk.sum()), int(fixed_bk.sum())],
                [int((bt == "variable").sum() - var_bk.sum()), int(var_bk.sum())]
            ])
            chi2, p, dof, expected = stats.chi2_contingency(contingency)
            pres["chi2_bk_rate"] = {"chi2": float(chi2), "p": float(p)}

            # Also: can R1 hidden state predict bet type? (should be ~1.0 per Table 9)
            scaler = StandardScaler()
            X = scaler.fit_transform(hs)
            pca = PCA(n_components=50, random_state=42)
            X_pca = pca.fit_transform(X)

            bt_labels = (bt == "variable").astype(int)
            clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            bt_aucs = []
            for train_idx, test_idx in cv.split(X_pca, bt_labels):
                clf.fit(X_pca[train_idx], bt_labels[train_idx])
                proba = clf.predict_proba(X_pca[test_idx])[:, 1]
                bt_aucs.append(roc_auc_score(bt_labels[test_idx], proba))
            pres["bet_type_classification_auc"] = float(np.mean(bt_aucs))

        results[paradigm] = pres
    return results


def main():
    print("=" * 70)
    print("V8 Condition × Cross-Domain Analysis")
    print("=" * 70)

    # Load data for all paradigms
    print("\n[1/6] Loading data...")
    paradigm_data_dp = {}
    paradigm_data_r1 = {}

    for paradigm in ["ic", "sm", "mw"]:
        print(f"  Loading {paradigm}...")
        hs_dp, meta_dp, labels_dp = load_paradigm_data(paradigm, mode="decision_point")
        hs_r1, meta_r1, labels_r1 = load_paradigm_data(paradigm, mode="round1")
        paradigm_data_dp[paradigm] = (hs_dp, meta_dp, labels_dp)
        paradigm_data_r1[paradigm] = (hs_r1, meta_r1, labels_r1)
        print(f"    DP: {hs_dp.shape}, R1: {hs_r1.shape}, BK_dp={labels_dp.sum()}, BK_r1={labels_r1.sum()}")

    all_results = {}

    # Part 1: R1 Validity Check
    print("\n[2/6] R1 Validity Check...")
    r1_validity = {}
    for paradigm in ["ic", "sm", "mw"]:
        hs_r1, meta_r1, labels_r1 = paradigm_data_r1[paradigm]
        r1_validity[paradigm] = analyze_r1_validity(hs_r1, meta_r1, labels_r1, paradigm)
    all_results["r1_validity"] = r1_validity

    # Part 2: Bet type behavioral analysis
    print("\n[3/6] Bet Type Behavioral Analysis...")
    bet_behavior_dp = {}
    for paradigm in ["ic", "sm", "mw"]:
        _, meta_dp, labels_dp = paradigm_data_dp[paradigm]
        bet_behavior_dp[paradigm] = analyze_bet_type_behavior(meta_dp, labels_dp, paradigm)
    all_results["bet_type_behavior_dp"] = bet_behavior_dp

    # R1 bet type analysis
    r1_bet_behavior = analyze_r1_bet_type_behavioral(paradigm_data_r1)
    all_results["r1_bet_type_behavior"] = r1_bet_behavior

    # Part 3: Compute shared 3D subspace
    print("\n[4/6] Computing shared 3D BK subspace...")
    pca_shared, bk_directions = compute_shared_subspace(paradigm_data_dp)
    all_results["shared_subspace_variance"] = pca_shared.explained_variance_ratio_.tolist()

    # Part 4: 3D condition analysis
    print("\n[5/6] Analyzing conditions in 3D shared space...")
    cond_3d = analyze_3d_conditions(pca_shared, paradigm_data_dp, paradigm_data_r1)
    all_results["conditions_in_3d"] = cond_3d

    # Part 5: Cross-domain condition consistency
    print("\n[6/6] Cross-domain condition consistency...")
    cross_consistency = analyze_cross_domain_condition_consistency(pca_shared, paradigm_data_dp)
    all_results["cross_domain_consistency"] = cross_consistency

    # Save results
    output_path = os.path.join(OUTPUT_DIR, "v8_condition_crossdomain.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n--- R1 Validity ---")
    for p in ["ic", "sm", "mw"]:
        v = r1_validity[p]
        print(f"  {p.upper()}: AUC={v['cv_auc_mean']:.3f}, perm_p={v['permutation_p']:.4f}, "
              f"z={v['z_score']:.1f}, BK={v['n_bk']}, sufficient={v['sample_sufficient']}")

    print("\n--- Bet Type BK Rates (DP) ---")
    for p in ["ic", "sm", "mw"]:
        bb = bet_behavior_dp[p]
        for bt in ["fixed", "variable"]:
            if bt in bb:
                print(f"  {p.upper()} {bt}: BK={bb[bt]['n_bk']}/{bb[bt]['n_games']} ({bb[bt]['bk_rate']:.3f}), "
                      f"mean_balance={bb[bt]['mean_balance_all']:.1f}")

    print("\n--- Bet Type BK Rates (R1) ---")
    for p in ["ic", "sm", "mw"]:
        bb = r1_bet_behavior[p]
        for bt in ["fixed", "variable"]:
            if bt in bb:
                print(f"  {p.upper()} {bt}: BK={bb[bt]['n_bk']}/{bb[bt]['n']} ({bb[bt]['bk_rate']:.3f})")
        if "chi2_bk_rate" in bb:
            print(f"  {p.upper()} chi2 p={bb['chi2_bk_rate']['p']:.4f}")

    print("\n--- Cross-Domain Condition Consistency (3D) ---")
    cc = cross_consistency
    if "bet_type_direction_cosines" in cc:
        print("  Bet-type direction cosines:")
        for k, v in cc["bet_type_direction_cosines"].items():
            print(f"    {k}: {v:.3f}")
    if "g_prompt_direction_cosines" in cc:
        print("  G-prompt direction cosines:")
        for k, v in cc["g_prompt_direction_cosines"].items():
            print(f"    {k}: {v:.3f}")
    if "condition_bk_alignment" in cc:
        print("  Condition-BK alignment:")
        for k, v in cc["condition_bk_alignment"].items():
            print(f"    {k}: {v:.3f}")

    print("\n--- 3D BK Separation ---")
    for mode in ["dp", "r1"]:
        if mode in cond_3d:
            for p in ["ic", "sm", "mw"]:
                if p in cond_3d[mode] and "auc_3d" in cond_3d[mode][p]:
                    print(f"  {p.upper()} {mode}: 3D AUC={cond_3d[mode][p]['auc_3d']:.3f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
