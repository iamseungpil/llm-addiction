#!/usr/bin/env python3
"""
Symmetric LLaMA analyses to match Gemma analyses in V9 report.

Computes:
1. LLaMA IC+SM Factor Decomposition (SAE features, L22)
2. LLaMA SM Prompt Component Analysis (SAE feature level)
3. Gemma IC Classification AUC (key layers)
4. Cross-model IC BK rate comparison
5. LLaMA IC within-bet-type SAE cross-domain (IC+SM)

Saves all results to results/json/llama_symmetric_20260318.json
"""

import sys
import json
import time
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")
np.random.seed(42)

# Paths
DATA_ROOT = Path("/home/v-seungplee/data/llm-addiction/sae_features_v3")
REPO_ROOT = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis")
OUTPUT_PATH = REPO_ROOT / "results" / "json" / "llama_symmetric_20260318.json"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Add src to path for Gemma loader
sys.path.insert(0, str(REPO_ROOT / "src"))


def timestamp():
    return datetime.now().strftime("%H:%M:%S")


def load_llama_npz(paradigm, layer, decision_point=True):
    """Load LLaMA sparse NPZ and return dense features + metadata at decision point."""
    npz_path = DATA_ROOT / paradigm / "llama" / f"sae_features_L{layer}.npz"
    data = np.load(npz_path, allow_pickle=False)

    shape = tuple(data["shape"])
    dense = np.zeros(shape, dtype=np.float32)
    dense[data["row_indices"], data["col_indices"]] = data["values"]

    meta = {
        "game_ids": data["game_ids"],
        "round_nums": data["round_nums"],
        "game_outcomes": data["game_outcomes"],
        "bet_types": data["bet_types"],
        "is_last_round": data["is_last_round"].astype(bool),
    }
    if "prompt_conditions" in data.files:
        meta["prompt_conditions"] = data["prompt_conditions"]
    if "balances" in data.files:
        meta["balances"] = data["balances"]

    if decision_point:
        mask = meta["is_last_round"]
        dense = dense[mask]
        meta = {k: v[mask] for k, v in meta.items()}

    return dense, meta


def load_gemma_npz(paradigm, layer, decision_point=True):
    """Load Gemma sparse NPZ and return dense features + metadata at decision point."""
    npz_path = DATA_ROOT / paradigm / "gemma" / f"sae_features_L{layer}.npz"
    data = np.load(npz_path, allow_pickle=False)

    shape = tuple(data["shape"])
    dense = np.zeros(shape, dtype=np.float32)
    dense[data["row_indices"], data["col_indices"]] = data["values"]

    meta = {
        "game_ids": data["game_ids"],
        "round_nums": data["round_nums"],
        "game_outcomes": data["game_outcomes"],
        "bet_types": data["bet_types"],
        "is_last_round": data["is_last_round"].astype(bool),
    }
    if "prompt_conditions" in data.files:
        meta["prompt_conditions"] = data["prompt_conditions"]

    if decision_point:
        mask = meta["is_last_round"]
        dense = dense[mask]
        meta = {k: v[mask] for k, v in meta.items()}

    return dense, meta


# ===========================================================================
# Analysis 1: LLaMA IC+SM Factor Decomposition
# ===========================================================================
def analysis1_factor_decomposition():
    print(f"[{timestamp()}] === Analysis 1: LLaMA IC+SM Factor Decomposition (L22) ===")

    # Load IC and SM at L22 decision point
    ic_feat, ic_meta = load_llama_npz("investment_choice", 22)
    sm_feat, sm_meta = load_llama_npz("slot_machine", 22)

    print(f"[{timestamp()}]   IC: {ic_feat.shape[0]} games, SM: {sm_feat.shape[0]} games")

    n_ic = ic_feat.shape[0]
    n_sm = sm_feat.shape[0]

    # Labels
    ic_labels = (ic_meta["game_outcomes"] == "bankruptcy").astype(np.float64)
    sm_labels = (sm_meta["game_outcomes"] == "bankruptcy").astype(np.float64)

    # Bet types (variable=1, fixed=0)
    ic_bet = (ic_meta["bet_types"] == "variable").astype(np.float64)
    sm_bet = (sm_meta["bet_types"] == "variable").astype(np.float64)

    # Paradigm indicator (SM=1, IC=0)
    ic_paradigm = np.zeros(n_ic, dtype=np.float64)
    sm_paradigm = np.ones(n_sm, dtype=np.float64)

    # Combine
    combined_feat = np.vstack([ic_feat, sm_feat])  # (4800, 32768)
    outcome = np.concatenate([ic_labels, sm_labels])
    bettype = np.concatenate([ic_bet, sm_bet])
    paradigm = np.concatenate([ic_paradigm, sm_paradigm])
    n_total = combined_feat.shape[0]

    print(f"[{timestamp()}]   Combined: {n_total} games, {combined_feat.shape[1]} features")

    # Union-active features (activation rate >= 0.01 in either paradigm)
    ic_rate = (ic_feat != 0).mean(axis=0)
    sm_rate = (sm_feat != 0).mean(axis=0)
    active_mask = (ic_rate >= 0.01) | (sm_rate >= 0.01)
    active_idx = np.where(active_mask)[0]
    n_active = len(active_idx)

    print(f"[{timestamp()}]   Union-active features: {n_active}")

    # Filter to active features
    Y = combined_feat[:, active_mask].astype(np.float64)  # (4800, n_active)

    # Design matrix: [ones, outcome, bettype, paradigm]
    X = np.column_stack([np.ones(n_total), outcome, bettype, paradigm])  # (4800, 4)

    # Vectorized OLS: Beta = (X'X)^-1 X' Y
    XtX = X.T @ X  # (4, 4)
    XtX_inv = np.linalg.inv(XtX)
    Beta = XtX_inv @ (X.T @ Y)  # (4, n_active)

    # Residuals and standard errors
    Y_hat = X @ Beta
    residuals = Y - Y_hat
    n, p = X.shape
    dof = n - p
    RSS = (residuals ** 2).sum(axis=0)  # (n_active,)
    sigma2 = RSS / dof  # (n_active,)

    # Standard errors for each coefficient
    se = np.sqrt(np.outer(np.diag(XtX_inv), sigma2))  # (4, n_active)

    # t-statistics
    t_stats = Beta / se  # (4, n_active)

    # p-values (two-sided)
    p_values = 2 * stats.t.sf(np.abs(t_stats), dof)  # (4, n_active)

    # Count significant at p<0.01
    coeff_names = ["intercept", "outcome_bk", "bet_type_variable", "paradigm_sm"]
    results = {}
    for i, name in enumerate(coeff_names):
        n_sig = (p_values[i] < 0.01).sum()
        pct_sig = n_sig / n_active * 100
        mean_beta = Beta[i].mean()
        median_beta = np.median(Beta[i])
        mean_t = t_stats[i].mean()
        results[name] = {
            "n_significant_p01": int(n_sig),
            "pct_significant": round(pct_sig, 2),
            "mean_beta": round(float(mean_beta), 6),
            "median_beta": round(float(median_beta), 6),
            "mean_t_stat": round(float(mean_t), 4),
            "n_positive_sig": int(((p_values[i] < 0.01) & (Beta[i] > 0)).sum()),
            "n_negative_sig": int(((p_values[i] < 0.01) & (Beta[i] < 0)).sum()),
        }
        print(f"[{timestamp()}]   {name}: {n_sig}/{n_active} sig ({pct_sig:.1f}%), mean_beta={mean_beta:.6f}")

    output = {
        "layer": 22,
        "n_ic_games": int(n_ic),
        "n_sm_games": int(n_sm),
        "n_total_games": int(n_total),
        "n_features_total": int(combined_feat.shape[1]),
        "n_union_active_features": int(n_active),
        "ic_bk_rate": round(float(ic_labels.mean()), 4),
        "sm_bk_rate": round(float(sm_labels.mean()), 4),
        "coefficients": results,
    }

    print(f"[{timestamp()}]   IC BK rate: {ic_labels.mean():.3f}, SM BK rate: {sm_labels.mean():.3f}")
    return output


# ===========================================================================
# Analysis 2: LLaMA SM Prompt Component Analysis
# ===========================================================================
def analysis2_prompt_component():
    print(f"\n[{timestamp()}] === Analysis 2: LLaMA SM Prompt Component Analysis ===")

    components = ["G", "M", "H", "W", "P"]
    layers = [8, 12, 20, 25, 30]
    results = {}

    for layer in layers:
        print(f"[{timestamp()}]   Loading SM L{layer}...")
        feat, meta = load_llama_npz("slot_machine", layer)
        n_games = feat.shape[0]
        n_features = feat.shape[1]
        outcomes = (meta["game_outcomes"] == "bankruptcy").astype(np.float64)
        conditions = meta["prompt_conditions"]

        layer_results = {}

        for comp in components:
            # Determine presence of component in each condition string
            has_comp = np.array([comp in cond for cond in conditions])
            n_with = has_comp.sum()
            n_without = (~has_comp).sum()

            # BK rates
            bk_with = outcomes[has_comp].mean()
            bk_without = outcomes[~has_comp].mean()
            bk_ratio = bk_with / max(bk_without, 1e-10)

            # Active features (>= 1% activation rate in full dataset)
            act_rate = (feat != 0).mean(axis=0)
            active_mask = act_rate >= 0.01
            active_idx = np.where(active_mask)[0]
            n_active = len(active_idx)

            feat_active = feat[:, active_mask].astype(np.float64)

            # Per-feature t-test: with component vs without component
            feat_with = feat_active[has_comp]
            feat_without = feat_active[~has_comp]

            # Vectorized Welch's t-test
            n1, n2 = feat_with.shape[0], feat_without.shape[0]
            mean1 = feat_with.mean(axis=0)
            mean2 = feat_without.mean(axis=0)
            var1 = feat_with.var(axis=0, ddof=1)
            var2 = feat_without.var(axis=0, ddof=1)

            se_diff = np.sqrt(var1 / n1 + var2 / n2)
            se_diff = np.where(se_diff == 0, 1e-10, se_diff)
            t_comp = (mean1 - mean2) / se_diff

            # Welch-Satterthwaite degrees of freedom
            num = (var1 / n1 + var2 / n2) ** 2
            denom = (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
            denom = np.where(denom == 0, 1e-10, denom)
            df_welch = num / denom
            df_welch = np.clip(df_welch, 1, 1e6)

            p_comp = 2 * stats.t.sf(np.abs(t_comp), df_welch)
            n_sig_comp = (p_comp < 0.01).sum()

            # Per-feature interaction: feature ~ outcome + component + outcome*component
            # Design: [1, outcome, comp, outcome*comp]
            comp_vec = has_comp.astype(np.float64)
            interaction = outcomes * comp_vec
            X_int = np.column_stack([np.ones(n_games), outcomes, comp_vec, interaction])

            XtX = X_int.T @ X_int
            try:
                XtX_inv = np.linalg.inv(XtX)
            except np.linalg.LinAlgError:
                XtX_inv = np.linalg.pinv(XtX)

            Beta_int = XtX_inv @ (X_int.T @ feat_active)  # (4, n_active)
            Y_hat_int = X_int @ Beta_int
            resid_int = feat_active - Y_hat_int
            dof_int = n_games - 4
            RSS_int = (resid_int ** 2).sum(axis=0)
            sigma2_int = RSS_int / dof_int

            se_int = np.sqrt(np.outer(np.diag(XtX_inv), sigma2_int))
            se_int = np.where(se_int == 0, 1e-10, se_int)
            t_int = Beta_int / se_int
            p_int = 2 * stats.t.sf(np.abs(t_int), dof_int)

            # Interaction term is index 3
            n_interaction_sig = (p_int[3] < 0.01).sum()

            # "amplifies_bk" features: interaction sig AND component amplifies BK-feature relationship
            # This means: the coefficient on outcome is larger when component is present
            # i.e., interaction beta has same sign as outcome beta (both positive or both negative)
            # More precisely: interaction is sig AND sign(interaction_beta) matches sign(outcome_beta)
            interaction_sig_mask = p_int[3] < 0.01
            amplifies_mask = interaction_sig_mask & (np.sign(Beta_int[3]) == np.sign(Beta_int[1]))
            # Also require outcome to be significant to be meaningful
            amplifies_mask = amplifies_mask & (p_int[1] < 0.05)
            n_amplifies = amplifies_mask.sum()

            layer_results[comp] = {
                "n_with_component": int(n_with),
                "n_without_component": int(n_without),
                "bk_rate_with": round(float(bk_with), 4),
                "bk_rate_without": round(float(bk_without), 4),
                "bk_ratio": round(float(bk_ratio), 4),
                "n_active_features": int(n_active),
                "n_component_sig_features": int(n_sig_comp),
                "pct_component_sig": round(float(n_sig_comp / max(n_active, 1) * 100), 2),
                "n_interaction_sig_features": int(n_interaction_sig),
                "pct_interaction_sig": round(float(n_interaction_sig / max(n_active, 1) * 100), 2),
                "n_amplifies_bk": int(n_amplifies),
                "pct_amplifies_bk": round(float(n_amplifies / max(n_active, 1) * 100), 2),
            }

            print(f"[{timestamp()}]     L{layer} {comp}: BK {bk_with:.3f}/{bk_without:.3f} (ratio={bk_ratio:.3f}), "
                  f"comp_sig={n_sig_comp}, interaction_sig={n_interaction_sig}, amplifies={n_amplifies}")

        results[f"L{layer}"] = layer_results

    return results


# ===========================================================================
# Analysis 3: Gemma IC Classification AUC
# ===========================================================================
def analysis3_gemma_auc():
    print(f"\n[{timestamp()}] === Analysis 3: Gemma IC Classification AUC ===")

    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    target_layers = [18, 20, 22, 26, 30]
    results = {}

    for layer in target_layers:
        print(f"[{timestamp()}]   Loading Gemma IC L{layer}...")
        feat, meta = load_gemma_npz("investment_choice", layer)
        labels = (meta["game_outcomes"] == "bankruptcy").astype(np.int32)

        n_games = feat.shape[0]
        n_bk = labels.sum()
        n_safe = n_games - n_bk

        print(f"[{timestamp()}]     {n_games} games, {n_bk} BK, {n_safe} safe")

        # Filter active features
        act_rate = (feat != 0).mean(axis=0)
        active_mask = act_rate >= 0.01
        feat_active = feat[:, active_mask]
        n_active = active_mask.sum()
        print(f"[{timestamp()}]     {n_active} active features")

        # PCA(50)
        n_components = min(50, feat_active.shape[1], feat_active.shape[0] - 1)
        pca = PCA(n_components=n_components, random_state=42)
        feat_pca = pca.fit_transform(feat_active)

        # LogReg(balanced) with 5-fold StratifiedKFold AUC
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs = []

        for train_idx, test_idx in skf.split(feat_pca, labels):
            X_train, X_test = feat_pca[train_idx], feat_pca[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            clf = LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000, random_state=42)
            clf.fit(X_train, y_train)

            if len(np.unique(y_test)) < 2:
                continue

            y_prob = clf.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
            aucs.append(auc)

        mean_auc = np.mean(aucs) if aucs else 0.0
        std_auc = np.std(aucs) if aucs else 0.0

        results[f"L{layer}"] = {
            "n_games": int(n_games),
            "n_bk": int(n_bk),
            "bk_rate": round(float(n_bk / n_games), 4),
            "n_active_features": int(n_active),
            "n_pca_components": int(n_components),
            "mean_auc": round(float(mean_auc), 4),
            "std_auc": round(float(std_auc), 4),
            "fold_aucs": [round(float(a), 4) for a in aucs],
        }

        print(f"[{timestamp()}]     AUC: {mean_auc:.4f} +/- {std_auc:.4f}")

    return results


# ===========================================================================
# Analysis 4: Cross-model IC BK rate comparison
# ===========================================================================
def analysis4_cross_model_bk():
    print(f"\n[{timestamp()}] === Analysis 4: Cross-model IC BK Rate Comparison ===")

    results = {}

    for model_name, loader_fn, paradigm_dir in [
        ("gemma", load_gemma_npz, "investment_choice"),
        ("llama", load_llama_npz, "investment_choice"),
    ]:
        feat, meta = loader_fn(paradigm_dir, 22)
        labels = (meta["game_outcomes"] == "bankruptcy").astype(np.int32)
        bet_types = meta["bet_types"]

        fixed_mask = bet_types == "fixed"
        variable_mask = bet_types == "variable"

        fixed_bk = labels[fixed_mask].mean() if fixed_mask.sum() > 0 else 0.0
        variable_bk = labels[variable_mask].mean() if variable_mask.sum() > 0 else 0.0
        overall_bk = labels.mean()

        results[model_name] = {
            "n_games": int(len(labels)),
            "n_fixed": int(fixed_mask.sum()),
            "n_variable": int(variable_mask.sum()),
            "fixed_bk_pct": round(float(fixed_bk * 100), 2),
            "variable_bk_pct": round(float(variable_bk * 100), 2),
            "overall_bk_pct": round(float(overall_bk * 100), 2),
            "autonomy_effect_pp": round(float((variable_bk - fixed_bk) * 100), 2),
        }

        print(f"[{timestamp()}]   {model_name}: Fixed BK={fixed_bk*100:.1f}%, Variable BK={variable_bk*100:.1f}%, "
              f"Autonomy effect={((variable_bk - fixed_bk) * 100):.1f}pp")

    return results


# ===========================================================================
# Analysis 5: LLaMA IC within-bet-type SAE cross-domain (IC+SM)
# ===========================================================================
def analysis5_within_bettype_crossdomain():
    print(f"\n[{timestamp()}] === Analysis 5: LLaMA IC Within-Bet-Type Cross-Domain (IC+SM L22) ===")

    ic_feat, ic_meta = load_llama_npz("investment_choice", 22)
    sm_feat, sm_meta = load_llama_npz("slot_machine", 22)

    ic_labels = (ic_meta["game_outcomes"] == "bankruptcy").astype(np.int32)
    sm_labels = (sm_meta["game_outcomes"] == "bankruptcy").astype(np.int32)
    ic_bets = ic_meta["bet_types"]
    sm_bets = sm_meta["bet_types"]

    results = {}

    for bet_type in ["fixed", "variable"]:
        print(f"[{timestamp()}]   Bet type: {bet_type}")

        # Filter by bet type
        ic_mask = ic_bets == bet_type
        sm_mask = sm_bets == bet_type

        ic_f = ic_feat[ic_mask]
        sm_f = sm_feat[sm_mask]
        ic_l = ic_labels[ic_mask]
        sm_l = sm_labels[sm_mask]

        n_ic = ic_f.shape[0]
        n_sm = sm_f.shape[0]
        ic_bk_n = ic_l.sum()
        sm_bk_n = sm_l.sum()

        print(f"[{timestamp()}]     IC: {n_ic} games ({ic_bk_n} BK), SM: {n_sm} games ({sm_bk_n} BK)")

        # Active features in BOTH paradigms (activation rate >= 0.01)
        ic_rate = (ic_f != 0).mean(axis=0)
        sm_rate = (sm_f != 0).mean(axis=0)
        both_active = (ic_rate >= 0.01) & (sm_rate >= 0.01)
        active_idx = np.where(both_active)[0]
        n_both_active = len(active_idx)

        print(f"[{timestamp()}]     Both-active features: {n_both_active}")

        if n_both_active == 0 or ic_bk_n < 5 or sm_bk_n < 5:
            results[bet_type] = {
                "n_ic": int(n_ic),
                "n_sm": int(n_sm),
                "n_both_active": int(n_both_active),
                "skipped": True,
                "reason": "too few BK games or no active features",
            }
            continue

        # Compute Cohen's d per feature for BK vs Safe in each paradigm
        def cohens_d_vectorized(features, labels):
            """Compute Cohen's d for BK (1) vs Safe (0) per feature."""
            bk_mask = labels == 1
            safe_mask = labels == 0
            n_bk = bk_mask.sum()
            n_safe = safe_mask.sum()

            if n_bk < 2 or n_safe < 2:
                return np.zeros(features.shape[1])

            feat_bk = features[bk_mask]
            feat_safe = features[safe_mask]

            mean_bk = feat_bk.mean(axis=0)
            mean_safe = feat_safe.mean(axis=0)

            var_bk = feat_bk.var(axis=0, ddof=1)
            var_safe = feat_safe.var(axis=0, ddof=1)

            # Pooled std
            pooled_var = ((n_bk - 1) * var_bk + (n_safe - 1) * var_safe) / (n_bk + n_safe - 2)
            pooled_std = np.sqrt(pooled_var)
            pooled_std = np.where(pooled_std == 0, 1e-10, pooled_std)

            d = (mean_bk - mean_safe) / pooled_std
            return d

        ic_d = cohens_d_vectorized(ic_f[:, both_active].astype(np.float64), ic_l)
        sm_d = cohens_d_vectorized(sm_f[:, both_active].astype(np.float64), sm_l)

        # Sign-consistent features: same sign AND |d| > small threshold to avoid noise
        min_d = 0.0  # No minimum threshold, just check sign consistency
        sign_consistent = (np.sign(ic_d) == np.sign(sm_d)) & (ic_d != 0) & (sm_d != 0)
        n_sign_consistent = sign_consistent.sum()

        # Also count with a meaningful effect size threshold
        meaningful_d = 0.1
        sign_consistent_meaningful = (
            sign_consistent
            & (np.abs(ic_d) >= meaningful_d)
            & (np.abs(sm_d) >= meaningful_d)
        )
        n_sign_consistent_meaningful = sign_consistent_meaningful.sum()

        # Correlation of d values
        d_corr, d_corr_p = stats.pearsonr(ic_d, sm_d)

        # Count by direction
        both_positive = (sign_consistent & (ic_d > 0)).sum()
        both_negative = (sign_consistent & (ic_d < 0)).sum()

        results[bet_type] = {
            "n_ic": int(n_ic),
            "n_sm": int(n_sm),
            "ic_bk_n": int(ic_bk_n),
            "sm_bk_n": int(sm_bk_n),
            "ic_bk_rate": round(float(ic_l.mean()), 4),
            "sm_bk_rate": round(float(sm_l.mean()), 4),
            "n_both_active_features": int(n_both_active),
            "n_sign_consistent": int(n_sign_consistent),
            "pct_sign_consistent": round(float(n_sign_consistent / max(n_both_active, 1) * 100), 2),
            "n_sign_consistent_d01": int(n_sign_consistent_meaningful),
            "pct_sign_consistent_d01": round(float(n_sign_consistent_meaningful / max(n_both_active, 1) * 100), 2),
            "both_positive": int(both_positive),
            "both_negative": int(both_negative),
            "d_correlation": round(float(d_corr), 4),
            "d_correlation_p": float(d_corr_p),
            "mean_ic_d": round(float(np.mean(np.abs(ic_d))), 4),
            "mean_sm_d": round(float(np.mean(np.abs(sm_d))), 4),
        }

        print(f"[{timestamp()}]     Sign-consistent: {n_sign_consistent}/{n_both_active} ({n_sign_consistent/n_both_active*100:.1f}%)")
        print(f"[{timestamp()}]     Sign-consistent (|d|>=0.1): {n_sign_consistent_meaningful}/{n_both_active}")
        print(f"[{timestamp()}]     d correlation: r={d_corr:.4f}, p={d_corr_p:.2e}")

    return results


# ===========================================================================
# Main
# ===========================================================================
def main():
    print(f"[{timestamp()}] Starting LLaMA Symmetric Analyses")
    print(f"[{timestamp()}] Output: {OUTPUT_PATH}")
    print()

    all_results = {}
    t0 = time.time()

    # Analysis 1
    t1 = time.time()
    all_results["analysis1_factor_decomposition"] = analysis1_factor_decomposition()
    print(f"[{timestamp()}]   -> Done in {time.time()-t1:.1f}s")

    # Analysis 2
    t2 = time.time()
    all_results["analysis2_prompt_component"] = analysis2_prompt_component()
    print(f"[{timestamp()}]   -> Done in {time.time()-t2:.1f}s")

    # Analysis 3
    t3 = time.time()
    all_results["analysis3_gemma_ic_auc"] = analysis3_gemma_auc()
    print(f"[{timestamp()}]   -> Done in {time.time()-t3:.1f}s")

    # Analysis 4
    t4 = time.time()
    all_results["analysis4_cross_model_bk"] = analysis4_cross_model_bk()
    print(f"[{timestamp()}]   -> Done in {time.time()-t4:.1f}s")

    # Analysis 5
    t5 = time.time()
    all_results["analysis5_within_bettype_crossdomain"] = analysis5_within_bettype_crossdomain()
    print(f"[{timestamp()}]   -> Done in {time.time()-t5:.1f}s")

    # Save
    all_results["metadata"] = {
        "script": "run_llama_symmetric.py",
        "timestamp": datetime.now().isoformat(),
        "total_runtime_seconds": round(time.time() - t0, 1),
        "random_seed": 42,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[{timestamp()}] All analyses complete in {time.time()-t0:.1f}s")
    print(f"[{timestamp()}] Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
