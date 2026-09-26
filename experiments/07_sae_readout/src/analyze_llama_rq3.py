#!/usr/bin/env python3
"""
V13 LLaMA RQ3: Condition-Dependent Modulation Analysis
=======================================================
Fills the gaps where only Gemma was analyzed (v8).

Analyses:
  1. R1 Within-Bet-Type Classification (confound-free BK vs Safe)
  2. G-Prompt BK Direction Alignment (+ all prompt components)
  3. Fixed/Variable Autonomy Paradox (BK-projection comparisons)
  4. Prompt Component Hierarchy (SM 32 conditions, MW 32 conditions)
  5. Prompt Condition BK-Projection (IC: BASE/G/M/GM)

Uses hidden_states_dp.npz files directly (no GPU required).

Usage:
    conda run -n llm-addiction python sae_v3_analysis/src/analyze_llama_rq3.py
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

from scipy import stats
from scipy.stats import pearsonr, mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATA_ROOT = Path("/home/v-seungplee/data/llm-addiction/sae_features_v3")
OUT_DIR = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis/results/json")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PARADIGM_PATHS = {
    "SM": DATA_ROOT / "slot_machine" / "llama" / "hidden_states_dp.npz",
    "IC": DATA_ROOT / "investment_choice" / "llama" / "hidden_states_dp.npz",
    "MW": DATA_ROOT / "mystery_wheel" / "llama" / "hidden_states_dp.npz",
}

# Layer 22 is index 2 in the [8, 12, 22, 25, 30] array
L22_INDEX = 2

# Gemma v8 reference values for comparison
GEMMA_V8_REF = {
    "within_bet_auc": {
        "IC_fixed": 0.753,
        "IC_variable": 0.692,
        "SM_variable": 0.805,
        "MW_fixed": 0.617,
    },
    "g_prompt_cos": {
        "SM": 0.69,
        "IC": -0.87,
        "MW": -0.79,
    },
    "variable_bk_proj_gap": {
        "note": "Variable had BK-proj 1.4-2.0 lower despite higher behavioral risk",
    },
    "prompt_hierarchy_sm": "G > M = W > H >> P",
}

N_PERMUTATIONS = 200
CV_FOLDS = 5


def log(msg: str) -> None:
    """Timestamped console output."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_paradigm(paradigm: str) -> Dict[str, np.ndarray]:
    """Load an NPZ file and return a dictionary of arrays."""
    path = PARADIGM_PATHS[paradigm]
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def get_l22_hidden(data: Dict[str, np.ndarray]) -> np.ndarray:
    """Extract layer-22 hidden states from (n, 5, 4096) array."""
    layers = data["layers"]
    l22_idx = int(np.where(layers == 22)[0][0])
    return data["hidden_states"][:, l22_idx, :]


def bk_labels(data: Dict[str, np.ndarray]) -> np.ndarray:
    """Return binary labels: 1 = bankruptcy, 0 = safe (voluntary_stop or max_rounds)."""
    return (data["game_outcomes"] == "bankruptcy").astype(np.int32)


# ---------------------------------------------------------------------------
# Utility: cosine similarity
# ---------------------------------------------------------------------------
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


# ---------------------------------------------------------------------------
# Analysis 1: R1 Within-Bet-Type Classification (confound-free)
# ---------------------------------------------------------------------------
def analysis1_within_bettype_classification() -> Dict[str, Any]:
    """
    For each paradigm, split by bet_type (fixed / variable).
    Within each subset: classify BK vs Safe using L22 hidden states.
    Pipeline: StandardScaler -> PCA(50) -> LogReg(balanced, C=1.0) -> 5-fold CV.
    Report AUC + permutation test (200 shuffles).

    Optimization: pre-compute PCA transform once, then run permutation tests
    on the reduced-dimensionality data (LogReg only) for speed.
    """
    log("=" * 72)
    log("ANALYSIS 1: R1 Within-Bet-Type Classification (confound-free)")
    log("=" * 72)

    results = {}

    for paradigm in ["SM", "IC", "MW"]:
        data = load_paradigm(paradigm)
        hidden = get_l22_hidden(data)
        labels = bk_labels(data)
        bet_types = data["bet_types"]

        paradigm_results = {}

        for bt in ["fixed", "variable"]:
            mask = bet_types == bt
            X_raw = hidden[mask]
            y = labels[mask]
            n_bk = int(y.sum())
            n_safe = int((y == 0).sum())

            if n_bk < 5 or n_safe < 5:
                log(f"  {paradigm} {bt}: SKIPPED (n_bk={n_bk}, n_safe={n_safe})")
                paradigm_results[bt] = {
                    "status": "skipped",
                    "n_total": int(mask.sum()),
                    "n_bk": n_bk,
                    "n_safe": n_safe,
                    "reason": f"Insufficient BK samples (n_bk={n_bk})",
                }
                continue

            # Step 1: Pre-compute PCA transform (deterministic, label-independent)
            n_comp = min(50, X_raw.shape[0] - 1, X_raw.shape[1])
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_raw)
            pca = PCA(n_components=n_comp, random_state=RANDOM_SEED)
            X_pca = pca.fit_transform(X_scaled)

            # Step 2: 5-fold CV on PCA-transformed data with held-out predictions
            cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
            auc_scores = []
            held_out_proba = np.zeros(len(y))  # Store held-out predictions

            for train_idx, test_idx in cv.split(X_pca, y):
                clf_fold = LogisticRegression(
                    C=1.0, class_weight="balanced",
                    max_iter=1000, random_state=RANDOM_SEED,
                )
                clf_fold.fit(X_pca[train_idx], y[train_idx])
                proba_fold = clf_fold.predict_proba(X_pca[test_idx])[:, 1]
                held_out_proba[test_idx] = proba_fold
                fold_auc = roc_auc_score(y[test_idx], proba_fold)
                auc_scores.append(fold_auc)

            auc_scores = np.array(auc_scores)
            mean_auc = float(auc_scores.mean())
            std_auc = float(auc_scores.std())

            # Step 3: Fast permutation test using held-out predictions
            # Shuffle true labels against fixed predictions (standard approach,
            # same as in run_hidden_state_analyses.py line 182)
            null_aucs = []
            for perm_i in range(N_PERMUTATIONS):
                y_perm = np.random.permutation(y)
                try:
                    null_auc = roc_auc_score(y_perm, held_out_proba)
                    null_aucs.append(null_auc)
                except ValueError:
                    null_aucs.append(0.5)

            null_aucs = np.array(null_aucs)
            perm_p = float((null_aucs >= mean_auc).mean())

            # Gemma v8 reference
            ref_key = f"{paradigm}_{bt}"
            gemma_ref = GEMMA_V8_REF["within_bet_auc"].get(ref_key, None)

            paradigm_results[bt] = {
                "status": "computed",
                "n_total": int(mask.sum()),
                "n_bk": n_bk,
                "n_safe": n_safe,
                "bk_rate": float(y.mean()),
                "n_pca_components": n_comp,
                "auc_mean": mean_auc,
                "auc_std": std_auc,
                "auc_folds": auc_scores.tolist(),
                "perm_p": perm_p,
                "perm_null_mean": float(null_aucs.mean()),
                "perm_null_std": float(null_aucs.std()),
                "gemma_v8_auc": gemma_ref,
            }

            sig_marker = "***" if perm_p < 0.005 else "**" if perm_p < 0.01 else "*" if perm_p < 0.05 else "ns"
            ref_str = f" (Gemma v8: {gemma_ref:.3f})" if gemma_ref else ""
            log(f"  {paradigm} {bt}: AUC={mean_auc:.3f}+/-{std_auc:.3f}, perm_p={perm_p:.3f} {sig_marker}{ref_str}")

        results[paradigm] = paradigm_results

    return results


# ---------------------------------------------------------------------------
# Analysis 2: G-Prompt BK Direction Alignment (+ all components)
# ---------------------------------------------------------------------------
def analysis2_g_prompt_alignment() -> Dict[str, Any]:
    """
    Compute BK direction at L22 for each paradigm: mean(BK) - mean(Safe).
    Compute prompt-component direction: mean(with-component) - mean(without).
    Compute cosine similarity between BK direction and component direction.
    """
    log("\n" + "=" * 72)
    log("ANALYSIS 2: G-Prompt BK Direction Alignment (all components)")
    log("=" * 72)

    # SM and MW have components G, M, H, W, P (32 conditions = 2^5)
    # IC has only G, M (4 conditions = 2^2)
    component_map = {
        "SM": ["G", "M", "H", "W", "P"],
        "IC": ["G", "M"],
        "MW": ["G", "M", "H", "W", "P"],
    }

    results = {}

    for paradigm in ["SM", "IC", "MW"]:
        data = load_paradigm(paradigm)
        hidden = get_l22_hidden(data)
        labels = bk_labels(data)
        conditions = data["prompt_conditions"]

        # BK direction
        bk_mean = hidden[labels == 1].mean(axis=0)
        safe_mean = hidden[labels == 0].mean(axis=0)
        bk_dir = bk_mean - safe_mean

        components = component_map[paradigm]
        comp_results = {}

        for comp in components:
            has_comp = np.array([comp in str(c) for c in conditions])
            n_with = int(has_comp.sum())
            n_without = int((~has_comp).sum())

            if n_with < 10 or n_without < 10:
                log(f"  {paradigm} {comp}: SKIPPED (n_with={n_with}, n_without={n_without})")
                continue

            comp_dir = hidden[has_comp].mean(axis=0) - hidden[~has_comp].mean(axis=0)
            cos = cosine_sim(bk_dir, comp_dir)

            # BK rates with/without component
            bk_rate_with = float(labels[has_comp].mean())
            bk_rate_without = float(labels[~has_comp].mean())
            bk_rate_diff = bk_rate_with - bk_rate_without

            comp_results[comp] = {
                "cos_with_bk_dir": cos,
                "n_with": n_with,
                "n_without": n_without,
                "bk_rate_with": bk_rate_with,
                "bk_rate_without": bk_rate_without,
                "bk_rate_diff": bk_rate_diff,
            }

            log(f"  {paradigm} {comp}: cos(comp_dir, BK_dir)={cos:+.3f}, "
                f"BK_rate: with={bk_rate_with:.3f}, without={bk_rate_without:.3f} "
                f"(diff={bk_rate_diff:+.3f})")

        # Gemma v8 reference for G
        gemma_g_ref = {
            "SM": GEMMA_V8_REF["g_prompt_cos"].get("SM"),
            "IC": GEMMA_V8_REF["g_prompt_cos"].get("IC"),
            "MW": GEMMA_V8_REF["g_prompt_cos"].get("MW"),
        }

        results[paradigm] = {
            "components": comp_results,
            "gemma_v8_g_cos": gemma_g_ref.get(paradigm),
            "n_bk": int(labels.sum()),
            "n_safe": int((labels == 0).sum()),
            "overall_bk_rate": float(labels.mean()),
        }

        if "G" in comp_results:
            ref = gemma_g_ref.get(paradigm)
            ref_str = f" (Gemma v8: {ref:+.3f})" if ref else ""
            log(f"  >> {paradigm} G cosine = {comp_results['G']['cos_with_bk_dir']:+.3f}{ref_str}")

    return results


# ---------------------------------------------------------------------------
# Analysis 3: Fixed/Variable Autonomy Paradox
# ---------------------------------------------------------------------------
def analysis3_autonomy_paradox() -> Dict[str, Any]:
    """
    Compute BK-projection for each game: dot(hidden, BK_direction_unit).
    Compare mean BK-projection: Fixed vs Variable, per paradigm.
    Compare with behavioral BK rates.
    Compute correlations: BK-proj vs balance, vs round_num.
    """
    log("\n" + "=" * 72)
    log("ANALYSIS 3: Fixed/Variable Autonomy Paradox")
    log("=" * 72)

    results = {}

    for paradigm in ["SM", "IC", "MW"]:
        data = load_paradigm(paradigm)
        hidden = get_l22_hidden(data)
        labels = bk_labels(data)
        bet_types = data["bet_types"]
        balances = data["balances"].astype(np.float64)
        round_nums = data["round_nums"].astype(np.float64)

        # Compute BK direction (unit vector)
        bk_dir = hidden[labels == 1].mean(axis=0) - hidden[labels == 0].mean(axis=0)
        bk_dir_norm = np.linalg.norm(bk_dir)
        if bk_dir_norm < 1e-12:
            log(f"  {paradigm}: BK direction near zero, SKIPPED")
            continue
        bk_dir_unit = bk_dir / bk_dir_norm

        # BK-projection for every game
        bk_proj = hidden @ bk_dir_unit  # (n_games,)

        fix_mask = bet_types == "fixed"
        var_mask = bet_types == "variable"

        # BK-projection stats
        fix_proj_mean = float(bk_proj[fix_mask].mean())
        fix_proj_std = float(bk_proj[fix_mask].std())
        var_proj_mean = float(bk_proj[var_mask].mean())
        var_proj_std = float(bk_proj[var_mask].std())
        proj_diff = var_proj_mean - fix_proj_mean

        # Mann-Whitney U test for projection difference
        u_stat, u_pval = mannwhitneyu(bk_proj[var_mask], bk_proj[fix_mask], alternative="two-sided")

        # Behavioral BK rates
        fix_bk_rate = float(labels[fix_mask].mean())
        var_bk_rate = float(labels[var_mask].mean())
        bk_rate_diff = var_bk_rate - fix_bk_rate

        # Paradox detection: variable has higher BK rate but lower BK-projection
        paradox = (bk_rate_diff > 0) and (proj_diff < 0)

        # Correlations: BK-proj vs balance, vs round_num
        r_balance, p_balance = pearsonr(bk_proj, balances)
        r_round, p_round = pearsonr(bk_proj, round_nums)

        # Per-bet-type correlations
        r_bal_fix, p_bal_fix = pearsonr(bk_proj[fix_mask], balances[fix_mask])
        r_bal_var, p_bal_var = pearsonr(bk_proj[var_mask], balances[var_mask])

        results[paradigm] = {
            "n_fixed": int(fix_mask.sum()),
            "n_variable": int(var_mask.sum()),
            "bk_rate_fixed": fix_bk_rate,
            "bk_rate_variable": var_bk_rate,
            "bk_rate_diff_var_minus_fix": bk_rate_diff,
            "bk_proj_fixed_mean": fix_proj_mean,
            "bk_proj_fixed_std": fix_proj_std,
            "bk_proj_variable_mean": var_proj_mean,
            "bk_proj_variable_std": var_proj_std,
            "bk_proj_diff_var_minus_fix": proj_diff,
            "mann_whitney_u": float(u_stat),
            "mann_whitney_p": float(u_pval),
            "autonomy_paradox_detected": paradox,
            "corr_bk_proj_balance": {
                "r": float(r_balance),
                "p": float(p_balance),
            },
            "corr_bk_proj_round_num": {
                "r": float(r_round),
                "p": float(p_round),
            },
            "corr_bk_proj_balance_fixed": {
                "r": float(r_bal_fix),
                "p": float(p_bal_fix),
            },
            "corr_bk_proj_balance_variable": {
                "r": float(r_bal_var),
                "p": float(p_bal_var),
            },
        }

        paradox_str = " ** PARADOX **" if paradox else ""
        log(f"  {paradigm}: BK_proj Fixed={fix_proj_mean:.3f}, Variable={var_proj_mean:.3f} "
            f"(diff={proj_diff:+.3f}, MWU p={u_pval:.4f}){paradox_str}")
        log(f"    Behavioral BK: Fixed={fix_bk_rate:.3f}, Variable={var_bk_rate:.3f} "
            f"(diff={bk_rate_diff:+.3f})")
        log(f"    r(BK_proj, balance)={r_balance:.3f} (p={p_balance:.4f}), "
            f"r(BK_proj, round)={r_round:.3f} (p={p_round:.4f})")

    return results


# ---------------------------------------------------------------------------
# Analysis 4: Prompt Component Hierarchy (SM and MW, 32 conditions)
# ---------------------------------------------------------------------------
def analysis4_prompt_hierarchy() -> Dict[str, Any]:
    """
    For each of 5 components (G, M, H, W, P):
    - Compute marginal BK rate with/without the component
    - Rank by effect size (difference in BK rate)
    - Also compute effect on BK-projection (representational effect)

    Apply to SM (32 conditions) and MW (32 conditions).
    """
    log("\n" + "=" * 72)
    log("ANALYSIS 4: Prompt Component Hierarchy (SM and MW, 32 conditions)")
    log("=" * 72)

    components = ["G", "M", "H", "W", "P"]
    results = {}

    for paradigm in ["SM", "MW"]:
        data = load_paradigm(paradigm)
        hidden = get_l22_hidden(data)
        labels = bk_labels(data)
        conditions = data["prompt_conditions"]

        # BK direction for projection
        bk_dir = hidden[labels == 1].mean(axis=0) - hidden[labels == 0].mean(axis=0)
        bk_dir_norm = np.linalg.norm(bk_dir)
        bk_dir_unit = bk_dir / bk_dir_norm if bk_dir_norm > 1e-12 else bk_dir
        bk_proj = hidden @ bk_dir_unit

        comp_effects = {}

        for comp in components:
            has = np.array([comp in str(c) for c in conditions])

            bk_rate_with = float(labels[has].mean())
            bk_rate_without = float(labels[~has].mean())
            bk_rate_diff = bk_rate_with - bk_rate_without

            proj_with = float(bk_proj[has].mean())
            proj_without = float(bk_proj[~has].mean())
            proj_diff = proj_with - proj_without

            comp_effects[comp] = {
                "n_with": int(has.sum()),
                "n_without": int((~has).sum()),
                "bk_rate_with": bk_rate_with,
                "bk_rate_without": bk_rate_without,
                "bk_rate_effect": bk_rate_diff,
                "bk_proj_with": proj_with,
                "bk_proj_without": proj_without,
                "bk_proj_effect": proj_diff,
            }

        # Rank components by behavioral effect size
        ranked = sorted(comp_effects.items(), key=lambda x: abs(x[1]["bk_rate_effect"]), reverse=True)
        ranking_bk = [(comp, eff["bk_rate_effect"]) for comp, eff in ranked]

        # Rank by representational effect
        ranked_proj = sorted(comp_effects.items(), key=lambda x: abs(x[1]["bk_proj_effect"]), reverse=True)
        ranking_proj = [(comp, eff["bk_proj_effect"]) for comp, eff in ranked_proj]

        results[paradigm] = {
            "components": comp_effects,
            "ranking_by_bk_rate": [{"component": c, "effect": e} for c, e in ranking_bk],
            "ranking_by_bk_proj": [{"component": c, "effect": e} for c, e in ranking_proj],
            "hierarchy_string_bk": " > ".join([f"{c}({e:+.3f})" for c, e in ranking_bk]),
            "hierarchy_string_proj": " > ".join([f"{c}({e:+.3f})" for c, e in ranking_proj]),
        }

        log(f"\n  {paradigm} BK rate hierarchy:")
        for i, (comp, eff) in enumerate(ranking_bk):
            log(f"    #{i+1}: {comp} effect={eff:+.3f} "
                f"(with={comp_effects[comp]['bk_rate_with']:.3f}, "
                f"without={comp_effects[comp]['bk_rate_without']:.3f})")

        log(f"  {paradigm} BK projection hierarchy:")
        for i, (comp, eff) in enumerate(ranking_proj):
            log(f"    #{i+1}: {comp} effect={eff:+.3f}")

    log(f"\n  Gemma v6 reference (SM): {GEMMA_V8_REF['prompt_hierarchy_sm']}")

    return results


# ---------------------------------------------------------------------------
# Analysis 5: Prompt Condition BK-Projection (IC: BASE/G/M/GM)
# ---------------------------------------------------------------------------
def analysis5_ic_prompt_conditions() -> Dict[str, Any]:
    """
    IC has 4 prompt conditions: BASE, G, M, GM.
    Compute BK-projection by condition.
    Also classify BK vs Safe within each condition.
    """
    log("\n" + "=" * 72)
    log("ANALYSIS 5: IC Prompt Condition BK-Projection")
    log("=" * 72)

    data = load_paradigm("IC")
    hidden = get_l22_hidden(data)
    labels = bk_labels(data)
    conditions = data["prompt_conditions"]
    bet_types = data["bet_types"]

    # BK direction (unit)
    bk_dir = hidden[labels == 1].mean(axis=0) - hidden[labels == 0].mean(axis=0)
    bk_dir_unit = bk_dir / np.linalg.norm(bk_dir)
    bk_proj = hidden @ bk_dir_unit

    unique_conds = sorted(np.unique(conditions))
    results = {"conditions": {}}

    for cond in unique_conds:
        cond_str = str(cond)
        mask = conditions == cond

        cond_bk_rate = float(labels[mask].mean())
        cond_proj_mean = float(bk_proj[mask].mean())
        cond_proj_std = float(bk_proj[mask].std())

        # Per bet-type within this condition
        bt_stats = {}
        for bt in ["fixed", "variable"]:
            bt_mask = mask & (bet_types == bt)
            if bt_mask.sum() > 0:
                bt_stats[bt] = {
                    "n": int(bt_mask.sum()),
                    "bk_rate": float(labels[bt_mask].mean()),
                    "bk_proj_mean": float(bk_proj[bt_mask].mean()),
                }

        results["conditions"][cond_str] = {
            "n": int(mask.sum()),
            "bk_rate": cond_bk_rate,
            "bk_proj_mean": cond_proj_mean,
            "bk_proj_std": cond_proj_std,
            "bet_type_breakdown": bt_stats,
        }

        log(f"  {cond_str:4s}: n={mask.sum():4d}, BK_rate={cond_bk_rate:.3f}, "
            f"BK_proj={cond_proj_mean:+.3f} +/- {cond_proj_std:.3f}")

    # Correlation: does having G or M increase BK-projection?
    has_g = np.array(["G" in str(c) for c in conditions])
    has_m = np.array(["M" in str(c) for c in conditions])

    g_proj_diff = float(bk_proj[has_g].mean() - bk_proj[~has_g].mean())
    m_proj_diff = float(bk_proj[has_m].mean() - bk_proj[~has_m].mean())

    results["g_effect_on_proj"] = g_proj_diff
    results["m_effect_on_proj"] = m_proj_diff
    log(f"\n  G effect on BK_proj: {g_proj_diff:+.3f}")
    log(f"  M effect on BK_proj: {m_proj_diff:+.3f}")

    return results


# ---------------------------------------------------------------------------
# Summary Comparison: Gemma v8 vs LLaMA v13
# ---------------------------------------------------------------------------
def print_comparison_summary(all_results: Dict[str, Any]) -> None:
    """Print a formatted comparison table between Gemma v8 and LLaMA v13."""
    log("\n" + "=" * 72)
    log("SUMMARY: Gemma (v8) vs LLaMA (v13) Comparison")
    log("=" * 72)

    # --- Analysis 1: Within-bet-type AUC ---
    log("\n--- Analysis 1: Within-Bet-Type Classification AUC ---")
    log(f"  {'Paradigm':>10s} {'BetType':>8s}  {'Gemma v8':>10s}  {'LLaMA v13':>10s}  {'Diff':>8s}  {'perm_p':>8s}")
    log(f"  {'----------':>10s} {'--------':>8s}  {'----------':>10s}  {'----------':>10s}  {'--------':>8s}  {'--------':>8s}")

    a1 = all_results.get("analysis1_within_bettype", {})
    for paradigm in ["SM", "IC", "MW"]:
        if paradigm not in a1:
            continue
        for bt in ["fixed", "variable"]:
            entry = a1[paradigm].get(bt, {})
            if entry.get("status") != "computed":
                status_str = entry.get("reason", "skipped")
                log(f"  {paradigm:>10s} {bt:>8s}  {'':>10s}  {status_str:>10s}")
                continue
            llama_auc = entry["auc_mean"]
            perm_p = entry["perm_p"]
            ref_key = f"{paradigm}_{bt}"
            gemma_auc = GEMMA_V8_REF["within_bet_auc"].get(ref_key)
            if gemma_auc is not None:
                diff = llama_auc - gemma_auc
                log(f"  {paradigm:>10s} {bt:>8s}  {gemma_auc:>10.3f}  {llama_auc:>10.3f}  {diff:>+8.3f}  {perm_p:>8.3f}")
            else:
                log(f"  {paradigm:>10s} {bt:>8s}  {'N/A':>10s}  {llama_auc:>10.3f}  {'':>8s}  {perm_p:>8.3f}")

    # --- Analysis 2: G-prompt cosine ---
    log("\n--- Analysis 2: G-Prompt BK Direction Cosine ---")
    log(f"  {'Paradigm':>10s}  {'Gemma v8':>10s}  {'LLaMA v13':>10s}")
    log(f"  {'----------':>10s}  {'----------':>10s}  {'----------':>10s}")

    a2 = all_results.get("analysis2_g_prompt_alignment", {})
    for paradigm in ["SM", "IC", "MW"]:
        if paradigm not in a2:
            continue
        comps = a2[paradigm].get("components", {})
        if "G" in comps:
            llama_cos = comps["G"]["cos_with_bk_dir"]
            gemma_cos = GEMMA_V8_REF["g_prompt_cos"].get(paradigm)
            gemma_str = f"{gemma_cos:+.3f}" if gemma_cos is not None else "N/A"
            log(f"  {paradigm:>10s}  {gemma_str:>10s}  {llama_cos:>+10.3f}")

    # --- Analysis 3: Autonomy Paradox ---
    log("\n--- Analysis 3: Autonomy Paradox (Variable higher BK rate, lower BK-proj?) ---")
    a3 = all_results.get("analysis3_autonomy_paradox", {})
    for paradigm in ["SM", "IC", "MW"]:
        if paradigm not in a3:
            continue
        entry = a3[paradigm]
        paradox = entry["autonomy_paradox_detected"]
        log(f"  {paradigm}: BK_rate diff (V-F)={entry['bk_rate_diff_var_minus_fix']:+.3f}, "
            f"BK_proj diff (V-F)={entry['bk_proj_diff_var_minus_fix']:+.3f} "
            f"{'** PARADOX **' if paradox else '(no paradox)'}")

    # --- Analysis 4: Prompt hierarchy ---
    log("\n--- Analysis 4: Prompt Component Hierarchy (BK rate effect) ---")
    a4 = all_results.get("analysis4_prompt_hierarchy", {})
    for paradigm in ["SM", "MW"]:
        if paradigm not in a4:
            continue
        hierarchy = a4[paradigm]["hierarchy_string_bk"]
        log(f"  {paradigm} LLaMA: {hierarchy}")
    log(f"  SM Gemma v6: {GEMMA_V8_REF['prompt_hierarchy_sm']}")


# ---------------------------------------------------------------------------
# JSON serializer for numpy types
# ---------------------------------------------------------------------------
def numpy_converter(obj):
    """Convert numpy types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log("=" * 72)
    log("V13 LLaMA RQ3: Condition-Dependent Modulation Analysis")
    log(f"Timestamp: {datetime.now().isoformat()}")
    log("=" * 72)

    # Verify all data files exist
    for paradigm, path in PARADIGM_PATHS.items():
        if not path.exists():
            log(f"ERROR: {paradigm} data not found at {path}")
            return
        log(f"  {paradigm}: {path} (OK)")

    all_results = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "version": "v13_llama_rq3",
            "model": "LLaMA-3.1-8B",
            "layer": 22,
            "layers_available": [8, 12, 22, 25, 30],
            "hidden_dim": 4096,
            "n_permutations": N_PERMUTATIONS,
            "cv_folds": CV_FOLDS,
            "random_seed": RANDOM_SEED,
            "paradigm_sizes": {
                "SM": 3200,
                "IC": 1600,
                "MW": 3200,
            },
            "gemma_v8_reference": GEMMA_V8_REF,
        }
    }

    # Run all analyses
    all_results["analysis1_within_bettype"] = analysis1_within_bettype_classification()
    all_results["analysis2_g_prompt_alignment"] = analysis2_g_prompt_alignment()
    all_results["analysis3_autonomy_paradox"] = analysis3_autonomy_paradox()
    all_results["analysis4_prompt_hierarchy"] = analysis4_prompt_hierarchy()
    all_results["analysis5_ic_prompt_conditions"] = analysis5_ic_prompt_conditions()

    # Print comparison summary
    print_comparison_summary(all_results)

    # Save results
    out_path = OUT_DIR / "v13_llama_rq3_analysis.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=numpy_converter)
    log(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
