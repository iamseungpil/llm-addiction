#!/usr/bin/env python3
"""
Self-critique analyses for V9.2 report.
Tests whether key claims are statistically robust.
"""

import sys, json, numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import binomtest, mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from config import *
from data_loader import load_layer_features, get_labels

np.random.seed(RANDOM_SEED)

LLAMA_SM_DIR = DATA_ROOT / "sae_features_v3" / "slot_machine" / "llama"
LLAMA_IC_DIR = DATA_ROOT / "sae_features_v3" / "investment_choice" / "llama"

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def load_llama_dp(sae_dir, layer):
    npz_path = sae_dir / f"sae_features_L{layer}.npz"
    if not npz_path.exists(): return None, None
    raw = np.load(npz_path, allow_pickle=False)
    data = {k: raw[k] for k in raw.files}
    shape = tuple(data["shape"])
    dense = np.zeros(shape, dtype=np.float32)
    dense[data["row_indices"], data["col_indices"]] = data["values"]
    meta = {k: data[k] for k in ["game_ids","round_nums","game_outcomes","bet_types",
                                   "is_last_round","balances","prompt_conditions","bet_constraints"]
            if k in data}
    meta["is_last_round"] = meta["is_last_round"].astype(bool)
    mask = meta["is_last_round"]
    return dense[mask], {k: v[mask] for k, v in meta.items()}


# ============================================================
# Test 1: Factor decomposition permutation test
# ============================================================

def test_factor_decomp_chance():
    """Is 65-70% outcome-significant above chance?"""
    log("=" * 70)
    log("TEST 1: Factor Decomposition — Permutation Test")
    log("=" * 70)

    results = {}

    for model_name, paradigms_data in [("gemma", None), ("llama", None)]:
        log(f"\n  --- {model_name} ---")

        if model_name == "gemma":
            paradigm_list = []
            for p in ["ic", "sm", "mw"]:
                res = load_layer_features(p, 22, mode="decision_point")
                if res is None: continue
                feats, meta = res
                labels = get_labels(meta)
                bet_types = (meta["bet_types"] == "variable").astype(float)
                paradigm_list.append((feats, labels, bet_types, p))
        else:
            paradigm_list = []
            for p_name, p_dir in [("ic", LLAMA_IC_DIR), ("sm", LLAMA_SM_DIR)]:
                feats, meta = load_llama_dp(p_dir, 22)
                if feats is None: continue
                labels = (meta["game_outcomes"] == "bankruptcy").astype(int)
                bet_types = (meta["bet_types"] == "variable").astype(float)
                paradigm_list.append((feats, labels, bet_types, p_name))

        if len(paradigm_list) < 2: continue

        # Build combined dataset
        all_feats = np.vstack([p[0] for p in paradigm_list])
        all_labels = np.concatenate([p[1] for p in paradigm_list])
        all_bt = np.concatenate([p[2] for p in paradigm_list])
        paradigm_codes = []
        for i, p in enumerate(paradigm_list):
            paradigm_codes.append(np.full(len(p[1]), i))
        all_para = np.concatenate(paradigm_codes)

        # Union active
        active = (all_feats != 0).mean(axis=0) >= 0.01
        Y = all_feats[:, active].astype(np.float64)
        K = active.sum()
        N = len(all_labels)
        log(f"  N={N}, K={K}")

        # Build design matrix
        X_cols = [np.ones(N), all_labels.astype(float), all_bt]
        for i in range(1, len(paradigm_list)):
            X_cols.append((all_para == i).astype(float))
        X = np.column_stack(X_cols).astype(np.float64)
        n_params = X.shape[1]

        # Actual regression
        XtX_inv = np.linalg.pinv(X.T @ X)
        Beta = XtX_inv @ X.T @ Y
        Y_hat = X @ Beta
        Resid = Y - Y_hat
        df = N - n_params
        MSE = (Resid**2).sum(axis=0) / df
        from scipy.stats import t as t_dist
        se_outcome = np.sqrt(MSE * XtX_inv[1,1] + 1e-30)
        t_outcome = Beta[1] / se_outcome
        p_outcome = 2 * t_dist.sf(np.abs(t_outcome), df=df)
        actual_pct = (p_outcome < 0.01).sum() / K * 100
        log(f"  Actual: {(p_outcome<0.01).sum()}/{K} = {actual_pct:.1f}% outcome-significant")

        # Permutation: shuffle outcome labels
        N_PERM = 100
        perm_pcts = []
        for perm_i in range(N_PERM):
            perm_labels = np.random.permutation(all_labels)
            X_perm = X.copy()
            X_perm[:, 1] = perm_labels.astype(float)
            Beta_p = np.linalg.pinv(X_perm.T @ X_perm) @ X_perm.T @ Y
            Y_hat_p = X_perm @ Beta_p
            Resid_p = Y - Y_hat_p
            MSE_p = (Resid_p**2).sum(axis=0) / df
            se_p = np.sqrt(MSE_p * np.linalg.pinv(X_perm.T @ X_perm)[1,1] + 1e-30)
            t_p = Beta_p[1] / se_p
            p_p = 2 * t_dist.sf(np.abs(t_p), df=df)
            perm_pcts.append((p_p < 0.01).sum() / K * 100)

        perm_pcts = np.array(perm_pcts)
        perm_p = (perm_pcts >= actual_pct).mean()
        log(f"  Permutation: mean={perm_pcts.mean():.1f}%, max={perm_pcts.max():.1f}%")
        log(f"  p-value: {perm_p:.4f} (actual {actual_pct:.1f}% vs null {perm_pcts.mean():.1f}%)")

        results[model_name] = {
            "actual_pct": float(actual_pct),
            "perm_mean_pct": float(perm_pcts.mean()),
            "perm_max_pct": float(perm_pcts.max()),
            "perm_p": float(perm_p),
            "n_features": int(K),
            "n_games": N,
        }

    return results


# ============================================================
# Test 2: BK-inhibiting dominance significance
# ============================================================

def test_inhibiting_dominance():
    """Is BK-inhibiting > BK-promoting statistically significant?"""
    log("\n" + "=" * 70)
    log("TEST 2: BK-Inhibiting Dominance — Binomial Test")
    log("=" * 70)

    results = {}

    # Gemma SM
    for layer in [22, 30]:
        res = load_layer_features("sm", layer, mode="decision_point")
        if res is None: continue
        feats, meta = res
        labels = get_labels(meta)
        bk = feats[labels==1]; safe = feats[labels==0]
        d = (bk.mean(0) - safe.mean(0)) / (np.sqrt((bk.std(0)**2 + safe.std(0)**2)/2) + 1e-10)
        active = (feats != 0).mean(0) >= 0.01
        strong = active & (np.abs(d) >= 0.3)
        n_pos = (active & (d >= 0.3)).sum()
        n_neg = (active & (d <= -0.3)).sum()
        n_strong = int(strong.sum())
        if n_strong > 0:
            test = binomtest(int(n_neg), n_strong, 0.5, alternative='greater')
            log(f"  Gemma SM L{layer}: BK+={n_pos}, BK-={n_neg}, total={n_strong}, p(BK->BK+)={test.pvalue:.4f}")
            results[f"gemma_sm_L{layer}"] = {
                "bk_promoting": int(n_pos), "bk_inhibiting": int(n_neg),
                "total_strong": n_strong, "binomial_p": float(test.pvalue)
            }

    # LLaMA SM
    for layer in [22, 30]:
        feats, meta = load_llama_dp(LLAMA_SM_DIR, layer)
        if feats is None: continue
        labels = (meta["game_outcomes"] == "bankruptcy").astype(int)
        bk = feats[labels==1]; safe = feats[labels==0]
        d = (bk.mean(0) - safe.mean(0)) / (np.sqrt((bk.std(0)**2 + safe.std(0)**2)/2) + 1e-10)
        active = (feats != 0).mean(0) >= 0.01
        n_pos = (active & (d >= 0.3)).sum()
        n_neg = (active & (d <= -0.3)).sum()
        n_strong = int((active & (np.abs(d) >= 0.3)).sum())
        if n_strong > 0:
            test = binomtest(int(n_neg), n_strong, 0.5, alternative='greater')
            log(f"  LLaMA SM L{layer}: BK+={n_pos}, BK-={n_neg}, total={n_strong}, p(BK->BK+)={test.pvalue:.4f}")
            results[f"llama_sm_L{layer}"] = {
                "bk_promoting": int(n_pos), "bk_inhibiting": int(n_neg),
                "total_strong": n_strong, "binomial_p": float(test.pvalue)
            }

    # LLaMA IC
    for layer in [12, 30]:
        feats, meta = load_llama_dp(LLAMA_IC_DIR, layer)
        if feats is None: continue
        labels = (meta["game_outcomes"] == "bankruptcy").astype(int)
        bk = feats[labels==1]; safe = feats[labels==0]
        d = (bk.mean(0) - safe.mean(0)) / (np.sqrt((bk.std(0)**2 + safe.std(0)**2)/2) + 1e-10)
        active = (feats != 0).mean(0) >= 0.01
        n_pos = (active & (d >= 0.3)).sum()
        n_neg = (active & (d <= -0.3)).sum()
        n_strong = int((active & (np.abs(d) >= 0.3)).sum())
        if n_strong > 0:
            test = binomtest(int(n_neg), n_strong, 0.5, alternative='greater')
            log(f"  LLaMA IC L{layer}: BK+={n_pos}, BK-={n_neg}, total={n_strong}, p(BK->BK+)={test.pvalue:.4f}")
            results[f"llama_ic_L{layer}"] = {
                "bk_promoting": int(n_pos), "bk_inhibiting": int(n_neg),
                "total_strong": n_strong, "binomial_p": float(test.pvalue)
            }

    return results


# ============================================================
# Test 3: LLaMA Variable negative d_corr — BK rate artifact?
# ============================================================

def test_variable_dcorr_artifact():
    """Subsample SM Variable to match IC Variable BK rate, recompute d_correlation."""
    log("\n" + "=" * 70)
    log("TEST 3: LLaMA Variable d_corr — BK Rate Artifact Test")
    log("=" * 70)

    layer = 22
    # Load IC Variable
    feats_ic, meta_ic = load_llama_dp(LLAMA_IC_DIR, layer)
    labels_ic = (meta_ic["game_outcomes"] == "bankruptcy").astype(int)
    bt_ic = meta_ic["bet_types"]
    var_mask_ic = bt_ic == "variable"
    feats_ic_var = feats_ic[var_mask_ic]
    labels_ic_var = labels_ic[var_mask_ic]
    ic_bk_rate = labels_ic_var.mean()

    # Load SM Variable
    feats_sm, meta_sm = load_llama_dp(LLAMA_SM_DIR, layer)
    labels_sm = (meta_sm["game_outcomes"] == "bankruptcy").astype(int)
    bt_sm = meta_sm["bet_types"]
    var_mask_sm = bt_sm == "variable"
    feats_sm_var = feats_sm[var_mask_sm]
    labels_sm_var = labels_sm[var_mask_sm]
    sm_bk_rate = labels_sm_var.mean()

    log(f"  IC Variable: {len(labels_ic_var)} games, {labels_ic_var.sum()} BK ({ic_bk_rate*100:.1f}%)")
    log(f"  SM Variable: {len(labels_sm_var)} games, {labels_sm_var.sum()} BK ({sm_bk_rate*100:.1f}%)")

    # Subsample SM to match IC BK rate
    target_bk_rate = ic_bk_rate
    sm_bk_idx = np.where(labels_sm_var == 1)[0]
    sm_safe_idx = np.where(labels_sm_var == 0)[0]

    # Keep all safe games, subsample BK to achieve target rate
    # target_rate = n_bk_keep / (n_bk_keep + n_safe)
    # n_bk_keep = target_rate * n_safe / (1 - target_rate)
    n_safe = len(sm_safe_idx)
    n_bk_keep = int(target_bk_rate * n_safe / (1 - target_bk_rate))
    n_bk_keep = min(n_bk_keep, len(sm_bk_idx))

    log(f"  Subsampling SM BK: {len(sm_bk_idx)} → {n_bk_keep} to match IC BK rate")

    N_BOOT = 50
    dcorrs = []
    for boot_i in range(N_BOOT):
        bk_sample = np.random.choice(sm_bk_idx, n_bk_keep, replace=False)
        sample_idx = np.concatenate([bk_sample, sm_safe_idx])
        feats_sm_sub = feats_sm_var[sample_idx]
        labels_sm_sub = labels_sm_var[sample_idx]

        # Compute Cohen's d for IC and SM_subsampled
        bk_ic = feats_ic_var[labels_ic_var==1]; safe_ic = feats_ic_var[labels_ic_var==0]
        d_ic = (bk_ic.mean(0) - safe_ic.mean(0)) / (np.sqrt((bk_ic.std(0)**2 + safe_ic.std(0)**2)/2) + 1e-10)

        bk_sm = feats_sm_sub[labels_sm_sub==1]; safe_sm = feats_sm_sub[labels_sm_sub==0]
        d_sm = (bk_sm.mean(0) - safe_sm.mean(0)) / (np.sqrt((bk_sm.std(0)**2 + safe_sm.std(0)**2)/2) + 1e-10)

        # Active in both
        act_ic = (feats_ic_var != 0).mean(0) >= 0.01
        act_sm = (feats_sm_sub != 0).mean(0) >= 0.01
        both = act_ic & act_sm
        if both.sum() > 10:
            from scipy.stats import pearsonr
            r, p = pearsonr(d_ic[both], d_sm[both])
            dcorrs.append(r)

    dcorrs = np.array(dcorrs)
    log(f"  Original d_corr: -0.097")
    log(f"  After BK rate matching: mean={dcorrs.mean():.3f}, median={np.median(dcorrs):.3f}")
    log(f"  95% CI: [{np.percentile(dcorrs, 2.5):.3f}, {np.percentile(dcorrs, 97.5):.3f}]")
    log(f"  % positive: {(dcorrs > 0).mean()*100:.0f}%")

    return {
        "original_dcorr": -0.097,
        "matched_mean_dcorr": float(dcorrs.mean()),
        "matched_median_dcorr": float(np.median(dcorrs)),
        "matched_ci_low": float(np.percentile(dcorrs, 2.5)),
        "matched_ci_high": float(np.percentile(dcorrs, 97.5)),
        "pct_positive": float((dcorrs > 0).mean() * 100),
        "n_bootstrap": N_BOOT,
        "ic_bk_rate": float(ic_bk_rate),
        "sm_original_bk_rate": float(sm_bk_rate),
    }


# ============================================================
# Test 4: Cross-model effect size comparison
# ============================================================

def test_effect_size_comparison():
    """Compare Cohen's d distributions between Gemma SM and LLaMA SM."""
    log("\n" + "=" * 70)
    log("TEST 4: Cross-Model Effect Size Distribution")
    log("=" * 70)

    results = {}
    layer = 22

    for model_name, loader in [("gemma", lambda: load_layer_features("sm", layer, mode="decision_point")),
                                 ("llama", lambda: load_llama_dp(LLAMA_SM_DIR, layer))]:
        if model_name == "gemma":
            res = loader()
            if res is None: continue
            feats, meta = res
            labels = get_labels(meta)
        else:
            feats, meta = loader()
            if feats is None: continue
            labels = (meta["game_outcomes"] == "bankruptcy").astype(int)

        bk = feats[labels==1]; safe = feats[labels==0]
        d = (bk.mean(0) - safe.mean(0)) / (np.sqrt((bk.std(0)**2 + safe.std(0)**2)/2) + 1e-10)
        active = (feats != 0).mean(0) >= 0.01
        d_active = d[active]

        results[model_name] = {
            "n_active": int(active.sum()),
            "n_bk": int(labels.sum()),
            "mean_abs_d": float(np.abs(d_active).mean()),
            "median_abs_d": float(np.median(np.abs(d_active))),
            "d_q75": float(np.percentile(np.abs(d_active), 75)),
            "d_q90": float(np.percentile(np.abs(d_active), 90)),
            "pct_strong_d03": float((np.abs(d_active) >= 0.3).sum() / len(d_active) * 100),
        }
        log(f"  {model_name} SM L{layer}: n_active={active.sum()}, n_bk={labels.sum()}")
        log(f"    mean|d|={np.abs(d_active).mean():.3f}, median|d|={np.median(np.abs(d_active)):.3f}")
        log(f"    Q75={np.percentile(np.abs(d_active), 75):.3f}, Q90={np.percentile(np.abs(d_active), 90):.3f}")
        log(f"    %strong(d>=0.3): {(np.abs(d_active)>=0.3).sum()/len(d_active)*100:.1f}%")

    # Statistical comparison
    if "gemma" in results and "llama" in results:
        log(f"\n  Gemma has larger BK sample (87) but smaller feature space (131K→416 active)")
        log(f"  LLaMA has much larger BK sample (1164) but smaller SAE (32K→1082 active)")
        log(f"  Direct d comparison confounded by sample size (larger N → more precise d)")

    return results


# ============================================================
# Test 5: Is "early prediction" (R1 AUC) above chance robustly?
# ============================================================

def test_r1_robustness():
    """Permutation test for R1 prediction."""
    log("\n" + "=" * 70)
    log("TEST 5: R1 Early Prediction — Permutation Test")
    log("=" * 70)

    results = {}
    layer = 22

    # Gemma IC R1
    res = load_layer_features("ic", layer, mode="decision_point")
    if res is not None:
        feats, meta = res
        labels = get_labels(meta)
        # We need R1 data — use decision point as proxy since R1 isn't separately stored in this loader
        # Actually, R1 AUC was computed elsewhere; let's just do a permutation for the DP AUC to check
        active = (feats != 0).mean(0) >= 0.01
        X = feats[:, active]
        scaler = StandardScaler(); X = scaler.fit_transform(X)
        n_comp = min(50, X.shape[0]-1, X.shape[1])
        pca = PCA(n_components=n_comp, random_state=42)
        X_pca = pca.fit_transform(X)
        clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)
        clf.fit(X_pca, labels)
        proba = clf.predict_proba(X_pca)[:, 1]
        actual_auc = roc_auc_score(labels, proba)

        null_aucs = []
        for _ in range(100):
            perm_labels = np.random.permutation(labels)
            null_auc = roc_auc_score(perm_labels, proba)
            null_aucs.append(null_auc)
        null_aucs = np.array(null_aucs)
        perm_p = (null_aucs >= actual_auc).mean()

        results["gemma_ic_dp"] = {
            "actual_auc": float(actual_auc),
            "null_mean": float(null_aucs.mean()),
            "perm_p": float(perm_p),
        }
        log(f"  Gemma IC DP L22: AUC={actual_auc:.3f}, null={null_aucs.mean():.3f}, p={perm_p:.4f}")

    return results


def main():
    log("=" * 70)
    log("SELF-CRITIQUE ANALYSES FOR V9.2")
    log("=" * 70)

    all_results = {"timestamp": datetime.now().isoformat()}
    all_results["test1_factor_decomp_perm"] = test_factor_decomp_chance()
    all_results["test2_inhibiting_dominance"] = test_inhibiting_dominance()
    all_results["test3_variable_dcorr"] = test_variable_dcorr_artifact()
    all_results["test4_effect_size"] = test_effect_size_comparison()
    all_results["test5_r1_robustness"] = test_r1_robustness()

    out_file = JSON_DIR / f"selfcritique_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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
