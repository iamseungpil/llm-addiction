#!/usr/bin/env python3
"""
LLaMA symmetric analyses using newly extracted hidden states.
Analysis 2 equivalent: Variable/Fixed BK direction cosine (IC)
Analysis 3 equivalent: Interaction regression (IC, SM)
+ Balance partial correlation (IC)
"""
import sys, json, numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import ttest_ind, pointbiserialr
import warnings; warnings.filterwarnings('ignore')
np.random.seed(42)

IC_HS = Path("/home/v-seungplee/data/llm-addiction/sae_features_v3/investment_choice/llama/hidden_states_dp.npz")
SM_HS = Path("/home/v-seungplee/data/llm-addiction/sae_features_v3/slot_machine/llama/hidden_states_dp.npz")
OUT = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis/results/json")

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def load_hs(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}

# ============================================================
# Analysis 2: BK-Only Direction Comparison (LLaMA IC)
# ============================================================
def analysis2_bk_direction():
    log("=" * 70)
    log("ANALYSIS 2 SYMMETRIC: LLaMA IC Variable/Fixed BK Direction")
    log("=" * 70)
    data = load_hs(IC_HS)
    hs = data["hidden_states"]  # (1600, 5, 4096)
    layers = data["layers"]
    labels = (data["game_outcomes"] == "bankruptcy").astype(int)
    bt = data["bet_types"]
    results = {}

    for j, layer in enumerate(layers):
        hidden = hs[:, j, :]
        var_bk = (labels == 1) & (bt == "variable")
        fix_bk = (labels == 1) & (bt == "fixed")
        var_safe = (labels == 0) & (bt == "variable")
        fix_safe = (labels == 0) & (bt == "fixed")

        n_vb, n_fb = var_bk.sum(), fix_bk.sum()
        if n_vb < 5 or n_fb < 5:
            log(f"  L{layer}: skipped (var_bk={n_vb}, fix_bk={n_fb})")
            continue

        var_bk_dir = hidden[var_bk].mean(0) - hidden[var_safe].mean(0)
        fix_bk_dir = hidden[fix_bk].mean(0) - hidden[fix_safe].mean(0)
        cos = np.dot(var_bk_dir, fix_bk_dir) / (np.linalg.norm(var_bk_dir) * np.linalg.norm(fix_bk_dir) + 1e-10)

        # Common BK neurons (sig in both Variable and Fixed, same sign)
        n_neurons = hidden.shape[1]
        var_sig = set(); fix_sig = set()
        for ni in range(n_neurons):
            if var_bk.sum() >= 3 and var_safe.sum() >= 3:
                _, p = ttest_ind(hidden[var_bk, ni], hidden[var_safe, ni])
                if p < 0.05: var_sig.add(ni)
            if fix_bk.sum() >= 3 and fix_safe.sum() >= 3:
                _, p = ttest_ind(hidden[fix_bk, ni], hidden[fix_safe, ni])
                if p < 0.05: fix_sig.add(ni)

        common = var_sig & fix_sig
        sign_consistent = 0
        for ni in common:
            v_d = hidden[var_bk, ni].mean() - hidden[var_safe, ni].mean()
            f_d = hidden[fix_bk, ni].mean() - hidden[fix_safe, ni].mean()
            if np.sign(v_d) == np.sign(f_d):
                sign_consistent += 1

        results[f"L{layer}"] = {
            "n_var_bk": int(n_vb), "n_fix_bk": int(n_fb),
            "cos_bk_direction": float(cos),
            "var_sig": len(var_sig), "fix_sig": len(fix_sig),
            "common": len(common), "sign_consistent_common": sign_consistent,
        }
        log(f"  L{layer}: var_bk={n_vb}, fix_bk={n_fb}, cos={cos:.3f}, common_sc={sign_consistent}")

    return results


# ============================================================
# Analysis 3: Interaction Regression (LLaMA IC and SM)
# ============================================================
def analysis3_interaction():
    log("\n" + "=" * 70)
    log("ANALYSIS 3 SYMMETRIC: LLaMA Interaction Regression")
    log("=" * 70)
    import statsmodels.api as sm

    results = {}

    for paradigm, hs_path in [("ic", IC_HS), ("sm", SM_HS)]:
        data = load_hs(hs_path)
        hs = data["hidden_states"]
        layers = data["layers"]
        labels = (data["game_outcomes"] == "bankruptcy").astype(int)
        bt = (data["bet_types"] == "variable").astype(float)

        n_bk = labels.sum()
        if n_bk < 10:
            continue

        # Use L22 (index 2 in [8,12,22,25,30])
        layer_idx = 2  # L22
        layer = int(layers[layer_idx])
        hidden = hs[:, layer_idx, :]
        n_neurons = hidden.shape[1]

        log(f"\n  {paradigm} L{layer}: {len(labels)} games, {n_bk} BK")

        outcome_sig = 0; bettype_sig = 0; interaction_sig = 0
        shared_bk = 0

        for ni in range(n_neurons):
            y = hidden[:, ni]
            X = np.column_stack([labels, bt, labels * bt])
            X = sm.add_constant(X)
            try:
                model = sm.OLS(y, X).fit()
                p_out = model.pvalues[1]
                p_bt = model.pvalues[2]
                p_int = model.pvalues[3]
                if p_out < 0.01: outcome_sig += 1
                if p_bt < 0.01: bettype_sig += 1
                if p_int < 0.01: interaction_sig += 1
                if p_out < 0.01 and p_int > 0.05: shared_bk += 1
            except:
                continue

        results[paradigm] = {
            "layer": layer, "n_neurons": n_neurons, "n_games": len(labels), "n_bk": int(n_bk),
            "outcome_sig_p01": outcome_sig,
            "bettype_sig_p01": bettype_sig,
            "interaction_sig_p01": interaction_sig,
            "shared_bk_neurons": shared_bk,
        }
        log(f"    outcome_sig={outcome_sig} ({outcome_sig/n_neurons*100:.1f}%)")
        log(f"    bettype_sig={bettype_sig} ({bettype_sig/n_neurons*100:.1f}%)")
        log(f"    interaction_sig={interaction_sig} ({interaction_sig/n_neurons*100:.1f}%)")
        log(f"    shared_bk={shared_bk} ({shared_bk/n_neurons*100:.1f}%)")

    return results


# ============================================================
# Partial Correlation: Balance confound (LLaMA IC)
# ============================================================
def partial_correlation():
    log("\n" + "=" * 70)
    log("BALANCE PARTIAL CORRELATION: LLaMA IC")
    log("=" * 70)

    data = load_hs(IC_HS)
    hs = data["hidden_states"]
    layers = data["layers"]
    labels = (data["game_outcomes"] == "bankruptcy").astype(int)
    balances = data["balances"].astype(float)

    results = {}
    layer_idx = 2  # L22
    layer = int(layers[layer_idx])
    hidden = hs[:, layer_idx, :]
    n_neurons = hidden.shape[1]

    raw_sig = 0; partial_sig = 0
    for ni in range(n_neurons):
        y = hidden[:, ni]
        r_raw, p_raw = pointbiserialr(labels, y)
        if p_raw < 0.01: raw_sig += 1
        beta = np.polyfit(balances, y, 1)
        residual = y - np.polyval(beta, balances)
        r_part, p_part = pointbiserialr(labels, residual)
        if p_part < 0.01: partial_sig += 1

    results["ic_L22"] = {
        "n_neurons": n_neurons,
        "raw_bk_sig": raw_sig,
        "partial_bk_sig": partial_sig,
        "retained_pct": float(partial_sig / max(raw_sig, 1) * 100),
    }
    log(f"  L{layer}: raw_sig={raw_sig}, partial={partial_sig} ({partial_sig/max(raw_sig,1)*100:.1f}% retained)")

    return results


def main():
    log("=" * 70)
    log("LLaMA HIDDEN STATE ANALYSES (Symmetric to Gemma Analysis 2, 3)")
    log("=" * 70)

    all_results = {"timestamp": datetime.now().isoformat(), "layers": [8, 12, 22, 25, 30]}
    all_results["analysis2_bk_direction"] = analysis2_bk_direction()
    all_results["analysis3_interaction"] = analysis3_interaction()
    all_results["partial_correlation"] = partial_correlation()

    out_file = OUT / f"llama_hidden_analyses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, 'w') as f:
        def convert(obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return obj
        json.dump(all_results, f, indent=2, default=convert)
    log(f"\nSaved to {out_file}")

if __name__ == "__main__":
    main()
