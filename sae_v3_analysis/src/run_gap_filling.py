#!/usr/bin/env python3
"""
Gap-filling analyses for V9 report.
Task 8: Llama IC→SM cross-domain transfer
Task 9: Llama cross-domain SAE feature consistency
Task 10: G/M/W/P/R prompt component activation analysis (Gemma SM)
"""

import sys, json, numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats
from scipy.stats import ttest_ind, pointbiserialr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from config import *
from data_loader import load_layer_features, load_hidden_states, get_labels, filter_active_features

np.random.seed(RANDOM_SEED)

LLAMA_SM_DIR = DATA_ROOT / "sae_features_v3" / "slot_machine" / "llama"
LLAMA_IC_DIR = DATA_ROOT / "sae_features_v3" / "investment_choice" / "llama"

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def load_llama_dp(sae_dir, layer):
    """Load Llama SAE features at decision point."""
    npz_path = sae_dir / f"sae_features_L{layer}.npz"
    if not npz_path.exists():
        return None, None
    raw = np.load(npz_path, allow_pickle=False)
    data = {k: raw[k] for k in raw.files}
    shape = tuple(data["shape"])
    dense = np.zeros(shape, dtype=np.float32)
    dense[data["row_indices"], data["col_indices"]] = data["values"]
    meta = {}
    for k in ["game_ids", "round_nums", "game_outcomes", "bet_types",
              "is_last_round", "balances", "prompt_conditions", "bet_constraints"]:
        if k in data:
            meta[k] = data[k]
    meta["is_last_round"] = meta["is_last_round"].astype(bool)
    mask = meta["is_last_round"]
    return dense[mask], {k: v[mask] for k, v in meta.items()}

def load_llama_all_rounds(sae_dir, layer):
    """Load all rounds for Llama."""
    npz_path = sae_dir / f"sae_features_L{layer}.npz"
    if not npz_path.exists():
        return None, None
    raw = np.load(npz_path, allow_pickle=False)
    data = {k: raw[k] for k in raw.files}
    shape = tuple(data["shape"])
    dense = np.zeros(shape, dtype=np.float32)
    dense[data["row_indices"], data["col_indices"]] = data["values"]
    meta = {}
    for k in ["game_ids", "round_nums", "game_outcomes", "bet_types",
              "is_last_round", "balances", "prompt_conditions", "bet_constraints"]:
        if k in data:
            meta[k] = data[k]
    meta["is_last_round"] = meta["is_last_round"].astype(bool)
    return dense, meta


# ============================================================
# Task 8: Llama IC→SM Cross-Domain Transfer
# ============================================================

def task8_llama_cross_domain_transfer():
    log("=" * 70)
    log("TASK 8: Llama IC↔SM Cross-Domain Transfer")
    log("=" * 70)

    results = {}

    for layer in [5, 8, 10, 12, 15, 20, 25, 30]:
        # Load IC
        ic_feats, ic_meta = load_llama_dp(LLAMA_IC_DIR, layer)
        if ic_feats is None:
            continue
        ic_labels = (ic_meta["game_outcomes"] == "bankruptcy").astype(int)

        # Load SM
        sm_feats, sm_meta = load_llama_dp(LLAMA_SM_DIR, layer)
        if sm_feats is None:
            continue
        sm_labels = (sm_meta["game_outcomes"] == "bankruptcy").astype(int)

        if ic_labels.sum() < 5 or sm_labels.sum() < 5:
            continue

        # Both use 32K features, same SAE — direct comparison possible
        for train_name, X_train, y_train, test_name, X_test, y_test in [
            ("IC", ic_feats, ic_labels, "SM", sm_feats, sm_labels),
            ("SM", sm_feats, sm_labels, "IC", ic_feats, ic_labels),
        ]:
            # Filter active in train
            active = (X_train != 0).mean(axis=0) >= 0.01
            Xtr = X_train[:, active]
            Xte = X_test[:, active]

            if Xtr.shape[1] < 10:
                continue

            scaler = StandardScaler()
            Xtr_s = scaler.fit_transform(Xtr)
            Xte_s = scaler.transform(Xte)

            n_comp = min(50, Xtr_s.shape[0] - 1, Xtr_s.shape[1])
            pca = PCA(n_components=n_comp)
            Xtr_pca = pca.fit_transform(Xtr_s)
            Xte_pca = pca.transform(Xte_s)

            clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
            clf.fit(Xtr_pca, y_train)

            y_prob = clf.predict_proba(Xte_pca)[:, 1]
            try:
                auc = roc_auc_score(y_test, y_prob)
            except:
                auc = 0.5

            # Permutation test
            n_perm = 200
            null_aucs = []
            for _ in range(n_perm):
                y_perm = np.random.permutation(y_test)
                try:
                    null_aucs.append(roc_auc_score(y_perm, y_prob))
                except:
                    null_aucs.append(0.5)
            null_aucs = np.array(null_aucs)
            p_perm = (null_aucs >= auc).mean()

            key = f"L{layer}_{train_name}_{test_name}"
            results[key] = {
                "layer": layer,
                "train": train_name,
                "test": test_name,
                "auc": float(auc),
                "perm_p": float(p_perm),
                "null_mean": float(null_aucs.mean()),
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "bk_train": int(y_train.sum()),
                "bk_test": int(y_test.sum()),
            }

            log(f"  L{layer} {train_name}→{test_name}: AUC={auc:.3f}, perm_p={p_perm:.3f} (n_train={len(y_train)}, n_test={len(y_test)})")

    # Also: R1 transfer (confound-free)
    log("\n  --- R1 Transfer (balance-controlled) ---")
    for layer in [10, 22, 30]:
        if layer >= 32:
            continue

        ic_dense, ic_meta_all = load_llama_all_rounds(LLAMA_IC_DIR, layer)
        sm_dense, sm_meta_all = load_llama_all_rounds(LLAMA_SM_DIR, layer)

        if ic_dense is None or sm_dense is None:
            continue

        ic_r1 = ic_meta_all["round_nums"] == 1
        sm_r1 = sm_meta_all["round_nums"] == 1

        if ic_r1.sum() < 20 or sm_r1.sum() < 20:
            continue

        ic_feats_r1 = ic_dense[ic_r1]
        ic_labels_r1 = (ic_meta_all["game_outcomes"][ic_r1] == "bankruptcy").astype(int)
        sm_feats_r1 = sm_dense[sm_r1]
        sm_labels_r1 = (sm_meta_all["game_outcomes"][sm_r1] == "bankruptcy").astype(int)

        if ic_labels_r1.sum() < 5 or sm_labels_r1.sum() < 5:
            continue

        # IC→SM R1 transfer
        active = (ic_feats_r1 != 0).mean(axis=0) >= 0.01
        Xtr = ic_feats_r1[:, active]
        Xte = sm_feats_r1[:, active]

        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)
        n_comp = min(50, Xtr_s.shape[0] - 1, Xtr_s.shape[1])
        pca = PCA(n_components=n_comp)
        Xtr_pca = pca.fit_transform(Xtr_s)
        Xte_pca = pca.transform(Xte_s)

        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
        clf.fit(Xtr_pca, ic_labels_r1)
        y_prob = clf.predict_proba(Xte_pca)[:, 1]
        try:
            auc = roc_auc_score(sm_labels_r1, y_prob)
        except:
            auc = 0.5

        log(f"  L{layer} IC→SM R1: AUC={auc:.3f} (IC BK={ic_labels_r1.sum()}, SM BK={sm_labels_r1.sum()})")

    return results


# ============================================================
# Task 9: Llama Cross-Domain SAE Feature Consistency (IC vs SM)
# ============================================================

def task9_llama_sae_crossdomain():
    log("\n" + "=" * 70)
    log("TASK 9: Llama IC vs SM SAE Feature Consistency")
    log("=" * 70)

    results = {}

    for layer in range(0, 32, 2):
        ic_feats, ic_meta = load_llama_dp(LLAMA_IC_DIR, layer)
        sm_feats, sm_meta = load_llama_dp(LLAMA_SM_DIR, layer)

        if ic_feats is None or sm_feats is None:
            continue

        ic_labels = (ic_meta["game_outcomes"] == "bankruptcy").astype(int)
        sm_labels = (sm_meta["game_outcomes"] == "bankruptcy").astype(int)

        if ic_labels.sum() < 5 or sm_labels.sum() < 5:
            continue

        # Cohen's d per feature
        def cohens_d(feats, labels):
            bk = feats[labels == 1]
            safe = feats[labels == 0]
            pooled = np.sqrt((bk.std(0)**2 + safe.std(0)**2) / 2) + 1e-10
            return (bk.mean(0) - safe.mean(0)) / pooled

        d_ic = cohens_d(ic_feats, ic_labels)
        d_sm = cohens_d(sm_feats, sm_labels)

        # Active in both
        ic_active = (ic_feats != 0).mean(axis=0) >= 0.01
        sm_active = (sm_feats != 0).mean(axis=0) >= 0.01
        both_active = ic_active & sm_active

        # Sign consistent
        sign_pos = (d_ic > 0) & (d_sm > 0) & both_active
        sign_neg = (d_ic < 0) & (d_sm < 0) & both_active
        sign_consistent = sign_pos | sign_neg

        # Strong
        geo_mean = np.sqrt(np.abs(d_ic) * np.abs(d_sm))
        strong = sign_consistent & (geo_mean >= 0.2)
        medium = sign_consistent & (geo_mean >= 0.1)

        n_active = both_active.sum()
        n_sign = sign_consistent.sum()

        results[layer] = {
            "n_active": int(n_active),
            "n_sign_consistent": int(n_sign),
            "sign_pct": f"{n_sign/max(n_active,1)*100:.1f}%",
            "n_strong": int(strong.sum()),
            "n_medium": int(medium.sum()),
        }

        if layer % 6 == 0:
            log(f"  L{layer}: active={n_active}, sign_consistent={n_sign} ({n_sign/max(n_active,1)*100:.1f}%), strong={strong.sum()}")

    total_sign = sum(r["n_sign_consistent"] for r in results.values())
    total_strong = sum(r["n_strong"] for r in results.values())
    log(f"\n  SUMMARY: Llama IC-SM sign-consistent={total_sign}, strong={total_strong}")
    log(f"  (Compare: Gemma 3-paradigm = 744 sign-consistent, 665 strong)")

    return results


# ============================================================
# Task 10: Prompt Component Activation Analysis (Gemma SM)
# ============================================================

def task10_prompt_component_analysis():
    log("\n" + "=" * 70)
    log("TASK 10: Prompt Component Activation Effects (Gemma SM)")
    log("=" * 70)

    results = {}

    for layer in [10, 18, 22, 26, 30, 33]:
        # Load Gemma SM hidden states
        hs = load_hidden_states("sm", layer, mode="decision_point")
        if hs is None:
            log(f"  L{layer}: no hidden states")
            continue

        hidden, meta = hs
        labels = get_labels(meta)
        n_neurons = hidden.shape[1]

        if "prompt_conditions" not in meta:
            log(f"  L{layer}: no prompt_conditions in metadata")
            continue

        prompts = meta["prompt_conditions"]
        unique_prompts = np.unique(prompts)
        log(f"  L{layer}: {len(unique_prompts)} unique prompt conditions, {hidden.shape[0]} games")

        # Parse prompt components: G, M, R, W, P
        components = ["G", "M", "R", "W", "P"]
        component_results = {}

        for comp in components:
            has_comp = np.array([comp in str(p) for p in prompts])
            no_comp = ~has_comp

            n_with = has_comp.sum()
            n_without = no_comp.sum()

            if n_with < 20 or n_without < 20:
                continue

            # BK rate difference
            bk_with = labels[has_comp].mean()
            bk_without = labels[no_comp].mean()

            # Per-neuron: activation difference due to component
            # AND interaction: does component change the BK-neuron relationship?
            comp_sig = 0
            comp_x_bk_sig = 0
            comp_amplifies_bk = 0  # component makes BK-neuron correlation stronger

            import statsmodels.api as sm

            for ni in range(n_neurons):
                y = hidden[:, ni]
                x_bk = labels.astype(float)
                x_comp = has_comp.astype(float)

                X = np.column_stack([x_bk, x_comp, x_bk * x_comp])
                X = sm.add_constant(X)

                try:
                    model = sm.OLS(y, X).fit()
                    if model.pvalues[2] < 0.01:  # component main effect
                        comp_sig += 1
                    if model.pvalues[3] < 0.01:  # component × BK interaction
                        comp_x_bk_sig += 1
                        # Does component amplify BK effect?
                        if np.sign(model.params[1]) == np.sign(model.params[3]):
                            comp_amplifies_bk += 1
                except:
                    continue

            component_results[comp] = {
                "n_with": int(n_with),
                "n_without": int(n_without),
                "bk_rate_with": float(bk_with),
                "bk_rate_without": float(bk_without),
                "bk_rate_ratio": float(bk_with / max(bk_without, 0.001)),
                "comp_sig_neurons": comp_sig,
                "comp_x_bk_interaction_neurons": comp_x_bk_sig,
                "comp_amplifies_bk": comp_amplifies_bk,
            }

            log(f"    {comp}: BK {bk_with*100:.1f}% vs {bk_without*100:.1f}% (ratio {bk_with/max(bk_without,0.001):.1f}x) | comp_sig={comp_sig}, interaction={comp_x_bk_sig}, amplifies_bk={comp_amplifies_bk}")

        results[f"L{layer}"] = component_results

    return results


# ============================================================
# Main
# ============================================================

def main():
    log("=" * 70)
    log("V9 GAP-FILLING ANALYSES")
    log("=" * 70)

    all_results = {"timestamp": datetime.now().isoformat()}

    all_results["task8_llama_transfer"] = task8_llama_cross_domain_transfer()
    all_results["task9_llama_sae_consistency"] = task9_llama_sae_crossdomain()
    all_results["task10_prompt_components"] = task10_prompt_component_analysis()

    # Save
    out_file = JSON_DIR / f"gap_filling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2, default=convert)

    log(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
