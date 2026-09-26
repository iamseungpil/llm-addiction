#!/usr/bin/env python3
"""
V8-style hidden state analyses missing from V9.
1. Universal BK Neurons (Gemma, L22, 3-paradigm FDR-corrected)
2. Hidden state cross-domain transfer (compare with SAE transfer)
3. 3D Shared Subspace via LR weight PCA
4. G-prompt BK direction alignment
5. Bet constraint linear mapping
"""
import sys, json, numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats
from scipy.stats import pearsonr, pointbiserialr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests
import warnings; warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from config import *
from data_loader import load_hidden_states, load_layer_features, get_labels

np.random.seed(RANDOM_SEED)

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================
# 1. Universal BK Neurons (V8 §2.1 equivalent)
# ============================================================
def universal_bk_neurons():
    """Find neurons significant in all 3 paradigms with FDR correction."""
    log("=" * 70)
    log("1. UNIVERSAL BK NEURONS (Gemma L22, 3-paradigm, FDR-corrected)")
    log("=" * 70)

    layer = 22
    paradigm_corrs = {}

    for p in ["ic", "sm", "mw"]:
        hs_result = load_hidden_states(p, layer, mode="decision_point")
        if hs_result is None:
            log(f"  {p}: hidden states not found")
            continue
        hidden, meta = hs_result
        labels = get_labels(meta)
        n_bk = labels.sum()
        n_neurons = hidden.shape[1]

        # Per-neuron point-biserial correlation with BK
        corrs = np.zeros(n_neurons)
        pvals = np.zeros(n_neurons)
        for ni in range(n_neurons):
            r, p_val = pointbiserialr(labels, hidden[:, ni])
            corrs[ni] = r
            pvals[ni] = p_val

        # FDR correction
        reject, pvals_fdr, _, _ = multipletests(pvals, alpha=0.01, method='fdr_bh')

        paradigm_corrs[p] = {
            "corrs": corrs,
            "pvals_fdr": pvals_fdr,
            "significant": reject,
            "n_sig": int(reject.sum()),
            "n_bk": int(n_bk),
        }
        log(f"  {p}: {reject.sum()}/{n_neurons} significant (FDR p<0.01), {n_bk} BK")

    if len(paradigm_corrs) < 3:
        return {}

    # Universal: significant in ALL 3 with same sign
    all_sig = paradigm_corrs["ic"]["significant"] & paradigm_corrs["sm"]["significant"] & paradigm_corrs["mw"]["significant"]
    ic_r = paradigm_corrs["ic"]["corrs"]
    sm_r = paradigm_corrs["sm"]["corrs"]
    mw_r = paradigm_corrs["mw"]["corrs"]

    sign_consistent = all_sig & (
        ((ic_r > 0) & (sm_r > 0) & (mw_r > 0)) |
        ((ic_r < 0) & (sm_r < 0) & (mw_r < 0))
    )

    n_universal = sign_consistent.sum()
    n_all_sig = all_sig.sum()

    # BK-promoting vs inhibiting
    promoting = sign_consistent & (ic_r > 0)
    inhibiting = sign_consistent & (ic_r < 0)

    log(f"\n  All-3-sig: {n_all_sig}, sign-consistent: {n_universal}")
    log(f"  BK-promoting: {promoting.sum()}, BK-inhibiting: {inhibiting.sum()}")

    # Top-10 by min|r|
    universal_idx = np.where(sign_consistent)[0]
    min_abs_r = np.minimum(np.abs(ic_r[universal_idx]),
                           np.minimum(np.abs(sm_r[universal_idx]), np.abs(mw_r[universal_idx])))
    top10_order = np.argsort(min_abs_r)[-10:][::-1]
    top10 = []
    for rank, oi in enumerate(top10_order):
        ni = universal_idx[oi]
        direction = "BK-promoting" if ic_r[ni] > 0 else "BK-inhibiting"
        top10.append({
            "rank": rank + 1,
            "neuron": int(ni),
            "min_abs_r": float(min_abs_r[oi]),
            "ic_r": float(ic_r[ni]),
            "sm_r": float(sm_r[ni]),
            "mw_r": float(mw_r[ni]),
            "direction": direction,
        })
        log(f"  #{rank+1}: Neuron {ni}, min|r|={min_abs_r[oi]:.3f}, IC={ic_r[ni]:.3f} SM={sm_r[ni]:.3f} MW={mw_r[ni]:.3f} [{direction}]")

    return {
        "layer": layer,
        "n_neurons": 3584,
        "n_all_3_sig_fdr": int(n_all_sig),
        "n_sign_consistent": int(n_universal),
        "n_promoting": int(promoting.sum()),
        "n_inhibiting": int(inhibiting.sum()),
        "top10": top10,
        "per_paradigm_sig": {p: v["n_sig"] for p, v in paradigm_corrs.items()},
    }


# ============================================================
# 2. Hidden State Cross-Domain Transfer (V8 §3.1)
# ============================================================
def hidden_state_transfer():
    """Cross-domain transfer using hidden states (not SAE)."""
    log("\n" + "=" * 70)
    log("2. HIDDEN STATE CROSS-DOMAIN TRANSFER (Gemma L22)")
    log("=" * 70)

    layer = 22
    N_PERM = 200
    paradigm_data = {}

    for p in ["ic", "sm", "mw"]:
        hs_result = load_hidden_states(p, layer, mode="decision_point")
        if hs_result is None:
            continue
        hidden, meta = hs_result
        labels = get_labels(meta)
        paradigm_data[p] = {"hidden": hidden, "labels": labels}
        log(f"  {p}: {len(labels)} games, {labels.sum()} BK")

    results = {}
    pairs = [("ic", "sm"), ("ic", "mw"), ("sm", "mw")]

    for train_p, test_p in pairs:
        if train_p not in paradigm_data or test_p not in paradigm_data:
            continue

        X_train = paradigm_data[train_p]["hidden"]
        y_train = paradigm_data[train_p]["labels"]
        X_test = paradigm_data[test_p]["hidden"]
        y_test = paradigm_data[test_p]["labels"]

        # StandardScaler → PCA(50) → LogReg
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)

        n_comp = min(50, X_tr_s.shape[0] - 1, X_tr_s.shape[1])
        pca = PCA(n_components=n_comp, random_state=RANDOM_SEED)
        X_tr_pca = pca.fit_transform(X_tr_s)
        X_te_pca = pca.transform(X_te_s)

        clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=RANDOM_SEED)
        clf.fit(X_tr_pca, y_train)
        proba = clf.predict_proba(X_te_pca)[:, 1]
        auc = roc_auc_score(y_test, proba)

        # Permutation test
        null_aucs = []
        for _ in range(N_PERM):
            null_auc = roc_auc_score(np.random.permutation(y_test), proba)
            null_aucs.append(null_auc)
        null_aucs = np.array(null_aucs)
        perm_p = (null_aucs >= auc).mean()

        key = f"{train_p}_{test_p}"
        results[key] = {
            "train": train_p, "test": test_p,
            "auc": float(auc), "perm_p": float(perm_p),
            "null_mean": float(null_aucs.mean()),
        }
        log(f"  {train_p}→{test_p}: AUC={auc:.3f}, perm_p={perm_p:.3f}")

    return results


# ============================================================
# 3. 3D Shared Subspace (V8 §3.2)
# ============================================================
def shared_subspace():
    """PCA on 3 LR weight vectors → 3D BK subspace."""
    log("\n" + "=" * 70)
    log("3. 3D SHARED SUBSPACE (Gemma L22)")
    log("=" * 70)

    layer = 22
    weight_vectors = {}

    for p in ["ic", "sm", "mw"]:
        hs_result = load_hidden_states(p, layer, mode="decision_point")
        if hs_result is None:
            continue
        hidden, meta = hs_result
        labels = get_labels(meta)

        scaler = StandardScaler()
        X = scaler.fit_transform(hidden)
        clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=RANDOM_SEED)
        clf.fit(X, labels)
        weight_vectors[p] = clf.coef_[0]  # (3584,)
        log(f"  {p}: weight vector norm={np.linalg.norm(clf.coef_[0]):.3f}")

    if len(weight_vectors) < 3:
        return {}

    # Stack weight vectors and find shared subspace via PCA
    W = np.stack([weight_vectors["ic"], weight_vectors["sm"], weight_vectors["mw"]])  # (3, 3584)
    pca = PCA(n_components=3)
    W_pca = pca.fit_transform(W)
    explained = pca.explained_variance_ratio_

    log(f"  PCA explained variance: {explained}")

    # Cosines between weight vectors
    cos_ic_sm = np.dot(weight_vectors["ic"], weight_vectors["sm"]) / (np.linalg.norm(weight_vectors["ic"]) * np.linalg.norm(weight_vectors["sm"]))
    cos_ic_mw = np.dot(weight_vectors["ic"], weight_vectors["mw"]) / (np.linalg.norm(weight_vectors["ic"]) * np.linalg.norm(weight_vectors["mw"]))
    cos_sm_mw = np.dot(weight_vectors["sm"], weight_vectors["mw"]) / (np.linalg.norm(weight_vectors["sm"]) * np.linalg.norm(weight_vectors["mw"]))

    log(f"  Cosines: IC-SM={cos_ic_sm:.3f}, IC-MW={cos_ic_mw:.3f}, SM-MW={cos_sm_mw:.3f}")

    # Project all paradigm data into 3D subspace and classify
    projection_basis = pca.components_  # (3, 3584)
    results_3d = {}

    for p in ["ic", "sm", "mw"]:
        hs_result = load_hidden_states(p, layer, mode="decision_point")
        hidden, meta = hs_result
        labels = get_labels(meta)

        X_3d = hidden @ projection_basis.T  # (n_games, 3)
        clf_3d = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=RANDOM_SEED)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        scores = cross_val_score(clf_3d, X_3d, labels, cv=cv, scoring='roc_auc')

        results_3d[p] = {"auc_mean": float(scores.mean()), "auc_std": float(scores.std())}
        log(f"  {p} 3D subspace AUC: {scores.mean():.3f} ± {scores.std():.3f}")

    return {
        "pca_explained_variance": explained.tolist(),
        "cosines": {"ic_sm": float(cos_ic_sm), "ic_mw": float(cos_ic_mw), "sm_mw": float(cos_sm_mw)},
        "3d_classification": results_3d,
    }


# ============================================================
# 4. G-Prompt BK Direction Alignment (V8 §4.2)
# ============================================================
def g_prompt_alignment():
    """Cosine between G-prompt direction and BK direction."""
    log("\n" + "=" * 70)
    log("4. G-PROMPT BK DIRECTION ALIGNMENT (Gemma L22)")
    log("=" * 70)

    layer = 22
    results = {}

    for p in ["ic", "sm", "mw"]:
        hs_result = load_hidden_states(p, layer, mode="decision_point")
        if hs_result is None:
            continue
        hidden, meta = hs_result
        labels = get_labels(meta)
        conditions = meta["prompt_conditions"]

        # BK direction
        bk_dir = hidden[labels == 1].mean(0) - hidden[labels == 0].mean(0)

        # G-prompt direction (has G vs no G)
        has_g = np.array(['G' in str(c) for c in conditions])
        if has_g.sum() < 10 or (~has_g).sum() < 10:
            log(f"  {p}: insufficient G/non-G samples")
            continue

        g_dir = hidden[has_g].mean(0) - hidden[~has_g].mean(0)
        cos = np.dot(bk_dir, g_dir) / (np.linalg.norm(bk_dir) * np.linalg.norm(g_dir) + 1e-10)

        results[p] = {
            "cos_g_bk": float(cos),
            "n_with_g": int(has_g.sum()),
            "n_without_g": int((~has_g).sum()),
            "bk_rate_with_g": float(labels[has_g].mean()),
            "bk_rate_without_g": float(labels[~has_g].mean()),
        }
        log(f"  {p}: cos(G_dir, BK_dir)={cos:.3f}, BK_with_G={labels[has_g].mean():.3f}, BK_without_G={labels[~has_g].mean():.3f}")

    return results


# ============================================================
# 5. Bet Constraint Linear Mapping (V8 §4.3)
# ============================================================
def bet_constraint_mapping():
    """BK-projection vs bet constraint (IC only: c10, c30, c50, c70)."""
    log("\n" + "=" * 70)
    log("5. BET CONSTRAINT LINEAR MAPPING (Gemma IC L22)")
    log("=" * 70)

    layer = 22
    hs_result = load_hidden_states("ic", layer, mode="decision_point")
    if hs_result is None:
        return {}
    hidden, meta = hs_result
    labels = get_labels(meta)
    constraints = meta.get("bet_constraints", None)
    if constraints is None:
        log("  No bet_constraints in metadata")
        return {}

    # Train BK classifier on all IC data
    scaler = StandardScaler()
    X = scaler.fit_transform(hidden)
    clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=RANDOM_SEED)
    clf.fit(X, labels)
    bk_proba = clf.predict_proba(X)[:, 1]

    # Group by constraint
    unique_c = sorted(set(constraints))
    log(f"  Constraints: {unique_c}")

    constraint_stats = {}
    means = []
    constraint_values = []

    for c in unique_c:
        mask = constraints == c
        mean_proj = bk_proba[mask].mean()
        bk_rate = labels[mask].mean()
        constraint_stats[str(c)] = {
            "n": int(mask.sum()),
            "mean_bk_proba": float(mean_proj),
            "bk_rate": float(bk_rate),
        }
        log(f"  {c}: n={mask.sum()}, mean_BK_proba={mean_proj:.3f}, BK_rate={bk_rate:.3f}")

        # Extract numeric value
        try:
            num = int(str(c).replace('c', '').replace('$', ''))
            means.append(mean_proj)
            constraint_values.append(num)
        except:
            pass

    # Linear correlation
    if len(means) >= 3:
        r, p = pearsonr(constraint_values, means)
        log(f"  Linear correlation: r={r:.3f}, p={p:.4f}")
        return {"constraints": constraint_stats, "linear_r": float(r), "linear_p": float(p)}

    return {"constraints": constraint_stats}


def main():
    log("=" * 70)
    log("V8-STYLE HIDDEN STATE ANALYSES FOR V9 UPDATE")
    log("=" * 70)

    all_results = {"timestamp": datetime.now().isoformat()}
    all_results["universal_bk_neurons"] = universal_bk_neurons()
    all_results["hidden_state_transfer"] = hidden_state_transfer()
    all_results["shared_subspace"] = shared_subspace()
    all_results["g_prompt_alignment"] = g_prompt_alignment()
    all_results["bet_constraint_mapping"] = bet_constraint_mapping()

    out = Path("results/json") / f"hidden_state_v8style_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out, 'w') as f:
        def convert(obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return obj
        json.dump(all_results, f, indent=2, default=convert)
    log(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
