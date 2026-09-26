#!/usr/bin/env python3
"""
V10 LLaMA Symmetric Analyses: 11 analyses for cross-domain cross-model comparison.

Phase 1 (LLaMA Symmetric, 7 analyses):
  1a. Universal BK Neurons (IC+SM, 2-paradigm, FDR-corrected)
  1b. Hidden State Cross-Domain Transfer (IC<->SM)
  1c. HS vs SAE Transfer Comparison
  1d. 2D Shared BK Subspace (L22)
  1e. G-Prompt Direction Alignment (SM)
  1f. Bet Constraint Mapping (IC, SAE level)
  1g. Prompt Component at HS Level (SM)

Phase 2 (Both-Model Reinforcement, 4 analyses):
  2a. Shared BK Neuron Profile
  2b. Var/Fix Common BK Features (SAE, IC)
  2c. Var/Fix BK Direction at SAE Level (IC)
  2d. Variable Anomaly Investigation

Usage:
    python run_llama_v10_symmetric.py
"""

import sys
import json
import time
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy import stats
from scipy.stats import pointbiserialr, pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
np.random.seed(42)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT = Path("/home/v-seungplee/data/llm-addiction/sae_features_v3")
IC_HS_PATH = DATA_ROOT / "investment_choice" / "llama" / "hidden_states_dp.npz"
SM_HS_PATH = DATA_ROOT / "slot_machine" / "llama" / "hidden_states_dp.npz"
GEMMA_IC_HS_PATH = DATA_ROOT / "investment_choice" / "gemma" / "hidden_states_dp.npz"

REPO_ROOT = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis")
JSON_DIR = REPO_ROOT / "results" / "json"
JSON_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
N_PERMUTATIONS = 200
FDR_ALPHA = 0.01
MIN_ACTIVATION_RATE = 0.01
HS_LAYERS = [8, 12, 22, 25, 30]
N_HS_LAYERS = len(HS_LAYERS)
HIDDEN_DIM = 4096
N_SAE_FEATURES = 32768


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    """Print a timestamped message."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def numpy_convert(obj):
    """JSON serialiser for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def load_hs(path: Path) -> dict:
    """Load a hidden-states decision-point NPZ and return dict of arrays."""
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def bk_labels(outcomes: np.ndarray) -> np.ndarray:
    """Binary labels: 1 = bankruptcy, 0 = voluntary_stop."""
    return (outcomes == "bankruptcy").astype(np.int32)


def load_sae_dp(paradigm: str, layer: int) -> tuple:
    """Load SAE features at the decision point for LLaMA.

    Returns (dense_features, metadata_dict).
    Uses efficient row filtering to avoid materialising the full matrix.
    """
    npz_path = DATA_ROOT / paradigm / "llama" / f"sae_features_L{layer}.npz"
    data = np.load(npz_path, allow_pickle=False)

    is_last = data["is_last_round"].astype(bool)
    shape = tuple(data["shape"])
    dp_indices = np.where(is_last)[0]
    n_dp = len(dp_indices)

    # Build a mapping from old row index -> new compressed row index
    row_map = np.full(shape[0], -1, dtype=np.int64)
    row_map[dp_indices] = np.arange(n_dp)

    row_idx = data["row_indices"]
    col_idx = data["col_indices"]
    vals = data["values"]

    # Filter to decision-point rows using vectorised operations
    dp_mask = np.isin(row_idx, dp_indices)
    new_rows = row_map[row_idx[dp_mask]]
    new_cols = col_idx[dp_mask]
    new_vals = vals[dp_mask]

    dense = np.zeros((n_dp, shape[1]), dtype=np.float32)
    dense[new_rows, new_cols] = new_vals

    meta = {
        "game_outcomes": data["game_outcomes"][is_last],
        "bet_types": data["bet_types"][is_last],
    }
    for field in ["bet_constraints", "prompt_conditions", "balances", "game_ids"]:
        if field in data.files:
            meta[field] = data[field][is_last]

    return dense, meta


def filter_active(features: np.ndarray, min_rate: float = MIN_ACTIVATION_RATE):
    """Return (filtered_features, active_column_indices)."""
    rate = (features != 0).mean(axis=0)
    mask = rate >= min_rate
    return features[:, mask], np.where(mask)[0]


def cohens_d_vec(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Vectorised Cohen's d (BK vs Safe) for every column."""
    bk = labels == 1
    safe = labels == 0
    n_bk, n_safe = bk.sum(), safe.sum()
    if n_bk < 2 or n_safe < 2:
        return np.zeros(features.shape[1])
    m_bk = features[bk].mean(axis=0)
    m_safe = features[safe].mean(axis=0)
    v_bk = features[bk].var(axis=0, ddof=1)
    v_safe = features[safe].var(axis=0, ddof=1)
    pooled = np.sqrt(((n_bk - 1) * v_bk + (n_safe - 1) * v_safe) / (n_bk + n_safe - 2))
    pooled = np.where(pooled == 0, 1e-10, pooled)
    return (m_bk - m_safe) / pooled


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


# ===================================================================
# 1a. Universal BK Neurons (IC+SM, 2-paradigm)
# ===================================================================
def analysis_1a_universal_bk_neurons() -> dict:
    """Point-biserial per neuron x paradigm, FDR, sign-consistency."""
    log("=" * 70)
    log("1a. UNIVERSAL BK NEURONS (LLaMA IC+SM, 2-paradigm, FDR)")
    log("=" * 70)

    paradigm_corrs = {}
    for name, path in [("ic", IC_HS_PATH), ("sm", SM_HS_PATH)]:
        data = load_hs(path)
        hs = data["hidden_states"]  # (n_games, 5, 4096)
        labels = bk_labels(data["game_outcomes"])

        per_layer = {}
        for j, layer in enumerate(HS_LAYERS):
            hidden = hs[:, j, :]
            n_neurons = hidden.shape[1]

            corrs = np.zeros(n_neurons)
            pvals = np.zeros(n_neurons)
            for ni in range(n_neurons):
                r, p = pointbiserialr(labels, hidden[:, ni])
                corrs[ni] = r
                pvals[ni] = p

            reject, pvals_fdr, _, _ = multipletests(pvals, alpha=FDR_ALPHA, method="fdr_bh")
            per_layer[layer] = {
                "corrs": corrs,
                "reject": reject,
                "n_sig": int(reject.sum()),
            }
            log(f"  {name} L{layer}: {reject.sum()}/{n_neurons} sig (FDR p<{FDR_ALPHA}), "
                f"n_bk={labels.sum()}")

        paradigm_corrs[name] = per_layer

    # Cross-paradigm sign-consistency per layer
    results = {}
    for layer in HS_LAYERS:
        ic_layer = paradigm_corrs["ic"][layer]
        sm_layer = paradigm_corrs["sm"][layer]

        both_sig = ic_layer["reject"] & sm_layer["reject"]
        ic_r = ic_layer["corrs"]
        sm_r = sm_layer["corrs"]

        sign_consistent = both_sig & (
            ((ic_r > 0) & (sm_r > 0)) | ((ic_r < 0) & (sm_r < 0))
        )
        n_universal = int(sign_consistent.sum())
        promoting = int((sign_consistent & (ic_r > 0)).sum())
        inhibiting = int((sign_consistent & (ic_r < 0)).sum())

        # Top-10 by min|r| among universal neurons
        uni_idx = np.where(sign_consistent)[0]
        top10 = []
        if len(uni_idx) > 0:
            min_abs_r = np.minimum(np.abs(ic_r[uni_idx]), np.abs(sm_r[uni_idx]))
            order = np.argsort(min_abs_r)[-10:][::-1]
            for rank, oi in enumerate(order):
                ni = int(uni_idx[oi])
                direction = "promoting" if ic_r[ni] > 0 else "inhibiting"
                top10.append({
                    "rank": rank + 1,
                    "neuron": ni,
                    "min_abs_r": round(float(min_abs_r[oi]), 4),
                    "ic_r": round(float(ic_r[ni]), 4),
                    "sm_r": round(float(sm_r[ni]), 4),
                    "direction": direction,
                })

        results[f"L{layer}"] = {
            "ic_sig": ic_layer["n_sig"],
            "sm_sig": sm_layer["n_sig"],
            "both_sig": int(both_sig.sum()),
            "n_universal": n_universal,
            "promoting": promoting,
            "inhibiting": inhibiting,
            "top10": top10,
        }
        log(f"  L{layer} universal: {n_universal} (promoting={promoting}, inhibiting={inhibiting})")

    return results


# ===================================================================
# 1b. Hidden State Cross-Domain Transfer (IC <-> SM)
# ===================================================================
def analysis_1b_hs_transfer() -> dict:
    """PCA(50)->LogReg transfer between IC and SM, permutation test."""
    log("\n" + "=" * 70)
    log("1b. HIDDEN STATE CROSS-DOMAIN TRANSFER (LLaMA IC<->SM)")
    log("=" * 70)

    paradigm_data = {}
    for name, path in [("ic", IC_HS_PATH), ("sm", SM_HS_PATH)]:
        data = load_hs(path)
        paradigm_data[name] = {
            "hs": data["hidden_states"],
            "labels": bk_labels(data["game_outcomes"]),
        }
        log(f"  {name}: {data['hidden_states'].shape[0]} games, "
            f"{(data['game_outcomes'] == 'bankruptcy').sum()} BK")

    results = {}
    directions = [("ic", "sm"), ("sm", "ic")]

    for layer_j, layer in enumerate(HS_LAYERS):
        layer_results = {}
        for train_p, test_p in directions:
            X_train = paradigm_data[train_p]["hs"][:, layer_j, :]
            y_train = paradigm_data[train_p]["labels"]
            X_test = paradigm_data[test_p]["hs"][:, layer_j, :]
            y_test = paradigm_data[test_p]["labels"]

            # PCA(50) on train
            n_comp = min(50, X_train.shape[0] - 1, X_train.shape[1])
            pca = PCA(n_components=n_comp, random_state=RANDOM_SEED)
            X_tr_pca = pca.fit_transform(X_train)
            X_te_pca = pca.transform(X_test)

            clf = LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED
            )
            clf.fit(X_tr_pca, y_train)
            proba = clf.predict_proba(X_te_pca)[:, 1]
            auc = roc_auc_score(y_test, proba)

            # Permutation test (shuffle test labels)
            rng = np.random.RandomState(RANDOM_SEED)
            null_aucs = np.array([
                roc_auc_score(rng.permutation(y_test), proba)
                for _ in range(N_PERMUTATIONS)
            ])
            perm_p = float((null_aucs >= auc).mean())

            key = f"{train_p}_to_{test_p}"
            layer_results[key] = {
                "auc": round(float(auc), 4),
                "perm_p": round(perm_p, 4),
                "null_mean": round(float(null_aucs.mean()), 4),
            }
            log(f"  L{layer} {train_p}->{test_p}: AUC={auc:.4f}, perm_p={perm_p:.4f}")

        results[f"L{layer}"] = layer_results

    return results


# ===================================================================
# 1c. HS vs SAE Transfer Comparison
# ===================================================================
def analysis_1c_hs_sae_comparison(hs_transfer_results: dict) -> dict:
    """Compare HS transfer AUC with SAE transfer AUC at matching layers."""
    log("\n" + "=" * 70)
    log("1c. HS vs SAE TRANSFER COMPARISON (LLaMA IC<->SM)")
    log("=" * 70)

    results = {}
    directions = [("ic", "sm"), ("sm", "ic")]
    paradigm_names = {"ic": "investment_choice", "sm": "slot_machine"}

    for layer in HS_LAYERS:
        layer_key = f"L{layer}"
        layer_results = {}

        # Load SAE features at this layer for both paradigms
        sae_data = {}
        for p_short, p_long in paradigm_names.items():
            log(f"  Loading SAE {p_short} L{layer}...")
            feat, meta = load_sae_dp(p_long, layer)
            feat_active, active_idx = filter_active(feat)
            labels = bk_labels(meta["game_outcomes"])
            sae_data[p_short] = {
                "features": feat_active,
                "labels": labels,
                "n_active": len(active_idx),
            }
            log(f"    {p_short}: {feat_active.shape[0]} games, {len(active_idx)} active features")

        for train_p, test_p in directions:
            direction_key = f"{train_p}_to_{test_p}"

            # SAE transfer
            X_train = sae_data[train_p]["features"]
            y_train = sae_data[train_p]["labels"]
            X_test = sae_data[test_p]["features"]
            y_test = sae_data[test_p]["labels"]

            # Need consistent feature space: use union-active features
            # Since we loaded separately, we must reload with aligned columns.
            # Simpler: load both, find union-active, filter both
            ic_full, ic_meta = load_sae_dp(paradigm_names["ic"], layer)
            sm_full, sm_meta = load_sae_dp(paradigm_names["sm"], layer)

            ic_rate = (ic_full != 0).mean(axis=0)
            sm_rate = (sm_full != 0).mean(axis=0)
            union_active = (ic_rate >= MIN_ACTIVATION_RATE) | (sm_rate >= MIN_ACTIVATION_RATE)
            union_idx = np.where(union_active)[0]
            n_union = len(union_idx)

            if train_p == "ic":
                X_tr_sae = ic_full[:, union_active].astype(np.float32)
                y_tr_sae = bk_labels(ic_meta["game_outcomes"])
                X_te_sae = sm_full[:, union_active].astype(np.float32)
                y_te_sae = bk_labels(sm_meta["game_outcomes"])
            else:
                X_tr_sae = sm_full[:, union_active].astype(np.float32)
                y_tr_sae = bk_labels(sm_meta["game_outcomes"])
                X_te_sae = ic_full[:, union_active].astype(np.float32)
                y_te_sae = bk_labels(ic_meta["game_outcomes"])

            n_comp_sae = min(50, X_tr_sae.shape[0] - 1, n_union)
            pca_sae = PCA(n_components=n_comp_sae, random_state=RANDOM_SEED)
            X_tr_pca = pca_sae.fit_transform(X_tr_sae)
            X_te_pca = pca_sae.transform(X_te_sae)

            clf_sae = LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED
            )
            clf_sae.fit(X_tr_pca, y_tr_sae)
            proba_sae = clf_sae.predict_proba(X_te_pca)[:, 1]
            auc_sae = roc_auc_score(y_te_sae, proba_sae)

            # Retrieve HS AUC from 1b results
            hs_auc = hs_transfer_results.get(layer_key, {}).get(direction_key, {}).get("auc", None)
            delta = round(float(hs_auc - auc_sae), 4) if hs_auc is not None else None

            layer_results[direction_key] = {
                "hs_auc": hs_auc,
                "sae_auc": round(float(auc_sae), 4),
                "delta_hs_minus_sae": delta,
                "n_union_active_sae": int(n_union),
            }
            log(f"  L{layer} {train_p}->{test_p}: HS={hs_auc}, SAE={auc_sae:.4f}, "
                f"delta={delta}")

        results[layer_key] = layer_results

    return results


# ===================================================================
# 1d. 2D Shared BK Subspace
# ===================================================================
def analysis_1d_shared_subspace() -> dict:
    """Train separate LogReg on IC and SM, PCA(2) on weight vectors."""
    log("\n" + "=" * 70)
    log("1d. 2D SHARED BK SUBSPACE (LLaMA L22)")
    log("=" * 70)

    layer_j = HS_LAYERS.index(22)  # Index into the 5-layer array

    weight_vectors = {}
    paradigm_hidden = {}
    paradigm_labels = {}

    for name, path in [("ic", IC_HS_PATH), ("sm", SM_HS_PATH)]:
        data = load_hs(path)
        hidden = data["hidden_states"][:, layer_j, :]
        labels = bk_labels(data["game_outcomes"])

        clf = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED
        )
        clf.fit(hidden, labels)
        weight_vectors[name] = clf.coef_[0]  # (4096,)
        paradigm_hidden[name] = hidden
        paradigm_labels[name] = labels

        log(f"  {name}: weight norm={np.linalg.norm(clf.coef_[0]):.4f}, "
            f"n={len(labels)}, BK={labels.sum()}")

    # PCA(2) on the 2 weight vectors
    W = np.stack([weight_vectors["ic"], weight_vectors["sm"]])  # (2, 4096)
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    pca.fit(W)
    basis = pca.components_  # (2, 4096)
    explained = pca.explained_variance_ratio_

    log(f"  PCA explained variance: {explained}")

    # Project each paradigm into 2D and classify
    results_2d = {}
    for name in ["ic", "sm"]:
        X_2d = paradigm_hidden[name] @ basis.T  # (n_games, 2)
        labels = paradigm_labels[name]
        clf_2d = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED
        )
        clf_2d.fit(X_2d, labels)
        proba = clf_2d.predict_proba(X_2d)[:, 1]
        auc = roc_auc_score(labels, proba)
        results_2d[name] = {"auc_2d": round(float(auc), 4)}
        log(f"  {name} 2D AUC (resubstitution): {auc:.4f}")

    # Cosine between IC and SM weight vectors
    weight_cos = cosine(weight_vectors["ic"], weight_vectors["sm"])
    log(f"  Cosine(IC_weights, SM_weights): {weight_cos:.4f}")

    return {
        "layer": 22,
        "pca_explained_variance": explained.tolist(),
        "weight_cosine_ic_sm": round(weight_cos, 4),
        "classification_2d": results_2d,
    }


# ===================================================================
# 1e. G-Prompt Direction Alignment (SM)
# ===================================================================
def analysis_1e_g_prompt_alignment() -> dict:
    """Cosine between G-direction and BK-direction in SM hidden states."""
    log("\n" + "=" * 70)
    log("1e. G-PROMPT DIRECTION ALIGNMENT (LLaMA SM)")
    log("=" * 70)

    data = load_hs(SM_HS_PATH)
    hs = data["hidden_states"]  # (3200, 5, 4096)
    labels = bk_labels(data["game_outcomes"])
    conditions = data["prompt_conditions"]

    has_g = np.array(["G" in str(c) for c in conditions])
    n_with_g = int(has_g.sum())
    n_without_g = int((~has_g).sum())
    log(f"  SM: {len(labels)} games, G-present={n_with_g}, G-absent={n_without_g}")

    # BK rates by G presence
    bk_rate_g = float(labels[has_g].mean()) if n_with_g > 0 else 0.0
    bk_rate_no_g = float(labels[~has_g].mean()) if n_without_g > 0 else 0.0
    log(f"  BK rate with G: {bk_rate_g:.4f}, without G: {bk_rate_no_g:.4f}")

    results = {
        "n_with_g": n_with_g,
        "n_without_g": n_without_g,
        "bk_rate_with_g": round(bk_rate_g, 4),
        "bk_rate_without_g": round(bk_rate_no_g, 4),
        "per_layer": {},
    }

    for j, layer in enumerate(HS_LAYERS):
        hidden = hs[:, j, :]

        # G direction: mean(G games) - mean(non-G games)
        g_dir = hidden[has_g].mean(axis=0) - hidden[~has_g].mean(axis=0)

        # BK direction: mean(BK games) - mean(Safe games)
        bk_dir = hidden[labels == 1].mean(axis=0) - hidden[labels == 0].mean(axis=0)

        cos_val = cosine(g_dir, bk_dir)
        results["per_layer"][f"L{layer}"] = {
            "cosine_g_bk": round(cos_val, 4),
            "g_dir_norm": round(float(np.linalg.norm(g_dir)), 4),
            "bk_dir_norm": round(float(np.linalg.norm(bk_dir)), 4),
        }
        log(f"  L{layer}: cos(G_dir, BK_dir) = {cos_val:.4f}")

    return results


# ===================================================================
# 1f. Bet Constraint Mapping (IC, SAE level)
# ===================================================================
def analysis_1f_bet_constraint_mapping() -> dict:
    """LogReg BK probability grouped by bet constraint (IC SAE L22)."""
    log("\n" + "=" * 70)
    log("1f. BET CONSTRAINT MAPPING (LLaMA IC, SAE L22)")
    log("=" * 70)

    feat, meta = load_sae_dp("investment_choice", 22)
    labels = bk_labels(meta["game_outcomes"])
    constraints = meta["bet_constraints"]

    log(f"  IC SAE L22: {feat.shape[0]} games, {feat.shape[1]} features, BK={labels.sum()}")
    log(f"  Constraints: {np.unique(constraints)}")

    # Filter to active features
    feat_active, active_idx = filter_active(feat)
    log(f"  Active features: {len(active_idx)}")

    # PCA to avoid high-dimensional LogReg instability
    n_comp = min(50, feat_active.shape[0] - 1, feat_active.shape[1])
    pca = PCA(n_components=n_comp, random_state=RANDOM_SEED)
    feat_pca = pca.fit_transform(feat_active)

    # Train LogReg(balanced) on BK
    clf = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED
    )
    clf.fit(feat_pca, labels)
    bk_proba = clf.predict_proba(feat_pca)[:, 1]

    # Group by bet constraint
    unique_c = sorted(np.unique(constraints))
    constraint_stats = {}
    numeric_vals = []
    mean_probas = []

    for c in unique_c:
        mask = constraints == c
        mean_p = float(bk_proba[mask].mean())
        bk_rate = float(labels[mask].mean())
        n = int(mask.sum())

        constraint_stats[str(c)] = {
            "n": n,
            "mean_bk_proba": round(mean_p, 4),
            "bk_rate": round(bk_rate, 4),
        }
        log(f"  c{c}: n={n}, mean_BK_proba={mean_p:.4f}, BK_rate={bk_rate:.4f}")

        try:
            num = int(str(c).replace("c", "").replace("$", ""))
            numeric_vals.append(num)
            mean_probas.append(mean_p)
        except ValueError:
            pass

    # Linear correlation
    result = {"constraints": constraint_stats, "n_active_features": int(len(active_idx))}
    if len(numeric_vals) >= 3:
        r, p = pearsonr(numeric_vals, mean_probas)
        result["linear_r"] = round(float(r), 4)
        result["linear_p"] = float(p)
        log(f"  Pearson r(constraint, BK_proba) = {r:.4f}, p = {p:.4e}")

    return result


# ===================================================================
# 1g. Prompt Component at HS Level (SM)
# ===================================================================
def analysis_1g_prompt_component_hs() -> dict:
    """Interaction regression: neuron ~ outcome + component + outcome*component."""
    log("\n" + "=" * 70)
    log("1g. PROMPT COMPONENT AT HS LEVEL (LLaMA SM)")
    log("=" * 70)

    data = load_hs(SM_HS_PATH)
    hs = data["hidden_states"]  # (3200, 5, 4096)
    labels = bk_labels(data["game_outcomes"])
    conditions = data["prompt_conditions"]

    components = ["G", "M", "H", "W", "P"]
    results = {}

    for comp in components:
        has_comp = np.array([comp in str(c) for c in conditions])
        n_with = int(has_comp.sum())
        n_without = int((~has_comp).sum())

        bk_rate_with = float(labels[has_comp].mean()) if n_with > 0 else 0.0
        bk_rate_without = float(labels[~has_comp].mean()) if n_without > 0 else 0.0
        bk_ratio = bk_rate_with / max(bk_rate_without, 1e-10)

        log(f"  Component {comp}: n_with={n_with}, n_without={n_without}, "
            f"BK_with={bk_rate_with:.4f}, BK_without={bk_rate_without:.4f}")

        comp_results = {
            "n_with": n_with,
            "n_without": n_without,
            "bk_rate_with": round(bk_rate_with, 4),
            "bk_rate_without": round(bk_rate_without, 4),
            "bk_ratio": round(bk_ratio, 4),
            "per_layer": {},
        }

        # Vectorised OLS per layer: Y(neuron) ~ beta0 + beta1*outcome + beta2*comp + beta3*outcome*comp
        outcome_vec = labels.astype(np.float64)
        comp_vec = has_comp.astype(np.float64)
        interaction_vec = outcome_vec * comp_vec
        n_games = len(labels)

        X = np.column_stack([
            np.ones(n_games),
            outcome_vec,
            comp_vec,
            interaction_vec,
        ])  # (n_games, 4)

        XtX = X.T @ X
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(XtX)

        for j, layer in enumerate(HS_LAYERS):
            hidden = hs[:, j, :].astype(np.float64)  # (n_games, 4096)
            n_neurons = hidden.shape[1]

            # Vectorised OLS: Beta = (X'X)^-1 X' Y
            Beta = XtX_inv @ (X.T @ hidden)  # (4, 4096)

            # Residuals and SE
            Y_hat = X @ Beta
            resid = hidden - Y_hat
            dof = n_games - 4
            RSS = (resid ** 2).sum(axis=0)  # (4096,)
            sigma2 = RSS / dof

            se = np.sqrt(np.outer(np.diag(XtX_inv), sigma2))  # (4, 4096)
            se = np.where(se < 1e-15, 1e-15, se)
            t_stats = Beta / se
            p_vals = 2 * stats.t.sf(np.abs(t_stats), dof)  # (4, 4096)

            # Interaction significance (p < 0.01 on beta3)
            interaction_sig = p_vals[3] < 0.01
            n_interaction_sig = int(interaction_sig.sum())

            # Amplifies BK: interaction sig AND sign(beta3) == sign(beta1)
            # Also require beta1 (outcome) to be at least marginally significant (p < 0.05)
            amplifies = (
                interaction_sig
                & (np.sign(Beta[3]) == np.sign(Beta[1]))
                & (p_vals[1] < 0.05)
            )
            n_amplifies = int(amplifies.sum())

            comp_results["per_layer"][f"L{layer}"] = {
                "n_interaction_sig": n_interaction_sig,
                "pct_interaction_sig": round(n_interaction_sig / n_neurons * 100, 2),
                "n_amplifies_bk": n_amplifies,
                "pct_amplifies_bk": round(n_amplifies / n_neurons * 100, 2),
            }
            log(f"    L{layer} {comp}: interaction_sig={n_interaction_sig} "
                f"({n_interaction_sig/n_neurons*100:.1f}%), "
                f"amplifies={n_amplifies} ({n_amplifies/n_neurons*100:.1f}%)")

        results[comp] = comp_results

    return results


# ===================================================================
# 2a. Shared BK Neuron Profile
# ===================================================================
def analysis_2a_shared_bk_profile() -> dict:
    """Interaction regression to find shared BK neurons (beta1 sig, beta3 NS)."""
    log("\n" + "=" * 70)
    log("2a. SHARED BK NEURON PROFILE (LLaMA IC, optionally Gemma)")
    log("=" * 70)

    results = {}

    # Load LLaMA IC
    ic_data = load_hs(IC_HS_PATH)
    ic_hs = ic_data["hidden_states"]
    ic_labels = bk_labels(ic_data["game_outcomes"])
    ic_bt = (ic_data["bet_types"] == "variable").astype(np.float64)
    n_ic = len(ic_labels)
    log(f"  LLaMA IC: {n_ic} games, {ic_labels.sum()} BK")

    # Interaction regression: neuron ~ outcome + bet_type + outcome*bet_type
    # "Shared BK neurons" = beta1 (outcome) significant AND beta3 (interaction) NOT significant
    X = np.column_stack([
        np.ones(n_ic),
        ic_labels.astype(np.float64),
        ic_bt,
        ic_labels.astype(np.float64) * ic_bt,
    ])
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)

    llama_results = {}
    for j, layer in enumerate(HS_LAYERS):
        hidden = ic_hs[:, j, :].astype(np.float64)
        n_neurons = hidden.shape[1]

        Beta = XtX_inv @ (X.T @ hidden)  # (4, n_neurons)
        Y_hat = X @ Beta
        resid = hidden - Y_hat
        dof = n_ic - 4
        RSS = (resid ** 2).sum(axis=0)
        sigma2 = RSS / dof
        se = np.sqrt(np.outer(np.diag(XtX_inv), sigma2))
        se = np.where(se < 1e-15, 1e-15, se)
        t_stats = Beta / se
        p_vals = 2 * stats.t.sf(np.abs(t_stats), dof)

        # Shared: beta1 (outcome) sig p<0.01 AND beta3 (interaction) NS p>0.05
        outcome_sig = p_vals[1] < 0.01
        interaction_ns = p_vals[3] > 0.05
        shared = outcome_sig & interaction_ns
        n_shared = int(shared.sum())

        # Direction and effect size for shared neurons
        shared_idx = np.where(shared)[0]
        promoting_count = int((Beta[1, shared_idx] > 0).sum())
        inhibiting_count = n_shared - promoting_count

        # Top-10 shared neurons by |beta1|
        top10 = []
        if len(shared_idx) > 0:
            abs_beta1 = np.abs(Beta[1, shared_idx])
            order = np.argsort(abs_beta1)[-10:][::-1]
            for rank, oi in enumerate(order):
                ni = int(shared_idx[oi])
                direction = "promoting" if Beta[1, ni] > 0 else "inhibiting"
                top10.append({
                    "rank": rank + 1,
                    "neuron": ni,
                    "beta_outcome": round(float(Beta[1, ni]), 6),
                    "p_outcome": float(p_vals[1, ni]),
                    "p_interaction": float(p_vals[3, ni]),
                    "direction": direction,
                })

        llama_results[f"L{layer}"] = {
            "n_outcome_sig": int(outcome_sig.sum()),
            "n_interaction_ns": int(interaction_ns.sum()),
            "n_shared": n_shared,
            "promoting": promoting_count,
            "inhibiting": inhibiting_count,
            "top10": top10,
        }
        log(f"  LLaMA L{layer}: outcome_sig={outcome_sig.sum()}, shared={n_shared} "
            f"(promoting={promoting_count}, inhibiting={inhibiting_count})")

    results["llama"] = llama_results

    # Try Gemma IC hidden states if they exist
    if GEMMA_IC_HS_PATH.exists():
        log(f"  Loading Gemma IC hidden states...")
        gemma_data = load_hs(GEMMA_IC_HS_PATH)
        gemma_hs = gemma_data["hidden_states"]
        gemma_labels = bk_labels(gemma_data["game_outcomes"])
        gemma_bt = (gemma_data["bet_types"] == "variable").astype(np.float64)
        gemma_layers = gemma_data["layers"]
        n_gemma = len(gemma_labels)
        log(f"  Gemma IC: {n_gemma} games, {gemma_labels.sum()} BK")

        X_g = np.column_stack([
            np.ones(n_gemma),
            gemma_labels.astype(np.float64),
            gemma_bt,
            gemma_labels.astype(np.float64) * gemma_bt,
        ])
        XtX_g = X_g.T @ X_g
        try:
            XtX_inv_g = np.linalg.inv(XtX_g)
        except np.linalg.LinAlgError:
            XtX_inv_g = np.linalg.pinv(XtX_g)

        gemma_results = {}
        for j_g, layer_g in enumerate(gemma_layers):
            hidden_g = gemma_hs[:, j_g, :].astype(np.float64)
            n_neurons_g = hidden_g.shape[1]

            Beta_g = XtX_inv_g @ (X_g.T @ hidden_g)
            Y_hat_g = X_g @ Beta_g
            resid_g = hidden_g - Y_hat_g
            dof_g = n_gemma - 4
            RSS_g = (resid_g ** 2).sum(axis=0)
            sigma2_g = RSS_g / dof_g
            se_g = np.sqrt(np.outer(np.diag(XtX_inv_g), sigma2_g))
            se_g = np.where(se_g < 1e-15, 1e-15, se_g)
            t_stats_g = Beta_g / se_g
            p_vals_g = 2 * stats.t.sf(np.abs(t_stats_g), dof_g)

            outcome_sig_g = p_vals_g[1] < 0.01
            interaction_ns_g = p_vals_g[3] > 0.05
            shared_g = outcome_sig_g & interaction_ns_g
            n_shared_g = int(shared_g.sum())

            gemma_results[f"L{int(layer_g)}"] = {
                "n_shared": n_shared_g,
                "n_neurons": n_neurons_g,
            }
            log(f"  Gemma L{layer_g}: shared={n_shared_g}/{n_neurons_g}")

        results["gemma"] = gemma_results
    else:
        log(f"  Gemma IC hidden states not found at {GEMMA_IC_HS_PATH}, skipping.")
        results["gemma"] = {"status": "not_found", "path": str(GEMMA_IC_HS_PATH)}

    return results


# ===================================================================
# 2b. Var/Fix Common BK Features (SAE, IC)
# ===================================================================
def analysis_2b_varfix_common_features() -> dict:
    """Find SAE features with Cohen's d >= 0.3 in BOTH Variable and Fixed (IC L22)."""
    log("\n" + "=" * 70)
    log("2b. VAR/FIX COMMON BK FEATURES (LLaMA IC SAE L22)")
    log("=" * 70)

    feat, meta = load_sae_dp("investment_choice", 22)
    labels = bk_labels(meta["game_outcomes"])
    bet_types = meta["bet_types"]

    var_mask = bet_types == "variable"
    fix_mask = bet_types == "fixed"

    log(f"  IC L22: Variable={var_mask.sum()} games ({labels[var_mask].sum()} BK), "
        f"Fixed={fix_mask.sum()} games ({labels[fix_mask].sum()} BK)")

    # Filter to active features (union of Variable and Fixed)
    var_rate = (feat[var_mask] != 0).mean(axis=0)
    fix_rate = (feat[fix_mask] != 0).mean(axis=0)
    active = (var_rate >= MIN_ACTIVATION_RATE) | (fix_rate >= MIN_ACTIVATION_RATE)
    active_idx = np.where(active)[0]
    n_active = len(active_idx)
    log(f"  Union-active features: {n_active}")

    feat_active = feat[:, active].astype(np.float64)

    # Cohen's d for Variable
    var_d = cohens_d_vec(feat_active[var_mask], labels[var_mask])
    fix_d = cohens_d_vec(feat_active[fix_mask], labels[fix_mask])

    # Common BK features: both |d| >= 0.3 AND same sign
    threshold = 0.3
    common = (
        (np.abs(var_d) >= threshold)
        & (np.abs(fix_d) >= threshold)
        & (np.sign(var_d) == np.sign(fix_d))
    )
    n_common = int(common.sum())

    common_idx = np.where(common)[0]
    promoting = int((var_d[common_idx] > 0).sum())
    inhibiting = n_common - promoting

    log(f"  Common BK features (|d|>=0.3 both, same sign): {n_common}")
    log(f"    Promoting: {promoting}, Inhibiting: {inhibiting}")

    # Top-20 by min|d|
    top_features = []
    if len(common_idx) > 0:
        min_abs_d = np.minimum(np.abs(var_d[common_idx]), np.abs(fix_d[common_idx]))
        order = np.argsort(min_abs_d)[-20:][::-1]
        for rank, oi in enumerate(order):
            fi = int(active_idx[common_idx[oi]])
            direction = "promoting" if var_d[common_idx[oi]] > 0 else "inhibiting"
            top_features.append({
                "rank": rank + 1,
                "feature_id": fi,
                "var_d": round(float(var_d[common_idx[oi]]), 4),
                "fix_d": round(float(fix_d[common_idx[oi]]), 4),
                "min_abs_d": round(float(min_abs_d[oi]), 4),
                "direction": direction,
            })

    return {
        "layer": 22,
        "n_active_features": n_active,
        "n_common_bk_features": n_common,
        "d_threshold": threshold,
        "promoting": promoting,
        "inhibiting": inhibiting,
        "top20": top_features,
        "n_var_games": int(var_mask.sum()),
        "n_fix_games": int(fix_mask.sum()),
        "n_var_bk": int(labels[var_mask].sum()),
        "n_fix_bk": int(labels[fix_mask].sum()),
    }


# ===================================================================
# 2c. Var/Fix BK Direction at SAE Level (IC)
# ===================================================================
def analysis_2c_varfix_direction_sae() -> dict:
    """Cosine between Variable BK direction and Fixed BK direction at SAE level."""
    log("\n" + "=" * 70)
    log("2c. VAR/FIX BK DIRECTION AT SAE LEVEL (LLaMA IC)")
    log("=" * 70)

    results = {}

    for layer in HS_LAYERS:
        log(f"  Loading IC SAE L{layer}...")
        feat, meta = load_sae_dp("investment_choice", layer)
        labels = bk_labels(meta["game_outcomes"])
        bet_types = meta["bet_types"]

        var_mask = bet_types == "variable"
        fix_mask = bet_types == "fixed"

        var_bk = var_mask & (labels == 1)
        var_safe = var_mask & (labels == 0)
        fix_bk = fix_mask & (labels == 1)
        fix_safe = fix_mask & (labels == 0)

        if var_bk.sum() < 5 or fix_bk.sum() < 5:
            log(f"  L{layer}: skipped (var_bk={var_bk.sum()}, fix_bk={fix_bk.sum()})")
            results[f"L{layer}"] = {"skipped": True}
            continue

        # Filter to active features
        feat_active, active_idx = filter_active(feat)
        n_active = len(active_idx)

        # BK direction for Variable: mean(Var BK) - mean(Var Safe)
        var_bk_dir = feat_active[var_bk].mean(axis=0) - feat_active[var_safe].mean(axis=0)
        fix_bk_dir = feat_active[fix_bk].mean(axis=0) - feat_active[fix_safe].mean(axis=0)

        cos_val = cosine(var_bk_dir.astype(np.float64), fix_bk_dir.astype(np.float64))

        results[f"L{layer}"] = {
            "cosine_var_fix_bk": round(cos_val, 4),
            "n_active_features": n_active,
            "n_var_bk": int(var_bk.sum()),
            "n_var_safe": int(var_safe.sum()),
            "n_fix_bk": int(fix_bk.sum()),
            "n_fix_safe": int(fix_safe.sum()),
        }
        log(f"  L{layer}: cos(Var_BK_dir, Fix_BK_dir) = {cos_val:.4f}, active={n_active}")

    return results


# ===================================================================
# 2d. Variable Anomaly Investigation
# ===================================================================
def analysis_2d_variable_anomaly() -> dict:
    """Cross-domain d-correlation for Variable games, with BK-rate-matched bootstrap."""
    log("\n" + "=" * 70)
    log("2d. VARIABLE ANOMALY INVESTIGATION (LLaMA IC+SM SAE L22)")
    log("=" * 70)

    # Load both paradigms at L22
    ic_feat, ic_meta = load_sae_dp("investment_choice", 22)
    sm_feat, sm_meta = load_sae_dp("slot_machine", 22)

    ic_labels = bk_labels(ic_meta["game_outcomes"])
    sm_labels = bk_labels(sm_meta["game_outcomes"])
    ic_bt = ic_meta["bet_types"]
    sm_bt = sm_meta["bet_types"]

    # Variable games only
    ic_var = ic_bt == "variable"
    sm_var = sm_bt == "variable"

    ic_v_feat = ic_feat[ic_var]
    sm_v_feat = sm_feat[sm_var]
    ic_v_labels = ic_labels[ic_var]
    sm_v_labels = sm_labels[sm_var]

    ic_bk_rate = float(ic_v_labels.mean())
    sm_bk_rate = float(sm_v_labels.mean())

    log(f"  IC Variable: {ic_var.sum()} games, BK rate={ic_bk_rate:.4f}")
    log(f"  SM Variable: {sm_var.sum()} games, BK rate={sm_bk_rate:.4f}")

    # Union-active features
    ic_rate = (ic_v_feat != 0).mean(axis=0)
    sm_rate = (sm_v_feat != 0).mean(axis=0)
    both_active = (ic_rate >= MIN_ACTIVATION_RATE) & (sm_rate >= MIN_ACTIVATION_RATE)
    both_idx = np.where(both_active)[0]
    n_both = len(both_idx)
    log(f"  Both-active features: {n_both}")

    ic_v_active = ic_v_feat[:, both_active].astype(np.float64)
    sm_v_active = sm_v_feat[:, both_active].astype(np.float64)

    # Raw d-correlation
    ic_d = cohens_d_vec(ic_v_active, ic_v_labels)
    sm_d = cohens_d_vec(sm_v_active, sm_v_labels)

    raw_corr, raw_p = pearsonr(ic_d, sm_d)
    log(f"  Raw d-correlation: r={raw_corr:.4f}, p={raw_p:.2e}")

    # Matched BK rate analysis: subsample SM Variable to match IC BK rate
    target_bk_rate = ic_bk_rate
    n_sm_var = len(sm_v_labels)
    n_sm_bk = sm_v_labels.sum()
    n_sm_safe = n_sm_var - n_sm_bk

    # Target number of BK games to achieve ic_bk_rate in SM subsample
    # We keep all safe games and subsample BK games:
    # target_bk_rate = n_target_bk / (n_target_bk + n_sm_safe)
    # n_target_bk = target_bk_rate * n_sm_safe / (1 - target_bk_rate)
    n_target_bk = int(np.round(target_bk_rate * n_sm_safe / (1 - target_bk_rate)))
    n_target_bk = max(1, min(n_target_bk, n_sm_bk))

    log(f"  BK rate matching: target_bk_rate={target_bk_rate:.4f}, "
        f"n_target_bk={n_target_bk} (from {n_sm_bk})")

    rng = np.random.RandomState(RANDOM_SEED)
    bk_indices = np.where(sm_v_labels == 1)[0]
    safe_indices = np.where(sm_v_labels == 0)[0]

    N_BOOTSTRAP = 50
    bootstrap_corrs = []

    for b in range(N_BOOTSTRAP):
        # Subsample BK games from SM Variable
        chosen_bk = rng.choice(bk_indices, size=n_target_bk, replace=False)
        subsample_idx = np.concatenate([chosen_bk, safe_indices])
        sub_feat = sm_v_active[subsample_idx]
        sub_labels = sm_v_labels[subsample_idx]

        sm_d_sub = cohens_d_vec(sub_feat, sub_labels)
        r_sub, _ = pearsonr(ic_d, sm_d_sub)
        bootstrap_corrs.append(r_sub)

    bootstrap_corrs = np.array(bootstrap_corrs)
    ci_low = float(np.percentile(bootstrap_corrs, 2.5))
    ci_high = float(np.percentile(bootstrap_corrs, 97.5))
    boot_mean = float(bootstrap_corrs.mean())
    boot_std = float(bootstrap_corrs.std())

    log(f"  Bootstrap d-corr (matched BK rate): mean={boot_mean:.4f}, "
        f"95% CI=[{ci_low:.4f}, {ci_high:.4f}]")

    # Additional diagnostics: sign consistency
    sign_consistent = ((np.sign(ic_d) == np.sign(sm_d)) & (ic_d != 0) & (sm_d != 0)).sum()
    total_nonzero = ((ic_d != 0) & (sm_d != 0)).sum()

    return {
        "layer": 22,
        "n_ic_variable": int(ic_var.sum()),
        "n_sm_variable": int(sm_var.sum()),
        "ic_bk_rate": round(ic_bk_rate, 4),
        "sm_bk_rate": round(sm_bk_rate, 4),
        "n_both_active_features": n_both,
        "raw_d_correlation": round(float(raw_corr), 4),
        "raw_d_correlation_p": float(raw_p),
        "matched_bk_rate_bootstrap": {
            "n_bootstrap": N_BOOTSTRAP,
            "target_bk_rate": round(target_bk_rate, 4),
            "n_target_bk": n_target_bk,
            "mean_d_corr": round(boot_mean, 4),
            "std_d_corr": round(boot_std, 4),
            "ci_95_low": round(ci_low, 4),
            "ci_95_high": round(ci_high, 4),
        },
        "sign_consistency": {
            "n_consistent": int(sign_consistent),
            "n_total_nonzero": int(total_nonzero),
            "pct_consistent": round(float(sign_consistent / max(total_nonzero, 1) * 100), 2),
        },
        "explanation": (
            "The raw d-correlation measures whether the same SAE features "
            "discriminate BK from Safe in both IC and SM for Variable-bet games. "
            "A negative or near-zero correlation suggests the bankruptcy mechanism "
            "operates through different features across paradigms. The bootstrap "
            "analysis controls for the large BK rate difference between IC (~9%) "
            "and SM (~36%) by subsampling SM BK games to match IC's rate."
        ),
    }


# ===================================================================
# Main
# ===================================================================
def main():
    log("=" * 70)
    log("V10 LLaMA SYMMETRIC ANALYSES (11 analyses)")
    log("=" * 70)

    t0 = time.time()
    all_results = {
        "metadata": {
            "script": "run_llama_v10_symmetric.py",
            "version": "v10",
            "timestamp": datetime.now().isoformat(),
            "random_seed": RANDOM_SEED,
            "n_permutations": N_PERMUTATIONS,
            "fdr_alpha": FDR_ALPHA,
            "hs_layers": HS_LAYERS,
            "model": "llama",
        }
    }

    # Phase 1: LLaMA Symmetric (7 analyses)
    log("\n>>> PHASE 1: LLaMA SYMMETRIC <<<\n")

    t = time.time()
    all_results["1a_universal_bk_neurons"] = analysis_1a_universal_bk_neurons()
    log(f"  -> 1a done in {time.time()-t:.1f}s\n")

    t = time.time()
    result_1b = analysis_1b_hs_transfer()
    all_results["1b_hs_cross_domain_transfer"] = result_1b
    log(f"  -> 1b done in {time.time()-t:.1f}s\n")

    t = time.time()
    all_results["1c_hs_vs_sae_transfer"] = analysis_1c_hs_sae_comparison(result_1b)
    log(f"  -> 1c done in {time.time()-t:.1f}s\n")

    t = time.time()
    all_results["1d_shared_bk_subspace_2d"] = analysis_1d_shared_subspace()
    log(f"  -> 1d done in {time.time()-t:.1f}s\n")

    t = time.time()
    all_results["1e_g_prompt_alignment"] = analysis_1e_g_prompt_alignment()
    log(f"  -> 1e done in {time.time()-t:.1f}s\n")

    t = time.time()
    all_results["1f_bet_constraint_mapping"] = analysis_1f_bet_constraint_mapping()
    log(f"  -> 1f done in {time.time()-t:.1f}s\n")

    t = time.time()
    all_results["1g_prompt_component_hs"] = analysis_1g_prompt_component_hs()
    log(f"  -> 1g done in {time.time()-t:.1f}s\n")

    # Phase 2: Both-Model Reinforcement (4 analyses)
    log("\n>>> PHASE 2: BOTH-MODEL REINFORCEMENT <<<\n")

    t = time.time()
    all_results["2a_shared_bk_profile"] = analysis_2a_shared_bk_profile()
    log(f"  -> 2a done in {time.time()-t:.1f}s\n")

    t = time.time()
    all_results["2b_varfix_common_features"] = analysis_2b_varfix_common_features()
    log(f"  -> 2b done in {time.time()-t:.1f}s\n")

    t = time.time()
    all_results["2c_varfix_direction_sae"] = analysis_2c_varfix_direction_sae()
    log(f"  -> 2c done in {time.time()-t:.1f}s\n")

    t = time.time()
    all_results["2d_variable_anomaly"] = analysis_2d_variable_anomaly()
    log(f"  -> 2d done in {time.time()-t:.1f}s\n")

    # Save
    total_time = time.time() - t0
    all_results["metadata"]["total_runtime_seconds"] = round(total_time, 1)

    out_file = JSON_DIR / f"llama_v10_symmetric_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2, default=numpy_convert)

    log("=" * 70)
    log(f"ALL 11 ANALYSES COMPLETE in {total_time:.1f}s")
    log(f"Results saved to {out_file}")
    log("=" * 70)


if __name__ == "__main__":
    main()
