#!/usr/bin/env python3
"""
Track C: Apply B1+B2 analyses to Llama IC data.
Compare with Gemma IC results for cross-model evidence.
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats
from scipy.stats import ttest_ind, pointbiserialr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from config import *

np.random.seed(RANDOM_SEED)

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_llama_layer(layer):
    """Load Llama IC SAE features for a layer."""
    sae_dir = LLAMA_PARADIGMS["ic"]["sae_dir"]
    npz_path = sae_dir / f"sae_features_L{layer}.npz"
    if not npz_path.exists():
        return None
    data = np.load(npz_path, allow_pickle=False)
    raw = {k: data[k] for k in data.files}
    return raw


def load_llama_features_dp(layer):
    """Load Llama IC decision-point features."""
    raw = load_llama_layer(layer)
    if raw is None:
        return None, None

    # Reconstruct dense from sparse COO
    shape = tuple(raw["shape"])
    dense = np.zeros(shape, dtype=np.float32)
    dense[raw["row_indices"], raw["col_indices"]] = raw["values"]

    meta = {
        "game_ids": raw["game_ids"],
        "round_nums": raw["round_nums"],
        "game_outcomes": raw["game_outcomes"],
        "is_last_round": raw["is_last_round"].astype(bool),
        "bet_types": raw["bet_types"],
    }
    for field in ["bet_constraints", "prompt_conditions", "balances"]:
        if field in raw:
            meta[field] = raw[field]

    # Decision point = last round
    mask = meta["is_last_round"]
    feats = dense[mask]
    filtered_meta = {k: v[mask] for k, v in meta.items()}
    return feats, filtered_meta


def load_llama_hidden_states_dp(layer):
    """Load Llama IC hidden states at decision point."""
    sae_dir = LLAMA_PARADIGMS["ic"]["sae_dir"]
    ckpt = sae_dir / "checkpoints" / "phase_a_hidden_states.npz"
    if not ckpt.exists():
        return None, None

    data = np.load(ckpt, allow_pickle=False)
    hidden_all = data["hidden_states"]  # (n_rounds, n_layers, hidden_dim)

    # Get metadata from any SAE feature file
    any_sae = list(sae_dir.glob("sae_features_L*.npz"))
    if not any_sae:
        return None, None
    data_ = np.load(any_sae[0], allow_pickle=False)
    raw = {k: data_[k] for k in data_.files}
    meta = {
        "game_ids": raw["game_ids"],
        "round_nums": raw["round_nums"],
        "game_outcomes": raw["game_outcomes"],
        "is_last_round": raw["is_last_round"].astype(bool),
        "bet_types": raw["bet_types"],
    }
    for field in ["bet_constraints", "prompt_conditions", "balances"]:
        if field in raw:
            meta[field] = raw[field]

    features = hidden_all[:, layer, :]

    if "valid_mask" in data:
        valid = data["valid_mask"].astype(bool)
        features = features[valid]
        meta = {k: v[valid] for k, v in meta.items()}

    mask = meta["is_last_round"]
    return features[mask], {k: v[mask] for k, v in meta.items()}


def get_labels(meta):
    return (meta["game_outcomes"] == "bankruptcy").astype(np.int32)


def llama_b1_sae_crossdomain():
    """B1 for Llama: SAE feature statistics per layer (IC only — no cross-domain yet)."""
    log("=== Llama IC: SAE feature BK differential ===")

    results = {}
    for layer in range(0, 32):
        feats, meta = load_llama_features_dp(layer)
        if feats is None:
            continue
        labels = get_labels(meta)
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

        # Significant features
        strong = active & (np.abs(cohens_d) >= 0.3)
        medium = active & (np.abs(cohens_d) >= 0.2)

        bk_promoting = active & (cohens_d >= 0.3)
        bk_inhibiting = active & (cohens_d <= -0.3)

        results[layer] = {
            "n_active": int(n_active),
            "n_strong_d03": int(strong.sum()),
            "n_medium_d02": int(medium.sum()),
            "n_bk_promoting": int(bk_promoting.sum()),
            "n_bk_inhibiting": int(bk_inhibiting.sum()),
            "n_bk": int(n_bk),
        }

        if layer % 5 == 0 or strong.sum() > 50:
            log(f"  L{layer}: active={n_active}, strong(d≥0.3)={strong.sum()} (BK+:{bk_promoting.sum()}, BK-:{bk_inhibiting.sum()}), BK={n_bk}")

    # Summary
    total_strong = sum(r["n_strong_d03"] for r in results.values())
    peak_layer = max(results.items(), key=lambda x: x[1]["n_strong_d03"])
    log(f"\n  SUMMARY: Total strong features across 32 layers: {total_strong}")
    log(f"  Peak layer: L{peak_layer[0]} ({peak_layer[1]['n_strong_d03']} strong features)")

    return results


def llama_b1_classification():
    """BK classification AUC per layer for Llama IC."""
    log("\n=== Llama IC: BK Classification (SAE features) ===")

    results = {}
    for layer in range(0, 32):
        feats, meta = load_llama_features_dp(layer)
        if feats is None:
            continue
        labels = get_labels(meta)
        n_bk = labels.sum()
        if n_bk < 10:
            continue

        # Filter active
        active = (feats != 0).mean(axis=0) >= 0.01
        feats_active = feats[:, active]
        if feats_active.shape[1] < 10:
            continue

        try:
            scaler = StandardScaler()
            X = scaler.fit_transform(feats_active)
            n_comp = min(50, X.shape[0] - 1, X.shape[1])
            pca = PCA(n_components=n_comp)
            X_pca = pca.fit_transform(X)

            clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=RANDOM_SEED)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
            scores = cross_val_score(clf, X_pca, labels, cv=cv, scoring='roc_auc')
            auc = scores.mean()

            results[layer] = {"auc": float(auc), "n_features": int(feats_active.shape[1])}

            if layer % 5 == 0 or auc > 0.9:
                log(f"  L{layer}: AUC={auc:.3f} ({feats_active.shape[1]} features)")
        except Exception as e:
            log(f"  L{layer}: error - {e}")

    if results:
        best = max(results.items(), key=lambda x: x[1]["auc"])
        log(f"\n  Best layer: L{best[0]} AUC={best[1]['auc']:.3f}")

    return results


def llama_b2_interaction():
    """B2-b for Llama: Interaction regression on SAE features."""
    log("\n=== Llama IC: Interaction regression (SAE features) ===")

    import statsmodels.api as sm

    # Use best layer from Llama
    for layer in [8, 10, 15, 20, 25, 30]:
        feats, meta = load_llama_features_dp(layer)
        if feats is None:
            continue
        labels = get_labels(meta)
        bet_types = (meta["bet_types"] == "variable").astype(float)
        n_bk = labels.sum()
        if n_bk < 10:
            continue

        active = (feats != 0).mean(axis=0) >= 0.01
        n_active = active.sum()
        if n_active < 10:
            continue

        outcome_sig = 0
        bettype_sig = 0
        interaction_sig = 0
        shared_bk = 0

        for fi_idx in range(n_active):
            fi = np.where(active)[0][fi_idx]
            y = feats[:, fi]
            X = np.column_stack([labels, bet_types, labels * bet_types])
            X = sm.add_constant(X)
            try:
                model = sm.OLS(y, X).fit()
                if model.pvalues[1] < 0.01:
                    outcome_sig += 1
                if model.pvalues[2] < 0.01:
                    bettype_sig += 1
                if model.pvalues[3] < 0.01:
                    interaction_sig += 1
                if model.pvalues[1] < 0.01 and model.pvalues[3] > 0.05:
                    shared_bk += 1
            except:
                continue

        log(f"  Llama L{layer}: outcome_sig={outcome_sig}/{n_active}, bettype_sig={bettype_sig}, interaction={interaction_sig}, shared_BK={shared_bk}")

    return {}


def llama_r1_classification():
    """R1 within-bet-type classification for Llama IC."""
    log("\n=== Llama IC: Round 1 BK Classification ===")

    results = {}
    layer = 22  # Use similar depth as Gemma analysis

    for layer in [10, 15, 20, 25, 30]:
        # Load all rounds
        raw = load_llama_layer(layer)
        if raw is None:
            continue

        shape = tuple(raw["shape"])
        dense = np.zeros(shape, dtype=np.float32)
        dense[raw["row_indices"], raw["col_indices"]] = raw["values"]

        round_nums = raw["round_nums"]
        r1_mask = round_nums == 1
        if r1_mask.sum() < 20:
            r1_mask = round_nums == 0

        if r1_mask.sum() < 20:
            continue

        feats_r1 = dense[r1_mask]
        labels_r1 = (raw["game_outcomes"][r1_mask] == "bankruptcy").astype(np.int32)
        bet_types_r1 = raw["bet_types"][r1_mask]
        n_bk = labels_r1.sum()

        if n_bk < 5:
            continue

        log(f"  L{layer} R1: {len(labels_r1)} games, {n_bk} BK")

        for bt in ["fixed", "variable"]:
            bt_mask = bet_types_r1 == bt
            n_bt = bt_mask.sum()
            if n_bt < 20:
                continue

            X_bt = feats_r1[bt_mask]
            y_bt = labels_r1[bt_mask]
            n_bk_bt = y_bt.sum()

            if n_bk_bt < 3 or n_bk_bt == len(y_bt):
                continue

            active = (X_bt != 0).mean(axis=0) >= 0.01
            X_active = X_bt[:, active]
            if X_active.shape[1] < 5:
                continue

            try:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_active)
                n_comp = min(50, X_scaled.shape[0] - 1, X_scaled.shape[1])
                pca = PCA(n_components=n_comp)
                X_pca = pca.fit_transform(X_scaled)

                clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=RANDOM_SEED)
                n_folds = min(5, n_bk_bt)
                cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
                scores = cross_val_score(clf, X_pca, y_bt, cv=cv, scoring='roc_auc')
                auc = scores.mean()

                results[f"L{layer}_{bt}"] = {"auc": float(auc), "n": int(n_bt), "n_bk": int(n_bk_bt)}
                log(f"    {bt}: AUC={auc:.3f} (n={n_bt}, BK={n_bk_bt})")
            except Exception as e:
                log(f"    {bt}: error - {e}")

    return results


def main():
    log("=" * 70)
    log("TRACK C: LLAMA IC ANALYSIS")
    log("=" * 70)

    # Check data
    sae_dir = LLAMA_PARADIGMS["ic"]["sae_dir"]
    n_layers = len(list(sae_dir.glob("sae_features_L*.npz")))
    ckpt = sae_dir / "checkpoints" / "phase_a_hidden_states.npz"
    log(f"Llama IC: {n_layers} SAE layers, hidden_states={'YES' if ckpt.exists() else 'NO'}")

    # Quick peek at data shape
    raw = load_llama_layer(0)
    if raw:
        log(f"  Keys: {list(raw.keys())}")
        log(f"  Shape: {tuple(raw['shape'])}")
        labels = (raw["game_outcomes"][raw["is_last_round"].astype(bool)] == "bankruptcy")
        log(f"  BK games (DP): {labels.sum()}")

    all_results = {"timestamp": datetime.now().isoformat(), "model": "llama"}

    all_results["b1_sae_features"] = llama_b1_sae_crossdomain()
    all_results["b1_classification"] = llama_b1_classification()
    all_results["b2_interaction"] = llama_b2_interaction()
    all_results["r1_classification"] = llama_r1_classification()

    # Save
    out_file = JSON_DIR / f"llama_ic_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, set): return list(obj)
        return obj

    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=convert)

    log(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
