#!/usr/bin/env python3
"""
LLaMA IC V2role SAE Analysis — symmetric with Gemma analyses.

Analyses:
1. BK Classification (all 32 layers, DP + R1 + balance-matched)
2. Round-level risk choice classification
3. Condition analysis (bet_constraint, bet_type, prompt)
4. Early BK prediction (R1-R10)
5. Feature importance and overlap with Gemma

Uses the V3 sparse COO format from extract_llama_ic.py output.
"""

import gc
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ===================================================================
# Config
# ===================================================================
LLAMA_SAE_DIR = Path("/home/jovyan/beomi/llm-addiction-data/sae_features_v3/investment_choice/llama")
GEMMA_SAE_DIR = Path("/home/jovyan/beomi/llm-addiction-data/sae_features_v3/investment_choice/gemma")
RESULTS_DIR = Path("/home/jovyan/llm-addiction/sae_v3_analysis/results")
FIGURE_DIR = RESULTS_DIR / "figures"
JSON_DIR = RESULTS_DIR / "json"
LOG_DIR = RESULTS_DIR / "logs"

N_LAYERS_LLAMA = 32
N_FEATURES_LLAMA = 32768
N_LAYERS_GEMMA = 42
N_FEATURES_GEMMA = 131072
RANDOM_SEED = 42
CV_FOLDS = 5
MIN_ACT_RATE = 0.01
N_PERMUTATIONS = 1000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / f"llama_ic_analyses_{datetime.now():%Y%m%d_%H%M%S}.log"),
    ],
)
logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


# ===================================================================
# Data loading
# ===================================================================
def load_layer(sae_dir: Path, layer: int):
    """Load sparse COO and return dense features + metadata."""
    path = sae_dir / f"sae_features_L{layer}.npz"
    if not path.exists():
        return None, None
    data = np.load(path, allow_pickle=False)
    shape = tuple(data["shape"])
    dense = np.zeros(shape, dtype=np.float32)
    dense[data["row_indices"], data["col_indices"]] = data["values"]
    meta = {
        "game_ids": data["game_ids"],
        "round_nums": data["round_nums"],
        "game_outcomes": data["game_outcomes"],
        "is_last_round": data["is_last_round"].astype(bool),
        "bet_types": data["bet_types"],
    }
    for field in ["bet_constraints", "prompt_conditions", "balances"]:
        if field in data:
            meta[field] = data[field]
    return dense, meta


def filter_active(X, min_rate=MIN_ACT_RATE):
    """Keep features active in >= min_rate fraction of samples."""
    active = (X != 0).mean(axis=0) >= min_rate
    return X[:, active], active


def get_dp_data(X, meta):
    """Extract decision-point (last round) data."""
    mask = meta["is_last_round"]
    return X[mask], {k: v[mask] for k, v in meta.items()}


def get_r1_data(X, meta):
    """Extract Round 1 data."""
    mask = meta["round_nums"] == 1
    return X[mask], {k: v[mask] for k, v in meta.items()}


def get_labels(meta, label_type="bk"):
    """Get binary labels from metadata."""
    if label_type == "bk":
        return (meta["game_outcomes"] == "bankruptcy").astype(int)
    elif label_type == "risky":
        # For IC: risky = choices 3,4 (high/very high risk)
        # This would need choice data which we don't have in the round records
        raise NotImplementedError("Need choice data for risky label")
    return None


# ===================================================================
# Classification utilities
# ===================================================================
def classify_cv(X, y, n_folds=CV_FOLDS, seed=RANDOM_SEED):
    """5-fold stratified CV with StandardScaler."""
    if len(np.unique(y)) < 2 or np.sum(y) < 5:
        return {"auc": float("nan"), "auc_std": float("nan"), "f1": float("nan"), "skipped": True}

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    aucs, f1s = [], []
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        clf = LogisticRegression(
            max_iter=1000, solver="lbfgs", class_weight="balanced",
            random_state=seed, C=1.0,
        )
        clf.fit(X_train, y[train_idx])
        proba = clf.predict_proba(X_test)[:, 1]
        preds = clf.predict(X_test)
        aucs.append(roc_auc_score(y[test_idx], proba))
        f1s.append(f1_score(y[test_idx], preds))

    return {
        "auc": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "f1": float(np.mean(f1s)),
        "skipped": False,
    }


def permutation_test(X, y, observed_auc, n_perm=N_PERMUTATIONS, seed=RANDOM_SEED):
    """Permutation test for AUC significance."""
    rng = np.random.RandomState(seed)
    null_aucs = []
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        result = classify_cv(X, y_perm, seed=seed)
        if not result["skipped"]:
            null_aucs.append(result["auc"])
    null_aucs = np.array(null_aucs)
    p_value = (np.sum(null_aucs >= observed_auc) + 1) / (len(null_aucs) + 1)
    return {
        "p_value": float(p_value),
        "null_mean": float(np.mean(null_aucs)),
        "null_std": float(np.std(null_aucs)),
        "n_perm": len(null_aucs),
    }


# ===================================================================
# Analysis 1: BK Classification (all layers, DP + R1 + BM)
# ===================================================================
def run_bk_classification(layers_to_run=None):
    """BK classification across all layers with DP, R1, and balance-matched."""
    logger.info("=" * 60)
    logger.info("ANALYSIS 1: BK Classification")
    logger.info("=" * 60)

    if layers_to_run is None:
        layers_to_run = list(range(N_LAYERS_LLAMA))

    results = {"dp": [], "r1": [], "bm": []}

    for layer in layers_to_run:
        logger.info(f"Layer {layer}...")
        X_all, meta_all = load_layer(LLAMA_SAE_DIR, layer)
        if X_all is None:
            continue

        # Decision-point
        X_dp, meta_dp = get_dp_data(X_all, meta_all)
        X_dp_f, _ = filter_active(X_dp)
        y_dp = get_labels(meta_dp)
        dp_result = classify_cv(X_dp_f, y_dp)
        dp_result.update({"layer": layer, "n_pos": int(y_dp.sum()), "n_neg": int(len(y_dp) - y_dp.sum()),
                          "n_features": X_dp_f.shape[1], "mode": "decision_point"})
        results["dp"].append(dp_result)
        logger.info(f"  DP: AUC={dp_result['auc']:.3f}, F1={dp_result['f1']:.3f}")

        # Round 1
        X_r1, meta_r1 = get_r1_data(X_all, meta_all)
        X_r1_f, _ = filter_active(X_r1)
        y_r1 = get_labels(meta_r1)
        r1_result = classify_cv(X_r1_f, y_r1)
        r1_result.update({"layer": layer, "n_pos": int(y_r1.sum()), "n_neg": int(len(y_r1) - y_r1.sum()),
                          "n_features": X_r1_f.shape[1], "mode": "r1"})
        results["r1"].append(r1_result)
        logger.info(f"  R1: AUC={r1_result['auc']:.3f}")

        # Balance-matched
        if "balances" in meta_dp:
            balances = meta_dp["balances"].astype(float)
            bk_idx = np.where(y_dp == 1)[0]
            safe_idx = np.where(y_dp == 0)[0]
            matched_safe = []
            used = set()
            for bi in bk_idx:
                bk_bal = balances[bi]
                diffs = np.abs(balances[safe_idx] - bk_bal)
                for si in np.argsort(diffs):
                    if safe_idx[si] not in used:
                        matched_safe.append(safe_idx[si])
                        used.add(safe_idx[si])
                        break
            if len(matched_safe) == len(bk_idx):
                bm_idx = np.concatenate([bk_idx, np.array(matched_safe)])
                X_bm = X_dp_f[bm_idx]
                y_bm = y_dp[bm_idx]
                bm_result = classify_cv(X_bm, y_bm)
                bm_result.update({"layer": layer, "n_pos": int(y_bm.sum()), "n_neg": int(len(y_bm) - y_bm.sum()),
                                  "mode": "balance_matched"})
                results["bm"].append(bm_result)
                logger.info(f"  BM: AUC={bm_result['auc']:.3f}")

        del X_all, meta_all
        gc.collect()

    return results


# ===================================================================
# Analysis 2: Condition Analysis
# ===================================================================
def run_condition_analysis(best_layer=None):
    """Analyze condition effects on BK rate and SAE classification."""
    logger.info("=" * 60)
    logger.info("ANALYSIS 2: Condition Analysis")
    logger.info("=" * 60)

    # Find best layer from BK classification if not provided
    if best_layer is None:
        best_layer = 9  # Default from previous V4 analysis

    X_all, meta_all = load_layer(LLAMA_SAE_DIR, best_layer)
    X_dp, meta_dp = get_dp_data(X_all, meta_all)
    X_dp_f, _ = filter_active(X_dp)
    y_dp = get_labels(meta_dp)

    result = {
        "layer": best_layer,
        "n_total": len(y_dp),
        "n_bk": int(y_dp.sum()),
        "bk_rate": float(y_dp.mean()),
    }

    # Bet type analysis
    if "bet_types" in meta_dp:
        for bt in ["fixed", "variable"]:
            mask = meta_dp["bet_types"] == bt
            n = int(mask.sum())
            n_bk = int(y_dp[mask].sum())
            auc_result = classify_cv(X_dp_f[mask], y_dp[mask]) if n_bk >= 5 else {"auc": float("nan"), "skipped": True}
            result[f"bet_{bt}"] = {
                "n": n, "n_bk": n_bk, "bk_rate": n_bk / n if n > 0 else 0,
                "auc": auc_result.get("auc", float("nan")),
            }
        # Fisher's exact
        from scipy.stats import fisher_exact
        a = result["bet_fixed"]["n_bk"]
        b = result["bet_fixed"]["n"] - a
        c = result["bet_variable"]["n_bk"]
        d = result["bet_variable"]["n"] - c
        _, p = fisher_exact([[a, b], [c, d]])
        result["bet_type_fisher_p"] = float(p)

    # Bet constraint analysis
    if "bet_constraints" in meta_dp:
        constraints = sorted(set(meta_dp["bet_constraints"]))
        result["bet_constraint"] = {}
        for bc in constraints:
            mask = meta_dp["bet_constraints"] == bc
            n = int(mask.sum())
            n_bk = int(y_dp[mask].sum())
            result["bet_constraint"][str(bc)] = {
                "n": n, "n_bk": n_bk, "bk_rate": n_bk / n if n > 0 else 0,
            }

        # Bet constraint 4-class classification across layers
        result["constraint_classification"] = []
        for layer in range(0, N_LAYERS_LLAMA, 2):
            X_l, meta_l = load_layer(LLAMA_SAE_DIR, layer)
            X_l_dp, meta_l_dp = get_dp_data(X_l, meta_l)
            X_l_f, _ = filter_active(X_l_dp)
            y_constraint = meta_l_dp["bet_constraints"]
            # Encode as integers
            constraint_map = {c: i for i, c in enumerate(sorted(set(y_constraint)))}
            y_c = np.array([constraint_map[c] for c in y_constraint])
            if len(constraint_map) >= 2:
                auc_result = classify_cv_multiclass(X_l_f, y_c)
                result["constraint_classification"].append({
                    "layer": layer, "auc": auc_result["auc"],
                    "auc_std": auc_result.get("auc_std", 0),
                    "n_classes": len(constraint_map),
                })
                logger.info(f"  Constraint L{layer}: AUC={auc_result['auc']:.3f}")
            del X_l, meta_l
            gc.collect()

    # Prompt condition analysis
    if "prompt_conditions" in meta_dp:
        conditions = sorted(set(meta_dp["prompt_conditions"]))
        result["prompt_conditions"] = {}
        for pc in conditions:
            mask = meta_dp["prompt_conditions"] == pc
            n = int(mask.sum())
            n_bk = int(y_dp[mask].sum())
            result["prompt_conditions"][pc] = {
                "n": n, "n_bk": n_bk, "bk_rate": n_bk / n if n > 0 else 0,
            }

        # Component marginal effects (G, M)
        result["component_marginal"] = {}
        for comp in ["G", "M"]:
            with_mask = np.array([comp in pc for pc in meta_dp["prompt_conditions"]])
            without_mask = ~with_mask
            n_with = int(with_mask.sum())
            n_without = int(without_mask.sum())
            bk_with = int(y_dp[with_mask].sum())
            bk_without = int(y_dp[without_mask].sum())
            rate_with = bk_with / n_with if n_with > 0 else 0
            rate_without = bk_without / n_without if n_without > 0 else 0

            from scipy.stats import fisher_exact
            _, p = fisher_exact([[bk_with, n_with - bk_with], [bk_without, n_without - bk_without]])
            result["component_marginal"][comp] = {
                "n_with": n_with, "n_without": n_without,
                "bk_with": bk_with, "bk_without": bk_without,
                "bk_rate_with": rate_with, "bk_rate_without": rate_without,
                "bk_rate_diff": rate_with - rate_without,
                "fisher_p": float(p),
            }
            logger.info(f"  {comp}: rate_with={rate_with:.4f}, rate_without={rate_without:.4f}, "
                        f"diff={rate_with - rate_without:.4f}, p={p:.4f}")

    del X_all, meta_all
    gc.collect()
    return result


def classify_cv_multiclass(X, y, n_folds=CV_FOLDS, seed=RANDOM_SEED):
    """Multiclass classification with OVR AUC."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    aucs = []
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        clf = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed, C=1.0)
        clf.fit(X_train, y[train_idx])
        proba = clf.predict_proba(X_test)
        aucs.append(roc_auc_score(y[test_idx], proba, multi_class="ovr", average="macro"))
    return {"auc": float(np.mean(aucs)), "auc_std": float(np.std(aucs))}


# ===================================================================
# Analysis 3: Early BK Prediction
# ===================================================================
def run_early_prediction(layers=None):
    """Predict BK from early rounds (R1-R10)."""
    logger.info("=" * 60)
    logger.info("ANALYSIS 3: Early BK Prediction")
    logger.info("=" * 60)

    if layers is None:
        layers = [0, 5, 9, 15, 22, 31]

    results = []
    for layer in layers:
        X_all, meta_all = load_layer(LLAMA_SAE_DIR, layer)
        if X_all is None:
            continue

        layer_result = {"layer": layer, "rounds": []}
        for round_n in range(1, 16):
            mask = meta_all["round_nums"] == round_n
            if mask.sum() < 20:
                break
            X_r = X_all[mask]
            X_r_f, _ = filter_active(X_r)
            y_r = get_labels({k: v[mask] for k, v in meta_all.items()})
            n_bk = int(y_r.sum())
            n_safe = int(len(y_r) - y_r.sum())
            if n_bk < 5:
                break
            result = classify_cv(X_r_f, y_r)
            layer_result["rounds"].append({
                "round": round_n,
                "n_games": int(mask.sum()),
                "n_bk": n_bk,
                "n_safe": n_safe,
                "auc_mean": result["auc"],
                "auc_std": result.get("auc_std", 0),
                "n_features": X_r_f.shape[1],
            })
            logger.info(f"  L{layer} R{round_n}: AUC={result['auc']:.3f} ({n_bk} BK / {n_safe} safe)")

        results.append(layer_result)
        del X_all, meta_all
        gc.collect()

    return results


# ===================================================================
# Analysis 4: R1 Permutation Test
# ===================================================================
def run_r1_permutation(layers=None, n_perm=100):
    """Permutation test for R1 BK classification."""
    logger.info("=" * 60)
    logger.info("ANALYSIS 4: R1 Permutation Test")
    logger.info("=" * 60)

    if layers is None:
        layers = [0, 9, 15, 22]

    results = {}
    for layer in layers:
        X_all, meta_all = load_layer(LLAMA_SAE_DIR, layer)
        if X_all is None:
            continue
        X_r1, meta_r1 = get_r1_data(X_all, meta_all)
        X_r1_f, _ = filter_active(X_r1)
        y_r1 = get_labels(meta_r1)

        observed = classify_cv(X_r1_f, y_r1)
        perm_result = permutation_test(X_r1_f, y_r1, observed["auc"], n_perm=n_perm)
        perm_result["observed_auc"] = observed["auc"]
        perm_result["observed_auc_std"] = observed.get("auc_std", 0)
        perm_result["layer"] = layer
        results[f"L{layer}"] = perm_result
        logger.info(f"  L{layer}: AUC={observed['auc']:.3f}, p={perm_result['p_value']:.4f}")

        del X_all, meta_all
        gc.collect()

    return results


# ===================================================================
# Analysis 5: Cross-Model Feature Comparison (Gemma vs LLaMA at shared equivalent layers)
# ===================================================================
def run_cross_model_comparison():
    """Compare Gemma and LLaMA IC BK classification layer profiles."""
    logger.info("=" * 60)
    logger.info("ANALYSIS 5: Cross-Model BK Classification Profiles")
    logger.info("=" * 60)

    gemma_results = []
    llama_results = []

    # Gemma: every other layer (21 layers)
    for layer in range(0, N_LAYERS_GEMMA, 2):
        X_all, meta_all = load_layer(GEMMA_SAE_DIR, layer)
        if X_all is None:
            continue
        X_dp, meta_dp = get_dp_data(X_all, meta_all)
        X_dp_f, _ = filter_active(X_dp)
        y_dp = get_labels(meta_dp)
        result = classify_cv(X_dp_f, y_dp)
        result.update({"layer": layer, "model": "gemma", "n_pos": int(y_dp.sum()), "n_neg": int(len(y_dp) - y_dp.sum())})
        gemma_results.append(result)
        logger.info(f"  Gemma L{layer}: AUC={result['auc']:.3f}")
        del X_all, meta_all
        gc.collect()

    # LLaMA: all 32 layers
    for layer in range(N_LAYERS_LLAMA):
        X_all, meta_all = load_layer(LLAMA_SAE_DIR, layer)
        if X_all is None:
            continue
        X_dp, meta_dp = get_dp_data(X_all, meta_all)
        X_dp_f, _ = filter_active(X_dp)
        y_dp = get_labels(meta_dp)
        result = classify_cv(X_dp_f, y_dp)
        result.update({"layer": layer, "model": "llama", "n_pos": int(y_dp.sum()), "n_neg": int(len(y_dp) - y_dp.sum())})
        llama_results.append(result)
        logger.info(f"  LLaMA L{layer}: AUC={result['auc']:.3f}")
        del X_all, meta_all
        gc.collect()

    return {"gemma": gemma_results, "llama": llama_results}


# ===================================================================
# Plotting
# ===================================================================
def plot_results(bk_results, condition_results, early_results, cross_model_results):
    """Generate publication-quality figures."""

    # Figure 1: BK Classification (DP, R1, BM) across layers
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, mode, title in zip(axes, ["dp", "r1", "bm"],
                                ["Decision-Point", "Round 1 (Balance-Controlled)", "Balance-Matched"]):
        data = bk_results[mode]
        if data:
            layers = [d["layer"] for d in data]
            aucs = [d["auc"] for d in data]
            ax.plot(layers, aucs, "o-", color="#D55E00", label="LLaMA IC")
            ax.set_xlabel("Layer")
            ax.set_ylabel("AUC")
            ax.set_title(f"LLaMA IC: {title}")
            ax.set_ylim(0.4, 1.0)
            ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
            ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "llama_ic_bk_classification.png", dpi=150)
    plt.close()

    # Figure 2: Cross-model comparison
    if cross_model_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        gemma = cross_model_results["gemma"]
        llama = cross_model_results["llama"]
        if gemma:
            g_layers = [d["layer"] for d in gemma]
            g_aucs = [d["auc"] for d in gemma]
            ax.plot(g_layers, g_aucs, "o-", color="#0072B2", label="Gemma-2-9B (131K SAE)", markersize=4)
        if llama:
            l_layers = [d["layer"] for d in llama]
            l_aucs = [d["auc"] for d in llama]
            ax.plot(l_layers, l_aucs, "s-", color="#D55E00", label="LLaMA-3.1-8B (32K SAE)", markersize=4)
        ax.set_xlabel("Layer", fontsize=13)
        ax.set_ylabel("AUC", fontsize=13)
        ax.set_title("Cross-Model BK Classification: IC V2role", fontsize=14)
        ax.set_ylim(0.85, 1.0)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "llama_vs_gemma_ic_bk.png", dpi=150)
        plt.close()

    # Figure 3: Early prediction
    if early_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#D55E00", "#0072B2", "#009E73", "#CC79A7", "#F0E442", "#56B4E9"]
        for i, lr in enumerate(early_results):
            rounds = [r["round"] for r in lr["rounds"]]
            aucs = [r["auc_mean"] for r in lr["rounds"]]
            ax.plot(rounds, aucs, "o-", color=colors[i % len(colors)],
                    label=f"L{lr['layer']}", markersize=5)
        ax.set_xlabel("Round", fontsize=13)
        ax.set_ylabel("AUC", fontsize=13)
        ax.set_title("LLaMA IC: Early BK Prediction by Round", fontsize=14)
        ax.set_ylim(0.4, 1.0)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "llama_ic_early_prediction.png", dpi=150)
        plt.close()

    logger.info("Figures saved.")


# ===================================================================
# Main
# ===================================================================
def main():
    np.random.seed(RANDOM_SEED)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("=" * 60)
    logger.info("LLaMA IC V2role SAE Analysis")
    logger.info(f"SAE dir: {LLAMA_SAE_DIR}")
    logger.info(f"Timestamp: {timestamp}")
    logger.info("=" * 60)

    all_results = {"timestamp": timestamp}

    # 1. BK Classification
    bk_results = run_bk_classification()
    all_results["bk_classification"] = bk_results

    # Find best DP layer
    dp_best = max(bk_results["dp"], key=lambda x: x.get("auc", 0))
    best_layer = dp_best["layer"]
    logger.info(f"Best DP layer: L{best_layer} (AUC={dp_best['auc']:.3f})")

    # 2. Condition Analysis
    condition_results = run_condition_analysis(best_layer=best_layer)
    all_results["condition_analysis"] = condition_results

    # 3. Early BK Prediction
    early_results = run_early_prediction()
    all_results["early_prediction"] = early_results

    # 4. R1 Permutation Test (reduced to 100 for speed)
    perm_results = run_r1_permutation(layers=[best_layer], n_perm=100)
    all_results["r1_permutation"] = perm_results

    # 5. Cross-model comparison
    cross_results = run_cross_model_comparison()
    all_results["cross_model"] = cross_results

    # Save results
    out_path = JSON_DIR / f"llama_ic_analyses_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    logger.info(f"Results saved: {out_path}")

    # Generate figures
    plot_results(bk_results, condition_results, early_results, cross_results)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"BK Classification (DP): Best L{best_layer}, AUC={dp_best['auc']:.3f}")
    r1_best = max(bk_results["r1"], key=lambda x: x.get("auc", 0))
    logger.info(f"BK Classification (R1): Best L{r1_best['layer']}, AUC={r1_best['auc']:.3f}")
    if bk_results["bm"]:
        bm_best = max(bk_results["bm"], key=lambda x: x.get("auc", 0))
        logger.info(f"BK Classification (BM): Best L{bm_best['layer']}, AUC={bm_best['auc']:.3f}")


if __name__ == "__main__":
    main()
