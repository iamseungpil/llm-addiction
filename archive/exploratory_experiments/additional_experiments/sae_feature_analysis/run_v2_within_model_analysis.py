#!/usr/bin/env python3
"""
V2 Within-Model SAE Analysis: Classification + Two-Way ANOVA + Improved Figures
================================================================================

Uses DECISION-POINT-FIXED V2 SAE features:
- LLaMA: fnlp SAE direct loading (NOT sae_lens) — verified match with original
- Gemma: sae_lens (same as original)
- Bankruptcy games: use pre-bankruptcy balance (not $0)

Data source: /home/jovyan/beomi/llm-addiction-data/sae_features_v2/{llama,gemma}/
Verified: safe games match old NPZ within float32 precision (max_diff < 0.0002)

Output: /home/jovyan/llm-addiction/exploratory_experiments/additional_experiments/
        sae_feature_analysis/results/within_model_v2/

Run:
    python run_v2_within_model_analysis.py
"""

import os
import sys
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Dict, List, Tuple, Optional

from scipy import stats
from scipy.stats import f_oneway
from statsmodels.stats.multitest import multipletests
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from matplotlib_venn import venn2
import seaborn as sns


# ===========================================================================
# V2 CONFIGURATION — Decision-Point-Fixed Features
# ===========================================================================
V2_SAE_DIR = Path("/home/jovyan/beomi/llm-addiction-data/sae_features_v2")
GAME_DIR = Path("/home/jovyan/beomi/llm-addiction-data/hf-dataset/slot_machine")

SAE_DIRS = {
    "llama": V2_SAE_DIR / "llama",
    "gemma": V2_SAE_DIR / "gemma",
}
GAME_FILES = {
    "llama": GAME_DIR / "llama" / "final_llama_20251004_021106.json",
    "gemma": GAME_DIR / "gemma" / "final_gemma_20251004_172426.json",
}

# V2 has all layers
LLAMA_LAYERS = list(range(0, 32))  # 0-31 (all 32 layers)
GEMMA_LAYERS = list(range(0, 42))  # 0-41 (all 42 layers)

# Output directories — clearly labeled V2
BASE_OUTPUT = Path("/home/jovyan/llm-addiction/exploratory_experiments/additional_experiments/"
                   "sae_feature_analysis/results/within_model_v2")
FIGURE_DIR = BASE_OUTPUT / "figures_improved"
RESULTS_DIR = BASE_OUTPUT / "json"
LOG_DIR = BASE_OUTPUT / "logs"

# Analysis parameters (same as original)
MIN_ACTIVATION_RATE = 0.01
CLASSIFICATION_CV_FOLDS = 5
CLASSIFICATION_C = 1.0
ANOVA_PRESCREENING_K = 5000
FDR_ALPHA = 0.05
N_PERMUTATIONS = 1000

# Visualization
MODEL_LABELS = {"llama": "LLaMA-3.1-8B", "gemma": "Gemma-2-9B"}
COLORS = {
    "llama": "#1f4e79", "gemma": "#c0392b", "all": "#2c3e50",
    "variable": "#e74c3c", "fixed": "#3498db",
    "median": "#c0392b", "p75": "#e67e22", "max": "#8e44ad", "iqr_fill": "#aec6cf",
}
ETA_SQ_SMALL, ETA_SQ_MEDIUM, ETA_SQ_LARGE = 0.01, 0.06, 0.14


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ===========================================================================
# Logging
# ===========================================================================
def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR / f'v2_analysis_{ts}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Log: {log_file}")
    return logger


# ===========================================================================
# Data Loading
# ===========================================================================
def load_game_results(model: str) -> List[Dict]:
    with open(GAME_FILES[model]) as f:
        return json.load(f)['results']


def get_game_metadata(games: List[Dict]) -> Dict[str, np.ndarray]:
    return {
        'bet_types': np.array([g['bet_type'] for g in games]),
        'outcomes': np.array([g['outcome'] for g in games]),
    }


def load_npz_features(model: str, layer: int) -> Optional[np.ndarray]:
    path = SAE_DIRS[model] / f"layer_{layer}_features.npz"
    if not path.exists():
        return None
    return np.load(path, allow_pickle=True)['features']


def filter_sparse_features(features: np.ndarray, min_rate: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    rate = (features != 0).mean(axis=0)
    mask = rate >= min_rate
    return features[:, mask], np.where(mask)[0]


def fdr_correction(p_values, alpha=0.05):
    if len(p_values) == 0:
        return np.array([]), np.array([])
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
    return reject, pvals_corrected


# ===========================================================================
# Part 1: Classification
# ===========================================================================
def run_classification_layer(features, active_indices, labels, n_folds=5, C=1.0, logger=None):
    n_samples, n_features = features.shape
    n_pos, n_neg = labels.sum(), n_samples - labels.sum()

    if n_pos < n_folds or n_neg < n_folds:
        return {'auc_mean': 0.5, 'auc_std': 0.0, 'accuracy_mean': 0.5, 'f1_mean': 0.0,
                'n_features': n_features, 'n_pos': int(n_pos), 'n_neg': int(n_neg),
                'skipped': True, 'top_features': []}

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    aucs, accs, f1s = [], [], []
    all_coefs = np.zeros(n_features)

    for train_idx, test_idx in skf.split(features, labels):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = LogisticRegression(C=C, penalty='l2', solver='lbfgs', max_iter=1000,
                                 class_weight='balanced', random_state=42)
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred = clf.predict(X_test)

        aucs.append(roc_auc_score(y_test, y_prob))
        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred, zero_division=0))
        all_coefs += clf.coef_[0]

    all_coefs /= n_folds
    top_k = min(50, n_features)
    top_idx = np.argsort(np.abs(all_coefs))[-top_k:][::-1]

    return {
        'auc_mean': float(np.mean(aucs)), 'auc_std': float(np.std(aucs)),
        'auc_folds': [float(a) for a in aucs],
        'accuracy_mean': float(np.mean(accs)), 'f1_mean': float(np.mean(f1s)),
        'n_features': n_features, 'n_pos': int(n_pos), 'n_neg': int(n_neg),
        'top_features': [{'feature_idx': int(active_indices[i]), 'coef': float(all_coefs[i])}
                         for i in top_idx],
        'skipped': False,
    }


def run_classification_permutation(features, labels, observed_auc, n_perm=1000, n_folds=5, C=1.0):
    perm_aucs = []
    for i in range(n_perm):
        perm_labels = np.random.permutation(labels)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=i)
        fold_aucs = []
        for train_idx, test_idx in skf.split(features, perm_labels):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(features[train_idx])
            X_test = scaler.transform(features[test_idx])
            clf = LogisticRegression(C=C, penalty='l2', solver='lbfgs', max_iter=500,
                                     class_weight='balanced', random_state=42)
            clf.fit(X_train, perm_labels[train_idx])
            fold_aucs.append(roc_auc_score(perm_labels[test_idx], clf.predict_proba(X_test)[:, 1]))
        perm_aucs.append(np.mean(fold_aucs))

    perm_aucs = np.array(perm_aucs)
    return {
        'p_value': float((perm_aucs >= observed_auc).mean()),
        'observed_auc': float(observed_auc),
        'perm_mean': float(perm_aucs.mean()),
        'perm_std': float(perm_aucs.std()),
        'perm_95': float(np.percentile(perm_aucs, 95)),
    }


def run_classification(model, meta, layers, logger):
    logger.info(f"[Part 1] Classification: {model.upper()}")
    labels = (meta['outcomes'] == 'bankruptcy').astype(int)
    bet_types = meta['bet_types']
    var_mask, fix_mask = bet_types == 'variable', bet_types == 'fixed'

    logger.info(f"  Total={len(labels)}, BK={labels.sum()}, Safe={(1-labels).sum()}")
    logger.info(f"  Variable: {var_mask.sum()} (BK={labels[var_mask].sum()}), "
                f"Fixed: {fix_mask.sum()} (BK={labels[fix_mask].sum()})")

    results_all, results_var, results_fix = [], [], []
    best_layer, best_auc = None, 0.0

    for layer in layers:
        raw = load_npz_features(model, layer)
        if raw is None:
            logger.warning(f"  Layer {layer}: not found")
            continue

        features, active_idx = filter_sparse_features(raw, MIN_ACTIVATION_RATE)
        logger.info(f"  Layer {layer}: {raw.shape[1]} -> {features.shape[1]} features")

        result = run_classification_layer(features, active_idx, labels, CLASSIFICATION_CV_FOLDS,
                                          CLASSIFICATION_C, logger)
        result['layer'] = layer
        results_all.append(result)

        if not result['skipped'] and result['auc_mean'] > best_auc:
            best_auc, best_layer = result['auc_mean'], layer

        if labels[var_mask].sum() >= CLASSIFICATION_CV_FOLDS:
            rv = run_classification_layer(features[var_mask], active_idx, labels[var_mask],
                                          CLASSIFICATION_CV_FOLDS, CLASSIFICATION_C, logger)
            rv['layer'] = layer
            results_var.append(rv)

        if labels[fix_mask].sum() >= CLASSIFICATION_CV_FOLDS:
            rf = run_classification_layer(features[fix_mask], active_idx, labels[fix_mask],
                                          CLASSIFICATION_CV_FOLDS, CLASSIFICATION_C, logger)
            rf['layer'] = layer
            results_fix.append(rf)

        logger.info(f"    AUC={result['auc_mean']:.4f}+-{result['auc_std']:.4f}")

    # Permutation test on best layer
    perm_result = None
    if best_layer is not None:
        logger.info(f"  Permutation test: layer {best_layer} (AUC={best_auc:.4f}), {N_PERMUTATIONS} perm...")
        raw = load_npz_features(model, best_layer)
        features, _ = filter_sparse_features(raw, MIN_ACTIVATION_RATE)
        perm_result = run_classification_permutation(features, labels, best_auc, N_PERMUTATIONS)
        perm_result['layer'] = best_layer
        logger.info(f"    p={perm_result['p_value']:.4f}")

    return {
        'model': model, 'all_games': results_all,
        'variable_only': results_var, 'fixed_only': results_fix,
        'best_layer': best_layer, 'best_auc': float(best_auc),
        'permutation_test': perm_result,
        'n_total': len(labels),
    }


# ===========================================================================
# Part 2: Two-Way ANOVA
# ===========================================================================
def fast_f_test_groups(features, group_labels):
    unique_groups = np.unique(group_labels)
    n, k = features.shape[0], len(unique_groups)
    grand_mean = features.mean(axis=0)
    ss_between = np.zeros(features.shape[1])
    ss_within = np.zeros(features.shape[1])
    for g in unique_groups:
        mask = group_labels == g
        ng = mask.sum()
        gm = features[mask].mean(axis=0)
        ss_between += ng * (gm - grand_mean) ** 2
        ss_within += ((features[mask] - gm) ** 2).sum(axis=0)
    ms_between = ss_between / max(k - 1, 1)
    ms_within = ss_within / max(n - k, 1)
    ms_within[ms_within == 0] = 1e-10
    return ms_between / ms_within


def run_anova_single_feature(activations, bet_type, outcome):
    df = pd.DataFrame({
        'activation': activations,
        'bet_type': pd.Categorical(bet_type),
        'outcome': pd.Categorical(outcome),
    })
    try:
        model = ols('activation ~ C(bet_type) + C(outcome) + C(bet_type):C(outcome)', data=df).fit()
        table = anova_lm(model, typ=2)
        ss_total = table['sum_sq'].sum()
        if ss_total == 0:
            return None

        interaction = table.loc['C(bet_type):C(outcome)', :]
        bet_row = table.loc['C(bet_type)', :]
        outcome_row = table.loc['C(outcome)', :]

        group_means = {}
        for bt_val, bt_label in [(0, 'variable'), (1, 'fixed')]:
            for oc_val, oc_label in [(0, 'bankrupt'), (1, 'safe')]:
                mask = (bet_type == bt_val) & (outcome == oc_val)
                group_means[f'{bt_label}_{oc_label}'] = float(activations[mask].mean()) if mask.sum() > 0 else 0.0

        return {
            'interaction_p': float(interaction['PR(>F)']),
            'interaction_f': float(interaction['F']),
            'interaction_eta_sq': float(interaction['sum_sq'] / ss_total),
            'bet_type_p': float(bet_row['PR(>F)']),
            'bet_type_eta_sq': float(bet_row['sum_sq'] / ss_total),
            'outcome_p': float(outcome_row['PR(>F)']),
            'outcome_eta_sq': float(outcome_row['sum_sq'] / ss_total),
            'group_means': group_means,
        }
    except Exception:
        return None


def run_anova_layer(features, active_indices, bet_types, outcomes, prescreening_k=5000, logger=None):
    n_samples, n_features = features.shape
    bet_enc = (bet_types == 'fixed').astype(int)
    out_enc = (outcomes == 'voluntary_stop').astype(int)
    group_4way = bet_enc * 2 + out_enc

    f_values = fast_f_test_groups(features, group_4way)
    n_screen = min(prescreening_k, n_features)
    top_idx = np.argsort(f_values)[-n_screen:]

    if logger:
        logger.info(f"    Pre-screen: {n_features} -> {n_screen}")

    results = []
    for local_idx in top_idx:
        r = run_anova_single_feature(features[:, local_idx], bet_enc, out_enc)
        if r is not None:
            r['feature_idx'] = int(active_indices[local_idx])
            r['local_idx'] = int(local_idx)
            results.append(r)
    return results


def run_anova(model, meta, layers, logger):
    logger.info(f"[Part 2] Two-Way ANOVA: {model.upper()}")
    bet_types, outcomes = meta['bet_types'], meta['outcomes']

    vb = ((bet_types == 'variable') & (outcomes == 'bankruptcy')).sum()
    vs = ((bet_types == 'variable') & (outcomes == 'voluntary_stop')).sum()
    fb = ((bet_types == 'fixed') & (outcomes == 'bankruptcy')).sum()
    fs = ((bet_types == 'fixed') & (outcomes == 'voluntary_stop')).sum()
    logger.info(f"  VB={vb}, VS={vs}, FB={fb}, FS={fs}")

    all_layer_results = []
    for layer in layers:
        raw = load_npz_features(model, layer)
        if raw is None:
            logger.warning(f"  Layer {layer}: not found")
            continue

        features, active_idx = filter_sparse_features(raw, MIN_ACTIVATION_RATE)
        logger.info(f"  Layer {layer}: {raw.shape[1]} -> {features.shape[1]} active")

        layer_results = run_anova_layer(features, active_idx, bet_types, outcomes,
                                        ANOVA_PRESCREENING_K, logger)

        if layer_results:
            ps = np.array([r['interaction_p'] for r in layer_results])
            reject, p_fdr = fdr_correction(ps, FDR_ALPHA)
            n_sig = reject.sum()
            for i, r in enumerate(layer_results):
                r['interaction_p_fdr'] = float(p_fdr[i])
                r['interaction_significant'] = bool(reject[i])
            sig = sorted([r for r in layer_results if r['interaction_significant']],
                         key=lambda x: x['interaction_eta_sq'], reverse=True)
            logger.info(f"    {len(layer_results)} tested, {n_sig} significant")
            if sig:
                logger.info(f"    Top: F{sig[0]['feature_idx']}, "
                            f"eta2={sig[0]['interaction_eta_sq']:.6f}")
        else:
            sig, n_sig = [], 0

        all_layer_results.append({
            'layer': layer, 'n_tested': len(layer_results),
            'n_significant': n_sig,
            'significant_features': sig[:100],
            'all_results': layer_results,
        })

    return {'model': model, 'layer_results': all_layer_results}


# ===========================================================================
# Save Results
# ===========================================================================
def save_clf(clf_data, path):
    out = {k: clf_data[k] for k in ['model', 'best_layer', 'best_auc', 'permutation_test',
                                      'all_games', 'variable_only', 'fixed_only', 'n_total']}
    with open(path, 'w') as f:
        json.dump(out, f, indent=2, cls=NumpyEncoder)


def save_anova(anova_data, path):
    out = {'model': anova_data['model'], 'layer_summary': []}
    for lr in anova_data['layer_results']:
        out['layer_summary'].append({
            'layer': lr['layer'], 'n_tested': lr['n_tested'],
            'n_significant': lr['n_significant'],
            'top_features': lr['significant_features'][:50],
        })
    with open(path, 'w') as f:
        json.dump(out, f, indent=2, cls=NumpyEncoder)


# ===========================================================================
# Improved Figures (same as visualize_improved_figures.py)
# ===========================================================================
def setup_style():
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.size': 10,
        'axes.titlesize': 12, 'axes.titleweight': 'bold',
        'axes.labelsize': 11, 'figure.dpi': 150, 'savefig.dpi': 300,
        'savefig.bbox': 'tight', 'axes.spines.top': False, 'axes.spines.right': False,
    })


def fig_a_improved(clf_dict, save_path):
    models = [m for m in ["llama", "gemma"] if m in clf_dict]
    if not models:
        return

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, len(models), hspace=0.35, wspace=0.3)

    for col, model in enumerate(models):
        clf = clf_dict[model]
        results = [r for r in clf["all_games"] if not r.get("skipped", False)]
        if not results:
            continue

        layers = np.array([r["layer"] for r in results])
        auc_mean = np.array([r["auc_mean"] for r in results])
        n_total = clf.get("n_total", 3200)

        ax1 = fig.add_subplot(gs[0, col])
        err = np.clip(1.0 - auc_mean, 1e-6, None)
        ax1.semilogy(layers, err, "-o", color=COLORS[model], markersize=4, lw=1.8,
                     label=MODEL_LABELS[model])

        for sub, style, clr in [("variable_only", "--", COLORS["variable"]),
                                 ("fixed_only", "--", COLORS["fixed"])]:
            sr = [r for r in clf.get(sub, []) if not r.get("skipped", False)]
            if sr:
                ax1.semilogy([r["layer"] for r in sr],
                             [max(1 - r["auc_mean"], 1e-6) for r in sr],
                             style, color=clr, lw=1.0, alpha=0.7,
                             label="Variable" if "variable" in sub else "Fixed")

        ax1.set_ylabel("Classification Error (1-AUC)", fontsize=11)
        ax1.set_title(MODEL_LABELS[model], fontsize=13, fontweight="bold")
        ax1.legend(fontsize=8, loc="upper right")
        ax1.grid(True, alpha=0.3, which="both")
        ax1.set_ylim(1e-5, 2e-1)
        ax1.axhline(y=0.01, color="gray", ls=":", alpha=0.5, lw=0.8)

        ax2 = fig.add_subplot(gs[1, col])
        nf = [r.get("n_features", 0) for r in results]
        ax2.bar(layers, nf, color=COLORS[model], alpha=0.6, width=0.8)
        max_idx = int(np.argmax(nf))
        ax2.annotate(f"{nf[max_idx]:,}", xy=(layers[max_idx], nf[max_idx]),
                     xytext=(0, 5), textcoords="offset points",
                     fontsize=7, ha="center", fontweight="bold", color=COLORS[model])
        total = 131072 if model == "gemma" else 32768
        avg_pct = np.mean([100 * n / total for n in nf])
        ax2.text(0.98, 0.95,
                 f"Avg: {avg_pct:.1f}% of {total//1000}K total\npass activation filter",
                 transform=ax2.transAxes, ha="right", va="top", fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
        ax2.set_xlabel("Layer", fontsize=11)
        ax2.set_ylabel("Active SAE Features\n(>1% activation rate)", fontsize=10)
        ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("SAE Features Predict Gambling Outcome — V2 (Decision-Point Fix)",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def fig_b_improved(anova_dict, save_path):
    models = [m for m in ["llama", "gemma"] if m in anova_dict]
    if not models:
        return

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5.5))
    if len(models) == 1:
        axes = [axes]

    cells = ["variable_bankrupt", "variable_safe", "fixed_bankrupt", "fixed_safe"]
    cell_labels = ["Var-BK", "Var-Safe", "Fix-BK", "Fix-Safe"]

    for ax, model in zip(axes, models):
        anova = anova_dict[model]
        lk = "layer_results" if "layer_results" in anova else "layer_summary"

        all_sig = []
        for lr in anova[lk]:
            fk = "significant_features" if "significant_features" in lr else "top_features"
            for feat in lr.get(fk, []):
                if feat.get("interaction_significant", True):
                    fc = dict(feat)
                    fc["layer"] = lr["layer"]
                    all_sig.append(fc)

        if not all_sig:
            ax.text(0.5, 0.5, f"No significant features for {model}",
                    transform=ax.transAxes, ha="center", va="center")
            continue

        all_sig.sort(key=lambda x: x["interaction_eta_sq"], reverse=True)
        top_n = min(10, len(all_sig))
        top = all_sig[:top_n]

        matrix = np.zeros((top_n, 4))
        for i, feat in enumerate(top):
            gm = feat["group_means"]
            for j, c in enumerate(cells):
                matrix[i, j] = gm.get(c, 0.0)

        row_means = matrix.mean(axis=1, keepdims=True)
        row_stds = matrix.std(axis=1, keepdims=True)
        row_stds[row_stds == 0] = 1.0
        matrix_z = (matrix - row_means) / row_stds

        pat_labels = []
        for i in range(top_n):
            mc = np.argmax(matrix_z[i])
            pat_labels.append(["Var-BK", "Var-Safe", "Fix-BK", "Fix-Safe"][mc])

        feat_labels = [f"L{f['layer']}/F{f['feature_idx']}" for f in top]
        eta_vals = [f"{f['interaction_eta_sq']:.3f}" for f in top]

        im = ax.imshow(matrix_z, cmap="RdBu_r", aspect="auto", vmin=-2.2, vmax=2.2)
        ax.set_xticks(range(4))
        ax.set_xticklabels(cell_labels, fontsize=9)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(feat_labels, fontsize=9)

        ax2 = ax.secondary_yaxis("right")
        ax2.set_yticks(range(top_n))
        ax2.set_yticklabels([f"eta2={v}" for v in eta_vals], fontsize=7.5, color="gray")

        for i, pat in enumerate(pat_labels):
            mc = COLORS["variable"] if "Var" in pat else COLORS["fixed"]
            ax.plot(-0.6, i, "s", color=mc, markersize=6,
                    transform=ax.get_yaxis_transform(), clip_on=False)

        ax.set_title(f"Top-10 Interaction Features - {MODEL_LABELS[model]}",
                     fontsize=11, fontweight="bold")
        cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label("Z-score", fontsize=8)

    fig.suptitle("V2: Decision-Point-Fixed SAE Features", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def fig_d_improved(anova_dict, save_path):
    models = [m for m in ["llama", "gemma"] if m in anova_dict]
    if not models:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    last_layer = 0

    for model in models:
        anova = anova_dict[model]
        lk = "layer_results" if "layer_results" in anova else "layer_summary"

        layers, medians, p75s, maxes = [], [], [], []
        for lr in anova[lk]:
            fk = "significant_features" if "significant_features" in lr else "top_features"
            results_to_use = lr.get("all_results", lr.get(fk, []))
            if not results_to_use:
                continue
            eta = [r["interaction_eta_sq"] for r in results_to_use]
            if not eta:
                continue
            layers.append(lr["layer"])
            medians.append(np.median(eta))
            p75s.append(np.percentile(eta, 75))
            maxes.append(np.max(eta))

        if not layers:
            continue
        last_layer = max(last_layer, max(layers))
        color = COLORS[model]
        ax.plot(layers, medians, "-", color=color, lw=2.0, label=f"{MODEL_LABELS[model]} (median)")
        ax.plot(layers, maxes, ":", color=color, lw=1.2, alpha=0.7, label=f"{MODEL_LABELS[model]} (max)")
        ax.fill_between(layers, medians, p75s, alpha=0.15, color=color)

    for y, label in [(ETA_SQ_LARGE, "Large"), (ETA_SQ_MEDIUM, "Medium"), (ETA_SQ_SMALL, "Small")]:
        ax.axhline(y=y, color="gray", ls="--", alpha=0.4, lw=0.8)
        ax.text(last_layer + 0.5, y, label, fontsize=7.5, color="gray", va="bottom")

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("eta2 (Interaction Effect Size)", fontsize=11)
    ax.set_title("Betting Condition x Outcome Interaction Effect (V2 Decision-Point Fix)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.grid(axis="y", alpha=0.3)
    ax.text(0.98, 0.02,
            "eta2 = fraction of variance in SAE activation\n"
            "explained by betting_type x outcome interaction.\n"
            "V2: bankruptcy games use pre-bankruptcy balance.",
            transform=ax.transAxes, fontsize=7.5, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.85))

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def fig_e_improved(anova_dict, save_path):
    models = [m for m in ["llama", "gemma"] if m in anova_dict]
    if not models:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    for model in models:
        anova = anova_dict[model]
        lk = "layer_results" if "layer_results" in anova else "layer_summary"

        layers, pcts, n_tested_list = [], [], []
        for lr in anova[lk]:
            nt = lr["n_tested"]
            ns = lr["n_significant"]
            if nt > 0:
                layers.append(lr["layer"])
                pcts.append(100.0 * ns / nt)
                n_tested_list.append(nt)

        if not layers:
            continue

        color = COLORS[model]
        ax.plot(layers, pcts, "-o", color=color, lw=1.8, markersize=4, label=MODEL_LABELS[model])

        valid_pcts = list(pcts)
        if model == "gemma":
            for i, (l, n) in enumerate(zip(layers, n_tested_list)):
                if n < 10:
                    valid_pcts[i] = 0

        peak_idx = int(np.argmax(valid_pcts))
        ax.plot(layers[peak_idx], pcts[peak_idx], "o", color=color,
                markersize=8, markeredgecolor="white", markeredgewidth=1.5, zorder=5)
        ax.annotate(f"L{layers[peak_idx]} ({pcts[peak_idx]:.1f}%)",
                    xy=(layers[peak_idx], pcts[peak_idx]),
                    xytext=(5, 8), textcoords="offset points",
                    fontsize=8, color=color, fontweight="bold")

        if model == "gemma":
            for i, (l, n) in enumerate(zip(layers, n_tested_list)):
                if l == 41 and n < 10:
                    ax.annotate(f"L41 (n={n}, artifact)",
                                xy=(l, pcts[i]),
                                xytext=(-50, -30), textcoords="offset points",
                                fontsize=7.5, color="gray",
                                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Features with Significant\nBetting x Outcome Interaction (%)", fontsize=10)
    ax.set_title("Interaction Feature Ratio by Layer (V2 Decision-Point Fix)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.text(0.98, 0.55,
            "Significant interaction = feature responds\n"
            "differently to bankruptcy vs safe outcomes\n"
            "depending on betting autonomy (var/fixed).\n\n"
            "V2: fixed circular reasoning in bankruptcy prompts.",
            transform=ax.transAxes, fontsize=7.5, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.85))

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    setup_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    logger.info("=" * 70)
    logger.info("V2 WITHIN-MODEL SAE ANALYSIS (Decision-Point Fix)")
    logger.info(f"SAE data: {V2_SAE_DIR}")
    logger.info(f"Output: {BASE_OUTPUT}")
    logger.info("=" * 70)

    # Verify data exists
    for model in ["llama", "gemma"]:
        d = SAE_DIRS[model]
        n_files = len(list(d.glob("layer_*_features.npz")))
        logger.info(f"  {model}: {n_files} layer NPZ files in {d}")

    all_clf, all_anova = {}, {}

    for model in ["llama", "gemma"]:
        logger.info(f"\n{'#' * 70}")
        logger.info(f"# {model.upper()}")
        logger.info(f"{'#' * 70}")

        layers = LLAMA_LAYERS if model == "llama" else GEMMA_LAYERS
        games = load_game_results(model)
        meta = get_game_metadata(games)
        logger.info(f"  Games: {len(games)}")
        logger.info(f"  Outcomes: {dict(Counter(meta['outcomes']))}")
        logger.info(f"  Bet types: {dict(Counter(meta['bet_types']))}")

        # Part 1: Classification
        clf = run_classification(model, meta, layers, logger)
        clf_path = RESULTS_DIR / f"classification_{model}_{ts}.json"
        save_clf(clf, clf_path)
        logger.info(f"  Saved: {clf_path}")
        all_clf[model] = clf

        # Part 2: ANOVA
        anova = run_anova(model, meta, layers, logger)
        anova_path = RESULTS_DIR / f"anova_{model}_{ts}.json"
        save_anova(anova, anova_path)
        logger.info(f"  Saved: {anova_path}")
        all_anova[model] = anova

    # Generate figures
    logger.info(f"\n{'=' * 70}")
    logger.info("GENERATING IMPROVED FIGURES")
    logger.info(f"{'=' * 70}")

    fig_a_improved(all_clf, FIGURE_DIR / "fig_a_error_rate_improved.png")
    fig_b_improved(all_anova, FIGURE_DIR / "fig_b_heatmap_combined_improved.png")
    fig_d_improved(all_anova, FIGURE_DIR / "fig_d_eta_combined_improved.png")
    fig_e_improved(all_anova, FIGURE_DIR / "fig_e_interaction_improved.png")

    # Save metadata
    meta_info = {
        "version": "V2",
        "description": "Decision-point-fixed SAE features",
        "sae_data_source": str(V2_SAE_DIR),
        "llama_sae": "fnlp direct (NOT sae_lens) — matching LlamaScopeWorking",
        "gemma_sae": "sae_lens gemma-scope-9b-pt-res-canonical",
        "decision_point_fix": "bankruptcy games use pre-bankruptcy balance, not $0",
        "verification": "safe games match old NPZ within float32 precision",
        "timestamp": ts,
        "llama_best_layer": all_clf.get("llama", {}).get("best_layer"),
        "llama_best_auc": all_clf.get("llama", {}).get("best_auc"),
        "gemma_best_layer": all_clf.get("gemma", {}).get("best_layer"),
        "gemma_best_auc": all_clf.get("gemma", {}).get("best_auc"),
    }
    with open(BASE_OUTPUT / "V2_METADATA.json", "w") as f:
        json.dump(meta_info, f, indent=2)

    logger.info(f"\n{'=' * 70}")
    logger.info("ALL COMPLETE")
    logger.info(f"Figures: {FIGURE_DIR}")
    logger.info(f"JSON results: {RESULTS_DIR}")
    logger.info(f"Metadata: {BASE_OUTPUT / 'V2_METADATA.json'}")
    logger.info(f"{'=' * 70}")


if __name__ == "__main__":
    main()
