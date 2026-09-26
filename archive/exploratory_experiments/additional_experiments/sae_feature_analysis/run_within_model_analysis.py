#!/usr/bin/env python3
"""
Within-Model SAE Analysis: Classification + Two-Way ANOVA
=========================================================

Analyzes whether SAE features encode outcome information within each model,
and whether variable betting activates risk-related features via interaction effects.

Part 1: Classification — SAE features → bankruptcy prediction (per-layer AUC)
Part 2: Two-way ANOVA — bet_type × outcome interaction features (optimized)
Part 3: Visualization — Layer-wise AUC, interaction heatmaps, feature overlap

Usage:
    python run_within_model_analysis.py
    python run_within_model_analysis.py --models llama
    python run_within_model_analysis.py --models gemma --skip-classification
    python run_within_model_analysis.py --skip-anova
"""

import os
import sys
import json
import argparse
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


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
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


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import seaborn as sns

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_ROOT = Path("/home/jovyan/beomi/llm-addiction-data/hf-dataset")
SAE_DIR = DATA_ROOT / "sae_patching" / "corrected_sae_analysis"
GAME_DIR = DATA_ROOT / "slot_machine"

OUTPUT_DIR = Path("/home/jovyan/beomi/llm-addiction-data/sae_condition_comparison/within_model")
FIGURE_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"

LLAMA_LAYERS = list(range(1, 32))  # 1-31
GEMMA_LAYERS = list(range(0, 42))  # 0-41 (full 42 layers)

# SAE directories per model (Gemma has a separate full-layer directory)
SAE_DIRS = {
    "llama": SAE_DIR / "llama",
    "gemma": SAE_DIR / "gemma_full_42layers",
}

GAME_FILES = {
    "llama": GAME_DIR / "llama" / "final_llama_20251004_021106.json",
    "gemma": GAME_DIR / "gemma" / "final_gemma_20251004_172426.json",
}

# Analysis parameters
MIN_ACTIVATION_RATE = 0.01   # Filter features active in < 1% of games
CLASSIFICATION_CV_FOLDS = 5
CLASSIFICATION_C = 1.0       # Logistic regression regularization
ANOVA_PRESCREENING_K = 5000  # Keep top K features by F-test before statsmodels
FDR_ALPHA = 0.05
N_PERMUTATIONS = 1000        # For permutation test


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'within_model_analysis_{timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Log file: {log_file}")
    return logger


# ---------------------------------------------------------------------------
# Data Loading (reused from run_all_analyses.py)
# ---------------------------------------------------------------------------
def load_game_results(model: str) -> List[Dict]:
    with open(GAME_FILES[model]) as f:
        data = json.load(f)
    return data['results']


def get_game_metadata(games: List[Dict]) -> Dict[str, np.ndarray]:
    bet_types = np.array([g['bet_type'] for g in games])
    outcomes = np.array([g['outcome'] for g in games])
    return {'bet_types': bet_types, 'outcomes': outcomes}


def load_npz_features(model: str, layer: int) -> Optional[np.ndarray]:
    sae_dir = SAE_DIRS.get(model, SAE_DIR / model)
    npz_path = sae_dir / f"layer_{layer}_features.npz"
    if not npz_path.exists():
        return None
    data = np.load(npz_path, allow_pickle=True)
    return data['features']  # (3200, n_features)


def fdr_correction(p_values, alpha=0.05):
    if len(p_values) == 0:
        return np.array([]), np.array([])
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
    return reject, pvals_corrected


# ---------------------------------------------------------------------------
# Feature Filtering
# ---------------------------------------------------------------------------
def filter_sparse_features(features: np.ndarray, min_rate: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """Filter out features with activation rate below threshold.

    Returns:
        filtered_features: (n_samples, n_active)
        active_indices: original feature indices that passed the filter
    """
    activation_rate = (features != 0).mean(axis=0)
    active_mask = activation_rate >= min_rate
    active_indices = np.where(active_mask)[0]
    return features[:, active_mask], active_indices


# ---------------------------------------------------------------------------
# Part 1: Classification Analysis
# ---------------------------------------------------------------------------
def run_classification_layer(
    features: np.ndarray,
    active_indices: np.ndarray,
    labels: np.ndarray,
    n_folds: int = 5,
    C: float = 1.0,
    logger: logging.Logger = None,
) -> Dict:
    """Run logistic regression classification for one layer.

    Args:
        features: (n_samples, n_features) — already filtered
        labels: binary (0=safe, 1=bankrupt)
    """
    n_samples, n_features = features.shape
    n_pos = labels.sum()
    n_neg = n_samples - n_pos

    if n_pos < n_folds or n_neg < n_folds:
        if logger:
            logger.warning(f"  Too few samples: pos={n_pos}, neg={n_neg}")
        return {'auc': 0.5, 'accuracy': 0.5, 'f1': 0.0, 'n_features': n_features,
                'n_pos': int(n_pos), 'n_neg': int(n_neg), 'skipped': True}

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    aucs, accs, f1s = [], [], []
    all_coefs = np.zeros(n_features)

    for train_idx, test_idx in skf.split(features, labels):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = LogisticRegression(
            C=C, penalty='l2', solver='lbfgs', max_iter=1000,
            class_weight='balanced', random_state=42
        )
        clf.fit(X_train, y_train)

        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred = clf.predict(X_test)

        aucs.append(roc_auc_score(y_test, y_prob))
        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred, zero_division=0))
        all_coefs += clf.coef_[0]

    all_coefs /= n_folds
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    # Top features by absolute coefficient
    top_k = min(50, n_features)
    top_idx = np.argsort(np.abs(all_coefs))[-top_k:][::-1]
    top_features = [
        {'feature_idx': int(active_indices[i]), 'coef': float(all_coefs[i])}
        for i in top_idx
    ]

    return {
        'auc_mean': float(mean_auc),
        'auc_std': float(std_auc),
        'auc_folds': [float(a) for a in aucs],
        'accuracy_mean': float(np.mean(accs)),
        'f1_mean': float(np.mean(f1s)),
        'n_features': n_features,
        'n_pos': int(n_pos),
        'n_neg': int(n_neg),
        'top_features': top_features,
        'skipped': False,
    }


def run_classification_permutation(
    features: np.ndarray,
    labels: np.ndarray,
    observed_auc: float,
    n_permutations: int = 1000,
    n_folds: int = 5,
    C: float = 1.0,
) -> Dict:
    """Permutation test for classification AUC significance."""
    perm_aucs = []
    for i in range(n_permutations):
        perm_labels = np.random.permutation(labels)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=i)
        fold_aucs = []
        for train_idx, test_idx in skf.split(features, perm_labels):
            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = perm_labels[train_idx], perm_labels[test_idx]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            clf = LogisticRegression(
                C=C, penalty='l2', solver='lbfgs', max_iter=500,
                class_weight='balanced', random_state=42
            )
            clf.fit(X_train, y_train)
            y_prob = clf.predict_proba(X_test)[:, 1]
            fold_aucs.append(roc_auc_score(y_test, y_prob))
        perm_aucs.append(np.mean(fold_aucs))

    perm_aucs = np.array(perm_aucs)
    p_value = (perm_aucs >= observed_auc).mean()

    return {
        'p_value': float(p_value),
        'observed_auc': float(observed_auc),
        'perm_mean': float(perm_aucs.mean()),
        'perm_std': float(perm_aucs.std()),
        'perm_95': float(np.percentile(perm_aucs, 95)),
    }


def run_classification_analysis(
    model: str, meta: Dict, layers: List[int], logger: logging.Logger,
    skip_permutation: bool = False,
) -> Dict:
    """Part 1: Classification analysis across all layers."""
    logger.info("=" * 70)
    logger.info(f"[Part 1] Classification: SAE → Bankruptcy — {model.upper()}")
    logger.info("=" * 70)

    # Labels: 1=bankrupt, 0=safe
    labels = (meta['outcomes'] == 'bankruptcy').astype(int)
    bet_types = meta['bet_types']
    var_mask = bet_types == 'variable'
    fix_mask = bet_types == 'fixed'

    logger.info(f"  Total: {len(labels)}, Bankrupt: {labels.sum()}, Safe: {(1-labels).sum()}")
    logger.info(f"  Variable: {var_mask.sum()} (BK={labels[var_mask].sum()}), "
                f"Fixed: {fix_mask.sum()} (BK={labels[fix_mask].sum()})")

    results_all = []       # All games
    results_var = []       # Variable-only
    results_fix = []       # Fixed-only
    best_layer = None
    best_auc = 0.0

    for layer in layers:
        features_raw = load_npz_features(model, layer)
        if features_raw is None:
            logger.warning(f"  Layer {layer}: NPZ not found, skipping")
            continue

        features, active_idx = filter_sparse_features(features_raw, MIN_ACTIVATION_RATE)
        logger.info(f"  Layer {layer}: {features_raw.shape[1]} → {features.shape[1]} features "
                     f"(activation rate ≥ {MIN_ACTIVATION_RATE})")

        # Classification on all games
        result = run_classification_layer(
            features, active_idx, labels, CLASSIFICATION_CV_FOLDS, CLASSIFICATION_C, logger
        )
        result['layer'] = layer
        results_all.append(result)

        if not result['skipped'] and result['auc_mean'] > best_auc:
            best_auc = result['auc_mean']
            best_layer = layer

        # Classification on variable-only
        if labels[var_mask].sum() >= CLASSIFICATION_CV_FOLDS:
            result_var = run_classification_layer(
                features[var_mask], active_idx, labels[var_mask],
                CLASSIFICATION_CV_FOLDS, CLASSIFICATION_C, logger
            )
            result_var['layer'] = layer
            results_var.append(result_var)

        # Classification on fixed-only
        if labels[fix_mask].sum() >= CLASSIFICATION_CV_FOLDS:
            result_fix = run_classification_layer(
                features[fix_mask], active_idx, labels[fix_mask],
                CLASSIFICATION_CV_FOLDS, CLASSIFICATION_C, logger
            )
            result_fix['layer'] = layer
            results_fix.append(result_fix)

        logger.info(f"    AUC(all)={result['auc_mean']:.4f}±{result['auc_std']:.4f}, "
                     f"Acc={result['accuracy_mean']:.4f}, F1={result['f1_mean']:.4f}")

    # Permutation test on best layer
    permutation_result = None
    if best_layer is not None and not skip_permutation:
        logger.info(f"\n  Permutation test on best layer {best_layer} (AUC={best_auc:.4f})...")
        features_raw = load_npz_features(model, best_layer)
        features, _ = filter_sparse_features(features_raw, MIN_ACTIVATION_RATE)
        permutation_result = run_classification_permutation(
            features, labels, best_auc, N_PERMUTATIONS, CLASSIFICATION_CV_FOLDS, CLASSIFICATION_C
        )
        permutation_result['layer'] = best_layer
        logger.info(f"    p-value={permutation_result['p_value']:.4f}, "
                     f"perm AUC={permutation_result['perm_mean']:.4f}±{permutation_result['perm_std']:.4f}")

    return {
        'model': model,
        'all_games': results_all,
        'variable_only': results_var,
        'fixed_only': results_fix,
        'best_layer': best_layer,
        'best_auc': float(best_auc),
        'permutation_test': permutation_result,
    }


# ---------------------------------------------------------------------------
# Part 2: Two-Way ANOVA (Optimized)
# ---------------------------------------------------------------------------
def fast_f_test_groups(features: np.ndarray, group_labels: np.ndarray) -> np.ndarray:
    """Vectorized one-way F-test across all features simultaneously.

    Args:
        features: (n_samples, n_features)
        group_labels: integer group labels (n_samples,)

    Returns:
        f_values: (n_features,)
    """
    unique_groups = np.unique(group_labels)
    n = features.shape[0]
    k = len(unique_groups)
    grand_mean = features.mean(axis=0)

    ss_between = np.zeros(features.shape[1])
    ss_within = np.zeros(features.shape[1])

    for g in unique_groups:
        mask = group_labels == g
        ng = mask.sum()
        group_mean = features[mask].mean(axis=0)
        ss_between += ng * (group_mean - grand_mean) ** 2
        ss_within += ((features[mask] - group_mean) ** 2).sum(axis=0)

    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (n - k)
    ms_within[ms_within == 0] = 1e-10  # avoid division by zero

    return ms_between / ms_within


def run_anova_single_feature(
    activations: np.ndarray, bet_type: np.ndarray, outcome: np.ndarray
) -> Optional[Dict]:
    """Run two-way ANOVA for a single feature using statsmodels."""
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

        # Group means
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


def run_anova_layer(
    features: np.ndarray,
    active_indices: np.ndarray,
    bet_types: np.ndarray,
    outcomes: np.ndarray,
    prescreening_k: int = 5000,
    logger: logging.Logger = None,
) -> List[Dict]:
    """Run optimized two-way ANOVA for one layer.

    Strategy:
    1. Sparse filter (already applied)
    2. Fast vectorized F-test pre-screening → top K features
    3. statsmodels OLS ANOVA on pre-screened features only
    """
    n_samples, n_features = features.shape

    # Encode groups: 4 groups for bet_type × outcome
    bet_type_enc = (bet_types == 'fixed').astype(int)   # 0=variable, 1=fixed
    outcome_enc = (outcomes == 'voluntary_stop').astype(int)  # 0=bankrupt, 1=safe
    group_4way = bet_type_enc * 2 + outcome_enc  # 0=VB, 1=VS, 2=FB, 3=FS

    # Pre-screening: fast F-test on 4-group comparison
    f_values = fast_f_test_groups(features, group_4way)
    n_to_screen = min(prescreening_k, n_features)
    top_f_idx = np.argsort(f_values)[-n_to_screen:]

    if logger:
        logger.info(f"    Pre-screening: {n_features} → {n_to_screen} features (top by F-test)")

    # Full ANOVA on pre-screened features
    results = []
    for local_idx in top_f_idx:
        activations = features[:, local_idx]
        original_idx = int(active_indices[local_idx])

        r = run_anova_single_feature(activations, bet_type_enc, outcome_enc)
        if r is not None:
            r['feature_idx'] = original_idx
            r['local_idx'] = int(local_idx)
            results.append(r)

    return results


def run_anova_analysis(
    model: str, meta: Dict, layers: List[int], logger: logging.Logger
) -> Dict:
    """Part 2: Two-way ANOVA analysis across all layers."""
    logger.info("=" * 70)
    logger.info(f"[Part 2] Two-Way ANOVA: bet_type × outcome — {model.upper()}")
    logger.info("=" * 70)

    bet_types = meta['bet_types']
    outcomes = meta['outcomes']
    logger.info(f"  Design: 2×2 (Variable/Fixed × Bankrupt/Safe)")
    logger.info(f"  Cells: VB={((bet_types=='variable')&(outcomes=='bankruptcy')).sum()}, "
                f"VS={((bet_types=='variable')&(outcomes=='voluntary_stop')).sum()}, "
                f"FB={((bet_types=='fixed')&(outcomes=='bankruptcy')).sum()}, "
                f"FS={((bet_types=='fixed')&(outcomes=='voluntary_stop')).sum()}")

    all_layer_results = []

    for layer in layers:
        features_raw = load_npz_features(model, layer)
        if features_raw is None:
            logger.warning(f"  Layer {layer}: NPZ not found, skipping")
            continue

        features, active_idx = filter_sparse_features(features_raw, MIN_ACTIVATION_RATE)
        logger.info(f"  Layer {layer}: {features_raw.shape[1]} → {features.shape[1]} active features")

        layer_results = run_anova_layer(
            features, active_idx, bet_types, outcomes, ANOVA_PRESCREENING_K, logger
        )

        # FDR correction on interaction p-values
        if layer_results:
            interaction_ps = np.array([r['interaction_p'] for r in layer_results])
            reject, p_fdr = fdr_correction(interaction_ps, FDR_ALPHA)

            n_sig = reject.sum()
            for i, r in enumerate(layer_results):
                r['interaction_p_fdr'] = float(p_fdr[i])
                r['interaction_significant'] = bool(reject[i])

            # Sort by interaction eta squared
            sig_results = [r for r in layer_results if r['interaction_significant']]
            sig_results.sort(key=lambda x: x['interaction_eta_sq'], reverse=True)

            logger.info(f"    ANOVA done: {len(layer_results)} tested, "
                         f"{n_sig} significant interactions (FDR < {FDR_ALPHA})")

            if sig_results:
                top = sig_results[0]
                logger.info(f"    Top interaction: feature {top['feature_idx']}, "
                             f"η²={top['interaction_eta_sq']:.6f}, "
                             f"p_FDR={top['interaction_p_fdr']:.2e}")
        else:
            sig_results = []
            n_sig = 0

        all_layer_results.append({
            'layer': layer,
            'n_tested': len(layer_results),
            'n_significant': n_sig,
            'significant_features': sig_results[:100],  # top 100
            'all_results': layer_results,  # keep for overlap analysis
        })

    return {
        'model': model,
        'layer_results': all_layer_results,
    }


# ---------------------------------------------------------------------------
# Part 3: Visualization
# ---------------------------------------------------------------------------
def setup_style():
    """Publication-quality matplotlib rcParams."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'axes.labelsize': 11,
        'axes.labelweight': 'normal',
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'legend.frameon': True,
        'legend.framealpha': 0.8,
        'legend.edgecolor': '0.8',
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
    })


# Consistent color palette
COLORS = {
    'llama': '#2c3e50',       # dark blue-gray
    'gemma': '#c0392b',       # dark red
    'all': '#2c3e50',         # primary line
    'variable': '#e74c3c',    # red for variable
    'fixed': '#3498db',       # blue for fixed
    'median': '#e74c3c',
    'p75': '#f39c12',
    'max': '#8e44ad',
    'iqr_fill': '#3498db',
}
MODEL_LABELS = {'llama': 'LLaMA-3.1-8B', 'gemma': 'Gemma-2-9B'}


def fig_a_layer_auc_combined(clf_data_dict: Dict[str, Dict], save_path: Path):
    """Fig A: Layer-wise AUC — two subplots (LLaMA, Gemma) side by side.

    Y-axis zoomed to 0.85–1.01 to show differences. "All games" as bold line,
    Variable/Fixed as thin auxiliary lines. Shaded error bands.
    """
    models = [m for m in ['llama', 'gemma'] if m in clf_data_dict]
    if not models:
        return

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        clf_data = clf_data_dict[model]
        results_all = [r for r in clf_data['all_games'] if not r.get('skipped', False)]
        results_var = [r for r in clf_data['variable_only'] if not r.get('skipped', False)]
        results_fix = [r for r in clf_data['fixed_only'] if not r.get('skipped', False)]

        if not results_all:
            continue

        # All games — bold line with shaded error
        layers_all = np.array([r['layer'] for r in results_all])
        auc_all = np.array([r['auc_mean'] for r in results_all])
        std_all = np.array([r['auc_std'] for r in results_all])
        ax.plot(layers_all, auc_all, '-', color=COLORS['all'], lw=2.0, label='All games')
        ax.fill_between(layers_all, auc_all - std_all, np.minimum(auc_all + std_all, 1.0),
                         alpha=0.15, color=COLORS['all'])

        # Variable — thin
        if results_var:
            layers_v = [r['layer'] for r in results_var]
            auc_v = [r['auc_mean'] for r in results_var]
            ax.plot(layers_v, auc_v, '--', color=COLORS['variable'], lw=1.0,
                    alpha=0.7, label='Variable')

        # Fixed — thin
        if results_fix:
            layers_f = [r['layer'] for r in results_fix]
            auc_f = [r['auc_mean'] for r in results_fix]
            ax.plot(layers_f, auc_f, '--', color=COLORS['fixed'], lw=1.0,
                    alpha=0.7, label='Fixed')

        ax.set_xlabel('Layer')
        ax.set_title(MODEL_LABELS.get(model, model))
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(axis='y')

    axes[0].set_ylabel('AUC-ROC')
    axes[0].set_ylim(0.85, 1.01)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def fig_b_heatmap(anova_data: Dict, save_path: Path):
    """Fig B: Top-10 interaction heatmap per model.

    No cell numbers — color only. Concise Y labels (L{layer}/F{idx}).
    """
    model = anova_data['model']
    layer_key = 'layer_results' if 'layer_results' in anova_data else 'layer_summary'
    layer_data = anova_data[layer_key]

    # Collect top interaction features across all layers
    all_sig = []
    for lr in layer_data:
        feat_key = 'significant_features' if 'significant_features' in lr else 'top_features'
        for feat in lr.get(feat_key, []):
            if feat.get('interaction_significant', True):
                feat_copy = dict(feat)
                feat_copy['layer'] = lr['layer']
                all_sig.append(feat_copy)

    if not all_sig:
        print(f"  No significant interactions for {model}, skipping heatmap")
        return

    all_sig.sort(key=lambda x: x['interaction_eta_sq'], reverse=True)
    top_n = min(10, len(all_sig))
    top_feats = all_sig[:top_n]

    cells = ['variable_bankrupt', 'variable_safe', 'fixed_bankrupt', 'fixed_safe']
    cell_labels = ['Var-BK', 'Var-Safe', 'Fix-BK', 'Fix-Safe']
    matrix = np.zeros((top_n, 4))

    for i, feat in enumerate(top_feats):
        gm = feat['group_means']
        for j, cell in enumerate(cells):
            matrix[i, j] = gm.get(cell, 0.0)

    # Z-score normalize each row
    row_means = matrix.mean(axis=1, keepdims=True)
    row_stds = matrix.std(axis=1, keepdims=True)
    row_stds[row_stds == 0] = 1.0
    matrix_z = (matrix - row_means) / row_stds

    feat_labels = [f"L{f['layer']}/F{f['feature_idx']}" for f in top_feats]

    fig, ax = plt.subplots(figsize=(5, max(4, top_n * 0.45)))
    im = ax.imshow(matrix_z, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)

    ax.set_xticks(range(4))
    ax.set_xticklabels(cell_labels)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(feat_labels, fontsize=9)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Z-score', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(f'Top-10 Interaction Features — {MODEL_LABELS.get(model, model)}')

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def fig_c_overlap(clf_data: Dict, anova_data: Dict, save_path: Path):
    """Fig C: Venn diagram — cleaner version with concise labels."""
    model = clf_data['model']
    best_layer = clf_data.get('best_layer')

    # Classification top features
    clf_top_set = set()
    for r in clf_data['all_games']:
        if r.get('layer') == best_layer and not r.get('skipped', False):
            clf_top_set = {f['feature_idx'] for f in r.get('top_features', [])}
            break

    # ANOVA interaction features
    layer_key = 'layer_results' if 'layer_results' in anova_data else 'layer_summary'
    anova_sig_set = set()
    for lr in anova_data[layer_key]:
        if lr['layer'] == best_layer:
            feat_key = 'significant_features' if 'significant_features' in lr else 'top_features'
            anova_sig_set = {f['feature_idx'] for f in lr.get(feat_key, [])
                            if f.get('interaction_significant', True)}
            break

    if not anova_sig_set:
        for lr in anova_data[layer_key]:
            feat_key = 'significant_features' if 'significant_features' in lr else 'top_features'
            for f in lr.get(feat_key, []):
                if f.get('interaction_significant', True):
                    anova_sig_set.add(f['feature_idx'])

    if not clf_top_set and not anova_sig_set:
        print(f"  No features for Venn diagram ({model})")
        return

    fig, ax = plt.subplots(figsize=(6, 5.5))
    overlap = clf_top_set & anova_sig_set
    union = clf_top_set | anova_sig_set
    jaccard = len(overlap) / max(1, len(union))

    try:
        v = venn2(
            [clf_top_set, anova_sig_set],
            set_labels=('Classification Top-50', 'ANOVA Interaction'),
            ax=ax
        )
        # Uniform colors
        if v.get_patch_by_id('10'):
            v.get_patch_by_id('10').set_color(COLORS['fixed'])
            v.get_patch_by_id('10').set_alpha(0.4)
        if v.get_patch_by_id('01'):
            v.get_patch_by_id('01').set_color(COLORS['variable'])
            v.get_patch_by_id('01').set_alpha(0.4)
        if v.get_patch_by_id('11'):
            v.get_patch_by_id('11').set_color('#8e44ad')
            v.get_patch_by_id('11').set_alpha(0.5)
    except Exception:
        ax.text(0.5, 0.5, f'Clf: {len(clf_top_set)} | ANOVA: {len(anova_sig_set)} | '
                           f'Overlap: {len(overlap)}',
                transform=ax.transAxes, ha='center', va='center', fontsize=12)

    ax.set_title(f'{MODEL_LABELS.get(model, model)} — Layer {best_layer}', pad=12)
    ax.text(0.5, -0.08, f'Jaccard = {jaccard:.3f}',
            transform=ax.transAxes, ha='center', fontsize=12, fontweight='bold')

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.08)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def fig_d_eta_summary(anova_data: Dict, save_path: Path):
    """Fig D: η² summary line plot — median, 75th percentile, max per layer.

    Replaces the unreadable per-layer histogram grid with a clean summary.
    """
    model = anova_data['model']
    layer_key = 'layer_results' if 'layer_results' in anova_data else 'layer_summary'
    layer_data = anova_data[layer_key]

    if not layer_data:
        return

    layers = []
    medians = []
    p75s = []
    maxes = []
    q25s = []

    for lr in layer_data:
        # Extract eta_sq values
        feat_key = 'significant_features' if 'significant_features' in lr else 'top_features'
        # Use all_results if available (has all tested features), otherwise top_features
        results_to_use = lr.get('all_results', lr.get(feat_key, []))
        if not results_to_use:
            continue

        eta_sq = [r['interaction_eta_sq'] for r in results_to_use]
        if not eta_sq:
            continue

        layers.append(lr['layer'])
        medians.append(np.median(eta_sq))
        q25s.append(np.percentile(eta_sq, 25))
        p75s.append(np.percentile(eta_sq, 75))
        maxes.append(np.max(eta_sq))

    if not layers:
        return

    layers = np.array(layers)
    medians = np.array(medians)
    q25s = np.array(q25s)
    p75s = np.array(p75s)
    maxes = np.array(maxes)

    fig, ax = plt.subplots(figsize=(8, 4))

    # IQR shaded area
    ax.fill_between(layers, q25s, p75s, alpha=0.2, color=COLORS['iqr_fill'], label='IQR')

    # Lines
    ax.plot(layers, medians, '-', color=COLORS['median'], lw=2.0, label='Median')
    ax.plot(layers, p75s, '--', color=COLORS['p75'], lw=1.2, label='75th pctl')
    ax.plot(layers, maxes, ':', color=COLORS['max'], lw=1.2, label='Max')

    ax.set_xlabel('Layer')
    ax.set_ylabel('η² (interaction)')
    ax.set_title(f'Interaction Effect Size Summary — {MODEL_LABELS.get(model, model)}')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(axis='y')

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def fig_e_interaction_combined(anova_data_dict: Dict[str, Dict], save_path: Path):
    """Fig E: Interaction ratio (%) per layer — both models on one line plot.

    Shows what percentage of tested features have significant interactions.
    """
    models = [m for m in ['llama', 'gemma'] if m in anova_data_dict]
    if not models:
        return

    fig, ax = plt.subplots(figsize=(8, 4))

    for model in models:
        anova_data = anova_data_dict[model]
        layer_key = 'layer_results' if 'layer_results' in anova_data else 'layer_summary'
        layer_data = anova_data[layer_key]

        layers = []
        pcts = []
        for lr in layer_data:
            n_tested = lr['n_tested']
            n_sig = lr['n_significant']
            if n_tested > 0:
                layers.append(lr['layer'])
                pcts.append(100.0 * n_sig / n_tested)

        if not layers:
            continue

        color = COLORS[model]
        ax.plot(layers, pcts, '-o', color=color, lw=1.8, markersize=4,
                label=MODEL_LABELS.get(model, model))

        # Mark peak
        peak_idx = int(np.argmax(pcts))
        ax.plot(layers[peak_idx], pcts[peak_idx], 'o', color=color,
                markersize=8, markeredgecolor='white', markeredgewidth=1.5, zorder=5)
        # Place label left of marker if near right edge
        n_layers_total = len(layers)
        near_right = peak_idx > n_layers_total * 0.8
        tx, ha = (-8, 'right') if near_right else (5, 'left')
        ax.annotate(f'L{layers[peak_idx]}',
                    xy=(layers[peak_idx], pcts[peak_idx]),
                    xytext=(tx, 8), textcoords='offset points',
                    fontsize=8, color=color, fontweight='bold', ha=ha)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Significant Interactions (%)')
    ax.set_title('Interaction Features by Layer')
    ax.legend(loc='upper left')
    ax.grid(axis='y')

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


# ---------------------------------------------------------------------------
# Save Results (JSON-serializable)
# ---------------------------------------------------------------------------
def save_classification_results(clf_data: Dict, path: Path):
    """Save classification results to JSON."""
    # Strip non-serializable fields
    output = {
        'model': clf_data['model'],
        'best_layer': clf_data['best_layer'],
        'best_auc': clf_data['best_auc'],
        'permutation_test': clf_data['permutation_test'],
        'all_games': clf_data['all_games'],
        'variable_only': clf_data['variable_only'],
        'fixed_only': clf_data['fixed_only'],
    }
    with open(path, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)


def save_anova_results(anova_data: Dict, path: Path):
    """Save ANOVA results to JSON (top features only, not all_results)."""
    output = {
        'model': anova_data['model'],
        'layer_summary': [],
    }
    for lr in anova_data['layer_results']:
        output['layer_summary'].append({
            'layer': lr['layer'],
            'n_tested': lr['n_tested'],
            'n_significant': lr['n_significant'],
            'top_features': lr['significant_features'][:50],  # top 50 per layer
        })

    with open(path, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def load_latest_json(results_dir: Path, prefix: str, model: str) -> Optional[Dict]:
    """Load the latest JSON result file matching prefix and model."""
    pattern = f"{prefix}_{model}_*.json"
    matches = sorted(results_dir.glob(pattern))
    if not matches:
        return None
    latest = matches[-1]  # sorted by name → latest timestamp
    with open(latest) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description='Within-Model SAE Analysis')
    parser.add_argument('--models', nargs='+', default=['llama', 'gemma'],
                        choices=['llama', 'gemma'])
    parser.add_argument('--skip-classification', action='store_true',
                        help='Skip Part 1 (Classification)')
    parser.add_argument('--skip-anova', action='store_true',
                        help='Skip Part 2 (Two-Way ANOVA)')
    parser.add_argument('--skip-permutation', action='store_true',
                        help='Skip permutation test (saves time)')
    parser.add_argument('--layers', type=int, nargs='+', default=None,
                        help='Specific layers to analyze (overrides defaults)')
    parser.add_argument('--viz-only', action='store_true',
                        help='Skip analysis, load existing JSON results and regenerate figures only')
    args = parser.parse_args()

    # Setup
    setup_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(OUTPUT_DIR / "logs")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    logger.info("=" * 70)
    logger.info("WITHIN-MODEL SAE ANALYSIS")
    logger.info(f"Models: {args.models}")
    logger.info(f"Viz-only: {args.viz_only}")
    logger.info(f"Skip classification: {args.skip_classification}")
    logger.info(f"Skip ANOVA: {args.skip_anova}")
    logger.info("=" * 70)

    # Collect per-model data for combined figures
    all_clf_data = {}   # model -> clf_data
    all_anova_data = {} # model -> anova_data

    if args.viz_only:
        # --viz-only: load existing JSON results
        logger.info("VIZ-ONLY MODE: Loading existing JSON results...")
        for model in args.models:
            clf_json = load_latest_json(RESULTS_DIR, 'classification', model)
            anova_json = load_latest_json(RESULTS_DIR, 'anova', model)

            if clf_json:
                all_clf_data[model] = clf_json
                logger.info(f"  Loaded classification for {model} (best_layer={clf_json.get('best_layer')})")
            else:
                logger.warning(f"  No classification JSON found for {model}")

            if anova_json:
                all_anova_data[model] = anova_json
                n_layers = len(anova_json.get('layer_summary', []))
                logger.info(f"  Loaded ANOVA for {model} ({n_layers} layers)")
            else:
                logger.warning(f"  No ANOVA JSON found for {model}")
    else:
        # Full analysis mode
        for model in args.models:
            logger.info(f"\n{'#' * 70}")
            logger.info(f"# Processing {model.upper()}")
            logger.info(f"{'#' * 70}")

            layers = args.layers or (LLAMA_LAYERS if model == 'llama' else GEMMA_LAYERS)
            games = load_game_results(model)
            meta = get_game_metadata(games)

            logger.info(f"  Games: {len(games)}")
            logger.info(f"  Outcomes: {dict(Counter(meta['outcomes']))}")
            logger.info(f"  Bet types: {dict(Counter(meta['bet_types']))}")

            # --- Part 1: Classification ---
            if not args.skip_classification:
                clf_data = run_classification_analysis(model, meta, layers, logger,
                                                        skip_permutation=args.skip_permutation)
                clf_path = RESULTS_DIR / f'classification_{model}_{timestamp}.json'
                save_classification_results(clf_data, clf_path)
                logger.info(f"  Classification results saved: {clf_path}")
                all_clf_data[model] = clf_data

            # --- Part 2: Two-Way ANOVA ---
            if not args.skip_anova:
                anova_data = run_anova_analysis(model, meta, layers, logger)
                anova_path = RESULTS_DIR / f'anova_{model}_{timestamp}.json'
                save_anova_results(anova_data, anova_path)
                logger.info(f"  ANOVA results saved: {anova_path}")
                all_anova_data[model] = anova_data

    # --- Generate all figures ---
    logger.info(f"\n{'=' * 70}")
    logger.info("GENERATING FIGURES")
    logger.info(f"{'=' * 70}")

    # Fig A: Combined AUC (both models, 1 figure)
    if all_clf_data:
        fig_a_layer_auc_combined(all_clf_data, FIGURE_DIR / 'fig_a_layer_auc_combined.png')

    # Fig B: Heatmap (per model)
    for model, anova in all_anova_data.items():
        fig_b_heatmap(anova, FIGURE_DIR / f'fig_b_heatmap_{model}.png')

    # Fig C: Overlap Venn (per model, needs both clf + anova)
    for model in args.models:
        if model in all_clf_data and model in all_anova_data:
            fig_c_overlap(all_clf_data[model], all_anova_data[model],
                          FIGURE_DIR / f'fig_c_overlap_{model}.png')

    # Fig D: η² summary (per model)
    for model, anova in all_anova_data.items():
        fig_d_eta_summary(anova, FIGURE_DIR / f'fig_d_eta_summary_{model}.png')

    # Fig E: Combined interaction % (both models, 1 figure)
    if all_anova_data:
        fig_e_interaction_combined(all_anova_data, FIGURE_DIR / 'fig_e_interaction_combined.png')

    logger.info(f"\n{'=' * 70}")
    logger.info("ALL COMPLETE")
    logger.info(f"Figures: {FIGURE_DIR}")
    logger.info(f"Results: {RESULTS_DIR}")
    logger.info(f"{'=' * 70}")


if __name__ == '__main__':
    main()
