#!/usr/bin/env python3
"""
IC SAE Analysis + Cross-Paradigm Comparison (SM vs IC)
======================================================

Module 1: IC Standalone Analysis — mirrors run_within_model_analysis.py
  1a. Classification (SAE → bankruptcy prediction, 5 subsets)
  1b. Two-Way ANOVA (bet_type × outcome interaction)

Module 2: IC-Specific Analyses
  2a. Constraint Effect (c30 vs c50 Welch t-test)
  2b. Constraint × Outcome ANOVA (2×2)
  2c. Stratified ANOVA (within c30/c50 separately)

Module 3: Cross-Paradigm Comparison (SM vs IC)
  - Layer-wise AUC comparison
  - Classification feature overlap (Jaccard + Spearman)
  - ANOVA interaction feature overlap

Usage:
    python run_ic_cross_paradigm_analysis.py
    python run_ic_cross_paradigm_analysis.py --skip-permutation
    python run_ic_cross_paradigm_analysis.py --viz-only
    python run_ic_cross_paradigm_analysis.py --layers 20 --skip-anova --skip-permutation
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

from scipy import stats
from scipy.stats import ttest_ind
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
import seaborn as sns

# Try to import venn2 (optional — fallback to text if unavailable)
try:
    from matplotlib_venn import venn2
    HAS_VENN = True
except ImportError:
    HAS_VENN = False


# ---------------------------------------------------------------------------
# JSON Encoder
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
IC_SAE_DIR = Path("/home/jovyan/beomi/llm-addiction-data/investment_choice_sae/gemma_42layers")
SM_SAE_DIR = Path("/home/jovyan/beomi/llm-addiction-data/hf-dataset/sae_patching/corrected_sae_analysis/gemma_full_42layers")
SM_JSON_PATH = Path("/home/jovyan/beomi/llm-addiction-data/hf-dataset/slot_machine/gemma/final_gemma_20251004_172426.json")
SM_RESULTS_DIR = Path("/home/jovyan/beomi/llm-addiction-data/sae_condition_comparison/within_model/results")

OUTPUT_DIR = Path("/home/jovyan/beomi/llm-addiction-data/sae_condition_comparison/ic_cross_paradigm")
FIGURE_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"

GEMMA_LAYERS = list(range(0, 42))

# Analysis parameters
MIN_ACTIVATION_RATE = 0.01
CLASSIFICATION_CV_FOLDS = 5
CLASSIFICATION_C = 1.0
ANOVA_PRESCREENING_K = 5000
FDR_ALPHA = 0.05
N_PERMUTATIONS = 1000
MIN_COHENS_D = 0.3


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass
class ParadigmData:
    """Container for paradigm-level SAE data + metadata."""
    name: str                       # 'ic' or 'sm'
    n_games: int
    outcomes: np.ndarray            # (N,) str: 'bankruptcy' / 'voluntary_stop'
    bet_types: np.ndarray           # (N,) str: 'variable' / 'fixed'
    bet_constraints: Optional[np.ndarray] = None   # IC only: '30' / '50'
    game_ids: Optional[np.ndarray] = None
    sae_dir: Path = None
    layers: List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'ic_cross_paradigm_{timestamp}.log'

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
# Data Loading
# ---------------------------------------------------------------------------
def load_ic_data() -> ParadigmData:
    """Load IC metadata from first NPZ (all layers share the same metadata)."""
    npz_path = IC_SAE_DIR / "layer_0_features.npz"
    data = np.load(npz_path, allow_pickle=True)
    return ParadigmData(
        name='ic',
        n_games=data['features'].shape[0],
        outcomes=data['outcomes'],
        bet_types=data['bet_types'],
        bet_constraints=data['bet_constraints'],
        game_ids=data['game_ids'],
        sae_dir=IC_SAE_DIR,
        layers=GEMMA_LAYERS,
    )


def load_sm_data() -> ParadigmData:
    """Load SM metadata from JSON (NPZ doesn't contain bet_type)."""
    with open(SM_JSON_PATH) as f:
        game_data = json.load(f)
    games = game_data['results']
    return ParadigmData(
        name='sm',
        n_games=len(games),
        outcomes=np.array([g['outcome'] for g in games]),
        bet_types=np.array([g['bet_type'] for g in games]),
        bet_constraints=None,
        game_ids=np.arange(len(games)),
        sae_dir=SM_SAE_DIR,
        layers=GEMMA_LAYERS,
    )


def load_npz_features(sae_dir: Path, layer: int) -> Optional[np.ndarray]:
    """Load SAE features for a single layer."""
    npz_path = sae_dir / f"layer_{layer}_features.npz"
    if not npz_path.exists():
        return None
    data = np.load(npz_path, allow_pickle=True)
    return data['features']


def load_sm_results() -> Tuple[Optional[Dict], Optional[Dict]]:
    """Load latest SM classification + ANOVA result JSONs."""
    def load_latest(prefix: str) -> Optional[Dict]:
        pattern = f"{prefix}_gemma_*.json"
        matches = sorted(SM_RESULTS_DIR.glob(pattern))
        if not matches:
            return None
        with open(matches[-1]) as f:
            return json.load(f)

    return load_latest('classification'), load_latest('anova')


# ---------------------------------------------------------------------------
# Core Analysis Functions (from run_within_model_analysis.py)
# ---------------------------------------------------------------------------
def filter_sparse_features(features: np.ndarray, min_rate: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """Filter features with activation rate below threshold."""
    activation_rate = (features != 0).mean(axis=0)
    active_mask = activation_rate >= min_rate
    active_indices = np.where(active_mask)[0]
    return features[:, active_mask], active_indices


def fdr_correction(p_values, alpha=0.05):
    if len(p_values) == 0:
        return np.array([]), np.array([])
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
    return reject, pvals_corrected


def run_classification_layer(
    features: np.ndarray,
    active_indices: np.ndarray,
    labels: np.ndarray,
    n_folds: int = 5,
    C: float = 1.0,
    logger: logging.Logger = None,
) -> Dict:
    """5-fold stratified CV logistic regression for one layer."""
    n_samples, n_features = features.shape
    n_pos = labels.sum()
    n_neg = n_samples - n_pos

    if n_pos < n_folds or n_neg < n_folds:
        if logger:
            logger.warning(f"  Too few samples: pos={n_pos}, neg={n_neg}")
        return {'auc_mean': 0.5, 'auc_std': 0.0, 'accuracy_mean': 0.5, 'f1_mean': 0.0,
                'n_features': n_features, 'n_pos': int(n_pos), 'n_neg': int(n_neg),
                'top_features': [], 'skipped': True}

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

    top_k = min(50, n_features)
    top_idx = np.argsort(np.abs(all_coefs))[-top_k:][::-1]
    top_features = [
        {'feature_idx': int(active_indices[i]), 'coef': float(all_coefs[i])}
        for i in top_idx
    ]

    return {
        'auc_mean': float(np.mean(aucs)),
        'auc_std': float(np.std(aucs)),
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


def fast_f_test_groups(features: np.ndarray, group_labels: np.ndarray) -> np.ndarray:
    """Vectorized one-way F-test across all features simultaneously."""
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

    ms_between = ss_between / max(k - 1, 1)
    ms_within = ss_within / max(n - k, 1)
    ms_within[ms_within == 0] = 1e-10

    return ms_between / ms_within


def run_anova_single_feature(
    activations: np.ndarray, factor_a: np.ndarray, factor_b: np.ndarray,
    factor_a_name: str = 'bet_type', factor_b_name: str = 'outcome',
) -> Optional[Dict]:
    """Run two-way ANOVA for a single feature using statsmodels."""
    df = pd.DataFrame({
        'activation': activations,
        'A': pd.Categorical(factor_a),
        'B': pd.Categorical(factor_b),
    })

    try:
        model = ols('activation ~ C(A) + C(B) + C(A):C(B)', data=df).fit()
        table = anova_lm(model, typ=2)

        ss_total = table['sum_sq'].sum()
        if ss_total == 0:
            return None

        interaction = table.loc['C(A):C(B)', :]
        row_a = table.loc['C(A)', :]
        row_b = table.loc['C(B)', :]

        # Group means
        group_means = {}
        for a_val in np.unique(factor_a):
            for b_val in np.unique(factor_b):
                mask = (factor_a == a_val) & (factor_b == b_val)
                label = f'{a_val}_{b_val}'
                group_means[label] = float(activations[mask].mean()) if mask.sum() > 0 else 0.0

        return {
            'interaction_p': float(interaction['PR(>F)']),
            'interaction_f': float(interaction['F']),
            'interaction_eta_sq': float(interaction['sum_sq'] / ss_total),
            f'{factor_a_name}_p': float(row_a['PR(>F)']),
            f'{factor_a_name}_eta_sq': float(row_a['sum_sq'] / ss_total),
            f'{factor_b_name}_p': float(row_b['PR(>F)']),
            f'{factor_b_name}_eta_sq': float(row_b['sum_sq'] / ss_total),
            'group_means': group_means,
        }
    except Exception:
        return None


def run_anova_layer(
    features: np.ndarray,
    active_indices: np.ndarray,
    factor_a: np.ndarray,
    factor_b: np.ndarray,
    prescreening_k: int = 5000,
    factor_a_name: str = 'bet_type',
    factor_b_name: str = 'outcome',
    logger: logging.Logger = None,
) -> List[Dict]:
    """Run two-way ANOVA for one layer with F-test pre-screening."""
    n_samples, n_features = features.shape

    # Encode 4 groups for pre-screening
    a_enc = (factor_a == np.unique(factor_a)[1]).astype(int) if len(np.unique(factor_a)) == 2 else factor_a.astype(int)
    b_enc = (factor_b == np.unique(factor_b)[1]).astype(int) if len(np.unique(factor_b)) == 2 else factor_b.astype(int)
    group_4way = a_enc * 2 + b_enc

    f_values = fast_f_test_groups(features, group_4way)
    n_to_screen = min(prescreening_k, n_features)
    top_f_idx = np.argsort(f_values)[-n_to_screen:]

    if logger:
        logger.info(f"    Pre-screening: {n_features} -> {n_to_screen} features (top by F-test)")

    results = []
    for local_idx in top_f_idx:
        activations = features[:, local_idx]
        original_idx = int(active_indices[local_idx])

        r = run_anova_single_feature(activations, factor_a, factor_b, factor_a_name, factor_b_name)
        if r is not None:
            r['feature_idx'] = original_idx
            r['local_idx'] = int(local_idx)
            results.append(r)

    return results


# ---------------------------------------------------------------------------
# Module 1: IC Standalone Analysis
# ---------------------------------------------------------------------------
def module1_classification(
    ic: ParadigmData, layers: List[int], logger: logging.Logger,
    skip_permutation: bool = False,
) -> Dict:
    """Module 1a: Classification — SAE features → bankruptcy prediction."""
    logger.info("=" * 70)
    logger.info("[Module 1a] IC Classification: SAE -> Bankruptcy")
    logger.info("=" * 70)

    labels = (ic.outcomes == 'bankruptcy').astype(int)
    var_mask = ic.bet_types == 'variable'
    fix_mask = ic.bet_types == 'fixed'
    c30_mask = ic.bet_constraints == '30'
    c50_mask = ic.bet_constraints == '50'

    logger.info(f"  Total: {len(labels)}, BK: {labels.sum()}, VS: {(1-labels).sum()}")
    logger.info(f"  Variable: {var_mask.sum()} (BK={labels[var_mask].sum()})")
    logger.info(f"  Fixed: {fix_mask.sum()} (BK={labels[fix_mask].sum()})")
    logger.info(f"  c30: {c30_mask.sum()} (BK={labels[c30_mask].sum()})")
    logger.info(f"  c50: {c50_mask.sum()} (BK={labels[c50_mask].sum()})")

    subsets = {
        'all_games': np.ones(len(labels), dtype=bool),
        'variable_only': var_mask,
        'fixed_only': fix_mask,
        'c30_only': c30_mask,
        'c50_only': c50_mask,
    }

    results_by_subset = {name: [] for name in subsets}
    best_layer = None
    best_auc = 0.0

    for layer in layers:
        features_raw = load_npz_features(ic.sae_dir, layer)
        if features_raw is None:
            logger.warning(f"  Layer {layer}: NPZ not found, skipping")
            continue

        features, active_idx = filter_sparse_features(features_raw, MIN_ACTIVATION_RATE)
        n_active = features.shape[1]

        if n_active < 10:
            logger.warning(f"  Layer {layer}: only {n_active} active features, skipping")
            continue

        logger.info(f"  Layer {layer}: {features_raw.shape[1]} -> {n_active} features")

        for subset_name, mask in subsets.items():
            subset_labels = labels[mask]
            if subset_labels.sum() < CLASSIFICATION_CV_FOLDS:
                continue

            result = run_classification_layer(
                features[mask], active_idx, subset_labels,
                CLASSIFICATION_CV_FOLDS, CLASSIFICATION_C, logger
            )
            result['layer'] = layer
            results_by_subset[subset_name].append(result)

            if subset_name == 'all_games' and not result['skipped'] and result['auc_mean'] > best_auc:
                best_auc = result['auc_mean']
                best_layer = layer

        # Log all-games AUC for this layer
        all_result = results_by_subset['all_games'][-1] if results_by_subset['all_games'] else None
        if all_result and not all_result.get('skipped', True):
            logger.info(f"    AUC(all)={all_result['auc_mean']:.4f}+/-{all_result['auc_std']:.4f}")

    # Permutation test on best layer
    permutation_result = None
    if best_layer is not None and not skip_permutation:
        logger.info(f"\n  Permutation test on best layer {best_layer} (AUC={best_auc:.4f})...")
        features_raw = load_npz_features(ic.sae_dir, best_layer)
        features, _ = filter_sparse_features(features_raw, MIN_ACTIVATION_RATE)
        permutation_result = run_classification_permutation(
            features, labels, best_auc, N_PERMUTATIONS, CLASSIFICATION_CV_FOLDS, CLASSIFICATION_C
        )
        permutation_result['layer'] = best_layer
        logger.info(f"    p={permutation_result['p_value']:.4f}, "
                     f"perm AUC={permutation_result['perm_mean']:.4f}+/-{permutation_result['perm_std']:.4f}")

    return {
        'paradigm': 'ic',
        'model': 'gemma',
        **results_by_subset,
        'best_layer': best_layer,
        'best_auc': float(best_auc),
        'permutation_test': permutation_result,
    }


def module1_anova(
    ic: ParadigmData, layers: List[int], logger: logging.Logger
) -> Dict:
    """Module 1b: Two-Way ANOVA — bet_type x outcome interaction."""
    logger.info("=" * 70)
    logger.info("[Module 1b] IC Two-Way ANOVA: bet_type x outcome")
    logger.info("=" * 70)

    bet_types = ic.bet_types
    outcomes = ic.outcomes

    # Encode for ANOVA
    bt_enc = (bet_types == 'fixed').astype(int)     # 0=variable, 1=fixed
    oc_enc = (outcomes == 'voluntary_stop').astype(int)  # 0=bankrupt, 1=safe

    logger.info(f"  Design: 2x2 (Variable/Fixed x Bankrupt/Safe)")
    logger.info(f"  Cells: VB={(bt_enc==0)&(oc_enc==0)}, VS={(bt_enc==0)&(oc_enc==1)}, "
                f"FB={(bt_enc==1)&(oc_enc==0)}, FS={(bt_enc==1)&(oc_enc==1)}")
    logger.info(f"  Cell counts: VB={((bt_enc==0)&(oc_enc==0)).sum()}, "
                f"VS={((bt_enc==0)&(oc_enc==1)).sum()}, "
                f"FB={((bt_enc==1)&(oc_enc==0)).sum()}, "
                f"FS={((bt_enc==1)&(oc_enc==1)).sum()}")

    all_layer_results = []

    for layer in layers:
        features_raw = load_npz_features(ic.sae_dir, layer)
        if features_raw is None:
            continue

        features, active_idx = filter_sparse_features(features_raw, MIN_ACTIVATION_RATE)
        if features.shape[1] < 10:
            logger.warning(f"  Layer {layer}: {features.shape[1]} active features, skipping")
            continue

        logger.info(f"  Layer {layer}: {features_raw.shape[1]} -> {features.shape[1]} active features")

        layer_results = run_anova_layer(
            features, active_idx, bt_enc, oc_enc,
            ANOVA_PRESCREENING_K, 'bet_type', 'outcome', logger
        )

        if layer_results:
            interaction_ps = np.array([r['interaction_p'] for r in layer_results])
            reject, p_fdr = fdr_correction(interaction_ps, FDR_ALPHA)

            n_sig = reject.sum()
            for i, r in enumerate(layer_results):
                r['interaction_p_fdr'] = float(p_fdr[i])
                r['interaction_significant'] = bool(reject[i])

            sig_results = [r for r in layer_results if r['interaction_significant']]
            sig_results.sort(key=lambda x: x['interaction_eta_sq'], reverse=True)

            logger.info(f"    ANOVA: {len(layer_results)} tested, {n_sig} significant (FDR<{FDR_ALPHA})")
        else:
            sig_results = []
            n_sig = 0

        all_layer_results.append({
            'layer': layer,
            'n_tested': len(layer_results),
            'n_significant': n_sig,
            'significant_features': sig_results[:100],
            'all_results': layer_results,
        })

    return {
        'paradigm': 'ic',
        'model': 'gemma',
        'layer_results': all_layer_results,
    }


# ---------------------------------------------------------------------------
# Module 2: IC-Specific Analyses
# ---------------------------------------------------------------------------
def module2a_constraint_welch(
    ic: ParadigmData, layers: List[int], logger: logging.Logger
) -> Dict:
    """Module 2a: Constraint effect — c30 vs c50 Welch t-test per feature."""
    logger.info("=" * 70)
    logger.info("[Module 2a] Constraint Effect: c30 vs c50 (Welch t-test)")
    logger.info("=" * 70)

    c30_mask = ic.bet_constraints == '30'
    c50_mask = ic.bet_constraints == '50'
    logger.info(f"  c30: {c30_mask.sum()}, c50: {c50_mask.sum()}")

    all_layer_results = []

    for layer in layers:
        features_raw = load_npz_features(ic.sae_dir, layer)
        if features_raw is None:
            continue

        features, active_idx = filter_sparse_features(features_raw, MIN_ACTIVATION_RATE)
        if features.shape[1] < 10:
            continue

        c30_feats = features[c30_mask]
        c50_feats = features[c50_mask]

        # Vectorized Welch t-test
        t_result = ttest_ind(c30_feats, c50_feats, axis=0, equal_var=False)
        t_stats = t_result.statistic
        p_values = t_result.pvalue

        # Handle NaN p-values
        nan_mask = np.isnan(p_values)
        p_values[nan_mask] = 1.0

        # Cohen's d
        n1, n2 = c30_feats.shape[0], c50_feats.shape[0]
        m1, m2 = c30_feats.mean(axis=0), c50_feats.mean(axis=0)
        s1, s2 = c30_feats.std(axis=0, ddof=1), c50_feats.std(axis=0, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        pooled_std[pooled_std == 0] = 1e-10
        cohens_d = (m1 - m2) / pooled_std

        # FDR correction
        reject, p_fdr = fdr_correction(p_values, FDR_ALPHA)

        # Filter by |d| >= threshold
        sig_mask = reject & (np.abs(cohens_d) >= MIN_COHENS_D)
        n_sig = sig_mask.sum()
        n_c30_up = (sig_mask & (cohens_d > 0)).sum()
        n_c50_up = (sig_mask & (cohens_d < 0)).sum()

        logger.info(f"  Layer {layer}: {features.shape[1]} features, "
                     f"{reject.sum()} FDR-sig, {n_sig} sig+|d|>={MIN_COHENS_D} "
                     f"(c30-up={n_c30_up}, c50-up={n_c50_up})")

        # Top significant features
        sig_indices = np.where(sig_mask)[0]
        top_features = []
        for idx in sig_indices:
            top_features.append({
                'feature_idx': int(active_idx[idx]),
                'local_idx': int(idx),
                't_stat': float(t_stats[idx]),
                'p_value': float(p_values[idx]),
                'p_fdr': float(p_fdr[idx]),
                'cohens_d': float(cohens_d[idx]),
                'direction': 'c30_up' if cohens_d[idx] > 0 else 'c50_up',
                'c30_mean': float(m1[idx]),
                'c50_mean': float(m2[idx]),
            })
        top_features.sort(key=lambda x: abs(x['cohens_d']), reverse=True)

        all_layer_results.append({
            'layer': layer,
            'n_tested': int(features.shape[1]),
            'n_fdr_sig': int(reject.sum()),
            'n_sig_with_d': int(n_sig),
            'n_c30_up': int(n_c30_up),
            'n_c50_up': int(n_c50_up),
            'top_features': top_features[:50],
        })

    return {
        'paradigm': 'ic',
        'model': 'gemma',
        'layer_results': all_layer_results,
    }


def module2b_constraint_outcome_anova(
    ic: ParadigmData, layers: List[int], logger: logging.Logger
) -> Dict:
    """Module 2b: Constraint x Outcome ANOVA (2x2: c30/c50 x BK/VS)."""
    logger.info("=" * 70)
    logger.info("[Module 2b] Constraint x Outcome ANOVA")
    logger.info("=" * 70)

    constraints = ic.bet_constraints
    outcomes = ic.outcomes

    c_enc = (constraints == '50').astype(int)  # 0=c30, 1=c50
    o_enc = (outcomes == 'voluntary_stop').astype(int)  # 0=BK, 1=VS

    cell_counts = {
        'c30_BK': ((c_enc == 0) & (o_enc == 0)).sum(),
        'c30_VS': ((c_enc == 0) & (o_enc == 1)).sum(),
        'c50_BK': ((c_enc == 1) & (o_enc == 0)).sum(),
        'c50_VS': ((c_enc == 1) & (o_enc == 1)).sum(),
    }
    logger.info(f"  Cell counts: {cell_counts}")

    all_layer_results = []

    for layer in layers:
        features_raw = load_npz_features(ic.sae_dir, layer)
        if features_raw is None:
            continue

        features, active_idx = filter_sparse_features(features_raw, MIN_ACTIVATION_RATE)
        if features.shape[1] < 10:
            continue

        logger.info(f"  Layer {layer}: {features.shape[1]} active features")

        layer_results = run_anova_layer(
            features, active_idx, c_enc, o_enc,
            ANOVA_PRESCREENING_K, 'constraint', 'outcome', logger
        )

        if layer_results:
            interaction_ps = np.array([r['interaction_p'] for r in layer_results])
            reject, p_fdr = fdr_correction(interaction_ps, FDR_ALPHA)
            n_sig = reject.sum()

            for i, r in enumerate(layer_results):
                r['interaction_p_fdr'] = float(p_fdr[i])
                r['interaction_significant'] = bool(reject[i])

            sig_results = [r for r in layer_results if r['interaction_significant']]
            sig_results.sort(key=lambda x: x['interaction_eta_sq'], reverse=True)

            logger.info(f"    {len(layer_results)} tested, {n_sig} significant interactions")
        else:
            sig_results = []
            n_sig = 0

        all_layer_results.append({
            'layer': layer,
            'n_tested': len(layer_results),
            'n_significant': n_sig,
            'significant_features': sig_results[:100],
            'all_results': layer_results,
        })

    return {
        'paradigm': 'ic',
        'model': 'gemma',
        'cell_counts': cell_counts,
        'layer_results': all_layer_results,
    }


def module2c_stratified_anova(
    ic: ParadigmData, layers: List[int], logger: logging.Logger
) -> Dict:
    """Module 2c: Stratified ANOVA — bet_type x outcome within c30 / c50 separately."""
    logger.info("=" * 70)
    logger.info("[Module 2c] Stratified ANOVA: bet_type x outcome within c30/c50")
    logger.info("=" * 70)

    c30_mask = ic.bet_constraints == '30'
    c50_mask = ic.bet_constraints == '50'

    strata = {'c30': c30_mask, 'c50': c50_mask}
    results_by_stratum = {}

    for stratum_name, stratum_mask in strata.items():
        bt_enc = (ic.bet_types[stratum_mask] == 'fixed').astype(int)
        oc_enc = (ic.outcomes[stratum_mask] == 'voluntary_stop').astype(int)

        cell_counts = {
            'var_BK': ((bt_enc == 0) & (oc_enc == 0)).sum(),
            'var_VS': ((bt_enc == 0) & (oc_enc == 1)).sum(),
            'fix_BK': ((bt_enc == 1) & (oc_enc == 0)).sum(),
            'fix_VS': ((bt_enc == 1) & (oc_enc == 1)).sum(),
        }
        min_cell = min(cell_counts.values())
        cell_warning = min_cell < 10

        logger.info(f"  {stratum_name}: cells={cell_counts}, min_cell={min_cell}"
                     f"{' (WARNING: small cell)' if cell_warning else ''}")

        layer_results = []

        for layer in layers:
            features_raw = load_npz_features(ic.sae_dir, layer)
            if features_raw is None:
                continue

            features, active_idx = filter_sparse_features(features_raw, MIN_ACTIVATION_RATE)
            if features.shape[1] < 10:
                continue

            feats_stratum = features[stratum_mask]
            anova_results = run_anova_layer(
                feats_stratum, active_idx, bt_enc, oc_enc,
                ANOVA_PRESCREENING_K, 'bet_type', 'outcome', logger
            )

            if anova_results:
                interaction_ps = np.array([r['interaction_p'] for r in anova_results])
                reject, p_fdr = fdr_correction(interaction_ps, FDR_ALPHA)
                n_sig = reject.sum()

                for i, r in enumerate(anova_results):
                    r['interaction_p_fdr'] = float(p_fdr[i])
                    r['interaction_significant'] = bool(reject[i])

                sig_results = [r for r in anova_results if r['interaction_significant']]
                sig_results.sort(key=lambda x: x['interaction_eta_sq'], reverse=True)
            else:
                sig_results = []
                n_sig = 0

            layer_results.append({
                'layer': layer,
                'n_tested': len(anova_results),
                'n_significant': n_sig,
                'significant_features': sig_results[:50],
            })

        results_by_stratum[stratum_name] = {
            'cell_counts': cell_counts,
            'cell_warning': cell_warning,
            'layer_results': layer_results,
        }

    return {
        'paradigm': 'ic',
        'model': 'gemma',
        'strata': results_by_stratum,
    }


# ---------------------------------------------------------------------------
# Module 3: Cross-Paradigm Comparison
# ---------------------------------------------------------------------------
def module3_cross_paradigm(
    ic_clf: Dict, ic_anova: Dict,
    sm_clf: Dict, sm_anova: Dict,
    logger: logging.Logger,
) -> Dict:
    """Module 3: Cross-paradigm comparison (SM vs IC)."""
    logger.info("=" * 70)
    logger.info("[Module 3] Cross-Paradigm Comparison: SM vs IC")
    logger.info("=" * 70)

    results = {
        'auc_comparison': {},
        'clf_overlap': {},
        'anova_overlap': {},
        'summary': {},
    }

    # --- AUC comparison ---
    ic_layers_auc = {}
    for r in ic_clf.get('all_games', []):
        if not r.get('skipped', False):
            ic_layers_auc[r['layer']] = {'mean': r['auc_mean'], 'std': r['auc_std']}

    sm_layers_auc = {}
    for r in sm_clf.get('all_games', []):
        if not r.get('skipped', False):
            sm_layers_auc[r['layer']] = {'mean': r['auc_mean'], 'std': r['auc_std']}

    common_layers = sorted(set(ic_layers_auc.keys()) & set(sm_layers_auc.keys()))
    logger.info(f"  Common layers: {len(common_layers)}")

    auc_comparison_layers = []
    for layer in common_layers:
        auc_comparison_layers.append({
            'layer': layer,
            'ic_auc': ic_layers_auc[layer]['mean'],
            'ic_std': ic_layers_auc[layer]['std'],
            'sm_auc': sm_layers_auc[layer]['mean'],
            'sm_std': sm_layers_auc[layer]['std'],
            'delta': ic_layers_auc[layer]['mean'] - sm_layers_auc[layer]['mean'],
        })

    results['auc_comparison'] = {
        'layers': auc_comparison_layers,
        'ic_best_layer': ic_clf.get('best_layer'),
        'ic_best_auc': ic_clf.get('best_auc'),
        'sm_best_layer': sm_clf.get('best_layer'),
        'sm_best_auc': sm_clf.get('best_auc'),
    }

    if common_layers:
        ic_aucs = [ic_layers_auc[l]['mean'] for l in common_layers]
        sm_aucs = [sm_layers_auc[l]['mean'] for l in common_layers]
        logger.info(f"  IC mean AUC: {np.mean(ic_aucs):.4f}, SM mean AUC: {np.mean(sm_aucs):.4f}")

    # --- Classification feature overlap (per-layer Jaccard + Spearman) ---
    # Build feature set maps: layer -> set of top-50 feature_idx
    def get_top_features_map(clf_data: Dict, subset_key: str = 'all_games') -> Dict[int, List]:
        """Returns {layer: [{'feature_idx': ..., 'coef': ...}, ...]}"""
        result = {}
        for r in clf_data.get(subset_key, []):
            if not r.get('skipped', False):
                result[r['layer']] = r.get('top_features', [])
        return result

    ic_top_map = get_top_features_map(ic_clf)
    sm_top_map = get_top_features_map(sm_clf)

    clf_overlap_layers = []
    for layer in common_layers:
        ic_feats = ic_top_map.get(layer, [])
        sm_feats = sm_top_map.get(layer, [])

        ic_set = {f['feature_idx'] for f in ic_feats}
        sm_set = {f['feature_idx'] for f in sm_feats}

        overlap = ic_set & sm_set
        union = ic_set | sm_set
        jaccard = len(overlap) / max(1, len(union))

        # Spearman correlation on |coef| for shared features
        spearman_rho = None
        if overlap:
            ic_coef_map = {f['feature_idx']: abs(f['coef']) for f in ic_feats}
            sm_coef_map = {f['feature_idx']: abs(f['coef']) for f in sm_feats}
            shared = sorted(overlap)
            ic_vals = [ic_coef_map[f] for f in shared]
            sm_vals = [sm_coef_map[f] for f in shared]
            if len(shared) >= 3:
                rho, p = stats.spearmanr(ic_vals, sm_vals)
                spearman_rho = {'rho': float(rho), 'p': float(p), 'n': len(shared)}

        clf_overlap_layers.append({
            'layer': layer,
            'ic_n': len(ic_set),
            'sm_n': len(sm_set),
            'overlap': len(overlap),
            'jaccard': float(jaccard),
            'spearman': spearman_rho,
        })

    results['clf_overlap'] = {'layers': clf_overlap_layers}

    if clf_overlap_layers:
        jaccards = [l['jaccard'] for l in clf_overlap_layers]
        logger.info(f"  Clf Jaccard: mean={np.mean(jaccards):.4f}, max={np.max(jaccards):.4f}")

    # --- ANOVA interaction feature overlap ---
    def get_anova_sig_map(anova_data: Dict) -> Dict[int, set]:
        """Returns {layer: set(feature_idx)} for significant interaction features."""
        result = {}
        layer_key = 'layer_results' if 'layer_results' in anova_data else 'layer_summary'
        for lr in anova_data.get(layer_key, []):
            feat_key = 'significant_features' if 'significant_features' in lr else 'top_features'
            sig_feats = set()
            for f in lr.get(feat_key, []):
                if f.get('interaction_significant', True):
                    sig_feats.add(f['feature_idx'])
            result[lr['layer']] = sig_feats
        return result

    ic_anova_map = get_anova_sig_map(ic_anova)
    sm_anova_map = get_anova_sig_map(sm_anova)

    anova_overlap_layers = []
    for layer in common_layers:
        ic_set = ic_anova_map.get(layer, set())
        sm_set = sm_anova_map.get(layer, set())

        overlap = ic_set & sm_set
        union = ic_set | sm_set
        jaccard = len(overlap) / max(1, len(union))

        anova_overlap_layers.append({
            'layer': layer,
            'ic_n': len(ic_set),
            'sm_n': len(sm_set),
            'overlap': len(overlap),
            'jaccard': float(jaccard),
        })

    results['anova_overlap'] = {'layers': anova_overlap_layers}

    if anova_overlap_layers:
        jaccards = [l['jaccard'] for l in anova_overlap_layers]
        logger.info(f"  ANOVA Jaccard: mean={np.mean(jaccards):.4f}, max={np.max(jaccards):.4f}")

    # --- Summary statistics ---
    ic_bk_rate = ic_clf.get('all_games', [{}])[0].get('n_pos', 0) / max(
        ic_clf.get('all_games', [{}])[0].get('n_pos', 0) + ic_clf.get('all_games', [{}])[0].get('n_neg', 1), 1
    ) if ic_clf.get('all_games') else 0

    sm_bk_rate = sm_clf.get('all_games', [{}])[0].get('n_pos', 0) / max(
        sm_clf.get('all_games', [{}])[0].get('n_pos', 0) + sm_clf.get('all_games', [{}])[0].get('n_neg', 1), 1
    ) if sm_clf.get('all_games') else 0

    # Count total significant ANOVA features
    ic_total_sig = sum(lr.get('n_significant', 0) for lr in ic_anova.get('layer_results', ic_anova.get('layer_summary', [])))
    sm_total_sig = sum(lr.get('n_significant', 0) for lr in sm_anova.get('layer_results', sm_anova.get('layer_summary', [])))

    results['summary'] = {
        'ic': {
            'n_games': ic_clf.get('all_games', [{}])[0].get('n_pos', 0) + ic_clf.get('all_games', [{}])[0].get('n_neg', 0) if ic_clf.get('all_games') else 0,
            'bankruptcy_rate': float(ic_bk_rate),
            'best_layer': ic_clf.get('best_layer'),
            'best_auc': ic_clf.get('best_auc'),
            'total_anova_sig': ic_total_sig,
        },
        'sm': {
            'n_games': sm_clf.get('all_games', [{}])[0].get('n_pos', 0) + sm_clf.get('all_games', [{}])[0].get('n_neg', 0) if sm_clf.get('all_games') else 0,
            'bankruptcy_rate': float(sm_bk_rate),
            'best_layer': sm_clf.get('best_layer'),
            'best_auc': sm_clf.get('best_auc'),
            'total_anova_sig': sm_total_sig,
        },
    }

    logger.info(f"  Summary: IC(N={results['summary']['ic']['n_games']}, BK={ic_bk_rate:.1%}, "
                f"AUC={ic_clf.get('best_auc', 0):.4f}), "
                f"SM(N={results['summary']['sm']['n_games']}, BK={sm_bk_rate:.1%}, "
                f"AUC={sm_clf.get('best_auc', 0):.4f})")

    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def setup_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'axes.labelsize': 11,
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
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
    })


COLORS = {
    'ic': '#27ae60',          # green for IC
    'sm': '#c0392b',          # red for SM
    'all': '#2c3e50',
    'variable': '#e74c3c',
    'fixed': '#3498db',
    'c30': '#f39c12',
    'c50': '#8e44ad',
    'median': '#e74c3c',
    'p75': '#f39c12',
    'max': '#8e44ad',
    'iqr_fill': '#3498db',
}


def _save_fig(fig, path):
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path.name}")


# --- Module 1 Figures ---

def ic_fig_a_layer_auc(clf_data: Dict, save_path: Path):
    """IC Fig A: Layer-wise AUC for 5 subsets."""
    fig, ax = plt.subplots(figsize=(10, 5))

    subset_styles = {
        'all_games': {'color': COLORS['all'], 'lw': 2.5, 'ls': '-', 'label': 'All games'},
        'variable_only': {'color': COLORS['variable'], 'lw': 1.2, 'ls': '--', 'label': 'Variable'},
        'fixed_only': {'color': COLORS['fixed'], 'lw': 1.2, 'ls': '--', 'label': 'Fixed'},
        'c30_only': {'color': COLORS['c30'], 'lw': 1.2, 'ls': '-.', 'label': 'c30'},
        'c50_only': {'color': COLORS['c50'], 'lw': 1.2, 'ls': '-.', 'label': 'c50'},
    }

    for subset_name, style in subset_styles.items():
        results = [r for r in clf_data.get(subset_name, []) if not r.get('skipped', False)]
        if not results:
            continue
        layers = [r['layer'] for r in results]
        aucs = [r['auc_mean'] for r in results]
        stds = [r['auc_std'] for r in results]

        ax.plot(layers, aucs, linestyle=style['ls'], color=style['color'],
                lw=style['lw'], label=style['label'])
        if subset_name == 'all_games':
            aucs_arr = np.array(aucs)
            stds_arr = np.array(stds)
            ax.fill_between(layers, aucs_arr - stds_arr, np.minimum(aucs_arr + stds_arr, 1.0),
                            alpha=0.15, color=style['color'])

    ax.axhline(y=0.5, color='gray', ls=':', alpha=0.5, label='Chance')
    ax.set_xlabel('Layer')
    ax.set_ylabel('AUC-ROC')
    ax.set_title('IC: Layer-wise Classification AUC (Gemma-2-9B)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(axis='y')

    _save_fig(fig, save_path)


def ic_fig_b_heatmap(anova_data: Dict, save_path: Path):
    """IC Fig B: Top-10 interaction feature 4-cell Z-score heatmap."""
    layer_key = 'layer_results' if 'layer_results' in anova_data else 'layer_summary'
    layer_data = anova_data[layer_key]

    all_sig = []
    for lr in layer_data:
        feat_key = 'significant_features' if 'significant_features' in lr else 'top_features'
        for feat in lr.get(feat_key, []):
            if feat.get('interaction_significant', True):
                feat_copy = dict(feat)
                feat_copy['layer'] = lr['layer']
                all_sig.append(feat_copy)

    if not all_sig:
        print("  No significant interactions for IC heatmap, skipping")
        return

    all_sig.sort(key=lambda x: x['interaction_eta_sq'], reverse=True)
    top_n = min(10, len(all_sig))
    top_feats = all_sig[:top_n]

    # IC ANOVA uses encoded factor values: 0/1 for bet_type, 0/1 for outcome
    cells = ['0_0', '0_1', '1_0', '1_1']
    cell_labels = ['Var-BK', 'Var-VS', 'Fix-BK', 'Fix-VS']
    matrix = np.zeros((top_n, 4))

    for i, feat in enumerate(top_feats):
        gm = feat['group_means']
        for j, cell in enumerate(cells):
            matrix[i, j] = gm.get(cell, 0.0)

    # Z-score normalize rows
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

    ax.set_title('IC: Top-10 Interaction Features (bet_type x outcome)')

    _save_fig(fig, save_path)


def ic_fig_c_overlap(clf_data: Dict, anova_data: Dict, save_path: Path):
    """IC Fig C: Venn diagram — Classification top-50 vs ANOVA sig at best layer."""
    best_layer = clf_data.get('best_layer')
    if best_layer is None:
        print("  No best layer for Venn diagram, skipping")
        return

    # Clf top features at best layer
    clf_top_set = set()
    for r in clf_data.get('all_games', []):
        if r.get('layer') == best_layer and not r.get('skipped', False):
            clf_top_set = {f['feature_idx'] for f in r.get('top_features', [])}
            break

    # ANOVA sig features at best layer
    layer_key = 'layer_results' if 'layer_results' in anova_data else 'layer_summary'
    anova_sig_set = set()
    for lr in anova_data.get(layer_key, []):
        if lr['layer'] == best_layer:
            feat_key = 'significant_features' if 'significant_features' in lr else 'top_features'
            anova_sig_set = {f['feature_idx'] for f in lr.get(feat_key, [])
                            if f.get('interaction_significant', True)}
            break

    if not clf_top_set and not anova_sig_set:
        print("  No features for Venn diagram")
        return

    overlap = clf_top_set & anova_sig_set
    union = clf_top_set | anova_sig_set
    jaccard = len(overlap) / max(1, len(union))

    fig, ax = plt.subplots(figsize=(6, 5.5))

    if HAS_VENN:
        try:
            v = venn2(
                [clf_top_set, anova_sig_set],
                set_labels=('Clf Top-50', 'ANOVA Interaction'),
                ax=ax
            )
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
            ax.text(0.5, 0.5, f'Clf: {len(clf_top_set)} | ANOVA: {len(anova_sig_set)} | Overlap: {len(overlap)}',
                    transform=ax.transAxes, ha='center', va='center', fontsize=12)
    else:
        ax.text(0.5, 0.5, f'Clf: {len(clf_top_set)} | ANOVA: {len(anova_sig_set)} | Overlap: {len(overlap)}',
                transform=ax.transAxes, ha='center', va='center', fontsize=12)

    ax.set_title(f'IC Gemma-2-9B — Layer {best_layer}', pad=12)
    ax.text(0.5, -0.08, f'Jaccard = {jaccard:.3f}',
            transform=ax.transAxes, ha='center', fontsize=12, fontweight='bold')

    fig.subplots_adjust(bottom=0.08)
    _save_fig(fig, save_path)


def ic_fig_d_eta_summary(anova_data: Dict, save_path: Path):
    """IC Fig D: Layer-wise interaction eta-squared (median/75th/max)."""
    layer_key = 'layer_results' if 'layer_results' in anova_data else 'layer_summary'
    layer_data = anova_data[layer_key]

    layers, medians, q25s, p75s, maxes = [], [], [], [], []

    for lr in layer_data:
        results_to_use = lr.get('all_results', lr.get('significant_features', lr.get('top_features', [])))
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

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(layers, q25s, p75s, alpha=0.2, color=COLORS['iqr_fill'], label='IQR')
    ax.plot(layers, medians, '-', color=COLORS['median'], lw=2.0, label='Median')
    ax.plot(layers, p75s, '--', color=COLORS['p75'], lw=1.2, label='75th pctl')
    ax.plot(layers, maxes, ':', color=COLORS['max'], lw=1.2, label='Max')

    ax.set_xlabel('Layer')
    ax.set_ylabel('eta-sq (interaction)')
    ax.set_title('IC: Interaction Effect Size Summary (Gemma-2-9B)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(axis='y')

    _save_fig(fig, save_path)


def ic_fig_e_interaction_pct(anova_data: Dict, save_path: Path):
    """IC Fig E: Layer-wise significant interaction feature percentage."""
    layer_key = 'layer_results' if 'layer_results' in anova_data else 'layer_summary'
    layer_data = anova_data[layer_key]

    layers, pcts = [], []
    for lr in layer_data:
        n_tested = lr['n_tested']
        n_sig = lr['n_significant']
        if n_tested > 0:
            layers.append(lr['layer'])
            pcts.append(100.0 * n_sig / n_tested)

    if not layers:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(layers, pcts, '-o', color=COLORS['ic'], lw=1.8, markersize=4)

    peak_idx = int(np.argmax(pcts))
    ax.plot(layers[peak_idx], pcts[peak_idx], 'o', color=COLORS['ic'],
            markersize=8, markeredgecolor='white', markeredgewidth=1.5, zorder=5)
    ax.annotate(f'L{layers[peak_idx]}',
                xy=(layers[peak_idx], pcts[peak_idx]),
                xytext=(5, 8), textcoords='offset points',
                fontsize=8, color=COLORS['ic'], fontweight='bold')

    ax.set_xlabel('Layer')
    ax.set_ylabel('Significant Interactions (%)')
    ax.set_title('IC: Interaction Features by Layer (Gemma-2-9B)')
    ax.grid(axis='y')

    _save_fig(fig, save_path)


# --- Module 2 Figures ---

def ic_fig_f_constraint_sig_count(welch_data: Dict, save_path: Path):
    """IC Fig F: Layer-wise c30-up / c50-up stacked bar."""
    layer_results = welch_data.get('layer_results', [])
    if not layer_results:
        return

    layers = [lr['layer'] for lr in layer_results]
    c30_up = [lr['n_c30_up'] for lr in layer_results]
    c50_up = [lr['n_c50_up'] for lr in layer_results]

    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(layers))
    width = 0.7

    ax.bar(x, c30_up, width, label='c30 > c50', color=COLORS['c30'], alpha=0.8)
    ax.bar(x, c50_up, width, bottom=c30_up, label='c50 > c30', color=COLORS['c50'], alpha=0.8)

    ax.set_xticks(x[::2])
    ax.set_xticklabels([layers[i] for i in range(0, len(layers), 2)], fontsize=8)
    ax.set_xlabel('Layer')
    ax.set_ylabel('# Significant Features (FDR + |d|>=0.3)')
    ax.set_title('IC: Constraint Effect — c30 vs c50 Significant Features')
    ax.legend(fontsize=9)
    ax.grid(axis='y')

    _save_fig(fig, save_path)


def ic_fig_g_constraint_outcome_heatmap(co_anova_data: Dict, save_path: Path):
    """IC Fig G: Top-10 constraint x outcome interaction heatmap."""
    layer_data = co_anova_data.get('layer_results', [])

    all_sig = []
    for lr in layer_data:
        for feat in lr.get('significant_features', []):
            if feat.get('interaction_significant', True):
                feat_copy = dict(feat)
                feat_copy['layer'] = lr['layer']
                all_sig.append(feat_copy)

    if not all_sig:
        print("  No significant constraint x outcome interactions, skipping")
        return

    all_sig.sort(key=lambda x: x['interaction_eta_sq'], reverse=True)
    top_n = min(10, len(all_sig))
    top_feats = all_sig[:top_n]

    cells = ['0_0', '0_1', '1_0', '1_1']
    cell_labels = ['c30-BK', 'c30-VS', 'c50-BK', 'c50-VS']
    matrix = np.zeros((top_n, 4))

    for i, feat in enumerate(top_feats):
        gm = feat['group_means']
        for j, cell in enumerate(cells):
            matrix[i, j] = gm.get(cell, 0.0)

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

    ax.set_title('IC: Constraint x Outcome Interaction')

    _save_fig(fig, save_path)


def ic_fig_h_stratified_comparison(stratified_data: Dict, save_path: Path):
    """IC Fig H: Stratified ANOVA comparison (c30 vs c50 interaction count)."""
    strata = stratified_data.get('strata', {})
    if not strata:
        return

    fig, ax = plt.subplots(figsize=(10, 4))

    for stratum_name, stratum_data in strata.items():
        color = COLORS.get(stratum_name, '#333333')
        layers_data = stratum_data.get('layer_results', [])

        layers = [lr['layer'] for lr in layers_data if lr['n_tested'] > 0]
        pcts = [100.0 * lr['n_significant'] / lr['n_tested']
                for lr in layers_data if lr['n_tested'] > 0]

        if not layers:
            continue

        label = stratum_name.upper()
        if stratum_data.get('cell_warning'):
            label += ' (small cell)'

        ax.plot(layers, pcts, '-o', color=color, lw=1.5, markersize=4, label=label)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Significant Interactions (%)')
    ax.set_title('IC: Stratified bet_type x outcome Interactions (c30 vs c50)')
    ax.legend(fontsize=9)
    ax.grid(axis='y')

    _save_fig(fig, save_path)


# --- Module 3 Figures ---

def cp_fig_1_auc_comparison(cross_data: Dict, save_path: Path):
    """CP Fig 1: SM vs IC AUC overlay with std shading."""
    auc_layers = cross_data.get('auc_comparison', {}).get('layers', [])
    if not auc_layers:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    layers = [l['layer'] for l in auc_layers]
    ic_aucs = np.array([l['ic_auc'] for l in auc_layers])
    ic_stds = np.array([l['ic_std'] for l in auc_layers])
    sm_aucs = np.array([l['sm_auc'] for l in auc_layers])
    sm_stds = np.array([l['sm_std'] for l in auc_layers])

    ax.plot(layers, sm_aucs, '-', color=COLORS['sm'], lw=2.0, label='Slot Machine')
    ax.fill_between(layers, sm_aucs - sm_stds, np.minimum(sm_aucs + sm_stds, 1.0),
                     alpha=0.15, color=COLORS['sm'])

    ax.plot(layers, ic_aucs, '-', color=COLORS['ic'], lw=2.0, label='Investment Choice')
    ax.fill_between(layers, ic_aucs - ic_stds, np.minimum(ic_aucs + ic_stds, 1.0),
                     alpha=0.15, color=COLORS['ic'])

    ax.axhline(y=0.5, color='gray', ls=':', alpha=0.5, label='Chance')
    ax.set_xlabel('Layer')
    ax.set_ylabel('AUC-ROC')
    ax.set_title('Cross-Paradigm: Classification AUC Comparison (Gemma-2-9B)')
    ax.legend(loc='best', fontsize=9)
    ax.grid(axis='y')

    _save_fig(fig, save_path)


def cp_fig_2_clf_overlap(cross_data: Dict, save_path: Path):
    """CP Fig 2: Layer-wise classification Jaccard bar chart."""
    clf_layers = cross_data.get('clf_overlap', {}).get('layers', [])
    if not clf_layers:
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    layers = [l['layer'] for l in clf_layers]
    jaccards = [l['jaccard'] for l in clf_layers]

    x = np.arange(len(layers))
    ax.bar(x, jaccards, color=COLORS['ic'], alpha=0.7)

    ax.set_xticks(x[::2])
    ax.set_xticklabels([layers[i] for i in range(0, len(layers), 2)], fontsize=8)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Jaccard Index')
    ax.set_title('Cross-Paradigm: Classification Top-50 Feature Overlap (SM vs IC)')
    ax.grid(axis='y')

    _save_fig(fig, save_path)


def cp_fig_3_anova_overlap(cross_data: Dict, save_path: Path):
    """CP Fig 3: Layer-wise ANOVA Jaccard bar chart."""
    anova_layers = cross_data.get('anova_overlap', {}).get('layers', [])
    if not anova_layers:
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    layers = [l['layer'] for l in anova_layers]
    jaccards = [l['jaccard'] for l in anova_layers]

    x = np.arange(len(layers))
    ax.bar(x, jaccards, color=COLORS['sm'], alpha=0.7)

    ax.set_xticks(x[::2])
    ax.set_xticklabels([layers[i] for i in range(0, len(layers), 2)], fontsize=8)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Jaccard Index')
    ax.set_title('Cross-Paradigm: ANOVA Interaction Feature Overlap (SM vs IC)')
    ax.grid(axis='y')

    _save_fig(fig, save_path)


def cp_fig_4_summary_table(cross_data: Dict, save_path: Path):
    """CP Fig 4: SM vs IC summary statistics table as figure."""
    summary = cross_data.get('summary', {})
    ic_s = summary.get('ic', {})
    sm_s = summary.get('sm', {})

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')

    rows = [
        ['Metric', 'Slot Machine', 'Investment Choice'],
        ['N games', str(sm_s.get('n_games', '?')), str(ic_s.get('n_games', '?'))],
        ['Bankruptcy rate', f"{sm_s.get('bankruptcy_rate', 0):.1%}", f"{ic_s.get('bankruptcy_rate', 0):.1%}"],
        ['Best AUC layer', str(sm_s.get('best_layer', '?')), str(ic_s.get('best_layer', '?'))],
        ['Best AUC', f"{sm_s.get('best_auc', 0):.4f}", f"{ic_s.get('best_auc', 0):.4f}"],
        ['Total ANOVA sig', str(sm_s.get('total_anova_sig', '?')), str(ic_s.get('total_anova_sig', '?'))],
    ]

    # Mean Jaccard values
    clf_layers = cross_data.get('clf_overlap', {}).get('layers', [])
    anova_layers = cross_data.get('anova_overlap', {}).get('layers', [])
    mean_clf_j = np.mean([l['jaccard'] for l in clf_layers]) if clf_layers else 0
    mean_anova_j = np.mean([l['jaccard'] for l in anova_layers]) if anova_layers else 0
    max_clf_j = np.max([l['jaccard'] for l in clf_layers]) if clf_layers else 0
    max_anova_j = np.max([l['jaccard'] for l in anova_layers]) if anova_layers else 0

    rows.append(['Clf Jaccard (mean/max)', f"{mean_clf_j:.4f} / {max_clf_j:.4f}", ''])
    rows.append(['ANOVA Jaccard (mean/max)', f"{mean_anova_j:.4f} / {max_anova_j:.4f}", ''])

    table = ax.table(
        cellText=rows[1:],
        colLabels=rows[0],
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Header row styling
    for j in range(3):
        cell = table[0, j]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')

    ax.set_title('Cross-Paradigm Summary: SM vs IC (Gemma-2-9B)', pad=20, fontweight='bold')

    _save_fig(fig, save_path)


# ---------------------------------------------------------------------------
# Save Results
# ---------------------------------------------------------------------------
def save_json(data: Dict, path: Path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)
    print(f"  Saved: {path.name}")


def save_classification_results(clf_data: Dict, path: Path):
    """Strip all_results (heavy) before saving."""
    output = {k: v for k, v in clf_data.items() if k != 'all_results'}
    save_json(output, path)


def save_anova_results(anova_data: Dict, path: Path):
    """Save ANOVA with top features only."""
    output = {
        'paradigm': anova_data.get('paradigm', 'ic'),
        'model': anova_data.get('model', 'gemma'),
        'layer_summary': [],
    }
    for lr in anova_data.get('layer_results', []):
        output['layer_summary'].append({
            'layer': lr['layer'],
            'n_tested': lr['n_tested'],
            'n_significant': lr['n_significant'],
            'top_features': lr.get('significant_features', [])[:50],
        })
    save_json(output, path)


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='IC SAE Analysis + Cross-Paradigm Comparison')
    parser.add_argument('--skip-classification', action='store_true')
    parser.add_argument('--skip-anova', action='store_true')
    parser.add_argument('--skip-permutation', action='store_true')
    parser.add_argument('--skip-module2', action='store_true', help='Skip IC-specific analyses')
    parser.add_argument('--skip-cross-paradigm', action='store_true', help='Skip Module 3')
    parser.add_argument('--layers', type=int, nargs='+', default=None)
    parser.add_argument('--viz-only', action='store_true',
                        help='Load existing JSONs and regenerate figures only')
    args = parser.parse_args()

    setup_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(OUTPUT_DIR / "logs")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    logger.info("=" * 70)
    logger.info("IC SAE ANALYSIS + CROSS-PARADIGM COMPARISON")
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Viz-only: {args.viz_only}")
    logger.info(f"Skip classification: {args.skip_classification}")
    logger.info(f"Skip ANOVA: {args.skip_anova}")
    logger.info(f"Skip permutation: {args.skip_permutation}")
    logger.info(f"Skip Module 2: {args.skip_module2}")
    logger.info(f"Skip cross-paradigm: {args.skip_cross_paradigm}")
    logger.info("=" * 70)

    layers = args.layers or GEMMA_LAYERS

    # Data holders for all modules
    ic_clf_data = None
    ic_anova_data = None
    welch_data = None
    co_anova_data = None
    stratified_data = None
    cross_data = None

    if args.viz_only:
        # Load existing results
        logger.info("VIZ-ONLY: Loading existing JSON results...")

        def load_latest(prefix: str) -> Optional[Dict]:
            matches = sorted(RESULTS_DIR.glob(f"{prefix}_*.json"))
            if not matches:
                return None
            with open(matches[-1]) as f:
                return json.load(f)

        ic_clf_data = load_latest('ic_classification')
        ic_anova_data = load_latest('ic_anova')
        welch_data = load_latest('ic_constraint_welch')
        co_anova_data = load_latest('ic_constraint_outcome_anova')
        stratified_data = load_latest('ic_stratified_anova')
        cross_data = load_latest('cross_paradigm_comparison')

        for name, data in [('ic_clf', ic_clf_data), ('ic_anova', ic_anova_data),
                           ('welch', welch_data), ('co_anova', co_anova_data),
                           ('stratified', stratified_data), ('cross', cross_data)]:
            logger.info(f"  {name}: {'loaded' if data else 'NOT FOUND'}")

    else:
        # Full analysis
        ic = load_ic_data()
        logger.info(f"IC data: {ic.n_games} games, BK={sum(ic.outcomes == 'bankruptcy')}, "
                     f"VS={sum(ic.outcomes == 'voluntary_stop')}")

        # ---------------------------------------------------------------
        # Module 1: IC Standalone
        # ---------------------------------------------------------------
        if not args.skip_classification:
            ic_clf_data = module1_classification(ic, layers, logger, args.skip_permutation)
            save_classification_results(ic_clf_data, RESULTS_DIR / f'ic_classification_gemma_{timestamp}.json')

        if not args.skip_anova:
            ic_anova_data = module1_anova(ic, layers, logger)
            save_anova_results(ic_anova_data, RESULTS_DIR / f'ic_anova_gemma_{timestamp}.json')

        # ---------------------------------------------------------------
        # Module 2: IC-Specific
        # ---------------------------------------------------------------
        if not args.skip_module2:
            welch_data = module2a_constraint_welch(ic, layers, logger)
            save_json(welch_data, RESULTS_DIR / f'ic_constraint_welch_gemma_{timestamp}.json')

            co_anova_data = module2b_constraint_outcome_anova(ic, layers, logger)
            save_anova_results(co_anova_data, RESULTS_DIR / f'ic_constraint_outcome_anova_gemma_{timestamp}.json')

            stratified_data = module2c_stratified_anova(ic, layers, logger)
            save_json(stratified_data, RESULTS_DIR / f'ic_stratified_anova_gemma_{timestamp}.json')

        # ---------------------------------------------------------------
        # Module 3: Cross-Paradigm
        # ---------------------------------------------------------------
        if not args.skip_cross_paradigm:
            sm_clf, sm_anova = load_sm_results()
            if sm_clf is None or sm_anova is None:
                logger.warning("SM results not found — skipping cross-paradigm comparison")
            else:
                # Use the IC clf/anova data just computed, or load if skipped
                clf_for_cross = ic_clf_data
                anova_for_cross = ic_anova_data

                if clf_for_cross is None:
                    def load_latest_ic(prefix):
                        matches = sorted(RESULTS_DIR.glob(f"{prefix}_*.json"))
                        return json.load(open(matches[-1])) if matches else None
                    clf_for_cross = load_latest_ic('ic_classification')
                if anova_for_cross is None:
                    def load_latest_ic2(prefix):
                        matches = sorted(RESULTS_DIR.glob(f"{prefix}_*.json"))
                        return json.load(open(matches[-1])) if matches else None
                    anova_for_cross = load_latest_ic2('ic_anova')

                if clf_for_cross and anova_for_cross:
                    cross_data = module3_cross_paradigm(
                        clf_for_cross, anova_for_cross, sm_clf, sm_anova, logger
                    )
                    save_json(cross_data, RESULTS_DIR / f'cross_paradigm_comparison_gemma_{timestamp}.json')
                else:
                    logger.warning("IC clf/anova results missing — cannot run cross-paradigm")

    # ---------------------------------------------------------------
    # Generate Figures
    # ---------------------------------------------------------------
    logger.info(f"\n{'=' * 70}")
    logger.info("GENERATING FIGURES")
    logger.info(f"{'=' * 70}")

    # Module 1 figures
    if ic_clf_data:
        ic_fig_a_layer_auc(ic_clf_data, FIGURE_DIR / 'ic_fig_a_layer_auc.png')

    if ic_anova_data:
        ic_fig_b_heatmap(ic_anova_data, FIGURE_DIR / 'ic_fig_b_heatmap.png')

    if ic_clf_data and ic_anova_data:
        ic_fig_c_overlap(ic_clf_data, ic_anova_data, FIGURE_DIR / 'ic_fig_c_overlap.png')

    if ic_anova_data:
        ic_fig_d_eta_summary(ic_anova_data, FIGURE_DIR / 'ic_fig_d_eta_summary.png')
        ic_fig_e_interaction_pct(ic_anova_data, FIGURE_DIR / 'ic_fig_e_interaction_pct.png')

    # Module 2 figures
    if welch_data:
        ic_fig_f_constraint_sig_count(welch_data, FIGURE_DIR / 'ic_fig_f_constraint_sig_count.png')

    if co_anova_data:
        ic_fig_g_constraint_outcome_heatmap(co_anova_data, FIGURE_DIR / 'ic_fig_g_constraint_outcome_heatmap.png')

    if stratified_data:
        ic_fig_h_stratified_comparison(stratified_data, FIGURE_DIR / 'ic_fig_h_stratified_comparison.png')

    # Module 3 figures
    if cross_data:
        cp_fig_1_auc_comparison(cross_data, FIGURE_DIR / 'cp_fig_1_auc_comparison.png')
        cp_fig_2_clf_overlap(cross_data, FIGURE_DIR / 'cp_fig_2_clf_overlap.png')
        cp_fig_3_anova_overlap(cross_data, FIGURE_DIR / 'cp_fig_3_anova_overlap.png')
        cp_fig_4_summary_table(cross_data, FIGURE_DIR / 'cp_fig_4_summary_table.png')

    logger.info(f"\n{'=' * 70}")
    logger.info("ALL COMPLETE")
    logger.info(f"Figures: {FIGURE_DIR}")
    logger.info(f"Results: {RESULTS_DIR}")
    logger.info(f"{'=' * 70}")


if __name__ == '__main__':
    main()
