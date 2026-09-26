"""
Statistical analysis utilities for SAE feature analysis.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size.

    Args:
        group1: First group values
        group2: Second group values

    Returns:
        Cohen's d
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def fdr_correction(p_values: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, float]:
    """
    Benjamini-Hochberg FDR correction.

    Args:
        p_values: Array of p-values
        alpha: FDR threshold

    Returns:
        Tuple of (significant boolean array, adjusted threshold)
    """
    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_pvalues = p_values[sorted_indices]

    # Find largest k such that P(k) <= (k/m) * alpha
    threshold_line = np.arange(1, m + 1) / m * alpha
    below_threshold = sorted_pvalues <= threshold_line

    if not np.any(below_threshold):
        return np.zeros(m, dtype=bool), 0.0

    max_k = np.where(below_threshold)[0][-1]
    adjusted_threshold = sorted_pvalues[max_k]

    # Create boolean array for significant results
    significant = np.zeros(m, dtype=bool)
    significant[sorted_indices[:max_k + 1]] = True

    return significant, adjusted_threshold


def binary_ttest_analysis(
    features: np.ndarray,
    labels: np.ndarray,
    label_0: int,
    label_1: int,
    min_cohens_d: float = 0.3,
    fdr_alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Perform binary t-test analysis on features.

    Args:
        features: Feature matrix [n_samples, n_features]
        labels: Binary labels [n_samples]
        label_0: First label value
        label_1: Second label value
        min_cohens_d: Minimum Cohen's d threshold
        fdr_alpha: FDR correction alpha

    Returns:
        Dictionary with analysis results
    """
    n_samples, n_features = features.shape

    group_0 = features[labels == label_0]
    group_1 = features[labels == label_1]

    logger.info(f"Group 0 ({label_0}): {len(group_0)} samples")
    logger.info(f"Group 1 ({label_1}): {len(group_1)} samples")

    t_stats = np.zeros(n_features)
    p_values = np.zeros(n_features)
    cohens_ds = np.zeros(n_features)

    for i in range(n_features):
        g0_vals = group_0[:, i]
        g1_vals = group_1[:, i]

        t_stat, p_val = stats.ttest_ind(g0_vals, g1_vals)
        cohens_d = compute_cohens_d(g0_vals, g1_vals)

        t_stats[i] = t_stat
        p_values[i] = p_val
        cohens_ds[i] = cohens_d

    # FDR correction
    significant_fdr, adj_threshold = fdr_correction(p_values, fdr_alpha)

    # Apply Cohen's d threshold
    significant_effect = np.abs(cohens_ds) >= min_cohens_d

    # Combined significance
    significant = significant_fdr & significant_effect

    logger.info(f"Significant features (FDR < {fdr_alpha}): {significant_fdr.sum()}")
    logger.info(f"Significant features (|d| > {min_cohens_d}): {significant_effect.sum()}")
    logger.info(f"Combined significant features: {significant.sum()}")

    return {
        't_stats': t_stats,
        'p_values': p_values,
        'cohens_ds': cohens_ds,
        'significant': significant,
        'significant_fdr': significant_fdr,
        'significant_effect': significant_effect,
        'fdr_threshold': adj_threshold,
        'n_significant': significant.sum(),
        'group_0_size': len(group_0),
        'group_1_size': len(group_1)
    }


def multiclass_anova_analysis(
    features: np.ndarray,
    labels: np.ndarray,
    min_eta_squared: float = 0.01,
    fdr_alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Perform multi-class ANOVA analysis on features.

    Args:
        features: Feature matrix [n_samples, n_features]
        labels: Multi-class labels [n_samples]
        min_eta_squared: Minimum eta-squared threshold
        fdr_alpha: FDR correction alpha

    Returns:
        Dictionary with analysis results
    """
    n_samples, n_features = features.shape
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)

    logger.info(f"Multi-class ANOVA: {n_classes} classes")
    for label in unique_labels:
        count = np.sum(labels == label)
        logger.info(f"  Class {label}: {count} samples ({count/n_samples*100:.1f}%)")

    f_stats = np.zeros(n_features)
    p_values = np.zeros(n_features)
    eta_squareds = np.zeros(n_features)

    for i in range(n_features):
        # Group features by label
        groups = [features[labels == label, i] for label in unique_labels]

        # One-way ANOVA
        f_stat, p_val = stats.f_oneway(*groups)

        # Eta-squared (effect size for ANOVA)
        ss_between = sum(len(g) * (np.mean(g) - np.mean(features[:, i]))**2 for g in groups)
        ss_total = np.sum((features[:, i] - np.mean(features[:, i]))**2)

        eta_squared = ss_between / ss_total if ss_total > 0 else 0.0

        f_stats[i] = f_stat
        p_values[i] = p_val
        eta_squareds[i] = eta_squared

    # FDR correction
    significant_fdr, adj_threshold = fdr_correction(p_values, fdr_alpha)

    # Apply eta-squared threshold
    significant_effect = eta_squareds >= min_eta_squared

    # Combined significance
    significant = significant_fdr & significant_effect

    logger.info(f"Significant features (FDR < {fdr_alpha}): {significant_fdr.sum()}")
    logger.info(f"Significant features (η² > {min_eta_squared}): {significant_effect.sum()}")
    logger.info(f"Combined significant features: {significant.sum()}")

    return {
        'f_stats': f_stats,
        'p_values': p_values,
        'eta_squareds': eta_squareds,
        'significant': significant,
        'significant_fdr': significant_fdr,
        'significant_effect': significant_effect,
        'fdr_threshold': adj_threshold,
        'n_significant': significant.sum(),
        'n_classes': n_classes
    }


def get_top_features(
    feature_ids: np.ndarray,
    scores: np.ndarray,
    significant: np.ndarray,
    n_top: int = 50,
    ascending: bool = False
) -> List[Tuple[int, float]]:
    """
    Get top N features by score.

    Args:
        feature_ids: Feature IDs
        scores: Feature scores (e.g., Cohen's d, eta-squared)
        significant: Boolean array of significant features
        n_top: Number of top features to return
        ascending: If True, return lowest scores; if False, return highest

    Returns:
        List of (feature_id, score) tuples
    """
    # Filter to significant features only
    sig_indices = np.where(significant)[0]

    if len(sig_indices) == 0:
        return []

    sig_feature_ids = feature_ids[sig_indices]
    sig_scores = scores[sig_indices]

    # Sort by score
    if ascending:
        sorted_indices = np.argsort(sig_scores)
    else:
        sorted_indices = np.argsort(sig_scores)[::-1]

    # Get top N
    top_n = min(n_top, len(sorted_indices))
    top_indices = sorted_indices[:top_n]

    top_features = [(int(sig_feature_ids[i]), float(sig_scores[i])) for i in top_indices]

    return top_features


def compute_feature_means_by_group(
    features: np.ndarray,
    labels: np.ndarray,
    feature_indices: List[int]
) -> Dict[int, Dict[int, float]]:
    """
    Compute mean feature activations by group.

    Args:
        features: Feature matrix [n_samples, n_features]
        labels: Group labels [n_samples]
        feature_indices: Feature indices to compute

    Returns:
        Dictionary mapping feature_id -> {label -> mean}
    """
    unique_labels = np.unique(labels)

    feature_means = {}

    for feat_idx in feature_indices:
        feat_means_by_label = {}

        for label in unique_labels:
            group_features = features[labels == label, feat_idx]
            feat_means_by_label[int(label)] = float(np.mean(group_features))

        feature_means[int(feat_idx)] = feat_means_by_label

    return feature_means


def compute_classification_accuracy(
    features: np.ndarray,
    labels: np.ndarray,
    feature_indices: List[int],
    cv_folds: int = 5
) -> float:
    """
    Compute classification accuracy using logistic regression.

    Args:
        features: Feature matrix [n_samples, n_features]
        labels: Labels [n_samples]
        feature_indices: Feature indices to use
        cv_folds: Number of cross-validation folds

    Returns:
        Mean cross-validation accuracy
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    # Select features
    X = features[:, feature_indices]

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Logistic regression with cross-validation
    clf = LogisticRegression(max_iter=1000, random_state=42)
    scores = cross_val_score(clf, X_scaled, labels, cv=cv_folds, scoring='accuracy')

    return float(np.mean(scores))


def clear_gpu_memory():
    """Clear GPU memory cache."""
    import torch
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU memory cleared")
