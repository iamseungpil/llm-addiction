#!/usr/bin/env python3
"""
Baseline Comparisons for LR Classification Experiment

Implements 4 baseline methods to compare against hidden state LR:
1. Chance Level - Majority class prediction
2. Text-Only (TF-IDF) - Prompt text features only
3. Metadata-Only - Game state metadata features
4. Random Projection - Random dimensionality reduction

Author: LLM Addiction Research Team
Last Updated: 2025-01
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


@dataclass
class BaselineResult:
    """Result for a single baseline method."""
    method: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    cv_mean: float
    cv_std: float
    n_samples: int
    n_positive: int
    n_features: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# Baseline 1: Chance Level
# =============================================================================

def chance_level_baseline(labels: np.ndarray) -> BaselineResult:
    """
    Baseline 1: Chance Level (Majority Class Prediction)

    Predicts the majority class for all samples.
    This is the minimum performance any useful classifier should beat.

    Args:
        labels: Binary labels (1=bankruptcy, 0=safe)

    Returns:
        BaselineResult with metrics
    """
    n_samples = len(labels)
    n_positive = int(labels.sum())
    n_negative = n_samples - n_positive

    # Majority class prediction
    majority_class = 1 if n_positive > n_negative else 0
    majority_count = max(n_positive, n_negative)

    # All predictions are majority class
    accuracy = majority_count / n_samples

    # Precision/Recall depend on which class is majority
    if majority_class == 1:
        # Predicting all as bankruptcy
        precision = n_positive / n_samples  # TP / (TP + FP) = n_pos / n_total
        recall = 1.0  # TP / (TP + FN) = n_pos / n_pos = 1
    else:
        # Predicting all as safe (no positive predictions)
        precision = 0.0  # No positive predictions
        recall = 0.0  # TP = 0

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    auc_roc = 0.5  # Random baseline

    return BaselineResult(
        method="chance_level",
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        auc_roc=auc_roc,
        cv_mean=accuracy,  # Same for all folds
        cv_std=0.0,
        n_samples=n_samples,
        n_positive=n_positive,
        n_features=0
    )


# =============================================================================
# Baseline 2: Text-Only (TF-IDF)
# =============================================================================

def text_only_baseline(
    prompts: List[str],
    labels: np.ndarray,
    max_features: int = 1000,
    test_size: float = 0.2,
    random_state: int = 42,
    n_cv_folds: int = 5
) -> BaselineResult:
    """
    Baseline 2: Text-Only (TF-IDF Features)

    Uses only the prompt text (TF-IDF features) without hidden states.
    Tests if hidden states provide information beyond the input text.

    Args:
        prompts: List of prompt strings
        labels: Binary labels
        max_features: Maximum number of TF-IDF features
        test_size: Test split ratio
        random_state: Random seed
        n_cv_folds: Number of CV folds

    Returns:
        BaselineResult with metrics
    """
    n_samples = len(labels)
    n_positive = int(labels.sum())

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    X = vectorizer.fit_transform(prompts).toarray()

    n_features = X.shape[1]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    # Train LR
    lr = LogisticRegression(
        max_iter=1000,
        solver='lbfgs',
        class_weight='balanced',
        random_state=random_state
    )
    lr.fit(X_train, y_train)

    # Predictions
    y_pred = lr.predict(X_test)
    y_prob = lr.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = 0.5

    # Cross-validation
    cv = StratifiedKFold(n_splits=min(n_cv_folds, n_positive, n_samples - n_positive),
                         shuffle=True, random_state=random_state)
    try:
        cv_scores = cross_val_score(
            LogisticRegression(max_iter=1000, solver='lbfgs', class_weight='balanced', random_state=random_state),
            X, labels, cv=cv, scoring='accuracy'
        )
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
    except ValueError:
        cv_mean = accuracy
        cv_std = 0.0

    return BaselineResult(
        method="text_only_tfidf",
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        auc_roc=auc,
        cv_mean=cv_mean,
        cv_std=cv_std,
        n_samples=n_samples,
        n_positive=n_positive,
        n_features=n_features
    )


# =============================================================================
# Baseline 3: Metadata-Only
# =============================================================================

def metadata_only_baseline(
    metadata_features: List[Dict],
    labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    n_cv_folds: int = 5
) -> BaselineResult:
    """
    Baseline 3: Metadata-Only

    Uses only game state metadata (balance, rounds, wins, losses) without hidden states.
    Tests if hidden states provide information beyond explicit game state.

    Expected features per sample:
    - balance: Current balance
    - n_rounds: Number of rounds played
    - n_wins: Number of wins
    - n_losses: Number of losses
    - consecutive_losses: Current consecutive loss streak
    - balance_change: Change from initial balance
    - win_rate: Win ratio

    Args:
        metadata_features: List of feature dicts from get_metadata_features()
        labels: Binary labels
        test_size: Test split ratio
        random_state: Random seed
        n_cv_folds: Number of CV folds

    Returns:
        BaselineResult with metrics
    """
    n_samples = len(labels)
    n_positive = int(labels.sum())

    # Convert to array
    feature_names = ['balance', 'n_rounds', 'n_wins', 'n_losses',
                     'consecutive_losses', 'balance_change', 'win_rate']
    X = np.array([[f.get(name, 0) for name in feature_names] for f in metadata_features])

    n_features = X.shape[1]

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    # Train LR
    lr = LogisticRegression(
        max_iter=1000,
        solver='lbfgs',
        class_weight='balanced',
        random_state=random_state
    )
    lr.fit(X_train, y_train)

    # Predictions
    y_pred = lr.predict(X_test)
    y_prob = lr.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = 0.5

    # Cross-validation
    cv = StratifiedKFold(n_splits=min(n_cv_folds, n_positive, n_samples - n_positive),
                         shuffle=True, random_state=random_state)
    try:
        cv_scores = cross_val_score(
            LogisticRegression(max_iter=1000, solver='lbfgs', class_weight='balanced', random_state=random_state),
            X_scaled, labels, cv=cv, scoring='accuracy'
        )
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
    except ValueError:
        cv_mean = accuracy
        cv_std = 0.0

    return BaselineResult(
        method="metadata_only",
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        auc_roc=auc,
        cv_mean=cv_mean,
        cv_std=cv_std,
        n_samples=n_samples,
        n_positive=n_positive,
        n_features=n_features
    )


# =============================================================================
# Baseline 4: Random Projection
# =============================================================================

def random_projection_baseline(
    hidden_states: Dict[int, np.ndarray],
    labels: np.ndarray,
    target_dim: int = 100,
    test_size: float = 0.2,
    random_state: int = 42,
    n_cv_folds: int = 5
) -> Dict[int, BaselineResult]:
    """
    Baseline 4: Random Projection

    Projects hidden states to lower dimension using random matrix.
    Tests if LR performance is due to structured information in hidden states
    vs. just having high-dimensional features.

    Args:
        hidden_states: Dict of layer -> array [n_samples, d_model]
        labels: Binary labels
        target_dim: Target dimensionality after projection
        test_size: Test split ratio
        random_state: Random seed
        n_cv_folds: Number of CV folds

    Returns:
        Dict of layer -> BaselineResult
    """
    np.random.seed(random_state)

    n_samples = len(labels)
    n_positive = int(labels.sum())

    results = {}

    for layer, X in hidden_states.items():
        d_model = X.shape[1]

        # Random projection matrix
        R = np.random.randn(d_model, target_dim) / np.sqrt(target_dim)
        X_proj = X @ R

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_proj)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, labels, test_size=test_size, random_state=random_state, stratify=labels
        )

        # Train LR
        lr = LogisticRegression(
            max_iter=1000,
            solver='lbfgs',
            class_weight='balanced',
            random_state=random_state
        )
        lr.fit(X_train, y_train)

        # Predictions
        y_pred = lr.predict(X_test)
        y_prob = lr.predict_proba(X_test)[:, 1]

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = 0.5

        # Cross-validation
        cv = StratifiedKFold(n_splits=min(n_cv_folds, n_positive, n_samples - n_positive),
                             shuffle=True, random_state=random_state)
        try:
            cv_scores = cross_val_score(
                LogisticRegression(max_iter=1000, solver='lbfgs', class_weight='balanced', random_state=random_state),
                X_scaled, labels, cv=cv, scoring='accuracy'
            )
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
        except ValueError:
            cv_mean = accuracy
            cv_std = 0.0

        results[layer] = BaselineResult(
            method=f"random_projection_layer{layer}",
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            auc_roc=auc,
            cv_mean=cv_mean,
            cv_std=cv_std,
            n_samples=n_samples,
            n_positive=n_positive,
            n_features=target_dim
        )

    return results


# =============================================================================
# All Baselines Runner
# =============================================================================

@dataclass
class AllBaselinesResult:
    """Container for all baseline results."""
    chance: BaselineResult
    text_only: Optional[BaselineResult] = None
    metadata: Optional[BaselineResult] = None
    random_projection: Optional[Dict[int, BaselineResult]] = None

    def to_dict(self) -> Dict:
        result = {
            'chance': self.chance.to_dict()
        }
        if self.text_only:
            result['text_only'] = self.text_only.to_dict()
        if self.metadata:
            result['metadata'] = self.metadata.to_dict()
        if self.random_projection:
            result['random_projection'] = {
                str(k): v.to_dict() for k, v in self.random_projection.items()
            }
        return result


def run_all_baselines(
    labels: np.ndarray,
    prompts: Optional[List[str]] = None,
    metadata_features: Optional[List[Dict]] = None,
    hidden_states: Optional[Dict[int, np.ndarray]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    n_cv_folds: int = 5,
    verbose: bool = True
) -> AllBaselinesResult:
    """
    Run all applicable baseline methods.

    Args:
        labels: Binary labels
        prompts: List of prompts (for text-only baseline)
        metadata_features: List of metadata dicts (for metadata baseline)
        hidden_states: Dict of hidden states (for random projection baseline)
        test_size: Test split ratio
        random_state: Random seed
        n_cv_folds: Number of CV folds
        verbose: Print results

    Returns:
        AllBaselinesResult containing all baseline results
    """
    if verbose:
        print("\n" + "="*70)
        print("BASELINE COMPARISONS")
        print("="*70)

    # 1. Chance level (always run)
    if verbose:
        print("\n[1/4] Chance Level...")
    chance_result = chance_level_baseline(labels)
    if verbose:
        print(f"  Accuracy: {chance_result.accuracy:.3f}")

    # 2. Text-only
    text_result = None
    if prompts is not None:
        if verbose:
            print("\n[2/4] Text-Only (TF-IDF)...")
        text_result = text_only_baseline(
            prompts, labels, test_size=test_size,
            random_state=random_state, n_cv_folds=n_cv_folds
        )
        if verbose:
            print(f"  Accuracy: {text_result.accuracy:.3f}, AUC: {text_result.auc_roc:.3f}")
    elif verbose:
        print("\n[2/4] Text-Only (TF-IDF)... SKIPPED (no prompts)")

    # 3. Metadata-only
    meta_result = None
    if metadata_features is not None:
        if verbose:
            print("\n[3/4] Metadata-Only...")
        meta_result = metadata_only_baseline(
            metadata_features, labels, test_size=test_size,
            random_state=random_state, n_cv_folds=n_cv_folds
        )
        if verbose:
            print(f"  Accuracy: {meta_result.accuracy:.3f}, AUC: {meta_result.auc_roc:.3f}")
    elif verbose:
        print("\n[3/4] Metadata-Only... SKIPPED (no metadata)")

    # 4. Random projection
    rp_result = None
    if hidden_states is not None:
        if verbose:
            print("\n[4/4] Random Projection...")
        rp_result = random_projection_baseline(
            hidden_states, labels, test_size=test_size,
            random_state=random_state, n_cv_folds=n_cv_folds
        )
        if verbose:
            for layer, r in sorted(rp_result.items()):
                print(f"  Layer {layer}: Accuracy={r.accuracy:.3f}, AUC={r.auc_roc:.3f}")
    elif verbose:
        print("\n[4/4] Random Projection... SKIPPED (no hidden states)")

    return AllBaselinesResult(
        chance=chance_result,
        text_only=text_result,
        metadata=meta_result,
        random_projection=rp_result
    )


def print_baseline_comparison(
    baselines: AllBaselinesResult,
    lr_best_accuracy: float,
    lr_best_auc: float
):
    """Print comparison table of baselines vs LR."""
    print("\n" + "="*70)
    print("BASELINE vs LR COMPARISON")
    print("="*70)

    print(f"\n{'Method':<25} | {'Accuracy':>8} | {'AUC':>8} | {'vs LR':>10}")
    print("-"*60)

    # Chance
    print(f"{'Chance Level':<25} | {baselines.chance.accuracy:>8.3f} | {baselines.chance.auc_roc:>8.3f} | "
          f"{lr_best_accuracy - baselines.chance.accuracy:>+10.3f}")

    # Text-only
    if baselines.text_only:
        print(f"{'Text-Only (TF-IDF)':<25} | {baselines.text_only.accuracy:>8.3f} | {baselines.text_only.auc_roc:>8.3f} | "
              f"{lr_best_accuracy - baselines.text_only.accuracy:>+10.3f}")

    # Metadata
    if baselines.metadata:
        print(f"{'Metadata-Only':<25} | {baselines.metadata.accuracy:>8.3f} | {baselines.metadata.auc_roc:>8.3f} | "
              f"{lr_best_accuracy - baselines.metadata.accuracy:>+10.3f}")

    # Random projection (best layer)
    if baselines.random_projection:
        best_rp = max(baselines.random_projection.values(), key=lambda x: x.accuracy)
        print(f"{'Random Projection (best)':<25} | {best_rp.accuracy:>8.3f} | {best_rp.auc_roc:>8.3f} | "
              f"{lr_best_accuracy - best_rp.accuracy:>+10.3f}")

    # LR (best)
    print("-"*60)
    print(f"{'LR Hidden State (best)':<25} | {lr_best_accuracy:>8.3f} | {lr_best_auc:>8.3f} | {'---':>10}")
    print("="*70)


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    print("Testing Baselines with synthetic data...")

    np.random.seed(42)
    n_samples = 500
    d_model = 100

    # Synthetic data
    labels = np.random.binomial(1, 0.3, n_samples)  # 30% positive
    prompts = [f"Balance ${np.random.randint(10, 200)}. Round {np.random.randint(1,20)}."
               for _ in range(n_samples)]

    metadata_features = [
        {
            'balance': np.random.randint(10, 200),
            'n_rounds': np.random.randint(1, 20),
            'n_wins': np.random.randint(0, 10),
            'n_losses': np.random.randint(0, 10),
            'consecutive_losses': np.random.randint(0, 5),
            'balance_change': np.random.randint(-90, 100),
            'win_rate': np.random.random()
        }
        for _ in range(n_samples)
    ]

    hidden_states = {
        15: np.random.randn(n_samples, d_model),
        20: np.random.randn(n_samples, d_model),
        25: np.random.randn(n_samples, d_model)
    }

    # Run all baselines
    results = run_all_baselines(
        labels=labels,
        prompts=prompts,
        metadata_features=metadata_features,
        hidden_states=hidden_states,
        verbose=True
    )

    # Print comparison (using random LR accuracy for demo)
    print_baseline_comparison(results, lr_best_accuracy=0.65, lr_best_auc=0.70)
