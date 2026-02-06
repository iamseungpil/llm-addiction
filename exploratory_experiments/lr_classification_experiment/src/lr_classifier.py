#!/usr/bin/env python3
"""
Logistic Regression Classifier for Hidden States

Trains and evaluates LR classifiers to predict bankruptcy vs voluntary_stop
from model hidden states. Includes all evaluation metrics and group analysis.

Author: LLM Addiction Research Team
Last Updated: 2025-01
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)


# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class ClassificationMetrics:
    """All classification metrics for a single experiment."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    cv_mean: float
    cv_std: float
    confusion_matrix: List[List[int]] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class LayerResult:
    """Result of LR classification for a single layer."""
    layer: int
    metrics: ClassificationMetrics
    n_samples: int
    n_positive: int
    n_negative: int
    coef_norm: float
    intercept: float

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['metrics'] = self.metrics.to_dict()
        return d


@dataclass
class GroupResult:
    """Result for an analysis group (e.g., bet_type=fixed)."""
    group_name: str
    group_value: str
    n_samples: int
    n_positive: int
    layer_results: Dict[int, LayerResult] = field(default_factory=dict)
    best_layer: int = 0
    best_accuracy: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'group_name': self.group_name,
            'group_value': self.group_value,
            'n_samples': self.n_samples,
            'n_positive': self.n_positive,
            'best_layer': self.best_layer,
            'best_accuracy': self.best_accuracy,
            'layer_results': {str(k): v.to_dict() for k, v in self.layer_results.items()}
        }


@dataclass
class ExperimentResult:
    """Full experiment result including all groups."""
    model_name: str
    option: str
    timestamp: str
    n_total_samples: int
    n_total_positive: int

    # Main result (all data)
    main_result: Optional[GroupResult] = None

    # Group results
    bet_type_results: Dict[str, GroupResult] = field(default_factory=dict)
    prompt_combo_results: Dict[str, GroupResult] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'model_name': self.model_name,
            'option': self.option,
            'timestamp': self.timestamp,
            'n_total_samples': self.n_total_samples,
            'n_total_positive': self.n_total_positive,
            'main_result': self.main_result.to_dict() if self.main_result else None,
            'bet_type_results': {k: v.to_dict() for k, v in self.bet_type_results.items()},
            'prompt_combo_results': {k: v.to_dict() for k, v in self.prompt_combo_results.items()}
        }


# =============================================================================
# LR Classifier
# =============================================================================

class LRClassifier:
    """
    Logistic Regression classifier for hidden states.
    Includes all evaluation metrics and cross-validation.
    """

    def __init__(
        self,
        max_iter: int = 1000,
        solver: str = 'lbfgs',
        class_weight: str = 'balanced',
        random_state: int = 42,
        test_size: float = 0.2,
        n_cv_folds: int = 5
    ):
        """
        Initialize LR classifier.

        Args:
            max_iter: Maximum iterations for LR optimization
            solver: Optimization solver ('lbfgs', 'liblinear', etc.)
            class_weight: Class weighting ('balanced' for imbalanced data)
            random_state: Random seed for reproducibility
            test_size: Proportion of data for test set
            n_cv_folds: Number of cross-validation folds
        """
        self.max_iter = max_iter
        self.solver = solver
        self.class_weight = class_weight
        self.random_state = random_state
        self.test_size = test_size
        self.n_cv_folds = n_cv_folds

    def _create_lr(self) -> LogisticRegression:
        """Create a new LR instance."""
        return LogisticRegression(
            max_iter=self.max_iter,
            solver=self.solver,
            class_weight=self.class_weight,
            random_state=self.random_state
        )

    def train_and_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        layer: int
    ) -> LayerResult:
        """
        Train LR and evaluate on a single layer's hidden states.

        Args:
            X: Hidden states [n_samples, d_model]
            y: Labels [n_samples] (1=bankruptcy, 0=safe)
            layer: Layer index (for recording)

        Returns:
            LayerResult with all metrics
        """
        n_samples = len(y)
        n_positive = int(y.sum())
        n_negative = n_samples - n_positive

        # Check minimum samples
        if n_samples < 10 or n_positive < 2 or n_negative < 2:
            # Return empty result for insufficient data
            return LayerResult(
                layer=layer,
                metrics=ClassificationMetrics(
                    accuracy=0.0, precision=0.0, recall=0.0, f1=0.0,
                    auc_roc=0.5, cv_mean=0.0, cv_std=0.0, confusion_matrix=[]
                ),
                n_samples=n_samples,
                n_positive=n_positive,
                n_negative=n_negative,
                coef_norm=0.0,
                intercept=0.0
            )

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train LR
        lr = self._create_lr()
        lr.fit(X_train_scaled, y_train)

        # Predictions
        y_pred = lr.predict(X_test_scaled)
        y_prob = lr.predict_proba(X_test_scaled)[:, 1]

        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = 0.5  # If only one class in test set

        cm = confusion_matrix(y_test, y_pred).tolist()

        # Cross-validation on full data
        X_scaled_full = scaler.fit_transform(X)
        cv = StratifiedKFold(n_splits=min(self.n_cv_folds, n_positive, n_negative), shuffle=True, random_state=self.random_state)

        try:
            cv_scores = cross_val_score(
                self._create_lr(),
                X_scaled_full, y,
                cv=cv,
                scoring='accuracy'
            )
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
        except ValueError:
            cv_mean = accuracy
            cv_std = 0.0

        metrics = ClassificationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            auc_roc=auc,
            cv_mean=cv_mean,
            cv_std=cv_std,
            confusion_matrix=cm
        )

        return LayerResult(
            layer=layer,
            metrics=metrics,
            n_samples=n_samples,
            n_positive=n_positive,
            n_negative=n_negative,
            coef_norm=float(np.linalg.norm(lr.coef_)),
            intercept=float(lr.intercept_[0])
        )

    def run_all_layers(
        self,
        hidden_states: Dict[int, np.ndarray],
        labels: np.ndarray,
        group_name: str = "all",
        group_value: str = "all",
        verbose: bool = True
    ) -> GroupResult:
        """
        Run LR classification on all layers.

        Args:
            hidden_states: Dict of layer -> array [n_samples, d_model]
            labels: Array of labels
            group_name: Name of the group (e.g., 'bet_type')
            group_value: Value of the group (e.g., 'fixed')
            verbose: Print progress

        Returns:
            GroupResult with all layer results
        """
        n_samples = len(labels)
        n_positive = int(labels.sum())

        result = GroupResult(
            group_name=group_name,
            group_value=group_value,
            n_samples=n_samples,
            n_positive=n_positive
        )

        if verbose:
            print(f"\n  Group: {group_name}={group_value} (N={n_samples}, pos={n_positive})")

        best_acc = 0.0
        best_layer = 0

        for layer in sorted(hidden_states.keys()):
            X = hidden_states[layer]

            if verbose:
                print(f"    Layer {layer}...", end=" ", flush=True)

            layer_result = self.train_and_evaluate(X, labels, layer)
            result.layer_results[layer] = layer_result

            if verbose:
                m = layer_result.metrics
                print(f"Acc={m.accuracy:.3f}, AUC={m.auc_roc:.3f}, F1={m.f1:.3f}, CV={m.cv_mean:.3f}±{m.cv_std:.3f}")

            if layer_result.metrics.accuracy > best_acc:
                best_acc = layer_result.metrics.accuracy
                best_layer = layer

        result.best_layer = best_layer
        result.best_accuracy = best_acc

        return result


# =============================================================================
# Result I/O
# =============================================================================

def save_experiment_result(result: ExperimentResult, output_path: str):
    """Save experiment result to JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)

    print(f"Saved: {output_path}")


def load_experiment_result(path: str) -> Dict:
    """Load experiment result from JSON."""
    with open(path, 'r') as f:
        return json.load(f)


# =============================================================================
# Summary Printing
# =============================================================================

def print_layer_summary(result: GroupResult, title: str = ""):
    """Print summary table for layer results."""
    print("\n" + "="*70)
    print(f"{title or result.group_name + '=' + result.group_value}")
    print("="*70)
    print(f"Samples: {result.n_samples} (positive={result.n_positive}, "
          f"rate={result.n_positive/result.n_samples*100:.1f}%)")
    print(f"Best Layer: {result.best_layer} (accuracy={result.best_accuracy:.3f})")
    print()

    print(f"{'Layer':>6} | {'Acc':>6} | {'AUC':>6} | {'Prec':>6} | {'Recall':>6} | {'F1':>6} | {'CV':>12}")
    print("-"*70)

    for layer in sorted(result.layer_results.keys()):
        r = result.layer_results[layer]
        m = r.metrics
        marker = " *" if layer == result.best_layer else ""
        print(f"{layer:>6} | {m.accuracy:>6.3f} | {m.auc_roc:>6.3f} | {m.precision:>6.3f} | "
              f"{m.recall:>6.3f} | {m.f1:>6.3f} | {m.cv_mean:>5.3f}±{m.cv_std:<5.3f}{marker}")

    print("-"*70)


def print_experiment_summary(result: ExperimentResult):
    """Print full experiment summary."""
    print("\n" + "="*70)
    print(f"EXPERIMENT SUMMARY: {result.model_name.upper()} - Option {result.option}")
    print("="*70)
    print(f"Total Samples: {result.n_total_samples}")
    print(f"Positive (bankruptcy): {result.n_total_positive} ({result.n_total_positive/result.n_total_samples*100:.1f}%)")
    print(f"Timestamp: {result.timestamp}")

    # Main result
    if result.main_result:
        print_layer_summary(result.main_result, "MAIN (All Data)")

    # bet_type results
    if result.bet_type_results:
        print("\n" + "-"*70)
        print("BY BET_TYPE")
        print("-"*70)
        for bt, gr in result.bet_type_results.items():
            print(f"\n{bt}: Best Layer={gr.best_layer}, Acc={gr.best_accuracy:.3f}, "
                  f"N={gr.n_samples}, Pos={gr.n_positive}")

    # prompt_combo summary (condensed)
    if result.prompt_combo_results:
        print("\n" + "-"*70)
        print("BY PROMPT_COMBO (Top 5 by accuracy)")
        print("-"*70)
        sorted_combos = sorted(
            result.prompt_combo_results.items(),
            key=lambda x: x[1].best_accuracy,
            reverse=True
        )
        for combo, gr in sorted_combos[:5]:
            print(f"  {combo}: Acc={gr.best_accuracy:.3f}, Layer={gr.best_layer}, N={gr.n_samples}")


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    print("Testing LRClassifier with synthetic data...")

    np.random.seed(42)
    n_samples = 500
    d_model = 100
    n_layers = 5

    # Create synthetic hidden states with increasing signal
    hidden_states = {}
    for layer in range(n_layers):
        X = np.random.randn(n_samples, d_model)
        # Add signal that increases with layer
        signal = np.random.randn(n_samples) * (0.1 + 0.2 * layer)
        X[:, 0] += signal
        hidden_states[layer] = X

    # Create labels correlated with the signal
    labels = (hidden_states[n_layers-1][:, 0] > 0).astype(int)

    print(f"\nSynthetic data: {n_samples} samples, {d_model} dims, {n_layers} layers")
    print(f"Labels: {labels.sum()} positive, {n_samples - labels.sum()} negative")

    # Run classifier
    classifier = LRClassifier()
    result = classifier.run_all_layers(hidden_states, labels, "test", "synthetic")

    print_layer_summary(result, "Synthetic Test")
