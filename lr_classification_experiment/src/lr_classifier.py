#!/usr/bin/env python3
"""
Logistic Regression Classifier for Hidden States

Trains and evaluates LR classifiers to predict bankruptcy vs voluntary_stop
from model hidden states.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import json
from pathlib import Path


@dataclass
class LRResult:
    """Result of LR classification for a single layer."""
    layer: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    cv_mean: float
    cv_std: float
    n_samples: int
    n_positive: int
    coef_norm: float


@dataclass
class ExperimentResult:
    """Full experiment result."""
    model_name: str
    option: str
    n_samples: int
    n_positive: int
    layer_results: Dict[int, LRResult] = field(default_factory=dict)
    best_layer: int = 0
    best_accuracy: float = 0.0


class LRClassifier:
    """
    Logistic Regression classifier for hidden states.
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
        self.max_iter = max_iter
        self.solver = solver
        self.class_weight = class_weight
        self.random_state = random_state
        self.test_size = test_size
        self.n_cv_folds = n_cv_folds

    def train_and_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        layer: int
    ) -> LRResult:
        """
        Train LR and evaluate on a single layer's hidden states.

        Args:
            X: Hidden states [n_samples, d_model]
            y: Labels [n_samples]
            layer: Layer index (for recording)

        Returns:
            LRResult with all metrics
        """
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
        lr = LogisticRegression(
            max_iter=self.max_iter,
            solver=self.solver,
            class_weight=self.class_weight,
            random_state=self.random_state
        )
        lr.fit(X_train_scaled, y_train)

        # Predictions
        y_pred = lr.predict(X_test_scaled)
        y_prob = lr.predict_proba(X_test_scaled)[:, 1]

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = 0.5  # If only one class in test set

        # Cross-validation on full data
        X_scaled = scaler.fit_transform(X)
        cv_scores = cross_val_score(
            LogisticRegression(
                max_iter=self.max_iter,
                solver=self.solver,
                class_weight=self.class_weight,
                random_state=self.random_state
            ),
            X_scaled, y,
            cv=self.n_cv_folds,
            scoring='accuracy'
        )

        return LRResult(
            layer=layer,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            auc_roc=auc,
            cv_mean=cv_scores.mean(),
            cv_std=cv_scores.std(),
            n_samples=len(y),
            n_positive=int(y.sum()),
            coef_norm=float(np.linalg.norm(lr.coef_))
        )

    def run_all_layers(
        self,
        hidden_states: Dict[int, np.ndarray],
        labels: np.ndarray,
        model_name: str,
        option: str
    ) -> ExperimentResult:
        """
        Run LR classification on all layers.

        Args:
            hidden_states: Dict of layer -> array [n_samples, d_model]
            labels: Array of labels
            model_name: Model name for recording
            option: Option name ('A', 'B', 'C')

        Returns:
            ExperimentResult with all layer results
        """
        result = ExperimentResult(
            model_name=model_name,
            option=option,
            n_samples=len(labels),
            n_positive=int(labels.sum())
        )

        best_acc = 0.0
        best_layer = 0

        for layer, X in sorted(hidden_states.items()):
            print(f"  Layer {layer}...", end=" ", flush=True)

            lr_result = self.train_and_evaluate(X, labels, layer)
            result.layer_results[layer] = lr_result

            print(f"Acc={lr_result.accuracy:.3f}, AUC={lr_result.auc_roc:.3f}, CV={lr_result.cv_mean:.3f}+/-{lr_result.cv_std:.3f}")

            if lr_result.accuracy > best_acc:
                best_acc = lr_result.accuracy
                best_layer = layer

        result.best_layer = best_layer
        result.best_accuracy = best_acc

        return result


def save_experiment_result(result: ExperimentResult, output_path: str):
    """Save experiment result to JSON."""
    data = {
        'model_name': result.model_name,
        'option': result.option,
        'n_samples': result.n_samples,
        'n_positive': result.n_positive,
        'best_layer': result.best_layer,
        'best_accuracy': result.best_accuracy,
        'layer_results': {
            str(k): {
                'layer': v.layer,
                'accuracy': v.accuracy,
                'precision': v.precision,
                'recall': v.recall,
                'f1': v.f1,
                'auc_roc': v.auc_roc,
                'cv_mean': v.cv_mean,
                'cv_std': v.cv_std,
                'n_samples': v.n_samples,
                'n_positive': v.n_positive,
                'coef_norm': v.coef_norm
            }
            for k, v in result.layer_results.items()
        }
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {output_path}")


def load_experiment_result(path: str) -> ExperimentResult:
    """Load experiment result from JSON."""
    with open(path, 'r') as f:
        data = json.load(f)

    result = ExperimentResult(
        model_name=data['model_name'],
        option=data['option'],
        n_samples=data['n_samples'],
        n_positive=data['n_positive'],
        best_layer=data['best_layer'],
        best_accuracy=data['best_accuracy']
    )

    for k, v in data['layer_results'].items():
        result.layer_results[int(k)] = LRResult(**v)

    return result


def print_summary(result: ExperimentResult):
    """Print summary of experiment result."""
    print("\n" + "="*60)
    print(f"SUMMARY: {result.model_name.upper()} - Option {result.option}")
    print("="*60)
    print(f"Samples: {result.n_samples} (positive={result.n_positive}, {result.n_positive/result.n_samples*100:.1f}%)")
    print(f"Best Layer: {result.best_layer} (accuracy={result.best_accuracy:.3f})")
    print()

    print("Layer Results:")
    print("-"*60)
    print(f"{'Layer':>6} | {'Acc':>6} | {'AUC':>6} | {'CV Mean':>8} | {'CV Std':>7}")
    print("-"*60)

    for layer in sorted(result.layer_results.keys()):
        r = result.layer_results[layer]
        marker = " *" if layer == result.best_layer else ""
        print(f"{layer:>6} | {r.accuracy:>6.3f} | {r.auc_roc:>6.3f} | {r.cv_mean:>8.3f} | {r.cv_std:>7.3f}{marker}")

    print("-"*60)


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    # Test with synthetic data
    print("Testing LRClassifier with synthetic data...")

    np.random.seed(42)
    n_samples = 1000
    d_model = 100
    n_layers = 5

    # Create synthetic hidden states with some signal
    hidden_states = {}
    for layer in range(n_layers):
        # Add increasing signal with layer depth
        signal_strength = 0.1 + 0.1 * layer
        X = np.random.randn(n_samples, d_model)
        hidden_states[layer] = X

    # Create labels with correlation to later layers
    labels = (hidden_states[n_layers-1][:, 0] > 0).astype(int)

    print(f"Synthetic data: {n_samples} samples, {d_model} dims, {n_layers} layers")
    print(f"Labels: {labels.sum()} positive, {n_samples - labels.sum()} negative")

    # Run classifier
    classifier = LRClassifier()
    result = classifier.run_all_layers(hidden_states, labels, 'synthetic', 'test')

    print_summary(result)
