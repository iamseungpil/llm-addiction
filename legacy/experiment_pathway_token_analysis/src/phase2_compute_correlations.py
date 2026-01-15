#!/usr/bin/env python3
"""
Phase 2: Feature-Feature Correlation Analysis
Compute co-activation patterns and correlations between features
Input: Phase 1 activation files
Output: Feature pair correlations (|r| > 0.7)
"""

import argparse
import json
import logging
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("phase2")

class CorrelationAnalyzer:
    def __init__(
        self,
        layers: list[int],
        condition: str,
        min_activation_mean: float = 1e-2,
        min_nonzero: int = 10,
        min_shared_trials: int = 10,
        min_abs_r: float = 0.7,
        max_p_value: float = 0.01,
        activation_threshold: float = 1e-2,
    ):
        self.layers = layers
        self.condition = condition
        self.min_activation_mean = min_activation_mean
        self.min_nonzero = min_nonzero
        self.min_shared_trials = min_shared_trials
        self.min_abs_r = min_abs_r
        self.max_p_value = max_p_value
        self.activation_threshold = activation_threshold

    def _load_layer_file(self, activation_file: Path) -> dict[str, dict]:
        """Load activations for a single layer filtered by condition."""
        layer_data: dict[str, dict] = {}

        with open(activation_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)

                if record.get("condition") != self.condition:
                    continue

                trial = record.get("trial")
                if trial is None:
                    continue

                feature = record["feature"]
                activation = record["activation"]

                entry = layer_data.setdefault(
                    feature,
                    {
                        "layer": record["layer"],
                        "feature_id": record["feature_id"],
                        "trials": {},
                    },
                )
                entry["trials"][trial] = activation

        LOGGER.info(
            "Layer %s: loaded %d features for condition %s",
            activation_file.stem.split('_')[-1],
            len(layer_data),
            self.condition,
        )
        return layer_data

    def load_activations(self, base_dir: Path) -> dict[str, dict]:
        """Load activations for all requested layers."""
        feature_data: dict[str, dict] = {}
        for layer in self.layers:
            activation_file = base_dir / f"feature_activations_L{layer}.jsonl"
            if not activation_file.exists():
                LOGGER.warning("Activation file missing for layer %s: %s", layer, activation_file)
                continue
            layer_entries = self._load_layer_file(activation_file)
            feature_data.update(layer_entries)

        LOGGER.info("Total features loaded across layers: %d", len(feature_data))
        return feature_data

    def filter_features(self, feature_data: dict[str, dict]) -> dict[str, dict]:
        """Filter out features with low activity or sparse coverage."""
        filtered: dict[str, dict] = {}
        for feature, data in feature_data.items():
            activations = np.array(list(data["trials"].values()))
            mean_activation = activations.mean()
            if mean_activation < self.min_activation_mean:
                continue

            nonzero = np.sum(activations > self.activation_threshold)
            if nonzero < self.min_nonzero:
                continue

            data["mean"] = mean_activation
            data["std"] = activations.std(ddof=1) if activations.size > 1 else 0.0
            filtered[feature] = data

        LOGGER.info("Filtered features: %d/%d retained", len(filtered), len(feature_data))
        return filtered

    def _aligned_samples(self, trials_a: dict, trials_b: dict) -> tuple[np.ndarray, np.ndarray, list]:
        shared_trials = sorted(set(trials_a.keys()) & set(trials_b.keys()))
        if len(shared_trials) < self.min_shared_trials:
            return np.array([]), np.array([]), []
        a_vals = np.array([trials_a[t] for t in shared_trials])
        b_vals = np.array([trials_b[t] for t in shared_trials])
        return a_vals, b_vals, shared_trials

    def compute_correlations(self, feature_data: dict[str, dict]):
        """Compute correlations for feature pairs and retain high-signal pairs."""
        feature_names = sorted(feature_data.keys())
        correlations = []

        for feature_a, feature_b in tqdm(
            combinations(feature_names, 2),
            desc="Computing correlations",
            total=len(feature_names) * (len(feature_names) - 1) // 2,
        ):
            data_a = feature_data[feature_a]
            data_b = feature_data[feature_b]

            samples_a, samples_b, shared_trials = self._aligned_samples(data_a["trials"], data_b["trials"])
            if shared_trials == []:
                continue

            co_active = np.sum(
                (samples_a > self.activation_threshold) & (samples_b > self.activation_threshold)
            )
            if co_active < self.min_shared_trials:
                continue

            try:
                pearson_r, p_value = stats.pearsonr(samples_a, samples_b)
            except ValueError:
                continue

            if np.isnan(pearson_r) or abs(pearson_r) < self.min_abs_r or p_value > self.max_p_value:
                continue

            spearman_rho, _ = stats.spearmanr(samples_a, samples_b)

            correlations.append(
                {
                    "feature_A": feature_a,
                    "feature_B": feature_b,
                    "layer_A": data_a["layer"],
                    "layer_B": data_b["layer"],
                    "condition": self.condition,
                    "shared_trials": len(shared_trials),
                    "co_activation_count": int(co_active),
                    "pearson_r": float(pearson_r),
                    "spearman_rho": float(spearman_rho),
                    "p_value": float(p_value),
                }
            )

        LOGGER.info("Found %d high-correlation pairs", len(correlations))
        return correlations

    def save_results(self, correlations, output_file: Path):
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            for record in correlations:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        LOGGER.info("Saved %d correlation records to %s", len(correlations), output_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers', type=str, required=True, help="Comma-separated layer numbers (e.g., '25,26')")
    parser.add_argument('--condition', type=str, required=True,
                       choices=['safe_baseline', 'safe_with_safe_patch', 'safe_with_risky_patch',
                               'risky_baseline', 'risky_with_risky_patch', 'risky_with_safe_patch'])
    parser.add_argument('--output-suffix', type=str, default='', help="Optional suffix for output filename")
    args = parser.parse_args()

    LOGGER.info(f"=== Phase 2: Feature Correlation Analysis ===")
    layers = [int(layer.strip()) for layer in args.layers.split(',') if layer.strip()]
    LOGGER.info("Layers: %s, Condition: %s", layers, args.condition)

    activation_dir = Path("/data/llm_addiction/experiment_pathway_token_analysis/results/phase1_activations")
    output_dir = Path("/data/llm_addiction/experiment_pathway_token_analysis/results/phase2_correlations")
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    output_file = output_dir / f"feature_correlations_L{'-'.join(map(str, layers))}_{args.condition}{suffix}.jsonl"

    analyzer = CorrelationAnalyzer(layers=layers, condition=args.condition)
    feature_data = analyzer.load_activations(activation_dir)
    if not feature_data:
        LOGGER.error("No activation data found for the requested layers/condition.")
        return

    filtered = analyzer.filter_features(feature_data)
    correlations = analyzer.compute_correlations(filtered)
    analyzer.save_results(correlations, output_file)

    LOGGER.info("✅ Phase 2 complete")

if __name__ == "__main__":
    main()
