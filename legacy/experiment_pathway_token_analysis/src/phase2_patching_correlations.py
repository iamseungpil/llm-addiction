#!/usr/bin/env python3
"""
Phase 2: Feature-Feature Correlation from Patching Data

Input: Phase 1 patching results (JSONL)
Output: High-correlation feature pairs (JSONL)

Key insight: When patching target features, we observe trial-level variation
in all other features' activations, enabling meaningful correlation analysis.
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

class PatchingCorrelationAnalyzer:
    def __init__(
        self,
        target_feature: str,
        condition: str,
        min_activation_mean: float = 1e-3,
        min_nonzero: int = 5,
        min_shared_trials: int = 10,
        min_abs_r: float = 0.7,
        max_p_value: float = 0.01,
        activation_threshold: float = 1e-3,
    ):
        """
        Args:
            target_feature: The feature being patched (e.g., "L11-1829")
            condition: Patching condition (e.g., "safe_mean_safe", "risky_mean_safe")
            min_activation_mean: Minimum mean activation to consider
            min_nonzero: Minimum number of non-zero activations
            min_shared_trials: Minimum shared trials for correlation
            min_abs_r: Minimum absolute correlation coefficient
            max_p_value: Maximum p-value for significance
            activation_threshold: Threshold for considering activation as "active"
        """
        self.target_feature = target_feature
        self.condition = condition
        self.min_activation_mean = min_activation_mean
        self.min_nonzero = min_nonzero
        self.min_shared_trials = min_shared_trials
        self.min_abs_r = min_abs_r
        self.max_p_value = max_p_value
        self.activation_threshold = activation_threshold

    def load_patching_data(self, patching_file: Path) -> dict[str, dict]:
        """Load Phase 1 patching data for specific target feature and condition"""
        feature_data: dict[str, dict] = {}

        with open(patching_file, 'r') as f:
            for line in tqdm(f, desc=f"Loading {patching_file.name}"):
                if not line.strip():
                    continue

                record = json.loads(line)

                # Filter by target feature and condition
                if record.get('target_feature') != self.target_feature:
                    continue

                record_cond = f"{record.get('patch_condition')}_{record.get('prompt_type')}"
                if record_cond != self.condition:
                    continue

                trial = record.get('trial')
                if trial is None:
                    continue

                # Extract all feature activations
                all_features = record.get('all_features', {})
                for feat_name, activation in all_features.items():
                    if feat_name == self.target_feature:
                        continue  # Skip target feature itself

                    entry = feature_data.setdefault(
                        feat_name,
                        {'trials': {}, 'layer': int(feat_name.split('-')[0][1:])}
                    )
                    entry['trials'][trial] = activation

        LOGGER.info(f"Loaded {len(feature_data)} features for {self.target_feature} / {self.condition}")
        return feature_data

    def filter_features(self, feature_data: dict[str, dict]) -> dict[str, dict]:
        """Filter out features with low activity or sparse coverage"""
        filtered: dict[str, dict] = {}

        for feature, data in feature_data.items():
            activations = np.array(list(data['trials'].values()))
            mean_activation = np.abs(activations).mean()  # Use absolute value for mean

            if mean_activation < self.min_activation_mean:
                continue

            nonzero = np.sum(np.abs(activations) > self.activation_threshold)
            if nonzero < self.min_nonzero:
                continue

            data['mean'] = mean_activation
            data['std'] = activations.std(ddof=1) if activations.size > 1 else 0.0
            filtered[feature] = data

        LOGGER.info(f"Filtered features: {len(filtered)}/{len(feature_data)} retained")
        return filtered

    def _aligned_samples(self, trials_a: dict, trials_b: dict) -> tuple[np.ndarray, np.ndarray, list]:
        """Get aligned samples for two features"""
        shared_trials = sorted(set(trials_a.keys()) & set(trials_b.keys()))
        if len(shared_trials) < self.min_shared_trials:
            return np.array([]), np.array([]), []

        a_vals = np.array([trials_a[t] for t in shared_trials])
        b_vals = np.array([trials_b[t] for t in shared_trials])
        return a_vals, b_vals, shared_trials

    def compute_correlations(self, feature_data: dict[str, dict]):
        """Compute correlations for feature pairs"""
        feature_names = sorted(feature_data.keys())
        correlations = []

        for feature_a, feature_b in tqdm(
            combinations(feature_names, 2),
            desc="Computing correlations",
            total=len(feature_names) * (len(feature_names) - 1) // 2,
        ):
            data_a = feature_data[feature_a]
            data_b = feature_data[feature_b]

            samples_a, samples_b, shared_trials = self._aligned_samples(data_a['trials'], data_b['trials'])
            if len(shared_trials) == 0:
                continue

            # Check co-activation
            co_active = np.sum(
                (np.abs(samples_a) > self.activation_threshold) &
                (np.abs(samples_b) > self.activation_threshold)
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

            correlations.append({
                'target_feature': self.target_feature,
                'condition': self.condition,
                'feature_A': feature_a,
                'feature_B': feature_b,
                'layer_A': data_a['layer'],
                'layer_B': data_b['layer'],
                'shared_trials': len(shared_trials),
                'co_activation_count': int(co_active),
                'pearson_r': float(pearson_r),
                'spearman_rho': float(spearman_rho),
                'p_value': float(p_value),
            })

        LOGGER.info(f"Found {len(correlations)} high-correlation pairs")
        return correlations

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patching-file', type=Path, required=True,
                       help="Phase 1 patching output file (JSONL)")
    parser.add_argument('--target-feature', type=str, required=True,
                       help="Target feature being patched (e.g., 'L11-1829')")
    parser.add_argument('--condition', type=str, required=True,
                       help="Patching condition (e.g., 'safe_mean_safe', 'risky_mean_risky')")
    parser.add_argument('--output', type=Path, required=True,
                       help="Output file for correlations (JSONL)")
    args = parser.parse_args()

    LOGGER.info(f"=== Phase 2: Feature Correlation from Patching ===")
    LOGGER.info(f"Target: {args.target_feature}, Condition: {args.condition}")

    analyzer = PatchingCorrelationAnalyzer(
        target_feature=args.target_feature,
        condition=args.condition
    )

    # Load and process data
    feature_data = analyzer.load_patching_data(args.patching_file)
    if not feature_data:
        LOGGER.error(f"No data found for {args.target_feature} / {args.condition}")
        return

    filtered = analyzer.filter_features(feature_data)
    correlations = analyzer.compute_correlations(filtered)

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        for record in correlations:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    LOGGER.info(f"✅ Phase 2 complete: {len(correlations)} correlations saved to {args.output}")

if __name__ == "__main__":
    main()
