#!/usr/bin/env python3
"""
Phase 5 (Redesigned): Input Prompt-Feature Correlation Analysis

Analyzes the correlation between:
- Input prompt type (safe vs risky)
- SAE feature activations (all 2,787 features)

This reveals:
- Which features are differentially activated by prompt type
- Feature sensitivity to input framing
- Prompt-feature interaction effects
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import logging
import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
LOGGER = logging.getLogger(__name__)


class PromptFeatureCorrelationAnalyzer:
    def __init__(self, patching_file: Path, output_file: Path):
        self.patching_file = patching_file
        self.output_file = output_file

        # Store feature activations by prompt type
        self.prompt_features = defaultdict(lambda: defaultdict(list))
        # prompt_features[prompt_type][feature] = [activation values]

    def analyze_patching_data(self):
        """Collect prompt-feature activation data"""
        LOGGER.info(f"Analyzing: {self.patching_file}")

        total_records = 0
        with open(self.patching_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue

                record = json.loads(line)
                prompt_type = record['prompt_type']  # 'safe' or 'risky'
                all_features = record['all_features']

                # Record all feature activations for this prompt type
                for feature_name, activation_value in all_features.items():
                    self.prompt_features[prompt_type][feature_name].append(activation_value)

                total_records += 1

                if total_records % 1000 == 0:
                    LOGGER.info(f"Processed {total_records:,} records...")

        LOGGER.info(f"Total records analyzed: {total_records:,}")
        LOGGER.info(f"Prompt types: {list(self.prompt_features.keys())}")

    def compute_correlations(self) -> Dict:
        """Compute prompt-feature differential activations"""
        LOGGER.info("Computing prompt-feature correlations...")

        results = {
            'prompt_types': list(self.prompt_features.keys()),
            'feature_comparisons': []
        }

        # Get all feature names
        all_features = set()
        for prompt_type, features in self.prompt_features.items():
            all_features.update(features.keys())

        LOGGER.info(f"Analyzing {len(all_features):,} features across prompt types")

        feature_comparisons = []

        for feature_idx, feature_name in enumerate(sorted(all_features)):
            if (feature_idx + 1) % 500 == 0:
                LOGGER.info(f"  Processing feature {feature_idx + 1}/{len(all_features)}...")

            # Get activations for safe and risky prompts
            safe_activations = self.prompt_features['safe'][feature_name]
            risky_activations = self.prompt_features['risky'][feature_name]

            if len(safe_activations) < 10 or len(risky_activations) < 10:
                continue

            # Compute statistics
            safe_mean = np.mean(safe_activations)
            safe_std = np.std(safe_activations)
            risky_mean = np.mean(risky_activations)
            risky_std = np.std(risky_activations)

            # Compute effect size (Cohen's d)
            pooled_std = np.sqrt((safe_std**2 + risky_std**2) / 2)
            if pooled_std > 0:
                cohens_d = (risky_mean - safe_mean) / pooled_std
            else:
                cohens_d = 0.0

            # Perform t-test
            t_stat, p_value = stats.ttest_ind(risky_activations, safe_activations)

            feature_comparisons.append({
                'feature': feature_name,
                'safe_mean': float(safe_mean),
                'safe_std': float(safe_std),
                'safe_count': len(safe_activations),
                'risky_mean': float(risky_mean),
                'risky_std': float(risky_std),
                'risky_count': len(risky_activations),
                'mean_difference': float(risky_mean - safe_mean),
                'cohens_d': float(cohens_d),
                't_statistic': float(t_stat),
                'p_value': float(p_value)
            })

        # Sort by absolute Cohen's d (effect size)
        feature_comparisons.sort(key=lambda x: abs(x['cohens_d']), reverse=True)

        results['feature_comparisons'] = feature_comparisons

        # Compute summary statistics
        results['summary'] = self._compute_summary(feature_comparisons)

        return results

    def _compute_summary(self, feature_comparisons: List[Dict]) -> Dict:
        """Compute summary statistics"""
        # Significant features (p < 0.05)
        significant = [f for f in feature_comparisons if f['p_value'] < 0.05]

        # Large effect size (|Cohen's d| > 0.5)
        large_effect = [f for f in feature_comparisons if abs(f['cohens_d']) > 0.5]

        # Very large effect (|Cohen's d| > 1.0)
        very_large_effect = [f for f in feature_comparisons if abs(f['cohens_d']) > 1.0]

        # Features more activated by risky prompts
        risky_biased = [f for f in significant if f['mean_difference'] > 0]

        # Features more activated by safe prompts
        safe_biased = [f for f in significant if f['mean_difference'] < 0]

        return {
            'total_features': len(feature_comparisons),
            'significant_features': len(significant),
            'large_effect_features': len(large_effect),
            'very_large_effect_features': len(very_large_effect),
            'risky_biased_features': len(risky_biased),
            'safe_biased_features': len(safe_biased),
            'top_risky_biased': risky_biased[:20],
            'top_safe_biased': safe_biased[:20],
            'top_effect_sizes': feature_comparisons[:50]
        }

    def save_results(self, results: Dict):
        """Save results to JSON file"""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_file, 'w') as f:
            json.dump(results, f, indent=2)

        LOGGER.info(f"Results saved to: {self.output_file}")


def main():
    parser = argparse.ArgumentParser(description='Phase 5: Input Prompt-Feature Correlation Analysis')
    parser.add_argument('--patching-file', type=str, required=True, help='Phase 1 patching JSONL file')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file')

    args = parser.parse_args()

    patching_file = Path(args.patching_file)
    output_file = Path(args.output)

    if not patching_file.exists():
        LOGGER.error(f"Patching file not found: {patching_file}")
        return

    analyzer = PromptFeatureCorrelationAnalyzer(patching_file, output_file)

    # Analyze
    analyzer.analyze_patching_data()

    # Compute correlations
    results = analyzer.compute_correlations()

    # Save
    analyzer.save_results(results)

    LOGGER.info("✅ Phase 5 prompt-feature correlation analysis complete!")


if __name__ == '__main__':
    main()
