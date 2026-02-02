#!/usr/bin/env python3
"""
Phase 2: Correlation Analysis

Compare SAE feature activations between bankruptcy and voluntary_stop cases.

Input:
    - Phase 1 outputs: layer_{L}_features.npz

Output:
    - phase2_results/significant_features.json
    - phase2_results/all_features_stats.json

Usage:
    python phase2_correlation_analysis.py
    python phase2_correlation_analysis.py --layers 25,30,35
"""

import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Optional
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))

from config import load_config, get_phase_output_dir
from utils import compute_cohens_d, benjamini_hochberg, logger


class Phase2CorrelationAnalysis:
    """Analyze correlations between SAE features and gambling outcomes."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        target_layers: Optional[List[int]] = None
    ):
        self.config = load_config(config_path)

        if target_layers:
            self.target_layers = target_layers
        else:
            self.target_layers = self.config.get('target_layers', list(range(20, 42)))

        # Directories
        self.input_dir = get_phase_output_dir(self.config, 1)
        self.output_dir = get_phase_output_dir(self.config, 2)

        # Statistical thresholds
        phase2_config = self.config.get('phase2', {})
        self.fdr_alpha = phase2_config.get('fdr_alpha', 0.05)
        self.min_cohens_d = phase2_config.get('min_cohens_d', 0.3)
        self.min_samples = phase2_config.get('min_samples', 10)

    def load_layer_features(self, layer: int) -> Optional[Dict]:
        """Load features for a specific layer."""
        feature_file = self.input_dir / f"layer_{layer}_features.npz"

        if not feature_file.exists():
            logger.warning(f"Feature file not found: {feature_file}")
            return None

        data = np.load(feature_file, allow_pickle=True)

        features = data['features']
        outcomes = data['outcomes']

        # Separate by outcome
        bankrupt_mask = outcomes == 'bankruptcy'
        safe_mask = outcomes == 'voluntary_stop'

        return {
            'bankrupt_features': features[bankrupt_mask],
            'safe_features': features[safe_mask],
            'n_features': features.shape[1],
            'n_bankrupt': np.sum(bankrupt_mask),
            'n_safe': np.sum(safe_mask)
        }

    def analyze_layer(self, layer: int) -> List[Dict]:
        """
        Analyze all features in a layer.

        Returns:
            List of feature statistics dictionaries
        """
        data = self.load_layer_features(layer)
        if data is None:
            return []

        bankrupt_features = data['bankrupt_features']
        safe_features = data['safe_features']
        n_features = data['n_features']

        logger.info(f"  Layer {layer}: {data['n_bankrupt']} bankrupt, {data['n_safe']} safe")

        if data['n_bankrupt'] < self.min_samples or data['n_safe'] < self.min_samples:
            logger.warning(f"  Insufficient samples for layer {layer}")
            return []

        results = []

        for feature_id in range(n_features):
            b_vals = bankrupt_features[:, feature_id]
            s_vals = safe_features[:, feature_id]

            # Skip if no variance
            if np.std(b_vals) == 0 and np.std(s_vals) == 0:
                continue

            # Compute statistics
            b_mean = np.mean(b_vals)
            s_mean = np.mean(s_vals)
            cohens_d = compute_cohens_d(b_vals, s_vals)

            # Welch's t-test
            try:
                t_stat, p_value = stats.ttest_ind(b_vals, s_vals, equal_var=False)
            except Exception:
                continue

            results.append({
                'layer': layer,
                'feature_id': int(feature_id),
                'bankrupt_mean': float(b_mean),
                'safe_mean': float(s_mean),
                'bankrupt_std': float(np.std(b_vals)),
                'safe_std': float(np.std(s_vals)),
                'difference': float(b_mean - s_mean),
                'cohens_d': float(cohens_d),
                't_stat': float(t_stat),
                'p_value': float(p_value),
                'direction': 'risky' if cohens_d > 0 else 'safe'
            })

        return results

    def run(self):
        """Run Phase 2 correlation analysis."""
        logger.info("=" * 70)
        logger.info("PHASE 2: CORRELATION ANALYSIS")
        logger.info("=" * 70)
        logger.info(f"FDR alpha: {self.fdr_alpha}")
        logger.info(f"Min Cohen's d: {self.min_cohens_d}")

        all_results = []

        # Analyze each layer
        for layer in tqdm(self.target_layers, desc="Analyzing layers"):
            layer_results = self.analyze_layer(layer)
            all_results.extend(layer_results)

        if not all_results:
            logger.error("No results to analyze!")
            return

        logger.info(f"\nTotal features analyzed: {len(all_results)}")

        # Apply FDR correction
        p_values = [r['p_value'] for r in all_results]
        significant, q_values = benjamini_hochberg(p_values, self.fdr_alpha)

        for i, result in enumerate(all_results):
            result['q_value'] = q_values[i]
            result['fdr_significant'] = significant[i]

        # Filter significant features
        significant_features = [
            r for r in all_results
            if r['fdr_significant'] and abs(r['cohens_d']) >= self.min_cohens_d
        ]

        # Separate by direction
        risky_features = [f for f in significant_features if f['direction'] == 'risky']
        safe_features = [f for f in significant_features if f['direction'] == 'safe']

        # Sort by absolute Cohen's d
        risky_features.sort(key=lambda x: abs(x['cohens_d']), reverse=True)
        safe_features.sort(key=lambda x: abs(x['cohens_d']), reverse=True)

        logger.info(f"\nSignificant features: {len(significant_features)}")
        logger.info(f"  Risky features (d > 0): {len(risky_features)}")
        logger.info(f"  Safe features (d < 0): {len(safe_features)}")

        # Print top features
        logger.info("\nTop 10 Risky Features:")
        for f in risky_features[:10]:
            logger.info(f"  L{f['layer']}:F{f['feature_id']} d={f['cohens_d']:.3f} q={f['q_value']:.4f}")

        logger.info("\nTop 10 Safe Features:")
        for f in safe_features[:10]:
            logger.info(f"  L{f['layer']}:F{f['feature_id']} d={f['cohens_d']:.3f} q={f['q_value']:.4f}")

        # Save results
        self._save_results(all_results, significant_features, risky_features, safe_features)

        logger.info("\nPhase 2 complete!")

    def _save_results(
        self,
        all_results: List[Dict],
        significant: List[Dict],
        risky: List[Dict],
        safe: List[Dict]
    ):
        """Save analysis results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save significant features
        significant_output = {
            'timestamp': timestamp,
            'config': {
                'fdr_alpha': self.fdr_alpha,
                'min_cohens_d': self.min_cohens_d
            },
            'summary': {
                'total_features_analyzed': len(all_results),
                'significant_features': len(significant),
                'risky_features': len(risky),
                'safe_features': len(safe)
            },
            'risky_features': risky,
            'safe_features': safe
        }

        sig_file = self.output_dir / "significant_features.json"
        with open(sig_file, 'w') as f:
            json.dump(significant_output, f, indent=2)
        logger.info(f"Saved: {sig_file}")

        # Save all features stats
        all_file = self.output_dir / "all_features_stats.json"
        with open(all_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'n_features': len(all_results),
                'results': all_results
            }, f, indent=2)
        logger.info(f"Saved: {all_file}")

        # Save per-layer summary
        layer_summary = {}
        for layer in self.target_layers:
            layer_features = [f for f in significant if f['layer'] == layer]
            layer_summary[layer] = {
                'n_significant': len(layer_features),
                'n_risky': sum(1 for f in layer_features if f['direction'] == 'risky'),
                'n_safe': sum(1 for f in layer_features if f['direction'] == 'safe')
            }

        summary_file = self.output_dir / "layer_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(layer_summary, f, indent=2)
        logger.info(f"Saved: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description='Phase 2: Correlation Analysis')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    parser.add_argument('--layers', type=str, default=None,
                       help='Comma-separated layer numbers')

    args = parser.parse_args()

    target_layers = None
    if args.layers:
        target_layers = [int(l) for l in args.layers.split(',')]

    phase2 = Phase2CorrelationAnalysis(
        config_path=args.config,
        target_layers=target_layers
    )
    phase2.run()


if __name__ == '__main__':
    main()
