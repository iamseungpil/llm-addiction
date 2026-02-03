#!/usr/bin/env python3
"""
Phase 2: Correlation Analysis for Alternative Paradigms

Identifies risky/safe SAE features from Phase 1 extracted features.
Follows same structure as paper_experiments/llama_sae_analysis/phase2_correlation_analysis.py

Usage:
    python src/common/phase2_correlation_analysis.py --paradigm lootbox --model llama
    python src/common/phase2_correlation_analysis.py --paradigm blackjack --model gemma
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common import setup_logger

logger = setup_logger(__name__)


class ParadigmCorrelationAnalyzer:
    """Analyze correlation between SAE features and outcomes"""

    def __init__(
        self,
        paradigm: str,
        model_name: str,
        fdr_alpha: float = 0.05,
        min_cohens_d: float = 0.3,
        data_dir: str = "/mnt/c/Users/oollccddss/git/data/llm-addiction/alternative_paradigms"
    ):
        """
        Initialize correlation analyzer.

        Args:
            paradigm: 'lootbox' or 'blackjack'
            model_name: 'llama' or 'gemma'
            fdr_alpha: FDR alpha level
            min_cohens_d: Minimum Cohen's d threshold
            data_dir: Data directory
        """
        self.paradigm = paradigm
        self.model_name = model_name
        self.fdr_alpha = fdr_alpha
        self.min_cohens_d = min_cohens_d

        self.features_dir = Path(data_dir) / paradigm / 'sae_features'
        self.output_dir = self.features_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_layer_features(self, layer: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load features for a specific layer.

        Args:
            layer: Layer number

        Returns:
            (features, outcomes)
            features: (n_samples, n_features)
            outcomes: (n_samples,) array of 'bankrupt' or 'voluntary_stop'
        """
        npz_file = self.features_dir / f'layer_{layer}_features.npz'

        if not npz_file.exists():
            raise FileNotFoundError(f"Features file not found: {npz_file}")

        data = np.load(npz_file, allow_pickle=True)
        features = data['features']
        outcomes = data['outcomes']

        logger.info(f"Layer {layer}: {features.shape[0]} samples, {features.shape[1]} features")

        return features, outcomes

    @staticmethod
    def welch_ttest(group1: np.ndarray, group2: np.ndarray) -> Tuple[float, float]:
        """
        Perform Welch's t-test.

        Args:
            group1: First group values
            group2: Second group values

        Returns:
            (t_stat, p_value)
        """
        if len(group1) < 2 or len(group2) < 2:
            return 0.0, 1.0
        if np.std(group1) == 0 and np.std(group2) == 0:
            return 0.0, 1.0

        try:
            t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
            if np.isnan(p_value):
                p_value = 1.0
            return float(t_stat), float(p_value)
        except Exception:
            return 0.0, 1.0

    @staticmethod
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
        if n1 < 2 or n2 < 2:
            return 0.0

        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0

        return (np.mean(group1) - np.mean(group2)) / pooled_std

    def analyze_layer(self, layer: int) -> List[Dict]:
        """
        Analyze one layer.

        Args:
            layer: Layer number

        Returns:
            List of feature results
        """
        logger.info(f"\nAnalyzing layer {layer}...")

        # Load features
        features, outcomes = self.load_layer_features(layer)

        # Split by outcome
        bankrupt_mask = (outcomes == 'bankrupt')
        safe_mask = (outcomes == 'voluntary_stop')

        n_bankrupt = bankrupt_mask.sum()
        n_safe = safe_mask.sum()

        logger.info(f"  Bankrupt: {n_bankrupt}, Voluntary stop: {n_safe}")

        if n_bankrupt < 2 or n_safe < 2:
            logger.warning(f"  Insufficient samples, skipping layer {layer}")
            return []

        # Analyze each feature
        results = []

        for feature_id in tqdm(range(features.shape[1]), desc=f"  Layer {layer}"):
            bankrupt_vals = features[bankrupt_mask, feature_id]
            safe_vals = features[safe_mask, feature_id]

            # Welch's t-test
            t_stat, p_value = self.welch_ttest(bankrupt_vals, safe_vals)

            # Cohen's d
            cohens_d = self.compute_cohens_d(bankrupt_vals, safe_vals)

            results.append({
                'layer': layer,
                'feature_id': feature_id,
                't_stat': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'bankrupt_mean': float(np.mean(bankrupt_vals)),
                'safe_mean': float(np.mean(safe_vals)),
                'bankrupt_std': float(np.std(bankrupt_vals)),
                'safe_std': float(np.std(safe_vals))
            })

        return results

    def run_analysis(self):
        """Run full Phase 2 correlation analysis"""
        logger.info(f"\n{'='*60}")
        logger.info(f"PHASE 2: CORRELATION ANALYSIS")
        logger.info(f"Paradigm: {self.paradigm.upper()}")
        logger.info(f"Model: {self.model_name.upper()}")
        logger.info(f"{'='*60}\n")

        # Find all layer files
        layer_files = sorted(self.features_dir.glob('layer_*_features.npz'))
        layers = [int(f.stem.split('_')[1]) for f in layer_files]

        logger.info(f"Found {len(layers)} layers: {layers}")

        # Analyze each layer
        all_results = []

        for layer in layers:
            layer_results = self.analyze_layer(layer)
            all_results.extend(layer_results)

        logger.info(f"\nTotal features analyzed: {len(all_results)}")

        # FDR correction
        logger.info("\nApplying FDR correction...")
        p_values = [r['p_value'] for r in all_results]

        reject, p_fdr, _, _ = multipletests(
            p_values,
            alpha=self.fdr_alpha,
            method='fdr_bh'
        )

        # Add FDR results
        for i, result in enumerate(all_results):
            result['p_fdr'] = float(p_fdr[i])
            result['fdr_significant'] = bool(reject[i])

        # Identify risky and safe features
        risky_features = []
        safe_features = []

        for result in all_results:
            if result['fdr_significant'] and abs(result['cohens_d']) >= self.min_cohens_d:
                if result['cohens_d'] > 0:
                    # Bankrupt > Safe = Risky
                    risky_features.append(result)
                else:
                    # Safe > Bankrupt = Safe
                    safe_features.append(result)

        logger.info(f"  FDR significant features: {sum(reject)}")
        logger.info(f"  Risky features (d > {self.min_cohens_d}): {len(risky_features)}")
        logger.info(f"  Safe features (d < -{self.min_cohens_d}): {len(safe_features)}")

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save all features
        all_file = self.output_dir / f'correlation_all_features_{timestamp}.json'
        with open(all_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"\nSaved all features to: {all_file}")

        # Save significant features
        sig_file = self.output_dir / f'correlation_significant_{timestamp}.json'
        with open(sig_file, 'w') as f:
            json.dump({
                'summary': {
                    'paradigm': self.paradigm,
                    'model_type': self.model_name,
                    'timestamp': timestamp,
                    'total_features_analyzed': len(all_results),
                    'fdr_alpha': self.fdr_alpha,
                    'min_cohens_d': self.min_cohens_d,
                    'fdr_significant_count': sum(reject),
                    'risky_features_count': len(risky_features),
                    'safe_features_count': len(safe_features)
                },
                'risky_features': risky_features[:100],  # Top 100 for inspection
                'safe_features': safe_features[:100]
            }, f, indent=2)
        logger.info(f"Saved significant features to: {sig_file}")

        # Save summary
        summary_file = self.output_dir / f'correlation_summary_{timestamp}.json'
        with open(summary_file, 'w') as f:
            json.dump({
                'paradigm': self.paradigm,
                'model_type': self.model_name,
                'timestamp': timestamp,
                'total_features_analyzed': len(all_results),
                'fdr_alpha': self.fdr_alpha,
                'min_cohens_d': self.min_cohens_d,
                'fdr_significant_count': sum(reject),
                'risky_features_count': len(risky_features),
                'safe_features_count': len(safe_features)
            }, f, indent=2)
        logger.info(f"Saved summary to: {summary_file}")

        logger.info(f"\n{'='*60}")
        logger.info("PHASE 2 COMPLETE")
        logger.info(f"{'='*60}")

        return {
            'all_results': all_results,
            'risky_features': risky_features,
            'safe_features': safe_features
        }


def main():
    parser = argparse.ArgumentParser(description='Phase 2: Correlation Analysis')
    parser.add_argument('--paradigm', type=str, required=True, choices=['lootbox', 'blackjack'],
                        help='Paradigm to analyze')
    parser.add_argument('--model', type=str, required=True, choices=['llama', 'gemma'],
                        help='Model type')
    parser.add_argument('--fdr-alpha', type=float, default=0.05,
                        help='FDR alpha level')
    parser.add_argument('--min-cohens-d', type=float, default=0.3,
                        help='Minimum Cohen\'s d threshold')
    parser.add_argument('--data-dir', type=str,
                        default='/mnt/c/Users/oollccddss/git/data/llm-addiction/alternative_paradigms',
                        help='Data directory')

    args = parser.parse_args()

    # Run analysis
    analyzer = ParadigmCorrelationAnalyzer(
        paradigm=args.paradigm,
        model_name=args.model,
        fdr_alpha=args.fdr_alpha,
        min_cohens_d=args.min_cohens_d,
        data_dir=args.data_dir
    )

    analyzer.run_analysis()


if __name__ == '__main__':
    main()
