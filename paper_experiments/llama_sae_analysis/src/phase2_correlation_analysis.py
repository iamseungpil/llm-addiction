#!/usr/bin/env python3
"""
Phase 2: Correlation Analysis with FDR Correction
Analyzes SAE feature correlations with bankruptcy/safe outcomes.
"""

import os
import sys
import json
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import argparse
import logging
from scipy import stats
from statsmodels.stats.multitest import multipletests


class CorrelationAnalyzer:
    """Analyze correlations between SAE features and gambling outcomes"""

    def __init__(self, config: dict, model_type: str):
        self.config = config
        self.model_type = model_type
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging"""
        log_dir = Path(self.config['data']['logs_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'phase2_{self.model_type}_{timestamp}.log'

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_layer_features(self, layer: int) -> tuple:
        """Load features for a specific layer"""
        data_dir = Path(self.config['data'][self.model_type]['output_dir'])
        feature_file = data_dir / f'layer_{layer}_features.npz'

        if not feature_file.exists():
            self.logger.warning(f"Feature file not found: {feature_file}")
            return None, None, None

        data = np.load(feature_file, allow_pickle=True)
        features = data['features']
        outcomes = data['outcomes']

        # Separate by outcome
        bankrupt_mask = outcomes == 'bankruptcy'
        safe_mask = outcomes == 'voluntary_stop'

        bankrupt_features = features[bankrupt_mask]
        safe_features = features[safe_mask]

        return bankrupt_features, safe_features, features.shape[1]

    def compute_cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Compute Cohen's d effect size"""
        n1, n2 = len(group1), len(group2)
        if n1 < 2 or n2 < 2:
            return 0.0

        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0

        return (np.mean(group1) - np.mean(group2)) / pooled_std

    def analyze_layer(self, layer: int) -> list:
        """Analyze all features in a layer"""
        self.logger.info(f"Analyzing layer {layer}")

        bankrupt_features, safe_features, n_features = self.load_layer_features(layer)

        if bankrupt_features is None:
            return []

        self.logger.info(f"  Bankrupt samples: {len(bankrupt_features)}")
        self.logger.info(f"  Safe samples: {len(safe_features)}")
        self.logger.info(f"  Features: {n_features}")

        results = []

        for feature_id in tqdm(range(n_features), desc=f"Layer {layer}", leave=False):
            b_vals = bankrupt_features[:, feature_id]
            s_vals = safe_features[:, feature_id]

            # Skip if no variance
            if np.std(b_vals) == 0 and np.std(s_vals) == 0:
                continue

            # t-test
            try:
                t_stat, p_value = stats.ttest_ind(b_vals, s_vals, equal_var=False)
            except:
                continue

            # Cohen's d
            cohens_d = self.compute_cohens_d(b_vals, s_vals)

            # Means
            b_mean = np.mean(b_vals)
            s_mean = np.mean(s_vals)

            results.append({
                'layer': layer,
                'feature_id': feature_id,
                't_stat': float(t_stat),
                'p_value': float(p_value) if not np.isnan(p_value) else 1.0,
                'cohens_d': float(cohens_d),
                'bankrupt_mean': float(b_mean),
                'safe_mean': float(s_mean),
                'bankrupt_std': float(np.std(b_vals)),
                'safe_std': float(np.std(s_vals))
            })

        return results

    def run_analysis(self):
        """Run full correlation analysis"""
        output_dir = Path(self.config['data'][self.model_type]['output_dir'])
        layers = self.config['models'][self.model_type]['layers']

        self.logger.info(f"Starting correlation analysis for {self.model_type}")
        self.logger.info(f"Layers: {layers}")

        # Collect all results
        all_results = []

        for layer in layers:
            layer_results = self.analyze_layer(layer)
            all_results.extend(layer_results)
            self.logger.info(f"Layer {layer}: {len(layer_results)} features analyzed")

        if not all_results:
            self.logger.error("No results collected!")
            return

        self.logger.info(f"\nTotal features analyzed: {len(all_results)}")

        # Apply multiple testing corrections
        self.logger.info("Applying FDR and Bonferroni corrections...")

        p_values = np.array([r['p_value'] for r in all_results])

        # Replace NaN with 1.0
        p_values = np.nan_to_num(p_values, nan=1.0)

        # FDR (Benjamini-Hochberg)
        fdr_alpha = self.config['correlation']['fdr_alpha']
        fdr_rejected, fdr_pvals, _, _ = multipletests(
            p_values, alpha=fdr_alpha, method='fdr_bh'
        )

        # Bonferroni
        bonf_alpha = self.config['correlation']['bonferroni_alpha']
        bonf_rejected, bonf_pvals, _, _ = multipletests(
            p_values, alpha=bonf_alpha, method='bonferroni'
        )

        # Add corrections to results
        for i, r in enumerate(all_results):
            r['p_fdr'] = float(fdr_pvals[i])
            r['fdr_significant'] = bool(fdr_rejected[i])
            r['p_bonferroni'] = float(bonf_pvals[i])
            r['bonferroni_significant'] = bool(bonf_rejected[i])

        # Filter significant features
        min_d = self.config['correlation']['min_cohens_d']

        fdr_significant = [r for r in all_results
                          if r['fdr_significant'] and abs(r['cohens_d']) >= min_d]
        bonf_significant = [r for r in all_results
                           if r['bonferroni_significant'] and abs(r['cohens_d']) >= min_d]

        # Separate by direction
        safe_features = [r for r in fdr_significant if r['cohens_d'] < 0]
        risky_features = [r for r in fdr_significant if r['cohens_d'] > 0]

        # Sort by effect size
        safe_features.sort(key=lambda x: x['cohens_d'])
        risky_features.sort(key=lambda x: x['cohens_d'], reverse=True)

        # Summary statistics
        summary = {
            'model_type': self.model_type,
            'timestamp': datetime.now().isoformat(),
            'total_features_analyzed': len(all_results),
            'fdr_alpha': fdr_alpha,
            'bonferroni_alpha': bonf_alpha,
            'min_cohens_d': min_d,
            'fdr_significant_count': len(fdr_significant),
            'bonferroni_significant_count': len(bonf_significant),
            'safe_features_count': len(safe_features),
            'risky_features_count': len(risky_features),
            'layer_breakdown': {},
            'effect_size_stats': {
                'mean_abs_d': float(np.mean([abs(r['cohens_d']) for r in fdr_significant])) if fdr_significant else 0,
                'max_safe_d': float(safe_features[0]['cohens_d']) if safe_features else 0,
                'max_risky_d': float(risky_features[0]['cohens_d']) if risky_features else 0
            }
        }

        # Layer breakdown
        for layer in layers:
            layer_fdr = [r for r in fdr_significant if r['layer'] == layer]
            summary['layer_breakdown'][str(layer)] = {
                'fdr_significant': len(layer_fdr),
                'safe': len([r for r in layer_fdr if r['cohens_d'] < 0]),
                'risky': len([r for r in layer_fdr if r['cohens_d'] > 0])
            }

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save summary
        summary_file = output_dir / f'correlation_summary_{timestamp}.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        # Save all results
        all_results_file = output_dir / f'correlation_all_features_{timestamp}.json'
        with open(all_results_file, 'w') as f:
            json.dump(all_results, f)

        # Save significant features (easier to work with)
        sig_file = output_dir / f'correlation_significant_{timestamp}.json'
        with open(sig_file, 'w') as f:
            json.dump({
                'summary': summary,
                'safe_features': safe_features[:100],  # Top 100
                'risky_features': risky_features[:100]
            }, f, indent=2)

        # Save top features for phase 3
        top_file = output_dir / 'top_features_for_analysis.json'
        n_top = self.config['semantic']['n_top_features']
        with open(top_file, 'w') as f:
            json.dump({
                'safe_features': safe_features[:n_top],
                'risky_features': risky_features[:n_top]
            }, f, indent=2)

        # Print summary
        self.logger.info("\n" + "="*60)
        self.logger.info("CORRELATION ANALYSIS SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"Total features analyzed: {len(all_results)}")
        self.logger.info(f"FDR significant (|d|>{min_d}): {len(fdr_significant)}")
        self.logger.info(f"Bonferroni significant (|d|>{min_d}): {len(bonf_significant)}")
        self.logger.info(f"  - Safe features (d<0): {len(safe_features)}")
        self.logger.info(f"  - Risky features (d>0): {len(risky_features)}")

        if safe_features:
            self.logger.info(f"\nTop 5 SAFE features:")
            for f in safe_features[:5]:
                self.logger.info(f"  L{f['layer']}-{f['feature_id']}: d={f['cohens_d']:.3f}, p_fdr={f['p_fdr']:.2e}")

        if risky_features:
            self.logger.info(f"\nTop 5 RISKY features:")
            for f in risky_features[:5]:
                self.logger.info(f"  L{f['layer']}-{f['feature_id']}: d={f['cohens_d']:.3f}, p_fdr={f['p_fdr']:.2e}")

        self.logger.info(f"\nResults saved to: {output_dir}")
        self.logger.info("="*60)

        return summary


def main():
    parser = argparse.ArgumentParser(description='Phase 2: Correlation Analysis')
    parser.add_argument('--model', type=str, required=True, choices=['llama', 'gemma'],
                        help='Model type to analyze')
    parser.add_argument('--config', type=str,
                        default='/home/ubuntu/llm_addiction/experiment_corrected_sae_analysis/configs/analysis_config.yaml',
                        help='Path to config file')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Run analysis
    analyzer = CorrelationAnalyzer(config, args.model)
    analyzer.run_analysis()


if __name__ == '__main__':
    main()
