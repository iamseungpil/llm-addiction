#!/usr/bin/env python3
"""
Prompt Complexity Analysis for SAE Features

Analyzes how prompt complexity (number of components: 0-5) affects
SAE feature activation in bankruptcy vs safe outcomes.

Complexity levels:
  0: BASE (100 games)
  1: Single component (500 games)
  2: Two components (1,000 games)
  3: Three components (1,000 games)
  4: Four components (500 games)
  5: Five components (100 games)

Usage:
    python prompt_complexity_analysis.py --model llama
    python prompt_complexity_analysis.py --model gemma
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
from typing import Dict, List, Optional

from .utils import (
    DataLoader, StatisticalAnalyzer,
    load_prompt_metadata, save_results
)


class PromptComplexityAnalyzer:
    """Analyze SAE feature differences by prompt complexity"""

    def __init__(self, config: dict, model_type: str):
        self.config = config
        self.model_type = model_type
        self.data_loader = DataLoader(config, model_type)
        self.stats = StatisticalAnalyzer()
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging"""
        log_dir = Path(self.config['data']['logs_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'prompt_complexity_{self.model_type}_{timestamp}.log'

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _filter_sparse_features(
        self,
        features: np.ndarray,
        complexity: np.ndarray,
        outcomes: np.ndarray
    ) -> np.ndarray:
        """Filter out extremely sparse features"""
        cfg = self.config.get('prompt_complexity_analysis', {}).get('sparse_filter', {})

        if not cfg.get('enabled', True):
            return np.ones(features.shape[1], dtype=bool)

        n_features = features.shape[1]
        valid_mask = np.ones(n_features, dtype=bool)

        min_rate = cfg.get('min_activation_rate', 0.01)
        min_mean = cfg.get('min_mean_activation', 0.001)

        for feat_id in range(n_features):
            feat_vals = features[:, feat_id]

            activation_rate = np.count_nonzero(feat_vals) / len(feat_vals)
            mean_activation = np.mean(feat_vals)

            if activation_rate < min_rate or mean_activation < min_mean:
                valid_mask[feat_id] = False

        return valid_mask

    def analyze_complexity_layer(self, layer: int) -> List[dict]:
        """
        Analyze Complexity × Outcome interaction for a single layer.

        Args:
            layer: Layer number

        Returns:
            List of feature results
        """
        # Load features and metadata
        result = self.data_loader.load_layer_features(layer)
        if result is None:
            return []

        features, outcomes, bet_types = result
        n_games = len(features)
        game_ids = np.arange(n_games)

        # Load prompt metadata
        prompt_meta = load_prompt_metadata(
            str(self.config['data'][self.model_type]['experiment_file']),
            game_ids
        )

        complexity = prompt_meta['complexity']

        # Filter sparse features
        valid_mask = self._filter_sparse_features(features, complexity, outcomes)
        n_valid = valid_mask.sum()
        n_total = len(valid_mask)

        if n_valid == 0:
            self.logger.warning(f"Layer {layer}: No valid features after sparse filtering")
            return []

        self.logger.info(
            f"Layer {layer}: {n_valid}/{n_total} features after sparse filtering "
            f"({100 * n_valid / n_total:.1f}%)"
        )

        # Two-way ANOVA: Complexity × Outcome
        results = []

        for feature_id in np.where(valid_mask)[0]:
            feat_vals = features[:, feature_id]

            try:
                # 2-way ANOVA
                main_complexity, main_outcome, interaction, group_means = \
                    self.stats.two_way_anova_simple(
                        feat_vals, complexity, outcomes
                    )

                # Calculate means per complexity level
                complexity_means = {}
                for level in range(6):
                    mask = complexity == level
                    if mask.sum() > 0:
                        complexity_means[f'level_{level}'] = float(np.mean(feat_vals[mask]))
                    else:
                        complexity_means[f'level_{level}'] = 0.0

                results.append({
                    'layer': layer,
                    'feature_id': int(feature_id),
                    # Main effect: Complexity
                    'complexity_f': main_complexity['f'],
                    'complexity_p': main_complexity['p'],
                    'complexity_eta': main_complexity['eta_squared'],
                    # Main effect: Outcome
                    'outcome_f': main_outcome['f'],
                    'outcome_p': main_outcome['p'],
                    'outcome_eta': main_outcome['eta_squared'],
                    # Interaction: Complexity × Outcome
                    'interaction_f': interaction['f'],
                    'interaction_p': interaction['p'],
                    'interaction_eta': interaction['eta_squared'],
                    # Group means
                    'group_means': group_means,
                    # Complexity means
                    'complexity_means': complexity_means,
                })

            except Exception as e:
                self.logger.warning(f"Layer {layer}, Feature {feature_id}: {e}")
                continue

        return results

    def run_analysis(self):
        """Run full complexity analysis"""
        layers = self.config['models'][self.model_type]['layers']
        fdr_alpha = self.config.get('prompt_complexity_analysis', {}).get('fdr_alpha', 0.05)
        min_eta = self.config.get('prompt_complexity_analysis', {}).get('min_eta_squared', 0.01)
        n_top = self.config.get('prompt_complexity_analysis', {}).get('n_top_features', 30)

        self.logger.info(f"Starting prompt complexity analysis for {self.model_type}")
        self.logger.info(f"Layers: {len(layers)}")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(self.config['data']['output_dir']) / 'prompt_complexity'
        output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("\n" + "=" * 70)
        self.logger.info("ANALYZING PROMPT COMPLEXITY (0-5)")
        self.logger.info("=" * 70)

        all_results = []
        for layer in tqdm(layers, desc="Complexity Analysis"):
            layer_results = self.analyze_complexity_layer(layer)
            all_results.extend(layer_results)

        if not all_results:
            self.logger.warning("No results generated")
            return

        self.logger.info(f"Total features analyzed: {len(all_results)}")

        # Apply FDR correction to interaction p-values
        p_values = np.array([r['interaction_p'] for r in all_results])
        fdr_rejected, fdr_pvals = self.stats.apply_fdr_correction(p_values, fdr_alpha)

        for i, r in enumerate(all_results):
            r['interaction_p_fdr'] = float(fdr_pvals[i])
            r['interaction_fdr_significant'] = bool(fdr_rejected[i])

        # Filter significant interactions
        sig_results = [
            r for r in all_results
            if r['interaction_fdr_significant'] and r['interaction_eta'] >= min_eta
        ]

        # Sort by interaction effect size
        sig_results.sort(key=lambda x: x['interaction_eta'], reverse=True)

        # Summary statistics
        summary = {
            'total_features_analyzed': len(all_results),
            'fdr_significant_count': int(fdr_rejected.sum()),
            'significant_with_min_eta': len(sig_results),
            'max_interaction_eta': sig_results[0]['interaction_eta'] if sig_results else 0.0,
            'timestamp': timestamp,
            'model_type': self.model_type,
        }

        # Log summary
        self.logger.info(f"\nSummary:")
        self.logger.info(f"  Total features: {summary['total_features_analyzed']}")
        self.logger.info(f"  FDR significant: {summary['fdr_significant_count']}")
        self.logger.info(f"  Significant (eta >= {min_eta}): {summary['significant_with_min_eta']}")

        if sig_results:
            self.logger.info(f"\nTop 5 features:")
            for i, feat in enumerate(sig_results[:5], 1):
                self.logger.info(
                    f"  {i}. L{feat['layer']}-{feat['feature_id']}: "
                    f"int_eta={feat['interaction_eta']:.4f}, "
                    f"p_fdr={feat['interaction_p_fdr']:.2e}"
                )
                # Show complexity trend
                comp_means = feat['complexity_means']
                trend = " -> ".join([f"{comp_means[f'level_{i}']:.4f}" for i in range(6)])
                self.logger.info(f"     Complexity trend (0->5): {trend}")

        # Save results
        result_data = {
            'summary': summary,
            'top_features': sig_results[:n_top],
            'all_results': all_results,
        }

        output_file = output_dir / f'complexity_{self.model_type}_{timestamp}.json'
        save_results(result_data, output_file)
        self.logger.info(f"\nResults saved to: {output_file}")

        self.logger.info("\n" + "=" * 70)
        self.logger.info("ANALYSIS COMPLETE")
        self.logger.info("=" * 70)

        return result_data


def main():
    parser = argparse.ArgumentParser(
        description='Prompt Complexity Analysis for SAE Features'
    )
    parser.add_argument(
        '--model', type=str, required=True, choices=['llama', 'gemma'],
        help='Model type to analyze'
    )
    parser.add_argument(
        '--config', type=str,
        default=str(Path(__file__).parent.parent / 'configs' / 'prompt_analysis_config.yaml'),
        help='Path to config file'
    )

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Run analysis
    analyzer = PromptComplexityAnalyzer(config, args.model)
    analyzer.run_analysis()


if __name__ == '__main__':
    main()
