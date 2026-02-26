#!/usr/bin/env python3
"""
Prompt Component Analysis for SAE Features

Analyzes how individual prompt components (G/M/R/W/P) affect SAE feature activation
in bankruptcy vs safe outcomes.

For each component:
  - Component 포함: 16 combos × 100 games = 1,600 games
  - Component 미포함: 16 combos × 100 games = 1,600 games
  - Two-way ANOVA: Component × Outcome

Usage:
    python prompt_component_analysis.py --model llama
    python prompt_component_analysis.py --model gemma --components G M
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


class PromptComponentAnalyzer:
    """Analyze SAE feature differences by prompt components"""

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
        log_file = log_dir / f'prompt_component_{self.model_type}_{timestamp}.log'

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
        has_component: np.ndarray,
        outcomes: np.ndarray
    ) -> np.ndarray:
        """
        Filter out extremely sparse features.

        Args:
            features: (n_games, n_features) array
            has_component: (n_games,) boolean array
            outcomes: (n_games,) string array

        Returns:
            Boolean mask of shape (n_features,) indicating valid features
        """
        cfg = self.config['prompt_component_analysis']['sparse_filter']

        if not cfg['enabled']:
            return np.ones(features.shape[1], dtype=bool)

        n_features = features.shape[1]
        valid_mask = np.ones(n_features, dtype=bool)

        min_rate = cfg['min_activation_rate']
        min_mean = cfg['min_mean_activation']

        for feat_id in range(n_features):
            feat_vals = features[:, feat_id]

            # Check activation rate
            activation_rate = np.count_nonzero(feat_vals) / len(feat_vals)
            mean_activation = np.mean(feat_vals)

            if activation_rate < min_rate or mean_activation < min_mean:
                valid_mask[feat_id] = False

        return valid_mask

    def analyze_component_layer(self, component: str, layer: int) -> List[dict]:
        """
        Analyze Component × Outcome interaction for a single layer.

        Args:
            component: 'G', 'M', 'R', 'W', or 'P'
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
        game_ids = np.arange(n_games)  # Sequential 0-3199

        # Load prompt metadata
        prompt_meta = load_prompt_metadata(
            str(self.config['data'][self.model_type]['experiment_file']),
            game_ids
        )

        has_component = prompt_meta[f'has_{component}']

        # Filter sparse features
        valid_mask = self._filter_sparse_features(features, has_component, outcomes)
        n_valid = valid_mask.sum()
        n_total = len(valid_mask)

        if n_valid == 0:
            self.logger.warning(f"Layer {layer}: No valid features after sparse filtering")
            return []

        self.logger.info(
            f"Layer {layer}, Component {component}: "
            f"{n_valid}/{n_total} features after sparse filtering "
            f"({100 * n_valid / n_total:.1f}%)"
        )

        # Two-way ANOVA: Component × Outcome
        results = []

        for feature_id in np.where(valid_mask)[0]:
            feat_vals = features[:, feature_id]

            # Skip if no variance in any group
            try:
                # 2-way ANOVA
                main_component, main_outcome, interaction, group_means = \
                    self.stats.two_way_anova_simple(
                        feat_vals, has_component, outcomes
                    )

                results.append({
                    'layer': layer,
                    'feature_id': int(feature_id),
                    'component': component,
                    # Main effect: Component
                    'component_f': main_component['f'],
                    'component_p': main_component['p'],
                    'component_eta': main_component['eta_squared'],
                    # Main effect: Outcome
                    'outcome_f': main_outcome['f'],
                    'outcome_p': main_outcome['p'],
                    'outcome_eta': main_outcome['eta_squared'],
                    # Interaction: Component × Outcome
                    'interaction_f': interaction['f'],
                    'interaction_p': interaction['p'],
                    'interaction_eta': interaction['eta_squared'],
                    # Group means
                    'group_means': group_means,
                    # Sample sizes
                    'n_with_component': int(has_component.sum()),
                    'n_without_component': int((~has_component).sum()),
                })

            except Exception as e:
                self.logger.warning(f"Layer {layer}, Feature {feature_id}: {e}")
                continue

        return results

    def run_analysis(self, components: Optional[List[str]] = None):
        """
        Run full component analysis for specified components.

        Args:
            components: List of components to analyze (default: all)
        """
        if components is None:
            components = self.config['prompt_component_analysis']['components']

        layers = self.config['models'][self.model_type]['layers']
        fdr_alpha = self.config['prompt_component_analysis']['fdr_alpha']
        min_eta = self.config['prompt_component_analysis']['min_eta_squared']
        n_top = self.config['prompt_component_analysis']['n_top_features']

        self.logger.info(f"Starting prompt component analysis for {self.model_type}")
        self.logger.info(f"Components: {components}")
        self.logger.info(f"Layers: {len(layers)}")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(self.config['data']['output_dir']) / 'prompt_component'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Analyze each component
        for component in components:
            comp_name = self.config['prompt_component_analysis']['component_names'][component]
            self.logger.info("\n" + "=" * 70)
            self.logger.info(f"ANALYZING COMPONENT: {component} ({comp_name})")
            self.logger.info("=" * 70)

            all_results = []
            for layer in tqdm(layers, desc=f"Component {component}"):
                layer_results = self.analyze_component_layer(component, layer)
                all_results.extend(layer_results)

            if not all_results:
                self.logger.warning(f"Component {component}: No results")
                continue

            self.logger.info(f"Component {component}: {len(all_results)} features analyzed")

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
                'component': component,
                'component_name': comp_name,
                'total_features_analyzed': len(all_results),
                'fdr_significant_count': int(fdr_rejected.sum()),
                'significant_with_min_eta': len(sig_results),
                'max_interaction_eta': sig_results[0]['interaction_eta'] if sig_results else 0.0,
                'timestamp': timestamp,
                'model_type': self.model_type,
            }

            # Log summary
            self.logger.info(f"\nSummary for Component {component}:")
            self.logger.info(f"  Total features: {summary['total_features_analyzed']}")
            self.logger.info(f"  FDR significant: {summary['fdr_significant_count']}")
            self.logger.info(f"  Significant (eta >= {min_eta}): {summary['significant_with_min_eta']}")

            if sig_results:
                self.logger.info(f"\nTop 5 features for Component {component}:")
                for i, feat in enumerate(sig_results[:5], 1):
                    self.logger.info(
                        f"  {i}. L{feat['layer']}-{feat['feature_id']}: "
                        f"int_eta={feat['interaction_eta']:.4f}, "
                        f"p_fdr={feat['interaction_p_fdr']:.2e}"
                    )
                    self.logger.info(f"     Group means: {feat['group_means']}")

            # Save results
            result_data = {
                'summary': summary,
                'top_features': sig_results[:n_top],
            }

            # Optionally save all results
            if self.config['prompt_component_analysis']['save_all_results']:
                result_data['all_results'] = all_results

            output_file = output_dir / f'{component}_{self.model_type}_{timestamp}.json'
            save_results(result_data, output_file)
            self.logger.info(f"Results saved to: {output_file}")

        self.logger.info("\n" + "=" * 70)
        self.logger.info("ANALYSIS COMPLETE")
        self.logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Prompt Component Analysis for SAE Features'
    )
    parser.add_argument(
        '--model', type=str, required=True, choices=['llama', 'gemma'],
        help='Model type to analyze'
    )
    parser.add_argument(
        '--components', type=str, nargs='+', default=None,
        choices=['G', 'M', 'R', 'W', 'P'],
        help='Components to analyze (default: all)'
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
    analyzer = PromptComponentAnalyzer(config, args.model)
    analyzer.run_analysis(components=args.components)


if __name__ == '__main__':
    main()
