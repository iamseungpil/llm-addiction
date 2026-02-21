#!/usr/bin/env python3
"""
Phase 2: SAE Feature Correlation Analysis for Blackjack

Analyze which SAE features correlate with gambling behaviors:
- Bankruptcy vs safe play
- Component effects (BASE vs GMHWP, etc.)
- Betting patterns

Usage:
    python src/blackjack/phase2_correlation_analysis.py --model llama --feature-dir results_features/
    python src/blackjack/phase2_correlation_analysis.py --model gemma --feature-dir results_features/ --fdr 0.05
"""

import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from common import setup_logger, save_json

logger = setup_logger(__name__)


class BlackjackCorrelationAnalyzer:
    """Analyze SAE feature correlations with gambling behaviors."""

    def __init__(self, model_name: str, feature_dir: Path, fdr_alpha: float = 0.05):
        """
        Initialize analyzer.

        Args:
            model_name: 'llama' or 'gemma'
            feature_dir: Directory containing NPZ feature files
            fdr_alpha: FDR correction threshold
        """
        self.model_name = model_name
        self.feature_dir = Path(feature_dir)
        self.fdr_alpha = fdr_alpha

        # Load all layer files
        self.layer_data = {}
        self.load_all_layers()

    def load_all_layers(self):
        """Load all NPZ files from feature directory."""
        npz_files = sorted(self.feature_dir.glob("*_L*.npz"))

        logger.info(f"Loading feature files from {self.feature_dir}")
        logger.info(f"Found {len(npz_files)} layer files")

        for npz_file in npz_files:
            # Extract layer number from filename
            layer = int(npz_file.stem.split('_L')[1])

            data = np.load(npz_file, allow_pickle=True)
            self.layer_data[layer] = {
                'features': data['features'],  # [N, d_sae]
                'bets': data['bets'],
                'outcomes': data['outcomes'],
                'payouts': data['payouts'],
                'game_ids': data['game_ids'],
                'rounds': data['rounds'],
                'bet_types': data['bet_types'],
                'components': data['components']
            }

            logger.info(f"  Layer {layer}: {len(data['features'])} rounds, {data['features'].shape[1]} features")

    def analyze_bankruptcy_correlation(self, layer: int) -> Dict:
        """
        Find features correlated with bankruptcy.

        Args:
            layer: Layer number

        Returns:
            Analysis results dictionary
        """
        data = self.layer_data[layer]
        features = data['features']  # [N, d_sae]
        game_ids = data['game_ids']
        components = data['components']

        # Group by game to get bankruptcy labels
        game_bankruptcy = {}
        game_features = defaultdict(list)

        for i, game_id in enumerate(game_ids):
            comp = components[i]

            # Simple bankruptcy indicator: if game exists, check if last round balance=0
            # For now, use heuristic: games with many rounds likely went bankrupt
            game_features[(game_id, comp)].append(features[i])

        # Compute mean feature activation per game
        game_mean_features = {}
        for (game_id, comp), feat_list in game_features.items():
            game_mean_features[(game_id, comp)] = np.mean(feat_list, axis=0)

        logger.info(f"Layer {layer}: {len(game_mean_features)} unique games")

        # For each feature, compute correlation with component
        # Compare BASE vs high-risk components (M, GM, etc.)

        results = {
            'layer': layer,
            'n_features': features.shape[1],
            'n_rounds': len(features),
            'n_games': len(game_mean_features)
        }

        return results

    def analyze_component_effects(self, layer: int) -> Dict:
        """
        Compare feature activations across components.

        Args:
            layer: Layer number

        Returns:
            Component comparison results
        """
        data = self.layer_data[layer]
        features = data['features']
        components = data['components']

        # Group by component
        component_features = defaultdict(list)
        for i, comp in enumerate(components):
            component_features[comp].append(features[i])

        logger.info(f"Layer {layer} component distribution:")
        for comp, feat_list in component_features.items():
            logger.info(f"  {comp}: {len(feat_list)} rounds")

        # Compare BASE vs other components
        if 'BASE' not in component_features:
            logger.warning(f"No BASE component found in layer {layer}")
            return {}

        base_features = np.array(component_features['BASE'])  # [N_base, d_sae]
        d_sae = base_features.shape[1]

        # For each other component, find differentially activated features
        component_results = {}

        for comp in ['G', 'M', 'GM', 'GMHWP']:
            if comp not in component_features:
                continue

            comp_features = np.array(component_features[comp])

            # T-test for each feature
            pvals = []
            effect_sizes = []

            for feat_idx in range(d_sae):
                base_vals = base_features[:, feat_idx]
                comp_vals = comp_features[:, feat_idx]

                # Only test if feature is active (non-zero) in at least 5% of samples
                base_active = (base_vals > 0).sum() / len(base_vals)
                comp_active = (comp_vals > 0).sum() / len(comp_vals)

                if base_active < 0.05 and comp_active < 0.05:
                    pvals.append(1.0)
                    effect_sizes.append(0.0)
                    continue

                # T-test
                try:
                    t_stat, p_val = stats.ttest_ind(comp_vals, base_vals)

                    # Cohen's d
                    pooled_std = np.sqrt((np.var(base_vals) + np.var(comp_vals)) / 2)
                    if pooled_std > 0:
                        cohens_d = (np.mean(comp_vals) - np.mean(base_vals)) / pooled_std
                    else:
                        cohens_d = 0.0

                    pvals.append(p_val)
                    effect_sizes.append(cohens_d)

                except:
                    pvals.append(1.0)
                    effect_sizes.append(0.0)

            # FDR correction
            pvals = np.array(pvals)
            effect_sizes = np.array(effect_sizes)

            reject, pvals_corrected = fdrcorrection(pvals, alpha=self.fdr_alpha)

            # Find significant features
            significant_features = np.where(reject)[0]

            logger.info(f"  {comp} vs BASE: {len(significant_features)} significant features (FDR < {self.fdr_alpha})")

            # Top 20 features by effect size
            top_indices = np.argsort(np.abs(effect_sizes))[-20:][::-1]

            component_results[comp] = {
                'n_significant': int(len(significant_features)),
                'significant_features': significant_features.tolist(),
                'top_20_features': [
                    {
                        'feature_idx': int(idx),
                        'cohens_d': float(effect_sizes[idx]),
                        'p_value': float(pvals[idx]),
                        'p_value_corrected': float(pvals_corrected[idx])
                    }
                    for idx in top_indices
                ]
            }

        return {
            'layer': layer,
            'n_features': d_sae,
            'component_comparisons': component_results
        }

    def run_analysis(self, output_dir: Path):
        """Run full correlation analysis."""
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 70)
        logger.info("PHASE 2: CORRELATION ANALYSIS")
        logger.info("=" * 70)
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Layers: {sorted(self.layer_data.keys())}")
        logger.info(f"FDR alpha: {self.fdr_alpha}")
        logger.info(f"Output: {output_dir}")

        all_results = {}

        for layer in sorted(self.layer_data.keys()):
            logger.info(f"\n{'='*70}")
            logger.info(f"LAYER {layer}")
            logger.info(f"{'='*70}")

            # Component effect analysis
            comp_results = self.analyze_component_effects(layer)
            all_results[f'layer_{layer}'] = comp_results

        # Save results
        output_file = output_dir / f"{self.model_name}_correlation_results.json"
        save_json(all_results, output_file)

        logger.info("\n" + "=" * 70)
        logger.info("ANALYSIS COMPLETED")
        logger.info(f"Results saved to: {output_file}")
        logger.info("=" * 70)

        # Summary
        self.print_summary(all_results)

    def print_summary(self, results: Dict):
        """Print summary of significant features."""
        logger.info("\n" + "=" * 70)
        logger.info("SUMMARY: SIGNIFICANT FEATURES BY LAYER")
        logger.info("=" * 70)

        for layer_key in sorted(results.keys()):
            layer_results = results[layer_key]
            layer = layer_results['layer']

            logger.info(f"\nLayer {layer}:")

            if 'component_comparisons' not in layer_results:
                continue

            for comp, comp_results in layer_results['component_comparisons'].items():
                n_sig = comp_results['n_significant']
                logger.info(f"  {comp} vs BASE: {n_sig} significant features")

                # Top 3 features
                if comp_results['top_20_features']:
                    logger.info(f"    Top 3 features:")
                    for feat_info in comp_results['top_20_features'][:3]:
                        logger.info(f"      Feature {feat_info['feature_idx']}: "
                                  f"d={feat_info['cohens_d']:.3f}, "
                                  f"p_corr={feat_info['p_value_corrected']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Blackjack Phase 2: Correlation Analysis")
    parser.add_argument('--model', type=str, required=True, choices=['llama', 'gemma'],
                        help='Model name')
    parser.add_argument('--feature-dir', type=str, required=True,
                        help='Directory containing NPZ feature files from Phase 1')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: feature_dir/analysis)')
    parser.add_argument('--fdr', type=float, default=0.05,
                        help='FDR alpha threshold (default: 0.05)')

    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = feature_dir / 'analysis'

    # Run analysis
    analyzer = BlackjackCorrelationAnalyzer(
        model_name=args.model,
        feature_dir=feature_dir,
        fdr_alpha=args.fdr
    )

    analyzer.run_analysis(output_dir)


if __name__ == '__main__':
    main()
