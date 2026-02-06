#!/usr/bin/env python3
"""
Individual Prompt Combo Explorer for SAE Features

Analyzes each of the 32 prompt combinations individually to find
unique patterns in bankruptcy vs safe outcomes.

WARNING: Small sample sizes per combo (50 games each)
Use as exploratory analysis only!

Usage:
    python prompt_combo_explorer.py --model llama
    python prompt_combo_explorer.py --model gemma --top-n 10
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
from collections import Counter

from .utils import (
    DataLoader, StatisticalAnalyzer,
    load_prompt_metadata, save_results
)


class PromptComboExplorer:
    """Explore SAE features for individual prompt combinations"""

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
        log_file = log_dir / f'prompt_combo_{self.model_type}_{timestamp}.log'

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def analyze_combo_layer(
        self,
        layer: int,
        combo: str,
        combo_mask: np.ndarray
    ) -> List[dict]:
        """
        Analyze bankruptcy vs safe for a specific combo and layer.

        Args:
            layer: Layer number
            combo: Prompt combo name (e.g., 'GM', 'GMRWP')
            combo_mask: Boolean mask for games with this combo

        Returns:
            List of feature results
        """
        result = self.data_loader.load_layer_features(layer)
        if result is None:
            return []

        features, outcomes, bet_types = result

        # Filter to this combo only
        combo_features = features[combo_mask]
        combo_outcomes = outcomes[combo_mask]

        # Check sample sizes
        n_total = len(combo_outcomes)
        n_bankrupt = (combo_outcomes == 'bankruptcy').sum()
        n_safe = (combo_outcomes == 'voluntary_stop').sum()

        if n_bankrupt < 5 or n_safe < 5:
            # Too few samples for reliable analysis
            return []

        n_features = combo_features.shape[1]
        results = []

        for feature_id in range(n_features):
            feat_vals = combo_features[:, feature_id]

            # Skip if no variance
            if np.std(feat_vals) == 0:
                continue

            # t-test: bankruptcy vs safe
            bankrupt_vals = feat_vals[combo_outcomes == 'bankruptcy']
            safe_vals = feat_vals[combo_outcomes == 'voluntary_stop']

            t_stat, p_value = self.stats.welch_ttest(bankrupt_vals, safe_vals)
            cohens_d = self.stats.compute_cohens_d(bankrupt_vals, safe_vals)

            results.append({
                'layer': layer,
                'feature_id': int(feature_id),
                'combo': combo,
                't_stat': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'bankrupt_mean': float(np.mean(bankrupt_vals)),
                'safe_mean': float(np.mean(safe_vals)),
                'bankrupt_std': float(np.std(bankrupt_vals)),
                'safe_std': float(np.std(safe_vals)),
                'n_bankrupt': int(n_bankrupt),
                'n_safe': int(n_safe),
                'direction': 'higher_in_bankrupt' if cohens_d > 0 else 'higher_in_safe'
            })

        return results

    def run_analysis(self, top_n: int = 10):
        """
        Run analysis for all 32 prompt combinations.

        Args:
            top_n: Number of top features to save per combo
        """
        layers = self.config['models'][self.model_type]['layers']
        fdr_alpha = 0.05
        min_d = 0.3  # Minimum Cohen's d

        self.logger.info(f"Starting individual combo exploration for {self.model_type}")
        self.logger.info(f"Layers: {len(layers)}")
        self.logger.info(f"WARNING: Small sample sizes per combo - exploratory only!")

        # Load game metadata
        experiment_file = self.config['data'][self.model_type]['experiment_file']
        game_ids = np.arange(3200)
        prompt_meta = load_prompt_metadata(str(experiment_file), game_ids)

        # Get unique combos and their counts
        combos = prompt_meta['prompt_combos']
        unique_combos = np.unique(combos)

        # Load outcomes to check bankruptcy rates
        result = self.data_loader.load_layer_features(layers[0])
        _, outcomes, _ = result

        combo_stats = {}
        for combo in unique_combos:
            combo_mask = combos == combo
            combo_outcomes = outcomes[combo_mask]
            n_total = len(combo_outcomes)
            n_bankrupt = (combo_outcomes == 'bankruptcy').sum()
            bankruptcy_rate = n_bankrupt / n_total if n_total > 0 else 0

            combo_stats[combo] = {
                'n_total': int(n_total),
                'n_bankrupt': int(n_bankrupt),
                'n_safe': int(n_total - n_bankrupt),
                'bankruptcy_rate': float(bankruptcy_rate)
            }

        self.logger.info(f"\nFound {len(unique_combos)} unique prompt combinations")
        self.logger.info(f"Sample size range: {min([s['n_total'] for s in combo_stats.values()])} - {max([s['n_total'] for s in combo_stats.values()])}")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(self.config['data']['output_dir']) / 'prompt_combo'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Analyze each combo
        all_combo_results = {}

        for combo in tqdm(unique_combos, desc="Analyzing combos"):
            combo_mask = combos == combo
            stats = combo_stats[combo]

            if stats['n_bankrupt'] < 5 or stats['n_safe'] < 5:
                self.logger.info(f"Skipping {combo}: insufficient samples (B={stats['n_bankrupt']}, S={stats['n_safe']})")
                continue

            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Combo: {combo} (n={stats['n_total']}, bankruptcy_rate={stats['bankruptcy_rate']:.1%})")
            self.logger.info(f"{'='*60}")

            combo_results = []
            for layer in layers:
                layer_results = self.analyze_combo_layer(layer, combo, combo_mask)
                combo_results.extend(layer_results)

            if not combo_results:
                continue

            # Apply FDR correction
            p_values = np.array([r['p_value'] for r in combo_results])
            fdr_rejected, fdr_pvals = self.stats.apply_fdr_correction(p_values, fdr_alpha)

            for i, r in enumerate(combo_results):
                r['p_fdr'] = float(fdr_pvals[i])
                r['fdr_significant'] = bool(fdr_rejected[i])

            # Filter significant features
            sig_results = [
                r for r in combo_results
                if r['fdr_significant'] and abs(r['cohens_d']) >= min_d
            ]

            # Sort by effect size
            sig_results.sort(key=lambda x: abs(x['cohens_d']), reverse=True)

            all_combo_results[combo] = {
                'stats': stats,
                'total_features_analyzed': len(combo_results),
                'fdr_significant_count': int(fdr_rejected.sum()),
                'significant_with_min_d': len(sig_results),
                'top_features': sig_results[:top_n],
            }

            self.logger.info(f"  Total features: {len(combo_results)}")
            self.logger.info(f"  FDR significant (|d|>={min_d}): {len(sig_results)}")

            if sig_results:
                self.logger.info(f"  Top feature: L{sig_results[0]['layer']}-{sig_results[0]['feature_id']} (d={sig_results[0]['cohens_d']:.3f})")

        # Save results
        result_data = {
            'timestamp': timestamp,
            'model_type': self.model_type,
            'summary': {
                'n_combos_analyzed': len(all_combo_results),
                'total_combos': len(unique_combos),
                'combos_with_significant_features': sum(1 for r in all_combo_results.values() if r['significant_with_min_d'] > 0)
            },
            'combo_results': all_combo_results,
            'combo_stats': combo_stats,
        }

        output_file = output_dir / f'combo_explorer_{self.model_type}_{timestamp}.json'
        save_results(result_data, output_file)
        self.logger.info(f"\n\nResults saved to: {output_file}")

        self.logger.info("\n" + "=" * 70)
        self.logger.info("EXPLORATION COMPLETE")
        self.logger.info("=" * 70)
        self.logger.info(f"Analyzed {len(all_combo_results)} combos (out of {len(unique_combos)} total)")
        self.logger.info(f"Combos with significant features: {result_data['summary']['combos_with_significant_features']}")

        return result_data


def main():
    parser = argparse.ArgumentParser(
        description='Individual Prompt Combo Explorer for SAE Features'
    )
    parser.add_argument(
        '--model', type=str, required=True, choices=['llama', 'gemma'],
        help='Model type to analyze'
    )
    parser.add_argument(
        '--top-n', type=int, default=10,
        help='Number of top features to save per combo (default: 10)'
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
    explorer = PromptComboExplorer(config, args.model)
    explorer.run_analysis(top_n=args.top_n)


if __name__ == '__main__':
    main()
