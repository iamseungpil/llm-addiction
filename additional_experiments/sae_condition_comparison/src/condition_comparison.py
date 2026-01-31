#!/usr/bin/env python3
"""
SAE Condition Comparison Analysis
Compares SAE features between variable/fixed betting conditions.

Analyses:
1. Variable vs Fixed (main effect)
2. 4-way comparison (variable-bankrupt, variable-safe, fixed-bankrupt, fixed-safe)
3. Interaction analysis (bet_type x outcome)
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
from typing import Dict, List, Optional, Tuple

from .utils import (
    DataLoader, StatisticalAnalyzer,
    FeatureResult, FourWayResult, InteractionResult,
    save_results
)


class ConditionComparisonAnalyzer:
    """Analyze SAE feature differences between variable/fixed betting conditions"""

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
        log_file = log_dir / f'condition_comparison_{self.model_type}_{timestamp}.log'

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def analyze_variable_vs_fixed_layer(self, layer: int) -> List[dict]:
        """
        Analyze variable vs fixed comparison for a single layer.
        Returns list of feature results with t-test and Cohen's d.
        """
        grouped = self.data_loader.load_layer_features_grouped(layer)
        if grouped is None:
            return []

        variable_features = grouped['variable']
        fixed_features = grouped['fixed']
        n_features = variable_features.shape[1]

        results = []

        for feature_id in range(n_features):
            v_vals = variable_features[:, feature_id]
            f_vals = fixed_features[:, feature_id]

            # Skip if no variance
            if np.std(v_vals) == 0 and np.std(f_vals) == 0:
                continue

            # t-test
            t_stat, p_value = self.stats.welch_ttest(v_vals, f_vals)

            # Cohen's d (positive = higher in variable)
            cohens_d = self.stats.compute_cohens_d(v_vals, f_vals)

            # Direction
            direction = 'higher_in_variable' if cohens_d > 0 else 'higher_in_fixed'

            results.append({
                'layer': layer,
                'feature_id': feature_id,
                't_stat': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'variable_mean': float(np.mean(v_vals)),
                'fixed_mean': float(np.mean(f_vals)),
                'variable_std': float(np.std(v_vals)),
                'fixed_std': float(np.std(f_vals)),
                'variable_n': len(v_vals),
                'fixed_n': len(f_vals),
                'direction': direction
            })

        return results

    def analyze_four_way_layer(self, layer: int) -> List[dict]:
        """
        Analyze 4-way comparison for a single layer.
        Groups: variable-bankrupt, variable-safe, fixed-bankrupt, fixed-safe
        """
        grouped = self.data_loader.load_layer_features_grouped(layer)
        if grouped is None:
            return []

        groups = {
            'variable_bankrupt': grouped['variable_bankrupt'],
            'variable_safe': grouped['variable_safe'],
            'fixed_bankrupt': grouped['fixed_bankrupt'],
            'fixed_safe': grouped['fixed_safe']
        }

        # Check minimum sample sizes
        min_samples = 5
        for name, arr in groups.items():
            if len(arr) < min_samples:
                self.logger.warning(f"Layer {layer}: Group {name} has only {len(arr)} samples")

        n_features = grouped['variable'].shape[1]
        results = []

        for feature_id in range(n_features):
            group_vals = {name: arr[:, feature_id] for name, arr in groups.items()}

            # One-way ANOVA
            group_list = [v for v in group_vals.values() if len(v) > 0]
            f_stat, p_value = self.stats.one_way_anova(group_list)

            # Eta-squared
            eta_sq = self.stats.compute_eta_squared(group_list) if len(group_list) >= 2 else 0.0

            results.append({
                'layer': layer,
                'feature_id': feature_id,
                'f_stat': f_stat,
                'p_value': p_value,
                'eta_squared': eta_sq,
                'group_means': {k: float(np.mean(v)) if len(v) > 0 else 0.0 for k, v in group_vals.items()},
                'group_stds': {k: float(np.std(v)) if len(v) > 0 else 0.0 for k, v in group_vals.items()},
                'group_ns': {k: len(v) for k, v in group_vals.items()}
            })

        return results

    def analyze_interaction_layer(self, layer: int) -> List[dict]:
        """
        Analyze bet_type x outcome interaction for a single layer.
        """
        result = self.data_loader.load_layer_features(layer)
        if result is None:
            return []

        features, outcomes, bet_types = result
        n_features = features.shape[1]

        results = []

        for feature_id in range(n_features):
            feat_vals = features[:, feature_id]

            # 2-way ANOVA
            main_bet, main_outcome, interaction, group_means = self.stats.two_way_anova_simple(
                feat_vals, bet_types, outcomes
            )

            results.append({
                'layer': layer,
                'feature_id': feature_id,
                'bet_type_f': main_bet['f'],
                'bet_type_p': main_bet['p'],
                'bet_type_eta': main_bet['eta_squared'],
                'outcome_f': main_outcome['f'],
                'outcome_p': main_outcome['p'],
                'outcome_eta': main_outcome['eta_squared'],
                'interaction_f': interaction['f'],
                'interaction_p': interaction['p'],
                'interaction_eta': interaction['eta_squared'],
                'group_means': group_means
            })

        return results

    def run_analysis(self):
        """Run full condition comparison analysis"""
        output_dir = Path(self.config['data']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        layers = self.config['models'][self.model_type]['layers']
        analyses = self.config['condition_comparison']['analyses']
        fdr_alpha = self.config['condition_comparison']['fdr_alpha']
        min_d = self.config['condition_comparison']['min_cohens_d']
        min_eta = self.config['condition_comparison']['min_eta_squared']
        n_top = self.config['condition_comparison']['n_top_features']

        self.logger.info(f"Starting condition comparison analysis for {self.model_type}")
        self.logger.info(f"Layers: {len(layers)}")
        self.logger.info(f"Analyses: {analyses}")

        # Get data summary
        data_summary = self.data_loader.get_data_summary()
        self.logger.info(f"Data summary: {data_summary}")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results = {
            'model_type': self.model_type,
            'timestamp': timestamp,
            'data_summary': data_summary,
        }

        # === Analysis 1: Variable vs Fixed ===
        if 'variable_vs_fixed' in analyses:
            self.logger.info("\n" + "="*60)
            self.logger.info("ANALYSIS 1: Variable vs Fixed")
            self.logger.info("="*60)

            all_vf_results = []
            for layer in tqdm(layers, desc="Variable vs Fixed"):
                layer_results = self.analyze_variable_vs_fixed_layer(layer)
                all_vf_results.extend(layer_results)
                self.logger.info(f"Layer {layer}: {len(layer_results)} features analyzed")

            if all_vf_results:
                # Apply FDR correction
                p_values = np.array([r['p_value'] for r in all_vf_results])
                fdr_rejected, fdr_pvals = self.stats.apply_fdr_correction(p_values, fdr_alpha)

                for i, r in enumerate(all_vf_results):
                    r['p_fdr'] = float(fdr_pvals[i])
                    r['fdr_significant'] = bool(fdr_rejected[i])

                # Filter significant features
                sig_results = [r for r in all_vf_results
                              if r['fdr_significant'] and abs(r['cohens_d']) >= min_d]

                # Separate by direction
                higher_variable = [r for r in sig_results if r['cohens_d'] > 0]
                higher_fixed = [r for r in sig_results if r['cohens_d'] < 0]

                # Sort by effect size
                higher_variable.sort(key=lambda x: x['cohens_d'], reverse=True)
                higher_fixed.sort(key=lambda x: x['cohens_d'])

                vf_summary = {
                    'total_features_analyzed': len(all_vf_results),
                    'fdr_significant_count': len(sig_results),
                    'higher_in_variable_count': len(higher_variable),
                    'higher_in_fixed_count': len(higher_fixed),
                    'max_d_variable': higher_variable[0]['cohens_d'] if higher_variable else 0,
                    'max_d_fixed': higher_fixed[0]['cohens_d'] if higher_fixed else 0,
                }

                results['variable_vs_fixed'] = {
                    'summary': vf_summary,
                    'top_higher_in_variable': higher_variable[:n_top],
                    'top_higher_in_fixed': higher_fixed[:n_top],
                }

                self.logger.info(f"Total features: {len(all_vf_results)}")
                self.logger.info(f"FDR significant (|d|>={min_d}): {len(sig_results)}")
                self.logger.info(f"  Higher in variable: {len(higher_variable)}")
                self.logger.info(f"  Higher in fixed: {len(higher_fixed)}")

                if higher_variable:
                    self.logger.info(f"\nTop 5 features higher in VARIABLE:")
                    for f in higher_variable[:5]:
                        self.logger.info(f"  L{f['layer']}-{f['feature_id']}: d={f['cohens_d']:.3f}, p_fdr={f['p_fdr']:.2e}")

                if higher_fixed:
                    self.logger.info(f"\nTop 5 features higher in FIXED:")
                    for f in higher_fixed[:5]:
                        self.logger.info(f"  L{f['layer']}-{f['feature_id']}: d={f['cohens_d']:.3f}, p_fdr={f['p_fdr']:.2e}")

                # Save detailed results
                vf_file = output_dir / f'variable_vs_fixed_{self.model_type}_{timestamp}.json'
                save_results({
                    'summary': vf_summary,
                    'all_results': all_vf_results
                }, vf_file)

        # === Analysis 2: Four-Way Comparison ===
        if 'four_way' in analyses:
            self.logger.info("\n" + "="*60)
            self.logger.info("ANALYSIS 2: Four-Way Comparison")
            self.logger.info("="*60)

            all_fw_results = []
            for layer in tqdm(layers, desc="Four-Way"):
                layer_results = self.analyze_four_way_layer(layer)
                all_fw_results.extend(layer_results)

            if all_fw_results:
                # Apply FDR correction
                p_values = np.array([r['p_value'] for r in all_fw_results])
                fdr_rejected, fdr_pvals = self.stats.apply_fdr_correction(p_values, fdr_alpha)

                for i, r in enumerate(all_fw_results):
                    r['p_fdr'] = float(fdr_pvals[i])
                    r['fdr_significant'] = bool(fdr_rejected[i])

                # Filter significant features
                sig_results = [r for r in all_fw_results
                              if r['fdr_significant'] and r['eta_squared'] >= min_eta]

                # Sort by effect size
                sig_results.sort(key=lambda x: x['eta_squared'], reverse=True)

                fw_summary = {
                    'total_features_analyzed': len(all_fw_results),
                    'fdr_significant_count': sum(1 for r in all_fw_results if r['fdr_significant']),
                    'significant_with_min_eta': len(sig_results),
                    'max_eta_squared': sig_results[0]['eta_squared'] if sig_results else 0,
                }

                results['four_way'] = {
                    'summary': fw_summary,
                    'top_features': sig_results[:n_top],
                }

                self.logger.info(f"Total features: {len(all_fw_results)}")
                self.logger.info(f"FDR significant (eta>={min_eta}): {len(sig_results)}")

                if sig_results:
                    self.logger.info(f"\nTop 5 features by eta-squared:")
                    for f in sig_results[:5]:
                        self.logger.info(f"  L{f['layer']}-{f['feature_id']}: eta^2={f['eta_squared']:.3f}, p_fdr={f['p_fdr']:.2e}")
                        self.logger.info(f"    Means: {f['group_means']}")

                # Save detailed results
                fw_file = output_dir / f'four_way_{self.model_type}_{timestamp}.json'
                save_results({
                    'summary': fw_summary,
                    'all_results': all_fw_results
                }, fw_file)

        # === Analysis 3: Interaction ===
        if 'interaction' in analyses:
            self.logger.info("\n" + "="*60)
            self.logger.info("ANALYSIS 3: Interaction (bet_type x outcome)")
            self.logger.info("="*60)

            all_int_results = []
            for layer in tqdm(layers, desc="Interaction"):
                layer_results = self.analyze_interaction_layer(layer)
                all_int_results.extend(layer_results)

            if all_int_results:
                # Apply FDR correction to interaction p-values
                p_values = np.array([r['interaction_p'] for r in all_int_results])
                fdr_rejected, fdr_pvals = self.stats.apply_fdr_correction(p_values, fdr_alpha)

                for i, r in enumerate(all_int_results):
                    r['interaction_p_fdr'] = float(fdr_pvals[i])
                    r['interaction_fdr_significant'] = bool(fdr_rejected[i])

                # Filter significant interactions
                sig_interactions = [r for r in all_int_results
                                   if r['interaction_fdr_significant'] and r['interaction_eta'] >= min_eta]

                # Sort by interaction effect size
                sig_interactions.sort(key=lambda x: x['interaction_eta'], reverse=True)

                int_summary = {
                    'total_features_analyzed': len(all_int_results),
                    'significant_interactions': len(sig_interactions),
                    'max_interaction_eta': sig_interactions[0]['interaction_eta'] if sig_interactions else 0,
                }

                results['interaction'] = {
                    'summary': int_summary,
                    'top_interactions': sig_interactions[:n_top],
                }

                self.logger.info(f"Total features: {len(all_int_results)}")
                self.logger.info(f"Significant interactions (eta>={min_eta}): {len(sig_interactions)}")

                if sig_interactions:
                    self.logger.info(f"\nTop 5 interaction effects:")
                    for f in sig_interactions[:5]:
                        self.logger.info(f"  L{f['layer']}-{f['feature_id']}: int_eta^2={f['interaction_eta']:.3f}")
                        self.logger.info(f"    bet_type: F={f['bet_type_f']:.2f}, p={f['bet_type_p']:.2e}")
                        self.logger.info(f"    outcome: F={f['outcome_f']:.2f}, p={f['outcome_p']:.2e}")

                # Save detailed results
                int_file = output_dir / f'interaction_{self.model_type}_{timestamp}.json'
                save_results({
                    'summary': int_summary,
                    'all_results': all_int_results
                }, int_file)

        # === Save Summary ===
        summary_file = output_dir / f'condition_comparison_summary_{self.model_type}_{timestamp}.json'
        save_results(results, summary_file)

        self.logger.info("\n" + "="*60)
        self.logger.info("ANALYSIS COMPLETE")
        self.logger.info("="*60)
        self.logger.info(f"Results saved to: {output_dir}")

        return results


def main():
    parser = argparse.ArgumentParser(description='SAE Condition Comparison Analysis')
    parser.add_argument('--model', type=str, required=True, choices=['llama', 'gemma'],
                        help='Model type to analyze')
    parser.add_argument('--config', type=str,
                        default=str(Path(__file__).parent.parent / 'configs' / 'analysis_config.yaml'),
                        help='Path to config file')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Run analysis
    analyzer = ConditionComparisonAnalyzer(config, args.model)
    analyzer.run_analysis()


if __name__ == '__main__':
    main()
