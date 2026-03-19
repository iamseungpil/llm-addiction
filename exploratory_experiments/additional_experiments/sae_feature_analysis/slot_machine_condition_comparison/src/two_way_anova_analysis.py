#!/usr/bin/env python3
"""
Proper Two-Way ANOVA Analysis for SAE Condition Comparison

Uses statsmodels for exact factorial ANOVA with main effects and interaction.
Compares SAE features across:
- Factor 1 (Bet Type): Variable vs Fixed
- Factor 2 (Outcome): Bankrupt vs Safe
- Interaction: Bet Type × Outcome

Usage:
    python src/two_way_anova_analysis.py --model llama
    python src/two_way_anova_analysis.py --model gemma
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import argparse
import logging
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# statsmodels for proper ANOVA
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import DataLoader


class TwoWayANOVAAnalyzer:
    """Proper Two-Way ANOVA analysis using statsmodels"""

    def __init__(self, config: dict, model_type: str):
        self.config = config
        self.model_type = model_type
        self.data_loader = DataLoader(config, model_type)
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging"""
        log_dir = Path(self.config['data']['logs_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'two_way_anova_{self.model_type}_{timestamp}.log'

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def run_two_way_anova_feature(
        self,
        feature_activations: np.ndarray,
        bet_type: np.ndarray,
        outcome: np.ndarray
    ) -> Dict:
        """
        Run Two-Way ANOVA for a single feature.

        Args:
            feature_activations: Feature activation values (n_samples,)
            bet_type: Bet type labels (0=Variable, 1=Fixed)
            outcome: Outcome labels (0=Bankrupt, 1=Safe)

        Returns:
            Dictionary with ANOVA results
        """
        # Create dataframe
        df = pd.DataFrame({
            'activation': feature_activations,
            'bet_type': bet_type,
            'outcome': outcome
        })

        # Convert to categorical
        df['bet_type'] = pd.Categorical(df['bet_type'])
        df['outcome'] = pd.Categorical(df['outcome'])

        try:
            # Fit model with interaction
            model = ols('activation ~ C(bet_type) + C(outcome) + C(bet_type):C(outcome)', data=df).fit()

            # ANOVA table (Type II)
            anova_table = anova_lm(model, typ=2)

            # Extract results
            bet_type_row = anova_table.loc['C(bet_type)', :]
            outcome_row = anova_table.loc['C(outcome)', :]
            interaction_row = anova_table.loc['C(bet_type):C(outcome)', :]
            residual_row = anova_table.loc['Residual', :]

            # Calculate eta-squared for each effect
            ss_total = anova_table['sum_sq'].sum()

            eta_sq_bet = bet_type_row['sum_sq'] / ss_total if ss_total > 0 else 0.0
            eta_sq_outcome = outcome_row['sum_sq'] / ss_total if ss_total > 0 else 0.0
            eta_sq_interaction = interaction_row['sum_sq'] / ss_total if ss_total > 0 else 0.0
            eta_sq_residual = residual_row['sum_sq'] / ss_total if ss_total > 0 else 0.0

            # Group means
            group_means = {}
            for bt in [0, 1]:
                for oc in [0, 1]:
                    mask = (df['bet_type'] == bt) & (df['outcome'] == oc)
                    bt_label = 'variable' if bt == 0 else 'fixed'
                    oc_label = 'bankrupt' if oc == 0 else 'safe'
                    key = f'{bt_label}_{oc_label}'
                    group_means[key] = float(df.loc[mask, 'activation'].mean())

            # Marginal means
            marginal_variable = float(df[df['bet_type'] == 0]['activation'].mean())
            marginal_fixed = float(df[df['bet_type'] == 1]['activation'].mean())
            marginal_bankrupt = float(df[df['outcome'] == 0]['activation'].mean())
            marginal_safe = float(df[df['outcome'] == 1]['activation'].mean())

            return {
                'success': True,
                'bet_type_effect': {
                    'f_stat': float(bet_type_row['F']),
                    'p_value': float(bet_type_row['PR(>F)']),
                    'df': int(bet_type_row['df']),
                    'sum_sq': float(bet_type_row['sum_sq']),
                    'eta_squared': float(eta_sq_bet)
                },
                'outcome_effect': {
                    'f_stat': float(outcome_row['F']),
                    'p_value': float(outcome_row['PR(>F)']),
                    'df': int(outcome_row['df']),
                    'sum_sq': float(outcome_row['sum_sq']),
                    'eta_squared': float(eta_sq_outcome)
                },
                'interaction_effect': {
                    'f_stat': float(interaction_row['F']),
                    'p_value': float(interaction_row['PR(>F)']),
                    'df': int(interaction_row['df']),
                    'sum_sq': float(interaction_row['sum_sq']),
                    'eta_squared': float(eta_sq_interaction)
                },
                'residual': {
                    'df': int(residual_row['df']),
                    'sum_sq': float(residual_row['sum_sq']),
                    'eta_squared': float(eta_sq_residual)
                },
                'total_sum_sq': float(ss_total),
                'group_means': group_means,
                'marginal_means': {
                    'variable': marginal_variable,
                    'fixed': marginal_fixed,
                    'bankrupt': marginal_bankrupt,
                    'safe': marginal_safe
                },
                'n_samples': len(df)
            }

        except Exception as e:
            self.logger.warning(f"ANOVA failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def analyze_layer(self, layer: int) -> List[Dict]:
        """
        Analyze all features in a layer with Two-Way ANOVA.

        Args:
            layer: Layer number

        Returns:
            List of feature results
        """
        self.logger.info(f"Analyzing Layer {layer}...")

        # Load grouped data
        grouped = self.data_loader.load_layer_features_grouped(layer)
        if grouped is None:
            self.logger.warning(f"Layer {layer}: No data loaded")
            return []

        # Prepare data arrays
        # Variable (bet_type=0), Fixed (bet_type=1)
        # Bankrupt (outcome=0), Safe (outcome=1)

        vb_features = grouped['variable_bankrupt']  # bet_type=0, outcome=0
        vs_features = grouped['variable_safe']       # bet_type=0, outcome=1
        fb_features = grouped['fixed_bankrupt']      # bet_type=1, outcome=0
        fs_features = grouped['fixed_safe']          # bet_type=1, outcome=1

        n_samples_vb = len(vb_features)
        n_samples_vs = len(vs_features)
        n_samples_fb = len(fb_features)
        n_samples_fs = len(fs_features)

        n_features = vb_features.shape[1]

        self.logger.info(f"  Sample sizes: VB={n_samples_vb}, VS={n_samples_vs}, FB={n_samples_fb}, FS={n_samples_fs}")
        self.logger.info(f"  Number of features: {n_features}")

        # Create combined arrays
        all_features = np.vstack([vb_features, vs_features, fb_features, fs_features])
        bet_type_labels = np.concatenate([
            np.zeros(n_samples_vb + n_samples_vs),  # Variable
            np.ones(n_samples_fb + n_samples_fs)     # Fixed
        ])
        outcome_labels = np.concatenate([
            np.zeros(n_samples_vb),  # Bankrupt
            np.ones(n_samples_vs),   # Safe
            np.zeros(n_samples_fb),  # Bankrupt
            np.ones(n_samples_fs)    # Safe
        ])

        results = []

        for feature_id in tqdm(range(n_features), desc=f"  Layer {layer}"):
            feature_activations = all_features[:, feature_id]

            # Skip if no variance
            if np.std(feature_activations) == 0:
                continue

            # Run Two-Way ANOVA
            anova_result = self.run_two_way_anova_feature(
                feature_activations,
                bet_type_labels,
                outcome_labels
            )

            if anova_result['success']:
                results.append({
                    'layer': layer,
                    'feature_id': feature_id,
                    **anova_result
                })

        self.logger.info(f"  Completed: {len(results)} features analyzed")
        return results

    def run_full_analysis(self, layers: List[int] = None):
        """
        Run Two-Way ANOVA for all specified layers.

        Args:
            layers: List of layer numbers (default: all available)
        """
        self.logger.info("=" * 70)
        self.logger.info("TWO-WAY ANOVA ANALYSIS")
        self.logger.info("=" * 70)
        self.logger.info(f"Model: {self.model_type.upper()}")
        self.logger.info(f"Design: 2×2 Factorial (Bet Type × Outcome)")
        self.logger.info("=" * 70)

        # Determine layers
        if layers is None:
            if self.model_type == 'llama':
                layers = list(range(25, 32))  # LLaMA: L25-L31
            elif self.model_type == 'gemma':
                layers = list(range(0, 42))   # Gemma: L0-L41
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

        self.logger.info(f"Layers to analyze: {layers}")

        all_results = []

        for layer in layers:
            layer_results = self.analyze_layer(layer)
            all_results.extend(layer_results)

        # Save results
        output_dir = Path(self.config['data']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f'two_way_anova_{self.model_type}_{timestamp}.json'

        # Compute summary statistics
        if all_results:
            # Separate by dominant effect
            bet_dominant = [r for r in all_results if r['bet_type_effect']['eta_squared'] > r['outcome_effect']['eta_squared']]
            outcome_dominant = [r for r in all_results if r['outcome_effect']['eta_squared'] > r['bet_type_effect']['eta_squared']]
            interaction_significant = [r for r in all_results if r['interaction_effect']['p_value'] < 0.05]

            summary = {
                'total_features': len(all_results),
                'bet_type_dominant': len(bet_dominant),
                'outcome_dominant': len(outcome_dominant),
                'interaction_significant': len(interaction_significant),
                'mean_eta_bet_type': float(np.mean([r['bet_type_effect']['eta_squared'] for r in all_results])),
                'mean_eta_outcome': float(np.mean([r['outcome_effect']['eta_squared'] for r in all_results])),
                'mean_eta_interaction': float(np.mean([r['interaction_effect']['eta_squared'] for r in all_results]))
            }
        else:
            summary = {}

        output_data = {
            'model_type': self.model_type,
            'timestamp': timestamp,
            'layers_analyzed': layers,
            'summary': summary,
            'all_results': all_results
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        self.logger.info("=" * 70)
        self.logger.info("ANALYSIS COMPLETE")
        self.logger.info("=" * 70)
        self.logger.info(f"Total features analyzed: {len(all_results)}")
        if summary:
            self.logger.info(f"Bet Type dominant: {summary['bet_type_dominant']} ({summary['bet_type_dominant']/len(all_results)*100:.1f}%)")
            self.logger.info(f"Outcome dominant: {summary['outcome_dominant']} ({summary['outcome_dominant']/len(all_results)*100:.1f}%)")
            self.logger.info(f"Significant interaction: {summary['interaction_significant']} ({summary['interaction_significant']/len(all_results)*100:.1f}%)")
            self.logger.info(f"Mean η² (Bet Type): {summary['mean_eta_bet_type']:.4f}")
            self.logger.info(f"Mean η² (Outcome): {summary['mean_eta_outcome']:.4f}")
            self.logger.info(f"Mean η² (Interaction): {summary['mean_eta_interaction']:.4f}")
        self.logger.info(f"Results saved: {output_file}")
        self.logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Two-Way ANOVA Analysis for SAE Condition Comparison")
    parser.add_argument('--model', type=str, required=True, choices=['llama', 'gemma'],
                        help='Model type')
    parser.add_argument('--config', type=str,
                        default='configs/analysis_config.yaml',
                        help='Config file path')
    parser.add_argument('--layers', type=int, nargs='+', default=None,
                        help='Specific layers to analyze (default: all)')

    args = parser.parse_args()

    # Load config
    import yaml
    config_path = Path(__file__).parent.parent / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Run analysis
    analyzer = TwoWayANOVAAnalyzer(config, args.model)
    analyzer.run_full_analysis(layers=args.layers)


if __name__ == '__main__':
    main()
