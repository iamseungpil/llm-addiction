"""
Phase 2: Correlation Analysis

Identify SAE features that predict investment choice decisions (1/2/3/4).
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import yaml

import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.analysis_utils import (
    binary_ttest_analysis,
    multiclass_anova_analysis,
    get_top_features,
    compute_feature_means_by_group
)


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CorrelationAnalyzer:
    """Analyze correlations between SAE features and choice decisions."""

    def __init__(self, config: Dict[str, Any], model_name: str):
        """
        Initialize analyzer.

        Args:
            config: Experiment configuration
            model_name: 'gemma' or 'llama'
        """
        self.config = config
        self.model_name = model_name
        self.model_config = config['models'][model_name]
        self.phase2_config = config['phase2']

        # Paths
        self.feature_dir = Path(config['data']['output_dir']) / model_name / 'features'
        self.output_dir = Path(config['data']['output_dir']) / model_name / 'correlations'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_layer_features(self, layer: int) -> Dict[str, np.ndarray]:
        """
        Load features for a specific layer.

        Args:
            layer: Layer number

        Returns:
            Dictionary with features and metadata
        """
        feature_file = self.feature_dir / f'layer_{layer}_features.npz'

        if not feature_file.exists():
            raise FileNotFoundError(f"Feature file not found: {feature_file}")

        data = np.load(feature_file, allow_pickle=True)

        return {
            'features': data['features'],
            'choices': data['choices'],
            'game_ids': data['game_ids'],
            'rounds': data['rounds'],
            'prompt_conditions': data['prompt_conditions'],
            'bet_types': data['bet_types'],
            'models': data['models'],
            'layer': int(data['layer'])
        }

    def binary_choice_analysis(self, layer: int) -> Dict[str, Any]:
        """
        Binary analysis: Safe (Option 1) vs Risky (Options 2/3/4).

        Args:
            layer: Layer number

        Returns:
            Analysis results
        """
        logger.info(f"Layer {layer}: Binary choice analysis (Safe vs Risky)")

        # Load features
        data = self.load_layer_features(layer)

        features = data['features']
        choices = data['choices']

        # Create binary labels: 1 (safe) vs 0 (risky)
        binary_labels = (choices == 1).astype(int)

        # Perform t-test analysis
        results = binary_ttest_analysis(
            features=features,
            labels=binary_labels,
            label_0=0,  # Risky (choices 2/3/4)
            label_1=1,  # Safe (choice 1)
            min_cohens_d=self.phase2_config['min_cohens_d'],
            fdr_alpha=self.phase2_config['fdr_alpha']
        )

        # Get top features
        feature_ids = np.arange(features.shape[1])

        # Safe features (higher in choice 1)
        safe_features = get_top_features(
            feature_ids=feature_ids,
            scores=results['cohens_ds'],
            significant=results['significant'] & (results['cohens_ds'] > 0),
            n_top=self.phase2_config['n_top_features'],
            ascending=False
        )

        # Risky features (higher in choices 2/3/4)
        risky_features = get_top_features(
            feature_ids=feature_ids,
            scores=-results['cohens_ds'],  # Negate for risky direction
            significant=results['significant'] & (results['cohens_ds'] < 0),
            n_top=self.phase2_config['n_top_features'],
            ascending=False
        )

        # Compute mean activations for top features
        top_feature_ids = [f[0] for f in safe_features[:10]] + [f[0] for f in risky_features[:10]]
        feature_means = compute_feature_means_by_group(features, binary_labels, top_feature_ids)

        return {
            'layer': layer,
            'analysis_type': 'binary_safe_vs_risky',
            'n_safe_samples': int(results['group_1_size']),
            'n_risky_samples': int(results['group_0_size']),
            'n_significant_features': int(results['n_significant']),
            'fdr_threshold': float(results['fdr_threshold']),
            'safe_features': safe_features,
            'risky_features': risky_features,
            'feature_means': feature_means
        }

    def multiclass_choice_analysis(self, layer: int) -> Dict[str, Any]:
        """
        Multi-class analysis: Option 1 vs 2 vs 3 vs 4.

        Args:
            layer: Layer number

        Returns:
            Analysis results
        """
        logger.info(f"Layer {layer}: Multi-class choice analysis (4-way)")

        # Load features
        data = self.load_layer_features(layer)

        features = data['features']
        choices = data['choices']

        # Perform ANOVA analysis
        results = multiclass_anova_analysis(
            features=features,
            labels=choices,
            min_eta_squared=0.01,
            fdr_alpha=self.phase2_config['fdr_alpha']
        )

        # Get top features
        feature_ids = np.arange(features.shape[1])

        top_features = get_top_features(
            feature_ids=feature_ids,
            scores=results['eta_squareds'],
            significant=results['significant'],
            n_top=self.phase2_config['n_top_features'],
            ascending=False
        )

        # Compute mean activations by choice for top features
        top_feature_ids = [f[0] for f in top_features[:20]]
        feature_means = compute_feature_means_by_group(features, choices, top_feature_ids)

        return {
            'layer': layer,
            'analysis_type': 'multiclass_4way',
            'n_classes': int(results['n_classes']),
            'n_significant_features': int(results['n_significant']),
            'fdr_threshold': float(results['fdr_threshold']),
            'top_features': top_features,
            'feature_means': feature_means
        }

    def prompt_condition_analysis(self, layer: int) -> Dict[str, Any]:
        """
        Analyze feature differences by prompt condition (BASE/G/M/GM).

        Args:
            layer: Layer number

        Returns:
            Analysis results
        """
        logger.info(f"Layer {layer}: Prompt condition analysis")

        # Load features
        data = self.load_layer_features(layer)

        features = data['features']
        conditions = data['prompt_conditions']

        # Perform ANOVA analysis
        results = multiclass_anova_analysis(
            features=features,
            labels=conditions,
            min_eta_squared=0.01,
            fdr_alpha=self.phase2_config['fdr_alpha']
        )

        # Get top features
        feature_ids = np.arange(features.shape[1])

        top_features = get_top_features(
            feature_ids=feature_ids,
            scores=results['eta_squareds'],
            significant=results['significant'],
            n_top=self.phase2_config['n_top_features'],
            ascending=False
        )

        # Compute mean activations by condition
        top_feature_ids = [f[0] for f in top_features[:20]]
        feature_means = compute_feature_means_by_group(features, conditions, top_feature_ids)

        return {
            'layer': layer,
            'analysis_type': 'prompt_conditions',
            'conditions': list(np.unique(conditions)),
            'n_significant_features': int(results['n_significant']),
            'fdr_threshold': float(results['fdr_threshold']),
            'top_features': top_features,
            'feature_means': feature_means
        }

    def betting_type_analysis(self, layer: int) -> Dict[str, Any]:
        """
        Analyze feature differences by betting type (fixed/variable).

        Args:
            layer: Layer number

        Returns:
            Analysis results
        """
        logger.info(f"Layer {layer}: Betting type analysis")

        # Load features
        data = self.load_layer_features(layer)

        features = data['features']
        bet_types = data['bet_types']

        # Create binary labels
        binary_labels = (bet_types == 'variable').astype(int)

        # Perform t-test analysis
        results = binary_ttest_analysis(
            features=features,
            labels=binary_labels,
            label_0=0,  # Fixed
            label_1=1,  # Variable
            min_cohens_d=self.phase2_config['min_cohens_d'],
            fdr_alpha=self.phase2_config['fdr_alpha']
        )

        # Get top features
        feature_ids = np.arange(features.shape[1])

        # Variable features (higher in variable betting)
        variable_features = get_top_features(
            feature_ids=feature_ids,
            scores=results['cohens_ds'],
            significant=results['significant'] & (results['cohens_ds'] > 0),
            n_top=self.phase2_config['n_top_features'],
            ascending=False
        )

        # Fixed features (higher in fixed betting)
        fixed_features = get_top_features(
            feature_ids=feature_ids,
            scores=-results['cohens_ds'],
            significant=results['significant'] & (results['cohens_ds'] < 0),
            n_top=self.phase2_config['n_top_features'],
            ascending=False
        )

        # Compute mean activations
        top_feature_ids = [f[0] for f in variable_features[:10]] + [f[0] for f in fixed_features[:10]]
        feature_means = compute_feature_means_by_group(features, binary_labels, top_feature_ids)

        return {
            'layer': layer,
            'analysis_type': 'betting_type',
            'n_fixed_samples': int(results['group_0_size']),
            'n_variable_samples': int(results['group_1_size']),
            'n_significant_features': int(results['n_significant']),
            'fdr_threshold': float(results['fdr_threshold']),
            'variable_features': variable_features,
            'fixed_features': fixed_features,
            'feature_means': feature_means
        }

    def analyze_all_layers(self):
        """Run all analyses for all target layers."""
        target_layers = self.model_config['target_layers']
        analyses_to_run = self.phase2_config['analyses']

        logger.info(f"Running analyses for {len(target_layers)} layers")
        logger.info(f"Analyses: {analyses_to_run}")

        all_results = {}

        for layer in tqdm(target_layers, desc="Analyzing layers"):
            layer_results = {'layer': layer}

            try:
                if 'binary_classification' in analyses_to_run:
                    layer_results['binary'] = self.binary_choice_analysis(layer)

                if 'multiclass_classification' in analyses_to_run:
                    layer_results['multiclass'] = self.multiclass_choice_analysis(layer)

                if 'prompt_condition_effects' in analyses_to_run:
                    layer_results['prompt_conditions'] = self.prompt_condition_analysis(layer)

                if 'betting_type_effects' in analyses_to_run:
                    layer_results['betting_type'] = self.betting_type_analysis(layer)

                all_results[f'layer_{layer}'] = layer_results

            except Exception as e:
                logger.error(f"Error analyzing layer {layer}: {e}")
                continue

        # Save results
        self.save_results(all_results)

        return all_results

    def save_results(self, results: Dict[str, Any]):
        """Save analysis results to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f'correlation_analysis_{self.model_name}_{timestamp}.json'

        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            else:
                return obj

        results_serializable = convert_to_serializable(results)

        with open(output_file, 'w') as f:
            json.dump(results_serializable, f, indent=2)

        logger.info(f"Results saved to {output_file}")

        # Print summary
        self.print_summary(results)

    def print_summary(self, results: Dict[str, Any]):
        """Print summary of results."""
        print("\n" + "="*60)
        print("PHASE 2: CORRELATION ANALYSIS SUMMARY")
        print("="*60)
        print(f"Model: {self.model_name}")
        print(f"Layers analyzed: {len(results)}")
        print()

        for layer_key, layer_results in results.items():
            print(f"\n{layer_key}:")

            if 'binary' in layer_results:
                binary = layer_results['binary']
                print(f"  Binary (Safe vs Risky):")
                print(f"    Significant features: {binary['n_significant_features']}")
                print(f"    Top safe features: {len(binary['safe_features'])}")
                print(f"    Top risky features: {len(binary['risky_features'])}")

            if 'multiclass' in layer_results:
                multi = layer_results['multiclass']
                print(f"  Multi-class (4-way):")
                print(f"    Significant features: {multi['n_significant_features']}")
                print(f"    Top features: {len(multi['top_features'])}")

            if 'prompt_conditions' in layer_results:
                prompt = layer_results['prompt_conditions']
                print(f"  Prompt conditions:")
                print(f"    Significant features: {prompt['n_significant_features']}")

            if 'betting_type' in layer_results:
                betting = layer_results['betting_type']
                print(f"  Betting type:")
                print(f"    Significant features: {betting['n_significant_features']}")

        print("\n" + "="*60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Phase 2: Correlation Analysis')
    parser.add_argument('--model', type=str, required=True, choices=['gemma', 'llama'],
                        help='Model to analyze')
    parser.add_argument('--config', type=str,
                        default='configs/experiment_config.yaml',
                        help='Path to config file')

    args = parser.parse_args()

    # Load config
    config_path = Path(__file__).parent.parent / args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from {config_path}")

    # Initialize analyzer
    analyzer = CorrelationAnalyzer(config, args.model)

    # Run all analyses
    analyzer.analyze_all_layers()

    logger.info("Phase 2 complete!")


if __name__ == '__main__':
    main()
