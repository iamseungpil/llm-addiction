#!/usr/bin/env python3
"""
Cross-Reference Addiction Features with Condition Features

Links Variable/Fixed condition features with Phase2 risky/safe features
to identify:
1. Risk Amplification: Variable-higher features that are also risky
2. Protective: Fixed-higher features that are also safe

This demonstrates that betting condition modulates addiction features.
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple
from datetime import datetime
import logging

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AddictionFeatureCrossReference:
    """Cross-reference condition features with addiction features"""

    def __init__(
        self,
        model_type: str,
        phase2_data_dir: str = "/mnt/c/Users/oollccddss/git/data/llm-addiction/sae_patching/corrected_sae_analysis",
        condition_data_dir: str = "results"
    ):
        """
        Initialize cross-reference analyzer.

        Args:
            model_type: 'llama' or 'gemma'
            phase2_data_dir: Directory containing Phase2 correlation results
            condition_data_dir: Directory containing condition comparison results
        """
        self.model_type = model_type
        self.phase2_dir = Path(phase2_data_dir) / model_type
        self.condition_dir = Path(condition_data_dir)

        self.risky_features = set()  # (layer, feature_id) tuples
        self.safe_features = set()
        self.variable_higher = set()
        self.fixed_higher = set()

    def load_phase2_features(self) -> Tuple[int, int]:
        """
        Load Phase2 correlation results (risky/safe features).

        Returns:
            (n_risky, n_safe) counts
        """
        logger.info(f"Loading Phase2 correlation results from {self.phase2_dir}")

        # Find correlation_all_features file
        all_features_files = list(self.phase2_dir.glob("correlation_all_features_*.json"))
        if not all_features_files:
            raise FileNotFoundError(f"No correlation_all_features file found in {self.phase2_dir}")

        all_features_file = sorted(all_features_files)[-1]  # Most recent
        logger.info(f"  File: {all_features_file.name}")

        with open(all_features_file) as f:
            all_features = json.load(f)

        # Filter risky and safe features
        # Risky: cohens_d > 0.3 AND fdr_significant
        # Safe: cohens_d < -0.3 AND fdr_significant
        for feat in all_features:
            layer = feat['layer']
            feature_id = feat['feature_id']
            cohens_d = feat['cohens_d']
            fdr_sig = feat.get('fdr_significant', feat.get('p_fdr', 1.0) < 0.05)

            if not fdr_sig:
                continue

            if cohens_d > 0.3:
                self.risky_features.add((layer, feature_id))
            elif cohens_d < -0.3:
                self.safe_features.add((layer, feature_id))

        logger.info(f"  Risky features (d > 0.3, FDR < 0.05): {len(self.risky_features)}")
        logger.info(f"  Safe features (d < -0.3, FDR < 0.05): {len(self.safe_features)}")

        return len(self.risky_features), len(self.safe_features)

    def load_condition_features(self) -> Tuple[int, int]:
        """
        Load condition comparison results (Variable/Fixed features).

        Returns:
            (n_variable_higher, n_fixed_higher) counts
        """
        logger.info(f"Loading condition comparison results from {self.condition_dir}")

        # Find variable_vs_fixed file
        var_fixed_files = list(self.condition_dir.glob(f"variable_vs_fixed_{self.model_type}_*.json"))
        if not var_fixed_files:
            raise FileNotFoundError(f"No variable_vs_fixed file found in {self.condition_dir}")

        var_fixed_file = sorted(var_fixed_files)[-1]  # Most recent
        logger.info(f"  File: {var_fixed_file.name}")

        with open(var_fixed_file) as f:
            data = json.load(f)

        # Extract features
        all_results = data.get('all_results', [])
        for feat in all_results:
            layer = feat['layer']
            feature_id = feat['feature_id']
            direction = feat.get('direction', '')

            if direction == 'higher_in_variable':
                self.variable_higher.add((layer, feature_id))
            elif direction == 'higher_in_fixed':
                self.fixed_higher.add((layer, feature_id))

        logger.info(f"  Variable-higher features: {len(self.variable_higher)}")
        logger.info(f"  Fixed-higher features: {len(self.fixed_higher)}")

        return len(self.variable_higher), len(self.fixed_higher)

    def compute_overlap(self) -> Dict:
        """
        Compute overlap between condition and addiction features.

        Returns:
            Dictionary with overlap statistics
        """
        logger.info("\nComputing overlaps...")

        # Risk Amplification: Variable-higher ∩ Risky
        risk_amplification = self.variable_higher & self.risky_features
        risk_amp_jaccard = len(risk_amplification) / len(self.variable_higher | self.risky_features) \
            if (self.variable_higher | self.risky_features) else 0.0

        # Protective: Fixed-higher ∩ Safe
        protective = self.fixed_higher & self.safe_features
        protective_jaccard = len(protective) / len(self.fixed_higher | self.safe_features) \
            if (self.fixed_higher | self.safe_features) else 0.0

        # Additional patterns
        # Variable-higher ∩ Safe (counterintuitive)
        variable_safe = self.variable_higher & self.safe_features
        # Fixed-higher ∩ Risky (counterintuitive)
        fixed_risky = self.fixed_higher & self.risky_features

        results = {
            'risk_amplification': {
                'description': 'Variable-higher features that are also risky (explains increased bankruptcy)',
                'count': len(risk_amplification),
                'jaccard_similarity': risk_amp_jaccard,
                'variable_higher_total': len(self.variable_higher),
                'risky_total': len(self.risky_features),
                'percentage_of_variable': 100 * len(risk_amplification) / len(self.variable_higher) if self.variable_higher else 0,
                'percentage_of_risky': 100 * len(risk_amplification) / len(self.risky_features) if self.risky_features else 0,
                'features': sorted(list(risk_amplification))[:50]  # Top 50 for inspection
            },
            'protective': {
                'description': 'Fixed-higher features that are also safe (explains reduced bankruptcy)',
                'count': len(protective),
                'jaccard_similarity': protective_jaccard,
                'fixed_higher_total': len(self.fixed_higher),
                'safe_total': len(self.safe_features),
                'percentage_of_fixed': 100 * len(protective) / len(self.fixed_higher) if self.fixed_higher else 0,
                'percentage_of_safe': 100 * len(protective) / len(self.safe_features) if self.safe_features else 0,
                'features': sorted(list(protective))[:50]
            },
            'counterintuitive_patterns': {
                'variable_safe': {
                    'count': len(variable_safe),
                    'description': 'Variable-higher but safe (unexpected)',
                    'features': sorted(list(variable_safe))[:20]
                },
                'fixed_risky': {
                    'count': len(fixed_risky),
                    'description': 'Fixed-higher but risky (unexpected)',
                    'features': sorted(list(fixed_risky))[:20]
                }
            }
        }

        # Log summary
        logger.info(f"\n{'='*60}")
        logger.info("OVERLAP SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Risk Amplification (Variable ∩ Risky):")
        logger.info(f"  Count: {results['risk_amplification']['count']}")
        logger.info(f"  Jaccard: {results['risk_amplification']['jaccard_similarity']:.3f}")
        logger.info(f"  {results['risk_amplification']['percentage_of_variable']:.1f}% of Variable-higher features")
        logger.info(f"  {results['risk_amplification']['percentage_of_risky']:.1f}% of Risky features")
        logger.info(f"\nProtective (Fixed ∩ Safe):")
        logger.info(f"  Count: {results['protective']['count']}")
        logger.info(f"  Jaccard: {results['protective']['jaccard_similarity']:.3f}")
        logger.info(f"  {results['protective']['percentage_of_fixed']:.1f}% of Fixed-higher features")
        logger.info(f"  {results['protective']['percentage_of_safe']:.1f}% of Safe features")

        if results['counterintuitive_patterns']['variable_safe']['count'] > 0:
            logger.info(f"\n⚠️  Counterintuitive: {results['counterintuitive_patterns']['variable_safe']['count']} Variable-Safe features")
        if results['counterintuitive_patterns']['fixed_risky']['count'] > 0:
            logger.info(f"⚠️  Counterintuitive: {results['counterintuitive_patterns']['fixed_risky']['count']} Fixed-Risky features")

        return results

    def run(self) -> Dict:
        """
        Run full cross-reference analysis.

        Returns:
            Complete analysis results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"ADDICTION FEATURE CROSS-REFERENCE ANALYSIS")
        logger.info(f"Model: {self.model_type.upper()}")
        logger.info(f"{'='*60}\n")

        # Load data
        n_risky, n_safe = self.load_phase2_features()
        n_variable, n_fixed = self.load_condition_features()

        # Compute overlaps
        overlap_results = self.compute_overlap()

        # Create final report
        report = {
            'model_type': self.model_type,
            'timestamp': datetime.now().isoformat(),
            'phase2_summary': {
                'risky_features': n_risky,
                'safe_features': n_safe
            },
            'condition_summary': {
                'variable_higher': n_variable,
                'fixed_higher': n_fixed
            },
            'overlap_analysis': overlap_results
        }

        return report

    def save_results(self, results: Dict, output_file: Path):
        """Save results to JSON"""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\n✅ Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Cross-reference addiction features with condition features')
    parser.add_argument('--model', type=str, required=True, choices=['llama', 'gemma'],
                        help='Model type to analyze')
    parser.add_argument('--phase2-dir', type=str,
                        default='/mnt/c/Users/oollccddss/git/data/llm-addiction/sae_patching/corrected_sae_analysis',
                        help='Phase2 correlation results directory')
    parser.add_argument('--condition-dir', type=str,
                        default='results',
                        help='Condition comparison results directory')
    parser.add_argument('--output-dir', type=str,
                        default='results',
                        help='Output directory')

    args = parser.parse_args()

    # Run analysis
    analyzer = AddictionFeatureCrossReference(
        model_type=args.model,
        phase2_data_dir=args.phase2_dir,
        condition_data_dir=args.condition_dir
    )

    results = analyzer.run()

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'cross_reference_{args.model}_{timestamp}.json'

    analyzer.save_results(results, output_file)


if __name__ == '__main__':
    main()
