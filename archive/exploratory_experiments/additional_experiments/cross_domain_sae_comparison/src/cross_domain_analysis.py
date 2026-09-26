#!/usr/bin/env python3
"""
Cross-Domain SAE Feature Comparison

Analyzes feature overlap across 3 gambling domains:
- Slot Machine (existing)
- Loot Box
- Blackjack

Usage:
    python cross_domain_analysis.py --model llama
    python cross_domain_analysis.py --model gemma
"""

import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple
from collections import Counter
import logging

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CrossDomainComparisonAnalyzer:
    """Analyze SAE feature overlap across gambling domains"""

    def __init__(
        self,
        model_type: str,
        slot_machine_dir: str = "/mnt/c/Users/oollccddss/git/data/llm-addiction/sae_patching/corrected_sae_analysis",
        alternative_dir: str = "/mnt/c/Users/oollccddss/git/data/llm-addiction/alternative_paradigms"
    ):
        """
        Initialize cross-domain analyzer.

        Args:
            model_type: 'llama' or 'gemma'
            slot_machine_dir: Slot machine Phase2 results directory
            alternative_dir: Alternative paradigms data directory
        """
        self.model_type = model_type
        self.slot_machine_dir = Path(slot_machine_dir) / model_type
        self.alternative_dir = Path(alternative_dir)

        self.domains = ['slot_machine', 'lootbox', 'blackjack']
        self.domain_features = {}  # {domain: {'risky': set(), 'safe': set()}}

    def load_slot_machine_features(self) -> Tuple[Set, Set]:
        """
        Load Slot Machine Phase2 correlation results.

        Returns:
            (risky_features, safe_features)
            Each set contains (layer, feature_id) tuples
        """
        logger.info(f"Loading Slot Machine features from {self.slot_machine_dir}")

        # Find correlation_all_features file
        all_files = list(self.slot_machine_dir.glob("correlation_all_features_*.json"))
        if not all_files:
            raise FileNotFoundError(f"No correlation file found in {self.slot_machine_dir}")

        all_file = sorted(all_files)[-1]  # Most recent
        logger.info(f"  File: {all_file.name}")

        with open(all_file) as f:
            all_features = json.load(f)

        risky = set()
        safe = set()

        for feat in all_features:
            if not feat.get('fdr_significant', False):
                continue

            layer = feat['layer']
            feature_id = feat['feature_id']
            cohens_d = feat['cohens_d']

            if cohens_d > 0.3:
                risky.add((layer, feature_id))
            elif cohens_d < -0.3:
                safe.add((layer, feature_id))

        logger.info(f"  Risky features: {len(risky)}")
        logger.info(f"  Safe features: {len(safe)}")

        return risky, safe

    def load_paradigm_features(self, paradigm: str) -> Tuple[Set, Set]:
        """
        Load alternative paradigm Phase2 correlation results.

        Args:
            paradigm: 'lootbox' or 'blackjack'

        Returns:
            (risky_features, safe_features)
        """
        logger.info(f"Loading {paradigm.capitalize()} features...")

        features_dir = self.alternative_dir / paradigm / 'sae_features'

        # Find correlation_all_features file
        all_files = list(features_dir.glob("correlation_all_features_*.json"))
        if not all_files:
            raise FileNotFoundError(f"No correlation file found in {features_dir}")

        all_file = sorted(all_files)[-1]
        logger.info(f"  File: {all_file.name}")

        with open(all_file) as f:
            all_features = json.load(f)

        risky = set()
        safe = set()

        for feat in all_features:
            if not feat.get('fdr_significant', False):
                continue

            layer = feat['layer']
            feature_id = feat['feature_id']
            cohens_d = feat['cohens_d']

            if cohens_d > 0.3:
                risky.add((layer, feature_id))
            elif cohens_d < -0.3:
                safe.add((layer, feature_id))

        logger.info(f"  Risky features: {len(risky)}")
        logger.info(f"  Safe features: {len(safe)}")

        return risky, safe

    def load_all_domains(self):
        """Load features from all domains"""
        logger.info(f"\n{'='*60}")
        logger.info("LOADING DOMAIN FEATURES")
        logger.info(f"Model: {self.model_type.upper()}")
        logger.info(f"{'='*60}\n")

        # Slot machine
        risky, safe = self.load_slot_machine_features()
        self.domain_features['slot_machine'] = {'risky': risky, 'safe': safe}

        # Loot box
        risky, safe = self.load_paradigm_features('lootbox')
        self.domain_features['lootbox'] = {'risky': risky, 'safe': safe}

        # Blackjack
        risky, safe = self.load_paradigm_features('blackjack')
        self.domain_features['blackjack'] = {'risky': risky, 'safe': safe}

    def compute_jaccard_overlap(self, domain1: str, domain2: str) -> Dict:
        """
        Compute Jaccard similarity between two domains.

        Args:
            domain1: First domain
            domain2: Second domain

        Returns:
            Overlap statistics
        """
        risky1 = self.domain_features[domain1]['risky']
        risky2 = self.domain_features[domain2]['risky']
        safe1 = self.domain_features[domain1]['safe']
        safe2 = self.domain_features[domain2]['safe']

        # Risky overlap
        risky_inter = risky1 & risky2
        risky_union = risky1 | risky2
        risky_jaccard = len(risky_inter) / len(risky_union) if risky_union else 0.0

        # Safe overlap
        safe_inter = safe1 & safe2
        safe_union = safe1 | safe2
        safe_jaccard = len(safe_inter) / len(safe_union) if safe_union else 0.0

        return {
            'domain1': domain1,
            'domain2': domain2,
            'risky_jaccard': risky_jaccard,
            'risky_overlap_count': len(risky_inter),
            'risky_total': len(risky_union),
            'safe_jaccard': safe_jaccard,
            'safe_overlap_count': len(safe_inter),
            'safe_total': len(safe_union)
        }

    def identify_core_features(self, min_domains: int = 2) -> Dict:
        """
        Identify core features present in multiple domains.

        Args:
            min_domains: Minimum number of domains

        Returns:
            Core feature analysis
        """
        # Count domain appearances for each feature
        risky_counter = Counter()
        safe_counter = Counter()

        for domain in self.domains:
            for feat in self.domain_features[domain]['risky']:
                risky_counter[feat] += 1
            for feat in self.domain_features[domain]['safe']:
                safe_counter[feat] += 1

        # Core features (2+ domains)
        core_risky = [(feat, count) for feat, count in risky_counter.items() if count >= min_domains]
        core_safe = [(feat, count) for feat, count in safe_counter.items() if count >= min_domains]

        # Universal features (all 3 domains)
        universal_risky = [(feat, count) for feat, count in risky_counter.items() if count == 3]
        universal_safe = [(feat, count) for feat, count in safe_counter.items() if count == 3]

        return {
            'core_risky_features': sorted(core_risky, key=lambda x: x[1], reverse=True),
            'core_safe_features': sorted(core_safe, key=lambda x: x[1], reverse=True),
            'universal_risky_features': universal_risky,
            'universal_safe_features': universal_safe,
            'core_risky_count': len(core_risky),
            'core_safe_count': len(core_safe),
            'universal_risky_count': len(universal_risky),
            'universal_safe_count': len(universal_safe)
        }

    def run(self) -> Dict:
        """Run full cross-domain analysis"""
        logger.info(f"\n{'='*60}")
        logger.info("CROSS-DOMAIN SAE FEATURE COMPARISON")
        logger.info(f"Model: {self.model_type.upper()}")
        logger.info(f"Domains: {', '.join([d.replace('_', ' ').title() for d in self.domains])}")
        logger.info(f"{'='*60}\n")

        # Load all domains
        self.load_all_domains()

        # Pairwise Jaccard similarities
        logger.info(f"\n{'='*60}")
        logger.info("PAIRWISE JACCARD SIMILARITIES")
        logger.info(f"{'='*60}\n")

        pairwise_overlaps = []
        for i, domain1 in enumerate(self.domains):
            for domain2 in self.domains[i+1:]:
                overlap = self.compute_jaccard_overlap(domain1, domain2)
                pairwise_overlaps.append(overlap)

                logger.info(f"{domain1.replace('_', ' ').title()} ↔ {domain2.replace('_', ' ').title()}:")
                logger.info(f"  Risky Jaccard: {overlap['risky_jaccard']:.3f} ({overlap['risky_overlap_count']} / {overlap['risky_total']})")
                logger.info(f"  Safe Jaccard: {overlap['safe_jaccard']:.3f} ({overlap['safe_overlap_count']} / {overlap['safe_total']})")
                logger.info("")

        # Core features
        logger.info(f"\n{'='*60}")
        logger.info("CORE FEATURES (2+ DOMAINS)")
        logger.info(f"{'='*60}\n")

        core_analysis = self.identify_core_features(min_domains=2)

        logger.info(f"Core Risky Features (2+ domains): {core_analysis['core_risky_count']}")
        logger.info(f"Core Safe Features (2+ domains): {core_analysis['core_safe_count']}")
        logger.info(f"\nUniversal Risky Features (all 3 domains): {core_analysis['universal_risky_count']}")
        logger.info(f"Universal Safe Features (all 3 domains): {core_analysis['universal_safe_count']}")

        if core_analysis['universal_risky_features']:
            logger.info(f"\nTop 10 Universal Risky Features:")
            for feat, count in core_analysis['universal_risky_features'][:10]:
                logger.info(f"  Layer {feat[0]}, Feature {feat[1]}: {count} domains")

        if core_analysis['universal_safe_features']:
            logger.info(f"\nTop 10 Universal Safe Features:")
            for feat, count in core_analysis['universal_safe_features'][:10]:
                logger.info(f"  Layer {feat[0]}, Feature {feat[1]}: {count} domains")

        # Compile results
        results = {
            'model_type': self.model_type,
            'timestamp': datetime.now().isoformat(),
            'domains': self.domains,
            'domain_summary': {
                domain: {
                    'risky_count': len(self.domain_features[domain]['risky']),
                    'safe_count': len(self.domain_features[domain]['safe'])
                }
                for domain in self.domains
            },
            'pairwise_overlaps': pairwise_overlaps,
            'core_features': core_analysis
        }

        return results

    def save_results(self, results: Dict, output_dir: Path):
        """Save results to JSON"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f'cross_domain_overlap_{self.model_type}_{timestamp}.json'

        # Convert sets to lists for JSON serialization
        results_copy = results.copy()
        core = results_copy['core_features']

        # Convert tuples to lists
        core['core_risky_features'] = [
            {'layer': feat[0], 'feature_id': feat[1], 'domain_count': count}
            for feat, count in core['core_risky_features']
        ]
        core['core_safe_features'] = [
            {'layer': feat[0], 'feature_id': feat[1], 'domain_count': count}
            for feat, count in core['core_safe_features']
        ]
        core['universal_risky_features'] = [
            {'layer': feat[0], 'feature_id': feat[1], 'domain_count': count}
            for feat, count in core['universal_risky_features']
        ]
        core['universal_safe_features'] = [
            {'layer': feat[0], 'feature_id': feat[1], 'domain_count': count}
            for feat, count in core['universal_safe_features']
        ]

        with open(output_file, 'w') as f:
            json.dump(results_copy, f, indent=2)

        logger.info(f"\n✅ Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Cross-Domain SAE Feature Comparison')
    parser.add_argument('--model', type=str, required=True, choices=['llama', 'gemma'],
                        help='Model type')
    parser.add_argument('--slot-machine-dir', type=str,
                        default='/mnt/c/Users/oollccddss/git/data/llm-addiction/sae_patching/corrected_sae_analysis',
                        help='Slot machine Phase2 results directory')
    parser.add_argument('--alternative-dir', type=str,
                        default='/mnt/c/Users/oollccddss/git/data/llm-addiction/alternative_paradigms',
                        help='Alternative paradigms data directory')
    parser.add_argument('--output-dir', type=str,
                        default='results',
                        help='Output directory')

    args = parser.parse_args()

    # Run analysis
    analyzer = CrossDomainComparisonAnalyzer(
        model_type=args.model,
        slot_machine_dir=args.slot_machine_dir,
        alternative_dir=args.alternative_dir
    )

    results = analyzer.run()

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    analyzer.save_results(results, output_dir)

    logger.info(f"\n{'='*60}")
    logger.info("ANALYSIS COMPLETE")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()
