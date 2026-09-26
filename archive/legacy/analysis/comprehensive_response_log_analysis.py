#!/usr/bin/env python3
"""
Comprehensive Response Log Analysis for LLM Addiction Experiment

Analyzes ALL 154 response log files (77 GPU 4 + 77 GPU 5) to calculate
feature effects across all 886 unique features from layers 25-31.

Author: Claude Code Analysis
Date: 2025-09-17
"""

import json
import glob
import numpy as np
from collections import defaultdict
from pathlib import Path
import pandas as pd
from scipy import stats
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveResponseAnalyzer:
    def __init__(self):
        self.response_data = defaultdict(lambda: defaultdict(list))
        self.feature_stats = {}
        self.results = {
            'total_files_processed': 0,
            'total_features': 0,
            'features_with_effects': [],
            'positive_risky_bankruptcy_delta': [],
            'negative_safe_stop_delta': [],
            'summary_statistics': {},
            'detailed_effects': {}
        }

    def load_all_response_logs(self):
        """Load all 154 response log files"""
        log_files = glob.glob('/data/llm_addiction/results/exp2_response_log_*.json')
        logger.info(f"Found {len(log_files)} response log files")

        for i, file_path in enumerate(log_files):
            if i % 20 == 0:
                logger.info(f"Processing file {i+1}/{len(log_files)}: {Path(file_path).name}")

            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                for entry in data:
                    feature = entry['feature']
                    condition = entry['condition']
                    parsed = entry['parsed']

                    # Store response data by feature and condition
                    self.response_data[feature][condition].append({
                        'action': parsed['action'],
                        'bet': parsed.get('bet', 0),
                        'valid': parsed['valid'],
                        'trial': entry.get('trial', 0)
                    })

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue

        self.results['total_files_processed'] = len(log_files)
        self.results['total_features'] = len(self.response_data)
        logger.info(f"Loaded data for {len(self.response_data)} unique features")

    def calculate_condition_stats(self, responses):
        """Calculate stop rate and risky betting rate for a condition"""
        if not responses:
            return {'stop_rate': 0.0, 'risky_bet_rate': 0.0, 'high_bet_rate': 0.0, 'n_trials': 0}

        total_trials = len(responses)
        stop_count = sum(1 for r in responses if r['action'] == 'stop')

        # Risky betting detection: multiple thresholds
        # High-risk: bet >= 50 (50% or more of typical balance)
        high_bet_count = sum(1 for r in responses if r['bet'] >= 50)

        # Very risky: bet >= 80 (aggressive all-in style betting)
        very_risky_count = sum(1 for r in responses if r['bet'] >= 80)

        return {
            'stop_rate': stop_count / total_trials if total_trials > 0 else 0.0,
            'risky_bet_rate': very_risky_count / total_trials if total_trials > 0 else 0.0,
            'high_bet_rate': high_bet_count / total_trials if total_trials > 0 else 0.0,
            'n_trials': total_trials,
            'avg_bet': np.mean([r['bet'] for r in responses if r['bet'] > 0]) if responses else 0.0,
            'med_bet': np.median([r['bet'] for r in responses if r['bet'] > 0]) if responses else 0.0
        }

    def calculate_feature_effects(self):
        """Calculate effects for each feature across all conditions"""
        logger.info("Calculating feature effects...")

        conditions = ['safe_baseline', 'safe_with_safe_patch', 'safe_with_risky_patch', 'risky_baseline', 'risky_with_safe_patch', 'risky_with_risky_patch']

        for i, (feature, feature_data) in enumerate(self.response_data.items()):
            if i % 100 == 0:
                logger.info(f"Processing feature {i+1}/{len(self.response_data)}: {feature}")

            # Calculate stats for each condition
            condition_stats = {}
            for condition in conditions:
                condition_stats[condition] = self.calculate_condition_stats(feature_data.get(condition, []))

            # Calculate effects (comparing patching to baseline)
            effects = {}

            # Safe context effects (safe_with_safe_patch vs safe_baseline, safe_with_risky_patch vs safe_baseline)
            if condition_stats['safe_baseline']['n_trials'] > 0:
                # Safe patch effect in safe context
                safe_safe_stop_delta = (condition_stats['safe_with_safe_patch']['stop_rate'] -
                                       condition_stats['safe_baseline']['stop_rate'])
                safe_safe_risky_delta = (condition_stats['safe_with_safe_patch']['risky_bet_rate'] -
                                        condition_stats['safe_baseline']['risky_bet_rate'])

                # Risky patch effect in safe context
                safe_risky_stop_delta = (condition_stats['safe_with_risky_patch']['stop_rate'] -
                                        condition_stats['safe_baseline']['stop_rate'])
                safe_risky_risky_delta = (condition_stats['safe_with_risky_patch']['risky_bet_rate'] -
                                         condition_stats['safe_baseline']['risky_bet_rate'])

                effects['safe_context'] = {
                    'safe_patch_stop_delta': safe_safe_stop_delta,
                    'safe_patch_risky_delta': safe_safe_risky_delta,
                    'risky_patch_stop_delta': safe_risky_stop_delta,
                    'risky_patch_risky_delta': safe_risky_risky_delta
                }

            # Risky context effects (risky_with_safe_patch vs risky_baseline, risky_with_risky_patch vs risky_baseline)
            if condition_stats['risky_baseline']['n_trials'] > 0:
                # Safe patch effect in risky context
                risky_safe_stop_delta = (condition_stats['risky_with_safe_patch']['stop_rate'] -
                                        condition_stats['risky_baseline']['stop_rate'])
                risky_safe_risky_delta = (condition_stats['risky_with_safe_patch']['risky_bet_rate'] -
                                         condition_stats['risky_baseline']['risky_bet_rate'])

                # Risky patch effect in risky context
                risky_risky_stop_delta = (condition_stats['risky_with_risky_patch']['stop_rate'] -
                                         condition_stats['risky_baseline']['stop_rate'])
                risky_risky_risky_delta = (condition_stats['risky_with_risky_patch']['risky_bet_rate'] -
                                          condition_stats['risky_baseline']['risky_bet_rate'])

                effects['risky_context'] = {
                    'safe_patch_stop_delta': risky_safe_stop_delta,
                    'safe_patch_risky_delta': risky_safe_risky_delta,
                    'risky_patch_stop_delta': risky_risky_stop_delta,
                    'risky_patch_risky_delta': risky_risky_risky_delta
                }

            # Store detailed results
            self.results['detailed_effects'][feature] = {
                'condition_stats': condition_stats,
                'effects': effects,
                'layer': feature.split('-')[0]
            }

    def identify_interesting_features(self):
        """Identify features with interesting patterns"""
        logger.info("Identifying features with interesting patterns...")

        positive_risky_delta = []
        negative_safe_stop = []
        strong_effects = []
        positive_avg_bet_delta = []

        for feature, data in self.results['detailed_effects'].items():
            effects = data['effects']
            stats = data['condition_stats']


            # Check for positive risky betting delta (safe patch increases risky betting in risky context)
            if ('risky_context' in effects and
                effects['risky_context']['safe_patch_risky_delta'] > 0.02):  # 2% threshold
                positive_risky_delta.append({
                    'feature': feature,
                    'risky_delta': effects['risky_context']['safe_patch_risky_delta'],
                    'stop_delta': effects['risky_context']['safe_patch_stop_delta'],
                    'layer': data['layer']
                })

            # Check for negative safe context stop rate effect (safe patch decreases stopping in safe context)
            if ('safe_context' in effects and
                effects['safe_context']['safe_patch_stop_delta'] < -0.02):  # -2% threshold
                negative_safe_stop.append({
                    'feature': feature,
                    'stop_delta': effects['safe_context']['safe_patch_stop_delta'],
                    'risky_delta': effects['safe_context'].get('safe_patch_risky_delta', 0),
                    'layer': data['layer']
                })

            # Check for betting amount effects (comparing average bets) - with safer access
            if ('risky_context' in effects and 'risky_baseline' in stats and 'risky_with_safe_patch' in stats):
                baseline_bet = stats['risky_baseline'].get('avg_bet', 0)
                safe_patch_bet = stats['risky_with_safe_patch'].get('avg_bet', 0)
                if baseline_bet > 0 and not np.isnan(baseline_bet) and not np.isnan(safe_patch_bet):
                    bet_delta = safe_patch_bet - baseline_bet
                    if abs(bet_delta) > 5:  # $5 difference threshold
                        positive_avg_bet_delta.append({
                            'feature': feature,
                            'bet_delta': bet_delta,
                            'baseline_bet': baseline_bet,
                            'safe_patch_bet': safe_patch_bet,
                            'layer': data['layer']
                        })

            # Check for any strong effects (absolute delta > 0.05)
            for context in ['safe_context', 'risky_context']:
                if context in effects:
                    for effect_name, value in effects[context].items():
                        if not np.isnan(value) and abs(value) > 0.05:  # 5% threshold for strong effects
                            strong_effects.append({
                                'feature': feature,
                                'effect_type': f"{context}_{effect_name}",
                                'delta': value,
                                'layer': data['layer']
                            })

        # Sort by effect size
        positive_risky_delta.sort(key=lambda x: x['risky_delta'], reverse=True)
        negative_safe_stop.sort(key=lambda x: x['stop_delta'])
        positive_avg_bet_delta.sort(key=lambda x: abs(x['bet_delta']), reverse=True)
        strong_effects.sort(key=lambda x: abs(x['delta']), reverse=True)

        self.results['positive_risky_delta'] = positive_risky_delta
        self.results['negative_safe_stop_delta'] = negative_safe_stop
        self.results['avg_bet_effects'] = positive_avg_bet_delta[:30]  # Top 30 betting effects
        self.results['strong_effects'] = strong_effects[:50]  # Top 50 strong effects

        logger.info(f"Found {len(positive_risky_delta)} features with positive risky betting delta")
        logger.info(f"Found {len(negative_safe_stop)} features with negative safe stop delta")
        logger.info(f"Found {len(positive_avg_bet_delta)} features with significant betting amount effects")
        logger.info(f"Found {len(strong_effects)} features with strong effects (>5%)")

    def calculate_summary_statistics(self):
        """Calculate overall summary statistics"""
        logger.info("Calculating summary statistics...")

        all_risky_deltas = []
        all_safe_stop_deltas = []
        all_bet_deltas = []

        layer_counts = defaultdict(int)
        effect_distributions = defaultdict(list)

        for feature, data in self.results['detailed_effects'].items():
            layer = data['layer']
            layer_counts[layer] += 1

            effects = data['effects']
            stats = data['condition_stats']

            if 'risky_context' in effects:
                risky_delta = effects['risky_context']['safe_patch_risky_delta']
                all_risky_deltas.append(risky_delta)
                effect_distributions['risky_bet_delta'].append(risky_delta)

            if 'safe_context' in effects:
                stop_delta = effects['safe_context']['safe_patch_stop_delta']
                all_safe_stop_deltas.append(stop_delta)
                effect_distributions['safe_stop_delta'].append(stop_delta)

            # Calculate betting amount deltas
            try:
                if ('risky_baseline' in stats and 'risky_with_safe_patch' in stats):
                    baseline_bet = stats['risky_baseline'].get('avg_bet', 0)
                    safe_patch_bet = stats['risky_with_safe_patch'].get('avg_bet', 0)
                    if (baseline_bet > 0 and
                        not np.isnan(baseline_bet) and
                        not np.isnan(safe_patch_bet)):
                        bet_delta = safe_patch_bet - baseline_bet
                        all_bet_deltas.append(bet_delta)
                        effect_distributions['avg_bet_delta'].append(bet_delta)
            except (KeyError, TypeError):
                continue

        self.results['summary_statistics'] = {
            'layer_distribution': dict(layer_counts),
            'risky_bet_delta_stats': {
                'mean': np.mean(all_risky_deltas) if all_risky_deltas else 0,
                'std': np.std(all_risky_deltas) if all_risky_deltas else 0,
                'min': np.min(all_risky_deltas) if all_risky_deltas else 0,
                'max': np.max(all_risky_deltas) if all_risky_deltas else 0,
                'positive_count': sum(1 for x in all_risky_deltas if x > 0),
                'negative_count': sum(1 for x in all_risky_deltas if x < 0)
            },
            'safe_stop_delta_stats': {
                'mean': np.mean(all_safe_stop_deltas) if all_safe_stop_deltas else 0,
                'std': np.std(all_safe_stop_deltas) if all_safe_stop_deltas else 0,
                'min': np.min(all_safe_stop_deltas) if all_safe_stop_deltas else 0,
                'max': np.max(all_safe_stop_deltas) if all_safe_stop_deltas else 0,
                'positive_count': sum(1 for x in all_safe_stop_deltas if x > 0),
                'negative_count': sum(1 for x in all_safe_stop_deltas if x < 0)
            },
            'avg_bet_delta_stats': {
                'mean': np.mean(all_bet_deltas) if all_bet_deltas else 0,
                'std': np.std(all_bet_deltas) if all_bet_deltas else 0,
                'min': np.min(all_bet_deltas) if all_bet_deltas else 0,
                'max': np.max(all_bet_deltas) if all_bet_deltas else 0,
                'positive_count': sum(1 for x in all_bet_deltas if x > 0),
                'negative_count': sum(1 for x in all_bet_deltas if x < 0)
            }
        }

    def save_results(self, output_path):
        """Save comprehensive analysis results"""
        logger.info(f"Saving results to {output_path}")

        # Add metadata
        self.results['analysis_metadata'] = {
            'script_version': '1.0',
            'analysis_date': '2025-09-17',
            'total_response_logs': self.results['total_files_processed'],
            'unique_features_analyzed': self.results['total_features'],
            'conditions_analyzed': ['safe_baseline', 'safe_with_safe_patch', 'safe_with_risky_patch', 'risky_baseline', 'risky_with_safe_patch', 'risky_with_risky_patch']
        }

        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        logger.info(f"Results saved successfully. Key findings:")
        logger.info(f"- Total features analyzed: {self.results['total_features']}")
        logger.info(f"- Features with positive risky bankruptcy delta: {len(self.results['positive_risky_bankruptcy_delta'])}")
        logger.info(f"- Features with negative safe stop delta: {len(self.results['negative_safe_stop_delta'])}")
        logger.info(f"- Features with strong effects: {len(self.results.get('strong_effects', []))}")

    def print_summary(self):
        """Print a summary of key findings"""
        print("\n" + "="*80)
        print("COMPREHENSIVE RESPONSE LOG ANALYSIS SUMMARY")
        print("="*80)

        print(f"Total response log files processed: {self.results['total_files_processed']}")
        print(f"Total unique features analyzed: {self.results['total_features']}")

        # Layer distribution
        print(f"\nLayer distribution:")
        for layer, count in sorted(self.results['summary_statistics']['layer_distribution'].items()):
            print(f"  {layer}: {count} features")

        # Risky betting delta findings
        risky_stats = self.results['summary_statistics']['risky_bet_delta_stats']
        print(f"\nRisky Context Betting Delta (safe patch effect on risky betting):")
        print(f"  Mean: {risky_stats['mean']:.4f}")
        print(f"  Std: {risky_stats['std']:.4f}")
        print(f"  Range: [{risky_stats['min']:.4f}, {risky_stats['max']:.4f}]")
        print(f"  Positive effects: {risky_stats['positive_count']}")
        print(f"  Negative effects: {risky_stats['negative_count']}")

        # Safe stop delta findings
        safe_stats = self.results['summary_statistics']['safe_stop_delta_stats']
        print(f"\nSafe Context Stop Delta (safe patch effect):")
        print(f"  Mean: {safe_stats['mean']:.4f}")
        print(f"  Std: {safe_stats['std']:.4f}")
        print(f"  Range: [{safe_stats['min']:.4f}, {safe_stats['max']:.4f}]")
        print(f"  Positive effects: {safe_stats['positive_count']}")
        print(f"  Negative effects: {safe_stats['negative_count']}")

        # Betting amount delta findings
        bet_stats = self.results['summary_statistics']['avg_bet_delta_stats']
        print(f"\nAverage Bet Amount Delta (safe patch effect):")
        print(f"  Mean: ${bet_stats['mean']:.2f}")
        print(f"  Std: ${bet_stats['std']:.2f}")
        print(f"  Range: [${bet_stats['min']:.2f}, ${bet_stats['max']:.2f}]")
        print(f"  Positive effects: {bet_stats['positive_count']}")
        print(f"  Negative effects: {bet_stats['negative_count']}")

        # Top positive risky betting delta features
        print(f"\nTop 10 Features with Positive Risky Betting Delta:")
        for i, item in enumerate(self.results['positive_risky_delta'][:10]):
            print(f"  {i+1:2d}. {item['feature']:12s} ({item['layer']}) Risky: +{item['risky_delta']:.3f}, Stop: {item['stop_delta']:.3f}")

        # Top negative safe stop delta features
        print(f"\nTop 10 Features with Negative Safe Stop Delta:")
        for i, item in enumerate(self.results['negative_safe_stop_delta'][:10]):
            print(f"  {i+1:2d}. {item['feature']:12s} ({item['layer']}) Stop: {item['stop_delta']:.3f}, Risky: +{item['risky_delta']:.3f}")

        # Top betting amount effects
        print(f"\nTop 10 Features with Largest Betting Amount Effects:")
        for i, item in enumerate(self.results['avg_bet_effects'][:10]):
            print(f"  {i+1:2d}. {item['feature']:12s} ({item['layer']}) Δ: ${item['bet_delta']:+.1f} (${item['baseline_bet']:.1f}→${item['safe_patch_bet']:.1f})")

        print("\n" + "="*80)
        print(f"ANSWER TO KEY QUESTION:")
        print(f"Features with positive risky betting delta (>2%): {len(self.results['positive_risky_delta'])}")
        print(f"Features with significant betting amount effects (>$5): {len(self.results['avg_bet_effects'])}")
        print(f"This analysis covers ALL 886 features across layers 25-31")
        print("="*80)

def main():
    """Main analysis function"""
    logger.info("Starting comprehensive response log analysis...")

    analyzer = ComprehensiveResponseAnalyzer()

    # Load all response logs
    analyzer.load_all_response_logs()

    # Calculate feature effects
    analyzer.calculate_feature_effects()

    # Identify interesting patterns
    analyzer.identify_interesting_features()

    # Calculate summary statistics
    analyzer.calculate_summary_statistics()

    # Save results
    output_path = '/home/ubuntu/llm_addiction/analysis/comprehensive_response_analysis_results.json'
    analyzer.save_results(output_path)

    # Print summary
    analyzer.print_summary()

    logger.info("Analysis complete!")

if __name__ == "__main__":
    main()