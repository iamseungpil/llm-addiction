#!/usr/bin/env python3
"""
Phase 2: Behavioral Analysis for Coin Flip Experiment

Analyzes game results and computes behavioral metrics across conditions.

Usage:
    python src/coin_flip/phase2_behavioral_analysis.py --input results.json
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from common import setup_logger, save_json

logger = setup_logger(__name__)


class BehavioralAnalyzer:
    """Behavioral analysis for Coin Flip experiments"""

    def __init__(self, results_file: Path):
        """
        Initialize analyzer.

        Args:
            results_file: Path to results JSON file
        """
        self.results_file = results_file
        self.results = self.load_results()

    def load_results(self) -> Dict:
        """Load results from JSON file"""
        logger.info(f"Loading results from: {self.results_file}")

        with open(self.results_file, 'r') as f:
            data = json.load(f)

        if 'game_results' in data:
            return data['game_results']
        else:
            return data

    def analyze_condition(self, condition_key: str, games: List[Dict]) -> Dict:
        """
        Analyze a single condition.

        Args:
            condition_key: Condition identifier
            games: List of game results

        Returns:
            Analysis results dictionary
        """
        num_games = len(games)
        metrics = {
            'condition': condition_key,
            'num_games': num_games,
            'bankruptcy_rate': 0.0,
            'voluntary_stop_rate': 0.0,
            'max_rounds_rate': 0.0,
            'avg_rounds': 0.0,
            'avg_final_chips': 0.0,
            'bet_ratios': [],
            'extreme_choices': [],
            'loss_chasing': []
        }

        total_rounds = 0
        end_reasons = {'bankrupt': 0, 'voluntary_stop': 0, 'max_rounds': 0}

        for game in games:
            # End reasons
            if 'end_reason' in game:
                end_reasons[game['end_reason']] += 1
            elif game['bankrupt']:
                end_reasons['bankrupt'] += 1

            # Round stats
            total_rounds += game['num_rounds']

            # Final chips
            metrics['avg_final_chips'] += game['final_chips']

            # Per-round analysis
            rounds = game.get('rounds', [])
            for i, round_data in enumerate(rounds):
                bet = round_data['bet']
                chips_before = round_data['chips_before']

                # Bet ratio
                if chips_before > 0:
                    metrics['bet_ratios'].append(bet / chips_before)

                # Extreme choice (assuming max_bet from metadata)
                # We'll calculate from data
                if i > 0:
                    prev_round = rounds[i-1]
                    # Loss chasing
                    if prev_round['outcome'] == 'lose' and bet > prev_round['bet']:
                        metrics['loss_chasing'].append(1)
                    else:
                        metrics['loss_chasing'].append(0)

        # Calculate averages
        metrics['bankruptcy_rate'] = end_reasons['bankrupt'] / num_games
        metrics['voluntary_stop_rate'] = end_reasons['voluntary_stop'] / num_games
        metrics['max_rounds_rate'] = end_reasons['max_rounds'] / num_games
        metrics['avg_rounds'] = total_rounds / num_games
        metrics['avg_final_chips'] /= num_games

        # Behavioral metrics
        if metrics['bet_ratios']:
            metrics['avg_bet_ratio'] = np.mean(metrics['bet_ratios'])
            metrics['std_bet_ratio'] = np.std(metrics['bet_ratios'])

        if metrics['loss_chasing']:
            metrics['loss_chase_rate'] = np.mean(metrics['loss_chasing'])

        return metrics

    def compare_conditions(self, metrics: Dict[str, Dict]) -> Dict:
        """
        Statistical comparison between conditions.

        Args:
            metrics: Metrics for all conditions

        Returns:
            Comparison results
        """
        comparisons = {}

        conditions = list(metrics.keys())
        for i, cond1 in enumerate(conditions):
            for cond2 in conditions[i+1:]:
                key = f"{cond1}_vs_{cond2}"

                # Compare bankruptcy rates
                # (Using binomial test or chi-square)
                n1 = metrics[cond1]['num_games']
                n2 = metrics[cond2]['num_games']
                k1 = int(metrics[cond1]['bankruptcy_rate'] * n1)
                k2 = int(metrics[cond2]['bankruptcy_rate'] * n2)

                # Chi-square test
                contingency = [[k1, n1-k1], [k2, n2-k2]]
                chi2, p_value = stats.chi2_contingency(contingency)[:2]

                comparisons[key] = {
                    'bankruptcy_diff': metrics[cond1]['bankruptcy_rate'] - metrics[cond2]['bankruptcy_rate'],
                    'bankruptcy_p_value': p_value,
                    'bankruptcy_significant': p_value < 0.05
                }

                # Compare bet ratios (t-test)
                if metrics[cond1]['bet_ratios'] and metrics[cond2]['bet_ratios']:
                    t_stat, t_p = stats.ttest_ind(
                        metrics[cond1]['bet_ratios'],
                        metrics[cond2]['bet_ratios']
                    )
                    comparisons[key]['bet_ratio_t_stat'] = t_stat
                    comparisons[key]['bet_ratio_p_value'] = t_p

        return comparisons

    def run_analysis(self) -> Dict:
        """
        Run full behavioral analysis.

        Returns:
            Complete analysis results
        """
        logger.info("Running behavioral analysis...")

        metrics = {}
        for condition_key, games in self.results.items():
            logger.info(f"Analyzing condition: {condition_key}")
            metrics[condition_key] = self.analyze_condition(condition_key, games)

        # Compare conditions
        comparisons = self.compare_conditions(metrics)

        # Summary
        self.print_summary(metrics, comparisons)

        return {
            'metrics': metrics,
            'comparisons': comparisons
        }

    def print_summary(self, metrics: Dict, comparisons: Dict):
        """Print analysis summary"""
        logger.info("\n" + "="*60)
        logger.info("BEHAVIORAL ANALYSIS SUMMARY")
        logger.info("="*60)

        for condition_key, cond_metrics in metrics.items():
            logger.info(f"\nCondition: {condition_key}")
            logger.info(f"  Games: {cond_metrics['num_games']}")
            logger.info(f"  Bankruptcy rate: {cond_metrics['bankruptcy_rate']*100:.1f}%")
            logger.info(f"  Voluntary stop rate: {cond_metrics['voluntary_stop_rate']*100:.1f}%")
            logger.info(f"  Avg rounds: {cond_metrics['avg_rounds']:.1f}")
            logger.info(f"  Avg final chips: ${cond_metrics['avg_final_chips']:.1f}")
            logger.info(f"  Avg bet ratio: {cond_metrics.get('avg_bet_ratio', 0):.3f}")
            logger.info(f"  Loss chase rate: {cond_metrics.get('loss_chase_rate', 0)*100:.1f}%")

        logger.info("\n" + "="*60)
        logger.info("CONDITION COMPARISONS")
        logger.info("="*60)

        for comp_key, comp_data in comparisons.items():
            logger.info(f"\n{comp_key}:")
            logger.info(f"  Bankruptcy diff: {comp_data['bankruptcy_diff']*100:.1f}%")
            logger.info(f"  p-value: {comp_data['bankruptcy_p_value']:.4f}")
            logger.info(f"  Significant: {comp_data['bankruptcy_significant']}")


def main():
    parser = argparse.ArgumentParser(description="Coin Flip Behavioral Analysis")
    parser.add_argument('--input', type=str, required=True,
                        help="Input results JSON file")
    parser.add_argument('--output-dir', type=str, default=None,
                        help="Output directory for analysis results")

    args = parser.parse_args()

    results_file = Path(args.input)
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    # Run analysis
    analyzer = BehavioralAnalyzer(results_file)
    analysis = analyzer.run_analysis()

    # Save analysis results
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_file.parent

    output_file = output_dir / f"behavioral_analysis_{results_file.stem}.json"
    save_json(analysis, output_file)
    logger.info(f"\nAnalysis saved to: {output_file}")


if __name__ == '__main__':
    main()
