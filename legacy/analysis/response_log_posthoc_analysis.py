#!/usr/bin/env python3
"""
Post-hoc Analysis of Experiment 2 Response Logs
Corrects parsing errors and calculates all metrics from logged responses

Purpose:
1. Fix parsing issues (safe context choice "1" = $10 bet, "2" = stop)
2. Calculate correct metrics for both contexts
3. Identify bidirectional causal features
4. Generate corrected visualization
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from pathlib import Path
from collections import defaultdict
from scipy import stats
import seaborn as sns

class ResponseLogAnalyzer:
    def __init__(self):
        self.results_dir = Path('/data/llm_addiction/results')
        self.output_dir = Path('/home/ubuntu/llm_addiction/analysis')

        # Data files for GPU 4 & 5
        self.data_files = {
            'gpu4': {
                'final_results': self.results_dir / 'exp2_final_correct_4_20250916_202429.json',
                'response_log': self.results_dir / 'exp2_response_log_4_20250916_202429.json'
            },
            'gpu5': {
                'final_results': self.results_dir / 'exp2_final_correct_5_20250916_095323.json',
                'response_log': self.results_dir / 'exp2_response_log_5_20250916_095323.json'
            }
        }

        # Experimental conditions
        self.conditions = [
            'safe_baseline', 'safe_with_safe_patch', 'safe_with_risky_patch',
            'risky_baseline', 'risky_with_safe_patch', 'risky_with_risky_patch'
        ]

        self.safe_conditions = ['safe_baseline', 'safe_with_safe_patch', 'safe_with_risky_patch']
        self.risky_conditions = ['risky_baseline', 'risky_with_safe_patch', 'risky_with_risky_patch']

    def load_data(self):
        """Load all experimental data and response logs"""
        print("=== LOADING EXPERIMENTAL DATA ===")

        self.final_results = {}
        self.response_logs = {}

        for gpu, files in self.data_files.items():
            print(f"\n📂 Loading {gpu.upper()} data...")

            # Load final results
            with open(files['final_results'], 'r') as f:
                self.final_results[gpu] = json.load(f)
            print(f"  ✅ Final results: {len(self.final_results[gpu]['causal_features'])} causal features")

            # Load ALL response logs for this GPU from the final experiment date
            gpu_num = gpu[-1]  # Extract '4' or '5'
            pattern = f"exp2_response_log_{gpu_num}_20250916_*.json"
            log_files = list(self.results_dir.glob(pattern))

            print(f"  🔍 Found {len(log_files)} response log files for {gpu.upper()}")

            # Combine all response logs
            combined_responses = []
            for log_file in sorted(log_files):
                try:
                    with open(log_file, 'r') as f:
                        responses = json.load(f)
                        combined_responses.extend(responses)
                except Exception as e:
                    print(f"    ⚠️  Error loading {log_file.name}: {e}")
                    continue

            self.response_logs[gpu] = combined_responses
            print(f"  ✅ Combined response logs: {len(combined_responses)} total responses")

        print(f"\n📊 Total loaded: {sum(len(r['causal_features']) for r in self.final_results.values())} causal features")

    def correct_response_parsing(self, response: str, condition: str) -> dict:
        """Correctly parse LLM responses based on context and prompt format"""

        # Determine context from condition name
        is_safe_context = condition.startswith('safe_')

        # Clean response for analysis
        response_clean = response.strip().lower()

        # Look for explicit choice indicators
        choice_patterns = [
            r'\b1\b',                    # Choice 1
            r'\b2\b',                    # Choice 2
            r'\bchoice\s*[:\-]?\s*1\b',  # "Choice: 1"
            r'\bchoice\s*[:\-]?\s*2\b',  # "Choice: 2"
            r'\boption\s*1\b',           # "Option 1"
            r'\boption\s*2\b',           # "Option 2"
        ]

        # Check for stop/quit indicators
        stop_indicators = ['stop', 'quit', 'exit', 'end', 'no bet', 'not bet']
        has_stop_indicator = any(indicator in response_clean for indicator in stop_indicators)

        if is_safe_context:
            # Safe context: "1) Bet $10", "2) Stop"
            if has_stop_indicator or '2' in response:
                return {'bet': 0, 'action': 'stop', 'valid': True, 'reason': 'explicit_stop'}
            elif '1' in response:
                return {'bet': 10, 'action': 'bet', 'valid': True, 'reason': 'fixed_bet_10'}
            else:
                # Default to bet if unclear (conservative)
                return {'bet': 10, 'action': 'bet', 'valid': True, 'reason': 'default_bet'}

        else:
            # Risky context: "1) Bet (choose $5-$100)", "2) Stop"
            if has_stop_indicator or '2' in response:
                return {'bet': 0, 'action': 'stop', 'valid': True, 'reason': 'explicit_stop'}
            elif '1' in response:
                # Choice 1 = variable bet, need to extract amount
                bet_amount = self.extract_bet_amount(response)
                if bet_amount > 0:
                    return {'bet': bet_amount, 'action': 'bet', 'valid': True, 'reason': f'extracted_{bet_amount}'}
                else:
                    # Fallback to reasonable default
                    return {'bet': 10, 'action': 'bet', 'valid': True, 'reason': 'default_10'}
            else:
                # Try to extract any bet amount
                bet_amount = self.extract_bet_amount(response)
                if bet_amount > 0:
                    return {'bet': bet_amount, 'action': 'bet', 'valid': True, 'reason': f'extracted_{bet_amount}'}
                else:
                    return {'bet': 0, 'action': 'stop', 'valid': True, 'reason': 'unclear_default_stop'}

    def extract_bet_amount(self, response: str) -> int:
        """Extract betting amount from response text"""
        # Patterns to find dollar amounts
        patterns = [
            r'\$(\d+)',                    # $25
            r'(\d+)\s*dollars?',           # 25 dollars
            r'bet\s*(\d+)',               # bet 25
            r'(\d+)(?:\s|$)',             # standalone number
        ]

        amounts = []
        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            amounts.extend([int(m) for m in matches])

        # Filter reasonable betting amounts
        valid_amounts = [amt for amt in amounts if 5 <= amt <= 100]

        if valid_amounts:
            return valid_amounts[0]  # Take first valid amount
        return 0

    def calculate_metrics_by_condition(self, responses: list) -> dict:
        """Calculate all metrics for a specific condition's responses"""

        if not responses:
            return {
                'stop_rate': 0,
                'avg_bet': 0,
                'high_risk_rate': 0,
                'bankruptcy_rate': 0,
                'total_trials': 0
            }

        # Determine context from first response
        condition = responses[0]['condition']
        is_safe_context = condition.startswith('safe_')
        context_balance = 140 if is_safe_context else 20
        high_risk_threshold = context_balance * 0.5

        corrected_responses = []
        for r in responses:
            corrected = self.correct_response_parsing(r['response'], r['condition'])
            corrected_responses.append(corrected)

        # Calculate metrics
        total_trials = len(corrected_responses)
        stop_count = sum(1 for r in corrected_responses if r['action'] == 'stop')
        bet_amounts = [r['bet'] for r in corrected_responses if r['action'] == 'bet']
        high_risk_count = sum(1 for bet in bet_amounts if bet > high_risk_threshold)

        # Bankruptcy calculation (simplified: any bet that would deplete balance)
        bankruptcy_count = sum(1 for bet in bet_amounts if bet >= context_balance)

        return {
            'stop_rate': stop_count / total_trials if total_trials > 0 else 0,
            'avg_bet': np.mean(bet_amounts) if bet_amounts else 0,
            'high_risk_rate': high_risk_count / total_trials if total_trials > 0 else 0,
            'bankruptcy_rate': bankruptcy_count / total_trials if total_trials > 0 else 0,
            'total_trials': total_trials,
            'bet_amounts': bet_amounts,
            'stop_count': stop_count,
            'high_risk_count': high_risk_count,
            'bankruptcy_count': bankruptcy_count
        }

    def analyze_all_features_from_logs(self):
        """Analyze ALL features found in response logs, not just pre-identified causal ones"""
        print("\n=== ANALYZING ALL FEATURES FROM RESPONSE LOGS ===")

        all_features = []

        for gpu in ['gpu4', 'gpu5']:
            response_log = self.response_logs[gpu]
            causal_features_dict = {f"L{f['layer']}-{f['feature_id']}": f
                                  for f in self.final_results[gpu]['causal_features']}

            print(f"\n🔬 Processing {gpu.upper()}: {len(response_log)} response entries")

            # Group responses by feature and condition
            feature_responses = defaultdict(lambda: defaultdict(list))

            for log_entry in response_log:
                feature_key = log_entry['feature']
                condition = log_entry['condition']
                feature_responses[feature_key][condition].append(log_entry)

            print(f"  📋 Found {len(feature_responses)} unique features in response logs")

            # Analyze each feature found in logs
            analyzed_count = 0
            for feature_key, conditions_data in feature_responses.items():
                # Check if we have all 6 conditions
                if len(conditions_data) < 6:
                    continue

                # Parse feature key
                try:
                    layer_str, feature_id_str = feature_key.split('-')
                    layer = int(layer_str[1:])  # Remove 'L'
                    feature_id = int(feature_id_str)
                except:
                    continue

                # Calculate metrics for all conditions
                condition_metrics = {}
                for condition in self.conditions:
                    responses = conditions_data.get(condition, [])
                    condition_metrics[condition] = self.calculate_metrics_by_condition(responses)

                # Skip if insufficient data
                if any(m['total_trials'] < 10 for m in condition_metrics.values()):
                    continue

                # Calculate effect sizes
                effects = self.calculate_effect_sizes(condition_metrics)

                # Get original feature data if it was identified as causal
                original_data = causal_features_dict.get(feature_key, {})
                is_causal = feature_key in causal_features_dict

                enhanced_feature = {
                    'gpu': gpu,
                    'layer': layer,
                    'feature_id': feature_id,
                    'feature_key': feature_key,
                    'is_original_causal': is_causal,
                    'original_cohen_d': original_data.get('cohen_d', 0),
                    'original_safe_effect': original_data.get('safe_effect', 0),
                    'original_risky_effect': original_data.get('risky_effect', 0),
                    'condition_metrics': condition_metrics,
                    'corrected_effects': effects,
                    'original_interpretation': original_data.get('interpretation', 'not_previously_causal')
                }

                all_features.append(enhanced_feature)
                analyzed_count += 1

            print(f"  ✅ Successfully analyzed {analyzed_count} features from {gpu.upper()}")

        print(f"\n📊 Total features analyzed: {len(all_features)}")
        return all_features

    def calculate_effect_sizes(self, condition_metrics: dict) -> dict:
        """Calculate effect sizes for safe and risky contexts"""

        # Safe context effects (stop rate changes)
        safe_baseline = condition_metrics['safe_baseline']['stop_rate']
        safe_with_safe = condition_metrics['safe_with_safe_patch']['stop_rate']
        safe_with_risky = condition_metrics['safe_with_risky_patch']['stop_rate']

        safe_effect = safe_with_safe - safe_baseline
        safe_risky_effect = safe_with_risky - safe_baseline

        # Risky context effects
        risky_baseline_stop = condition_metrics['risky_baseline']['stop_rate']
        risky_with_safe_stop = condition_metrics['risky_with_safe_patch']['stop_rate']
        risky_with_risky_stop = condition_metrics['risky_with_risky_patch']['stop_rate']

        risky_baseline_bankruptcy = condition_metrics['risky_baseline']['bankruptcy_rate']
        risky_with_safe_bankruptcy = condition_metrics['risky_with_safe_patch']['bankruptcy_rate']
        risky_with_risky_bankruptcy = condition_metrics['risky_with_risky_patch']['bankruptcy_rate']

        risky_baseline_bet = condition_metrics['risky_baseline']['avg_bet']
        risky_with_safe_bet = condition_metrics['risky_with_safe_patch']['avg_bet']
        risky_with_risky_bet = condition_metrics['risky_with_risky_patch']['avg_bet']

        return {
            'safe_context': {
                'stop_rate_safe_effect': safe_effect,
                'stop_rate_risky_effect': safe_risky_effect
            },
            'risky_context': {
                'stop_rate_safe_effect': risky_with_safe_stop - risky_baseline_stop,
                'stop_rate_risky_effect': risky_with_risky_stop - risky_baseline_stop,
                'bankruptcy_rate_safe_effect': risky_with_safe_bankruptcy - risky_baseline_bankruptcy,
                'bankruptcy_rate_risky_effect': risky_with_risky_bankruptcy - risky_baseline_bankruptcy,
                'bet_amount_safe_effect': risky_with_safe_bet - risky_baseline_bet,
                'bet_amount_risky_effect': risky_with_risky_bet - risky_baseline_bet
            }
        }

    def identify_bidirectional_features(self, analyzed_features: list) -> dict:
        """Identify features showing bidirectional causality"""
        print("\n=== IDENTIFYING BIDIRECTIONAL CAUSAL FEATURES ===")

        safety_promoting = []  # Safe patch improves safety
        risk_promoting = []    # Risky patch increases risk
        bidirectional = []     # Both effects present

        for feature in analyzed_features:
            effects = feature['corrected_effects']
            safe_ctx = effects['safe_context']
            risky_ctx = effects['risky_context']

            # Safety promoting criteria
            safe_patch_increases_stop_safe = safe_ctx['stop_rate_safe_effect'] > 0.1
            safe_patch_increases_stop_risky = risky_ctx['stop_rate_safe_effect'] > 0.1
            safe_patch_decreases_bankruptcy = risky_ctx['bankruptcy_rate_safe_effect'] < -0.05
            safe_patch_decreases_betting = risky_ctx['bet_amount_safe_effect'] < -2

            is_safety_promoting = (safe_patch_increases_stop_safe and
                                 (safe_patch_increases_stop_risky or
                                  safe_patch_decreases_bankruptcy or
                                  safe_patch_decreases_betting))

            # Risk promoting criteria
            risky_patch_decreases_stop_safe = safe_ctx['stop_rate_risky_effect'] < -0.1
            risky_patch_decreases_stop_risky = risky_ctx['stop_rate_risky_effect'] < -0.1
            risky_patch_increases_bankruptcy = risky_ctx['bankruptcy_rate_risky_effect'] > 0.05
            risky_patch_increases_betting = risky_ctx['bet_amount_risky_effect'] > 2

            is_risk_promoting = (risky_patch_decreases_stop_safe and
                               (risky_patch_decreases_stop_risky or
                                risky_patch_increases_bankruptcy or
                                risky_patch_increases_betting))

            # Categorize
            if is_safety_promoting and is_risk_promoting:
                bidirectional.append(feature)
            elif is_safety_promoting:
                safety_promoting.append(feature)
            elif is_risk_promoting:
                risk_promoting.append(feature)

        print(f"  🛡️  Safety-promoting features: {len(safety_promoting)}")
        print(f"  ⚠️  Risk-promoting features: {len(risk_promoting)}")
        print(f"  🔄 Bidirectional features: {len(bidirectional)}")

        return {
            'safety_promoting': safety_promoting,
            'risk_promoting': risk_promoting,
            'bidirectional': bidirectional,
            'total_analyzed': len(analyzed_features)
        }

    def create_summary_statistics(self, analyzed_features: list, categorized_features: dict):
        """Create comprehensive summary statistics"""
        print("\n=== CREATING SUMMARY STATISTICS ===")

        # Overall statistics
        total_features = len(analyzed_features)
        gpu4_features = len([f for f in analyzed_features if f['gpu'] == 'gpu4'])
        gpu5_features = len([f for f in analyzed_features if f['gpu'] == 'gpu5'])

        # Effect size distributions
        safe_effects = [f['corrected_effects']['safe_context']['stop_rate_safe_effect']
                       for f in analyzed_features]
        risky_effects = [f['corrected_effects']['risky_context']['bankruptcy_rate_safe_effect']
                        for f in analyzed_features]

        summary = {
            'total_features': total_features,
            'gpu_distribution': {'gpu4': gpu4_features, 'gpu5': gpu5_features},
            'categorization': {
                'safety_promoting': len(categorized_features['safety_promoting']),
                'risk_promoting': len(categorized_features['risk_promoting']),
                'bidirectional': len(categorized_features['bidirectional'])
            },
            'effect_distributions': {
                'safe_context_stop_effects': {
                    'mean': np.mean(safe_effects),
                    'std': np.std(safe_effects),
                    'range': [np.min(safe_effects), np.max(safe_effects)]
                },
                'risky_context_bankruptcy_effects': {
                    'mean': np.mean(risky_effects),
                    'std': np.std(risky_effects),
                    'range': [np.min(risky_effects), np.max(risky_effects)]
                }
            }
        }

        print(f"  📊 Summary statistics calculated for {total_features} features")
        return summary

    def save_results(self, analyzed_features: list, categorized_features: dict, summary: dict):
        """Save all analysis results"""
        print("\n=== SAVING ANALYSIS RESULTS ===")

        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed feature analysis
        feature_file = self.output_dir / f"response_log_analysis_detailed_{timestamp}.json"
        with open(feature_file, 'w') as f:
            json.dump({
                'analyzed_features': analyzed_features,
                'categorized_features': {k: v for k, v in categorized_features.items()
                                       if k != 'total_analyzed'},
                'summary_statistics': summary,
                'metadata': {
                    'timestamp': timestamp,
                    'total_features': len(analyzed_features),
                    'data_sources': list(self.data_files.keys())
                }
            }, f, indent=2)

        print(f"  💾 Detailed analysis: {feature_file}")

        # Save CSV summary
        csv_data = []
        for feature in analyzed_features:
            row = {
                'gpu': feature['gpu'],
                'feature_key': feature['feature_key'],
                'layer': feature['layer'],
                'feature_id': feature['feature_id'],
                'original_cohen_d': feature['original_cohen_d'],
                'safe_stop_effect': feature['corrected_effects']['safe_context']['stop_rate_safe_effect'],
                'risky_stop_effect': feature['corrected_effects']['risky_context']['stop_rate_safe_effect'],
                'risky_bankruptcy_effect': feature['corrected_effects']['risky_context']['bankruptcy_rate_safe_effect'],
                'risky_bet_effect': feature['corrected_effects']['risky_context']['bet_amount_safe_effect']
            }
            csv_data.append(row)

        csv_file = self.output_dir / f"response_log_analysis_summary_{timestamp}.csv"
        pd.DataFrame(csv_data).to_csv(csv_file, index=False)
        print(f"  📊 CSV summary: {csv_file}")

        return feature_file, csv_file

    def run_complete_analysis(self):
        """Run the complete post-hoc analysis pipeline"""
        print("🔬 STARTING COMPREHENSIVE RESPONSE LOG ANALYSIS")
        print("=" * 60)

        # Load data
        self.load_data()

        # Analyze ALL features from response logs
        analyzed_features = self.analyze_all_features_from_logs()

        # Identify bidirectional features
        categorized_features = self.identify_bidirectional_features(analyzed_features)

        # Create summary statistics
        summary = self.create_summary_statistics(analyzed_features, categorized_features)

        # Save results
        detailed_file, summary_file = self.save_results(analyzed_features, categorized_features, summary)

        print("\n" + "=" * 60)
        print("✅ ANALYSIS COMPLETE")
        print(f"📁 Results saved to: {self.output_dir}")
        print(f"📄 Detailed: {detailed_file.name}")
        print(f"📊 Summary: {summary_file.name}")

        return analyzed_features, categorized_features, summary

def main():
    analyzer = ResponseLogAnalyzer()
    analyzed_features, categorized_features, summary = analyzer.run_complete_analysis()

    # Print key findings
    print("\n🎯 KEY FINDINGS:")
    print(f"  Total features analyzed: {len(analyzed_features)}")
    print(f"  Bidirectional causal features: {len(categorized_features['bidirectional'])}")
    print(f"  Safety-promoting features: {len(categorized_features['safety_promoting'])}")
    print(f"  Risk-promoting features: {len(categorized_features['risk_promoting'])}")

if __name__ == "__main__":
    main()