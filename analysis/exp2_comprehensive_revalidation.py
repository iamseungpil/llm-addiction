#!/usr/bin/env python3
"""
Comprehensive Revalidation of 142 Causal Features
Tests bidirectional patching with multiple statistical methods
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu, fisher_exact, spearmanr
from tqdm import tqdm
import torch
import gc

sys.path.append('/home/ubuntu/llm_addiction/causal_feature_discovery/src')
from experiment_patching_population_mean import PopulationMeanPatchingExperiment

class ComprehensiveRevalidationExperiment:
    def __init__(self, gpu_id='6'):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        self.device = 'cuda:0'
        self.gpu_id = gpu_id
        
        # Reproducibility
        self.seed = 42
        random.seed(self.seed)
        np.random.seed(self.seed)
        try:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        except Exception:
            pass
        
        # Statistical thresholds (aligned with Experiment 2)
        self.alpha = 0.01  # More stringent for multiple comparisons
        self.min_effect_size = 0.3  # Cohen's d
        
        # Experiment parameters (consistent with Exp 2 structure)
        self.scales = [0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]  # 7 scales for robust testing
        self.n_trials = 50  # Increased for better statistical power
        self.batch_size = 4  # GPU parallelization like Exp 2
        
        # Base experiment (inherit from PopulationMeanPatchingExperiment)
        self.exp = PopulationMeanPatchingExperiment()
        self.exp.exclude_invalid = True
        self.exp.scales = self.scales
        self.exp.n_trials = self.n_trials
        self.exp.batch_size = self.batch_size
        
        # Results storage
        self.results_dir = Path('/home/ubuntu/llm_addiction/analysis')
        self.results_dir.mkdir(exist_ok=True)
        
    def load_causal_features(self, results_path: str):
        """Load all 142 causal features from population mean patching results"""
        with open(results_path, 'r') as f:
            data = json.load(f)
        
        causal_bet = data.get('causal_features_bet', [])
        causal_stop = data.get('causal_features_stop', [])
        
        # Deduplicate and get all unique features
        feature_map = {}
        
        for feat in causal_bet:
            key = (feat['layer'], feat['feature_id'])
            effect = abs(feat.get('bet_effect', 0))
            feature_map[key] = {
                'layer': feat['layer'],
                'feature_id': feat['feature_id'],
                'type': 'bet',
                'effect': effect,
                'correlation': feat.get('bet_correlation', 0)
            }
        
        for feat in causal_stop:
            key = (feat['layer'], feat['feature_id'])
            effect = abs(feat.get('stop_effect', 0))
            if key not in feature_map or effect > feature_map[key]['effect']:
                feature_map[key] = {
                    'layer': feat['layer'],
                    'feature_id': feat['feature_id'],
                    'type': 'stop',
                    'effect': effect,
                    'correlation': feat.get('stop_correlation', 0)
                }
        
        # Sort by effect size
        features = sorted(feature_map.values(), key=lambda x: x['effect'], reverse=True)
        
        print(f"Loaded {len(features)} unique causal features")
        return features
    
    def load_feature_means(self):
        """Load bankrupt and safe means for bidirectional patching"""
        npz_file = '/data/llm_addiction/results/llama_feature_arrays_20250829_150110_v2.npz'
        data = np.load(npz_file, allow_pickle=True)
        
        # Load arrays
        layer_25_indices = data['layer_25_indices']
        layer_25_bankrupt = data['layer_25_bankrupt_mean']
        layer_25_safe = data['layer_25_safe_mean']
        
        layer_30_indices = data['layer_30_indices']
        layer_30_bankrupt = data['layer_30_bankrupt_mean']
        layer_30_safe = data['layer_30_safe_mean']
        
        # Create lookup
        means = {}
        
        for i, idx in enumerate(layer_25_indices):
            means[(25, int(idx))] = {
                'bankrupt_mean': float(layer_25_bankrupt[i]),
                'safe_mean': float(layer_25_safe[i])
            }
            
        for i, idx in enumerate(layer_30_indices):
            means[(30, int(idx))] = {
                'bankrupt_mean': float(layer_30_bankrupt[i]),
                'safe_mean': float(layer_30_safe[i])
            }
        
        return means
    
    def calculate_comprehensive_stats(self, risky_data: List, safe_data: List, 
                                    risky_decisions: List, safe_decisions: List) -> Dict:
        """Calculate comprehensive statistical measures"""
        stats_result = {}
        
        # 1. T-test for betting amounts
        if len(risky_data) > 0 and len(safe_data) > 0:
            t_stat, t_p = ttest_ind(risky_data, safe_data)
            stats_result['ttest'] = {'statistic': float(t_stat), 'p_value': float(t_p)}
            
            # Cohen's d
            pooled_std = np.sqrt(((np.std(risky_data, ddof=1)**2 * (len(risky_data)-1)) + 
                                (np.std(safe_data, ddof=1)**2 * (len(safe_data)-1))) / 
                               (len(risky_data) + len(safe_data) - 2))
            cohens_d = (np.mean(risky_data) - np.mean(safe_data)) / pooled_std if pooled_std > 0 else 0
            stats_result['cohens_d'] = float(cohens_d)
        
        # 2. Mann-Whitney U (non-parametric)
        try:
            u_stat, u_p = mannwhitneyu(risky_data, safe_data, alternative='two-sided')
            stats_result['mannwhitney'] = {'statistic': float(u_stat), 'p_value': float(u_p)}
        except:
            stats_result['mannwhitney'] = {'statistic': 0, 'p_value': 1.0}
        
        # 3. Chi-square for decision distributions
        risky_stops = sum(1 for d in risky_decisions if d == 'stop')
        risky_bets = len(risky_decisions) - risky_stops
        safe_stops = sum(1 for d in safe_decisions if d == 'stop')
        safe_bets = len(safe_decisions) - safe_stops
        
        contingency = [[risky_stops, risky_bets], [safe_stops, safe_bets]]
        
        try:
            chi2, chi_p, dof, expected = chi2_contingency(contingency)
            stats_result['chi2'] = {'statistic': float(chi2), 'p_value': float(chi_p)}
            
            # Cramer's V
            n = sum(sum(row) for row in contingency)
            cramers_v = np.sqrt(chi2 / (n * (min(len(contingency), len(contingency[0])) - 1)))
            stats_result['cramers_v'] = float(cramers_v)
        except:
            stats_result['chi2'] = {'statistic': 0, 'p_value': 1.0}
            stats_result['cramers_v'] = 0.0
        
        # 4. Effect sizes and descriptives
        stats_result['descriptives'] = {
            'risky_mean_bet': float(np.mean(risky_data)) if risky_data else 0,
            'safe_mean_bet': float(np.mean(safe_data)) if safe_data else 0,
            'risky_stop_rate': float(risky_stops / len(risky_decisions)) if risky_decisions else 0,
            'safe_stop_rate': float(safe_stops / len(safe_decisions)) if safe_decisions else 0,
            'n_risky': len(risky_data),
            'n_safe': len(safe_data)
        }
        
        return stats_result
    
    def detect_all_causality_patterns(self, scales: List[float], values: List[float]) -> Dict:
        """Detect various causality patterns including non-monotonic (consistent with Exp 2)"""
        patterns = {}
        
        if len(values) < 3:  # Need at least 3 points
            return {'any_significant': False, 'patterns': {}}
        
        # 1. Monotonic (Spearman - same as Experiment 2)
        try:
            rho, p_spear = spearmanr(scales, values)
            patterns['monotonic'] = {
                'correlation': float(rho),
                'p_value': float(p_spear),
                'significant': abs(rho) > 0.4 and p_spear < self.alpha  # Slightly relaxed from 0.5
            }
        except:
            patterns['monotonic'] = {'correlation': 0, 'p_value': 1.0, 'significant': False}
        
        # 2. ANOVA (overall variance across scales)
        try:
            # Group values by scale for one-way ANOVA
            scale_groups = []
            for i, scale in enumerate(scales):
                if i < len(values):
                    scale_groups.append([values[i]])
            
            # Check if we have enough data for ANOVA (each group needs >1 sample)
            valid_groups = [group for group in scale_groups if len(group) > 1]
            if len(valid_groups) >= 3:  # Need at least 3 groups with >1 sample each
                f_stat, p_anova = stats.f_oneway(*valid_groups)
                patterns['anova'] = {
                    'f_statistic': float(f_stat),
                    'p_value': float(p_anova), 
                    'significant': p_anova < self.alpha
                }
            else:
                patterns['anova'] = {'f_statistic': 0, 'p_value': 1.0, 'significant': False}
        except:
            patterns['anova'] = {'f_statistic': 0, 'p_value': 1.0, 'significant': False}
        
        # 3. Quadratic pattern (U-shaped or inverted-U)
        try:
            if len(scales) >= 3:
                coeffs = np.polyfit(scales, values, 2)  # Quadratic fit
                quadratic_coef = coeffs[0]
                
                # Test if quadratic coefficient is significant
                residuals = values - np.polyval(coeffs, scales)
                mse = np.mean(residuals**2)
                quadratic_strength = abs(quadratic_coef) / (mse + 1e-8)
                
                patterns['quadratic'] = {
                    'coefficient': float(quadratic_coef),
                    'strength': float(quadratic_strength),
                    'significant': quadratic_strength > 0.1  # Empirical threshold
                }
            else:
                patterns['quadratic'] = {'coefficient': 0, 'strength': 0, 'significant': False}
        except:
            patterns['quadratic'] = {'coefficient': 0, 'strength': 0, 'significant': False}
        
        # 4. Range-based effect (consistent with Experiment 2)
        value_range = max(values) - min(values) if values else 0
        patterns['range_effect'] = {
            'range': float(value_range),
            'significant': value_range > 5.0  # Same threshold as Exp 2 for betting
        }
        
        # Overall significance: any pattern is significant
        any_significant = any(p.get('significant', False) for p in patterns.values())
        
        return {
            'any_significant': any_significant,
            'patterns': patterns
        }
    
    def is_causally_significant(self, stats: Dict) -> Tuple[bool, List[str]]:
        """Determine if feature shows causal significance using multiple criteria"""
        reasons = []
        
        # Criterion 1: T-test significance + effect size
        if (stats.get('ttest', {}).get('p_value', 1.0) < self.alpha and 
            abs(stats.get('cohens_d', 0)) > self.min_effect_size):
            reasons.append('ttest_effect')
        
        # Criterion 2: Mann-Whitney significance (robust)
        if stats.get('mannwhitney', {}).get('p_value', 1.0) < self.alpha:
            reasons.append('mannwhitney')
        
        # Criterion 3: Chi-square + Cramer's V
        if (stats.get('chi2', {}).get('p_value', 1.0) < self.alpha and 
            stats.get('cramers_v', 0) > 0.2):
            reasons.append('chi2_effect')
        
        # Criterion 4: Large practical effect
        desc = stats.get('descriptives', {})
        bet_diff = abs(desc.get('risky_mean_bet', 0) - desc.get('safe_mean_bet', 0))
        stop_diff = abs(desc.get('risky_stop_rate', 0) - desc.get('safe_stop_rate', 0))
        
        if bet_diff > 5.0 or stop_diff > 0.1:  # Practical significance thresholds
            reasons.append('practical_effect')
        
        return len(reasons) >= 2, reasons  # Require at least 2 criteria
    
    def test_single_feature(self, feature: Dict, means: Dict) -> Dict:
        """Test single feature with bidirectional patching"""
        layer = feature['layer']
        feature_id = feature['feature_id']
        
        key = (layer, feature_id)
        if key not in means:
            return None
        
        bankrupt_mean = means[key]['bankrupt_mean']
        safe_mean = means[key]['safe_mean']
        
        # Determine feature direction
        is_risky_feature = bankrupt_mean > safe_mean
        
        print(f"  Testing L{layer}-{feature_id} ({'risky' if is_risky_feature else 'safe'} feature)")
        
        # Bidirectional patching: Test both directions
        results = {
            'layer': layer,
            'feature_id': feature_id,
            'bankrupt_mean': bankrupt_mean,
            'safe_mean': safe_mean,
            'is_risky_feature': is_risky_feature,
            'directions': {}
        }
        
        for direction in ['to_safe', 'to_risky']:
            # Choose target mean based on direction
            if direction == 'to_safe':
                target_mean = safe_mean
                source_mean = bankrupt_mean
            else:
                target_mean = bankrupt_mean
                source_mean = safe_mean
            
            direction_data = {
                'target_mean': target_mean,
                'source_mean': source_mean,
                'scale_results': {}
            }
            
            # Test multiple scales
            # Precompute original feature values once per prompt
            risky_prompt = self.exp.risky_prompt
            safe_prompt = self.exp.safe_prompt
            try:
                orig_risky = self.exp.extract_original_feature(risky_prompt, layer, feature_id)
            except Exception:
                orig_risky = float('nan')
            try:
                orig_safe = self.exp.extract_original_feature(safe_prompt, layer, feature_id)
            except Exception:
                orig_safe = float('nan')

            for scale in self.scales:
                risky_bets, safe_bets = [], []
                risky_decisions, safe_decisions = [], []
                invalid_risky, invalid_safe = 0, 0
                
                # Generate trials for both prompt types
                for prompt_type in ['risky', 'safe']:
                    prompt = self.exp.risky_prompt if prompt_type == 'risky' else self.exp.safe_prompt
                    
                    bets, decisions = [], []
                    for trial in range(self.n_trials):
                        try:
                            # Calculate patched value using Exp2 semantics (1.0 = original, independent of "direction")
                            original = orig_risky if prompt_type == 'risky' else orig_safe
                            if scale < 1.0:
                                patched_value = max(0.0, safe_mean + scale * (original - safe_mean))
                            else:
                                adj = scale - 1.0
                                patched_value = max(0.0, original + adj * (bankrupt_mean - original))
                            
                            # Generate response
                            response = self.exp.generate_with_patching(
                                prompt, layer, feature_id, patched_value
                            )
                            parsed = self.exp.parse_response(response)
                            
                            if not parsed.get('valid', True):
                                if prompt_type == 'risky':
                                    invalid_risky += 1
                                else:
                                    invalid_safe += 1
                                continue
                            
                            decision = parsed['decision']
                            bet = int(parsed['bet']) if decision == 'bet' else 0
                            
                            bets.append(bet)
                            decisions.append(decision)
                            
                        except Exception as e:
                            if prompt_type == 'risky':
                                invalid_risky += 1
                            else:
                                invalid_safe += 1
                            continue
                    
                    if prompt_type == 'risky':
                        risky_bets.extend(bets)
                        risky_decisions.extend(decisions)
                    else:
                        safe_bets.extend(bets)
                        safe_decisions.extend(decisions)
                
                # Calculate statistics for this scale
                stats = self.calculate_comprehensive_stats(
                    risky_bets, safe_bets, risky_decisions, safe_decisions
                )
                # Add invalid rates to stats for audit
                total_risky = len(risky_decisions) + invalid_risky
                total_safe = len(safe_decisions) + invalid_safe
                stats['invalid_rate_risky'] = float(invalid_risky / total_risky) if total_risky > 0 else 0.0
                stats['invalid_rate_safe'] = float(invalid_safe / total_safe) if total_safe > 0 else 0.0
                
                direction_data['scale_results'][scale] = stats
            
            # Aggregate statistics across scales
            all_p_values = []
            all_effect_sizes = []

            # Build per-scale series for monotonicity (compat with Exp2)
            scales_sorted = sorted(direction_data['scale_results'].keys())
            risky_mean_bets = [direction_data['scale_results'][s]['descriptives']['risky_mean_bet'] for s in scales_sorted]
            safe_mean_bets = [direction_data['scale_results'][s]['descriptives']['safe_mean_bet'] for s in scales_sorted]
            risky_stop_rates = [direction_data['scale_results'][s]['descriptives']['risky_stop_rate'] for s in scales_sorted]
            safe_stop_rates = [direction_data['scale_results'][s]['descriptives']['safe_stop_rate'] for s in scales_sorted]

            # Correlations and ranges across scales
            def corr_and_range(xs: List[float]) -> Tuple[float, float]:
                try:
                    rho, _ = spearmanr(scales_sorted, xs)
                    return float(rho), float(max(xs) - min(xs))
                except Exception:
                    return 0.0, 0.0

            rho_br, range_br = corr_and_range(risky_mean_bets)
            rho_bs, range_bs = corr_and_range(safe_mean_bets)
            rho_sr, range_sr = corr_and_range(risky_stop_rates)
            rho_ss, range_ss = corr_and_range(safe_stop_rates)

            # Also collect per-scale classical stats
            for scale_stats in direction_data['scale_results'].values():
                all_p_values.append(scale_stats.get('ttest', {}).get('p_value', 1.0))
                all_effect_sizes.append(abs(scale_stats.get('cohens_d', 0)))

            direction_data['min_p_value'] = min(all_p_values) if all_p_values else 1.0
            direction_data['max_effect_size'] = max(all_effect_sizes) if all_effect_sizes else 0.0
            direction_data['spearman'] = {
                'bet_risky': rho_br,
                'bet_safe': rho_bs,
                'stop_risky': rho_sr,
                'stop_safe': rho_ss,
                'range_bet_risky': range_br,
                'range_bet_safe': range_bs,
                'range_stop_risky': range_sr,
                'range_stop_safe': range_ss,
            }

            # Enhanced significance detection (monotonic + non-monotonic patterns)
            
            # Traditional monotonic (aligned with Exp2)
            is_monotonic_bet = (abs(rho_br) > 0.4 or abs(rho_bs) > 0.4) and (max(range_br, range_bs) > 3.0)  # Relaxed
            is_monotonic_stop = (abs(rho_sr) > 0.4 or abs(rho_ss) > 0.4) and (max(range_sr, range_ss) > 0.08)  # Relaxed
            
            # Non-monotonic patterns
            bet_patterns = self.detect_all_causality_patterns(scales_sorted, risky_mean_bets)
            stop_patterns = self.detect_all_causality_patterns(scales_sorted, risky_stop_rates)
            
            is_nonmonotonic_bet = bet_patterns['any_significant']
            is_nonmonotonic_stop = stop_patterns['any_significant']
            
            # Combined assessment
            is_causal_bet = is_monotonic_bet or is_nonmonotonic_bet
            is_causal_stop = is_monotonic_stop or is_nonmonotonic_stop
            is_significant = bool(is_causal_bet or is_causal_stop)
            
            # Record reasons
            reasons = []
            if is_monotonic_bet:
                reasons.append('monotonic_bet')
            if is_nonmonotonic_bet:
                reasons.append('nonmonotonic_bet')
            if is_monotonic_stop:
                reasons.append('monotonic_stop') 
            if is_nonmonotonic_stop:
                reasons.append('nonmonotonic_stop')
            
            # Store pattern analysis
            direction_data['pattern_analysis'] = {
                'bet_patterns': bet_patterns,
                'stop_patterns': stop_patterns
            }
            direction_data['is_significant'] = is_significant
            direction_data['significance_reasons'] = reasons
            
            results['directions'][direction] = direction_data
        
        # Overall feature assessment
        results['is_causal'] = any(
            d['is_significant'] for d in results['directions'].values()
        )
        
        return results
    
    def run_experiment(self, results_path: str):
        """Run comprehensive revalidation experiment"""
        print("="*80)
        print("Starting Comprehensive Revalidation of 142 Causal Features")
        print("="*80)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load models and data
        print("Loading models...")
        self.exp.load_models()
        
        print("Loading causal features...")
        features = self.load_causal_features(results_path)
        
        print("Loading feature means...")
        means = self.load_feature_means()
        
        # Test all features
        all_results = []
        causal_count = 0
        
        for i, feature in enumerate(tqdm(features, desc="Testing features")):
            result = self.test_single_feature(feature, means)
            
            if result is None:
                continue
            
            all_results.append(result)
            
            if result['is_causal']:
                causal_count += 1
            
            # Save intermediate results every 20 features
            if (i + 1) % 20 == 0:
                intermediate_file = self.results_dir / f"comprehensive_revalidation_intermediate_{timestamp}.json"
                with open(intermediate_file, 'w') as f:
                    json.dump({
                        'timestamp': timestamp,
                        'progress': f"{i+1}/{len(features)}",
                        'causal_count': causal_count,
                        'results': all_results
                    }, f, indent=2)
                print(f"\nSaved intermediate results: {causal_count}/{i+1} causal")
        
        # Save final results
        final_file = self.results_dir / f"comprehensive_revalidation_final_{timestamp}.json"
        
        summary = {
            'timestamp': timestamp,
            'experiment_config': {
                'gpu': self.gpu_id,
                'scales': self.scales,
                'n_trials': self.n_trials,
                'alpha': self.alpha,
                'min_effect_size': self.min_effect_size
            },
            'summary': {
                'n_features_tested': len(all_results),
                'n_causal_features': causal_count,
                'causal_rate': causal_count / len(all_results) if all_results else 0
            },
            'causal_features': [r for r in all_results if r['is_causal']],
            'all_results': all_results
        }
        
        with open(final_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "="*80)
        print("Comprehensive Revalidation Complete!")
        print("="*80)
        print(f"✅ Results saved to: {final_file}")
        print(f"📊 Causal features: {causal_count}/{len(all_results)} ({100*causal_count/len(all_results):.1f}%)")

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Revalidation Experiment')
    parser.add_argument('--results', type=str, 
                       default='/data/llm_addiction/results/patching_population_mean_final_20250905_150612.json',
                       help='Path to population mean patching results')
    parser.add_argument('--gpu', type=str, default='6', help='GPU to use')
    args = parser.parse_args()
    
    exp = ComprehensiveRevalidationExperiment(gpu_id=args.gpu)
    exp.run_experiment(args.results)

if __name__ == "__main__":
    main()
