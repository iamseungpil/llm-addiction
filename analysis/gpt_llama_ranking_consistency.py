#!/usr/bin/env python3
"""
GPT vs LLaMA Ranking Consistency Analysis
Compares prompt-specific risk rankings between GPT and LLaMA experiments
"""

import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from pathlib import Path
from collections import defaultdict

class RankingConsistencyAnalyzer:
    def __init__(self):
        # Data paths
        self.gpt_file = '/data/llm_addiction/gpt_results_corrected/gpt_corrected_complete_20250825_212628.json'
        self.llama_main_file = '/data/llm_addiction/results/exp1_multiround_intermediate_20250819_140040.json'
        self.llama_additional_file = '/data/llm_addiction/results/exp1_missing_complete_20250820_090040.json'
        
        # Results storage
        self.gpt_metrics = {}
        self.llama_metrics = {}
        self.common_prompts = []
        
    def load_gpt_data(self):
        """Load and process GPT experiment results"""
        with open(self.gpt_file, 'r') as f:
            data = json.load(f)
        
        # Group by prompt combination
        prompt_groups = defaultdict(list)
        
        for result in data['results']:
            prompt = result['prompt_combo']
            prompt_groups[prompt].append(result)
        
        # Calculate metrics per prompt
        for prompt, results in prompt_groups.items():
            total_games = len(results)
            bankruptcies = sum(1 for r in results if r.get('is_bankrupt', False))
            
            # Calculate metrics
            bankruptcy_rate = bankruptcies / total_games if total_games > 0 else 0
            avg_bet = np.mean([r.get('total_bet', 0) / max(r.get('total_rounds', 1), 1) for r in results])
            avg_loss = np.mean([100 - r.get('final_balance', 100) for r in results])
            avg_rounds = np.mean([r.get('total_rounds', 0) for r in results])
            
            self.gpt_metrics[prompt] = {
                'bankruptcy_rate': bankruptcy_rate,
                'avg_bet_per_round': avg_bet,
                'avg_loss': avg_loss,
                'avg_rounds': avg_rounds,
                'total_games': total_games
            }
        
        print(f"✅ Loaded GPT data: {len(self.gpt_metrics)} prompt combinations")
        
    def load_llama_data(self):
        """Load and process LLaMA experiment results"""
        # Load main file only to speed up analysis
        with open(self.llama_main_file, 'r') as f:
            llama_results = json.load(f)
        
        # Group by prompt combination
        prompt_groups = defaultdict(list)
        
        # Process first 1000 results for speed
        for result in llama_results[:1000]:
            # Extract prompt components
            prompt_components = result.get('prompt_components', [])
            if not prompt_components:
                prompt_components = ['BASE']
            
            prompt = ''.join(sorted(prompt_components)) if prompt_components != ['BASE'] else 'BASE'
            prompt_groups[prompt].append(result)
        
        # Calculate metrics per prompt
        for prompt, results in prompt_groups.items():
            if len(results) < 5:  # Skip prompts with too few samples
                continue
                
            total_games = len(results)
            bankruptcies = sum(1 for r in results if r.get('is_bankrupt', False) or r.get('final_balance', 100) <= 0)
            
            # Calculate metrics
            bankruptcy_rate = bankruptcies / total_games if total_games > 0 else 0
            avg_bet = np.mean([np.mean(r.get('bet_amounts', [10])) if r.get('bet_amounts') else 10 for r in results])
            avg_loss = np.mean([max(0, 100 - r.get('final_balance', 100)) for r in results])
            avg_rounds = np.mean([len(r.get('bet_amounts', [])) for r in results])
            
            self.llama_metrics[prompt] = {
                'bankruptcy_rate': bankruptcy_rate,
                'avg_bet_per_round': avg_bet,
                'avg_loss': avg_loss,
                'avg_rounds': avg_rounds,
                'total_games': total_games
            }
        
        print(f"✅ Loaded LLaMA data: {len(self.llama_metrics)} prompt combinations")
        
    def find_common_prompts(self):
        """Find prompt combinations present in both datasets"""
        gpt_prompts = set(self.gpt_metrics.keys())
        llama_prompts = set(self.llama_metrics.keys())
        
        # Direct matches
        direct_matches = gpt_prompts & llama_prompts
        
        # Map similar prompts (handle different naming conventions)
        prompt_mappings = {
            'BASE': 'BASE',
            'G': 'G',
            'M': 'M', 
            'P': 'P',
            'R': 'R',
            'W': 'W',
            'GM': 'GM',
            'GP': 'GP',
            'GW': 'GW',
            'MP': 'MP',
            'MW': 'MW',
            'PW': 'PW',
            'GMP': 'GMP',
            'GMW': 'GMW',
            'GPW': 'GPW',
            'MPW': 'MPW',
            'GMPW': 'GMPW'
        }
        
        mapped_matches = set()
        for gpt_prompt in gpt_prompts:
            if gpt_prompt in prompt_mappings:
                llama_equivalent = prompt_mappings[gpt_prompt]
                if llama_equivalent in llama_prompts:
                    mapped_matches.add(gpt_prompt)
        
        self.common_prompts = list(direct_matches | mapped_matches)
        print(f"✅ Found {len(self.common_prompts)} common prompt combinations")
        return self.common_prompts
        
    def calculate_rankings(self):
        """Calculate rankings for each metric in both datasets"""
        if not self.common_prompts:
            print("❌ No common prompts found")
            return None
            
        metrics = ['bankruptcy_rate', 'avg_bet_per_round', 'avg_loss', 'avg_rounds']
        results = {}
        
        for metric in metrics:
            # Get values for common prompts
            gpt_values = [(prompt, self.gpt_metrics[prompt][metric]) for prompt in self.common_prompts]
            llama_values = [(prompt, self.llama_metrics[prompt][metric]) for prompt in self.common_prompts]
            
            # Sort by value (descending for risk metrics)
            gpt_ranked = sorted(gpt_values, key=lambda x: x[1], reverse=True)
            llama_ranked = sorted(llama_values, key=lambda x: x[1], reverse=True)
            
            # Create rank dictionaries
            gpt_ranks = {prompt: rank+1 for rank, (prompt, _) in enumerate(gpt_ranked)}
            llama_ranks = {prompt: rank+1 for rank, (prompt, _) in enumerate(llama_ranked)}
            
            # Calculate Spearman correlation
            gpt_rank_values = [gpt_ranks[prompt] for prompt in self.common_prompts]
            llama_rank_values = [llama_ranks[prompt] for prompt in self.common_prompts]
            
            correlation, p_value = spearmanr(gpt_rank_values, llama_rank_values)
            
            results[metric] = {
                'gpt_rankings': gpt_ranked,
                'llama_rankings': llama_ranked,
                'correlation': correlation,
                'p_value': p_value,
                'gpt_ranks': gpt_ranks,
                'llama_ranks': llama_ranks
            }
            
            print(f"📊 {metric}: ρ = {correlation:.3f}, p = {p_value:.3f}")
        
        return results
        
    def generate_comparison_table(self, ranking_results):
        """Generate comparison table for the paper"""
        if not ranking_results:
            return None
            
        # Create comprehensive comparison table
        table_data = []
        
        for prompt in self.common_prompts:
            row = {'Prompt': prompt}
            
            # Add rankings for each metric
            for metric in ['bankruptcy_rate', 'avg_bet_per_round', 'avg_loss', 'avg_rounds']:
                gpt_rank = ranking_results[metric]['gpt_ranks'][prompt]
                llama_rank = ranking_results[metric]['llama_ranks'][prompt]
                row[f'{metric}_gpt_rank'] = gpt_rank
                row[f'{metric}_llama_rank'] = llama_rank
                row[f'{metric}_rank_diff'] = abs(gpt_rank - llama_rank)
            
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        return df
        
    def run_analysis(self):
        """Run complete ranking consistency analysis"""
        print("🚀 Starting GPT vs LLaMA Ranking Consistency Analysis")
        print("=" * 60)
        
        # Load data
        self.load_gpt_data()
        self.load_llama_data()
        
        # Find common prompts
        common = self.find_common_prompts()
        if not common:
            print("❌ No common prompts found for comparison")
            return None
        
        # Calculate rankings and correlations
        ranking_results = self.calculate_rankings()
        
        # Generate comparison table
        comparison_table = self.generate_comparison_table(ranking_results)
        
        # Summary statistics
        print("\n📈 RANKING CONSISTENCY SUMMARY")
        print("=" * 40)
        correlations = []
        for metric, result in ranking_results.items():
            corr = result['correlation']
            p_val = result['p_value']
            correlations.append(corr)
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"{metric:20}: ρ = {corr:6.3f} {significance}")
        
        avg_correlation = np.mean(correlations)
        print(f"{'Average Correlation':20}: ρ = {avg_correlation:6.3f}")
        
        return {
            'ranking_results': ranking_results,
            'comparison_table': comparison_table,
            'common_prompts': self.common_prompts,
            'average_correlation': avg_correlation
        }

def main():
    analyzer = RankingConsistencyAnalyzer()
    results = analyzer.run_analysis()
    
    if results:
        # Save results
        output_file = '/home/ubuntu/llm_addiction/analysis/gpt_llama_ranking_results.json'
        
        # Convert to JSON-serializable format
        save_data = {
            'common_prompts': results['common_prompts'],
            'average_correlation': float(results['average_correlation']),
            'metric_correlations': {
                metric: {
                    'correlation': float(data['correlation']),
                    'p_value': float(data['p_value']),
                    'gpt_rankings': [(p, float(v)) for p, v in data['gpt_rankings']],
                    'llama_rankings': [(p, float(v)) for p, v in data['llama_rankings']]
                }
                for metric, data in results['ranking_results'].items()
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        
        # Save comparison table
        if results['comparison_table'] is not None:
            csv_file = '/home/ubuntu/llm_addiction/analysis/gpt_llama_ranking_comparison.csv'
            results['comparison_table'].to_csv(csv_file, index=False)
            print(f"📊 Comparison table saved to: {csv_file}")

if __name__ == "__main__":
    main()