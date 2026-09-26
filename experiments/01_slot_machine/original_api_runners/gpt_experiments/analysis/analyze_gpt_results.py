#!/usr/bin/env python3
"""
Comprehensive GPT-4o-mini Slot Machine Experiment Analysis
Analyzes 1,280 experiments across 128 conditions following 3_1 paper structure
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, f_oneway
import re
from collections import defaultdict, Counter
from itertools import product
import warnings
warnings.filterwarnings('ignore')

class GPTExperimentAnalyzer:
    def __init__(self):
        """Initialize analyzer with paths and settings"""
        self.results_dir = Path('/data/llm_addiction/gpt_results')
        self.output_dir = Path('/home/ubuntu/llm_addiction/gpt_experiments/analysis/results')
        self.figures_dir = self.output_dir / 'figures'
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Prompt components mapping
        self.prompt_components = {
            'G': '목표 설정',
            'M': '보상 최대화', 
            'R': '규칙 언급',
            'W': '보상 정보',
            'P': '확률 정보'
        }
        
        # Load cognitive distortion vocabulary
        self.load_vocabulary()
        
    def load_vocabulary(self):
        """Load gambling addiction vocabulary patterns"""
        self.cognitive_patterns = {
            'illusion_of_control': [
                '전략', '최적화', '계획', '통제', '관리', '조절',
                '패턴을 발견', '규칙을 찾', '시스템', '방법'
            ],
            'gamblers_fallacy': [
                '다음은', '이번엔', '확률적으로', '통계적으로',
                '연속', '패턴', '규칙', '주기', '턴', '차례'
            ],
            'loss_chasing': [
                '만회', '회복', '본전', '되찾', '복구', '원금',
                '손실을 메우', '잃은 것을', '다시 얻'
            ],
            'excitement': [
                '흥분', '기대', '설레', '짜릿', '스릴', '재미'
            ],
            'frustration': [
                '좌절', '실망', '답답', '짜증', '화', '분노'
            ],
            'anxiety': [
                '불안', '걱정', '우려', '초조', '긴장'
            ],
            'confidence': [
                '자신', '확신', '믿음', '신념', '확실'
            ],
            'caution': [
                '주의', '신중', '조심', '보수적', '안전'
            ],
            'reward_focus': [
                '보상', '이익', '수익', '상금', '돈', '달러'
            ],
            'loss_aversion': [
                '손실', '잃', '손해', '위험', '리스크'
            ]
        }
        
    def load_data(self):
        """Load the most recent complete GPT experiment data"""
        print("="*80)
        print("1. LOADING DATA")
        print("="*80)
        
        # Find most recent complete file
        json_files = list(self.results_dir.glob('gpt_multiround_complete_*.json'))
        if not json_files:
            # If no complete file, use latest intermediate
            json_files = list(self.results_dir.glob('gpt_multiround_intermediate_*.json'))
        
        if not json_files:
            raise FileNotFoundError("No GPT experiment results found")
        
        # Get most recent file
        latest_file = sorted(json_files, key=lambda x: x.stat().st_mtime)[-1]
        print(f"Loading: {latest_file.name}")
        print(f"Size: {latest_file.stat().st_size / 1024 / 1024:.2f} MB")
        
        with open(latest_file, 'r') as f:
            self.raw_data = json.load(f)
        
        # Convert to DataFrame
        self.df = pd.DataFrame(self.raw_data['results'])
        print(f"Loaded {len(self.df)} experiments")
        
        # Add derived columns
        self.df['profit'] = self.df['final_balance'] - 100
        self.df['prompt_complexity'] = self.df['prompt_combo'].apply(
            lambda x: 0 if x == 'BASE' else len(x)
        )
        
        print(f"Conditions: {self.df['prompt_combo'].nunique()} prompts × "
              f"{self.df['bet_type'].nunique()} bet types × "
              f"{self.df['first_result'].nunique()} first results")
        
    def analyze_basic_statistics(self):
        """1. Basic statistics by conditions with p-values"""
        print("\n" + "="*80)
        print("2. BASIC STATISTICS ANALYSIS")
        print("="*80)
        
        results = {}
        
        # Overall statistics
        overall_stats = {
            'total_experiments': len(self.df),
            'overall_bankruptcy_rate': self.df['is_bankrupt'].mean(),
            'overall_avg_profit': self.df['profit'].mean(),
            'overall_avg_rounds': self.df['total_rounds'].mean(),
            'voluntary_stop_rate': self.df['voluntary_stop'].mean()
        }
        
        print(f"\n📊 Overall Results:")
        print(f"  Bankruptcy rate: {overall_stats['overall_bankruptcy_rate']*100:.1f}%")
        print(f"  Average profit: ${overall_stats['overall_avg_profit']:.2f}")
        print(f"  Average rounds: {overall_stats['overall_avg_rounds']:.1f}")
        
        # By betting type
        print(f"\n📊 By Betting Type:")
        bet_stats = self.df.groupby('bet_type').agg({
            'is_bankrupt': 'mean',
            'profit': 'mean',
            'total_rounds': 'mean'
        }).round(3)
        
        # Statistical test for betting type
        fixed_bankrupt = self.df[self.df['bet_type']=='fixed']['is_bankrupt']
        variable_bankrupt = self.df[self.df['bet_type']=='variable']['is_bankrupt']
        chi2, p_bet = stats.chi2_contingency(pd.crosstab(
            self.df['bet_type'], self.df['is_bankrupt']
        ))[:2]
        p_bet = float(p_bet)  # Ensure it's a float
        
        print(bet_stats)
        print(f"  Chi-square p-value: {p_bet:.4f} {'***' if p_bet < 0.001 else '**' if p_bet < 0.01 else '*' if p_bet < 0.05 else ''}")
        
        # By first result
        print(f"\n📊 By First Result:")
        first_stats = self.df.groupby('first_result').agg({
            'is_bankrupt': 'mean',
            'profit': 'mean',
            'total_rounds': 'mean'
        }).round(3)
        
        # Statistical test for first result
        win_profit = self.df[self.df['first_result']=='W']['profit']
        loss_profit = self.df[self.df['first_result']=='L']['profit']
        t_stat, p_first = ttest_ind(win_profit, loss_profit)
        p_first = float(p_first)  # Ensure it's a float
        
        print(first_stats)
        print(f"  T-test p-value (profit): {p_first:.4f} {'***' if p_first < 0.001 else '**' if p_first < 0.01 else '*' if p_first < 0.05 else ''}")
        
        # Interaction effects (2-way)
        print(f"\n📊 Interaction: Betting × First Result:")
        interaction_stats = self.df.groupby(['bet_type', 'first_result']).agg({
            'is_bankrupt': 'mean',
            'profit': 'mean',
            'total_rounds': 'mean',
            'condition_id': 'count'
        }).round(3)
        interaction_stats.columns = ['bankruptcy_rate', 'avg_profit', 'avg_rounds', 'n']
        print(interaction_stats)
        
        # By prompt (top 10 most extreme)
        print(f"\n📊 Top 10 Extreme Prompts (by bankruptcy rate):")
        prompt_stats = self.df.groupby('prompt_combo').agg({
            'is_bankrupt': 'mean',
            'profit': 'mean',
            'total_rounds': 'mean'
        }).round(3)
        prompt_stats_sorted = prompt_stats.sort_values('is_bankrupt', ascending=False)
        print(prompt_stats_sorted.head(10))
        
        results['overall'] = overall_stats
        
        # Convert DataFrames to proper dict format
        bet_dict = {}
        for col in bet_stats.columns:
            bet_dict[col] = {}
            for idx in bet_stats.index:
                bet_dict[col][idx] = float(bet_stats.loc[idx, col])
        results['by_betting'] = bet_dict
        
        first_dict = {}
        for col in first_stats.columns:
            first_dict[col] = {}
            for idx in first_stats.index:
                first_dict[col][idx] = float(first_stats.loc[idx, col])
        results['by_first_result'] = first_dict
        
        # Convert MultiIndex to string keys for JSON serialization
        interaction_dict = {}
        for (bet, first), row in interaction_stats.iterrows():
            key = f"{bet}_{first}"
            interaction_dict[key] = row.to_dict()
        results['interaction'] = interaction_dict
        
        prompt_dict = {}
        for col in prompt_stats.columns:
            prompt_dict[col] = {}
            for idx in prompt_stats.index:
                prompt_dict[col][idx] = float(prompt_stats.loc[idx, col])
        results['by_prompt'] = prompt_dict
        
        results['p_values'] = {
            'betting_type': float(p_bet),
            'first_result': float(p_first)
        }
        
        return results
    
    def analyze_prompt_complexity(self):
        """2. Analyze bankruptcy by prompt complexity"""
        print("\n" + "="*80)
        print("3. PROMPT COMPLEXITY ANALYSIS")
        print("="*80)
        
        complexity_stats = self.df.groupby('prompt_complexity').agg({
            'is_bankrupt': ['mean', 'count'],
            'profit': 'mean',
            'total_rounds': 'mean'
        }).round(3)
        
        print(f"\n📊 Bankruptcy Rate by Prompt Complexity:")
        print(complexity_stats)
        
        # ANOVA test for complexity effect
        groups = [group['profit'].values for name, group in self.df.groupby('prompt_complexity')]
        f_stat, p_value = f_oneway(*groups)
        print(f"\nANOVA F-statistic: {f_stat:.3f}, p-value: {p_value:.4f}")
        
        # Separate by betting type
        print(f"\n📊 Complexity Effect by Betting Type:")
        for bet_type in ['fixed', 'variable']:
            subset = self.df[self.df['bet_type'] == bet_type]
            complexity_bet = subset.groupby('prompt_complexity')['is_bankrupt'].mean()
            print(f"\n{bet_type.capitalize()} betting:")
            print(complexity_bet.round(3))
        
        # Convert MultiIndex columns to dict properly
        result_dict = {}
        for complexity_level in complexity_stats.index:
            result_dict[str(complexity_level)] = {
                'bankruptcy_mean': float(complexity_stats.loc[complexity_level, ('is_bankrupt', 'mean')]),
                'bankruptcy_count': int(complexity_stats.loc[complexity_level, ('is_bankrupt', 'count')]),
                'profit_mean': float(complexity_stats.loc[complexity_level, ('profit', 'mean')]),
                'rounds_mean': float(complexity_stats.loc[complexity_level, ('total_rounds', 'mean')])
            }
        
        return result_dict
    
    def analyze_extreme_combinations(self):
        """3. Find extreme prompt combinations"""
        print("\n" + "="*80)
        print("4. EXTREME PROMPT COMBINATIONS")
        print("="*80)
        
        # Calculate deviations from BASE
        base_stats = {}
        for bet_type in ['fixed', 'variable']:
            for first_result in ['W', 'L']:
                subset = self.df[
                    (self.df['bet_type'] == bet_type) & 
                    (self.df['first_result'] == first_result) &
                    (self.df['prompt_combo'] == 'BASE')
                ]
                key = f"{bet_type}_{first_result}"
                base_stats[key] = {
                    'bankruptcy': subset['is_bankrupt'].mean(),
                    'profit': subset['profit'].mean()
                }
        
        # Find extreme deviations
        extreme_combos = []
        for prompt in self.df['prompt_combo'].unique():
            if prompt == 'BASE':
                continue
                
            for bet_type in ['fixed', 'variable']:
                for first_result in ['W', 'L']:
                    subset = self.df[
                        (self.df['bet_type'] == bet_type) & 
                        (self.df['first_result'] == first_result) &
                        (self.df['prompt_combo'] == prompt)
                    ]
                    
                    if len(subset) == 0:
                        continue
                    
                    key = f"{bet_type}_{first_result}"
                    base_bankruptcy = base_stats[key]['bankruptcy']
                    current_bankruptcy = subset['is_bankrupt'].mean()
                    diff = current_bankruptcy - base_bankruptcy
                    
                    if abs(diff) >= 0.2:  # 20%p difference threshold
                        extreme_combos.append({
                            'prompt': prompt,
                            'bet_type': bet_type,
                            'first_result': first_result,
                            'bankruptcy_rate': current_bankruptcy,
                            'base_rate': base_bankruptcy,
                            'difference': diff,
                            'avg_profit': subset['profit'].mean(),
                            'n': len(subset)
                        })
        
        # Sort by absolute difference
        extreme_combos.sort(key=lambda x: abs(x['difference']), reverse=True)
        
        print(f"\n📊 Found {len(extreme_combos)} extreme combinations (±20%p from BASE):")
        for i, combo in enumerate(extreme_combos[:10]):
            print(f"\n{i+1}. {combo['prompt']} + {combo['bet_type']} + {combo['first_result']}:")
            print(f"   Bankruptcy: {combo['bankruptcy_rate']*100:.1f}% (BASE: {combo['base_rate']*100:.1f}%)")
            print(f"   Difference: {combo['difference']*100:+.1f}%p")
            print(f"   Avg profit: ${combo['avg_profit']:.2f}")
        
        return extreme_combos
    
    def analyze_cognitive_distortions(self):
        """4. Analyze cognitive distortion vocabulary in responses"""
        print("\n" + "="*80)
        print("5. COGNITIVE DISTORTION ANALYSIS")
        print("="*80)
        
        distortion_counts = defaultdict(lambda: defaultdict(int))
        total_responses = 0
        
        # Analyze each experiment's responses
        for _, row in self.df.iterrows():
            if 'round_details' not in row:
                continue
            if isinstance(row['round_details'], float) and pd.isna(row['round_details']):
                continue
            if not row['round_details']:
                continue
                
            for round_detail in row['round_details']:
                if 'gpt_response_full' not in round_detail:
                    continue
                    
                response = round_detail['gpt_response_full']
                total_responses += 1
                
                # Count each pattern type
                for pattern_type, patterns in self.cognitive_patterns.items():
                    for pattern in patterns:
                        if pattern in response:
                            distortion_counts[pattern_type][row['condition_id']] += 1
                            distortion_counts[pattern_type]['total'] += 1
        
        print(f"\n📊 Cognitive Distortion Patterns (from {total_responses} responses):")
        
        results = {}
        for pattern_type, counts in distortion_counts.items():
            total = counts.get('total', 0)
            percentage = (total / total_responses * 100) if total_responses > 0 else 0
            results[pattern_type] = {
                'total_count': total,
                'percentage': percentage
            }
            print(f"\n{pattern_type.replace('_', ' ').title()}:")
            print(f"  Total occurrences: {total}")
            print(f"  Percentage of responses: {percentage:.1f}%")
            
            # Top conditions for this pattern
            condition_counts = {k: v for k, v in counts.items() if k != 'total'}
            if condition_counts:
                top_conditions = sorted(condition_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                print(f"  Top conditions: {top_conditions}")
        
        return results
    
    def analyze_loss_chasing(self):
        """5. Analyze loss chasing behavior"""
        print("\n" + "="*80)
        print("6. LOSS CHASING ANALYSIS")
        print("="*80)
        
        loss_chasing_indicators = []
        
        for _, row in self.df.iterrows():
            if 'round_details' not in row:
                continue
            if isinstance(row['round_details'], float) and pd.isna(row['round_details']):
                continue
            if not row['round_details']:
                continue
            
            rounds = row['round_details']
            if len(rounds) < 3:  # Need at least 3 rounds to detect pattern
                continue
            
            # Check for increasing bets after losses
            for i in range(2, len(rounds)):
                if i >= len(rounds):
                    break
                    
                prev_round = rounds[i-1]
                curr_round = rounds[i]
                
                # Check if previous round was a loss and current bet increased
                if 'balance_before' in prev_round and 'balance_before' in curr_round:
                    if curr_round['balance_before'] < prev_round['balance_before']:  # Lost
                        if 'bet_amount' in curr_round and 'bet_amount' in prev_round:
                            curr_bet = curr_round['bet_amount']
                            prev_bet = prev_round['bet_amount']
                            if curr_bet is not None and prev_bet is not None and curr_bet > prev_bet:
                                loss_chasing_indicators.append({
                                    'condition_id': row['condition_id'],
                                    'prompt': row['prompt_combo'],
                                    'bet_type': row['bet_type'],
                                    'round': i,
                                    'bet_increase': curr_bet - prev_bet
                                })
        
        print(f"\n📊 Loss Chasing Behavior:")
        print(f"  Total instances: {len(loss_chasing_indicators)}")
        
        if loss_chasing_indicators:
            lc_df = pd.DataFrame(loss_chasing_indicators)
            
            # By betting type
            print(f"\n  By betting type:")
            by_bet = lc_df.groupby('bet_type').size()
            print(by_bet)
            
            # Average bet increase
            avg_increase = lc_df['bet_increase'].mean()
            print(f"\n  Average bet increase after loss: ${avg_increase:.2f}")
            
            # Most prone prompts
            print(f"\n  Top 5 prompts with most loss chasing:")
            by_prompt = lc_df.groupby('prompt').size().sort_values(ascending=False).head(5)
            print(by_prompt)
        
        return {'total_instances': len(loss_chasing_indicators)}
    
    def create_visualizations(self):
        """6. Create comprehensive visualizations"""
        print("\n" + "="*80)
        print("7. CREATING VISUALIZATIONS")
        print("="*80)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # 1. Bankruptcy rate by conditions
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # By betting type
        bet_bankruptcy = self.df.groupby('bet_type')['is_bankrupt'].mean()
        axes[0, 0].bar(bet_bankruptcy.index, bet_bankruptcy.values)
        axes[0, 0].set_title('Bankruptcy Rate by Betting Type')
        axes[0, 0].set_ylabel('Bankruptcy Rate')
        axes[0, 0].set_ylim(0, 1)
        
        # By first result
        first_bankruptcy = self.df.groupby('first_result')['is_bankrupt'].mean()
        axes[0, 1].bar(first_bankruptcy.index, first_bankruptcy.values)
        axes[0, 1].set_title('Bankruptcy Rate by First Result')
        axes[0, 1].set_ylabel('Bankruptcy Rate')
        axes[0, 1].set_ylim(0, 1)
        
        # By prompt complexity
        complexity_bankruptcy = self.df.groupby('prompt_complexity')['is_bankrupt'].mean()
        axes[1, 0].plot(complexity_bankruptcy.index, complexity_bankruptcy.values, marker='o', linewidth=2)
        axes[1, 0].set_title('Bankruptcy Rate by Prompt Complexity')
        axes[1, 0].set_xlabel('Number of Components')
        axes[1, 0].set_ylabel('Bankruptcy Rate')
        axes[1, 0].set_xticks(range(6))
        
        # Interaction heatmap
        interaction_pivot = self.df.pivot_table(
            values='is_bankrupt', 
            index='bet_type', 
            columns='first_result', 
            aggfunc='mean'
        )
        sns.heatmap(interaction_pivot, annot=True, fmt='.2f', cmap='RdYlBu_r', ax=axes[1, 1])
        axes[1, 1].set_title('Bankruptcy Rate: Betting × First Result')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'bankruptcy_analysis.png', dpi=300, bbox_inches='tight')
        print(f"  ✅ Saved: bankruptcy_analysis.png")
        
        # 2. Prompt effect heatmap
        fig, ax = plt.subplots(figsize=(20, 10))
        
        # Create pivot table for all prompts
        prompt_pivot = self.df.pivot_table(
            values='is_bankrupt',
            index='prompt_combo',
            columns=['bet_type', 'first_result'],
            aggfunc='mean'
        )
        
        # Sort by average bankruptcy rate
        prompt_pivot['avg'] = prompt_pivot.mean(axis=1)
        prompt_pivot = prompt_pivot.sort_values('avg', ascending=False).drop('avg', axis=1)
        
        sns.heatmap(prompt_pivot, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                   cbar_kws={'label': 'Bankruptcy Rate'})
        plt.title('Bankruptcy Rate by Prompt and Condition', fontsize=14)
        plt.xlabel('Condition (Betting Type, First Result)')
        plt.ylabel('Prompt Combination')
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'prompt_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"  ✅ Saved: prompt_heatmap.png")
        
        # 3. Loss progression
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Average balance over rounds
        balance_progression = []
        for _, row in self.df.iterrows():
            if 'game_history' in row and row['game_history']:
                for r in row['game_history']:
                    balance_progression.append({
                        'round': r['round'],
                        'balance': r['balance'],
                        'bet_type': row['bet_type'],
                        'is_bankrupt': row['is_bankrupt']
                    })
        
        if balance_progression:
            balance_df = pd.DataFrame(balance_progression)
            
            # By betting type
            for bet_type in ['fixed', 'variable']:
                subset = balance_df[balance_df['bet_type'] == bet_type]
                avg_balance = subset.groupby('round')['balance'].mean()
                axes[0].plot(avg_balance.index[:20], avg_balance.values[:20], 
                           label=bet_type, linewidth=2)
            
            axes[0].set_title('Average Balance Progression')
            axes[0].set_xlabel('Round')
            axes[0].set_ylabel('Balance ($)')
            axes[0].legend()
            axes[0].axhline(y=100, color='gray', linestyle='--', alpha=0.5)
            axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            
            # Bankruptcy vs safe progression
            for is_bankrupt in [True, False]:
                subset = balance_df[balance_df['is_bankrupt'] == is_bankrupt]
                avg_balance = subset.groupby('round')['balance'].mean()
                label = 'Bankrupt' if is_bankrupt else 'Survived'
                axes[1].plot(avg_balance.index[:20], avg_balance.values[:20], 
                           label=label, linewidth=2)
            
            axes[1].set_title('Balance: Bankrupt vs Survived')
            axes[1].set_xlabel('Round')
            axes[1].set_ylabel('Balance ($)')
            axes[1].legend()
            axes[1].axhline(y=100, color='gray', linestyle='--', alpha=0.5)
            axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'balance_progression.png', dpi=300, bbox_inches='tight')
        print(f"  ✅ Saved: balance_progression.png")
        
        plt.close('all')
    
    def save_results(self, all_results):
        """Save all analysis results"""
        print("\n" + "="*80)
        print("8. SAVING RESULTS")
        print("="*80)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON results
        output_file = self.output_dir / f'gpt_analysis_{timestamp}.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
        print(f"  ✅ Saved analysis: {output_file}")
        
        # Save summary report
        report_file = self.output_dir / f'gpt_analysis_report_{timestamp}.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("GPT-4O-MINI SLOT MACHINE EXPERIMENT ANALYSIS REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            # Overall statistics
            f.write("OVERALL STATISTICS\n")
            f.write("-"*40 + "\n")
            f.write(f"Total experiments: {all_results['basic_stats']['overall']['total_experiments']}\n")
            f.write(f"Overall bankruptcy rate: {all_results['basic_stats']['overall']['overall_bankruptcy_rate']*100:.1f}%\n")
            f.write(f"Average profit/loss: ${all_results['basic_stats']['overall']['overall_avg_profit']:.2f}\n")
            f.write(f"Average rounds played: {all_results['basic_stats']['overall']['overall_avg_rounds']:.1f}\n")
            f.write("\n")
            
            # Key findings
            f.write("KEY FINDINGS\n")
            f.write("-"*40 + "\n")
            
            # Betting type effect
            bet_stats = all_results['basic_stats']['by_betting']
            fixed_bankruptcy = bet_stats['is_bankrupt']['fixed']
            variable_bankruptcy = bet_stats['is_bankrupt']['variable']
            f.write(f"1. Variable betting bankruptcy rate: {variable_bankruptcy*100:.1f}%\n")
            f.write(f"   Fixed betting bankruptcy rate: {fixed_bankruptcy*100:.1f}%\n")
            if fixed_bankruptcy > 0:
                f.write(f"   Ratio: {variable_bankruptcy/fixed_bankruptcy:.1f}x higher\n\n")
            else:
                f.write(f"   Note: Fixed betting had ZERO bankruptcies!\n\n")
            
            # Complexity effect
            f.write("2. Prompt Complexity Effect:\n")
            complexity = all_results['prompt_complexity']
            for i in range(6):
                key = str(i)
                if key in complexity:
                    rate = complexity[key]['bankruptcy_mean']
                    f.write(f"   {i} components: {rate*100:.1f}% bankruptcy\n")
            f.write("\n")
            
            # Extreme combinations
            f.write("3. Most Extreme Prompt Combinations:\n")
            for i, combo in enumerate(all_results['extreme_combinations'][:5]):
                f.write(f"   {i+1}. {combo['prompt']} + {combo['bet_type']} + {combo['first_result']}: ")
                f.write(f"{combo['difference']*100:+.1f}%p from BASE\n")
            f.write("\n")
            
            # Cognitive distortions
            f.write("4. Cognitive Distortion Patterns:\n")
            for pattern, stats in all_results['cognitive_distortions'].items():
                f.write(f"   {pattern.replace('_', ' ').title()}: {stats['percentage']:.1f}% of responses\n")
            
        print(f"  ✅ Saved report: {report_file}")
    
    def run_analysis(self):
        """Run complete analysis pipeline"""
        print("\n" + "🎰"*20)
        print("GPT-4O-MINI SLOT MACHINE EXPERIMENT ANALYSIS")
        print("🎰"*20 + "\n")
        
        # Load data
        self.load_data()
        
        # Run all analyses
        all_results = {
            'basic_stats': self.analyze_basic_statistics(),
            'prompt_complexity': self.analyze_prompt_complexity(),
            'extreme_combinations': self.analyze_extreme_combinations(),
            'cognitive_distortions': self.analyze_cognitive_distortions(),
            'loss_chasing': self.analyze_loss_chasing()
        }
        
        # Create visualizations
        self.create_visualizations()
        
        # Save results
        self.save_results(all_results)
        
        print("\n" + "="*80)
        print("✅ ANALYSIS COMPLETE!")
        print("="*80)
        
        return all_results

def main():
    """Main execution function"""
    analyzer = GPTExperimentAnalyzer()
    results = analyzer.run_analysis()
    return results

if __name__ == "__main__":
    main()