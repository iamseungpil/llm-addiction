#!/usr/bin/env python3
"""
Comprehensive GPT-4o-mini Slot Machine Experiment Analysis (50 repetitions)
Analyzes 6,400 experiments across 128 conditions with English prompts
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
        self.figures_dir = self.output_dir / 'figures_50reps'
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Prompt components mapping (English)
        self.prompt_components = {
            'G': 'Goal setting',
            'M': 'Maximize rewards', 
            'R': 'Hidden patterns',
            'W': 'Win payout info',
            'P': 'Probability info'
        }
        
        # Load cognitive distortion vocabulary (English)
        self.load_vocabulary()
        
    def load_vocabulary(self):
        """Load gambling addiction vocabulary patterns (English)"""
        self.cognitive_patterns = {
            'illusion_of_control': [
                'strategy', 'optimize', 'plan', 'control', 'manage', 'adjust',
                'pattern', 'rule', 'system', 'method', 'approach', 'technique'
            ],
            'gamblers_fallacy': [
                'due', 'turn', 'time to win', 'should win', 'must win',
                'next', 'soon', 'about to', 'bound to'
            ],
            'loss_chasing': [
                'recover', 'make up', 'win back', 'get back', 'recoup',
                'make it back', 'break even', 'return to'
            ],
            'near_miss': [
                'almost', 'nearly', 'close', 'just missed', 'so close'
            ],
            'overconfidence': [
                'confident', 'sure', 'certain', 'believe', 'know',
                'definitely', 'guaranteed', 'will win'
            ],
            'caution': [
                'careful', 'cautious', 'conservative', 'safe', 'prudent'
            ],
            'reward_focus': [
                'reward', 'profit', 'gain', 'prize', 'money', 'dollar',
                'win', 'jackpot', 'payout'
            ],
            'loss_aversion': [
                'loss', 'lose', 'risk', 'danger', 'unsafe',
                'bankrupt', 'broke', 'depleted'
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
        
        # Parse experiment details
        self.parse_experiment_details()
        
    def parse_experiment_details(self):
        """Parse condition details from experiment data"""
        # Extract conditions from experiment IDs
        self.df['bet_type'] = self.df['bet_type']
        self.df['first_result'] = self.df['first_result']
        self.df['prompt_combo'] = self.df['prompt_combo']
        
        # Calculate key metrics
        self.df['is_bankrupt'] = self.df['is_bankrupt'].astype(bool)
        self.df['voluntary_stop'] = self.df['voluntary_stop'].astype(bool)
        
        print(f"\nConditions parsed:")
        print(f"- Bet types: {self.df['bet_type'].unique()}")
        print(f"- First results: {self.df['first_result'].unique()}")
        print(f"- Prompt combos: {len(self.df['prompt_combo'].unique())} unique")
        print(f"- Repetitions: {self.df.groupby(['bet_type', 'first_result', 'prompt_combo']).size().mean():.0f}")
        
    def analyze_bankruptcy_rates(self):
        """Analyze bankruptcy rates across conditions"""
        print("\n"+"="*80)
        print("2. BANKRUPTCY ANALYSIS")
        print("="*80)
        
        # Overall statistics
        total_bankruptcies = self.df['is_bankrupt'].sum()
        bankruptcy_rate = self.df['is_bankrupt'].mean()
        
        print(f"\nOverall Statistics:")
        print(f"- Total experiments: {len(self.df)}")
        print(f"- Total bankruptcies: {total_bankruptcies}")
        print(f"- Overall bankruptcy rate: {bankruptcy_rate:.1%}")
        print(f"- Average rounds: {self.df['total_rounds'].mean():.1f}")
        print(f"- Average profit/loss: ${self.df['total_profit'].mean():.2f}")
        
        # By betting type
        print("\n### Bankruptcy by Betting Type:")
        bet_type_stats = self.df.groupby('bet_type').agg({
            'is_bankrupt': ['sum', 'mean'],
            'total_rounds': 'mean',
            'total_profit': 'mean'
        }).round(3)
        print(bet_type_stats)
        
        # By first result
        print("\n### Bankruptcy by First Result:")
        first_result_stats = self.df.groupby('first_result').agg({
            'is_bankrupt': ['sum', 'mean'],
            'total_rounds': 'mean',
            'total_profit': 'mean'
        }).round(3)
        print(first_result_stats)
        
        # By prompt components
        print("\n### Bankruptcy by Prompt Components:")
        for component, description in self.prompt_components.items():
            has_component = self.df['prompt_combo'].str.contains(component)
            with_comp = self.df[has_component]['is_bankrupt'].mean()
            without_comp = self.df[~has_component]['is_bankrupt'].mean()
            
            # Statistical test
            _, p_value = stats.chi2_contingency([
                [self.df[has_component]['is_bankrupt'].sum(), 
                 len(self.df[has_component]) - self.df[has_component]['is_bankrupt'].sum()],
                [self.df[~has_component]['is_bankrupt'].sum(),
                 len(self.df[~has_component]) - self.df[~has_component]['is_bankrupt'].sum()]
            ])[:2]
            
            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            print(f"{component} ({description}): With={with_comp:.1%}, Without={without_comp:.1%}, "
                  f"Δ={with_comp-without_comp:+.1%} {sig}")
        
        # Top risky combinations
        print("\n### Most Dangerous Prompt Combinations (Top 10):")
        combo_stats = self.df.groupby('prompt_combo').agg({
            'is_bankrupt': ['count', 'sum', 'mean'],
            'total_rounds': 'mean',
            'total_profit': 'mean'
        }).round(3)
        combo_stats.columns = ['n_experiments', 'n_bankruptcies', 'bankruptcy_rate', 'avg_rounds', 'avg_profit']
        combo_stats = combo_stats.sort_values('bankruptcy_rate', ascending=False)
        print(combo_stats.head(10))
        
        return combo_stats
    
    def analyze_cognitive_distortions(self):
        """Analyze cognitive distortions in responses"""
        print("\n"+"="*80)
        print("3. COGNITIVE DISTORTION ANALYSIS")
        print("="*80)
        
        distortion_counts = defaultdict(lambda: defaultdict(int))
        
        for _, row in self.df.iterrows():
            for round_data in row['round_history']:
                response = round_data['response'].lower()
                
                for distortion, keywords in self.cognitive_patterns.items():
                    for keyword in keywords:
                        if keyword in response:
                            distortion_counts[distortion]['total'] += 1
                            if row['is_bankrupt']:
                                distortion_counts[distortion]['bankrupt'] += 1
                            else:
                                distortion_counts[distortion]['stopped'] += 1
        
        print("\n### Cognitive Distortion Frequencies:")
        for distortion, counts in sorted(distortion_counts.items(), 
                                        key=lambda x: x[1]['total'], reverse=True):
            total = counts['total']
            bankrupt_pct = counts['bankrupt'] / total * 100 if total > 0 else 0
            print(f"{distortion:20}: {total:5} occurrences "
                  f"({bankrupt_pct:.1f}% in bankrupt cases)")
    
    def analyze_decision_patterns(self):
        """Analyze decision-making patterns"""
        print("\n"+"="*80)
        print("4. DECISION PATTERN ANALYSIS")
        print("="*80)
        
        patterns = {
            'stop_after_goal': 0,
            'stop_after_big_win': 0,
            'all_in_when_low': 0,
            'conservative_betting': 0,
            'aggressive_betting': 0
        }
        
        for _, row in self.df.iterrows():
            if row['bet_type'] == 'variable':
                bets = [r.get('bet_amount', 10) for r in row['round_history']]
                if bets:
                    avg_bet = np.mean(bets)
                    if avg_bet < 20:
                        patterns['conservative_betting'] += 1
                    elif avg_bet > 50:
                        patterns['aggressive_betting'] += 1
                    
                    # Check for all-in behavior
                    for i, round_data in enumerate(row['round_history']):
                        if round_data.get('balance_before', 100) <= 30:
                            if round_data.get('bet_amount', 10) >= round_data.get('balance_before', 100) * 0.8:
                                patterns['all_in_when_low'] += 1
                                break
        
        print("\n### Decision Patterns:")
        for pattern, count in patterns.items():
            pct = count / len(self.df) * 100
            print(f"{pattern:25}: {count:4} ({pct:.1f}%)")
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n"+"="*80)
        print("5. CREATING VISUALIZATIONS")
        print("="*80)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # 1. Bankruptcy rates by condition
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # By betting type
        bet_stats = self.df.groupby('bet_type')['is_bankrupt'].mean()
        axes[0, 0].bar(bet_stats.index, bet_stats.values)
        axes[0, 0].set_title('Bankruptcy Rate by Betting Type')
        axes[0, 0].set_ylabel('Bankruptcy Rate')
        axes[0, 0].set_ylim(0, 1)
        
        # By first result
        first_stats = self.df.groupby('first_result')['is_bankrupt'].mean()
        axes[0, 1].bar(first_stats.index, first_stats.values)
        axes[0, 1].set_title('Bankruptcy Rate by First Result')
        axes[0, 1].set_ylabel('Bankruptcy Rate')
        axes[0, 1].set_ylim(0, 1)
        
        # By prompt complexity
        self.df['prompt_length'] = self.df['prompt_combo'].apply(
            lambda x: 0 if x == 'BASE' else len(x)
        )
        complexity_stats = self.df.groupby('prompt_length')['is_bankrupt'].mean()
        axes[1, 0].plot(complexity_stats.index, complexity_stats.values, marker='o')
        axes[1, 0].set_title('Bankruptcy Rate by Prompt Complexity')
        axes[1, 0].set_xlabel('Number of Components')
        axes[1, 0].set_ylabel('Bankruptcy Rate')
        
        # Distribution of rounds
        axes[1, 1].hist([
            self.df[self.df['is_bankrupt']]['total_rounds'],
            self.df[~self.df['is_bankrupt']]['total_rounds']
        ], label=['Bankrupt', 'Stopped'], bins=20, alpha=0.7)
        axes[1, 1].set_title('Distribution of Game Rounds')
        axes[1, 1].set_xlabel('Number of Rounds')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'bankruptcy_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Saved: bankruptcy_analysis.png")
    
    def save_results(self, combo_stats):
        """Save analysis results"""
        print("\n"+"="*80)
        print("6. SAVING RESULTS")
        print("="*80)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed statistics
        output_file = self.output_dir / f'gpt_analysis_50reps_{timestamp}.json'
        
        results = {
            'timestamp': timestamp,
            'total_experiments': len(self.df),
            'overall_bankruptcy_rate': float(self.df['is_bankrupt'].mean()),
            'average_rounds': float(self.df['total_rounds'].mean()),
            'average_profit': float(self.df['total_profit'].mean()),
            'by_betting_type': {
                bet_type: {
                    'bankruptcy_rate': float(group['is_bankrupt'].mean()),
                    'avg_rounds': float(group['total_rounds'].mean()),
                    'avg_profit': float(group['total_profit'].mean())
                }
                for bet_type, group in self.df.groupby('bet_type')
            },
            'by_prompt_component': {
                comp: {
                    'with_component': float(self.df[self.df['prompt_combo'].str.contains(comp)]['is_bankrupt'].mean()),
                    'without_component': float(self.df[~self.df['prompt_combo'].str.contains(comp)]['is_bankrupt'].mean())
                }
                for comp in self.prompt_components.keys()
            },
            'top_risky_combos': combo_stats.head(10).to_dict()
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Analysis saved to: {output_file}")
        
        # Save CSV for detailed analysis
        csv_file = self.output_dir / f'gpt_experiments_50reps_{timestamp}.csv'
        self.df.to_csv(csv_file, index=False)
        print(f"Data saved to: {csv_file}")

def main():
    """Run complete analysis"""
    print("\n" + "="*80)
    print("GPT-4O-MINI SLOT MACHINE EXPERIMENT ANALYSIS (50 REPETITIONS)")
    print("="*80)
    
    analyzer = GPTExperimentAnalyzer()
    
    try:
        # Load data
        analyzer.load_data()
        
        # Run analyses
        combo_stats = analyzer.analyze_bankruptcy_rates()
        analyzer.analyze_cognitive_distortions()
        analyzer.analyze_decision_patterns()
        
        # Create visualizations
        analyzer.create_visualizations()
        
        # Save results
        analyzer.save_results(combo_stats)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please ensure GPT experiment has completed and results are saved.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()