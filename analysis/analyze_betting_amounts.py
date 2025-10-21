#!/usr/bin/env python3
"""
Comprehensive Betting Amount Analysis for LLaMA and GPT Experiments
Analyzes betting patterns, amounts, and their relationship with bankruptcy
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class BettingAmountAnalyzer:
    def __init__(self):
        self.llama_file = Path('/data/llm_addiction/results/exp1_multiround_intermediate_20250819_140040.json')
        self.gpt_file = Path('/data/llm_addiction/gpt_results/gpt_multiround_intermediate_20250820_025809.json')
        
    def analyze_llama_betting(self):
        """Analyze LLaMA betting amounts by prompt and outcome"""
        print("="*80)
        print("LLAMA BETTING AMOUNT ANALYSIS")
        print("="*80)
        
        with open(self.llama_file, 'r') as f:
            data = json.load(f)
        
        results = []
        for exp in data['results']:
            # Extract betting amounts
            bets = []
            for round_data in exp['round_history']:
                if round_data['action'] == 'bet':
                    bets.append(round_data['amount'])
            
            if bets:
                results.append({
                    'prompt': exp['prompt_combo'],
                    'bet_type': exp['bet_type'],
                    'is_bankrupt': exp['is_bankrupt'],
                    'total_rounds': exp['total_rounds'],
                    'avg_bet': np.mean(bets),
                    'max_bet': max(bets),
                    'min_bet': min(bets),
                    'std_bet': np.std(bets),
                    'final_bet': bets[-1] if bets else 0,
                    'increasing_pattern': 1 if len(bets) > 1 and bets[-1] > bets[0] else 0
                })
        
        df = pd.DataFrame(results)
        
        # Analysis by prompt
        print("\n📊 Top 10 Prompts by Average Bet Amount (Variable Betting Only):")
        variable_df = df[df['bet_type'] == 'variable']
        
        prompt_stats = variable_df.groupby('prompt').agg({
            'avg_bet': 'mean',
            'max_bet': 'mean',
            'total_rounds': 'mean',
            'is_bankrupt': 'mean',
            'increasing_pattern': 'mean'
        }).round(2)
        
        prompt_stats = prompt_stats.sort_values('avg_bet', ascending=False)
        
        print(f"{'Prompt':<10} {'Avg Bet':<10} {'Avg Max':<10} {'Rounds':<8} {'Bank %':<8} {'Incr %':<8}")
        print("-"*60)
        for prompt, row in prompt_stats.head(10).iterrows():
            print(f"{prompt:<10} ${row['avg_bet']:<9.1f} ${row['max_bet']:<9.1f} "
                  f"{row['total_rounds']:<7.1f} {row['is_bankrupt']*100:<7.1f}% "
                  f"{row['increasing_pattern']*100:<7.1f}%")
        
        # Bankruptcy correlation
        print("\n📈 Betting Patterns and Bankruptcy:")
        bankrupt_df = df[df['is_bankrupt'] == True]
        safe_df = df[df['is_bankrupt'] == False]
        
        print(f"Bankrupt games - Avg bet: ${bankrupt_df['avg_bet'].mean():.1f} (±{bankrupt_df['avg_bet'].std():.1f})")
        print(f"Safe games - Avg bet: ${safe_df['avg_bet'].mean():.1f} (±{safe_df['avg_bet'].std():.1f})")
        
        # Statistical test
        if len(bankrupt_df) > 0 and len(safe_df) > 0:
            t_stat, p_value = stats.ttest_ind(bankrupt_df['avg_bet'], safe_df['avg_bet'])
            print(f"T-test: t={t_stat:.3f}, p={p_value:.6f}")
        
        # Progressive betting analysis
        print(f"\n🎰 Progressive Betting Patterns:")
        print(f"Games with increasing bets: {df['increasing_pattern'].sum()} ({df['increasing_pattern'].mean()*100:.1f}%)")
        print(f"Bankruptcy rate when increasing: {df[df['increasing_pattern']==1]['is_bankrupt'].mean()*100:.1f}%")
        print(f"Bankruptcy rate when not increasing: {df[df['increasing_pattern']==0]['is_bankrupt'].mean()*100:.1f}%")
        
        return df
    
    def analyze_gpt_betting(self):
        """Analyze GPT betting amounts"""
        print("\n" + "="*80)
        print("GPT-4O-MINI BETTING AMOUNT ANALYSIS")
        print("="*80)
        
        with open(self.gpt_file, 'r') as f:
            data = json.load(f)
        
        results = []
        for exp in data['results']:
            # Extract betting amounts
            bets = []
            for round_data in exp['round_history']:
                if 'bet_amount' in round_data:
                    bets.append(round_data['bet_amount'])
            
            if bets:
                results.append({
                    'prompt': exp['prompt_combo'],
                    'bet_type': exp['bet_type'],
                    'is_bankrupt': exp['is_bankrupt'],
                    'total_rounds': exp['total_rounds'],
                    'avg_bet': np.mean(bets),
                    'max_bet': max(bets),
                    'min_bet': min(bets),
                    'std_bet': np.std(bets),
                    'final_bet': bets[-1] if bets else 0,
                    'all_in_count': sum(1 for b in bets if b >= 80)
                })
        
        df = pd.DataFrame(results)
        
        # Analysis by prompt (variable only)
        print("\n📊 Top 10 Prompts by Average Bet Amount (Variable Betting Only):")
        variable_df = df[df['bet_type'] == 'variable']
        
        if len(variable_df) > 0:
            prompt_stats = variable_df.groupby('prompt').agg({
                'avg_bet': 'mean',
                'max_bet': 'mean',
                'total_rounds': 'mean',
                'is_bankrupt': 'mean',
                'all_in_count': 'mean'
            }).round(2)
            
            prompt_stats = prompt_stats.sort_values('avg_bet', ascending=False)
            
            print(f"{'Prompt':<10} {'Avg Bet':<10} {'Avg Max':<10} {'Rounds':<8} {'Bank %':<8} {'All-ins':<8}")
            print("-"*60)
            for prompt, row in prompt_stats.head(10).iterrows():
                print(f"{prompt:<10} ${row['avg_bet']:<9.1f} ${row['max_bet']:<9.1f} "
                      f"{row['total_rounds']:<7.1f} {row['is_bankrupt']*100:<7.1f}% "
                      f"{row['all_in_count']:<7.1f}")
        
        # All-in behavior
        print(f"\n💸 All-in Behavior (bet >= $80):")
        all_in_games = df[df['all_in_count'] > 0]
        print(f"Games with all-ins: {len(all_in_games)} ({len(all_in_games)/len(df)*100:.1f}%)")
        if len(all_in_games) > 0:
            print(f"Bankruptcy rate with all-ins: {all_in_games['is_bankrupt'].mean()*100:.1f}%")
            print(f"Avg all-ins per game: {all_in_games['all_in_count'].mean():.1f}")
        
        return df
    
    def compare_models(self, llama_df, gpt_df):
        """Compare betting patterns between models"""
        print("\n" + "="*80)
        print("MODEL COMPARISON: LLAMA vs GPT")
        print("="*80)
        
        # Overall statistics
        print("\n📊 Overall Betting Statistics:")
        print(f"{'Metric':<25} {'LLaMA':<15} {'GPT-4o-mini':<15}")
        print("-"*55)
        
        metrics = [
            ('Average bet amount', llama_df['avg_bet'].mean(), gpt_df['avg_bet'].mean()),
            ('Std dev of bets', llama_df['std_bet'].mean(), gpt_df['std_bet'].mean()),
            ('Average max bet', llama_df['max_bet'].mean(), gpt_df['max_bet'].mean()),
            ('Bankruptcy rate', llama_df['is_bankrupt'].mean()*100, gpt_df['is_bankrupt'].mean()*100),
            ('Avg rounds played', llama_df['total_rounds'].mean(), gpt_df['total_rounds'].mean())
        ]
        
        for name, llama_val, gpt_val in metrics:
            print(f"{name:<25} {llama_val:<14.2f} {gpt_val:<14.2f}")
        
        # Variable betting comparison
        print("\n🎲 Variable Betting Comparison:")
        llama_var = llama_df[llama_df['bet_type'] == 'variable']
        gpt_var = gpt_df[gpt_df['bet_type'] == 'variable']
        
        if len(llama_var) > 0 and len(gpt_var) > 0:
            print(f"LLaMA variable avg bet: ${llama_var['avg_bet'].mean():.1f}")
            print(f"GPT variable avg bet: ${gpt_var['avg_bet'].mean():.1f}")
            print(f"LLaMA variable bankruptcy: {llama_var['is_bankrupt'].mean()*100:.1f}%")
            print(f"GPT variable bankruptcy: {gpt_var['is_bankrupt'].mean()*100:.1f}%")
        
        # Risk-taking comparison
        print("\n⚠️ Risk-Taking Indicators:")
        llama_risky = llama_df[llama_df['max_bet'] >= 50]
        gpt_risky = gpt_df[gpt_df['max_bet'] >= 50]
        
        print(f"LLaMA high-bet games (>=$50): {len(llama_risky)} ({len(llama_risky)/len(llama_df)*100:.1f}%)")
        print(f"GPT high-bet games (>=$50): {len(gpt_risky)} ({len(gpt_risky)/len(gpt_df)*100:.1f}%)")

def main():
    analyzer = BettingAmountAnalyzer()
    
    # Analyze each model
    llama_df = analyzer.analyze_llama_betting()
    gpt_df = analyzer.analyze_gpt_betting()
    
    # Compare models
    analyzer.compare_models(llama_df, gpt_df)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()