#!/usr/bin/env python3
"""
Comprehensive GPT experiment analysis
Following the order: betting type metrics → losing streaks → prompt complexity → case studies
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats

def load_gpt_data():
    """Load corrected GPT experiment results"""
    with open('/data/llm_addiction/gpt_results_corrected/gpt_corrected_complete_20250825_212628.json', 'r') as f:
        data = json.load(f)
    return data

def analyze_comprehensive_metrics(data):
    """1. Analyze comprehensive metrics by betting type, prompt, and first result"""
    print("="*80)
    print("1. COMPREHENSIVE METRICS ANALYSIS")
    print("="*80)
    
    results = data['results']
    
    # Create DataFrame for easier analysis
    df_data = []
    for exp in results:
        df_data.append({
            'bet_type': exp['bet_type'],
            'first_result': exp['first_result'],
            'prompt_combo': exp['prompt_combo'],
            'is_bankrupt': exp.get('is_bankrupt', False),
            'final_balance': exp.get('final_balance', 100),
            'total_rounds': exp.get('total_rounds', 0),
            'profit': exp.get('final_balance', 100) - 100,
            'avg_bet': np.mean([r.get('bet', 0) for r in exp.get('game_history', [])])
        })
    
    df = pd.DataFrame(df_data)
    
    # A. Betting Type Analysis
    print("\nA. BETTING TYPE ANALYSIS")
    print("-"*60)
    
    betting_metrics = df.groupby('bet_type').agg({
        'is_bankrupt': ['count', 'sum', 'mean'],
        'profit': 'mean',
        'total_rounds': 'mean',
        'avg_bet': 'mean'
    }).round(2)
    
    print(betting_metrics)
    
    # Statistical test for betting type
    fixed_bankrupt = df[df['bet_type'] == 'fixed']['is_bankrupt'].values
    variable_bankrupt = df[df['bet_type'] == 'variable']['is_bankrupt'].values
    chi2, p_bet = stats.chi2_contingency([[sum(fixed_bankrupt), len(fixed_bankrupt) - sum(fixed_bankrupt)],
                                          [sum(variable_bankrupt), len(variable_bankrupt) - sum(variable_bankrupt)]])[:2]
    print(f"\nBetting type bankruptcy rate difference p-value: {p_bet:.4f}")
    
    # B. First Result Analysis
    print("\nB. FIRST RESULT ANALYSIS")
    print("-"*60)
    
    first_result_metrics = df.groupby('first_result').agg({
        'is_bankrupt': ['count', 'sum', 'mean'],
        'profit': 'mean',
        'total_rounds': 'mean',
        'avg_bet': 'mean'
    }).round(2)
    
    print(first_result_metrics)
    
    # C. Prompt Component Analysis
    print("\nC. PROMPT COMPONENT EFFECTS")
    print("-"*60)
    
    components = ['G', 'M', 'R', 'W', 'P']
    component_effects = []
    
    for comp in components:
        with_comp = df[df['prompt_combo'].str.contains(comp)]
        without_comp = df[~df['prompt_combo'].str.contains(comp)]
        
        with_metrics = {
            'bankruptcy_rate': with_comp['is_bankrupt'].mean() * 100,
            'avg_profit': with_comp['profit'].mean(),
            'avg_rounds': with_comp['total_rounds'].mean(),
            'avg_bet': with_comp['avg_bet'].mean()
        }
        
        without_metrics = {
            'bankruptcy_rate': without_comp['is_bankrupt'].mean() * 100,
            'avg_profit': without_comp['profit'].mean(),
            'avg_rounds': without_comp['total_rounds'].mean(),
            'avg_bet': without_comp['avg_bet'].mean()
        }
        
        # Statistical test
        chi2, p_val = stats.chi2_contingency([[with_comp['is_bankrupt'].sum(), len(with_comp) - with_comp['is_bankrupt'].sum()],
                                              [without_comp['is_bankrupt'].sum(), len(without_comp) - without_comp['is_bankrupt'].sum()]])[:2]
        
        component_effects.append({
            'component': comp,
            'with_bankruptcy': with_metrics['bankruptcy_rate'],
            'without_bankruptcy': without_metrics['bankruptcy_rate'],
            'diff_bankruptcy': with_metrics['bankruptcy_rate'] - without_metrics['bankruptcy_rate'],
            'with_profit': with_metrics['avg_profit'],
            'without_profit': without_metrics['avg_profit'],
            'diff_profit': with_metrics['avg_profit'] - without_metrics['avg_profit'],
            'with_rounds': with_metrics['avg_rounds'],
            'without_rounds': without_metrics['avg_rounds'],
            'with_bet': with_metrics['avg_bet'],
            'without_bet': without_metrics['avg_bet'],
            'p_value': p_val
        })
    
    comp_df = pd.DataFrame(component_effects)
    print(comp_df.to_string(index=False))
    
    # D. Combined Analysis Table
    print("\n" + "="*80)
    print("TABLE: COMPREHENSIVE METRICS BY CONDITIONS")
    print("="*80)
    
    # Create comprehensive table
    table_data = []
    
    # By betting type
    for bet_type in ['fixed', 'variable']:
        subset = df[df['bet_type'] == bet_type]
        table_data.append({
            'Condition': f'Betting: {bet_type.capitalize()}',
            'N': len(subset),
            'Bankruptcy Rate (%)': f"{subset['is_bankrupt'].mean() * 100:.1f}",
            'Avg Profit ($)': f"{subset['profit'].mean():.2f}",
            'Avg Rounds': f"{subset['total_rounds'].mean():.1f}",
            'Avg Bet ($)': f"{subset['avg_bet'].mean():.2f}"
        })
    
    # By first result
    for first in ['W', 'L']:
        subset = df[df['first_result'] == first]
        result_name = 'Win' if first == 'W' else 'Loss'
        table_data.append({
            'Condition': f'First Game: {result_name}',
            'N': len(subset),
            'Bankruptcy Rate (%)': f"{subset['is_bankrupt'].mean() * 100:.1f}",
            'Avg Profit ($)': f"{subset['profit'].mean():.2f}",
            'Avg Rounds': f"{subset['total_rounds'].mean():.1f}",
            'Avg Bet ($)': f"{subset['avg_bet'].mean():.2f}"
        })
    
    # Top 5 highest bankruptcy prompt combinations
    prompt_bankruptcy = df.groupby('prompt_combo').agg({
        'is_bankrupt': ['count', 'sum', 'mean'],
        'profit': 'mean',
        'total_rounds': 'mean',
        'avg_bet': 'mean'
    })
    
    prompt_bankruptcy.columns = ['n', 'bankruptcies', 'bankruptcy_rate', 'avg_profit', 'avg_rounds', 'avg_bet']
    prompt_bankruptcy = prompt_bankruptcy.sort_values('bankruptcy_rate', ascending=False)
    
    print("\nTOP 5 HIGHEST RISK PROMPT COMBINATIONS:")
    for i, (prompt, row) in enumerate(prompt_bankruptcy.head(5).iterrows(), 1):
        table_data.append({
            'Condition': f'Prompt: {prompt}',
            'N': int(row['n']),
            'Bankruptcy Rate (%)': f"{row['bankruptcy_rate'] * 100:.1f}",
            'Avg Profit ($)': f"{row['avg_profit']:.2f}",
            'Avg Rounds': f"{row['avg_rounds']:.1f}",
            'Avg Bet ($)': f"{row['avg_bet']:.2f}"
        })
    
    table_df = pd.DataFrame(table_data)
    print(table_df.to_string(index=False))
    
    return df, prompt_bankruptcy

def analyze_losing_streaks(data):
    """2. Analyze behavior during losing streaks"""
    print("\n" + "="*80)
    print("2. LOSING STREAK ANALYSIS")
    print("="*80)
    
    results = data['results']
    
    streak_behaviors = {
        '3_losses': {'continue': 0, 'quit': 0, 'bet_increase': 0, 'bet_same': 0, 'bet_decrease': 0},
        '5_losses': {'continue': 0, 'quit': 0, 'bet_increase': 0, 'bet_same': 0, 'bet_decrease': 0}
    }
    
    for exp in results:
        if 'game_history' not in exp:
            continue
            
        history = exp['game_history']
        
        # Check for consecutive losses
        for i in range(len(history)):
            # Check for 3 consecutive losses
            if i >= 2:
                if all(h['result'] == 'L' for h in history[i-2:i+1]):
                    # Found 3 consecutive losses
                    if i+1 < len(history):
                        # Player continued
                        streak_behaviors['3_losses']['continue'] += 1
                        
                        # Check bet change (only for variable betting)
                        if exp['bet_type'] == 'variable' and i >= 1:
                            prev_bet = history[i]['bet']
                            next_bet = history[i+1]['bet']
                            if next_bet > prev_bet:
                                streak_behaviors['3_losses']['bet_increase'] += 1
                            elif next_bet == prev_bet:
                                streak_behaviors['3_losses']['bet_same'] += 1
                            else:
                                streak_behaviors['3_losses']['bet_decrease'] += 1
                    else:
                        # Player quit after 3 losses
                        streak_behaviors['3_losses']['quit'] += 1
            
            # Check for 5 consecutive losses
            if i >= 4:
                if all(h['result'] == 'L' for h in history[i-4:i+1]):
                    # Found 5 consecutive losses
                    if i+1 < len(history):
                        streak_behaviors['5_losses']['continue'] += 1
                        
                        # Check bet change
                        if exp['bet_type'] == 'variable' and i >= 1:
                            prev_bet = history[i]['bet']
                            next_bet = history[i+1]['bet']
                            if next_bet > prev_bet:
                                streak_behaviors['5_losses']['bet_increase'] += 1
                            elif next_bet == prev_bet:
                                streak_behaviors['5_losses']['bet_same'] += 1
                            else:
                                streak_behaviors['5_losses']['bet_decrease'] += 1
                    else:
                        streak_behaviors['5_losses']['quit'] += 1
    
    print("\nLOSING STREAK BEHAVIOR PATTERNS:")
    print("-"*60)
    
    for streak_type, behaviors in streak_behaviors.items():
        total = behaviors['continue'] + behaviors['quit']
        if total > 0:
            print(f"\n{streak_type.replace('_', ' ').title()}:")
            print(f"  Total occurrences: {total}")
            print(f"  Continue rate: {behaviors['continue']/total*100:.1f}%")
            print(f"  Quit rate: {behaviors['quit']/total*100:.1f}%")
            
            if behaviors['bet_increase'] + behaviors['bet_same'] + behaviors['bet_decrease'] > 0:
                total_bet_changes = behaviors['bet_increase'] + behaviors['bet_same'] + behaviors['bet_decrease']
                print(f"  Bet changes (variable betting only):")
                print(f"    Increase: {behaviors['bet_increase']/total_bet_changes*100:.1f}%")
                print(f"    Same: {behaviors['bet_same']/total_bet_changes*100:.1f}%")
                print(f"    Decrease: {behaviors['bet_decrease']/total_bet_changes*100:.1f}%")
    
    return streak_behaviors

def analyze_prompt_complexity(df):
    """3. Analyze bankruptcy rate by prompt complexity"""
    print("\n" + "="*80)
    print("3. PROMPT COMPLEXITY ANALYSIS")
    print("="*80)
    
    # Count number of components in each prompt
    df['prompt_complexity'] = df['prompt_combo'].apply(
        lambda x: 0 if x == 'BASE' else len(x)
    )
    
    complexity_metrics = df.groupby('prompt_complexity').agg({
        'is_bankrupt': ['count', 'sum', 'mean'],
        'profit': 'mean',
        'total_rounds': 'mean',
        'avg_bet': 'mean'
    }).round(2)
    
    complexity_metrics.columns = ['N', 'Bankruptcies', 'Bankruptcy_Rate', 'Avg_Profit', 'Avg_Rounds', 'Avg_Bet']
    
    print("\nBankruptcy Rate by Prompt Complexity:")
    print("-"*60)
    print(complexity_metrics.to_string())
    
    # Test for trend
    complexities = df['prompt_complexity'].values
    bankruptcies = df['is_bankrupt'].values
    correlation, p_value = stats.spearmanr(complexities, bankruptcies)
    print(f"\nSpearman correlation between complexity and bankruptcy: {correlation:.3f} (p={p_value:.4f})")
    
    return complexity_metrics

def analyze_bankruptcy_cases(data):
    """4. Detailed case study of top bankruptcy cases"""
    print("\n" + "="*80)
    print("4. BANKRUPTCY CASE STUDIES")
    print("="*80)
    
    results = data['results']
    
    # Find bankrupt cases with details
    bankrupt_cases = []
    for exp in results:
        if exp.get('is_bankrupt', False):
            bankrupt_cases.append({
                'exp_id': exp.get('experiment_id', 'unknown'),
                'prompt': exp['prompt_combo'],
                'bet_type': exp['bet_type'],
                'total_rounds': exp['total_rounds'],
                'first_result': exp['first_result'],
                'final_bet': exp['game_history'][-1]['bet'] if 'game_history' in exp and exp['game_history'] else 0,
                'history': exp.get('game_history', []),
                'reasoning': exp.get('reasoning_history', [])
            })
    
    # Sort by total rounds (longer games = more persistent)
    bankrupt_cases.sort(key=lambda x: x['total_rounds'], reverse=True)
    
    print(f"\nTotal bankruptcy cases: {len(bankrupt_cases)}")
    print("\nTOP 5 BANKRUPTCY CASES (by persistence):")
    print("-"*60)
    
    for i, case in enumerate(bankrupt_cases[:5], 1):
        print(f"\nCASE {i}: Experiment {case['exp_id']}")
        print(f"  Prompt: {case['prompt']}")
        print(f"  Betting type: {case['bet_type']}")
        print(f"  Total rounds: {case['total_rounds']}")
        print(f"  First result: {case['first_result']}")
        print(f"  Final bet: ${case['final_bet']}")
        
        # Analyze betting pattern
        if case['history']:
            bets = [h['bet'] for h in case['history']]
            results_pattern = ''.join([h['result'] for h in case['history'][-10:]])  # Last 10 results
            print(f"  Betting progression: ${bets[0]} → ... → ${bets[-1]}")
            print(f"  Last 10 results: {results_pattern}")
            
            # Check for all-in behavior
            if case['bet_type'] == 'variable':
                all_ins = sum(1 for h in case['history'] if h['bet'] >= h.get('balance', 100) * 0.9)
                print(f"  All-in attempts: {all_ins}")
        
        # Look for reasoning patterns if available
        if case['reasoning']:
            # This would contain actual GPT reasoning, but may not be in the data
            print(f"  Reasoning samples: [Not available in current data]")
    
    # Statistical summary of bankruptcy cases
    print("\n" + "="*80)
    print("BANKRUPTCY PATTERNS SUMMARY")
    print("-"*60)
    
    if bankrupt_cases:
        avg_rounds = np.mean([c['total_rounds'] for c in bankrupt_cases])
        variable_bankruptcies = sum(1 for c in bankrupt_cases if c['bet_type'] == 'variable')
        
        print(f"Average rounds to bankruptcy: {avg_rounds:.1f}")
        print(f"Variable betting bankruptcies: {variable_bankruptcies}/{len(bankrupt_cases)} ({variable_bankruptcies/len(bankrupt_cases)*100:.1f}%)")
        
        # Most common prompts in bankruptcies
        prompt_counts = {}
        for case in bankrupt_cases:
            prompt = case['prompt']
            prompt_counts[prompt] = prompt_counts.get(prompt, 0) + 1
        
        print("\nMost common prompts in bankruptcies:")
        for prompt, count in sorted(prompt_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {prompt}: {count} cases")
    
    return bankrupt_cases

def main():
    print("="*80)
    print("GPT CORRECTED EXPERIMENT - COMPREHENSIVE ANALYSIS")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Load data
    print("\nLoading corrected GPT data...")
    data = load_gpt_data()
    print(f"Total experiments: {len(data['results'])}")
    
    # Run analyses in order
    df, prompt_bankruptcy = analyze_comprehensive_metrics(data)
    streak_behaviors = analyze_losing_streaks(data)
    complexity_metrics = analyze_prompt_complexity(df)
    bankrupt_cases = analyze_bankruptcy_cases(data)
    
    # Save results
    analysis_results = {
        'timestamp': datetime.now().isoformat(),
        'total_experiments': len(data['results']),
        'overall_bankruptcy_rate': df['is_bankrupt'].mean() * 100,
        'streak_behaviors': streak_behaviors,
        'complexity_effects': complexity_metrics.to_dict(),
        'top_bankrupt_prompts': prompt_bankruptcy.head(10).to_dict()
    }
    
    output_file = '/home/ubuntu/llm_addiction/analysis/gpt_detailed_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    print(f"\n✅ Analysis complete! Results saved to: {output_file}")

if __name__ == "__main__":
    main()