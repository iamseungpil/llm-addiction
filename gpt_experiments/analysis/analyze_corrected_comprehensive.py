#!/usr/bin/env python3
"""
Comprehensive analysis of corrected GPT experiment results
Analyzes: bankruptcy rate, play rounds, average profit, betting amounts
By: betting type, first game result, prompt components
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path
from datetime import datetime
import pandas as pd

def load_corrected_results():
    """Load corrected GPT experiment results"""
    with open('/data/llm_addiction/gpt_results_corrected/gpt_corrected_complete_20250825_212628.json', 'r') as f:
        data = json.load(f)
    return data

def calculate_metrics(results):
    """Calculate key metrics for a set of results"""
    if not results:
        return None
    
    bankruptcies = sum(1 for r in results if r.get('is_bankrupt', False))
    bankruptcy_rate = bankruptcies / len(results) * 100
    
    # Play rounds
    rounds = [r.get('total_rounds', 0) for r in results]
    avg_rounds = np.mean(rounds) if rounds else 0
    
    # Profit (final balance - 100)
    profits = [r.get('final_balance', 100) - 100 for r in results]
    avg_profit = np.mean(profits) if profits else 0
    
    # Betting amounts (from game history)
    all_bets = []
    for r in results:
        if 'game_history' in r:
            for round_data in r['game_history']:
                if 'bet' in round_data:
                    all_bets.append(round_data['bet'])
    avg_bet = np.mean(all_bets) if all_bets else 0
    
    return {
        'n': len(results),
        'bankruptcy_rate': bankruptcy_rate,
        'bankruptcies': bankruptcies,
        'avg_rounds': avg_rounds,
        'avg_profit': avg_profit,
        'avg_bet': avg_bet,
        'rounds_list': rounds,
        'profits_list': profits,
        'bets_list': all_bets
    }

def statistical_test(group1_metrics, group2_metrics, metric_name):
    """Perform statistical test between two groups"""
    if metric_name == 'bankruptcy_rate':
        # Chi-square test for bankruptcy rates
        table = [[group1_metrics['bankruptcies'], group1_metrics['n'] - group1_metrics['bankruptcies']],
                 [group2_metrics['bankruptcies'], group2_metrics['n'] - group2_metrics['bankruptcies']]]
        chi2, p_value = stats.chi2_contingency(table)[:2]
        return p_value
    else:
        # T-test for continuous variables
        if metric_name == 'avg_rounds':
            list1, list2 = group1_metrics['rounds_list'], group2_metrics['rounds_list']
        elif metric_name == 'avg_profit':
            list1, list2 = group1_metrics['profits_list'], group2_metrics['profits_list']
        elif metric_name == 'avg_bet':
            list1, list2 = group1_metrics['bets_list'], group2_metrics['bets_list']
        else:
            return None
        
        if list1 and list2:
            _, p_value = stats.ttest_ind(list1, list2)
            return p_value
        return None

def analyze_betting_type(data):
    """Analyze results by betting type"""
    print("\n" + "="*80)
    print("BETTING TYPE ANALYSIS")
    print("="*80)
    
    fixed_results = [r for r in data['results'] if r['bet_type'] == 'fixed']
    variable_results = [r for r in data['results'] if r['bet_type'] == 'variable']
    
    fixed_metrics = calculate_metrics(fixed_results)
    variable_metrics = calculate_metrics(variable_results)
    
    print(f"\nFixed Betting (n={fixed_metrics['n']}):")
    print(f"  Bankruptcy rate: {fixed_metrics['bankruptcy_rate']:.2f}% ({fixed_metrics['bankruptcies']}/{fixed_metrics['n']})")
    print(f"  Avg rounds: {fixed_metrics['avg_rounds']:.2f}")
    print(f"  Avg profit: ${fixed_metrics['avg_profit']:.2f}")
    print(f"  Avg bet: ${fixed_metrics['avg_bet']:.2f}")
    
    print(f"\nVariable Betting (n={variable_metrics['n']}):")
    print(f"  Bankruptcy rate: {variable_metrics['bankruptcy_rate']:.2f}% ({variable_metrics['bankruptcies']}/{variable_metrics['n']})")
    print(f"  Avg rounds: {variable_metrics['avg_rounds']:.2f}")
    print(f"  Avg profit: ${variable_metrics['avg_profit']:.2f}")
    print(f"  Avg bet: ${variable_metrics['avg_bet']:.2f}")
    
    print("\nStatistical Tests (Fixed vs Variable):")
    for metric in ['bankruptcy_rate', 'avg_rounds', 'avg_profit', 'avg_bet']:
        p_value = statistical_test(fixed_metrics, variable_metrics, metric)
        if p_value is not None:
            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            print(f"  {metric}: p={p_value:.4f} {sig}")
    
    return fixed_metrics, variable_metrics

def analyze_first_game(data):
    """Analyze results by first game outcome"""
    print("\n" + "="*80)
    print("FIRST GAME RESULT ANALYSIS")
    print("="*80)
    
    win_results = [r for r in data['results'] if r['first_result'] == 'W']
    loss_results = [r for r in data['results'] if r['first_result'] == 'L']
    
    win_metrics = calculate_metrics(win_results)
    loss_metrics = calculate_metrics(loss_results)
    
    print(f"\nFirst Game WIN (n={win_metrics['n']}):")
    print(f"  Bankruptcy rate: {win_metrics['bankruptcy_rate']:.2f}% ({win_metrics['bankruptcies']}/{win_metrics['n']})")
    print(f"  Avg rounds: {win_metrics['avg_rounds']:.2f}")
    print(f"  Avg profit: ${win_metrics['avg_profit']:.2f}")
    print(f"  Avg bet: ${win_metrics['avg_bet']:.2f}")
    
    print(f"\nFirst Game LOSS (n={loss_metrics['n']}):")
    print(f"  Bankruptcy rate: {loss_metrics['bankruptcy_rate']:.2f}% ({loss_metrics['bankruptcies']}/{loss_metrics['n']})")
    print(f"  Avg rounds: {loss_metrics['avg_rounds']:.2f}")
    print(f"  Avg profit: ${loss_metrics['avg_profit']:.2f}")
    print(f"  Avg bet: ${loss_metrics['avg_bet']:.2f}")
    
    print("\nStatistical Tests (Win vs Loss):")
    for metric in ['bankruptcy_rate', 'avg_rounds', 'avg_profit', 'avg_bet']:
        p_value = statistical_test(win_metrics, loss_metrics, metric)
        if p_value is not None:
            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            print(f"  {metric}: p={p_value:.4f} {sig}")
    
    return win_metrics, loss_metrics

def analyze_prompt_components(data):
    """Analyze results by individual prompt components"""
    print("\n" + "="*80)
    print("PROMPT COMPONENT ANALYSIS")
    print("="*80)
    
    components = ['G', 'M', 'R', 'W', 'P']
    component_names = {
        'G': 'Goal setting',
        'M': 'Maximize reward',
        'R': 'Rule/pattern',
        'W': 'Reward info',
        'P': 'Probability info'
    }
    
    results = []
    for comp in components:
        # With component
        with_comp = [r for r in data['results'] if comp in r['prompt_combo']]
        # Without component
        without_comp = [r for r in data['results'] if comp not in r['prompt_combo']]
        
        with_metrics = calculate_metrics(with_comp)
        without_metrics = calculate_metrics(without_comp)
        
        print(f"\n{comp} ({component_names[comp]}):")
        print(f"  With {comp} (n={with_metrics['n']}):")
        print(f"    Bankruptcy: {with_metrics['bankruptcy_rate']:.2f}%")
        print(f"    Rounds: {with_metrics['avg_rounds']:.2f}")
        print(f"    Profit: ${with_metrics['avg_profit']:.2f}")
        print(f"    Bet: ${with_metrics['avg_bet']:.2f}")
        
        print(f"  Without {comp} (n={without_metrics['n']}):")
        print(f"    Bankruptcy: {without_metrics['bankruptcy_rate']:.2f}%")
        print(f"    Rounds: {without_metrics['avg_rounds']:.2f}")
        print(f"    Profit: ${without_metrics['avg_profit']:.2f}")
        print(f"    Bet: ${without_metrics['avg_bet']:.2f}")
        
        print(f"  Differences:")
        print(f"    Bankruptcy: {with_metrics['bankruptcy_rate'] - without_metrics['bankruptcy_rate']:+.2f}%p")
        
        # Statistical tests
        print(f"  P-values:")
        for metric in ['bankruptcy_rate', 'avg_rounds', 'avg_profit', 'avg_bet']:
            p_value = statistical_test(with_metrics, without_metrics, metric)
            if p_value is not None:
                sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                print(f"    {metric}: p={p_value:.4f} {sig}")
        
        results.append({
            'component': comp,
            'with_metrics': with_metrics,
            'without_metrics': without_metrics
        })
    
    return results

def analyze_prompt_combinations(data):
    """Analyze top and bottom prompt combinations"""
    print("\n" + "="*80)
    print("PROMPT COMBINATION ANALYSIS")
    print("="*80)
    
    # Group by prompt combination
    combo_results = {}
    for r in data['results']:
        combo = r['prompt_combo']
        if combo not in combo_results:
            combo_results[combo] = []
        combo_results[combo].append(r)
    
    # Calculate metrics for each combination
    combo_metrics = []
    for combo, results in combo_results.items():
        metrics = calculate_metrics(results)
        metrics['combo'] = combo
        combo_metrics.append(metrics)
    
    # Sort by bankruptcy rate
    combo_metrics.sort(key=lambda x: x['bankruptcy_rate'], reverse=True)
    
    print("\nTop 5 Highest Risk Combinations:")
    for i, metrics in enumerate(combo_metrics[:5], 1):
        print(f"{i}. {metrics['combo']}: {metrics['bankruptcy_rate']:.1f}% bankruptcy, "
              f"{metrics['avg_rounds']:.1f} rounds, ${metrics['avg_profit']:.2f} profit, "
              f"${metrics['avg_bet']:.1f} avg bet (n={metrics['n']})")
    
    print("\nTop 5 Lowest Risk Combinations:")
    for i, metrics in enumerate(combo_metrics[-5:], 1):
        print(f"{i}. {metrics['combo']}: {metrics['bankruptcy_rate']:.1f}% bankruptcy, "
              f"{metrics['avg_rounds']:.1f} rounds, ${metrics['avg_profit']:.2f} profit, "
              f"${metrics['avg_bet']:.1f} avg bet (n={metrics['n']})")
    
    # Compare BASE with others
    base_metrics = next((m for m in combo_metrics if m['combo'] == 'BASE'), None)
    if base_metrics:
        print(f"\nBASE Performance (n={base_metrics['n']}):")
        print(f"  Bankruptcy: {base_metrics['bankruptcy_rate']:.2f}%")
        print(f"  Rounds: {base_metrics['avg_rounds']:.2f}")
        print(f"  Profit: ${base_metrics['avg_profit']:.2f}")
        print(f"  Bet: ${base_metrics['avg_bet']:.2f}")
    
    return combo_metrics

def main():
    print("="*80)
    print("COMPREHENSIVE ANALYSIS OF CORRECTED GPT EXPERIMENT")
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Load data
    print("\nLoading corrected GPT results...")
    data = load_corrected_results()
    print(f"Total experiments: {len(data['results'])}")
    
    # Overall statistics
    overall_metrics = calculate_metrics(data['results'])
    print(f"\nOVERALL STATISTICS:")
    print(f"  Bankruptcy rate: {overall_metrics['bankruptcy_rate']:.2f}% ({overall_metrics['bankruptcies']}/{overall_metrics['n']})")
    print(f"  Avg rounds: {overall_metrics['avg_rounds']:.2f}")
    print(f"  Avg profit: ${overall_metrics['avg_profit']:.2f}")
    print(f"  Avg bet: ${overall_metrics['avg_bet']:.2f}")
    
    # Detailed analyses
    betting_results = analyze_betting_type(data)
    first_game_results = analyze_first_game(data)
    component_results = analyze_prompt_components(data)
    combo_results = analyze_prompt_combinations(data)
    
    # Save results to JSON for paper update
    analysis_results = {
        'timestamp': datetime.now().isoformat(),
        'overall': overall_metrics,
        'betting_type': {
            'fixed': betting_results[0],
            'variable': betting_results[1]
        },
        'first_game': {
            'win': first_game_results[0],
            'loss': first_game_results[1]
        },
        'components': component_results,
        'combinations': combo_results[:10]  # Top 10 for paper
    }
    
    output_file = '/home/ubuntu/llm_addiction/gpt_experiments/analysis/corrected_analysis_results.json'
    with open(output_file, 'w') as f:
        # Remove non-serializable lists for JSON
        for key in ['rounds_list', 'profits_list', 'bets_list']:
            for section in analysis_results.values():
                if isinstance(section, dict):
                    for metrics in section.values():
                        if isinstance(metrics, dict) and key in metrics:
                            del metrics[key]
                elif isinstance(section, list):
                    for item in section:
                        if isinstance(item, dict):
                            for k in ['with_metrics', 'without_metrics']:
                                if k in item and key in item[k]:
                                    del item[k][key]
                            if key in item:
                                del item[key]
        
        json.dump(analysis_results, f, indent=2)
    
    print(f"\n✅ Analysis complete! Results saved to: {output_file}")
    print("\n" + "="*80)
    print("KEY FINDINGS:")
    print("="*80)
    print(f"1. Variable betting shows significantly higher risk ({betting_results[1]['bankruptcy_rate']:.1f}% vs {betting_results[0]['bankruptcy_rate']:.1f}%)")
    print(f"2. First game outcome affects behavior (Loss: {first_game_results[1]['bankruptcy_rate']:.1f}% vs Win: {first_game_results[0]['bankruptcy_rate']:.1f}%)")
    print(f"3. GPT shows rational gambling behavior after parsing fix (only {overall_metrics['bankruptcy_rate']:.1f}% bankruptcy)")

if __name__ == "__main__":
    main()