#!/usr/bin/env python3
"""
Verify Claude comprehensive analysis table data
Generate accurate table with real experimental data
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict
import os
import warnings
warnings.filterwarnings('ignore')

def load_claude_data():
    """Load the Claude experimental data from multiple files"""
    print("Loading Claude experimental data...")

    # Load all Claude data files
    data_files = [
        '/data/llm_addiction/claude_experiment/claude_experiment_20250920_031403.json',  # Main file (62MB)
        '/data/llm_addiction/claude_experiment/claude_experiment_20250920_003210.json',  # Additional (2.7MB)
    ]

    all_experiments = []
    total_loaded = 0

    for file_path in data_files:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
            experiments = data.get('results', [])
            all_experiments.extend(experiments)
            total_loaded += len(experiments)
            print(f"Loaded {len(experiments)} experiments from {os.path.basename(file_path)}")

    print(f"Total loaded: {total_loaded} Claude experiments")
    return all_experiments

def analyze_by_bet_type(experiments):
    """Analyze results by betting type"""
    print("Analyzing by betting type...")

    bet_type_stats = {}

    for bet_type in ['fixed', 'variable']:
        type_experiments = [exp for exp in experiments if exp['bet_type'] == bet_type]

        total_experiments = len(type_experiments)
        bankruptcies = sum(1 for exp in type_experiments if exp['is_bankrupt'])
        immediate_stops = sum(1 for exp in type_experiments if exp['total_rounds'] == 0)

        # Calculate averages
        avg_rounds = np.mean([exp['total_rounds'] for exp in type_experiments])
        avg_bet = np.mean([exp['total_bet'] for exp in type_experiments])

        # Calculate net profit/loss
        net_profits = []
        for exp in type_experiments:
            initial_balance = 100  # All experiments start with $100
            final_balance = exp['final_balance']
            net_profit = final_balance - initial_balance
            net_profits.append(net_profit)

        avg_net_profit = np.mean(net_profits)

        bet_type_stats[bet_type] = {
            'total_experiments': total_experiments,
            'bankruptcies': bankruptcies,
            'bankruptcy_rate': bankruptcies / total_experiments * 100,
            'immediate_stops': immediate_stops,
            'immediate_stop_rate': immediate_stops / total_experiments * 100,
            'avg_rounds': avg_rounds,
            'avg_bet': avg_bet,
            'avg_net_profit': avg_net_profit
        }

    return bet_type_stats

def analyze_by_prompt_combo(experiments):
    """Analyze results by prompt combination"""
    print("Analyzing by prompt combination...")

    prompt_stats = defaultdict(lambda: {
        'total': 0, 'bankruptcies': 0, 'rounds': [], 'bets': [], 'profits': []
    })

    for exp in experiments:
        prompt = exp.get('prompt_combo', 'BASE')
        if not prompt:
            prompt = 'BASE'

        prompt_stats[prompt]['total'] += 1
        if exp['is_bankrupt']:
            prompt_stats[prompt]['bankruptcies'] += 1

        prompt_stats[prompt]['rounds'].append(exp['total_rounds'])
        prompt_stats[prompt]['bets'].append(exp['total_bet'])

        # Calculate net profit
        net_profit = exp['final_balance'] - 100
        prompt_stats[prompt]['profits'].append(net_profit)

    # Convert to final stats
    final_prompt_stats = {}
    for prompt, stats in prompt_stats.items():
        if stats['total'] > 0:
            final_prompt_stats[prompt] = {
                'total_experiments': stats['total'],
                'bankruptcies': stats['bankruptcies'],
                'bankruptcy_rate': stats['bankruptcies'] / stats['total'] * 100,
                'avg_rounds': np.mean(stats['rounds']),
                'avg_bet': np.mean(stats['bets']),
                'avg_net_profit': np.mean(stats['profits'])
            }

    return final_prompt_stats

def analyze_by_language(experiments):
    """Analyze results by language (Korean/English)"""
    print("Analyzing by language...")

    language_stats = {}
    korean_prompts = {'BASE', 'G', 'M', 'P', 'R', 'W', 'GM', 'GP', 'GR', 'GW', 'MP', 'MR', 'MW', 'PR', 'PW', 'RW',
                      'GMP', 'GMR', 'GMW', 'GPR', 'GPW', 'GRW', 'MPR', 'MPW', 'MRW', 'PRW', 'GMPR', 'GMPW', 'GMRW',
                      'GPRW', 'MPRW', 'GMPRW'}

    for lang in ['Korean', 'English']:
        if lang == 'Korean':
            lang_experiments = [exp for exp in experiments if exp.get('prompt_combo', 'BASE') in korean_prompts]
        else:
            lang_experiments = [exp for exp in experiments if exp.get('prompt_combo', 'BASE') not in korean_prompts]

        if len(lang_experiments) == 0:
            continue

        total_experiments = len(lang_experiments)
        bankruptcies = sum(1 for exp in lang_experiments if exp['is_bankrupt'])

        avg_rounds = np.mean([exp['total_rounds'] for exp in lang_experiments])
        avg_bet = np.mean([exp['total_bet'] for exp in lang_experiments])

        net_profits = [exp['final_balance'] - 100 for exp in lang_experiments]
        avg_net_profit = np.mean(net_profits)

        language_stats[lang] = {
            'total_experiments': total_experiments,
            'bankruptcies': bankruptcies,
            'bankruptcy_rate': bankruptcies / total_experiments * 100,
            'avg_rounds': avg_rounds,
            'avg_bet': avg_bet,
            'avg_net_profit': avg_net_profit
        }

    return language_stats

def create_comprehensive_table():
    """Create comprehensive analysis table"""
    print("Creating comprehensive Claude analysis table...")

    experiments = load_claude_data()

    # Overall statistics
    total_experiments = len(experiments)
    total_bankruptcies = sum(1 for exp in experiments if exp['is_bankrupt'])
    overall_bankruptcy_rate = total_bankruptcies / total_experiments * 100

    print(f"\n=== CLAUDE COMPREHENSIVE ANALYSIS ===")
    print(f"Total Experiments: {total_experiments:,}")
    print(f"Total Bankruptcies: {total_bankruptcies}")
    print(f"Overall Bankruptcy Rate: {overall_bankruptcy_rate:.2f}%")

    # Analysis by betting type
    bet_type_stats = analyze_by_bet_type(experiments)
    print(f"\n=== BY BETTING TYPE ===")
    for bet_type, stats in bet_type_stats.items():
        print(f"\n{bet_type.upper()} BETTING:")
        print(f"  Experiments: {stats['total_experiments']:,}")
        print(f"  Bankruptcies: {stats['bankruptcies']} ({stats['bankruptcy_rate']:.2f}%)")
        print(f"  Immediate Stops: {stats['immediate_stops']} ({stats['immediate_stop_rate']:.2f}%)")
        print(f"  Avg Rounds: {stats['avg_rounds']:.2f}")
        print(f"  Avg Total Bet: ${stats['avg_bet']:.2f}")
        print(f"  Avg Net Profit: ${stats['avg_net_profit']:.2f}")

    # Analysis by prompt combination
    prompt_stats = analyze_by_prompt_combo(experiments)
    print(f"\n=== TOP 10 RISKIEST PROMPT COMBINATIONS ===")
    sorted_prompts = sorted(prompt_stats.items(), key=lambda x: x[1]['bankruptcy_rate'], reverse=True)
    for i, (prompt, stats) in enumerate(sorted_prompts[:10]):
        print(f"{i+1:2d}. {prompt:6s}: {stats['bankruptcy_rate']:5.1f}% bankruptcy ({stats['bankruptcies']:2d}/{stats['total_experiments']:2d})")

    print(f"\n=== TOP 10 SAFEST PROMPT COMBINATIONS ===")
    for i, (prompt, stats) in enumerate(sorted_prompts[-10:]):
        print(f"{i+1:2d}. {prompt:6s}: {stats['bankruptcy_rate']:5.1f}% bankruptcy ({stats['bankruptcies']:2d}/{stats['total_experiments']:2d})")

    # Save detailed results
    output_file = '/home/ubuntu/llm_addiction/claude_experiment/results/claude_comprehensive_analysis.json'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    comprehensive_results = {
        'total_experiments': total_experiments,
        'total_bankruptcies': total_bankruptcies,
        'overall_bankruptcy_rate': overall_bankruptcy_rate,
        'bet_type_stats': bet_type_stats,
        'prompt_stats': dict(prompt_stats),
        'analysis_timestamp': '2025-09-23'
    }

    with open(output_file, 'w') as f:
        json.dump(comprehensive_results, f, indent=2)

    print(f"\n✅ Comprehensive analysis saved to: {output_file}")

    # Generate LaTeX table
    latex_file = '/home/ubuntu/llm_addiction/claude_experiment/results/claude_comprehensive_table.tex'
    with open(latex_file, 'w') as f:
        f.write("\\begin{table}[ht!]\n")
        f.write("\\centering\n")
        f.write("\\caption{Claude-3.5-Haiku Gambling Experiment Results}\n")
        f.write("\\label{tab:claude-comprehensive}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\n")
        f.write("\\textbf{Category} & \\textbf{Experiments} & \\textbf{Bankruptcies} & \\textbf{Rate (\\%)} & \\textbf{Avg Rounds} \\\\\n")
        f.write("\\midrule\n")

        # Overall
        f.write(f"Overall & {total_experiments:,} & {total_bankruptcies} & {overall_bankruptcy_rate:.1f} & {np.mean([exp['total_rounds'] for exp in experiments]):.1f} \\\\\n")

        # By betting type
        for bet_type, stats in bet_type_stats.items():
            f.write(f"{bet_type.title()} Betting & {stats['total_experiments']:,} & {stats['bankruptcies']} & {stats['bankruptcy_rate']:.1f} & {stats['avg_rounds']:.1f} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"✅ LaTeX table saved to: {latex_file}")
    return comprehensive_results

if __name__ == "__main__":
    create_comprehensive_table()