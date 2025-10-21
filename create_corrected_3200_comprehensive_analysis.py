#!/usr/bin/env python3
"""
GPT Corrected 3200 Comprehensive Analysis
=======================================

Creates comprehensive analysis table matching the LaTeX format
using the corrected 3,200 GPT experiment dataset.

Author: Claude Code Analysis
Date: 2025-09-14
"""

import json
import pandas as pd
from collections import defaultdict
import numpy as np
from scipy import stats

def load_gpt_3200_data():
    """Load the corrected 3200 GPT experiments"""
    with open('/data/llm_addiction/gpt_results_fixed_parsing/gpt_fixed_parsing_complete_20250919_151240.json', 'r') as f:
        data = json.load(f)

    print(f"✅ Loaded {len(data['results'])} experiments")
    print(f"📊 Total experiments: {data['experiment_config']['total_experiments']}")
    print(f"📈 Overall bankruptcy rate: {data['summary_statistics']['bankruptcy_rate']:.3%}")

    return data['results']

def analyze_by_conditions(experiments):
    """Comprehensive analysis by prompt combinations and bet types"""
    print("\n🔍 COMPREHENSIVE CONDITION ANALYSIS")
    print("=" * 50)

    # Group by prompt combo and bet type
    condition_analysis = defaultdict(lambda: {
        'fixed': {'experiments': [], 'bankruptcies': 0, 'stops': 0, 'total_rounds': 0, 'total_bet': 0},
        'variable': {'experiments': [], 'bankruptcies': 0, 'stops': 0, 'total_rounds': 0, 'total_bet': 0}
    })

    for exp in experiments:
        prompt_combo = exp['prompt_combo']
        bet_type = exp['bet_type']

        condition_analysis[prompt_combo][bet_type]['experiments'].append(exp)

        if exp['is_bankrupt']:
            condition_analysis[prompt_combo][bet_type]['bankruptcies'] += 1
        else:
            condition_analysis[prompt_combo][bet_type]['stops'] += 1

        condition_analysis[prompt_combo][bet_type]['total_rounds'] += exp['total_rounds']
        condition_analysis[prompt_combo][bet_type]['total_bet'] += exp['total_bet']

    # Calculate statistics for each condition
    results = []

    for prompt_combo in sorted(condition_analysis.keys()):
        for bet_type in ['fixed', 'variable']:
            data = condition_analysis[prompt_combo][bet_type]
            n_experiments = len(data['experiments'])

            if n_experiments == 0:
                continue

            bankruptcies = data['bankruptcies']
            stops = data['stops']
            bankruptcy_rate = bankruptcies / n_experiments if n_experiments > 0 else 0
            stop_rate = stops / n_experiments if n_experiments > 0 else 0
            avg_rounds = data['total_rounds'] / n_experiments if n_experiments > 0 else 0
            avg_bet = data['total_bet'] / n_experiments if n_experiments > 0 else 0

            results.append({
                'prompt_combo': prompt_combo,
                'bet_type': bet_type,
                'n': n_experiments,
                'bankruptcies': bankruptcies,
                'bankruptcy_rate': bankruptcy_rate,
                'stops': stops,
                'stop_rate': stop_rate,
                'avg_rounds': avg_rounds,
                'avg_bet': avg_bet
            })

    return results

def create_comprehensive_table(results):
    """Create comprehensive LaTeX table with all conditions"""

    print(f"\n📋 CREATING COMPREHENSIVE TABLE")
    print("=" * 40)

    # Separate by bet type for cleaner presentation
    fixed_results = [r for r in results if r['bet_type'] == 'fixed']
    variable_results = [r for r in results if r['bet_type'] == 'variable']

    latex_content = """
\\begin{table}[htbp]
\\centering
\\caption{Comprehensive Analysis of GPT-4o-mini Gambling Behavior by Condition (N=3,200)}
\\label{tab:gpt_comprehensive_real}
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{llrrrrr}
\\toprule
\\multirow{2}{*}{\\textbf{Prompt}} & \\multirow{2}{*}{\\textbf{Bet Type}} & \\multirow{2}{*}{\\textbf{N}} & \\multicolumn{2}{c}{\\textbf{Bankruptcy}} & \\textbf{Avg} & \\textbf{Avg} \\\\
\\cmidrule(lr){4-5}
& & & \\textbf{Count} & \\textbf{Rate (\%)} & \\textbf{Rounds} & \\textbf{Bet (\$)} \\\\
\\midrule
"""

    # Add fixed betting results
    for result in sorted(fixed_results, key=lambda x: x['prompt_combo']):
        prompt = result['prompt_combo'] if result['prompt_combo'] else 'BASE'
        bet_type = 'Fixed'
        n = result['n']
        bankruptcies = result['bankruptcies']
        bankruptcy_rate = result['bankruptcy_rate'] * 100
        avg_rounds = result['avg_rounds']
        avg_bet = result['avg_bet']

        latex_content += f"{prompt} & {bet_type} & {n} & {bankruptcies} & {bankruptcy_rate:.1f} & {avg_rounds:.1f} & {avg_bet:.1f} \\\\\n"

    latex_content += "\\midrule\n"

    # Add variable betting results
    for result in sorted(variable_results, key=lambda x: x['prompt_combo']):
        prompt = result['prompt_combo'] if result['prompt_combo'] else 'BASE'
        bet_type = 'Variable'
        n = result['n']
        bankruptcies = result['bankruptcies']
        bankruptcy_rate = result['bankruptcy_rate'] * 100
        avg_rounds = result['avg_rounds']
        avg_bet = result['avg_bet']

        latex_content += f"{prompt} & {bet_type} & {n} & {bankruptcies} & {bankruptcy_rate:.1f} & {avg_rounds:.1f} & {avg_bet:.1f} \\\\\n"

    latex_content += """\\bottomrule
\\end{tabular}
}
\\begin{tablenotes}
\\footnotesize
\\item Note: Analysis of 3,200 GPT-4o-mini experiments across 64 conditions (32 prompt combinations × 2 bet types) with 50 repetitions each. Fixed betting used \\$10 per round, variable betting allowed \\$5-\\$100 range. Bankruptcy occurs when balance reaches \\$0.
\\end{tablenotes}
\\end{table}
"""

    return latex_content

def validate_totals(results):
    """Validate that totals match expected values"""
    print("\n🔍 VALIDATION CHECKS")
    print("=" * 30)

    total_experiments = sum(r['n'] for r in results)
    total_bankruptcies = sum(r['bankruptcies'] for r in results)
    overall_bankruptcy_rate = total_bankruptcies / total_experiments if total_experiments > 0 else 0

    fixed_experiments = sum(r['n'] for r in results if r['bet_type'] == 'fixed')
    variable_experiments = sum(r['n'] for r in results if r['bet_type'] == 'variable')

    fixed_bankruptcies = sum(r['bankruptcies'] for r in results if r['bet_type'] == 'fixed')
    variable_bankruptcies = sum(r['bankruptcies'] for r in results if r['bet_type'] == 'variable')

    print(f"Total experiments: {total_experiments} (expected: 3,200)")
    print(f"Total bankruptcies: {total_bankruptcies}")
    print(f"Overall bankruptcy rate: {overall_bankruptcy_rate:.3%}")
    print(f"Fixed betting experiments: {fixed_experiments}")
    print(f"Variable betting experiments: {variable_experiments}")
    print(f"Fixed bankruptcies: {fixed_bankruptcies}")
    print(f"Variable bankruptcies: {variable_bankruptcies}")

    # Check if matches expected structure
    expected_conditions = 32  # prompt combinations
    unique_prompts = len(set(r['prompt_combo'] for r in results))

    print(f"\\nUnique prompt combinations found: {unique_prompts} (expected: {expected_conditions})")

    return {
        'total_experiments': total_experiments,
        'total_bankruptcies': total_bankruptcies,
        'overall_bankruptcy_rate': overall_bankruptcy_rate,
        'fixed_experiments': fixed_experiments,
        'variable_experiments': variable_experiments,
        'validation_passed': total_experiments == 3200
    }

def main():
    """Main analysis pipeline"""
    print("🎰 GPT COMPREHENSIVE ANALYSIS - 3,200 EXPERIMENTS")
    print("=" * 60)

    # Load data
    experiments = load_gpt_3200_data()

    # Analyze by conditions
    results = analyze_by_conditions(experiments)

    # Validate totals
    validation = validate_totals(results)

    # Create LaTeX table
    latex_table = create_comprehensive_table(results)

    # Save results
    output_file = '/home/ubuntu/llm_addiction/CORRECTED_3200_comprehensive_table.tex'
    with open(output_file, 'w') as f:
        f.write(latex_table)

    print(f"\\n✅ Comprehensive table saved to: {output_file}")

    # Save raw analysis data
    analysis_data = {
        'results': results,
        'validation': validation,
        'timestamp': '2025-09-14',
        'total_experiments': 3200
    }

    analysis_file = '/home/ubuntu/llm_addiction/CORRECTED_3200_comprehensive_analysis.json'
    with open(analysis_file, 'w') as f:
        json.dump(analysis_data, f, indent=2)

    print(f"📋 Analysis data saved to: {analysis_file}")
    print(f"\\n🎯 Validation: {'✅ PASSED' if validation['validation_passed'] else '❌ FAILED'}")
    print("\\n🎯 Analysis complete! Ready for paper integration.")

if __name__ == '__main__':
    main()