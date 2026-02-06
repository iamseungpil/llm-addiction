#!/usr/bin/env python3
"""
Verify GPT comprehensive analysis table data
Generate accurate table with real experimental data
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def load_gpt_data():
    """Load the GPT experimental data"""
    print("Loading GPT experimental data...")

    with open('/data/llm_addiction/gpt5_experiment/gpt5_experiment_20250921_174509.json') as f:
        data = json.load(f)

    print(f"Loaded {len(data['results'])} GPT5 experiments")
    return data['results']

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
            'N': total_experiments,
            'bankruptcy_rate': bankruptcies / total_experiments * 100,
            'immediate_stop_rate': immediate_stops / total_experiments * 100,
            'avg_rounds': avg_rounds,
            'avg_bet': avg_bet,
            'avg_net_profit': avg_net_profit
        }

    return bet_type_stats

def analyze_by_prompt_combo(experiments):
    """Analyze results by prompt combination - INCLUDING BOTH BETTING TYPES"""
    print("Analyzing by prompt combination (both betting types combined)...")

    # Include ALL experiments (both fixed and variable betting)
    all_experiments = experiments

    prompt_stats = defaultdict(lambda: {
        'experiments': [],
        'bankruptcies': 0,
        'immediate_stops': 0,
        'total_rounds': [],
        'total_bets': [],
        'net_profits': []
    })

    for exp in all_experiments:
        combo = exp['prompt_combo']
        prompt_stats[combo]['experiments'].append(exp)

        if exp['is_bankrupt']:
            prompt_stats[combo]['bankruptcies'] += 1
        if exp['total_rounds'] == 0:
            prompt_stats[combo]['immediate_stops'] += 1

        prompt_stats[combo]['total_rounds'].append(exp['total_rounds'])
        prompt_stats[combo]['total_bets'].append(exp['total_bet'])

        # Calculate net profit
        net_profit = exp['final_balance'] - 100
        prompt_stats[combo]['net_profits'].append(net_profit)

    # Calculate final statistics
    prompt_results = {}
    for combo, stats in prompt_stats.items():
        n = len(stats['experiments'])
        if n > 0:
            prompt_results[combo] = {
                'N': n,
                'bankruptcy_rate': stats['bankruptcies'] / n * 100,
                'immediate_stop_rate': stats['immediate_stops'] / n * 100,
                'avg_rounds': np.mean(stats['total_rounds']),
                'avg_bet': np.mean(stats['total_bets']),
                'avg_net_profit': np.mean(stats['net_profits'])
            }

    return prompt_results

def verify_table_claims(bet_type_stats, prompt_results):
    """Verify the specific claims in the table"""
    print("Verifying table claims...")

    # Claims from the table to verify
    table_claims = {
        'fixed_betting': {
            'N': 1600,
            'bankruptcy_rate': 0.0,
            'immediate_stop_rate': 99.9,
            'avg_rounds': 0.04,
            'avg_bet': 0.35,
            'avg_net_profit': 0.04
        },
        'variable_betting': {
            'N': 1600,
            'bankruptcy_rate': 11.4,
            'immediate_stop_rate': 58.8,
            'avg_rounds': 1.95,
            'avg_bet': 21.49,
            'avg_net_profit': 0.21
        },
        'high_risk_prompts': {
            'MPRW': {'N': 50, 'bankruptcy_rate': 46.0, 'immediate_stop_rate': 54.0, 'avg_rounds': 1.56, 'avg_bet': 103.78, 'avg_net_profit': -38.38},
            'MPW': {'N': 50, 'bankruptcy_rate': 32.0, 'immediate_stop_rate': 68.0, 'avg_rounds': 1.18, 'avg_bet': 93.38, 'avg_net_profit': -1.58},
            'PW': {'N': 50, 'bankruptcy_rate': 26.0, 'immediate_stop_rate': 74.0, 'avg_rounds': 0.88, 'avg_bet': 54.80, 'avg_net_profit': 7.30},
            'MW': {'N': 50, 'bankruptcy_rate': 24.0, 'immediate_stop_rate': 76.0, 'avg_rounds': 1.04, 'avg_bet': 50.20, 'avg_net_profit': -3.10},
            'GPW': {'N': 50, 'bankruptcy_rate': 22.0, 'immediate_stop_rate': 78.0, 'avg_rounds': 2.82, 'avg_bet': 80.30, 'avg_net_profit': -26.30}
        },
        'low_risk_prompts': {
            'BASE': {'N': 50, 'bankruptcy_rate': 0.0, 'immediate_stop_rate': 100.0, 'avg_rounds': 0.18, 'avg_bet': 2.20, 'avg_net_profit': 0.80},
            'R': {'N': 50, 'bankruptcy_rate': 0.0, 'immediate_stop_rate': 100.0, 'avg_rounds': 0.50, 'avg_bet': 5.40, 'avg_net_profit': 0.30},
            'M': {'N': 50, 'bankruptcy_rate': 0.0, 'immediate_stop_rate': 100.0, 'avg_rounds': 0.74, 'avg_bet': 15.10, 'avg_net_profit': -3.10},
            'MR': {'N': 50, 'bankruptcy_rate': 0.0, 'immediate_stop_rate': 100.0, 'avg_rounds': 0.92, 'avg_bet': 16.30, 'avg_net_profit': -1.00},
            'GR': {'N': 50, 'bankruptcy_rate': 0.0, 'immediate_stop_rate': 100.0, 'avg_rounds': 1.78, 'avg_bet': 26.00, 'avg_net_profit': 0.10}
        }
    }

    print(f"\n{'='*80}")
    print("VERIFICATION RESULTS")
    print(f"{'='*80}")

    # Check bet type statistics
    print("\nBET TYPE VERIFICATION:")
    for bet_type in ['fixed', 'variable']:
        if bet_type in bet_type_stats:
            real_data = bet_type_stats[bet_type]
            table_key = f'{bet_type}_betting'
            if table_key in table_claims:
                table_data = table_claims[table_key]

                print(f"\n{bet_type.upper()} BETTING:")
                print(f"  N: Table={table_data['N']}, Real={real_data['N']}, Match={table_data['N'] == real_data['N']}")
                print(f"  Bankruptcy Rate: Table={table_data['bankruptcy_rate']:.1f}%, Real={real_data['bankruptcy_rate']:.1f}%, Match={abs(table_data['bankruptcy_rate'] - real_data['bankruptcy_rate']) < 1}")
                print(f"  Immediate Stop: Table={table_data['immediate_stop_rate']:.1f}%, Real={real_data['immediate_stop_rate']:.1f}%, Match={abs(table_data['immediate_stop_rate'] - real_data['immediate_stop_rate']) < 5}")
                print(f"  Avg Rounds: Table={table_data['avg_rounds']:.2f}, Real={real_data['avg_rounds']:.2f}, Match={abs(table_data['avg_rounds'] - real_data['avg_rounds']) < 0.5}")
                print(f"  Avg Bet: Table=${table_data['avg_bet']:.2f}, Real=${real_data['avg_bet']:.2f}, Match={abs(table_data['avg_bet'] - real_data['avg_bet']) < 5}")
                print(f"  Net Profit: Table=${table_data['avg_net_profit']:.2f}, Real=${real_data['avg_net_profit']:.2f}, Match={abs(table_data['avg_net_profit'] - real_data['avg_net_profit']) < 5}")

    # Check prompt combinations
    print(f"\nPROMPT COMBINATION VERIFICATION:")

    # High risk prompts
    print(f"\nHIGH RISK PROMPTS:")
    for combo in ['MPRW', 'MPW', 'PW', 'MW', 'GPW']:
        if combo in prompt_results:
            real_data = prompt_results[combo]
            table_data = table_claims['high_risk_prompts'][combo]

            print(f"\n{combo}:")
            print(f"  N: Table={table_data['N']}, Real={real_data['N']}, Match={table_data['N'] == real_data['N']}")
            print(f"  Bankruptcy: Table={table_data['bankruptcy_rate']:.1f}%, Real={real_data['bankruptcy_rate']:.1f}%, Match={abs(table_data['bankruptcy_rate'] - real_data['bankruptcy_rate']) < 5}")
            print(f"  Immediate Stop: Table={table_data['immediate_stop_rate']:.1f}%, Real={real_data['immediate_stop_rate']:.1f}%, Match={abs(table_data['immediate_stop_rate'] - real_data['immediate_stop_rate']) < 10}")
            print(f"  Net Profit: Table=${table_data['avg_net_profit']:.2f}, Real=${real_data['avg_net_profit']:.2f}, Match={abs(table_data['avg_net_profit'] - real_data['avg_net_profit']) < 10}")
        else:
            print(f"\n{combo}: NOT FOUND IN REAL DATA")

    # Low risk prompts
    print(f"\nLOW RISK PROMPTS:")
    for combo in ['BASE', 'R', 'M', 'MR', 'GR']:
        if combo in prompt_results:
            real_data = prompt_results[combo]
            table_data = table_claims['low_risk_prompts'][combo]

            print(f"\n{combo}:")
            print(f"  N: Table={table_data['N']}, Real={real_data['N']}, Match={table_data['N'] == real_data['N']}")
            print(f"  Bankruptcy: Table={table_data['bankruptcy_rate']:.1f}%, Real={real_data['bankruptcy_rate']:.1f}%, Match={abs(table_data['bankruptcy_rate'] - real_data['bankruptcy_rate']) < 1}")
            print(f"  Net Profit: Table=${table_data['avg_net_profit']:.2f}, Real=${real_data['avg_net_profit']:.2f}, Match={abs(table_data['avg_net_profit'] - real_data['avg_net_profit']) < 3}")
        else:
            print(f"\n{combo}: NOT FOUND IN REAL DATA")

def generate_accurate_latex_table(bet_type_stats, prompt_results):
    """Generate accurate LaTeX tables with real data - both Korean and English versions"""
    print("Generating accurate LaTeX tables (Korean and English)...")

    # Sort prompt results by bankruptcy rate for high/low risk classification
    sorted_prompts = sorted(prompt_results.items(), key=lambda x: x[1]['bankruptcy_rate'], reverse=True)

    high_risk = sorted_prompts[:5]  # Top 5 highest bankruptcy rates
    low_risk = sorted_prompts[-5:]  # Bottom 5 lowest bankruptcy rates

    # Korean version
    korean_latex = [
        "\\begin{table}[ht!]",
        "\\centering",
        "\\caption{실제 GPT 실험 결과 종합 분석 (3,200개 실험, 64개 조건) - 두 베팅 타입 합산}",
        "\\label{tab:gpt-comprehensive-analysis-korean2}",
        "\\resizebox{\\columnwidth}{!}{",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "\\textbf{조건} & \\textbf{N} & \\textbf{파산율 (\\%)} & \\textbf{즉시중단율 (\\%)} & \\textbf{평균 라운드} & \\textbf{평균 베팅 (\\$)} & \\textbf{순손익 (\\$)} \\\\",
        "\\midrule",
        "\\multicolumn{7}{c}{\\textbf{베팅 타입별 결과}} \\\\"
    ]

    # Add betting type results - Korean
    for bet_type in ['fixed', 'variable']:
        if bet_type in bet_type_stats:
            stats = bet_type_stats[bet_type]
            korean_type = "고정 베팅" if bet_type == 'fixed' else "가변 베팅"
            korean_latex.append(
                f"{korean_type} & {stats['N']:,} & {stats['bankruptcy_rate']:.1f} & "
                f"{stats['immediate_stop_rate']:.1f} & {stats['avg_rounds']:.2f} & "
                f"{stats['avg_bet']:.2f} & {stats['avg_net_profit']:+.2f} \\\\"
            )

    korean_latex.extend([
        "\\midrule",
        "\\multicolumn{7}{c}{\\textbf{고위험 프롬프트 조합 (상위 5개)}} \\\\"
    ])

    # Add high risk prompts - Korean
    for combo, stats in high_risk:
        korean_latex.append(
            f"\\texttt{{{combo}}} & {stats['N']} & {stats['bankruptcy_rate']:.1f} & "
            f"{stats['immediate_stop_rate']:.1f} & {stats['avg_rounds']:.2f} & "
            f"{stats['avg_bet']:.2f} & {stats['avg_net_profit']:+.2f} \\\\"
        )

    korean_latex.extend([
        "\\midrule",
        "\\multicolumn{7}{c}{\\textbf{저위험 프롬프트 조합 (하위 5개)}} \\\\"
    ])

    # Add low risk prompts - Korean
    for combo, stats in low_risk:
        korean_latex.append(
            f"\\texttt{{{combo}}} & {stats['N']} & {stats['bankruptcy_rate']:.1f} & "
            f"{stats['immediate_stop_rate']:.1f} & {stats['avg_rounds']:.2f} & "
            f"{stats['avg_bet']:.2f} & {stats['avg_net_profit']:+.2f} \\\\"
        )

    korean_latex.extend([
        "\\bottomrule",
        "\\end{tabular}}",
        "\\end{table}"
    ])

    # English version
    english_latex = [
        "\\begin{table}[ht!]",
        "\\centering",
        "\\caption{Comprehensive Analysis of Actual GPT Experimental Results (3,200 experiments, 64 conditions) - Combined Betting Types}",
        "\\vspace{5pt}",
        "\\label{tab:gpt-comprehensive-analysis-english2}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "\\multirow{2}{*}{\\textbf{Condition}} & ",
        "\\multirow{2}{*}{\\textbf{N}} & ",
        "\\multirow{2}{*}{\\textbf{\\makecell{Bankruptcy \\\\ Rate (\\%)}}} & ",
        "\\multirow{2}{*}{\\textbf{\\makecell{Immediate \\\\ Quit Rate (\\%)}}} & ",
        "\\multirow{2}{*}{\\textbf{\\makecell{Avg. \\\\ Rounds}}} & ",
        "\\multirow{2}{*}{\\textbf{\\makecell{Avg. Bet \\\\ (\\$)}}} & ",
        "\\multirow{2}{*}{\\textbf{\\makecell{Net P/L \\\\ (\\$)}}} \\\\",
        "& & & & & & \\\\",
        "\\midrule",
        "\\multicolumn{7}{c}{\\textbf{Results by Betting Type}} \\\\"
    ]

    # Add betting type results - English
    for bet_type in ['fixed', 'variable']:
        if bet_type in bet_type_stats:
            stats = bet_type_stats[bet_type]
            english_type = "Fixed Betting" if bet_type == 'fixed' else "Variable Betting"
            english_latex.append(
                f"{english_type} & {stats['N']:,} & {stats['bankruptcy_rate']:.1f} & "
                f"{stats['immediate_stop_rate']:.1f} & {stats['avg_rounds']:.2f} & "
                f"{stats['avg_bet']:.2f} & {stats['avg_net_profit']:+.2f} \\\\"
            )

    english_latex.extend([
        "\\midrule",
        "\\multicolumn{7}{c}{\\textbf{High-Risk Prompt Combinations (Top 5)}} \\\\"
    ])

    # Add high risk prompts - English
    for combo, stats in high_risk:
        english_latex.append(
            f"\\texttt{{{combo}}} & {stats['N']} & {stats['bankruptcy_rate']:.1f} & "
            f"{stats['immediate_stop_rate']:.1f} & {stats['avg_rounds']:.2f} & "
            f"{stats['avg_bet']:.2f} & {stats['avg_net_profit']:+.2f} \\\\"
        )

    english_latex.extend([
        "\\midrule",
        "\\multicolumn{7}{c}{\\textbf{Low-Risk Prompt Combinations (Bottom 5)}} \\\\"
    ])

    # Add low risk prompts - English
    for combo, stats in low_risk:
        english_latex.append(
            f"\\texttt{{{combo}}} & {stats['N']} & {stats['bankruptcy_rate']:.1f} & "
            f"{stats['immediate_stop_rate']:.1f} & {stats['avg_rounds']:.2f} & "
            f"{stats['avg_bet']:.2f} & {stats['avg_net_profit']:+.2f} \\\\"
        )

    english_latex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])

    return "\n".join(korean_latex), "\n".join(english_latex)

def main():
    print("=== GPT COMPREHENSIVE TABLE VERIFICATION ===")

    # Load data
    experiments = load_gpt_data()

    # Analyze by betting type
    bet_type_stats = analyze_by_bet_type(experiments)

    # Analyze by prompt combination
    prompt_results = analyze_by_prompt_combo(experiments)

    # Verify table claims
    verify_table_claims(bet_type_stats, prompt_results)

    # Save detailed results
    output_data = {
        'timestamp': '2025-09-14',
        'total_experiments': len(experiments),
        'bet_type_stats': bet_type_stats,
        'prompt_results': prompt_results
    }

    output_file = '/home/ubuntu/llm_addiction/gpt5_experiment/results/REAL_gpt41_comprehensive_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nDetailed results saved to: {output_file}")

    # Generate accurate LaTeX tables
    korean_latex, english_latex = generate_accurate_latex_table(bet_type_stats, prompt_results)

    # Save Korean version
    korean_file = '/home/ubuntu/llm_addiction/gpt5_experiment/results/REAL_gpt41_comprehensive_table_korean.tex'
    with open(korean_file, 'w') as f:
        f.write(korean_latex)

    # Save English version
    english_file = '/home/ubuntu/llm_addiction/gpt5_experiment/results/REAL_gpt41_comprehensive_table_english.tex'
    with open(english_file, 'w') as f:
        f.write(english_latex)

    print(f"Korean LaTeX table saved to: {korean_file}")
    print(f"English LaTeX table saved to: {english_file}")

    return output_file, korean_file, english_file

if __name__ == "__main__":
    output_file, korean_file, english_file = main()
    print(f"\n✅ COMPLETED: GPT comprehensive analysis verification")
    print(f"📊 Results: {output_file}")
    print(f"📄 Korean LaTeX: {korean_file}")
    print(f"📄 English LaTeX: {english_file}")
