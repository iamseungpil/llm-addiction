#!/usr/bin/env python3
"""
Create accurate streak analysis from Gemini 3200 experiments
Generate real data for LaTeX table
"""

import json
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def load_gemini_data():
    """Load the Gemini experimental data"""
    print("Loading Gemini experimental data...")

    # Load multiple data files for Gemini
    data_files = [
        '/data/llm_addiction/gemini_experiment/gemini_experiment_20250920_042809.json',  # Main file (48.8MB)
        '/data/llm_addiction/gemini_experiment/gemini_experiment_20250920_024529.json',  # Additional (851KB)
    ]

    all_results = []
    for file_path in data_files:
        try:
            with open(file_path) as f:
                data = json.load(f)
                if 'results' in data:
                    all_results.extend(data['results'])
                else:
                    all_results.extend(data)
            print(f"  Loaded {len(data.get('results', data))} experiments from {file_path.split('/')[-1]}")
        except Exception as e:
            print(f"  Warning: Could not load {file_path}: {e}")

    print(f"Total loaded: {len(all_results)} Gemini experiments")
    return all_results

def identify_streaks(game_history, streak_type='win', streak_length=2):
    """
    Identify consecutive win/loss streaks in game history
    Returns list of positions where streak completes
    """
    if len(game_history) < streak_length:
        return []

    streaks = []

    for i in range(streak_length - 1, len(game_history)):
        # Check if last 'streak_length' games match the streak type
        if streak_type == 'win':
            is_streak = all(game_history[j]['win'] for j in range(i - streak_length + 1, i + 1))
        else:  # loss
            is_streak = all(not game_history[j]['win'] for j in range(i - streak_length + 1, i + 1))

        if is_streak:
            streaks.append(i)  # Position where streak completes

    return streaks

def analyze_post_streak_behavior(experiments):
    """
    Analyze behavior immediately after win/loss streaks
    """
    print("Analyzing post-streak behavior...")

    streak_results = {}

    # Define streak types to analyze
    streak_configs = [
        ('win', 2), ('loss', 2),
        ('win', 3), ('loss', 3),
        ('win', 4), ('loss', 4)
    ]

    for streak_type, streak_length in streak_configs:
        print(f"Analyzing {streak_length}-{streak_type} streaks...")

        streak_data = {
            'occurrences': 0,
            'continued': 0,
            'stopped': 0,
            'bet_increases': 0,
            'bet_changes': [],
            'all_next_bets': [],
            'all_current_bets': []
        }

        for exp in experiments:
            # Extract game history from round_details for Gemini data
            history = []
            if 'round_details' in exp and exp['round_details']:
                for round_detail in exp['round_details']:
                    if 'game_result' in round_detail and round_detail['game_result']:
                        game_result = round_detail['game_result']
                        history.append({
                            'win': game_result['win'],
                            'bet': game_result['bet']
                        })

            # Check if game has sufficient length for streak analysis
            if len(history) >= streak_length:

                # Find all streak occurrences
                streak_positions = identify_streaks(history, streak_type, streak_length)

                for pos in streak_positions:
                    streak_data['occurrences'] += 1
                    current_bet = history[pos]['bet']

                    # Check if there's a next round after the streak
                    if pos + 1 < len(history):
                        # Game continued - use actual next bet
                        next_bet = history[pos + 1]['bet']
                        streak_data['continued'] += 1

                        # Track betting behavior changes
                        bet_change = next_bet - current_bet
                        streak_data['bet_changes'].append(bet_change)

                        if next_bet > current_bet:
                            streak_data['bet_increases'] += 1
                    else:
                        # Game stopped - treat as 0 bet change
                        streak_data['stopped'] += 1
                        streak_data['bet_changes'].append(0)  # No bet change for stopped games
                        # No bet increase for stopped games

                    streak_data['all_current_bets'].append(current_bet)

        # Calculate statistics
        total = streak_data['occurrences']
        if total > 0:
            continuation_rate = streak_data['continued'] / total
            bet_increase_rate = streak_data['bet_increases'] / total

            # Calculate 95% confidence intervals for proportions
            cont_se = np.sqrt(continuation_rate * (1 - continuation_rate) / total)
            cont_ci = 1.96 * cont_se  # 95% CI

            bet_inc_se = np.sqrt(bet_increase_rate * (1 - bet_increase_rate) / total)
            bet_inc_ci = 1.96 * bet_inc_se  # 95% CI

            # Bet change statistics (including stopped games as 0 change)
            bet_changes = np.array(streak_data['bet_changes'])
            mean_bet_change = np.mean(bet_changes)
            bet_change_se = stats.sem(bet_changes) if len(bet_changes) > 1 else 0
            bet_change_ci = 1.96 * bet_change_se  # 95% CI

            # Calculate percentage changes (for all games, including 0 for stopped games)
            current_bets = np.array(streak_data['all_current_bets'])

            # Create percentage changes array including stopped games
            pct_changes = []
            for i, (current_bet, bet_change) in enumerate(zip(current_bets, bet_changes)):
                if current_bet > 0:
                    pct_change = (bet_change / current_bet) * 100
                    pct_changes.append(pct_change)
                else:
                    pct_changes.append(0)  # If current bet is 0, set change to 0

            if len(pct_changes) > 0:
                pct_changes = np.array(pct_changes)
                mean_pct_change = np.mean(pct_changes)
                pct_change_se = stats.sem(pct_changes) if len(pct_changes) > 1 else 0
                pct_change_ci = 1.96 * pct_change_se  # 95% CI
            else:
                mean_pct_change = 0
                pct_change_se = 0
                pct_change_ci = 0
        else:
            continuation_rate = bet_increase_rate = 0
            cont_se = bet_inc_se = cont_ci = bet_inc_ci = 0
            mean_bet_change = bet_change_se = bet_change_ci = 0
            mean_pct_change = pct_change_se = pct_change_ci = 0

        streak_key = f"{streak_length}-{streak_type}"
        streak_results[streak_key] = {
            'streak_type': streak_type,
            'streak_length': streak_length,
            'occurrences': total,
            'continuation_rate': continuation_rate,
            'continuation_se': cont_se,
            'continuation_ci': cont_ci,
            'bet_increase_rate': bet_increase_rate,
            'bet_increase_se': bet_inc_se,
            'bet_increase_ci': bet_inc_ci,
            'avg_bet_change': mean_bet_change,
            'bet_change_se': bet_change_se,
            'bet_change_ci': bet_change_ci,
            'avg_pct_change': mean_pct_change,
            'pct_change_se': pct_change_se,
            'pct_change_ci': pct_change_ci,
            'continued': streak_data['continued'],
            'stopped': streak_data['stopped']
        }

        print(f"  {streak_key}: {total} occurrences, {continuation_rate:.1%} continuation")

    return streak_results

def calculate_statistical_comparisons(streak_results):
    """Calculate statistical comparisons between win and loss streaks"""
    print("Calculating statistical comparisons...")

    comparisons = []

    for length in [2, 3, 4]:
        win_key = f"{length}-win"
        loss_key = f"{length}-loss"

        if win_key in streak_results and loss_key in streak_results:
            win_data = streak_results[win_key]
            loss_data = streak_results[loss_key]

            # Only compare if both have sufficient data
            if win_data['occurrences'] >= 5 and loss_data['occurrences'] >= 5:
                # Chi-square test for continuation rates
                win_cont = win_data['continued']
                win_stop = win_data['stopped']
                loss_cont = loss_data['continued']
                loss_stop = loss_data['stopped']

                if win_cont + win_stop > 0 and loss_cont + loss_stop > 0:
                    contingency_table = [[win_cont, win_stop], [loss_cont, loss_stop]]
                    chi2, p_value = stats.chi2_contingency(contingency_table)[:2]
                else:
                    p_value = 1.0

                comparisons.append({
                    'streak_length': length,
                    'win_occurrences': win_data['occurrences'],
                    'loss_occurrences': loss_data['occurrences'],
                    'win_continuation_rate': win_data['continuation_rate'],
                    'loss_continuation_rate': loss_data['continuation_rate'],
                    'win_bet_increase_rate': win_data['bet_increase_rate'],
                    'loss_bet_increase_rate': loss_data['bet_increase_rate'],
                    'win_avg_pct_change': win_data['avg_pct_change'],
                    'loss_avg_pct_change': loss_data['avg_pct_change'],
                    'p_value': p_value
                })

    return comparisons

def generate_korean_latex_table(streak_results):
    """Generate Korean LaTeX table with confidence intervals"""
    print("Generating Korean LaTeX table...")

    latex_lines = [
        "\\begin{table}[ht!]",
        "\\centering",
        "\\caption{Gemini-2.5-Flash 연승/연패 후 행동 패턴 분석 (N=2,750)}",
        "\\label{tab:gemini-streak-comparison-korean}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "\\multirow{2}{*}{\\textbf{연속 결과}} & ",
        "\\multirow{2}{*}{\\textbf{발생 빈도}} & ",
        "\\multirow{2}{*}{\\textbf{\\makecell{지속률 \\\\ (\\%)}}} & ",
        "\\multirow{2}{*}{\\textbf{\\makecell{베팅 증가율 \\\\ (\\%)}}} & ",
        "\\multirow{2}{*}{\\textbf{\\makecell{평균 베팅 변화 \\\\ (\\%)}}} & ",
        "\\multirow{2}{*}{\\textbf{p-value}} \\\\",
        "& & & & & \\\\",
        "\\midrule"
    ]

    # Organize data by streak length
    for length in [2, 3, 4]:
        win_key = f"{length}-win"
        loss_key = f"{length}-loss"

        if win_key in streak_results and loss_key in streak_results:
            win_data = streak_results[win_key]
            loss_data = streak_results[loss_key]

            if win_data['occurrences'] > 0 and loss_data['occurrences'] > 0:
                # Calculate p-value between win and loss
                win_cont = win_data['continued']
                win_stop = win_data['stopped']
                loss_cont = loss_data['continued']
                loss_stop = loss_data['stopped']

                if win_cont + win_stop > 0 and loss_cont + loss_stop > 0:
                    from scipy.stats import chi2_contingency
                    contingency_table = [[win_cont, win_stop], [loss_cont, loss_stop]]
                    chi2, p_value = chi2_contingency(contingency_table)[:2]
                else:
                    p_value = 1.0

                # Win streak row
                latex_lines.append(
                    f"{length}연승 & {win_data['occurrences']} & "
                    f"{win_data['continuation_rate']*100:.1f} ± {win_data['continuation_ci']*100:.1f} & "
                    f"{win_data['bet_increase_rate']*100:.1f} ± {win_data['bet_increase_ci']*100:.1f} & "
                    f"{win_data['avg_pct_change']:+.1f} ± {win_data['pct_change_ci']:.1f} & "
                    f"\\multirow{{2}}{{*}}{{{p_value:.2e}{'*' if p_value < 0.05 else ''}}} \\\\"
                )

                # Loss streak row
                latex_lines.append(
                    f"{length}연패 & {loss_data['occurrences']} & "
                    f"{loss_data['continuation_rate']*100:.1f} ± {loss_data['continuation_ci']*100:.1f} & "
                    f"{loss_data['bet_increase_rate']*100:.1f} ± {loss_data['bet_increase_ci']*100:.1f} & "
                    f"{loss_data['avg_pct_change']:+.1f} ± {loss_data['pct_change_ci']:.1f} & \\\\"
                )

                latex_lines.append("\\midrule")

    # Remove last midrule and add footer
    if latex_lines[-1] == "\\midrule":
        latex_lines.pop()

    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])

    return "\n".join(latex_lines)

def generate_english_latex_table(streak_results):
    """Generate English LaTeX table with confidence intervals"""
    print("Generating English LaTeX table...")

    latex_lines = [
        "\\begin{table}[ht!]",
        "\\centering",
        "\\caption{Comparison of Behavioral Patterns After Winning and Losing Streaks in Gemini-2.5-Flash (N=2,750)}",
        "\\vspace{5pt}",
        "\\label{tab:gemini-streak-comparison-english}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "\\multirow{2}{*}{\\textbf{Streak}} & ",
        "\\multirow{2}{*}{\\textbf{Frequency}} & ",
        "\\multirow{2}{*}{\\textbf{\\makecell{Persistence \\\\ Rate (\\%)}}} & ",
        "\\multirow{2}{*}{\\textbf{\\makecell{Bet Increase \\\\ Rate (\\%)}}} & ",
        "\\multirow{2}{*}{\\textbf{\\makecell{Mean Bet \\\\ Change (\\%)}}} & ",
        "\\multirow{2}{*}{\\textbf{p-value}} \\\\",
        "& & & & & \\\\",
        "\\midrule"
    ]

    # Organize data by streak length
    for length in [2, 3, 4]:
        win_key = f"{length}-win"
        loss_key = f"{length}-loss"

        if win_key in streak_results and loss_key in streak_results:
            win_data = streak_results[win_key]
            loss_data = streak_results[loss_key]

            if win_data['occurrences'] > 0 and loss_data['occurrences'] > 0:
                # Calculate p-value between win and loss
                win_cont = win_data['continued']
                win_stop = win_data['stopped']
                loss_cont = loss_data['continued']
                loss_stop = loss_data['stopped']

                if win_cont + win_stop > 0 and loss_cont + loss_stop > 0:
                    from scipy.stats import chi2_contingency
                    contingency_table = [[win_cont, win_stop], [loss_cont, loss_stop]]
                    chi2, p_value = chi2_contingency(contingency_table)[:2]
                else:
                    p_value = 1.0

                # Win streak row
                latex_lines.append(
                    f"{length}-Win Streak & {win_data['occurrences']} & "
                    f"{win_data['continuation_rate']*100:.1f} ± {win_data['continuation_ci']*100:.1f} & "
                    f"{win_data['bet_increase_rate']*100:.1f} ± {win_data['bet_increase_ci']*100:.1f} & "
                    f"{win_data['avg_pct_change']:+.1f} ± {win_data['pct_change_ci']:.1f} & "
                    f"\\multirow{{2}}{{*}}{{{p_value:.2e}{'*' if p_value < 0.05 else ''}}} \\\\"
                )

                # Loss streak row
                latex_lines.append(
                    f"{length}-Loss Streak & {loss_data['occurrences']} & "
                    f"{loss_data['continuation_rate']*100:.1f} ± {loss_data['continuation_ci']*100:.1f} & "
                    f"{loss_data['bet_increase_rate']*100:.1f} ± {loss_data['bet_increase_ci']*100:.1f} & "
                    f"{loss_data['avg_pct_change']:+.1f} ± {loss_data['pct_change_ci']:.1f} & \\\\"
                )

                latex_lines.append("\\midrule")

    # Remove last midrule and add footer
    if latex_lines[-1] == "\\midrule":
        latex_lines.pop()

    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])

    return "\n".join(latex_lines)

def main():
    print("=== ACCURATE STREAK ANALYSIS FOR GEMINI DATA ===")

    # Load real data
    experiments = load_gemini_data()

    # Analyze streaks
    streak_results = analyze_post_streak_behavior(experiments)

    # Calculate comparisons
    comparisons = calculate_statistical_comparisons(streak_results)

    # Save detailed results
    output_data = {
        'timestamp': '2025-09-23',
        'data_source': '/data/llm_addiction/gemini_experiment/',
        'total_experiments': len(experiments),
        'streak_results': streak_results,
        'comparisons': comparisons,
        'notes': 'Gemini streak analysis including stopped games with 0 bet change'
    }

    output_file = '/home/ubuntu/llm_addiction/gemini_experiment/results/gemini_streak_analysis_results.json'
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nDetailed results saved to: {output_file}")

    # Generate Korean LaTeX table
    korean_latex_table = generate_korean_latex_table(streak_results)
    korean_latex_file = '/home/ubuntu/llm_addiction/gemini_experiment/results/gemini_streak_analysis_korean.tex'
    with open(korean_latex_file, 'w') as f:
        f.write(korean_latex_table)
    print(f"Korean LaTeX table saved to: {korean_latex_file}")

    # Generate English LaTeX table
    english_latex_table = generate_english_latex_table(streak_results)
    english_latex_file = '/home/ubuntu/llm_addiction/gemini_experiment/results/gemini_streak_analysis_english.tex'
    with open(english_latex_file, 'w') as f:
        f.write(english_latex_table)
    print(f"English LaTeX table saved to: {english_latex_file}")

    # Print summary
    print(f"\n{'='*60}")
    print("GEMINI STREAK ANALYSIS SUMMARY (INCLUDING STOPPED GAMES)")
    print(f"{'='*60}")

    for result_key, result in streak_results.items():
        if result['occurrences'] > 0:
            print(f"{result_key}: {result['occurrences']} occurrences, "
                  f"{result['continuation_rate']:.1%} continuation (±{result['continuation_ci']:.1%}), "
                  f"{result['bet_increase_rate']:.1%} bet increases (±{result['bet_increase_ci']:.1%})")

    print(f"\nStatistical comparisons:")
    for comp in comparisons:
        print(f"{comp['streak_length']}-streak: Win vs Loss continuation "
              f"({comp['win_continuation_rate']:.1%} vs {comp['loss_continuation_rate']:.1%}), "
              f"p = {comp['p_value']:.3f}")

    return output_file, korean_latex_file, english_latex_file

if __name__ == "__main__":
    output_file, korean_latex_file, english_latex_file = main()
    print(f"\n✅ COMPLETED: Gemini streak analysis")
    print(f"📊 Results: {output_file}")
    print(f"📄 Korean LaTeX: {korean_latex_file}")
    print(f"📄 English LaTeX: {english_latex_file}")