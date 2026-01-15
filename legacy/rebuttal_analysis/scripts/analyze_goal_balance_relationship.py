#!/usr/bin/env python3
"""
Goal-Balance Relationship Analysis

Analyzes whether models set goals based on current balance,
rather than truly "escalating" from previous goals they cannot see.

Key question: Do goals correlate with current balance?
"""

import json
import re
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
from scipy import stats

RESULTS_DIR = Path('/data/llm_addiction/investment_choice_experiment/results')

def extract_goal_amount(text: str) -> Optional[int]:
    """Extract goal/target amount from model response"""
    text_lower = text.lower()

    patterns = [
        r'(?:target|goal).*?\$(\d+)',
        r'(?:my target is|target is|goal is)\s*\$?(\d+)',
        r'(?:aim for|reach|achieve)\s*\$(\d+)',
        r'(?:target amount|goal amount).*?\$(\d+)',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        if matches:
            return int(matches[0])

    return None

def analyze_goal_balance_correlation():
    """Analyze correlation between balance and goal setting"""

    all_rounds = []  # (balance, goal, model, condition)

    # Load all result files
    for result_file in sorted(RESULTS_DIR.glob('*.json')):
        with open(result_file) as f:
            data = json.load(f)

        for game in data.get('results', []):
            model = game.get('model')
            condition = game.get('prompt_condition')

            for decision in game.get('decisions', []):
                balance = decision.get('balance_before')
                response = decision.get('response', '')
                goal = extract_goal_amount(response)

                if goal is not None and balance is not None:
                    all_rounds.append({
                        'balance': balance,
                        'goal': goal,
                        'model': model,
                        'condition': condition,
                        'ratio': goal / balance if balance > 0 else None
                    })

    print("="*80)
    print(f"Total rounds with goals: {len(all_rounds)}")
    print("="*80)

    # Overall correlation
    balances = [r['balance'] for r in all_rounds]
    goals = [r['goal'] for r in all_rounds]

    correlation, p_value = stats.pearsonr(balances, goals)

    print(f"\n{'='*80}")
    print("Overall Correlation: Balance vs Goal")
    print(f"{'='*80}")
    print(f"Pearson r = {correlation:.3f}, p = {p_value:.3e}")
    print(f"Interpretation: {'Strong' if abs(correlation) > 0.7 else 'Moderate' if abs(correlation) > 0.4 else 'Weak'} correlation")

    # By condition
    print(f"\n{'='*80}")
    print("Correlation by Prompt Condition")
    print(f"{'='*80}")

    for condition in ['BASE', 'G', 'M', 'GM']:
        cond_rounds = [r for r in all_rounds if r['condition'] == condition]

        if len(cond_rounds) < 10:
            continue

        cond_balances = [r['balance'] for r in cond_rounds]
        cond_goals = [r['goal'] for r in cond_rounds]

        corr, pval = stats.pearsonr(cond_balances, cond_goals)

        avg_ratio = np.mean([r['ratio'] for r in cond_rounds if r['ratio'] is not None])

        print(f"\n{condition:4} (n={len(cond_rounds):4}): r={corr:+.3f}, p={pval:.3e}, avg ratio={avg_ratio:.2f}x")

    # By model
    print(f"\n{'='*80}")
    print("Correlation by Model")
    print(f"{'='*80}")

    for model in ['gpt4o_mini', 'gpt41_mini', 'gemini_flash', 'claude_haiku']:
        model_rounds = [r for r in all_rounds if r['model'] == model]

        if len(model_rounds) < 10:
            continue

        model_balances = [r['balance'] for r in model_rounds]
        model_goals = [r['goal'] for r in model_rounds]

        corr, pval = stats.pearsonr(model_balances, model_goals)

        avg_ratio = np.mean([r['ratio'] for r in model_rounds if r['ratio'] is not None])
        median_ratio = np.median([r['ratio'] for r in model_rounds if r['ratio'] is not None])

        print(f"\n{model:20}: r={corr:+.3f}, p={pval:.3e}")
        print(f"                      avg ratio={avg_ratio:.2f}x, median={median_ratio:.2f}x")

    # Ratio distribution analysis
    print(f"\n{'='*80}")
    print("Goal/Balance Ratio Distribution (G condition)")
    print(f"{'='*80}")

    g_rounds = [r for r in all_rounds if r['condition'] == 'G']

    for model in ['gpt4o_mini', 'gpt41_mini', 'gemini_flash', 'claude_haiku']:
        model_g = [r for r in g_rounds if r['model'] == model]

        if not model_g:
            continue

        ratios = [r['ratio'] for r in model_g if r['ratio'] is not None]

        print(f"\n{model:20}:")
        print(f"  Mean ratio: {np.mean(ratios):.2f}x")
        print(f"  Median ratio: {np.median(ratios):.2f}x")
        print(f"  Std: {np.std(ratios):.2f}")
        print(f"  Range: {np.min(ratios):.2f}x - {np.max(ratios):.2f}x")
        print(f"  25th percentile: {np.percentile(ratios, 25):.2f}x")
        print(f"  75th percentile: {np.percentile(ratios, 75):.2f}x")

    # Example: Does goal increase when balance increases?
    print(f"\n{'='*80}")
    print("Balance Increase → Goal Increase Analysis")
    print(f"{'='*80}")
    print("\nDo goals increase when balance increases within same game?")
    print("(Only for G condition, games with 3+ goals)")

    # Track within-game patterns
    result_files = sorted(RESULTS_DIR.glob('*.json'))

    same_game_patterns = []

    for result_file in result_files:
        with open(result_file) as f:
            data = json.load(f)

        for game in data.get('results', []):
            if game.get('prompt_condition') != 'G':
                continue

            decisions = game.get('decisions', [])

            game_data = []
            for decision in decisions:
                balance = decision.get('balance_before')
                goal = extract_goal_amount(decision.get('response', ''))

                if goal is not None and balance is not None:
                    game_data.append((balance, goal))

            if len(game_data) >= 3:
                # Check if balance increase → goal increase
                for i in range(len(game_data) - 1):
                    balance_change = game_data[i+1][0] - game_data[i][0]
                    goal_change = game_data[i+1][1] - game_data[i][1]

                    same_game_patterns.append({
                        'balance_change': balance_change,
                        'goal_change': goal_change,
                        'balance_increased': balance_change > 0,
                        'goal_increased': goal_change > 0,
                        'model': game.get('model')
                    })

    print(f"\nTotal consecutive round pairs: {len(same_game_patterns)}")

    # Contingency table
    balance_inc_goal_inc = sum(1 for p in same_game_patterns if p['balance_increased'] and p['goal_increased'])
    balance_inc_goal_dec = sum(1 for p in same_game_patterns if p['balance_increased'] and not p['goal_increased'])
    balance_dec_goal_inc = sum(1 for p in same_game_patterns if not p['balance_increased'] and p['goal_increased'])
    balance_dec_goal_dec = sum(1 for p in same_game_patterns if not p['balance_increased'] and not p['goal_increased'])

    print("\nContingency Table:")
    print(f"                     Goal ↑      Goal ↓")
    print(f"Balance ↑         {balance_inc_goal_inc:6}      {balance_inc_goal_dec:6}")
    print(f"Balance ↓         {balance_dec_goal_inc:6}      {balance_dec_goal_dec:6}")

    # Chi-square test
    from scipy.stats import chi2_contingency
    contingency = [
        [balance_inc_goal_inc, balance_inc_goal_dec],
        [balance_dec_goal_inc, balance_dec_goal_dec]
    ]
    chi2, pval, dof, expected = chi2_contingency(contingency)

    print(f"\nChi-square test: χ²={chi2:.2f}, p={pval:.3e}")
    print(f"Association: {'Significant' if pval < 0.001 else 'Not significant'}")

    # Save results
    output = {
        'overall_correlation': {
            'r': correlation,
            'p': p_value,
            'n': len(all_rounds)
        },
        'balance_increase_goal_increase': {
            'balance_up_goal_up': balance_inc_goal_inc,
            'balance_up_goal_down': balance_inc_goal_dec,
            'balance_down_goal_up': balance_dec_goal_inc,
            'balance_down_goal_down': balance_dec_goal_dec,
            'chi2': chi2,
            'p': pval
        }
    }

    output_file = Path('/home/ubuntu/llm_addiction/rebuttal_analysis/goal_balance_correlation.json')
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*80}")
    print(f"✅ Results saved to: {output_file}")
    print(f"{'='*80}")

if __name__ == '__main__':
    analyze_goal_balance_correlation()
