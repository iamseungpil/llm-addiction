#!/usr/bin/env python3
"""
Realized Risk Analysis: Option × Bet Amount → Variance

Calculates actual variance for each decision based on:
- Option choice (1/2/3/4)
- Bet amount (fixed $10 or variable $1-$100)

Key questions:
1. Does high Option 4 rate → high actual risk?
2. How does bet amount modulate risk?
3. Which model has highest realized variance?
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = Path('/data/llm_addiction/investment_choice_experiment/results')
OUTPUT_DIR = Path('/home/ubuntu/llm_addiction/rebuttal_analysis/figures')
OUTPUT_DIR.mkdir(exist_ok=True)

def calculate_variance(option: int, bet_amount: float) -> float:
    """
    Calculate variance for a single decision

    Args:
        option: 1, 2, 3, or 4
        bet_amount: Amount invested

    Returns:
        Variance of outcome
    """
    if option == 1:
        # Option 1: Certain outcome (get bet back)
        return 0.0

    elif option == 2:
        # Option 2: 50% of 1.8x, 50% of 0
        ev = bet_amount * 0.9  # 0.5 * 1.8 = 0.9
        outcomes = [bet_amount * 1.8, 0]
        probs = [0.5, 0.5]

    elif option == 3:
        # Option 3: 25% of 3.2x, 75% of 0
        ev = bet_amount * 0.8  # 0.25 * 3.2 = 0.8
        outcomes = [bet_amount * 3.2, 0]
        probs = [0.25, 0.75]

    elif option == 4:
        # Option 4: 10% of 9.0x, 90% of 0
        ev = bet_amount * 0.9  # 0.1 * 9.0 = 0.9
        outcomes = [bet_amount * 9.0, 0]
        probs = [0.1, 0.9]

    else:
        return 0.0

    # Variance = E[(X - μ)²]
    variance = sum(p * (o - ev)**2 for p, o in zip(probs, outcomes))

    return variance

def calculate_expected_value(option: int, bet_amount: float) -> float:
    """Calculate expected net profit/loss"""
    if option == 1:
        return 0.0  # Break even
    elif option == 2:
        return bet_amount * 0.9 - bet_amount  # -0.1 * bet
    elif option == 3:
        return bet_amount * 0.8 - bet_amount  # -0.2 * bet
    elif option == 4:
        return bet_amount * 0.9 - bet_amount  # -0.1 * bet
    else:
        return 0.0

def analyze_all_decisions():
    """Analyze all decisions from all games"""

    all_decisions = []
    game_level_stats = []

    print("="*80)
    print("Loading Data and Calculating Realized Variance")
    print("="*80)

    for result_file in sorted(RESULTS_DIR.glob('*.json')):
        print(f"Processing: {result_file.name}")

        with open(result_file) as f:
            data = json.load(f)

        model = data['experiment_config']['model']
        bet_type = data['experiment_config']['bet_type']

        for game in data['results']:
            game_variance = 0
            game_total_bet = 0
            game_decisions = []

            for decision in game.get('decisions', []):
                choice = decision.get('choice')
                bet = decision.get('bet')
                balance_before = decision.get('balance_before')

                if choice is None or bet is None:
                    continue

                # Calculate variance for this decision
                variance = calculate_variance(choice, bet)
                ev = calculate_expected_value(choice, bet)

                # Bet ratio
                bet_ratio = bet / balance_before if balance_before > 0 else 0

                decision_data = {
                    'model': model,
                    'bet_type': bet_type,
                    'option': choice,
                    'bet_amount': bet,
                    'balance_before': balance_before,
                    'bet_ratio': bet_ratio,
                    'variance': variance,
                    'std_dev': np.sqrt(variance),
                    'expected_value': ev,
                    'prompt_condition': game.get('prompt_condition'),
                }

                all_decisions.append(decision_data)
                game_decisions.append(decision_data)
                game_variance += variance
                game_total_bet += bet

            # Game-level aggregates
            if game_decisions:
                game_level_stats.append({
                    'model': model,
                    'bet_type': bet_type,
                    'prompt_condition': game.get('prompt_condition'),
                    'total_variance': game_variance,
                    'avg_variance': game_variance / len(game_decisions),
                    'total_bet': game_total_bet,
                    'rounds': len(game_decisions),
                    'final_balance': game.get('final_balance', 100),
                    'net_pl': game.get('final_balance', 100) - 100,
                })

    print(f"\nTotal decisions analyzed: {len(all_decisions)}")
    print(f"Total games analyzed: {len(game_level_stats)}")

    return all_decisions, game_level_stats

def summarize_by_model_bettype(decisions, game_stats):
    """Summarize statistics by model and bet type"""

    print("\n" + "="*80)
    print("Summary Statistics: Model × Bet Type")
    print("="*80)

    # Group decisions
    grouped = defaultdict(list)
    for d in decisions:
        key = (d['model'], d['bet_type'])
        grouped[key].append(d)

    # Group games
    game_grouped = defaultdict(list)
    for g in game_stats:
        key = (g['model'], g['bet_type'])
        game_grouped[key].append(g)

    results = {}

    print(f"\n{'Model':<20} {'Bet Type':<10} {'N Dec':<8} {'Avg Var':<12} {'Avg Bet':<10} "
          f"{'Opt4%':<8} {'Net P/L':<10}")
    print("-"*100)

    for (model, bet_type) in sorted(grouped.keys()):
        decs = grouped[(model, bet_type)]
        games = game_grouped[(model, bet_type)]

        avg_variance = np.mean([d['variance'] for d in decs])
        avg_bet = np.mean([d['bet_amount'] for d in decs])
        opt4_rate = sum(1 for d in decs if d['option'] == 4) / len(decs) * 100
        avg_net_pl = np.mean([g['net_pl'] for g in games])

        results[(model, bet_type)] = {
            'n_decisions': len(decs),
            'avg_variance': avg_variance,
            'avg_bet': avg_bet,
            'opt4_rate': opt4_rate,
            'avg_net_pl': avg_net_pl,
        }

        print(f"{model:<20} {bet_type:<10} {len(decs):<8} "
              f"{avg_variance:<12.2f} ${avg_bet:<9.2f} "
              f"{opt4_rate:<7.1f}% ${avg_net_pl:<9.2f}")

    return results

def analyze_option_bet_combinations(decisions):
    """Analyze Option × Bet Amount → Variance patterns"""

    print("\n" + "="*80)
    print("Option × Bet Amount Analysis (Variable Betting Only)")
    print("="*80)

    variable_decs = [d for d in decisions if d['bet_type'] == 'variable']

    # Group by option
    by_option = defaultdict(list)
    for d in variable_decs:
        by_option[d['option']].append(d)

    print(f"\n{'Option':<10} {'N':<8} {'Avg Bet':<12} {'Avg Var':<12} {'Bet Range':<20}")
    print("-"*70)

    for opt in sorted(by_option.keys()):
        opt_decs = by_option[opt]
        avg_bet = np.mean([d['bet_amount'] for d in opt_decs])
        avg_var = np.mean([d['variance'] for d in opt_decs])
        min_bet = min(d['bet_amount'] for d in opt_decs)
        max_bet = max(d['bet_amount'] for d in opt_decs)

        print(f"Option {opt}  {len(opt_decs):<8} ${avg_bet:<11.2f} "
              f"{avg_var:<12.2f} ${min_bet:.0f} - ${max_bet:.0f}")

    # Model × Option analysis
    print("\n" + "="*80)
    print("Model-Specific: Option 4 Behavior (Variable Betting)")
    print("="*80)

    opt4_variable = [d for d in variable_decs if d['option'] == 4]

    by_model = defaultdict(list)
    for d in opt4_variable:
        by_model[d['model']].append(d)

    print(f"\n{'Model':<20} {'N Opt4':<10} {'Avg Bet':<12} {'Avg Var':<12} {'Avg Bet Ratio':<15}")
    print("-"*80)

    for model in sorted(by_model.keys()):
        model_decs = by_model[model]
        avg_bet = np.mean([d['bet_amount'] for d in model_decs])
        avg_var = np.mean([d['variance'] for d in model_decs])
        avg_ratio = np.mean([d['bet_ratio'] for d in model_decs])

        print(f"{model:<20} {len(model_decs):<10} ${avg_bet:<11.2f} "
              f"{avg_var:<12.2f} {avg_ratio*100:<14.1f}%")

def check_hypothesis(decisions, game_stats):
    """Test key hypotheses"""

    print("\n" + "="*80)
    print("HYPOTHESIS TESTING")
    print("="*80)

    # H1: Option 4 rate correlates with actual variance?
    print("\n[H1] Does Option 4 selection rate correlate with realized variance?")
    print("-"*80)

    # Group by model × bet_type
    grouped = defaultdict(lambda: {'opt4_rate': 0, 'avg_variance': 0, 'n': 0})

    for d in decisions:
        key = (d['model'], d['bet_type'])
        grouped[key]['avg_variance'] += d['variance']
        if d['option'] == 4:
            grouped[key]['opt4_rate'] += 1
        grouped[key]['n'] += 1

    # Calculate rates
    for key in grouped:
        grouped[key]['opt4_rate'] = (grouped[key]['opt4_rate'] / grouped[key]['n']) * 100
        grouped[key]['avg_variance'] /= grouped[key]['n']

    print(f"\n{'Model':<20} {'Bet Type':<10} {'Opt4 Rate':<12} {'Avg Variance':<15}")
    print("-"*70)

    for (model, bet_type) in sorted(grouped.keys()):
        stats = grouped[(model, bet_type)]
        print(f"{model:<20} {bet_type:<10} {stats['opt4_rate']:<11.1f}% "
              f"{stats['avg_variance']:<15.2f}")

    # Correlation
    opt4_rates = [grouped[k]['opt4_rate'] for k in sorted(grouped.keys())]
    variances = [grouped[k]['avg_variance'] for k in sorted(grouped.keys())]

    correlation = np.corrcoef(opt4_rates, variances)[0, 1]
    print(f"\nPearson correlation: r = {correlation:.3f}")

    if correlation > 0.5:
        print("→ Strong positive correlation: High Opt4 rate → High variance ✓")
    elif correlation > 0:
        print("→ Weak positive correlation: Opt4 rate weakly predicts variance")
    else:
        print("→ Negative/No correlation: Opt4 rate does NOT predict variance!")

    # H2: Variable betting riskier than fixed?
    print("\n\n[H2] Is Variable betting riskier than Fixed betting?")
    print("-"*80)

    fixed_var = [d['variance'] for d in decisions if d['bet_type'] == 'fixed']
    variable_var = [d['variance'] for d in decisions if d['bet_type'] == 'variable']

    print(f"Fixed betting:    Mean variance = {np.mean(fixed_var):.2f}")
    print(f"Variable betting: Mean variance = {np.mean(variable_var):.2f}")
    print(f"Ratio: {np.mean(variable_var) / np.mean(fixed_var):.2f}x")

    from scipy import stats
    t_stat, p_val = stats.ttest_ind(variable_var, fixed_var)
    print(f"\nT-test: t = {t_stat:.3f}, p = {p_val:.3e}")

    if p_val < 0.001 and np.mean(variable_var) > np.mean(fixed_var):
        print("→ Variable betting is significantly riskier ✓")
    else:
        print("→ No significant difference in risk")

    # H3: Claude paradox - low Opt4 but high loss?
    print("\n\n[H3] Claude Paradox: Low Opt4 rate but high loss in Variable?")
    print("-"*80)

    claude_var = [d for d in decisions if d['model'] == 'claude_haiku' and d['bet_type'] == 'variable']

    opt4_decs = [d for d in claude_var if d['option'] == 4]
    opt2_decs = [d for d in claude_var if d['option'] == 2]
    opt3_decs = [d for d in claude_var if d['option'] == 3]

    print(f"Option 2 choices: {len(opt2_decs)} (avg bet: ${np.mean([d['bet_amount'] for d in opt2_decs]):.2f})")
    print(f"Option 3 choices: {len(opt3_decs)} (avg bet: ${np.mean([d['bet_amount'] for d in opt3_decs]):.2f})")
    print(f"Option 4 choices: {len(opt4_decs)} (avg bet: ${np.mean([d['bet_amount'] for d in opt4_decs]) if opt4_decs else 0:.2f})")

    opt2_total_var = sum(d['variance'] for d in opt2_decs)
    opt3_total_var = sum(d['variance'] for d in opt3_decs)
    opt4_total_var = sum(d['variance'] for d in opt4_decs) if opt4_decs else 0

    print(f"\nTotal variance contribution:")
    print(f"  Option 2: {opt2_total_var:.0f} ({opt2_total_var/(opt2_total_var+opt3_total_var+opt4_total_var)*100:.1f}%)")
    print(f"  Option 3: {opt3_total_var:.0f} ({opt3_total_var/(opt2_total_var+opt3_total_var+opt4_total_var)*100:.1f}%)")
    print(f"  Option 4: {opt4_total_var:.0f} ({opt4_total_var/(opt2_total_var+opt3_total_var+opt4_total_var)*100:.1f}%)")

    print("\n→ Claude's high loss comes from:", "Option 2/3 with large bets" if opt2_total_var > opt4_total_var else "Option 4")

def save_results(decisions, game_stats, summary):
    """Save analysis results"""

    output_file = Path('/home/ubuntu/llm_addiction/rebuttal_analysis/realized_risk_analysis.json')

    # Prepare summary for JSON
    summary_json = {}
    for (model, bet_type), stats in summary.items():
        key = f"{model}_{bet_type}"
        summary_json[key] = stats

    # Game-level summary
    game_summary = {}
    grouped = defaultdict(list)
    for g in game_stats:
        key = f"{g['model']}_{g['bet_type']}"
        grouped[key].append(g)

    for key, games in grouped.items():
        game_summary[key] = {
            'n_games': len(games),
            'avg_total_variance': np.mean([g['total_variance'] for g in games]),
            'avg_net_pl': np.mean([g['net_pl'] for g in games]),
        }

    output = {
        'decision_level_summary': summary_json,
        'game_level_summary': game_summary,
        'total_decisions': len(decisions),
        'total_games': len(game_stats),
    }

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print("\n" + "="*80)
    print(f"✅ Results saved to: {output_file}")
    print("="*80)

def main():
    # Load and calculate
    decisions, game_stats = analyze_all_decisions()

    # Summarize
    summary = summarize_by_model_bettype(decisions, game_stats)

    # Detailed analysis
    analyze_option_bet_combinations(decisions)

    # Hypothesis testing
    check_hypothesis(decisions, game_stats)

    # Save
    save_results(decisions, game_stats, summary)

if __name__ == '__main__':
    main()
