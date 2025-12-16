#!/usr/bin/env python3
"""
Realized Risk Analysis: Model-Specific Comparison

Compares Fixed vs Variable betting risk WITHIN each model separately.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy import stats

RESULTS_DIR = Path('/data/llm_addiction/investment_choice_experiment/results')

def calculate_variance(option: int, bet_amount: float) -> float:
    """Calculate variance for a single decision"""
    if option == 1:
        return 0.0
    elif option == 2:
        ev = bet_amount * 0.9
        outcomes = [bet_amount * 1.8, 0]
        probs = [0.5, 0.5]
    elif option == 3:
        ev = bet_amount * 0.8
        outcomes = [bet_amount * 3.2, 0]
        probs = [0.25, 0.75]
    elif option == 4:
        ev = bet_amount * 0.9
        outcomes = [bet_amount * 9.0, 0]
        probs = [0.1, 0.9]
    else:
        return 0.0

    variance = sum(p * (o - ev)**2 for p, o in zip(probs, outcomes))
    return variance

def load_and_calculate():
    """Load data and calculate variance for all decisions"""

    all_decisions = []

    for result_file in sorted(RESULTS_DIR.glob('*.json')):
        with open(result_file) as f:
            data = json.load(f)

        model = data['experiment_config']['model']
        bet_type = data['experiment_config']['bet_type']

        for game in data['results']:
            for decision in game.get('decisions', []):
                choice = decision.get('choice')
                bet = decision.get('bet')

                if choice is None or bet is None:
                    continue

                variance = calculate_variance(choice, bet)

                all_decisions.append({
                    'model': model,
                    'bet_type': bet_type,
                    'option': choice,
                    'bet_amount': bet,
                    'variance': variance,
                })

    return all_decisions

def analyze_by_model(decisions):
    """Analyze Fixed vs Variable for EACH model separately"""

    print("="*80)
    print("Model-Specific Analysis: Fixed vs Variable Betting Risk")
    print("="*80)

    # Group by model
    by_model = defaultdict(lambda: {'fixed': [], 'variable': []})

    for d in decisions:
        by_model[d['model']][d['bet_type']].append(d)

    results = {}

    print(f"\n{'Model':<20} {'Bet Type':<10} {'N':<8} {'Mean Var':<15} {'Median Var':<15} {'SD Var':<15}")
    print("-"*100)

    for model in sorted(by_model.keys()):
        fixed_decs = by_model[model]['fixed']
        var_decs = by_model[model]['variable']

        # Fixed betting stats
        if fixed_decs:
            fixed_vars = [d['variance'] for d in fixed_decs]
            fixed_mean = np.mean(fixed_vars)
            fixed_median = np.median(fixed_vars)
            fixed_std = np.std(fixed_vars)

            print(f"{model:<20} {'Fixed':<10} {len(fixed_decs):<8} "
                  f"{fixed_mean:<15.2f} {fixed_median:<15.2f} {fixed_std:<15.2f}")

        # Variable betting stats
        if var_decs:
            var_vars = [d['variance'] for d in var_decs]
            var_mean = np.mean(var_vars)
            var_median = np.median(var_vars)
            var_std = np.std(var_vars)

            print(f"{model:<20} {'Variable':<10} {len(var_decs):<8} "
                  f"{var_mean:<15.2f} {var_median:<15.2f} {var_std:<15.2f}")

        # Statistical comparison
        if fixed_decs and var_decs:
            ratio_mean = var_mean / fixed_mean if fixed_mean > 0 else 0
            ratio_median = var_median / fixed_median if fixed_median > 0 else 0

            # T-test
            t_stat, p_val = stats.ttest_ind(var_vars, fixed_vars)

            # Mann-Whitney U (non-parametric)
            u_stat, u_pval = stats.mannwhitneyu(var_vars, fixed_vars, alternative='two-sided')

            print(f"{'':20} {'→ Ratio':<10} {'':<8} "
                  f"{ratio_mean:<15.2f}x {ratio_median:<15.2f}x")
            print(f"{'':20} {'→ T-test':<10} {'':<8} "
                  f"t={t_stat:<14.3f} p={p_val:<15.3e}")
            print(f"{'':20} {'→ M-W U':<10} {'':<8} "
                  f"U={u_stat:<14.0f} p={u_pval:<15.3e}")
            print()

            results[model] = {
                'fixed': {
                    'n': len(fixed_decs),
                    'mean_var': fixed_mean,
                    'median_var': fixed_median,
                    'std_var': fixed_std,
                },
                'variable': {
                    'n': len(var_decs),
                    'mean_var': var_mean,
                    'median_var': var_median,
                    'std_var': var_std,
                },
                'ratio_mean': ratio_mean,
                'ratio_median': ratio_median,
                't_test': {'t': t_stat, 'p': p_val},
                'mannwhitneyu': {'U': u_stat, 'p': u_pval},
            }

    return results

def analyze_option_distribution(decisions):
    """Analyze option distribution by model and bet type"""

    print("\n" + "="*80)
    print("Option Distribution by Model × Bet Type")
    print("="*80)

    # Group by model × bet_type
    grouped = defaultdict(lambda: defaultdict(int))

    for d in decisions:
        key = (d['model'], d['bet_type'])
        grouped[key][d['option']] += 1

    print(f"\n{'Model':<20} {'Bet Type':<10} {'Total':<8} {'Opt1%':<10} {'Opt2%':<10} {'Opt3%':<10} {'Opt4%':<10}")
    print("-"*90)

    for (model, bet_type) in sorted(grouped.keys()):
        opt_counts = grouped[(model, bet_type)]
        total = sum(opt_counts.values())

        opt1_pct = (opt_counts.get(1, 0) / total * 100) if total > 0 else 0
        opt2_pct = (opt_counts.get(2, 0) / total * 100) if total > 0 else 0
        opt3_pct = (opt_counts.get(3, 0) / total * 100) if total > 0 else 0
        opt4_pct = (opt_counts.get(4, 0) / total * 100) if total > 0 else 0

        print(f"{model:<20} {bet_type:<10} {total:<8} "
              f"{opt1_pct:<9.1f}% {opt2_pct:<9.1f}% {opt3_pct:<9.1f}% {opt4_pct:<9.1f}%")

def analyze_variance_contribution_by_option(decisions):
    """Show how much variance comes from each option"""

    print("\n" + "="*80)
    print("Variance Contribution by Option (Model × Bet Type)")
    print("="*80)

    # Group by model × bet_type
    grouped = defaultdict(lambda: defaultdict(lambda: {'count': 0, 'total_var': 0}))

    for d in decisions:
        key = (d['model'], d['bet_type'])
        opt = d['option']
        grouped[key][opt]['count'] += 1
        grouped[key][opt]['total_var'] += d['variance']

    for (model, bet_type) in sorted(grouped.keys()):
        print(f"\n{model} - {bet_type.upper()}")
        print("-"*60)

        opt_data = grouped[(model, bet_type)]
        total_var = sum(data['total_var'] for data in opt_data.values())

        print(f"{'Option':<10} {'N':<8} {'Total Var':<15} {'% of Total':<12} {'Avg Var':<15}")
        print("-"*60)

        for opt in sorted(opt_data.keys()):
            data = opt_data[opt]
            count = data['count']
            tot_var = data['total_var']
            pct = (tot_var / total_var * 100) if total_var > 0 else 0
            avg_var = tot_var / count if count > 0 else 0

            print(f"Option {opt}  {count:<8} {tot_var:<15.0f} {pct:<11.1f}% {avg_var:<15.0f}")

def analyze_bet_amount_by_option(decisions):
    """Analyze betting amounts by option for Variable betting"""

    print("\n" + "="*80)
    print("Bet Amount by Option (Variable Betting Only)")
    print("="*80)

    variable_decs = [d for d in decisions if d['bet_type'] == 'variable']

    # Group by model × option
    grouped = defaultdict(lambda: defaultdict(list))

    for d in variable_decs:
        grouped[d['model']][d['option']].append(d['bet_amount'])

    for model in sorted(grouped.keys()):
        print(f"\n{model}")
        print("-"*60)
        print(f"{'Option':<10} {'N':<8} {'Mean Bet':<12} {'Median Bet':<12} {'Max Bet':<12}")
        print("-"*60)

        for opt in sorted(grouped[model].keys()):
            bets = grouped[model][opt]
            mean_bet = np.mean(bets)
            median_bet = np.median(bets)
            max_bet = max(bets)

            print(f"Option {opt}  {len(bets):<8} ${mean_bet:<11.2f} ${median_bet:<11.2f} ${max_bet:<11.2f}")

def summary_table(results):
    """Print summary table"""

    print("\n" + "="*80)
    print("SUMMARY: Fixed → Variable Risk Increase by Model")
    print("="*80)

    print(f"\n{'Model':<20} {'Mean Ratio':<15} {'Median Ratio':<15} {'Significance':<20}")
    print("-"*80)

    for model in sorted(results.keys()):
        r = results[model]
        ratio_mean = r['ratio_mean']
        ratio_median = r['ratio_median']
        p_val = r['t_test']['p']

        if p_val < 0.001:
            sig = "***"
        elif p_val < 0.01:
            sig = "**"
        elif p_val < 0.05:
            sig = "*"
        else:
            sig = "n.s."

        print(f"{model:<20} {ratio_mean:<14.1f}x {ratio_median:<14.1f}x {sig:<20} (p={p_val:.3e})")

    print("\n*** p < 0.001, ** p < 0.01, * p < 0.05, n.s. = not significant")

def main():
    print("Loading data...")
    decisions = load_and_calculate()
    print(f"Total decisions: {len(decisions)}\n")

    # Main analysis
    results = analyze_by_model(decisions)

    # Additional analyses
    analyze_option_distribution(decisions)
    analyze_variance_contribution_by_option(decisions)
    analyze_bet_amount_by_option(decisions)

    # Summary
    summary_table(results)

    # Save
    output_file = Path('/home/ubuntu/llm_addiction/rebuttal_analysis/model_specific_risk_analysis.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*80)
    print(f"✅ Results saved to: {output_file}")
    print("="*80)

if __name__ == '__main__':
    main()
