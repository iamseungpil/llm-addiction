#!/usr/bin/env python3
"""
Option-Level Risk Distribution Analysis

For each model × bet_type × option combination:
- Calculate variance distribution
- Show mean, median, std, percentiles
- Compare risk across options within same model
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

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
    """Load all decisions and calculate variance"""
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
                balance_before = decision.get('balance_before')

                if choice is None or bet is None:
                    continue

                variance = calculate_variance(choice, bet)

                all_decisions.append({
                    'model': model,
                    'bet_type': bet_type,
                    'option': choice,
                    'bet_amount': bet,
                    'balance_before': balance_before,
                    'variance': variance,
                    'std_dev': np.sqrt(variance),
                })

    return all_decisions

def analyze_option_risk_by_model(decisions):
    """Analyze risk distribution for each option by model"""

    print("="*100)
    print("OPTION RISK DISTRIBUTION BY MODEL")
    print("="*100)

    # Group by model × bet_type × option
    grouped = defaultdict(lambda: defaultdict(list))

    for d in decisions:
        key = (d['model'], d['bet_type'])
        grouped[key][d['option']].append(d)

    results = {}

    for model in sorted(set(d['model'] for d in decisions)):
        for bet_type in ['fixed', 'variable']:
            print(f"\n{'='*100}")
            print(f"MODEL: {model.upper()} - BET TYPE: {bet_type.upper()}")
            print(f"{'='*100}")

            key = (model, bet_type)
            if key not in grouped:
                continue

            option_data = grouped[key]

            # Header
            print(f"\n{'Option':<10} {'N':<8} {'Mean Var':<15} {'Median Var':<15} {'SD Var':<15} "
                  f"{'Min Var':<12} {'Max Var':<15}")
            print("-"*100)

            model_results = {}

            for opt in sorted(option_data.keys()):
                opt_decisions = option_data[opt]
                variances = [d['variance'] for d in opt_decisions]
                bet_amounts = [d['bet_amount'] for d in opt_decisions]

                # Statistics
                n = len(variances)
                mean_var = np.mean(variances)
                median_var = np.median(variances)
                std_var = np.std(variances)
                min_var = np.min(variances)
                max_var = np.max(variances)

                # Percentiles
                p25 = np.percentile(variances, 25)
                p75 = np.percentile(variances, 75)

                # Bet amount stats
                mean_bet = np.mean(bet_amounts)
                median_bet = np.median(bet_amounts)

                print(f"Option {opt}  {n:<8} {mean_var:<15.2f} {median_var:<15.2f} {std_var:<15.2f} "
                      f"{min_var:<12.2f} {max_var:<15.2f}")

                model_results[f'option_{opt}'] = {
                    'n': n,
                    'variance': {
                        'mean': mean_var,
                        'median': median_var,
                        'std': std_var,
                        'min': min_var,
                        'max': max_var,
                        'p25': p25,
                        'p75': p75,
                    },
                    'bet_amount': {
                        'mean': mean_bet,
                        'median': median_bet,
                    }
                }

            # Additional detail: percentiles and bet amounts
            print(f"\n{'Option':<10} {'25th %ile':<15} {'75th %ile':<15} {'Mean Bet':<15} {'Median Bet':<15}")
            print("-"*100)

            for opt in sorted(option_data.keys()):
                opt_key = f'option_{opt}'
                if opt_key in model_results:
                    r = model_results[opt_key]
                    print(f"Option {opt}  {r['variance']['p25']:<15.2f} {r['variance']['p75']:<15.2f} "
                          f"${r['bet_amount']['mean']:<14.2f} ${r['bet_amount']['median']:<14.2f}")

            results[f"{model}_{bet_type}"] = model_results

    return results

def compare_options_within_model(decisions):
    """Compare variance across options within same model"""

    print("\n" + "="*100)
    print("OPTION COMPARISON WITHIN EACH MODEL (Variable Betting)")
    print("="*100)

    # Focus on variable betting
    variable_decs = [d for d in decisions if d['bet_type'] == 'variable']

    # Group by model
    by_model = defaultdict(lambda: defaultdict(list))

    for d in variable_decs:
        by_model[d['model']][d['option']].append(d['variance'])

    for model in sorted(by_model.keys()):
        print(f"\n{model.upper()}")
        print("-"*100)

        opt_data = by_model[model]

        # Calculate ratios between options
        print(f"\nVariance Ratios (Reference: Option 2 = 1.0x)")
        print(f"{'Option':<10} {'Mean Variance':<20} {'Ratio to Opt2':<20}")
        print("-"*60)

        opt2_mean = np.mean(opt_data[2]) if 2 in opt_data else 1

        for opt in sorted(opt_data.keys()):
            mean_var = np.mean(opt_data[opt])
            ratio = mean_var / opt2_mean if opt2_mean > 0 else 0

            print(f"Option {opt}  {mean_var:<20.2f} {ratio:<20.2f}x")

def variance_by_bet_amount_quantiles(decisions):
    """Analyze how variance changes with bet amount"""

    print("\n" + "="*100)
    print("VARIANCE BY BET AMOUNT QUANTILES (Variable Betting, Option 2 & 4)")
    print("="*100)

    variable_decs = [d for d in decisions if d['bet_type'] == 'variable']

    # Group by model × option
    for model in sorted(set(d['model'] for d in decisions)):
        print(f"\n{model.upper()}")
        print("-"*100)

        model_decs = [d for d in variable_decs if d['model'] == model]

        for opt in [2, 4]:
            opt_decs = [d for d in model_decs if d['option'] == opt]

            if not opt_decs:
                continue

            print(f"\n  Option {opt} (N={len(opt_decs)})")
            print(f"  {'Bet Quantile':<20} {'N':<8} {'Mean Bet':<15} {'Mean Variance':<20}")
            print(f"  {'-'*70}")

            # Sort by bet amount
            sorted_decs = sorted(opt_decs, key=lambda x: x['bet_amount'])

            # Quartiles
            n = len(sorted_decs)
            quantiles = {
                'Q1 (0-25%)': sorted_decs[:n//4],
                'Q2 (25-50%)': sorted_decs[n//4:n//2],
                'Q3 (50-75%)': sorted_decs[n//2:3*n//4],
                'Q4 (75-100%)': sorted_decs[3*n//4:],
            }

            for q_name, q_decs in quantiles.items():
                if not q_decs:
                    continue

                mean_bet = np.mean([d['bet_amount'] for d in q_decs])
                mean_var = np.mean([d['variance'] for d in q_decs])

                print(f"  {q_name:<20} {len(q_decs):<8} ${mean_bet:<14.2f} {mean_var:<20.2f}")

def summary_statistics_table(results):
    """Create summary table across all models"""

    print("\n" + "="*100)
    print("SUMMARY TABLE: Mean Variance by Model × Bet Type × Option")
    print("="*100)

    print(f"\n{'Model':<20} {'Bet Type':<12} {'Opt1':<12} {'Opt2':<12} {'Opt3':<12} {'Opt4':<12}")
    print("-"*90)

    for key in sorted(results.keys()):
        model, bet_type = key.rsplit('_', 1)

        opt1 = results[key].get('option_1', {}).get('variance', {}).get('mean', 0)
        opt2 = results[key].get('option_2', {}).get('variance', {}).get('mean', 0)
        opt3 = results[key].get('option_3', {}).get('variance', {}).get('mean', 0)
        opt4 = results[key].get('option_4', {}).get('variance', {}).get('mean', 0)

        print(f"{model:<20} {bet_type:<12} {opt1:<12.0f} {opt2:<12.0f} {opt3:<12.0f} {opt4:<12.0f}")

def main():
    print("Loading data...")
    decisions = load_and_calculate()
    print(f"Total decisions: {len(decisions)}\n")

    # Main analysis: detailed option stats
    results = analyze_option_risk_by_model(decisions)

    # Comparisons
    compare_options_within_model(decisions)

    # Bet amount effect
    variance_by_bet_amount_quantiles(decisions)

    # Summary table
    summary_statistics_table(results)

    # Save
    output_file = Path('/home/ubuntu/llm_addiction/rebuttal_analysis/option_risk_distribution.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*100)
    print(f"✅ Results saved to: {output_file}")
    print("="*100)

if __name__ == '__main__':
    main()
