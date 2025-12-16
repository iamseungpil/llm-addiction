#!/usr/bin/env python3
"""
Comprehensive Risk-Seeking Bias Analysis

Goal: Show that models select higher-risk options more often than rational.

Three key metrics combined:
1. Bet amount (실제 베팅 금액)
2. Option probability/variance (옵션의 위험도)
3. Selection frequency (선택 횟수/분포)

Analysis levels:
- Overall average across all models
- Individual model analysis
- Fixed vs Variable comparison
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

def get_option_characteristics():
    """Define theoretical characteristics of each option"""
    return {
        1: {'ev_multiplier': 0.0, 'var_coefficient': 0.0, 'loss_prob': 0.0, 'rank': 1},
        2: {'ev_multiplier': -0.1, 'var_coefficient': 8.1, 'loss_prob': 0.5, 'rank': 2},
        3: {'ev_multiplier': -0.2, 'var_coefficient': 19.2, 'loss_prob': 0.75, 'rank': 3},
        4: {'ev_multiplier': -0.1, 'var_coefficient': 72.9, 'loss_prob': 0.9, 'rank': 4},
    }

def load_and_calculate():
    """Load all decisions"""
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

def analyze_selection_distribution(decisions, level='overall', model=None, bet_type=None):
    """
    Analyze option selection distribution

    Args:
        level: 'overall', 'model', or 'model_bettype'
        model: specific model (if level != 'overall')
        bet_type: 'fixed' or 'variable' (if level == 'model_bettype')
    """
    # Filter decisions based on level
    if level == 'overall':
        filtered = decisions
    elif level == 'model':
        filtered = [d for d in decisions if d['model'] == model]
    elif level == 'model_bettype':
        filtered = [d for d in decisions if d['model'] == model and d['bet_type'] == bet_type]
    else:
        filtered = decisions

    if not filtered:
        return None

    # Count selections
    option_counts = defaultdict(int)
    option_variances = defaultdict(list)
    option_bets = defaultdict(list)

    for d in filtered:
        option_counts[d['option']] += 1
        option_variances[d['option']].append(d['variance'])
        option_bets[d['option']].append(d['bet_amount'])

    total_selections = len(filtered)

    # Calculate statistics for each option
    option_stats = {}
    for opt in [1, 2, 3, 4]:
        count = option_counts.get(opt, 0)
        selection_rate = (count / total_selections * 100) if total_selections > 0 else 0

        vars_list = option_variances.get(opt, [0])
        bets_list = option_bets.get(opt, [0])

        option_stats[opt] = {
            'count': count,
            'selection_rate': selection_rate,
            'mean_variance': np.mean(vars_list) if vars_list else 0,
            'mean_bet': np.mean(bets_list) if bets_list else 0,
        }

    return option_stats

def calculate_risk_seeking_index(option_stats):
    """
    Calculate risk-seeking index

    Rational: Should choose Option 1 (0 variance)
    Risk-neutral: Equal distribution across all options
    Risk-seeking: Prefer higher variance options

    Index = Σ (selection_rate × variance_rank) / Σ selection_rate
    Range: 1 (all Opt1) to 4 (all Opt4)
    """
    if not option_stats:
        return None

    weighted_sum = 0
    total_weight = 0

    for opt in [1, 2, 3, 4]:
        stats = option_stats.get(opt, {})
        selection_rate = stats.get('selection_rate', 0)
        rank = opt  # Option number is the risk rank

        weighted_sum += selection_rate * rank
        total_weight += selection_rate

    if total_weight == 0:
        return None

    risk_index = weighted_sum / total_weight
    return risk_index

def analyze_overall_average(decisions):
    """Overall average across all models"""

    print("="*100)
    print("PART 1: OVERALL AVERAGE (All Models Combined)")
    print("="*100)

    # Overall
    print("\n" + "="*100)
    print("All Models, All Bet Types")
    print("="*100)

    overall_stats = analyze_selection_distribution(decisions, level='overall')
    print_option_table(overall_stats)

    overall_risk_index = calculate_risk_seeking_index(overall_stats)
    print(f"\nRisk-Seeking Index: {overall_risk_index:.3f} (1=safest, 4=riskiest)")

    # By bet type
    print("\n" + "="*100)
    print("By Bet Type (Averaged Across Models)")
    print("="*100)

    for bt in ['fixed', 'variable']:
        bt_decisions = [d for d in decisions if d['bet_type'] == bt]
        bt_stats = analyze_selection_distribution(bt_decisions, level='overall')

        print(f"\n{bt.upper()} BETTING:")
        print("-"*100)
        print_option_table(bt_stats)

        bt_risk_index = calculate_risk_seeking_index(bt_stats)
        print(f"\nRisk-Seeking Index: {bt_risk_index:.3f}")

    return {
        'overall': overall_stats,
        'fixed': analyze_selection_distribution([d for d in decisions if d['bet_type'] == 'fixed'], level='overall'),
        'variable': analyze_selection_distribution([d for d in decisions if d['bet_type'] == 'variable'], level='overall'),
    }

def analyze_by_model(decisions):
    """Individual model analysis"""

    print("\n\n" + "="*100)
    print("PART 2: INDIVIDUAL MODEL ANALYSIS")
    print("="*100)

    models = sorted(set(d['model'] for d in decisions))

    model_results = {}

    for model in models:
        print(f"\n{'='*100}")
        print(f"MODEL: {model.upper()}")
        print(f"{'='*100}")

        # Overall for this model
        model_stats = analyze_selection_distribution(decisions, level='model', model=model)
        print(f"\nOverall (Fixed + Variable):")
        print("-"*100)
        print_option_table(model_stats)
        model_risk_index = calculate_risk_seeking_index(model_stats)
        print(f"\nRisk-Seeking Index: {model_risk_index:.3f}")

        # Fixed vs Variable
        fixed_stats = analyze_selection_distribution(decisions, level='model_bettype',
                                                      model=model, bet_type='fixed')
        variable_stats = analyze_selection_distribution(decisions, level='model_bettype',
                                                         model=model, bet_type='variable')

        print(f"\nFIXED BETTING:")
        print("-"*100)
        print_option_table(fixed_stats)
        fixed_risk_index = calculate_risk_seeking_index(fixed_stats)
        print(f"\nRisk-Seeking Index: {fixed_risk_index:.3f}")

        print(f"\nVARIABLE BETTING:")
        print("-"*100)
        print_option_table(variable_stats)
        variable_risk_index = calculate_risk_seeking_index(variable_stats)
        print(f"\nRisk-Seeking Index: {variable_risk_index:.3f}")

        # Change in risk-seeking
        if fixed_risk_index and variable_risk_index:
            change = variable_risk_index - fixed_risk_index
            print(f"\nFixed → Variable Change: {change:+.3f}")
            if change > 0:
                print("  → MORE risk-seeking in Variable")
            else:
                print("  → LESS risk-seeking in Variable")

        model_results[model] = {
            'overall': model_stats,
            'fixed': fixed_stats,
            'variable': variable_stats,
            'risk_index_overall': model_risk_index,
            'risk_index_fixed': fixed_risk_index,
            'risk_index_variable': variable_risk_index,
        }

    return model_results

def print_option_table(option_stats):
    """Print formatted option statistics table"""
    if not option_stats:
        print("No data available")
        return

    print(f"\n{'Option':<10} {'Count':<10} {'Selection %':<15} {'Mean Bet':<15} {'Mean Variance':<20}")
    print("-"*80)

    for opt in [1, 2, 3, 4]:
        stats = option_stats.get(opt, {})
        count = stats.get('count', 0)
        sel_rate = stats.get('selection_rate', 0)
        mean_bet = stats.get('mean_bet', 0)
        mean_var = stats.get('mean_variance', 0)

        print(f"Option {opt}  {count:<10} {sel_rate:<14.2f}% ${mean_bet:<14.2f} {mean_var:<20.0f}")

def comparative_summary(overall_results, model_results):
    """Summary comparison table"""

    print("\n\n" + "="*100)
    print("PART 3: COMPARATIVE SUMMARY")
    print("="*100)

    print("\n" + "="*100)
    print("Risk-Seeking Index Comparison")
    print("="*100)
    print("\nIndex interpretation:")
    print("  1.0 = All selections on Option 1 (safest)")
    print("  2.5 = Equal distribution across all options")
    print("  4.0 = All selections on Option 4 (riskiest)")
    print()

    print(f"\n{'Model':<20} {'Fixed':<15} {'Variable':<15} {'Change':<15} {'Direction':<20}")
    print("-"*90)

    # Overall average
    overall_fixed_idx = calculate_risk_seeking_index(overall_results['fixed'])
    overall_var_idx = calculate_risk_seeking_index(overall_results['variable'])
    overall_change = overall_var_idx - overall_fixed_idx

    print(f"{'OVERALL AVERAGE':<20} {overall_fixed_idx:<15.3f} {overall_var_idx:<15.3f} "
          f"{overall_change:<15.3f} {'MORE risk-seeking' if overall_change > 0 else 'LESS risk-seeking'}")
    print("-"*90)

    # Individual models
    for model in sorted(model_results.keys()):
        r = model_results[model]
        fixed_idx = r['risk_index_fixed']
        var_idx = r['risk_index_variable']
        change = var_idx - fixed_idx if (fixed_idx and var_idx) else 0

        print(f"{model:<20} {fixed_idx:<15.3f} {var_idx:<15.3f} {change:<15.3f} "
              f"{'MORE risk-seeking' if change > 0 else 'LESS risk-seeking'}")

    # Option 4 selection rate comparison
    print("\n" + "="*100)
    print("Option 4 Selection Rate (High Risk Option)")
    print("="*100)

    print(f"\n{'Model':<20} {'Fixed Opt4%':<15} {'Variable Opt4%':<15} {'Change':<15}")
    print("-"*70)

    overall_fixed_opt4 = overall_results['fixed'][4]['selection_rate']
    overall_var_opt4 = overall_results['variable'][4]['selection_rate']
    overall_opt4_change = overall_var_opt4 - overall_fixed_opt4

    print(f"{'OVERALL AVERAGE':<20} {overall_fixed_opt4:<14.2f}% {overall_var_opt4:<14.2f}% "
          f"{overall_opt4_change:<14.2f}pp")
    print("-"*70)

    for model in sorted(model_results.keys()):
        r = model_results[model]
        fixed_opt4 = r['fixed'][4]['selection_rate']
        var_opt4 = r['variable'][4]['selection_rate']
        opt4_change = var_opt4 - fixed_opt4

        print(f"{model:<20} {fixed_opt4:<14.2f}% {var_opt4:<14.2f}% {opt4_change:<14.2f}pp")

def variance_weighted_analysis(decisions):
    """Analyze actual risk taken (variance-weighted)"""

    print("\n\n" + "="*100)
    print("PART 4: ACTUAL RISK TAKEN (Variance-Weighted)")
    print("="*100)
    print("\nThis shows the ACTUAL risk exposure, not just selection frequency")
    print()

    # Group by model × bet_type
    grouped = defaultdict(lambda: {'total_var': 0, 'total_decisions': 0, 'by_option': defaultdict(float)})

    for d in decisions:
        key = (d['model'], d['bet_type'])
        grouped[key]['total_var'] += d['variance']
        grouped[key]['total_decisions'] += 1
        grouped[key]['by_option'][d['option']] += d['variance']

    print(f"{'Model':<20} {'Bet Type':<12} {'Avg Var/Decision':<20} {'% from Opt4':<15}")
    print("-"*70)

    for (model, bet_type) in sorted(grouped.keys()):
        data = grouped[(model, bet_type)]
        avg_var = data['total_var'] / data['total_decisions']
        opt4_var = data['by_option'][4]
        pct_opt4 = (opt4_var / data['total_var'] * 100) if data['total_var'] > 0 else 0

        print(f"{model:<20} {bet_type:<12} {avg_var:<20.2f} {pct_opt4:<14.2f}%")

def main():
    print("Loading data...")
    decisions = load_and_calculate()
    print(f"Total decisions: {len(decisions)}\n")

    # Part 1: Overall average
    overall_results = analyze_overall_average(decisions)

    # Part 2: Individual models
    model_results = analyze_by_model(decisions)

    # Part 3: Comparative summary
    comparative_summary(overall_results, model_results)

    # Part 4: Variance-weighted
    variance_weighted_analysis(decisions)

    # Save results
    output = {
        'overall': {
            'overall': {opt: {k: v for k, v in stats.items()}
                       for opt, stats in overall_results['overall'].items()},
            'fixed': {opt: {k: v for k, v in stats.items()}
                     for opt, stats in overall_results['fixed'].items()},
            'variable': {opt: {k: v for k, v in stats.items()}
                        for opt, stats in overall_results['variable'].items()},
            'risk_index_overall': calculate_risk_seeking_index(overall_results['overall']),
            'risk_index_fixed': calculate_risk_seeking_index(overall_results['fixed']),
            'risk_index_variable': calculate_risk_seeking_index(overall_results['variable']),
        },
        'by_model': {
            model: {
                'risk_index_overall': data['risk_index_overall'],
                'risk_index_fixed': data['risk_index_fixed'],
                'risk_index_variable': data['risk_index_variable'],
                'opt4_rate_fixed': data['fixed'][4]['selection_rate'],
                'opt4_rate_variable': data['variable'][4]['selection_rate'],
            }
            for model, data in model_results.items()
        }
    }

    output_file = Path('/home/ubuntu/llm_addiction/rebuttal_analysis/comprehensive_risk_seeking_analysis.json')
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print("\n" + "="*100)
    print(f"✅ Results saved to: {output_file}")
    print("="*100)

if __name__ == '__main__':
    main()
