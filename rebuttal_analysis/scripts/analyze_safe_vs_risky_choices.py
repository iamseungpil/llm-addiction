#!/usr/bin/env python3
"""
Safe (Option 1) vs Risky (Options 2-4) Analysis

Key question:
- Rational choice = Option 1 (exit with 0 loss)
- How often do models choose safe exit vs risky continuation?
- Fixed vs Variable differences
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy import stats

RESULTS_DIR = Path('/data/llm_addiction/investment_choice_experiment/results')

def load_all_decisions():
    """Load all decisions from all games"""
    all_decisions = []

    for result_file in sorted(RESULTS_DIR.glob('*.json')):
        with open(result_file) as f:
            data = json.load(f)

        model = data['experiment_config']['model']
        bet_type = data['experiment_config']['bet_type']

        for game in data['results']:
            for decision in game.get('decisions', []):
                choice = decision.get('choice')
                if choice is None:
                    continue

                all_decisions.append({
                    'model': model,
                    'bet_type': bet_type,
                    'choice': choice,
                    'is_safe': choice == 1,
                    'is_risky': choice in [2, 3, 4],
                })

    return all_decisions

def analyze_safe_vs_risky(decisions, level='overall', model=None, bet_type=None):
    """
    Analyze safe vs risky choices

    Args:
        level: 'overall', 'by_bettype', 'by_model', 'by_model_bettype'
    """
    # Filter
    if level == 'by_bettype':
        filtered = [d for d in decisions if d['bet_type'] == bet_type]
    elif level == 'by_model':
        filtered = [d for d in decisions if d['model'] == model]
    elif level == 'by_model_bettype':
        filtered = [d for d in decisions if d['model'] == model and d['bet_type'] == bet_type]
    else:  # overall
        filtered = decisions

    if not filtered:
        return None

    total = len(filtered)
    safe_count = sum(1 for d in filtered if d['is_safe'])
    risky_count = sum(1 for d in filtered if d['is_risky'])

    safe_rate = (safe_count / total * 100) if total > 0 else 0
    risky_rate = (risky_count / total * 100) if total > 0 else 0

    return {
        'total': total,
        'safe_count': safe_count,
        'risky_count': risky_count,
        'safe_rate': safe_rate,
        'risky_rate': risky_rate,
    }

def overall_analysis(decisions):
    """Overall average across all models"""

    print("="*100)
    print("PART 1: OVERALL ANALYSIS (All Models Combined)")
    print("="*100)

    # All decisions
    overall = analyze_safe_vs_risky(decisions, level='overall')

    print("\n" + "="*100)
    print("All Models, All Bet Types")
    print("="*100)
    print_safe_risky_stats(overall)

    print("\n해석:")
    print(f"  - 전체 {overall['total']:,}개 결정 중")
    print(f"  - Option 1 (안전 탈출): {overall['safe_rate']:.2f}%")
    print(f"  - Options 2-4 (위험 추구): {overall['risky_rate']:.2f}%")
    print(f"  → 합리적 선택(Option 1)보다 위험 추구가 {overall['risky_rate'] / overall['safe_rate']:.1f}배 많음!")

    # By bet type
    print("\n" + "="*100)
    print("By Bet Type")
    print("="*100)

    for bt in ['fixed', 'variable']:
        bt_stats = analyze_safe_vs_risky(decisions, level='by_bettype', bet_type=bt)

        print(f"\n{bt.upper()} BETTING:")
        print("-"*80)
        print_safe_risky_stats(bt_stats)

    # Comparison
    fixed_stats = analyze_safe_vs_risky(decisions, level='by_bettype', bet_type='fixed')
    var_stats = analyze_safe_vs_risky(decisions, level='by_bettype', bet_type='variable')

    print("\n" + "="*100)
    print("Fixed vs Variable Comparison")
    print("="*100)

    print(f"\n{'Metric':<30} {'Fixed':<20} {'Variable':<20} {'Change':<20}")
    print("-"*90)

    print(f"{'Option 1 (Safe) Rate':<30} {fixed_stats['safe_rate']:<19.2f}% "
          f"{var_stats['safe_rate']:<19.2f}% "
          f"{var_stats['safe_rate'] - fixed_stats['safe_rate']:<19.2f}pp")

    print(f"{'Options 2-4 (Risky) Rate':<30} {fixed_stats['risky_rate']:<19.2f}% "
          f"{var_stats['risky_rate']:<19.2f}% "
          f"{var_stats['risky_rate'] - fixed_stats['risky_rate']:<19.2f}pp")

    safe_ratio_change = (var_stats['safe_rate'] / fixed_stats['safe_rate'] - 1) * 100

    print(f"\n해석:")
    if var_stats['safe_rate'] > fixed_stats['safe_rate']:
        print(f"  → Variable에서 안전 선택이 {safe_ratio_change:.1f}% 증가")
        print(f"  → 베팅 금액 선택권이 주어지면 더 보수적 선택")
    else:
        print(f"  → Variable에서 안전 선택이 {-safe_ratio_change:.1f}% 감소")
        print(f"  → Fixed에서 더 보수적 선택")

    return {
        'overall': overall,
        'fixed': fixed_stats,
        'variable': var_stats,
    }

def model_specific_analysis(decisions):
    """Individual model analysis"""

    print("\n\n" + "="*100)
    print("PART 2: MODEL-SPECIFIC ANALYSIS")
    print("="*100)

    models = sorted(set(d['model'] for d in decisions))

    model_results = {}

    for model in models:
        print(f"\n{'='*100}")
        print(f"MODEL: {model.upper()}")
        print(f"{'='*100}")

        # Overall
        model_overall = analyze_safe_vs_risky(decisions, level='by_model', model=model)

        print(f"\nOverall (Fixed + Variable):")
        print("-"*80)
        print_safe_risky_stats(model_overall)

        # Fixed
        model_fixed = analyze_safe_vs_risky(decisions, level='by_model_bettype',
                                             model=model, bet_type='fixed')

        print(f"\nFIXED BETTING:")
        print("-"*80)
        print_safe_risky_stats(model_fixed)

        # Variable
        model_var = analyze_safe_vs_risky(decisions, level='by_model_bettype',
                                           model=model, bet_type='variable')

        print(f"\nVARIABLE BETTING:")
        print("-"*80)
        print_safe_risky_stats(model_var)

        # Change
        safe_change = model_var['safe_rate'] - model_fixed['safe_rate']
        risky_change = model_var['risky_rate'] - model_fixed['risky_rate']

        print(f"\nFixed → Variable Change:")
        print(f"  Option 1 (Safe):     {model_fixed['safe_rate']:.2f}% → {model_var['safe_rate']:.2f}% "
              f"({safe_change:+.2f}pp)")
        print(f"  Options 2-4 (Risky): {model_fixed['risky_rate']:.2f}% → {model_var['risky_rate']:.2f}% "
              f"({risky_change:+.2f}pp)")

        if safe_change > 0:
            print(f"  → Variable에서 더 보수적 ({safe_change:.2f}pp 증가)")
        else:
            print(f"  → Fixed에서 더 보수적 ({-safe_change:.2f}pp 더 많은 안전 선택)")

        model_results[model] = {
            'overall': model_overall,
            'fixed': model_fixed,
            'variable': model_var,
            'safe_change': safe_change,
        }

    return model_results

def print_safe_risky_stats(stats):
    """Print safe vs risky statistics"""
    if not stats:
        print("No data available")
        return

    print(f"\nTotal Decisions: {stats['total']:,}")
    print(f"\n{'Category':<20} {'Count':<15} {'Rate':<15}")
    print("-"*50)
    print(f"{'Option 1 (Safe)':<20} {stats['safe_count']:<15,} {stats['safe_rate']:<14.2f}%")
    print(f"{'Options 2-4 (Risky)':<20} {stats['risky_count']:<15,} {stats['risky_rate']:<14.2f}%")

    ratio = stats['risky_rate'] / stats['safe_rate'] if stats['safe_rate'] > 0 else float('inf')
    print(f"\nRisk/Safe Ratio: {ratio:.2f}× (위험 선택이 안전 선택의 {ratio:.1f}배)")

def comparative_summary(overall_results, model_results):
    """Summary comparison"""

    print("\n\n" + "="*100)
    print("PART 3: COMPARATIVE SUMMARY")
    print("="*100)

    # Table 1: Option 1 (Safe) selection rates
    print("\n" + "="*100)
    print("Option 1 (Safe Exit) Selection Rates")
    print("="*100)

    print(f"\n{'Model':<20} {'Overall':<15} {'Fixed':<15} {'Variable':<15} {'Change (F→V)':<20}")
    print("-"*90)

    # Overall average first
    overall_overall = overall_results['overall']['safe_rate']
    overall_fixed = overall_results['fixed']['safe_rate']
    overall_var = overall_results['variable']['safe_rate']
    overall_change = overall_var - overall_fixed

    print(f"{'OVERALL AVERAGE':<20} {overall_overall:<14.2f}% {overall_fixed:<14.2f}% "
          f"{overall_var:<14.2f}% {overall_change:<19.2f}pp")
    print("-"*90)

    # Individual models
    for model in sorted(model_results.keys()):
        r = model_results[model]
        overall_rate = r['overall']['safe_rate']
        fixed_rate = r['fixed']['safe_rate']
        var_rate = r['variable']['safe_rate']
        change = r['safe_change']

        print(f"{model:<20} {overall_rate:<14.2f}% {fixed_rate:<14.2f}% "
              f"{var_rate:<14.2f}% {change:<19.2f}pp")

    # Table 2: Risk/Safe Ratio
    print("\n" + "="*100)
    print("Risk/Safe Ratio (높을수록 위험 추구적)")
    print("="*100)

    print(f"\n{'Model':<20} {'Fixed Ratio':<20} {'Variable Ratio':<20}")
    print("-"*60)

    overall_fixed_ratio = overall_results['fixed']['risky_rate'] / overall_results['fixed']['safe_rate']
    overall_var_ratio = overall_results['variable']['risky_rate'] / overall_results['variable']['safe_rate']

    print(f"{'OVERALL AVERAGE':<20} {overall_fixed_ratio:<19.2f}× {overall_var_ratio:<19.2f}×")
    print("-"*60)

    for model in sorted(model_results.keys()):
        r = model_results[model]
        fixed_ratio = r['fixed']['risky_rate'] / r['fixed']['safe_rate'] if r['fixed']['safe_rate'] > 0 else float('inf')
        var_ratio = r['variable']['risky_rate'] / r['variable']['safe_rate'] if r['variable']['safe_rate'] > 0 else float('inf')

        print(f"{model:<20} {fixed_ratio:<19.2f}× {var_ratio:<19.2f}×")

    # Interpretation
    print("\n" + "="*100)
    print("핵심 발견")
    print("="*100)

    print("\n1. 합리성 vs 실제 선택:")
    print(f"   - 합리적 선택: Option 1 (100% 확률로 베팅금 회수, 게임 종료)")
    print(f"   - 실제 전체 평균: Option 1 선택률 {overall_overall:.2f}%")
    print(f"   → {100 - overall_overall:.2f}%가 비합리적 위험 추구!")

    print("\n2. Fixed vs Variable:")
    if overall_change > 0:
        print(f"   - Variable에서 안전 선택 {overall_change:.2f}pp 증가")
        print(f"   - 베팅 금액 통제권이 주어지면 더 보수적")
    else:
        print(f"   - Fixed에서 안전 선택 {-overall_change:.2f}pp 더 많음")

    print("\n3. 모델별 차이:")
    safest_model = max(model_results.items(), key=lambda x: x[1]['overall']['safe_rate'])
    riskiest_model = min(model_results.items(), key=lambda x: x[1]['overall']['safe_rate'])

    print(f"   - 가장 보수적: {safest_model[0]} ({safest_model[1]['overall']['safe_rate']:.2f}% 안전 선택)")
    print(f"   - 가장 위험 추구: {riskiest_model[0]} ({riskiest_model[1]['overall']['safe_rate']:.2f}% 안전 선택)")

def statistical_test(decisions):
    """Statistical significance test"""

    print("\n\n" + "="*100)
    print("PART 4: STATISTICAL SIGNIFICANCE TEST")
    print("="*100)

    # Chi-square test: Fixed vs Variable
    print("\n" + "="*100)
    print("Chi-Square Test: Fixed vs Variable")
    print("="*100)

    fixed_decs = [d for d in decisions if d['bet_type'] == 'fixed']
    var_decs = [d for d in decisions if d['bet_type'] == 'variable']

    fixed_safe = sum(1 for d in fixed_decs if d['is_safe'])
    fixed_risky = sum(1 for d in fixed_decs if d['is_risky'])
    var_safe = sum(1 for d in var_decs if d['is_safe'])
    var_risky = sum(1 for d in var_decs if d['is_risky'])

    print(f"\nContingency Table:")
    print(f"                 Safe (Opt1)    Risky (Opt2-4)    Total")
    print(f"Fixed            {fixed_safe:>10}     {fixed_risky:>10}        {fixed_safe + fixed_risky:>10}")
    print(f"Variable         {var_safe:>10}     {var_risky:>10}        {var_safe + var_risky:>10}")

    contingency = [[fixed_safe, fixed_risky], [var_safe, var_risky]]
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

    print(f"\nChi-square test:")
    print(f"  χ² = {chi2:.3f}")
    print(f"  p-value = {p_value:.3e}")
    print(f"  degrees of freedom = {dof}")

    if p_value < 0.001:
        print(f"\n  → Highly significant difference (p < 0.001) ***")
    elif p_value < 0.01:
        print(f"\n  → Significant difference (p < 0.01) **")
    elif p_value < 0.05:
        print(f"\n  → Significant difference (p < 0.05) *")
    else:
        print(f"\n  → No significant difference (p >= 0.05)")

def main():
    print("Loading data...")
    decisions = load_all_decisions()
    print(f"Total decisions: {len(decisions):,}\n")

    # Part 1: Overall
    overall_results = overall_analysis(decisions)

    # Part 2: Model-specific
    model_results = model_specific_analysis(decisions)

    # Part 3: Summary
    comparative_summary(overall_results, model_results)

    # Part 4: Statistical test
    statistical_test(decisions)

    # Save
    output = {
        'overall': {
            'overall': overall_results['overall'],
            'fixed': overall_results['fixed'],
            'variable': overall_results['variable'],
        },
        'by_model': {
            model: {
                'overall_safe_rate': data['overall']['safe_rate'],
                'fixed_safe_rate': data['fixed']['safe_rate'],
                'variable_safe_rate': data['variable']['safe_rate'],
                'safe_change': data['safe_change'],
            }
            for model, data in model_results.items()
        }
    }

    output_file = Path('/home/ubuntu/llm_addiction/rebuttal_analysis/safe_vs_risky_analysis.json')
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print("\n" + "="*100)
    print(f"✅ Results saved to: {output_file}")
    print("="*100)

if __name__ == '__main__':
    main()
