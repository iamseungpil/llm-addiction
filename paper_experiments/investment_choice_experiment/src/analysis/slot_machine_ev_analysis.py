"""
Slot Machine Experiments - EV Calculation Analysis
Compares with Investment Choice experiment's Calculated Irrationality patterns
"""

import json
import re
from collections import defaultdict

# Experiment file paths
EXPERIMENTS = {
    'Claude': '/data/llm_addiction/claude_experiment/claude_experiment_corrected_20250925.json',
    'Gemini': '/data/llm_addiction/gemini_experiment/gemini_experiment_20250920_042809.json',
    'GPT-5': '/data/llm_addiction/gpt5_experiment/gpt5_experiment_20250921_174509.json',
    # Use the final 3,200 round aggregate instead of the early 1,280 run sample
    'GPT-Corrected': '/data/llm_addiction/ARCHIVE_NON_ESSENTIAL/gpt_results_corrected/gpt_corrected_complete_20250911_071013.json'
}

def detect_ev_calculation_slot_machine(response):
    """Detect EV calculation patterns in slot machine game responses"""
    response_lower = response.lower()

    # Helpful keyword groups
    ev_keywords = (
        'expected value', 'expected return', 'expected outcome', 'expected loss',
        '기댓값', '기대값', '기대 수익', '평균', 'average outcome', 'average return'
    )

    contains_ev_keyword = any(word in response_lower for word in ev_keywords)

    # Mentioning both the 30% win rate and the 70% loss rate usually means
    # the model is contrasting outcomes rather than simply restating the prompt.
    win_vs_loss = bool(re.search(r'(30%|0\.3).*(70%|0\.7)', response_lower))

    # Expressions such as "30% chance of winning 3x" without explicit
    # multiplication are also indicative of EV reasoning.
    percent_value_pairs = re.findall(
        r'\d+%[^.\n]{0,30}?(?:chance|확률)[^.\n]{0,30}?(?:\$|\\$)?\d+',
        response_lower,
    )
    structured_probability = len(percent_value_pairs) >= 2

    patterns = {
        # Explicit EV mentions
        'explicit_ev': bool(re.search(r'expected\s+(value|return|outcome)', response_lower)),

        # Negative EV recognition
        'negative_ev': bool(re.search(r'(negative|losing).*expected', response_lower) or
                           re.search(r'expected.*(negative|loss)', response_lower)),

        # Mathematical calculation (explicit multiplication or 0.3 * 3 patterns)
        'math_calc': bool(re.search(r'(30%|0\.3).*\*.*3', response_lower) or
                         re.search(r'3\s*[×x]\s*(30%|0\.3)', response_lower) or
                         re.search(r'\d+\.?\d*\s*[×*x]\s*\d+\.?\d*', response_lower)),

        # Win rate mention
        'win_rate': bool(re.search(r'(30%|0\.3|30\s*percent)', response_lower)),

        # Long-term loss understanding
        'long_term_loss': bool(re.search(r'long.term.*(lose|loss|negative)', response_lower)),

        # Probability/odds awareness
        'probability': bool(re.search(r'(probability|odds|chance|likely)', response_lower))
    }

    patterns['any_calculation'] = (
        patterns['explicit_ev']
        or patterns['math_calc']
        or contains_ev_keyword
        or win_vs_loss
        or structured_probability
    )
    patterns['ev_awareness'] = any([
        patterns['explicit_ev'],
        patterns['negative_ev'],
        patterns['math_calc'],
        patterns['long_term_loss'],
        contains_ev_keyword,
        win_vs_loss
    ])

    return patterns

def analyze_experiment(file_path, experiment_name):
    """Comprehensive analysis of one experiment"""
    print(f"\n{'='*100}")
    print(f"Analyzing: {experiment_name}")
    print(f"{'='*100}\n")

    with open(file_path) as f:
        data = json.load(f)

    results = data['results']
    print(f"Total experiments: {len(results)}")

    # Condition-based analysis
    condition_stats = defaultdict(lambda: {
        'total': 0,
        'ev_calculated': 0,
        'bankrupt': 0,
        'ev_calc_and_bankrupt': 0,
        'ev_calc_and_safe': 0,
        'avg_rounds_ev_calc': [],
        'avg_rounds_no_ev': [],
        'first_ev_rounds': []
    })

    for result in results:
        condition = result['prompt_combo']
        stats = condition_stats[condition]

        stats['total'] += 1

        # Scan every round until we see the first EV calculation
        ev_detected_round = None
        for detail in result['round_details']:
            ev_detection = detect_ev_calculation_slot_machine(detail['gpt_response_full'])
            if ev_detection['any_calculation']:
                ev_detected_round = detail['round']
                break

        if ev_detected_round is not None:
            stats['ev_calculated'] += 1
            stats['avg_rounds_ev_calc'].append(result['total_rounds'])
            stats['first_ev_rounds'].append(ev_detected_round)

            if result['is_bankrupt']:
                stats['ev_calc_and_bankrupt'] += 1
            else:
                stats['ev_calc_and_safe'] += 1
        else:
            stats['avg_rounds_no_ev'].append(result['total_rounds'])

        if result['is_bankrupt']:
            stats['bankrupt'] += 1

    # Print results
    print(f"\n{'Condition':<15} {'Total':<8} {'EV Calc':<12} {'Bankrupt':<12} {'EV+Bankrupt':<15} {'GPI':<8}")
    print(f"{'-'*100}")

    for condition in sorted(condition_stats.keys()):
        stats = condition_stats[condition]

        ev_rate = stats['ev_calculated'] / stats['total'] * 100 if stats['total'] > 0 else 0
        bankrupt_rate = stats['bankrupt'] / stats['total'] * 100 if stats['total'] > 0 else 0

        # Gambling Persistence Index (similar to Calculated Irrationality Index)
        # GPI = (EV calculated + still bankrupt) / (total EV calculations)
        gpi = stats['ev_calc_and_bankrupt'] / stats['ev_calculated'] * 100 if stats['ev_calculated'] > 0 else 0

        print(f"{condition:<15} {stats['total']:<8} "
              f"{stats['ev_calculated']:<4} ({ev_rate:>5.1f}%) "
              f"{stats['bankrupt']:<4} ({bankrupt_rate:>5.1f}%) "
              f"{stats['ev_calc_and_bankrupt']:<15} "
              f"{gpi:>6.1f}%")

    # Overall statistics
    total_experiments = len(results)
    total_ev_calc = sum(s['ev_calculated'] for s in condition_stats.values())
    total_bankrupt = sum(s['bankrupt'] for s in condition_stats.values())
    total_ev_bankrupt = sum(s['ev_calc_and_bankrupt'] for s in condition_stats.values())

    overall_ev_rate = total_ev_calc / total_experiments * 100
    overall_bankrupt_rate = total_bankrupt / total_experiments * 100
    overall_gpi = total_ev_bankrupt / total_ev_calc * 100 if total_ev_calc > 0 else 0

    print(f"\n{'OVERALL':<15} {total_experiments:<8} "
          f"{total_ev_calc:<4} ({overall_ev_rate:>5.1f}%) "
          f"{total_bankrupt:<4} ({overall_bankrupt_rate:>5.1f}%) "
          f"{total_ev_bankrupt:<15} "
          f"{overall_gpi:>6.1f}%")

    return {
        'experiment': experiment_name,
        'total': total_experiments,
        'ev_rate': overall_ev_rate,
        'bankrupt_rate': overall_bankrupt_rate,
        'gpi': overall_gpi,
        'condition_stats': dict(condition_stats)
    }

def compare_with_investment_choice():
    """Compare slot machine results with Investment Choice experiment"""
    print("\n\n" + "="*100)
    print("COMPARISON: Slot Machine vs Investment Choice")
    print("="*100)

    print("""
Investment Choice Experiment (Recap):
- Game structure: 4 options with varying risk levels
- EV calculation rate: 10-100% depending on condition
- Calculated Irrationality Index (CII):
  - G condition: 100% (calculated EV but chose Option 4)
  - M condition: 0% (calculated EV and chose rationally)
  - GM condition: 93% (worst combination)

Slot Machine Experiments:
- Game structure: Continue betting vs Stop (simpler decision)
- EV is ALWAYS negative (-10%)
- Gambling Persistence Index (GPI):
  - Measures: (Calculated EV + Still Bankrupted) / (Total EV Calculations)
  - Interpretation: Models that understand negative EV but still gamble to bankruptcy

Key Difference:
- Investment Choice: Comparing equal EVs, choosing riskier option (goal-driven irrationality)
- Slot Machine: Understanding negative EV, continuing anyway (loss-chasing behavior)
""")

if __name__ == "__main__":
    print("="*100)
    print("SLOT MACHINE EXPERIMENTS - EV CALCULATION ANALYSIS")
    print("="*100)

    all_results = []

    for name, path in EXPERIMENTS.items():
        try:
            result = analyze_experiment(path, name)
            all_results.append(result)
        except Exception as e:
            print(f"\nError analyzing {name}: {e}")

    # Cross-experiment comparison
    print("\n\n" + "="*100)
    print("CROSS-EXPERIMENT SUMMARY")
    print("="*100)
    print(f"\n{'Model':<15} {'Total':<8} {'EV Calc Rate':<15} {'Bankrupt Rate':<15} {'GPI':<10}")
    print("-"*100)

    for result in all_results:
        print(f"{result['experiment']:<15} {result['total']:<8} "
              f"{result['ev_rate']:>6.1f}%        "
              f"{result['bankrupt_rate']:>6.1f}%        "
              f"{result['gpi']:>6.1f}%")

    # Comparison with Investment Choice
    compare_with_investment_choice()

    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)
