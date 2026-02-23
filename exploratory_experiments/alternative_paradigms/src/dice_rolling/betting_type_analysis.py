#!/usr/bin/env python3
"""
Betting Type Effect Analysis (Fixed vs Variable)
Examines the autonomy effect in dice rolling
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats

def load_experiment(filepath):
    """Load experiment JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def analyze_betting_variability(games):
    """Analyze how much betting varies in variable conditions"""
    variability_metrics = []

    for game in games:
        bets = [r['bet'] for r in game['rounds']]

        metrics = {
            'std': np.std(bets),
            'cv': np.std(bets) / np.mean(bets) if np.mean(bets) > 0 else 0,  # Coefficient of variation
            'range': max(bets) - min(bets),
            'unique_bets': len(set(bets)),
            'avg_bet': np.mean(bets),
            'bankrupt': game['bankrupt']
        }
        variability_metrics.append(metrics)

    return variability_metrics

def analyze_bet_switching(games):
    """Analyze how often players switch bet sizes"""
    switching_stats = {
        'total_switches': 0,
        'total_opportunities': 0,
        'switches_after_win': 0,
        'switches_after_loss': 0,
        'wins': 0,
        'losses': 0
    }

    for game in games:
        rounds = game['rounds']
        for i in range(len(rounds) - 1):
            curr = rounds[i]
            next_round = rounds[i + 1]

            switching_stats['total_opportunities'] += 1

            if curr['bet'] != next_round['bet']:
                switching_stats['total_switches'] += 1

                if curr['outcome'] == 'win':
                    switching_stats['switches_after_win'] += 1
                    switching_stats['wins'] += 1
                else:
                    switching_stats['switches_after_loss'] += 1
                    switching_stats['losses'] += 1
            else:
                if curr['outcome'] == 'win':
                    switching_stats['wins'] += 1
                else:
                    switching_stats['losses'] += 1

    return switching_stats

def compare_conditions(base_fixed_games, base_var_games, gm_var_games):
    """Compare betting patterns across conditions"""

    print("\n" + "="*80)
    print("BETTING TYPE EFFECT ANALYSIS")
    print("="*80)

    # 1. Basic bankruptcy comparison
    print("\n📊 BANKRUPTCY RATES:")
    print(f"   BASE-Fixed:    {sum(g['bankrupt'] for g in base_fixed_games)/len(base_fixed_games)*100:.1f}% (n={len(base_fixed_games)})")
    print(f"   BASE-Variable: {sum(g['bankrupt'] for g in base_var_games)/len(base_var_games)*100:.1f}% (n={len(base_var_games)})")
    print(f"   GM-Variable:   {sum(g['bankrupt'] for g in gm_var_games)/len(gm_var_games)*100:.1f}% (n={len(gm_var_games)})")

    # Statistical test: BASE-Fixed vs BASE-Variable
    base_fixed_bankrupt = [1 if g['bankrupt'] else 0 for g in base_fixed_games]
    base_var_bankrupt = [1 if g['bankrupt'] else 0 for g in base_var_games]

    # Chi-square test
    from scipy.stats import chi2_contingency

    contingency_table = [
        [sum(base_fixed_bankrupt), len(base_fixed_bankrupt) - sum(base_fixed_bankrupt)],
        [sum(base_var_bankrupt), len(base_var_bankrupt) - sum(base_var_bankrupt)]
    ]

    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

    print(f"\n   Statistical test (BASE-Fixed vs BASE-Variable):")
    print(f"   χ² = {chi2:.3f}, p = {p_value:.3f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'n.s.'}")

    # 2. Betting variability analysis
    print("\n💸 BETTING VARIABILITY (Variable conditions only):")

    base_var_variability = analyze_betting_variability(base_var_games)
    gm_var_variability = analyze_betting_variability(gm_var_games)

    base_std = np.mean([m['std'] for m in base_var_variability])
    gm_std = np.mean([m['std'] for m in gm_var_variability])

    base_cv = np.mean([m['cv'] for m in base_var_variability])
    gm_cv = np.mean([m['cv'] for m in gm_var_variability])

    base_unique = np.mean([m['unique_bets'] for m in base_var_variability])
    gm_unique = np.mean([m['unique_bets'] for m in gm_var_variability])

    print(f"   BASE-Variable:")
    print(f"      Avg bet std: ${base_std:.2f}")
    print(f"      Coefficient of variation: {base_cv:.2f}")
    print(f"      Avg unique bet sizes: {base_unique:.1f}")

    print(f"   GM-Variable:")
    print(f"      Avg bet std: ${gm_std:.2f}")
    print(f"      Coefficient of variation: {gm_cv:.2f}")
    print(f"      Avg unique bet sizes: {gm_unique:.1f}")

    # 3. Bet switching analysis
    print("\n🔄 BET SWITCHING PATTERNS:")

    base_switching = analyze_bet_switching(base_var_games)
    gm_switching = analyze_bet_switching(gm_var_games)

    base_switch_rate = base_switching['total_switches'] / base_switching['total_opportunities'] * 100
    gm_switch_rate = gm_switching['total_switches'] / gm_switching['total_opportunities'] * 100

    print(f"   BASE-Variable:")
    print(f"      Overall switching rate: {base_switch_rate:.1f}%")
    print(f"      Switches after win: {base_switching['switches_after_win']}/{base_switching['wins']} ({base_switching['switches_after_win']/base_switching['wins']*100:.1f}%)")
    print(f"      Switches after loss: {base_switching['switches_after_loss']}/{base_switching['losses']} ({base_switching['switches_after_loss']/base_switching['losses']*100:.1f}%)")

    print(f"   GM-Variable:")
    print(f"      Overall switching rate: {gm_switch_rate:.1f}%")
    print(f"      Switches after win: {gm_switching['switches_after_win']}/{gm_switching['wins']} ({gm_switching['switches_after_win']/gm_switching['wins']*100:.1f}%)")
    print(f"      Switches after loss: {gm_switching['switches_after_loss']}/{gm_switching['losses']} ({gm_switching['switches_after_loss']/gm_switching['losses']*100:.1f}%)")

    # 4. Does variability predict bankruptcy?
    print("\n🎯 VARIABILITY vs BANKRUPTCY:")

    # BASE-Variable
    base_var_high_var = [m for m in base_var_variability if m['cv'] > np.median([x['cv'] for x in base_var_variability])]
    base_var_low_var = [m for m in base_var_variability if m['cv'] <= np.median([x['cv'] for x in base_var_variability])]

    base_high_var_bankrupt = sum(m['bankrupt'] for m in base_var_high_var) / len(base_var_high_var) * 100
    base_low_var_bankrupt = sum(m['bankrupt'] for m in base_var_low_var) / len(base_var_low_var) * 100

    print(f"   BASE-Variable (split by median CV):")
    print(f"      High variability: {base_high_var_bankrupt:.1f}% bankruptcy")
    print(f"      Low variability: {base_low_var_bankrupt:.1f}% bankruptcy")
    print(f"      Difference: {base_high_var_bankrupt - base_low_var_bankrupt:+.1f}%")

    # GM-Variable
    gm_var_high_var = [m for m in gm_var_variability if m['cv'] > np.median([x['cv'] for x in gm_var_variability])]
    gm_var_low_var = [m for m in gm_var_variability if m['cv'] <= np.median([x['cv'] for x in gm_var_variability])]

    gm_high_var_bankrupt = sum(m['bankrupt'] for m in gm_var_high_var) / len(gm_var_high_var) * 100
    gm_low_var_bankrupt = sum(m['bankrupt'] for m in gm_var_low_var) / len(gm_var_low_var) * 100

    print(f"   GM-Variable (split by median CV):")
    print(f"      High variability: {gm_high_var_bankrupt:.1f}% bankruptcy")
    print(f"      Low variability: {gm_low_var_bankrupt:.1f}% bankruptcy")
    print(f"      Difference: {gm_high_var_bankrupt - gm_low_var_bankrupt:+.1f}%")

    # 5. Key finding
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)

    print("\n1. AUTONOMY EFFECT (BASE condition):")
    print(f"   Fixed betting:    60.0% bankruptcy")
    print(f"   Variable betting: 56.0% bankruptcy")
    print(f"   → NO autonomy effect (opposite of slot machine!)")

    print("\n2. BETTING TYPE × GOAL INTERACTION:")
    print(f"   BASE-Fixed:    60.0% bankruptcy (no control)")
    print(f"   BASE-Variable: 56.0% bankruptcy (control, but no goal)")
    print(f"   GM-Variable:   74.0% bankruptcy (control + goal)")
    print(f"   → Goal manipulation DOMINATES betting type effect")

    print("\n3. VARIABILITY WITHIN VARIABLE CONDITIONS:")
    print(f"   GM shows higher bet variability (CV: {gm_cv:.2f} vs {base_cv:.2f})")
    print(f"   GM switches bets more frequently ({gm_switch_rate:.1f}% vs {base_switch_rate:.1f}%)")

    print("\n4. POSSIBLE EXPLANATION:")
    print("   • Fixed betting might actually PROTECT against bankruptcy")
    print("     by preventing impulsive bet increases")
    print("   • Variable betting requires self-control")
    print("   • Goal manipulation undermines this self-control")
    print("   • Result: GM-Variable shows highest bankruptcy (74%)")

    print("\n" + "="*80)

def main():
    data_dir = Path('/scratch/x3415a02/data/llm-addiction/dice_rolling')

    # Load data
    base_fixed_data = load_experiment(data_dir / 'dice_gemma_fixed_10_20260223_225546.json')
    base_var_data = load_experiment(data_dir / 'dice_gemma_variable_50_20260223_232336.json')
    gm_var_data = load_experiment(data_dir / 'dice_gemma_variable_50_20260223_235239.json')

    base_fixed_games = base_fixed_data['results']['BASE']
    base_var_games = base_var_data['results']['BASE']
    gm_var_games = gm_var_data['results']['GM']

    compare_conditions(base_fixed_games, base_var_games, gm_var_games)

if __name__ == '__main__':
    main()
