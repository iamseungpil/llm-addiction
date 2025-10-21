#!/usr/bin/env python3
"""
LLaMA Streak Analysis
Analyze behavior after winning/losing streaks
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18
})

def load_llama_data():
    """Load LLaMA data"""
    print("Loading LLaMA data...")
    with open('/data/llm_addiction/experiment_0_llama_corrected/final_llama_20251004_021106.json', 'r') as f:
        data = json.load(f)
    return data['results']

def identify_streaks(history, streak_type='win', streak_length=2):
    """Identify consecutive win/loss streaks"""
    if len(history) < streak_length:
        return []

    streaks = []
    for i in range(streak_length - 1, len(history)):
        if streak_type == 'win':
            is_streak = all(history[j]['win'] for j in range(i - streak_length + 1, i + 1))
        else:
            is_streak = all(not history[j]['win'] for j in range(i - streak_length + 1, i + 1))

        if is_streak:
            streaks.append(i)

    return streaks

def analyze_streaks(experiments):
    """Analyze post-streak behavior"""
    print("Analyzing streak patterns...")

    streak_configs = [
        ('win', 2), ('loss', 2),
        ('win', 3), ('loss', 3),
        ('win', 4), ('loss', 4)
    ]

    results = {}

    for streak_type, streak_length in streak_configs:
        streak_data = {
            'occurrences': 0,
            'continued': 0,
            'stopped': 0,
            'bet_increases': 0,
            'bet_changes': [],
            'all_current_bets': []
        }

        for exp in experiments:
            history = exp.get('history', [])

            if len(history) >= streak_length:
                streak_positions = identify_streaks(history, streak_type, streak_length)

                for pos in streak_positions:
                    streak_data['occurrences'] += 1
                    current_bet = history[pos]['bet']

                    if pos + 1 < len(history):
                        next_bet = history[pos + 1]['bet']
                        streak_data['continued'] += 1

                        bet_change = next_bet - current_bet
                        streak_data['bet_changes'].append(bet_change)

                        if next_bet > current_bet:
                            streak_data['bet_increases'] += 1
                    else:
                        streak_data['stopped'] += 1
                        streak_data['bet_changes'].append(0)

                    streak_data['all_current_bets'].append(current_bet)

        # Calculate statistics
        total = streak_data['occurrences']
        if total > 0:
            continuation_rate = streak_data['continued'] / total
            bet_increase_rate = streak_data['bet_increases'] / total

            cont_se = np.sqrt(continuation_rate * (1 - continuation_rate) / total)
            cont_ci = 1.96 * cont_se

            bet_inc_se = np.sqrt(bet_increase_rate * (1 - bet_increase_rate) / total)
            bet_inc_ci = 1.96 * bet_inc_se

            bet_changes = np.array(streak_data['bet_changes'])
            mean_bet_change = np.mean(bet_changes)
            bet_change_se = stats.sem(bet_changes) if len(bet_changes) > 1 else 0
            bet_change_ci = 1.96 * bet_change_se

            current_bets = np.array(streak_data['all_current_bets'])
            pct_changes = []
            for current_bet, bet_change in zip(current_bets, bet_changes):
                if current_bet > 0:
                    pct_changes.append((bet_change / current_bet) * 100)
                else:
                    pct_changes.append(0)

            pct_changes = np.array(pct_changes)
            mean_pct_change = np.mean(pct_changes)
            pct_change_se = stats.sem(pct_changes) if len(pct_changes) > 1 else 0
            pct_change_ci = 1.96 * pct_change_se
        else:
            continuation_rate = bet_increase_rate = 0
            cont_se = bet_inc_se = cont_ci = bet_inc_ci = 0
            mean_bet_change = bet_change_se = bet_change_ci = 0
            mean_pct_change = pct_change_se = pct_change_ci = 0

        streak_key = f"{streak_length}-{streak_type}"
        results[streak_key] = {
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

    return results

def create_streak_figure(results):
    """Create streak analysis figure"""
    print("Creating streak figure...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Continuation Rate
    ax = axes[0]
    x_pos = np.arange(3)
    width = 0.35

    win_cont = [results[f'{i}-win']['continuation_rate'] * 100 for i in [2, 3, 4]]
    loss_cont = [results[f'{i}-loss']['continuation_rate'] * 100 for i in [2, 3, 4]]
    win_err = [results[f'{i}-win']['continuation_ci'] * 100 for i in [2, 3, 4]]
    loss_err = [results[f'{i}-loss']['continuation_ci'] * 100 for i in [2, 3, 4]]

    ax.bar(x_pos - width/2, win_cont, width, yerr=win_err, label='Win Streak',
           capsize=5, color='#2E86AB', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.bar(x_pos + width/2, loss_cont, width, yerr=loss_err, label='Loss Streak',
           capsize=5, color='#A23B72', alpha=0.7, edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Streak Length', fontsize=20)
    ax.set_ylabel('Continuation Rate (%)', fontsize=20)
    ax.set_title('LLaMA: Post-Streak Continuation', fontsize=22)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['2', '3', '4'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: Bet Increase Rate
    ax = axes[1]

    win_inc = [results[f'{i}-win']['bet_increase_rate'] * 100 for i in [2, 3, 4]]
    loss_inc = [results[f'{i}-loss']['bet_increase_rate'] * 100 for i in [2, 3, 4]]
    win_inc_err = [results[f'{i}-win']['bet_increase_ci'] * 100 for i in [2, 3, 4]]
    loss_inc_err = [results[f'{i}-loss']['bet_increase_ci'] * 100 for i in [2, 3, 4]]

    ax.bar(x_pos - width/2, win_inc, width, yerr=win_inc_err, label='Win Streak',
           capsize=5, color='#F18F01', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.bar(x_pos + width/2, loss_inc, width, yerr=loss_inc_err, label='Loss Streak',
           capsize=5, color='#C73E1D', alpha=0.7, edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Streak Length', fontsize=20)
    ax.set_ylabel('Bet Increase Rate (%)', fontsize=20)
    ax.set_title('LLaMA: Post-Streak Bet Increases', fontsize=22)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['2', '3', '4'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = '/home/ubuntu/llm_addiction/experiment_0_llama_gemma_restart/analysis/figures/llama/streak_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def print_summary(results):
    """Print summary statistics"""
    print("\nStreak Analysis Summary:")
    print("-" * 80)
    print(f"{'Streak':<15} {'Occurrences':<12} {'Continuation':<15} {'Bet Increase':<15}")
    print("-" * 80)

    for length in [2, 3, 4]:
        for stype in ['win', 'loss']:
            key = f'{length}-{stype}'
            r = results[key]
            print(f"{length}-{stype:<12} {r['occurrences']:<12} "
                  f"{r['continuation_rate']*100:>5.1f}%±{r['continuation_ci']*100:>4.1f}%  "
                  f"{r['bet_increase_rate']*100:>5.1f}%±{r['bet_increase_ci']*100:>4.1f}%")

def calculate_comparisons(results):
    """Calculate statistical comparisons"""
    print("\nStatistical Comparisons:")
    print("-" * 60)

    for length in [2, 3, 4]:
        win_data = results[f'{length}-win']
        loss_data = results[f'{length}-loss']

        if win_data['occurrences'] >= 5 and loss_data['occurrences'] >= 5:
            win_cont = win_data['continued']
            win_stop = win_data['stopped']
            loss_cont = loss_data['continued']
            loss_stop = loss_data['stopped']

            if win_cont + win_stop > 0 and loss_cont + loss_stop > 0:
                contingency = [[win_cont, win_stop], [loss_cont, loss_stop]]
                _, p_value = stats.chi2_contingency(contingency)[:2]
                sig = '*' if p_value < 0.05 else ''

                print(f"{length}-streak: Win {win_data['continuation_rate']*100:.1f}% vs "
                      f"Loss {loss_data['continuation_rate']*100:.1f}%, p={p_value:.4f}{sig}")

def save_csv(results):
    """Save to CSV"""
    import csv
    output_path = '/home/ubuntu/llm_addiction/experiment_0_llama_gemma_restart/analysis/figures/llama/streak_analysis.csv'

    rows = []
    for key, r in results.items():
        rows.append({
            'streak': key,
            'occurrences': r['occurrences'],
            'continuation_rate': r['continuation_rate'],
            'continuation_ci': r['continuation_ci'],
            'bet_increase_rate': r['bet_increase_rate'],
            'bet_increase_ci': r['bet_increase_ci'],
            'avg_pct_change': r['avg_pct_change'],
            'pct_change_ci': r['pct_change_ci']
        })

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {output_path}")

if __name__ == '__main__':
    experiments = load_llama_data()
    results = analyze_streaks(experiments)
    create_streak_figure(results)
    print_summary(results)
    calculate_comparisons(results)
    save_csv(results)
    print("✅ LLaMA Streak Analysis complete!")
