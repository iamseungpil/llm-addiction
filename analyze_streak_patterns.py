#!/usr/bin/env python3
"""
Analyze streak patterns (2/3/5 consecutive wins/losses) from GPT gambling data
Calculate: Frequency, Persistence rate, Bet increase rate, Mean bet change, p-values
"""

import json
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def load_gpt_data():
    """Load GPT gambling experiment data"""
    print("Loading GPT experimental data...")

    with open('/data/llm_addiction/gpt_results_corrected/gpt_corrected_complete_20250911_071013.json') as f:
        data = json.load(f)

    print(f"Loaded {len(data['results'])} experiments")
    print(f"Config: {data['experiment_config']}")

    return data['results']

def analyze_data_structure(experiments):
    """Analyze the structure of game history data"""
    print("\n=== DATA STRUCTURE ANALYSIS ===")

    total_with_history = 0
    total_rounds = 0
    max_rounds = 0

    for exp in experiments[:20]:  # Sample first 20
        if 'game_history' in exp and exp['game_history']:
            total_with_history += 1
            rounds = len(exp['game_history'])
            total_rounds += rounds
            max_rounds = max(max_rounds, rounds)

            print(f"Exp: {exp['bet_type']}_{exp['prompt_combo']} - {rounds} rounds")
            if rounds > 0:
                for i, round_data in enumerate(exp['game_history'][:3]):
                    print(f"  Round {i+1}: Bet ${round_data['bet']}, Result {round_data['result']}, Win {round_data['win']}, Balance ${round_data['balance']}")

    print(f"\nSample summary: {total_with_history}/{20} have game history")
    print(f"Max rounds in sample: {max_rounds}")

    return True

def identify_streaks(game_history, streak_type='win', streak_length=2):
    """
    Identify consecutive win/loss streaks in game history

    Args:
        game_history: List of round data
        streak_type: 'win' or 'loss'
        streak_length: Length of streak (2, 3, or 5)

    Returns:
        List of streak positions (round numbers where streak completes)
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
    Analyze behavior after streak patterns
    """
    print("\n=== STREAK PATTERN ANALYSIS ===")

    # Streak configurations to analyze
    streak_configs = [
        ('win', 2), ('loss', 2),
        ('win', 3), ('loss', 3),
        ('win', 5), ('loss', 5)
    ]

    results = []

    for streak_type, streak_length in streak_configs:
        print(f"\nAnalyzing {streak_length} consecutive {streak_type}s...")

        # Data collection
        streak_occurrences = []
        persistence_data = []  # 1 if continued, 0 if stopped
        bet_increase_data = []  # 1 if bet increased, 0 if not
        bet_changes = []  # Actual bet change amounts

        for exp in experiments:
            if 'game_history' not in exp or len(exp['game_history']) < streak_length + 1:
                continue

            game_history = exp['game_history']
            streaks = identify_streaks(game_history, streak_type, streak_length)

            for streak_end_pos in streaks:
                streak_occurrences.append(1)

                # Check if there's a next round (persistence)
                if streak_end_pos + 1 < len(game_history):
                    # Game continued
                    persistence_data.append(1)

                    # Analyze bet change
                    current_bet = game_history[streak_end_pos]['bet']
                    next_bet = game_history[streak_end_pos + 1]['bet']

                    bet_change = next_bet - current_bet
                    bet_changes.append(bet_change)

                    # Bet increase (binary)
                    bet_increase_data.append(1 if bet_change > 0 else 0)

                else:
                    # Game ended after streak
                    persistence_data.append(0)

        # Calculate statistics
        frequency = len(streak_occurrences)
        persistence_rate = np.mean(persistence_data) if persistence_data else 0
        bet_increase_rate = np.mean(bet_increase_data) if bet_increase_data else 0
        mean_bet_change = np.mean(bet_changes) if bet_changes else 0
        std_bet_change = np.std(bet_changes, ddof=1) if len(bet_changes) > 1 else 0

        # Calculate standard errors and confidence intervals for rates
        # For binary outcomes (persistence, bet increase), use binomial distribution
        if len(persistence_data) > 0:
            # Standard error for proportion
            persist_se = np.sqrt(persistence_rate * (1 - persistence_rate) / len(persistence_data))
            # 95% confidence interval
            persist_ci_lower = max(0, persistence_rate - 1.96 * persist_se)
            persist_ci_upper = min(1, persistence_rate + 1.96 * persist_se)
        else:
            persist_se = 0
            persist_ci_lower = persist_ci_upper = 0

        if len(bet_increase_data) > 0:
            bet_inc_se = np.sqrt(bet_increase_rate * (1 - bet_increase_rate) / len(bet_increase_data))
            bet_inc_ci_lower = max(0, bet_increase_rate - 1.96 * bet_inc_se)
            bet_inc_ci_upper = min(1, bet_increase_rate + 1.96 * bet_inc_se)
        else:
            bet_inc_se = 0
            bet_inc_ci_lower = bet_inc_ci_upper = 0

        # For continuous outcomes (bet changes), use normal distribution
        if len(bet_changes) > 1:
            # Standard error for mean
            bet_change_se = std_bet_change / np.sqrt(len(bet_changes))
            # 95% confidence interval
            bet_change_ci_lower = mean_bet_change - 1.96 * bet_change_se
            bet_change_ci_upper = mean_bet_change + 1.96 * bet_change_se
        else:
            bet_change_se = 0
            bet_change_ci_lower = bet_change_ci_upper = mean_bet_change

        # Statistical significance tests
        # Persistence rate vs 50% (null hypothesis: no effect)
        if len(persistence_data) > 0:
            persist_tstat, persist_pval = stats.ttest_1samp(persistence_data, 0.5)
        else:
            persist_pval = 1.0

        # Bet increase rate vs 50% (null hypothesis: no effect)
        if len(bet_increase_data) > 0:
            increase_tstat, increase_pval = stats.ttest_1samp(bet_increase_data, 0.5)
        else:
            increase_pval = 1.0

        # Mean bet change vs 0 (null hypothesis: no change)
        if len(bet_changes) > 0:
            change_tstat, change_pval = stats.ttest_1samp(bet_changes, 0)
        else:
            change_pval = 1.0

        results.append({
            'streak_type': f"{streak_length} {streak_type}s",
            'frequency': frequency,
            'persistence_rate': persistence_rate,
            'persistence_se': persist_se,
            'persistence_ci_lower': persist_ci_lower,
            'persistence_ci_upper': persist_ci_upper,
            'bet_increase_rate': bet_increase_rate,
            'bet_increase_se': bet_inc_se,
            'bet_increase_ci_lower': bet_inc_ci_lower,
            'bet_increase_ci_upper': bet_inc_ci_upper,
            'mean_bet_change': mean_bet_change,
            'std_bet_change': std_bet_change,
            'bet_change_se': bet_change_se,
            'bet_change_ci_lower': bet_change_ci_lower,
            'bet_change_ci_upper': bet_change_ci_upper,
            'persistence_pval': persist_pval,
            'bet_increase_pval': increase_pval,
            'bet_change_pval': change_pval,
            'n_streaks': len(persistence_data),
            'n_bet_changes': len(bet_increase_data)
        })

        print(f"  Frequency: {frequency}")
        print(f"  Persistence rate: {persistence_rate:.3f} ± {persist_se:.3f} [95% CI: {persist_ci_lower:.3f}-{persist_ci_upper:.3f}] (n={len(persistence_data)})")
        print(f"  Bet increase rate: {bet_increase_rate:.3f} ± {bet_inc_se:.3f} [95% CI: {bet_inc_ci_lower:.3f}-{bet_inc_ci_upper:.3f}] (n={len(bet_increase_data)})")
        print(f"  Mean bet change: ${mean_bet_change:.2f} ± ${bet_change_se:.2f} [95% CI: ${bet_change_ci_lower:.2f}-${bet_change_ci_upper:.2f}] (SD={std_bet_change:.2f})")
        print(f"  P-values: persist={persist_pval:.4f}, increase={increase_pval:.4f}, change={change_pval:.4f}")

    return results

def create_results_table(results):
    """Create formatted table of results"""
    print("\n" + "="*120)
    print("STREAK PATTERN ANALYSIS RESULTS")
    print("="*120)

    df = pd.DataFrame(results)

    # Format table with confidence intervals
    print(f"{'Streak Pattern':<15} {'Frequency':<10} {'Persistence Rate':<25} {'Bet Increase Rate':<25} {'Mean Change':<30}")
    print(f"{'':15} {'(Count)':<10} {'[95% CI]':<25} {'[95% CI]':<25} {'[95% CI] ($)':<30}")
    print("-" * 140)

    for _, row in df.iterrows():
        persist_col = f"{row['persistence_rate']:.3f} [{row['persistence_ci_lower']:.3f}-{row['persistence_ci_upper']:.3f}]"
        bet_inc_col = f"{row['bet_increase_rate']:.3f} [{row['bet_increase_ci_lower']:.3f}-{row['bet_increase_ci_upper']:.3f}]"
        bet_change_col = f"{row['mean_bet_change']:+.2f} [{row['bet_change_ci_lower']:+.2f} to {row['bet_change_ci_upper']:+.2f}]"

        print(f"{row['streak_type']:<15} {row['frequency']:<10} "
              f"{persist_col:<25} "
              f"{bet_inc_col:<25} "
              f"{bet_change_col:<30}")

    print("-" * 140)
    print("95% CI = 95% Confidence Interval")
    print("Standard errors: binomial for rates, normal for bet changes")

    print("="*120)
    print("P-values: P=Persistence, I=Bet Increase, C=Bet Change")
    print("Significance tests: Persistence & Bet Increase vs 50%, Bet Change vs $0")

    return df

def main():
    """Main analysis function"""
    print("="*80)
    print("GPT GAMBLING STREAK PATTERN ANALYSIS")
    print("="*80)

    # Load data
    experiments = load_gpt_data()

    # Analyze data structure
    analyze_data_structure(experiments)

    # Perform streak analysis
    results = analyze_post_streak_behavior(experiments)

    # Create results table
    results_df = create_results_table(results)

    # Save results
    results_df.to_csv('/home/ubuntu/llm_addiction/streak_analysis_results.csv', index=False)
    print(f"\nResults saved to: /home/ubuntu/llm_addiction/streak_analysis_results.csv")

    return results_df

if __name__ == "__main__":
    results = main()