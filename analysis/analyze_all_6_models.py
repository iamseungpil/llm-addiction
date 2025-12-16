#!/usr/bin/env python3
"""
Comprehensive Analysis of 6 LLM Slot Machine Experiments
Models: LLaMA-3.1-8B, Gemma-2-9B, GPT-4o-mini, GPT-4.1-mini, Claude-3.5-Sonnet, Gemini-1.5-Flash
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Data file paths
DATA_FILES = {
    'LLaMA-3.1-8B': '/data/llm_addiction/experiment_0_llama_corrected/final_llama_20251004_021106.json',
    'Gemma-2-9B': '/data/llm_addiction/experiment_0_gemma_corrected/final_gemma_20251004_172426.json',
    'GPT-4o-mini': '/data/llm_addiction/gpt_results_fixed_parsing/gpt_fixed_parsing_complete_20250919_151240.json',
    'GPT-4.1-mini': '/data/llm_addiction/gpt5_experiment/gpt5_experiment_20250921_174509.json',
    'Claude-3.5-Sonnet': '/data/llm_addiction/claude_experiment/claude_experiment_corrected_20250925.json',
    'Gemini-1.5-Flash': '/data/llm_addiction/gemini_experiment/gemini_experiment_20250920_042809.json',
}

def load_data(filepath, model_name):
    """Load and normalize data from different formats"""
    with open(filepath, 'r') as f:
        data = json.load(f)

    results = data.get('results', [])
    normalized = []

    for exp in results:
        # Determine bet type
        bet_type = exp.get('bet_type', 'unknown')

        # Determine outcome
        if 'outcome' in exp:
            is_bankrupt = exp['outcome'] in ['bankrupt', 'bankruptcy']
            voluntary_stop = exp['outcome'] == 'voluntary_stop'
        else:
            is_bankrupt = exp.get('is_bankrupt', False)
            voluntary_stop = exp.get('voluntary_stop', False)

        # Get betting history
        if 'history' in exp:
            history = exp['history']
        elif 'round_details' in exp:
            history = exp['round_details']
        elif 'game_history' in exp:
            history = exp['game_history']
        else:
            history = []

        # Extract bet amounts from history
        bets = []
        wins = []
        for h in history:
            if isinstance(h, dict):
                bet = h.get('bet', h.get('bet_amount', 0))
                if bet is None:
                    bet = 0
                # Check for win/loss
                if 'win' in h:
                    wins.append(h['win'])
                elif 'result' in h:
                    wins.append(h['result'] == 'W')
                else:
                    wins.append(None)
            else:
                bet = 0
                wins.append(None)
            bets.append(bet)

        normalized.append({
            'model': model_name,
            'bet_type': bet_type,
            'prompt_combo': exp.get('prompt_combo', 'BASE'),
            'is_bankrupt': is_bankrupt,
            'voluntary_stop': voluntary_stop,
            'final_balance': exp.get('final_balance', 100),
            'total_rounds': exp.get('total_rounds', len(history)),
            'bets': bets,
            'wins': wins,
        })

    return normalized

def calculate_irrationality_metrics(experiments):
    """Calculate irrationality indices for a set of experiments"""
    metrics = defaultdict(list)

    for exp in experiments:
        bets = exp['bets']
        wins = exp.get('wins', [])
        if len(bets) < 2:
            continue

        # Filter out None/0 bets
        valid_bets = [b for b in bets if b and b > 0]
        if len(valid_bets) < 2:
            continue

        # Loss chasing: betting more after a loss
        loss_chase_count = 0
        total_after_loss = 0

        # Excessive betting: betting > 50% of balance
        excessive_count = 0

        balance = 100
        for i, bet in enumerate(bets):
            if bet is None or bet == 0:
                continue

            if bet > balance * 0.5:
                excessive_count += 1

            # Check if previous round was a loss
            if i > 0 and len(wins) > i-1 and wins[i-1] == False:
                total_after_loss += 1
                if bets[i] > bets[i-1]:
                    loss_chase_count += 1

            # Update balance (approximate)
            if len(wins) > i and wins[i]:
                balance = balance - bet + bet * 3
            else:
                balance = balance - bet

        if total_after_loss > 0:
            metrics['loss_chasing'].append(loss_chase_count / total_after_loss)

        if len(valid_bets) > 0:
            metrics['excessive_betting'].append(excessive_count / len(valid_bets))

        # Average bet ratio
        avg_bet = np.mean(valid_bets) if valid_bets else 0
        metrics['avg_bet_ratio'].append(avg_bet / 100)

    return {k: np.mean(v) if v else 0 for k, v in metrics.items()}

def analyze_model(model_name, experiments):
    """Analyze a single model's experiments"""
    df = pd.DataFrame(experiments)

    results = {
        'model': model_name,
        'total_experiments': len(df),
    }

    # Overall bankruptcy rate
    results['bankruptcy_rate'] = df['is_bankrupt'].mean() * 100
    results['voluntary_stop_rate'] = df['voluntary_stop'].mean() * 100

    # By bet type
    for bet_type in ['fixed', 'variable']:
        subset = df[df['bet_type'] == bet_type]
        if len(subset) > 0:
            results[f'{bet_type}_bankruptcy_rate'] = subset['is_bankrupt'].mean() * 100
            results[f'{bet_type}_count'] = len(subset)
            results[f'{bet_type}_avg_rounds'] = subset['total_rounds'].mean()
            results[f'{bet_type}_avg_final_balance'] = subset['final_balance'].mean()

            # Irrationality metrics
            metrics = calculate_irrationality_metrics(subset.to_dict('records'))
            results[f'{bet_type}_loss_chasing'] = metrics.get('loss_chasing', 0)
            results[f'{bet_type}_excessive_betting'] = metrics.get('excessive_betting', 0)
            results[f'{bet_type}_avg_bet_ratio'] = metrics.get('avg_bet_ratio', 0)

    # Top risky prompt combos (variable betting)
    variable_df = df[df['bet_type'] == 'variable']
    if len(variable_df) > 0:
        prompt_bankruptcy = variable_df.groupby('prompt_combo')['is_bankrupt'].mean() * 100
        top_risky = prompt_bankruptcy.nlargest(5)
        results['top_risky_prompts'] = top_risky.to_dict()

    return results

def main():
    print("=" * 80)
    print("6-MODEL SLOT MACHINE EXPERIMENT ANALYSIS")
    print("=" * 80)
    print()

    all_results = []
    all_experiments = []

    # Load and analyze each model
    for model_name, filepath in DATA_FILES.items():
        print(f"Loading {model_name}...")
        try:
            experiments = load_data(filepath, model_name)
            all_experiments.extend(experiments)
            results = analyze_model(model_name, experiments)
            all_results.append(results)
            print(f"  Loaded {len(experiments)} experiments")
        except Exception as e:
            print(f"  Error: {e}")

    print()
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    # Create summary table
    summary_data = []
    for r in all_results:
        summary_data.append({
            'Model': r['model'],
            'Total': r['total_experiments'],
            'Overall Bankruptcy %': f"{r['bankruptcy_rate']:.2f}%",
            'Fixed Bankruptcy %': f"{r.get('fixed_bankruptcy_rate', 0):.2f}%",
            'Variable Bankruptcy %': f"{r.get('variable_bankruptcy_rate', 0):.2f}%",
            'Variable/Fixed Ratio': f"{r.get('variable_bankruptcy_rate', 0) / max(r.get('fixed_bankruptcy_rate', 0.01), 0.01):.2f}x",
        })

    summary_df = pd.DataFrame(summary_data)
    print()
    print(summary_df.to_string(index=False))

    # Detailed comparison
    print()
    print("=" * 80)
    print("DETAILED COMPARISON: FIXED vs VARIABLE BETTING")
    print("=" * 80)

    for r in all_results:
        print(f"\n{r['model']}")
        print("-" * 40)
        print(f"  Fixed betting:")
        print(f"    Bankruptcy rate: {r.get('fixed_bankruptcy_rate', 0):.2f}%")
        print(f"    Avg rounds: {r.get('fixed_avg_rounds', 0):.1f}")
        print(f"    Avg final balance: ${r.get('fixed_avg_final_balance', 0):.2f}")
        print(f"  Variable betting:")
        print(f"    Bankruptcy rate: {r.get('variable_bankruptcy_rate', 0):.2f}%")
        print(f"    Avg rounds: {r.get('variable_avg_rounds', 0):.1f}")
        print(f"    Avg final balance: ${r.get('variable_avg_final_balance', 0):.2f}")

        if 'top_risky_prompts' in r:
            print(f"  Top risky prompts (variable):")
            for prompt, rate in list(r['top_risky_prompts'].items())[:3]:
                print(f"    {prompt}: {rate:.1f}%")

    # Cross-model comparison
    print()
    print("=" * 80)
    print("CROSS-MODEL RANKING")
    print("=" * 80)

    # Sort by variable bankruptcy rate
    sorted_results = sorted(all_results, key=lambda x: x.get('variable_bankruptcy_rate', 0), reverse=True)

    print("\nMost Vulnerable (Variable Betting Bankruptcy Rate):")
    for i, r in enumerate(sorted_results, 1):
        print(f"  {i}. {r['model']}: {r.get('variable_bankruptcy_rate', 0):.2f}%")

    # Statistical tests
    print()
    print("=" * 80)
    print("STATISTICAL ANALYSIS")
    print("=" * 80)

    # Create DataFrame for statistical tests
    df_all = pd.DataFrame(all_experiments)

    print("\nFixed vs Variable Betting (All Models Combined):")
    fixed = df_all[df_all['bet_type'] == 'fixed']['is_bankrupt']
    variable = df_all[df_all['bet_type'] == 'variable']['is_bankrupt']

    if len(fixed) > 0 and len(variable) > 0:
        chi2, p_value = stats.chi2_contingency([
            [fixed.sum(), len(fixed) - fixed.sum()],
            [variable.sum(), len(variable) - variable.sum()]
        ])[:2]
        print(f"  Fixed bankruptcy: {fixed.mean()*100:.2f}% (n={len(fixed)})")
        print(f"  Variable bankruptcy: {variable.mean()*100:.2f}% (n={len(variable)})")
        print(f"  Chi-squared test p-value: {p_value:.2e}")

    # Model comparison
    print("\nBetween-Model Comparison (Variable Betting):")
    model_groups = df_all[df_all['bet_type'] == 'variable'].groupby('model')['is_bankrupt']
    model_rates = model_groups.mean() * 100
    print(model_rates.sort_values(ascending=False).to_string())

    # Save results to JSON
    output_path = '/home/ubuntu/llm_addiction/analysis/6_model_analysis_results.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()
