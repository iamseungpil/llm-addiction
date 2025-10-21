#!/usr/bin/env python3
import json
import re
from collections import defaultdict
import numpy as np
from scipy import stats

def extract_bet_amount(response):
    amounts = re.findall(r'\$(\d+)', response)
    if amounts:
        return int(amounts[-1])
    return None

def analyze_model_data(data, model_name):
    print(f"\n{'='*60}")
    print(f"Analyzing {model_name}")
    print(f"{'='*60}")

    if isinstance(data, dict) and 'results' in data:
        experiments = data['results']
    elif isinstance(data, list):
        experiments = data
    else:
        experiments = [data]

    fixed_betting = []
    variable_betting = []

    for exp in experiments:
        bet_type = exp.get('bet_type', 'unknown')
        outcome = exp.get('outcome', 'unknown')

        if bet_type == 'fixed':
            fixed_betting.append(exp)
        elif bet_type == 'variable':
            variable_betting.append(exp)

    print(f"\nTotal experiments: {len(experiments)}")
    print(f"Fixed betting: {len(fixed_betting)}")
    print(f"Variable betting: {len(variable_betting)}")

    results = {}

    for bet_type_name, bet_experiments in [('Fixed', fixed_betting), ('Variable', variable_betting)]:
        if not bet_experiments:
            continue

        print(f"\n{bet_type_name} Betting Analysis:")
        print(f"-" * 40)

        bankruptcies = [e for e in bet_experiments if e.get('is_bankrupt') or e.get('outcome') == 'bankruptcy']
        voluntary_stops = [e for e in bet_experiments if e.get('voluntary_stop') or e.get('outcome') == 'voluntary_stop']

        bankruptcy_rate = len(bankruptcies) / len(bet_experiments) * 100 if bet_experiments else 0

        total_rounds = []
        total_bets = []

        for exp in bet_experiments:
            rounds = exp.get('total_rounds', len(exp.get('history', [])))
            total_rounds.append(rounds)

            bet_sum = exp.get('total_bet', 0)
            won_sum = exp.get('total_won', 0)

            if bet_sum == 0:
                for entry in exp.get('history', []):
                    response = entry.get('response', '')
                    bet = extract_bet_amount(response)
                    if bet:
                        bet_sum += bet
            total_bets.append(bet_sum)

        avg_rounds = np.mean(total_rounds) if total_rounds else 0
        avg_total_bet = np.mean(total_bets) if total_bets else 0

        total_won_all = [exp.get('total_won', 0) for exp in bet_experiments]
        avg_total_won = np.mean(total_won_all) if total_won_all else 0
        avg_net_pl = avg_total_won - avg_total_bet

        print(f"Experiments: {len(bet_experiments)}")
        print(f"Bankruptcies: {len(bankruptcies)} ({bankruptcy_rate:.2f}%)")
        print(f"Voluntary stops: {len(voluntary_stops)}")
        print(f"Avg rounds: {avg_rounds:.2f}")
        print(f"Avg total bet: ${avg_total_bet:.2f}")
        print(f"Avg total won: ${avg_total_won:.2f}")
        print(f"Avg net P/L: ${avg_net_pl:.2f}")

        results[bet_type_name] = {
            'n': len(bet_experiments),
            'bankruptcies': len(bankruptcies),
            'bankruptcy_rate': bankruptcy_rate,
            'voluntary_stops': len(voluntary_stops),
            'avg_rounds': avg_rounds,
            'avg_total_bet': avg_total_bet,
            'avg_total_won': avg_total_won,
            'avg_net_pl': avg_net_pl
        }

    return results

def main():
    models_data = {}

    print("Loading GPT-4o-mini data...")
    with open('/data/llm_addiction/gpt_results_fixed_parsing/gpt_fixed_parsing_complete_20250919_151240.json', 'r') as f:
        gpt_data = json.load(f)
    models_data['GPT-4o-mini'] = analyze_model_data(gpt_data, 'GPT-4o-mini')

    print("\n\nLoading Claude data...")
    with open('/data/llm_addiction/claude_experiment/claude_experiment_20250920_031403.json', 'r') as f:
        claude_data = json.load(f)
    models_data['Claude'] = analyze_model_data(claude_data, 'Claude')

    print("\n\nLoading Gemini data...")
    with open('/data/llm_addiction/gemini_experiment/gemini_experiment_20250920_042809.json', 'r') as f:
        gemini_data = json.load(f)
    models_data['Gemini'] = analyze_model_data(gemini_data, 'Gemini')

    print("\n\nLoading LLaMA data (this may take a while, 14GB main + 453MB additional)...")
    with open('/data/llm_addiction/results/exp1_multiround_intermediate_20250819_140040.json', 'r') as f:
        llama_main_raw = json.load(f)
    with open('/data/llm_addiction/results/exp1_missing_complete_20250820_090040.json', 'r') as f:
        llama_additional_raw = json.load(f)

    if isinstance(llama_main_raw, list):
        llama_main = llama_main_raw
    else:
        llama_main = llama_main_raw.get('results', llama_main_raw)

    if isinstance(llama_additional_raw, list):
        llama_additional = llama_additional_raw
    else:
        llama_additional = llama_additional_raw.get('results', llama_additional_raw)

    llama_data = {'results': llama_main + llama_additional}
    print(f"Combined LLaMA data: {len(llama_main)} + {len(llama_additional)} = {len(llama_data['results'])}")
    models_data['LLaMA'] = analyze_model_data(llama_data, 'LLaMA')

    with open('/home/ubuntu/llm_addiction/all_models_comprehensive_stats.json', 'w') as f:
        json.dump(models_data, f, indent=2)

    print("\n\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)

    print("\n{:<15} {:<12} {:<10} {:<15} {:<15} {:<15}".format(
        "Model", "Bet Type", "N", "Bankrupt %", "Avg Rounds", "Avg Total Bet"
    ))
    print("-" * 90)

    for model in ['GPT-4o-mini', 'Claude', 'Gemini', 'LLaMA']:
        if model in models_data:
            for bet_type in ['Fixed', 'Variable']:
                if bet_type in models_data[model]:
                    data = models_data[model][bet_type]
                    print("{:<15} {:<12} {:<10} {:<15.2f} {:<15.2f} ${:<14.2f}".format(
                        model, bet_type, data['n'], data['bankruptcy_rate'],
                        data['avg_rounds'], data['avg_total_bet']
                    ))

    print("\n\nResults saved to: all_models_comprehensive_stats.json")

if __name__ == '__main__':
    main()