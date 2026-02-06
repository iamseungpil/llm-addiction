#!/usr/bin/env python3
"""
Analyze API errors in GPT experiment logs
"""

import json
from pathlib import Path
from collections import defaultdict
import re

def analyze_gpt_results():
    results_dir = Path('/data/llm_addiction/gpt_results')
    latest_file = sorted(results_dir.glob('gpt_multiround_intermediate_*.json'))[-1]
    
    print(f"Analyzing: {latest_file}")
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    print(f"Total experiments: {len(results)}")
    
    # Check for API error patterns
    api_error_experiments = []
    bankruptcy_rates = defaultdict(list)
    round_counts = defaultdict(list)
    
    for i, exp in enumerate(results):
        # Check if this experiment has signs of API errors
        # (Short rounds, immediate bankruptcy, default betting patterns)
        
        rounds = exp.get('rounds_played', 0)
        is_bankrupt = exp.get('is_bankrupt', False)
        final_balance = exp.get('final_balance', 0)
        
        # Track bankruptcy rates and round counts
        condition = f"{exp['bet_type']}_{exp['first_result']}_{exp['prompt_combo']}"
        bankruptcy_rates[condition].append(1 if is_bankrupt else 0)
        round_counts[condition].append(rounds)
        
        # Look for suspicious patterns (very short games, immediate bankruptcy)
        if is_bankrupt and rounds <= 3:
            api_error_experiments.append({
                'index': i,
                'condition': condition,
                'rounds': rounds,
                'final_balance': final_balance,
                'experiment_id': exp.get('experiment_id', i+1)
            })
    
    print(f"\nSuspicious experiments (bankrupt in ≤3 rounds): {len(api_error_experiments)}")
    print(f"Percentage of total: {len(api_error_experiments)/len(results)*100:.1f}%")
    
    # Show some examples
    print(f"\nFirst 10 suspicious experiments:")
    for exp in api_error_experiments[:10]:
        print(f"  Exp {exp['experiment_id']}: {exp['condition']} - {exp['rounds']} rounds, ${exp['final_balance']}")
    
    # Analyze bankruptcy rates by condition
    high_bankruptcy_conditions = []
    for condition, bankruptcies in bankruptcy_rates.items():
        if len(bankruptcies) >= 10:  # At least 10 experiments
            bankruptcy_rate = sum(bankruptcies) / len(bankruptcies)
            avg_rounds = sum(round_counts[condition]) / len(round_counts[condition])
            
            if bankruptcy_rate > 0.8:  # >80% bankruptcy rate
                high_bankruptcy_conditions.append({
                    'condition': condition,
                    'bankruptcy_rate': bankruptcy_rate,
                    'avg_rounds': avg_rounds,
                    'n_experiments': len(bankruptcies)
                })
    
    high_bankruptcy_conditions.sort(key=lambda x: x['bankruptcy_rate'], reverse=True)
    
    print(f"\nConditions with >80% bankruptcy rate:")
    for cond in high_bankruptcy_conditions[:10]:
        print(f"  {cond['condition']}: {cond['bankruptcy_rate']:.1%} bankrupt, avg {cond['avg_rounds']:.1f} rounds, n={cond['n_experiments']}")
    
    # Look for time patterns (when did rate limits start?)
    print(f"\nAnalyzing experiment progression...")
    
    # Check last 500 experiments for patterns
    recent_experiments = results[-500:]
    recent_bankruptcies = [exp for exp in recent_experiments if exp.get('is_bankrupt', False)]
    recent_short_games = [exp for exp in recent_experiments if exp.get('rounds_played', 0) <= 3 and exp.get('is_bankrupt', False)]
    
    print(f"Last 500 experiments:")
    print(f"  Bankruptcies: {len(recent_bankruptcies)} ({len(recent_bankruptcies)/500*100:.1f}%)")
    print(f"  Short bankruptcies (≤3 rounds): {len(recent_short_games)} ({len(recent_short_games)/500*100:.1f}%)")
    
    # Check if there's a pattern in experiment IDs where errors increased
    short_bankruptcy_ids = [exp.get('experiment_id', 0) for exp in api_error_experiments]
    if short_bankruptcy_ids:
        print(f"\nShort bankruptcy experiment IDs range: {min(short_bankruptcy_ids)} to {max(short_bankruptcy_ids)}")
        
        # Group by ranges
        ranges = {
            '1-1000': len([x for x in short_bankruptcy_ids if 1 <= x <= 1000]),
            '1001-2000': len([x for x in short_bankruptcy_ids if 1001 <= x <= 2000]),
            '2001-3000': len([x for x in short_bankruptcy_ids if 2001 <= x <= 3000]),
            '3001-4000': len([x for x in short_bankruptcy_ids if 3001 <= x <= 4000]),
        }
        
        print(f"Short bankruptcies by experiment range:")
        for range_name, count in ranges.items():
            if count > 0:
                print(f"  {range_name}: {count} experiments")

if __name__ == "__main__":
    analyze_gpt_results()