#!/usr/bin/env python3
"""
Analyze GPT experiment progress and identify where it stopped
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Load latest GPT results
results_dir = Path('/data/llm_addiction/gpt_results')
latest_file = sorted(results_dir.glob('gpt_multiround_intermediate_*.json'))[-1]

print(f"Analyzing: {latest_file}")

with open(latest_file, 'r') as f:
    data = json.load(f)

results = data['results']
print(f"Total experiments completed: {len(results)}")

# Count by condition
condition_counts = defaultdict(int)
for exp in results:
    key = f"{exp['bet_type']}_{exp['first_result']}_{exp['prompt_combo']}"
    condition_counts[key] += 1

# Expected conditions
bet_types = ['fixed', 'variable']
first_results = ['W', 'L']
prompt_combos = ['BASE', 'G', 'M', 'R', 'W', 'P', 
                'GM', 'GR', 'GW', 'GP', 'MR', 'MW', 'MP', 'RW', 'RP', 'WP',
                'GMR', 'GMW', 'GMP', 'GRW', 'GRP', 'GWP', 'MRW', 'MRP', 'MWP', 'RWP',
                'GMRW', 'GMRP', 'GMWP', 'GRWP', 'MRWP', 'GMRWP']

print(f"\nExpected: {len(bet_types)} × {len(first_results)} × {len(prompt_combos)} × 50 = 6400 experiments")
print(f"Completed: {len(results)}/6400 ({len(results)/6400*100:.1f}%)")

# Find incomplete conditions
incomplete = []
complete = []

for bet_type in bet_types:
    for first_result in first_results:
        for prompt in prompt_combos:
            key = f"{bet_type}_{first_result}_{prompt}"
            count = condition_counts.get(key, 0)
            
            if count < 50:
                incomplete.append({
                    'condition': key,
                    'completed': count,
                    'needed': 50 - count
                })
            else:
                complete.append(key)

print(f"\nComplete conditions: {len(complete)}/128")
print(f"Incomplete conditions: {len(incomplete)}")

# Show last completed and first incomplete
if results:
    last_exp = results[-1]
    print(f"\nLast completed experiment:")
    print(f"  {last_exp['bet_type']}_{last_exp['first_result']}_{last_exp['prompt_combo']} (Rep {last_exp.get('repetition', 'N/A')})")

print(f"\nFirst 10 incomplete conditions:")
for cond in incomplete[:10]:
    print(f"  {cond['condition']}: {cond['completed']}/50 (need {cond['needed']} more)")

# Save resume info
resume_info = {
    'total_completed': len(results),
    'total_expected': 6400,
    'last_experiment': {
        'bet_type': last_exp['bet_type'],
        'first_result': last_exp['first_result'],  
        'prompt_combo': last_exp['prompt_combo'],
        'repetition': last_exp.get('repetition', len([r for r in results if 
                                                     r['bet_type'] == last_exp['bet_type'] and
                                                     r['first_result'] == last_exp['first_result'] and
                                                     r['prompt_combo'] == last_exp['prompt_combo']]))
    },
    'incomplete_conditions': incomplete,
    'total_remaining': sum(c['needed'] for c in incomplete)
}

resume_file = Path('/home/ubuntu/llm_addiction/gpt_experiments/src/gpt_resume_info.json')
with open(resume_file, 'w') as f:
    json.dump(resume_info, f, indent=2)

print(f"\nResume info saved to: {resume_file}")
print(f"Total experiments remaining: {resume_info['total_remaining']}")

# Check API error rate
api_errors = 0
for exp in results[-100:]:  # Check last 100 experiments
    if 'api_error' in str(exp).lower():
        api_errors += 1

print(f"\nAPI error rate (last 100): {api_errors}%")