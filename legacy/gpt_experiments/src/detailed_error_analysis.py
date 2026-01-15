#!/usr/bin/env python3
"""
Detailed analysis of when API errors started and their impact
"""

import json
from pathlib import Path
from collections import defaultdict

def analyze_experiment_timeline():
    gpt_results_dir = Path('/data/llm_addiction/gpt_results')
    
    # Analyze several key files to see the progression
    key_files = [
        'gpt_multiround_intermediate_20250819_055659.json',  # Early (9.1 MB)
        'gpt_multiround_intermediate_20250819_072009.json',  # Mid-early (11.8 MB) 
        'gpt_multiround_intermediate_20250819_091630.json',  # Mid (15.6 MB)
        'gpt_multiround_intermediate_20250819_140206.json',  # Later (23.6 MB)
        'gpt_multiround_intermediate_20250820_075617.json',  # Latest (32.8 MB)
    ]
    
    for filename in key_files:
        file_path = gpt_results_dir / filename
        if not file_path.exists():
            print(f"File not found: {filename}")
            continue
            
        print(f"\n{'='*80}")
        print(f"File: {filename}")
        print(f"{'='*80}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        results = data['results']
        print(f"Total experiments: {len(results)}")
        
        # Analyze error patterns
        zero_rounds = [exp for exp in results if exp.get('rounds_played', 0) == 0]
        valid_games = [exp for exp in results if exp.get('rounds_played', 0) > 0]
        bankruptcies = [exp for exp in results if exp.get('is_bankrupt', False)]
        
        print(f"Zero-round games: {len(zero_rounds)} ({len(zero_rounds)/len(results)*100:.1f}%)")
        print(f"Valid games: {len(valid_games)} ({len(valid_games)/len(results)*100:.1f}%)")
        print(f"Total bankruptcies: {len(bankruptcies)} ({len(bankruptcies)/len(results)*100:.1f}%)")
        
        if valid_games:
            avg_rounds = sum(exp.get('rounds_played', 0) for exp in valid_games) / len(valid_games)
            valid_bankruptcies = [exp for exp in valid_games if exp.get('is_bankrupt', False)]
            print(f"Valid games - Avg rounds: {avg_rounds:.1f}")
            print(f"Valid games - Bankruptcy rate: {len(valid_bankruptcies)/len(valid_games)*100:.1f}%")
            
            # Show some sample valid games
            print(f"\nSample valid games:")
            for i, exp in enumerate(valid_games[:3]):
                status = "BANKRUPT" if exp.get('is_bankrupt', False) else f"${exp.get('final_balance', 0)}"
                print(f"  {exp['bet_type']}_{exp['first_result']}_{exp['prompt_combo']}: {exp.get('rounds_played', 0)} rounds → {status}")
        
        # Check when zero-round games started appearing
        if len(results) > 100:
            first_100 = results[:100]
            last_100 = results[-100:]
            
            zero_in_first = len([exp for exp in first_100 if exp.get('rounds_played', 0) == 0])
            zero_in_last = len([exp for exp in last_100 if exp.get('rounds_played', 0) == 0])
            
            print(f"\nProgression analysis:")
            print(f"  First 100 experiments - Zero rounds: {zero_in_first}%")
            print(f"  Last 100 experiments - Zero rounds: {zero_in_last}%")
            
        # Find the transition point
        if zero_rounds:
            first_zero_exp = min(i for i, exp in enumerate(results) if exp.get('rounds_played', 0) == 0)
            print(f"\nFirst zero-round experiment at index: {first_zero_exp}")
            if first_zero_exp < len(results):
                exp = results[first_zero_exp]
                print(f"  Condition: {exp['bet_type']}_{exp['first_result']}_{exp['prompt_combo']}")
                print(f"  Experiment ID: {exp.get('experiment_id', 'N/A')}")

if __name__ == "__main__":
    analyze_experiment_timeline()