#!/usr/bin/env python3
"""
Check if previous 10-repetition experiments also had API errors
"""

import json
from pathlib import Path
from collections import defaultdict

def check_10rep_experiments():
    # Find 10-repetition experiment files
    gpt_results_dir = Path('/data/llm_addiction/gpt_results')
    
    # Look for files that might be 10-rep experiments
    all_files = sorted(gpt_results_dir.glob('*.json'))
    
    print("Available GPT result files:")
    for f in all_files:
        print(f"  {f.name} - {f.stat().st_size / 1024 / 1024:.1f} MB")
    
    # The 10-rep experiment would be smaller than current 50-rep ones
    # Look for files with different naming or smaller sizes
    
    smaller_files = [f for f in all_files if f.stat().st_size < 10_000_000]  # Less than 10MB
    
    if smaller_files:
        print(f"\nAnalyzing smaller files (likely 10-rep experiments):")
        
        for file_path in smaller_files:
            print(f"\n{'='*60}")
            print(f"File: {file_path.name}")
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                if 'results' in data:
                    results = data['results']
                    
                    print(f"Total experiments: {len(results)}")
                    
                    # Count repetitions per condition
                    condition_counts = defaultdict(int)
                    for exp in results:
                        condition = f"{exp['bet_type']}_{exp['first_result']}_{exp['prompt_combo']}"
                        condition_counts[condition] += 1
                    
                    if condition_counts:
                        max_reps = max(condition_counts.values())
                        min_reps = min(condition_counts.values())
                        avg_reps = sum(condition_counts.values()) / len(condition_counts)
                        
                        print(f"Repetitions per condition: min={min_reps}, max={max_reps}, avg={avg_reps:.1f}")
                        
                        # Check for API error patterns (0 rounds, immediate bankruptcy)
                        zero_round_experiments = [exp for exp in results if exp.get('rounds_played', 0) == 0]
                        short_bankruptcies = [exp for exp in results if exp.get('is_bankrupt', False) and exp.get('rounds_played', 0) <= 3]
                        
                        print(f"Zero-round experiments: {len(zero_round_experiments)} ({len(zero_round_experiments)/len(results)*100:.1f}%)")
                        print(f"Short bankruptcies (≤3 rounds): {len(short_bankruptcies)} ({len(short_bankruptcies)/len(results)*100:.1f}%)")
                        
                        # Sample bankruptcy rates
                        bankruptcy_rates = defaultdict(list)
                        for exp in results:
                            condition = f"{exp['bet_type']}_{exp['first_result']}_{exp['prompt_combo']}"
                            bankruptcy_rates[condition].append(1 if exp.get('is_bankrupt', False) else 0)
                        
                        # Show conditions with high bankruptcy rates
                        high_bankruptcy = []
                        for condition, bankruptcies in bankruptcy_rates.items():
                            if len(bankruptcies) >= 5:  # At least 5 experiments
                                rate = sum(bankruptcies) / len(bankruptcies)
                                if rate > 0.7:  # >70% bankruptcy
                                    high_bankruptcy.append((condition, rate, len(bankruptcies)))
                        
                        if high_bankruptcy:
                            print(f"High bankruptcy conditions (>70%):")
                            for condition, rate, count in sorted(high_bankruptcy, key=lambda x: x[1], reverse=True)[:5]:
                                print(f"  {condition}: {rate:.1%} (n={count})")
                
            except json.JSONDecodeError:
                print(f"  Cannot parse {file_path.name} as JSON")
            except Exception as e:
                print(f"  Error reading {file_path.name}: {e}")
    
    else:
        print(f"\nNo smaller files found - all files are from 50-rep experiments")

if __name__ == "__main__":
    check_10rep_experiments()