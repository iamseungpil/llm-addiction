#!/usr/bin/env python3
"""Quick check of GPT experiment data"""

import json
from pathlib import Path
import pandas as pd

# Find latest GPT results
results_dir = Path('/data/llm_addiction/gpt_results')
json_files = list(results_dir.glob('*.json'))

print("Available files:")
for f in sorted(json_files, key=lambda x: x.stat().st_mtime)[-5:]:
    size_mb = f.stat().st_size / 1024 / 1024
    print(f"  {f.name}: {size_mb:.2f} MB")

# Load most recent
latest = sorted(json_files, key=lambda x: x.stat().st_mtime)[-1]
print(f"\nLoading: {latest.name}")

with open(latest, 'r') as f:
    data = json.load(f)

print(f"\nData structure:")
print(f"  Keys: {list(data.keys())}")

if 'results' in data:
    df = pd.DataFrame(data['results'])
    print(f"\nResults DataFrame:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    
    print(f"\nConditions:")
    print(f"  Unique prompts: {df['prompt_combo'].nunique()}")
    print(f"  Unique bet types: {df['bet_type'].unique()}")
    print(f"  Unique first results: {df['first_result'].unique()}")
    
    print(f"\nBasic stats:")
    print(f"  Total experiments: {len(df)}")
    print(f"  Bankruptcy rate: {df['is_bankrupt'].mean():.2%}")
    print(f"  Avg final balance: ${df['final_balance'].mean():.2f}")
    print(f"  Avg rounds: {df['total_rounds'].mean():.1f}")
    
    # Check for round_details
    if 'round_details' in df.columns:
        has_details = df['round_details'].notna().sum()
        print(f"\nRound details: {has_details}/{len(df)} experiments have details")
        
        # Check first experiment with details
        for idx, row in df.iterrows():
            if row['round_details'] and len(row['round_details']) > 0:
                print(f"\nFirst round detail structure:")
                print(f"  Keys: {list(row['round_details'][0].keys())}")
                break