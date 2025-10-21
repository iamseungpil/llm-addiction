#!/usr/bin/env python3
"""
Load COMPLETE GPT fixed vs variable betting data including $10 fixed.

Data sources:
- $10 fixed: gpt_fixed_parsing_complete (1,600 experiments) - CORRECTED
- $30, $50, $70 fixed: gpt_fixed_bet_size (4,800 experiments)
- $10, $30, $50, $70 variable: gpt_variable_max_bet (to be verified)

TOTAL: Should be 6,400 + 6,400 = 12,800 experiments
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))
from irrationality_metrics import compute_all_metrics

# File paths
FIXED_10_FILE = '/data/llm_addiction/gpt_results_fixed_parsing/gpt_fixed_parsing_complete_20250919_151240.json'  # CORRECTED: 1,600 experiments
FIXED_30_50_70_FILE = '/home/ubuntu/llm_addiction/gpt_fixed_bet_size_experiment/results/complete_20251016_010653.json'
VARIABLE_BET_DIR = Path('/home/ubuntu/llm_addiction/gpt_variable_max_bet_experiment/results')
OUTPUT_DIR = Path('/home/ubuntu/llm_addiction/gpt_variable_max_bet_experiment/analysis/figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_fixed_10_data() -> pd.DataFrame:
    """Load $10 fixed bet experiment data from corrected results."""
    print("Loading $10 fixed bet data...")
    with open(FIXED_10_FILE, 'r') as f:
        data = json.load(f)

    results = data.get('results', [])

    # Filter only fixed betting
    fixed_results = [exp for exp in results if exp.get('bet_type') == 'fixed']

    print(f"  Found {len(fixed_results)} $10 fixed experiments")

    # Compute metrics
    records = []
    for exp in fixed_results:
        metrics = compute_all_metrics(exp)

        record = {
            'bet_type': 'fixed',
            'bet_amount': 10,  # All are $10
            'prompt_combo': exp.get('prompt_combo', 'BASE'),
            'is_bankrupt': exp.get('is_bankrupt', False),
            'total_rounds': exp.get('total_rounds', 0),
            'total_bet': exp.get('total_bet', 0),
            'final_balance': exp.get('final_balance', 0),
            **metrics
        }
        records.append(record)

    df = pd.DataFrame(records)
    print(f"  ✅ Loaded {len(df)} $10 fixed bet experiments")
    return df


def load_fixed_30_50_70_data() -> pd.DataFrame:
    """Load $30, $50, $70 fixed bet experiment data."""
    print("Loading $30, $50, $70 fixed bet data...")
    with open(FIXED_30_50_70_FILE, 'r') as f:
        data = json.load(f)

    results = data.get('results', [])

    # Compute metrics
    records = []
    for exp in results:
        metrics = compute_all_metrics(exp)

        record = {
            'bet_type': 'fixed',
            'bet_amount': exp.get('bet_size'),
            'prompt_combo': exp.get('prompt_combo', 'BASE'),
            'is_bankrupt': exp.get('is_bankrupt', False),
            'total_rounds': exp.get('total_rounds', 0),
            'total_bet': exp.get('total_bet', 0),
            'final_balance': exp.get('final_balance', 0),
            **metrics
        }
        records.append(record)

    df = pd.DataFrame(records)
    print(f"  ✅ Loaded {len(df)} fixed bet experiments ($30, $50, $70)")
    return df


def load_variable_bet_data() -> pd.DataFrame:
    """Load variable bet experiment data including intermediate files."""
    print("Loading variable bet data...")

    # Load ALL intermediate files
    intermediate_files = sorted(VARIABLE_BET_DIR.glob('intermediate_*.json'))
    print(f"  Found {len(intermediate_files)} intermediate files")

    all_experiments = []

    # Load intermediate files
    for f in intermediate_files:
        with open(f, 'r') as file:
            data = json.load(file)
            results = data.get('results', [])
            all_experiments.extend(results)

    print(f"  Loaded {len(all_experiments)} experiments from intermediate files")

    # Load restart complete files
    files_10_30 = sorted(VARIABLE_BET_DIR.glob('restart_complete_10_30_*.json'),
                        key=lambda x: x.stat().st_mtime, reverse=True)
    files_50_70 = sorted(VARIABLE_BET_DIR.glob('restart_complete_50_70_*.json'),
                        key=lambda x: x.stat().st_mtime, reverse=True)

    if files_10_30:
        with open(files_10_30[0], 'r') as f:
            data = json.load(f)
            all_experiments.extend(data.get('results', []))
            print(f"  Loaded {len(data.get('results', []))} experiments from restart_10_30")

    if files_50_70:
        with open(files_50_70[0], 'r') as f:
            data = json.load(f)
            all_experiments.extend(data.get('results', []))
            print(f"  Loaded {len(data.get('results', []))} experiments from restart_50_70")

    print(f"  Total experiments (before deduplication): {len(all_experiments)}")

    # Deduplicate based on (max_bet, prompt_combo, repetition)
    unique_experiments = {}
    for exp in all_experiments:
        key = (exp.get('max_bet'), exp.get('prompt_combo'), exp.get('repetition'))
        if key not in unique_experiments:
            unique_experiments[key] = exp

    print(f"  Unique experiments (after deduplication): {len(unique_experiments)}")

    # Process into records
    records = []
    for exp in unique_experiments.values():
        metrics = compute_all_metrics(exp)

        record = {
            'bet_type': 'variable',
            'bet_amount': exp.get('max_bet'),
            'prompt_combo': exp.get('prompt_combo', 'BASE'),
            'is_bankrupt': exp.get('is_bankrupt', False),
            'total_rounds': exp.get('total_rounds', 0),
            'total_bet': exp.get('total_bet', 0),
            'final_balance': exp.get('final_balance', 0),
            **metrics
        }

        # Compute actual average bet
        round_details = exp.get('round_details', [])
        if round_details:
            bets = [rd.get('bet_amount') for rd in round_details if rd.get('bet_amount') is not None]
            record['avg_bet'] = np.mean(bets) if bets else 0
        else:
            record['avg_bet'] = 0

        records.append(record)

    df = pd.DataFrame(records)
    print(f"  ✅ Loaded {len(df)} variable bet experiments (deduplicated)")
    return df


def complexity_from_combo(combo: str) -> int:
    """Calculate prompt complexity from combo string."""
    if not combo or combo == 'BASE':
        return 0
    components = ['G', 'M', 'P', 'R', 'W']
    return sum(1 for c in components if c in combo)


def main():
    """Main analysis pipeline."""
    print("="*80)
    print("GPT Fixed vs Variable Betting Analysis - COMPLETE DATA")
    print("="*80)

    # Load all data
    fixed_10_df = load_fixed_10_data()
    fixed_30_50_70_df = load_fixed_30_50_70_data()
    variable_df = load_variable_bet_data()

    # Combine
    df = pd.concat([fixed_10_df, fixed_30_50_70_df, variable_df], ignore_index=True)
    df['complexity'] = df['prompt_combo'].apply(complexity_from_combo)

    print(f"\n{'='*80}")
    print("COMPLETE DATA SUMMARY:")
    print("="*80)
    print(f"Total experiments: {len(df)}")
    print(f"  Fixed: {len(df[df['bet_type']=='fixed'])}")
    print(f"    $10: {len(df[(df['bet_type']=='fixed') & (df['bet_amount']==10)])}")
    print(f"    $30: {len(df[(df['bet_type']=='fixed') & (df['bet_amount']==30)])}")
    print(f"    $50: {len(df[(df['bet_type']=='fixed') & (df['bet_amount']==50)])}")
    print(f"    $70: {len(df[(df['bet_type']=='fixed') & (df['bet_amount']==70)])}")
    print(f"  Variable: {len(df[df['bet_type']=='variable'])}")
    print(f"    $10: {len(df[(df['bet_type']=='variable') & (df['bet_amount']==10)])}")
    print(f"    $30: {len(df[(df['bet_type']=='variable') & (df['bet_amount']==30)])}")
    print(f"    $50: {len(df[(df['bet_type']=='variable') & (df['bet_amount']==50)])}")
    print(f"    $70: {len(df[(df['bet_type']=='variable') & (df['bet_amount']==70)])}")

    # Save combined data for visualization scripts
    combined_path = OUTPUT_DIR.parent / 'combined_data_complete.csv'
    df.to_csv(combined_path, index=False)
    print(f"\n✅ Combined data saved to: {combined_path}")

    print("\nData loading complete!")
    print("="*80)


if __name__ == '__main__':
    main()
