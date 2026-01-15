#!/usr/bin/env python3
"""
GPT Fixed vs Variable Betting Analysis.

Generates comprehensive comparison figures between fixed and variable betting experiments.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))
from irrationality_metrics import compute_all_metrics

plt.style.use('seaborn-v0_8-whitegrid')

# File paths
FIXED_BET_FILE = '/home/ubuntu/llm_addiction/gpt_fixed_bet_size_experiment/results/complete_20251016_010653.json'
VARIABLE_BET_DIR = Path('/home/ubuntu/llm_addiction/gpt_variable_max_bet_experiment/results')
OUTPUT_DIR = Path('/home/ubuntu/llm_addiction/gpt_variable_max_bet_experiment/analysis/figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_fixed_bet_data() -> pd.DataFrame:
    """Load fixed bet experiment data."""
    print("Loading fixed bet data...")
    with open(FIXED_BET_FILE, 'r') as f:
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
    print(f"  Loaded {len(df)} fixed bet experiments")
    return df


def load_variable_bet_data() -> pd.DataFrame:
    """Load variable bet experiment data."""
    print("Loading variable bet data...")

    # Find latest complete files
    files_10_30 = sorted(VARIABLE_BET_DIR.glob('restart_complete_10_30_*.json'),
                        key=lambda x: x.stat().st_mtime, reverse=True)
    files_50_70 = sorted(VARIABLE_BET_DIR.glob('restart_complete_50_70_*.json'),
                        key=lambda x: x.stat().st_mtime, reverse=True)

    records = []

    if files_10_30:
        with open(files_10_30[0], 'r') as f:
            data = json.load(f)
            for exp in data.get('results', []):
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

    if files_50_70:
        with open(files_50_70[0], 'r') as f:
            data = json.load(f)
            for exp in data.get('results', []):
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
    print(f"  Loaded {len(df)} variable bet experiments")
    return df


def complexity_from_combo(combo: str) -> int:
    """Calculate prompt complexity from combo string."""
    if not combo or combo == 'BASE':
        return 0
    components = ['G', 'M', 'P', 'R', 'W']
    return sum(1 for c in components if c in combo)


def main():
    """Main analysis pipeline."""
    print("="*70)
    print("GPT Fixed vs Variable Betting Analysis")
    print("="*70)

    # Load data
    fixed_df = load_fixed_bet_data()
    variable_df = load_variable_bet_data()

    # Combine
    df = pd.concat([fixed_df, variable_df], ignore_index=True)
    df['complexity'] = df['prompt_combo'].apply(complexity_from_combo)

    print(f"\nTotal experiments: {len(df)}")
    print(f"  Fixed: {len(fixed_df)}")
    print(f"  Variable: {len(variable_df)}")

    # Save combined data for visualization scripts
    combined_path = OUTPUT_DIR.parent / 'combined_data.csv'
    df.to_csv(combined_path, index=False)
    print(f"\nCombined data saved to: {combined_path}")

    print("\nData loading complete!")
    print("="*70)


if __name__ == '__main__':
    main()
