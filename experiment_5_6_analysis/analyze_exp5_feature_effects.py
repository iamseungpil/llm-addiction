#!/usr/bin/env python3
"""
Experiment 5 Feature Effect Analysis
Analyzes the effect of mean-value feature patching on bankruptcy rate and final balance
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import sys

def load_exp5_data(filepath: str) -> Dict:
    """Load Experiment 5 data"""
    print(f"Loading Experiment 5 data from {filepath}...")
    with open(filepath, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data['results'])} features")
    return data

def calculate_feature_effects(data: Dict) -> pd.DataFrame:
    """
    Calculate the effect of each feature on bankruptcy rate and final balance

    Returns DataFrame with columns:
    - feature_id, layer
    - baseline_bankruptcy_rate, safe_bankruptcy_rate, risky_bankruptcy_rate
    - baseline_balance, safe_balance, risky_balance
    - delta_bankruptcy_safe, delta_bankruptcy_risky
    - delta_balance_safe, delta_balance_risky
    """

    results_list = []

    for feature_result in data['results']:
        feature_id = feature_result['feature_id']
        layer = feature_result.get('layer', 'unknown')

        # Organize by condition
        by_condition = {}
        for result in feature_result['results']:
            condition = result['condition']
            by_condition[condition] = result

        # Extract metrics for each condition
        baseline = by_condition.get('baseline', {})
        safe = by_condition.get('patch_to_safe_mean', {})
        risky = by_condition.get('patch_to_bankrupt_mean', {})

        # Calculate rates
        def get_stats(cond_result):
            if not cond_result:
                return None, None, None
            total = cond_result.get('n_trials', 0)
            bankruptcy_rate = cond_result.get('bankruptcy_rate', 0)
            balance = cond_result.get('avg_final_balance', 0)
            return bankruptcy_rate, balance, total

        baseline_br, baseline_bal, baseline_n = get_stats(baseline)
        safe_br, safe_bal, safe_n = get_stats(safe)
        risky_br, risky_bal, risky_n = get_stats(risky)

        if baseline_br is None:
            continue

        # Calculate deltas
        delta_br_safe = safe_br - baseline_br if safe_br is not None else None
        delta_br_risky = risky_br - baseline_br if risky_br is not None else None
        delta_bal_safe = safe_bal - baseline_bal if safe_bal is not None else None
        delta_bal_risky = risky_bal - baseline_bal if risky_bal is not None else None

        results_list.append({
            'feature_id': feature_id,
            'layer': layer,
            'baseline_bankruptcy_rate': baseline_br,
            'safe_bankruptcy_rate': safe_br,
            'risky_bankruptcy_rate': risky_br,
            'baseline_balance': baseline_bal,
            'safe_balance': safe_bal,
            'risky_balance': risky_bal,
            'delta_bankruptcy_safe': delta_br_safe,
            'delta_bankruptcy_risky': delta_br_risky,
            'delta_balance_safe': delta_bal_safe,
            'delta_balance_risky': delta_bal_risky,
            'baseline_n': baseline_n,
            'safe_n': safe_n,
            'risky_n': risky_n
        })

    df = pd.DataFrame(results_list)
    return df

def rank_features(df: pd.DataFrame, metric: str = 'delta_bankruptcy_safe', top_n: int = 50) -> pd.DataFrame:
    """Rank features by specified metric"""
    # Filter out None values and sort by absolute value
    df_filtered = df[df[metric].notna()].copy()
    df_sorted = df_filtered.sort_values(by=metric, key=lambda x: x.abs(), ascending=False)
    return df_sorted.head(top_n)

def identify_harmful_features(df: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
    """
    Identify features where patching INCREASED bankruptcy rate
    (harmful intervention)
    """
    harmful_safe = df[df['delta_bankruptcy_safe'] > threshold].copy()
    harmful_risky = df[df['delta_bankruptcy_risky'] > threshold].copy()

    # Features harmful in both conditions
    harmful_both = df[
        (df['delta_bankruptcy_safe'] > threshold) &
        (df['delta_bankruptcy_risky'] > threshold)
    ].copy()

    return {
        'harmful_safe': harmful_safe,
        'harmful_risky': harmful_risky,
        'harmful_both': harmful_both
    }

def identify_protective_features(df: pd.DataFrame, threshold: float = -0.05) -> pd.DataFrame:
    """
    Identify features where patching DECREASED bankruptcy rate
    (protective intervention)
    """
    protective_safe = df[df['delta_bankruptcy_safe'] < threshold].copy()
    protective_risky = df[df['delta_bankruptcy_risky'] < threshold].copy()

    return {
        'protective_safe': protective_safe,
        'protective_risky': protective_risky
    }

def analyze_by_layer(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze effects by layer"""

    layer_stats = df.groupby('layer').agg({
        'delta_bankruptcy_safe': ['mean', 'std', 'count'],
        'delta_bankruptcy_risky': ['mean', 'std', 'count'],
        'delta_balance_safe': ['mean', 'std'],
        'delta_balance_risky': ['mean', 'std']
    }).reset_index()

    return layer_stats

def main():
    # Paths
    exp5_path = "/data/llm_addiction/experiment_5_multiround_patching/multiround_patching_final_20251012_021759.json"
    output_dir = Path("/home/ubuntu/llm_addiction/experiment_5_6_analysis/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data = load_exp5_data(exp5_path)

    # Calculate effects
    print("\nCalculating feature effects...")
    df = calculate_feature_effects(data)

    # Save full results
    df.to_csv(output_dir / "exp5_all_features.csv", index=False)
    print(f"Saved all features to {output_dir / 'exp5_all_features.csv'}")

    # Overall statistics
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    print(f"\nBaseline bankruptcy rate: {df['baseline_bankruptcy_rate'].mean():.3f}")
    print(f"Safe mean bankruptcy rate: {df['safe_bankruptcy_rate'].mean():.3f}")
    print(f"Risky mean bankruptcy rate: {df['risky_bankruptcy_rate'].mean():.3f}")
    print(f"\nBaseline balance: ${df['baseline_balance'].mean():.2f}")
    print(f"Safe mean balance: ${df['safe_balance'].mean():.2f}")
    print(f"Risky mean balance: ${df['risky_balance'].mean():.2f}")

    # Top features by bankruptcy rate change
    print("\n" + "="*80)
    print("TOP 20 FEATURES BY BANKRUPTCY RATE CHANGE (SAFE MEAN)")
    print("="*80)
    top_safe = rank_features(df, 'delta_bankruptcy_safe', 20)
    top_safe.to_csv(output_dir / "exp5_top20_bankruptcy_safe.csv", index=False)
    print(top_safe[['feature_id', 'layer', 'baseline_bankruptcy_rate', 'safe_bankruptcy_rate', 'delta_bankruptcy_safe']].to_string())

    print("\n" + "="*80)
    print("TOP 20 FEATURES BY BANKRUPTCY RATE CHANGE (RISKY MEAN)")
    print("="*80)
    top_risky = rank_features(df, 'delta_bankruptcy_risky', 20)
    top_risky.to_csv(output_dir / "exp5_top20_bankruptcy_risky.csv", index=False)
    print(top_risky[['feature_id', 'layer', 'baseline_bankruptcy_rate', 'risky_bankruptcy_rate', 'delta_bankruptcy_risky']].to_string())

    # Top features by balance change
    print("\n" + "="*80)
    print("TOP 20 FEATURES BY BALANCE CHANGE (SAFE MEAN)")
    print("="*80)
    top_balance_safe = rank_features(df, 'delta_balance_safe', 20)
    top_balance_safe.to_csv(output_dir / "exp5_top20_balance_safe.csv", index=False)
    print(top_balance_safe[['feature_id', 'layer', 'baseline_balance', 'safe_balance', 'delta_balance_safe']].to_string())

    # Harmful features
    print("\n" + "="*80)
    print("HARMFUL INTERVENTION FEATURES")
    print("="*80)
    harmful = identify_harmful_features(df, threshold=0.05)
    print(f"\nFeatures that increased bankruptcy when patched to SAFE mean: {len(harmful['harmful_safe'])}")
    print(f"Features that increased bankruptcy when patched to RISKY mean: {len(harmful['harmful_risky'])}")
    print(f"Features that increased bankruptcy in BOTH conditions: {len(harmful['harmful_both'])}")

    if len(harmful['harmful_both']) > 0:
        print("\nTop 10 'Harmful Both' features:")
        harmful_both_sorted = harmful['harmful_both'].sort_values('delta_bankruptcy_safe', ascending=False).head(10)
        print(harmful_both_sorted[['feature_id', 'layer', 'delta_bankruptcy_safe', 'delta_bankruptcy_risky']].to_string())
        harmful['harmful_both'].to_csv(output_dir / "exp5_harmful_both.csv", index=False)

    harmful['harmful_safe'].to_csv(output_dir / "exp5_harmful_safe.csv", index=False)
    harmful['harmful_risky'].to_csv(output_dir / "exp5_harmful_risky.csv", index=False)

    # Protective features
    print("\n" + "="*80)
    print("PROTECTIVE INTERVENTION FEATURES")
    print("="*80)
    protective = identify_protective_features(df, threshold=-0.05)
    print(f"\nFeatures that decreased bankruptcy when patched to SAFE mean: {len(protective['protective_safe'])}")
    print(f"Features that decreased bankruptcy when patched to RISKY mean: {len(protective['protective_risky'])}")

    if len(protective['protective_safe']) > 0:
        print("\nTop 10 protective (safe mean) features:")
        protective_safe_sorted = protective['protective_safe'].sort_values('delta_bankruptcy_safe').head(10)
        print(protective_safe_sorted[['feature_id', 'layer', 'delta_bankruptcy_safe', 'delta_balance_safe']].to_string())
        protective['protective_safe'].to_csv(output_dir / "exp5_protective_safe.csv", index=False)

    # Layer analysis
    print("\n" + "="*80)
    print("LAYER-WISE ANALYSIS")
    print("="*80)
    layer_stats = analyze_by_layer(df)
    layer_stats.to_csv(output_dir / "exp5_layer_stats.csv", index=False)
    print(layer_stats.to_string())

    print(f"\n\nAll results saved to: {output_dir}")
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
