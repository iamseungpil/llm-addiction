#!/usr/bin/env python3
"""
Calculate behavioral effects (including bankruptcy) from REPARSED response data.
This calculates 3 metrics for the figure:
1. Safe Context Stop Rate change
2. Risky Context Stop Rate change
3. Risky Context Bankruptcy change

Uses reparsed_responses_20251125_043458.json which contains all 6 conditions:
- safe_baseline, safe_with_safe_patch, safe_with_risky_patch
- risky_baseline, risky_with_safe_patch, risky_with_risky_patch
"""

import json
import re
from pathlib import Path
import numpy as np
from scipy import stats

REPARSED_DIR = Path("/data/llm_addiction/experiment_2_multilayer_patching/reparsed")
OUTPUT_DIR = Path("/home/ubuntu/llm_addiction/rebuttal_analysis/figures_REPARSED")

RESPONSES_FILE = REPARSED_DIR / "reparsed_responses_20251125_043458.json"
SAFE_FEATURES_FILE = REPARSED_DIR / "safe_features_20251125_043600.json"
RISKY_FEATURES_FILE = REPARSED_DIR / "risky_features_20251125_043600.json"

# Risky context balance ($30) - bankruptcy if bet > 50% of balance
RISKY_BALANCE = 30
HIGH_BET_THRESHOLD = 15  # > 50% of $30


def detect_bankruptcy_from_response(response: str) -> bool:
    """Detect if the response indicates bankruptcy (balance = 0)."""
    resp_lower = response.lower()

    # Check for explicit bankruptcy indicators
    if 'bankrupt' in resp_lower:
        return True

    # Check for zero balance patterns
    zero_balance_patterns = [
        r'balance[:\s]*\$?0(?:\.|$|\s|,)',
        r'balance\s+\$0',
        r'your balance is \$?0',
        r'current balance[:\s]*\$?0',
        r'balance: \$0',
    ]

    for pattern in zero_balance_patterns:
        if re.search(pattern, resp_lower):
            return True

    return False


def load_data():
    """Load all required data files."""
    print("Loading data files...")

    with open(RESPONSES_FILE) as f:
        responses = json.load(f)
    print(f"  Loaded {len(responses)} features from responses file")

    with open(SAFE_FEATURES_FILE) as f:
        safe_data = json.load(f)
    safe_features = {f['feature'] for f in safe_data['features']}
    print(f"  Loaded {len(safe_features)} safe features")

    with open(RISKY_FEATURES_FILE) as f:
        risky_data = json.load(f)
    risky_features = {f['feature'] for f in risky_data['features']}
    print(f"  Loaded {len(risky_features)} risky features")

    return responses, safe_features, risky_features


def calculate_feature_effects(feature_data: dict) -> dict:
    """
    Calculate behavioral effects for a single feature.

    Returns dict with:
    - safe_stop_delta: stop rate change in safe context (safe_patch - baseline)
    - risky_stop_delta: stop rate change in risky context (safe_patch - baseline)
    - risky_bankruptcy_delta: bankruptcy rate change in risky context (safe_patch - baseline)
    """
    effects = {
        'safe_stop_delta': None,
        'risky_stop_delta': None,
        'risky_bankruptcy_delta': None,
    }

    # Safe context: Compare safe_with_safe_patch vs safe_baseline
    safe_baseline = feature_data.get('safe_baseline', [])
    safe_with_safe = feature_data.get('safe_with_safe_patch', [])

    if len(safe_baseline) >= 10 and len(safe_with_safe) >= 10:
        # Stop rate = proportion of trials where bet = 0
        baseline_stop_rate = sum(1 for t in safe_baseline if t['reparsed_bet'] == 0) / len(safe_baseline)
        safe_patch_stop_rate = sum(1 for t in safe_with_safe if t['reparsed_bet'] == 0) / len(safe_with_safe)
        effects['safe_stop_delta'] = safe_patch_stop_rate - baseline_stop_rate

    # Risky context: Compare risky_with_safe_patch vs risky_baseline
    risky_baseline = feature_data.get('risky_baseline', [])
    risky_with_safe = feature_data.get('risky_with_safe_patch', [])

    if len(risky_baseline) >= 10 and len(risky_with_safe) >= 10:
        # Stop rate
        baseline_stop_rate = sum(1 for t in risky_baseline if t['reparsed_bet'] == 0) / len(risky_baseline)
        safe_patch_stop_rate = sum(1 for t in risky_with_safe if t['reparsed_bet'] == 0) / len(risky_with_safe)
        effects['risky_stop_delta'] = safe_patch_stop_rate - baseline_stop_rate

        # Bankruptcy rate (high bet > 50% of balance or actual balance=0 in response)
        baseline_high_bet = sum(1 for t in risky_baseline if t['reparsed_bet'] >= HIGH_BET_THRESHOLD) / len(risky_baseline)
        safe_patch_high_bet = sum(1 for t in risky_with_safe if t['reparsed_bet'] >= HIGH_BET_THRESHOLD) / len(risky_with_safe)

        # Also check for actual bankruptcy in response
        baseline_bankrupt = sum(1 for t in risky_baseline if detect_bankruptcy_from_response(t.get('response', ''))) / len(risky_baseline)
        safe_patch_bankrupt = sum(1 for t in risky_with_safe if detect_bankruptcy_from_response(t.get('response', ''))) / len(risky_with_safe)

        # Use high bet rate as proxy for bankruptcy risk (consistent with original analysis)
        effects['risky_bankruptcy_delta'] = safe_patch_high_bet - baseline_high_bet

    return effects


def calculate_risky_feature_effects(feature_data: dict) -> dict:
    """
    Calculate behavioral effects for risky features.
    Uses risky_with_risky_patch vs baseline (opposite direction).
    """
    effects = {
        'safe_stop_delta': None,
        'risky_stop_delta': None,
        'risky_bankruptcy_delta': None,
    }

    # Safe context: Compare safe_with_risky_patch vs safe_baseline
    safe_baseline = feature_data.get('safe_baseline', [])
    safe_with_risky = feature_data.get('safe_with_risky_patch', [])

    if len(safe_baseline) >= 10 and len(safe_with_risky) >= 10:
        baseline_stop_rate = sum(1 for t in safe_baseline if t['reparsed_bet'] == 0) / len(safe_baseline)
        risky_patch_stop_rate = sum(1 for t in safe_with_risky if t['reparsed_bet'] == 0) / len(safe_with_risky)
        effects['safe_stop_delta'] = risky_patch_stop_rate - baseline_stop_rate

    # Risky context: Compare risky_with_risky_patch vs risky_baseline
    risky_baseline = feature_data.get('risky_baseline', [])
    risky_with_risky = feature_data.get('risky_with_risky_patch', [])

    if len(risky_baseline) >= 10 and len(risky_with_risky) >= 10:
        # Stop rate
        baseline_stop_rate = sum(1 for t in risky_baseline if t['reparsed_bet'] == 0) / len(risky_baseline)
        risky_patch_stop_rate = sum(1 for t in risky_with_risky if t['reparsed_bet'] == 0) / len(risky_with_risky)
        effects['risky_stop_delta'] = risky_patch_stop_rate - baseline_stop_rate

        # Bankruptcy rate (high bet)
        baseline_high_bet = sum(1 for t in risky_baseline if t['reparsed_bet'] >= HIGH_BET_THRESHOLD) / len(risky_baseline)
        risky_patch_high_bet = sum(1 for t in risky_with_risky if t['reparsed_bet'] >= HIGH_BET_THRESHOLD) / len(risky_with_risky)
        effects['risky_bankruptcy_delta'] = risky_patch_high_bet - baseline_high_bet

    return effects


def aggregate_effects(all_effects: dict, safe_features: set, risky_features: set) -> dict:
    """Aggregate effects by feature type."""
    safe_effects = {'safe_stop': [], 'risky_stop': [], 'risky_bankruptcy': []}
    risky_effects = {'safe_stop': [], 'risky_stop': [], 'risky_bankruptcy': []}

    for feature, effects in all_effects.items():
        if feature in safe_features:
            if effects['safe_stop_delta'] is not None:
                safe_effects['safe_stop'].append(effects['safe_stop_delta'])
            if effects['risky_stop_delta'] is not None:
                safe_effects['risky_stop'].append(effects['risky_stop_delta'])
            if effects['risky_bankruptcy_delta'] is not None:
                safe_effects['risky_bankruptcy'].append(effects['risky_bankruptcy_delta'])

        if feature in risky_features:
            if effects['safe_stop_delta'] is not None:
                risky_effects['safe_stop'].append(effects['safe_stop_delta'])
            if effects['risky_stop_delta'] is not None:
                risky_effects['risky_stop'].append(effects['risky_stop_delta'])
            if effects['risky_bankruptcy_delta'] is not None:
                risky_effects['risky_bankruptcy'].append(effects['risky_bankruptcy_delta'])

    # Calculate means and standard errors
    def calc_stats(values):
        if not values:
            return {'mean': 0, 'se': 0, 'n': 0}
        return {
            'mean': np.mean(values),
            'se': np.std(values) / np.sqrt(len(values)),
            'n': len(values)
        }

    summary = {
        'safe_features': {
            'n': len([f for f in all_effects if f in safe_features]),
            'safe_stop': calc_stats(safe_effects['safe_stop']),
            'risky_stop': calc_stats(safe_effects['risky_stop']),
            'risky_bankruptcy': calc_stats(safe_effects['risky_bankruptcy']),
        },
        'risky_features': {
            'n': len([f for f in all_effects if f in risky_features]),
            'safe_stop': calc_stats(risky_effects['safe_stop']),
            'risky_stop': calc_stats(risky_effects['risky_stop']),
            'risky_bankruptcy': calc_stats(risky_effects['risky_bankruptcy']),
        }
    }

    return summary


def main():
    print("=" * 60)
    print("CALCULATING BEHAVIORAL EFFECTS FROM REPARSED DATA")
    print("=" * 60)

    # Load data
    responses, safe_features, risky_features = load_data()

    # Calculate effects for each feature
    print("\nCalculating effects for each feature...")
    all_effects = {}

    safe_count = 0
    risky_count = 0

    for feature, feature_data in responses.items():
        if feature in safe_features:
            # Safe features: use safe_patch vs baseline
            effects = calculate_feature_effects(feature_data)
            all_effects[feature] = effects
            safe_count += 1
        elif feature in risky_features:
            # Risky features: use risky_patch vs baseline
            effects = calculate_risky_feature_effects(feature_data)
            all_effects[feature] = effects
            risky_count += 1

    print(f"  Processed {safe_count} safe features, {risky_count} risky features")

    # Aggregate effects
    print("\nAggregating effects...")
    summary = aggregate_effects(all_effects, safe_features, risky_features)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS (REPARSED DATA)")
    print("=" * 60)

    sf = summary['safe_features']
    rf = summary['risky_features']

    print(f"\nSafe Features (n={sf['n']}, with data: {sf['safe_stop']['n']}):")
    print(f"  Safe Context Stop Rate:     {sf['safe_stop']['mean']*100:+.1f}% (SE: {sf['safe_stop']['se']*100:.1f}%)")
    print(f"  Risky Context Stop Rate:    {sf['risky_stop']['mean']*100:+.1f}% (SE: {sf['risky_stop']['se']*100:.1f}%)")
    print(f"  Risky Context Bankruptcy:   {sf['risky_bankruptcy']['mean']*100:+.1f}% (SE: {sf['risky_bankruptcy']['se']*100:.1f}%)")

    print(f"\nRisky Features (n={rf['n']}, with data: {rf['safe_stop']['n']}):")
    print(f"  Safe Context Stop Rate:     {rf['safe_stop']['mean']*100:+.1f}% (SE: {rf['safe_stop']['se']*100:.1f}%)")
    print(f"  Risky Context Stop Rate:    {rf['risky_stop']['mean']*100:+.1f}% (SE: {rf['risky_stop']['se']*100:.1f}%)")
    print(f"  Risky Context Bankruptcy:   {rf['risky_bankruptcy']['mean']*100:+.1f}% (SE: {rf['risky_bankruptcy']['se']*100:.1f}%)")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "behavioral_effects_REPARSED.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return summary


if __name__ == "__main__":
    main()
