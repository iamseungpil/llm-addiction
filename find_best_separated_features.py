#!/usr/bin/env python3
"""
Find Best Separated Features from GPU 4/5 Results
===============================================

Analyzes latest experiment 2 results to find features with
the clearest causal effects for visualization.

Author: Claude Code Analysis
Date: 2025-09-14
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

def load_latest_results():
    """Load the most recent experiment 2 results from both GPUs"""
    results_dir = Path('/data/llm_addiction/results')

    # Get latest files from each GPU
    gpu4_files = sorted(list(results_dir.glob('exp2_final_intermediate_4_20250914*.json')))
    gpu5_files = sorted(list(results_dir.glob('exp2_final_intermediate_5_20250914*.json')))

    if not gpu4_files or not gpu5_files:
        print("❌ No recent experiment files found")
        return None

    gpu4_latest = gpu4_files[-1]
    gpu5_latest = gpu5_files[-1]

    print(f"📂 Loading GPU 4: {gpu4_latest}")
    print(f"📂 Loading GPU 5: {gpu5_latest}")

    all_results = []

    # Load GPU 4 results
    with open(gpu4_latest, 'r') as f:
        gpu4_data = json.load(f)
        all_results.extend(gpu4_data.get('results', []))

    # Load GPU 5 results
    with open(gpu5_latest, 'r') as f:
        gpu5_data = json.load(f)
        all_results.extend(gpu5_data.get('results', []))

    print(f"✅ Loaded {len(all_results)} total feature results")
    return all_results

def analyze_causal_effects(results):
    """Analyze causal effects and find best separated features"""

    feature_analysis = []

    for result in results:
        feature_id = result.get('feature_id')
        layer = result.get('layer')

        if not feature_id or not layer:
            continue

        # Extract safe context results
        safe_stats = result.get('safe_context', {})
        safe_safe = safe_stats.get('safe_patch', {})
        safe_risky = safe_stats.get('risky_patch', {})

        if not safe_safe or not safe_risky:
            continue

        # Calculate effect sizes
        safe_safe_stop = safe_safe.get('stop_rate', 0)
        safe_risky_stop = safe_risky.get('stop_rate', 0)

        safe_safe_bet = safe_safe.get('avg_bet', 0)
        safe_risky_bet = safe_risky.get('avg_bet', 0)

        safe_safe_bankrupt = safe_safe.get('bankruptcy_rate', 0)
        safe_risky_bankrupt = safe_risky.get('bankruptcy_rate', 0)

        # Effect calculations
        stop_effect = safe_risky_stop - safe_safe_stop
        bet_effect = safe_risky_bet - safe_safe_bet
        bankrupt_effect = safe_risky_bankrupt - safe_safe_bankrupt

        # P-values
        stop_p = safe_stats.get('stop_p_value', 1.0)
        bet_p = safe_stats.get('bet_p_value', 1.0)
        bankrupt_p = safe_stats.get('bankruptcy_p_value', 1.0)

        # Cohen's d
        cohens_d = result.get('cohens_d', 0)

        # Overall effect strength (combination of all metrics)
        effect_magnitude = abs(stop_effect) + abs(bet_effect) + abs(bankrupt_effect)

        feature_analysis.append({
            'feature_id': feature_id,
            'layer': layer,
            'feature_name': f'L{layer}-{feature_id}',
            'stop_effect': stop_effect,
            'bet_effect': bet_effect,
            'bankrupt_effect': bankrupt_effect,
            'stop_p': stop_p,
            'bet_p': bet_p,
            'bankrupt_p': bankrupt_p,
            'cohens_d': cohens_d,
            'effect_magnitude': effect_magnitude,
            'safe_safe_stop': safe_safe_stop,
            'safe_risky_stop': safe_risky_stop,
            'safe_safe_bet': safe_safe_bet,
            'safe_risky_bet': safe_risky_bet,
            'safe_safe_bankrupt': safe_safe_bankrupt,
            'safe_risky_bankrupt': safe_risky_bankrupt
        })

    return feature_analysis

def find_best_features(feature_analysis, top_n=10):
    """Find features with strongest and clearest effects"""

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(feature_analysis)

    if df.empty:
        print("❌ No valid feature data found")
        return []

    print(f"📊 Analyzing {len(df)} features...")

    # Filter for significant effects (at least one p < 0.05)
    significant_mask = (df['stop_p'] < 0.05) | (df['bet_p'] < 0.05) | (df['bankrupt_p'] < 0.05)
    significant_df = df[significant_mask]

    print(f"🔍 Found {len(significant_df)} features with significant effects")

    if significant_df.empty:
        print("⚠️ No statistically significant features found, showing top by effect magnitude")
        significant_df = df.nlargest(top_n, 'effect_magnitude')

    # Sort by combined criteria: effect magnitude and significance
    significant_df['combined_score'] = (
        significant_df['effect_magnitude'] * 2 +  # Effect size weight
        (1 - significant_df[['stop_p', 'bet_p', 'bankrupt_p']].min(axis=1)) * 3  # Significance weight
    )

    best_features = significant_df.nlargest(top_n, 'combined_score')

    return best_features.to_dict('records')

def display_results(best_features):
    """Display the best features in a readable format"""

    print(f"\n🏆 TOP {len(best_features)} FEATURES WITH STRONGEST CAUSAL EFFECTS")
    print("=" * 80)

    for i, feature in enumerate(best_features, 1):
        name = feature['feature_name']
        stop_effect = feature['stop_effect']
        bet_effect = feature['bet_effect']
        bankrupt_effect = feature['bankrupt_effect']

        stop_p = feature['stop_p']
        bet_p = feature['bet_p']
        bankrupt_p = feature['bankrupt_p']

        cohens_d = feature['cohens_d']

        print(f"\n{i}. {name}")
        print(f"   Stop Rate Effect: {stop_effect:+.3f} (p={stop_p:.3f})")
        print(f"   Bet Amount Effect: {bet_effect:+.3f} (p={bet_p:.3f})")
        print(f"   Bankruptcy Effect: {bankrupt_effect:+.3f} (p={bankrupt_p:.3f})")
        print(f"   Cohen's d: {cohens_d:.3f}")

        # Significance markers
        sig_markers = []
        if stop_p < 0.001: sig_markers.append("Stop***")
        elif stop_p < 0.01: sig_markers.append("Stop**")
        elif stop_p < 0.05: sig_markers.append("Stop*")

        if bet_p < 0.001: sig_markers.append("Bet***")
        elif bet_p < 0.01: sig_markers.append("Bet**")
        elif bet_p < 0.05: sig_markers.append("Bet*")

        if bankrupt_p < 0.001: sig_markers.append("Bankrupt***")
        elif bankrupt_p < 0.01: sig_markers.append("Bankrupt**")
        elif bankrupt_p < 0.05: sig_markers.append("Bankrupt*")

        if sig_markers:
            print(f"   Significant: {', '.join(sig_markers)}")

    return best_features

def save_best_features(best_features):
    """Save best features for visualization"""
    output_file = '/home/ubuntu/llm_addiction/best_separated_features.npz'

    # Convert to arrays for saving
    data_arrays = {}
    for key in best_features[0].keys():
        if isinstance(best_features[0][key], (int, float)):
            data_arrays[key] = np.array([f[key] for f in best_features])
        else:
            data_arrays[key] = [f[key] for f in best_features]

    np.savez(output_file, **data_arrays)
    print(f"\n💾 Best features saved to: {output_file}")

    return output_file

def main():
    """Main analysis pipeline"""
    print("🔍 FINDING BEST SEPARATED FEATURES FROM GPU 4/5 RESULTS")
    print("=" * 60)

    # Load latest results
    results = load_latest_results()
    if not results:
        return

    # Analyze causal effects
    feature_analysis = analyze_causal_effects(results)

    # Find best features
    best_features = find_best_features(feature_analysis, top_n=10)

    # Display results
    display_results(best_features)

    # Save for visualization
    if best_features:
        save_best_features(best_features)
        print(f"\n🎯 Ready to create visualization with top feature: {best_features[0]['feature_name']}")
    else:
        print("\n❌ No suitable features found for visualization")

if __name__ == '__main__':
    main()