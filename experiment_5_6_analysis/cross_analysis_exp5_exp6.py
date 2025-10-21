#!/usr/bin/env python3
"""
Cross-Analysis: Experiment 5 & 6
Maps harmful/protective features from Exp 5 to their natural token activation patterns in Exp 6
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

def load_exp5_features(results_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load Experiment 5 feature analysis results"""
    return {
        'all': pd.read_csv(results_dir / "exp5_all_features.csv"),
        'harmful_both': pd.read_csv(results_dir / "exp5_harmful_both.csv"),
        'harmful_safe': pd.read_csv(results_dir / "exp5_harmful_safe.csv"),
        'harmful_risky': pd.read_csv(results_dir / "exp5_harmful_risky.csv"),
        'protective_safe': pd.read_csv(results_dir / "exp5_protective_safe.csv")
    }

def load_exp6_data(filepath: str) -> Dict:
    """Load Experiment 6 token-level tracking data"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def extract_feature_token_profile(data: Dict, layer: int, feature_id: int) -> pd.DataFrame:
    """
    Extract a specific feature's activation profile across all scenarios and token positions

    Returns DataFrame with columns:
    - scenario, balance, token_position, token_type, activation
    """

    layer_str = f'L{layer}'
    profiles = []

    for result in data['results']:
        scenario = result['scenario']
        balance = result['balance']
        key_positions = result['key_positions']

        if layer_str not in result['layers']:
            continue

        features = result['layers'][layer_str]['features']

        # Extract activation at balance token
        if key_positions['balance']:
            pos = key_positions['balance'][0]
            profiles.append({
                'scenario': scenario,
                'balance': balance,
                'token_position': pos,
                'token_type': 'balance',
                'activation': features[pos][feature_id]
            })

        # Extract activation at goal token
        if key_positions['goal']:
            pos = key_positions['goal'][0]
            profiles.append({
                'scenario': scenario,
                'balance': balance,
                'token_position': pos,
                'token_type': 'goal',
                'activation': features[pos][feature_id]
            })

        # Extract activation at probability token
        if key_positions['probability']:
            pos = key_positions['probability'][0]
            profiles.append({
                'scenario': scenario,
                'balance': balance,
                'token_position': pos,
                'token_type': 'probability',
                'activation': features[pos][feature_id]
            })

        # Extract activation at choice tokens (average)
        if key_positions['choices']:
            choice_acts = [features[pos][feature_id] for pos in key_positions['choices']]
            profiles.append({
                'scenario': scenario,
                'balance': balance,
                'token_position': -1,  # Marker for averaged position
                'token_type': 'choice',
                'activation': np.mean(choice_acts)
            })

    return pd.DataFrame(profiles)

def analyze_harmful_feature_activations(exp5_features: pd.DataFrame, exp6_data: Dict,
                                       top_n: int = 20) -> pd.DataFrame:
    """
    Analyze token activation patterns for top harmful features from Exp 5
    Note: Only analyzes features from layers present in Exp 6 (8, 15, 31)
    """

    # Filter to only layers present in Exp 6
    exp6_layers = [8, 15, 31]
    features_in_exp6 = exp5_features[exp5_features['layer'].isin(exp6_layers)]

    if len(features_in_exp6) == 0:
        print(f"WARNING: No features from Exp 5 exist in Exp 6 layers {exp6_layers}")
        return pd.DataFrame()

    # Sort by absolute delta bankruptcy to get most impactful features
    harmful_sorted = features_in_exp6.sort_values(
        by='delta_bankruptcy_safe',
        key=lambda x: x.abs(),
        ascending=False
    ).head(top_n)

    results = []

    for _, row in harmful_sorted.iterrows():
        feature_id = int(row['feature_id'])
        layer = int(row['layer'])
        feature_name = f"L{layer}-{feature_id}"

        # Extract token profile from Exp 6
        profile = extract_feature_token_profile(exp6_data, layer, feature_id)

        # Skip if profile is empty
        if len(profile) == 0:
            continue

        # Calculate statistics by token type
        token_stats = profile.groupby('token_type')['activation'].agg(['mean', 'std', 'max'])

        result = {
            'feature': feature_name,
            'layer': layer,
            'feature_id': feature_id,
            'delta_bankruptcy_safe': row['delta_bankruptcy_safe'],
            'delta_bankruptcy_risky': row['delta_bankruptcy_risky'],
            'baseline_bankruptcy_rate': row['baseline_bankruptcy_rate']
        }

        # Add token-specific statistics
        for token_type in ['balance', 'goal', 'probability', 'choice']:
            if token_type in token_stats.index:
                result[f'{token_type}_mean'] = token_stats.loc[token_type, 'mean']
                result[f'{token_type}_max'] = token_stats.loc[token_type, 'max']
            else:
                result[f'{token_type}_mean'] = 0.0
                result[f'{token_type}_max'] = 0.0

        # Identify dominant token position
        means = {
            'balance': result['balance_mean'],
            'goal': result['goal_mean'],
            'probability': result['probability_mean'],
            'choice': result['choice_mean']
        }
        result['dominant_position'] = max(means, key=means.get)
        result['dominant_activation'] = max(means.values())

        results.append(result)

    return pd.DataFrame(results)

def compare_scenarios_by_risk(exp6_data: Dict, feature_layer: int, feature_id: int) -> Dict:
    """
    Compare feature activation patterns across different risk scenarios
    """

    layer_str = f'L{feature_layer}'

    # Categorize scenarios
    categories = {
        'desperate': [],      # balance < 30
        'risky': [],          # 30 <= balance < 60
        'safe': [],           # balance >= 120
        'bankrupt_edge': []   # bankruptcy scenarios
    }

    for result in exp6_data['results']:
        scenario = result['scenario']
        balance = result['balance']

        if layer_str not in result['layers']:
            continue

        features = result['layers'][layer_str]['features']
        key_positions = result['key_positions']

        # Extract mean activation across key positions
        activations = []
        for pos_key in ['balance', 'goal', 'probability']:
            if key_positions[pos_key]:
                pos = key_positions[pos_key][0]
                activations.append(features[pos][feature_id])

        if key_positions['choices']:
            choice_acts = [features[pos][feature_id] for pos in key_positions['choices']]
            activations.append(np.mean(choice_acts))

        mean_activation = np.mean(activations) if activations else 0.0

        # Categorize
        if 'Bankruptcy' in scenario:
            categories['bankrupt_edge'].append((scenario, balance, mean_activation))
        elif balance < 30:
            categories['desperate'].append((scenario, balance, mean_activation))
        elif balance < 60:
            categories['risky'].append((scenario, balance, mean_activation))
        elif balance >= 120:
            categories['safe'].append((scenario, balance, mean_activation))

    return categories

def create_heatmap_harmful_features(harmful_df: pd.DataFrame, output_path: Path):
    """
    Create heatmap showing token position activations for harmful features
    """

    # Prepare data for heatmap
    features = harmful_df['feature'].values
    token_types = ['balance', 'goal', 'probability', 'choice']

    heatmap_data = []
    for token_type in token_types:
        heatmap_data.append(harmful_df[f'{token_type}_mean'].values)

    heatmap_data = np.array(heatmap_data)

    # Create figure
    plt.figure(figsize=(16, 8))
    sns.heatmap(
        heatmap_data,
        xticklabels=features,
        yticklabels=token_types,
        cmap='YlOrRd',
        cbar_kws={'label': 'Mean Activation'},
        annot=False
    )
    plt.title('Token Position Activations for Top Harmful Features (Exp 5)', fontsize=14, pad=20)
    plt.xlabel('Feature ID', fontsize=12)
    plt.ylabel('Token Position', fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap to {output_path}")

def create_scenario_comparison_plot(exp6_data: Dict, top_features: List[Tuple], output_path: Path):
    """
    Create plot comparing feature activations across scenario risk levels
    """

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, (feature_name, layer, feature_id) in enumerate(top_features[:4]):
        categories = compare_scenarios_by_risk(exp6_data, layer, feature_id)

        ax = axes[idx]

        # Plot each category
        for cat_name, scenarios in categories.items():
            if scenarios:
                balances = [s[1] for s in scenarios]
                activations = [s[2] for s in scenarios]
                ax.scatter(balances, activations, label=cat_name, s=100, alpha=0.7)

        ax.set_xlabel('Balance ($)', fontsize=11)
        ax.set_ylabel('Mean Activation', fontsize=11)
        ax.set_title(f'Feature {feature_name} - L{layer}', fontsize=12, weight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Top Harmful Features: Activation vs Scenario Balance', fontsize=14, weight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved scenario comparison to {output_path}")

def main():
    # Paths
    results_dir = Path("/home/ubuntu/llm_addiction/experiment_5_6_analysis/results")
    exp6_path = "/data/llm_addiction/experiment_6_token_level/token_level_tracking_20251013_145433.json"

    print("="*80)
    print("CROSS-ANALYSIS: EXPERIMENT 5 & 6")
    print("="*80)

    # Load data
    print("\nLoading Experiment 5 features...")
    exp5_features = load_exp5_features(results_dir)
    print(f"  Loaded {len(exp5_features['harmful_both'])} features harmful in both conditions")
    print(f"  Loaded {len(exp5_features['protective_safe'])} protective features")

    print("\nLoading Experiment 6 token-level data...")
    exp6_data = load_exp6_data(exp6_path)
    print(f"  Loaded {len(exp6_data['results'])} scenarios")

    # Analyze harmful features
    print("\n" + "="*80)
    print("ANALYZING TOP HARMFUL FEATURES (Exp 5) IN TOKEN SPACE (Exp 6)")
    print("="*80)

    harmful_token_analysis = analyze_harmful_feature_activations(
        exp5_features['harmful_both'],
        exp6_data,
        top_n=20
    )

    # Save results
    harmful_token_analysis.to_csv(results_dir / "cross_analysis_harmful_features.csv", index=False)
    print(f"\nSaved cross-analysis to {results_dir / 'cross_analysis_harmful_features.csv'}")

    # Display summary
    print("\nTop 10 Harmful Features - Token Activation Patterns:")
    print("-" * 80)

    display_cols = [
        'feature', 'layer', 'delta_bankruptcy_safe',
        'dominant_position', 'dominant_activation',
        'balance_mean', 'goal_mean', 'probability_mean', 'choice_mean'
    ]
    print(harmful_token_analysis[display_cols].head(10).to_string(index=False))

    # Token position distribution
    print("\n" + "="*80)
    print("DOMINANT TOKEN POSITION DISTRIBUTION")
    print("="*80)
    position_counts = harmful_token_analysis['dominant_position'].value_counts()
    print(position_counts)
    print(f"\nInterpretation:")
    print(f"  - {position_counts.get('balance', 0)} features most active at BALANCE token")
    print(f"  - {position_counts.get('goal', 0)} features most active at GOAL token")
    print(f"  - {position_counts.get('probability', 0)} features most active at PROBABILITY token")
    print(f"  - {position_counts.get('choice', 0)} features most active at CHOICE token")

    # Create visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    # Heatmap
    create_heatmap_harmful_features(
        harmful_token_analysis.head(20),
        results_dir / "heatmap_harmful_features_tokens.png"
    )

    # Scenario comparison
    top_features = [
        (row['feature'], row['layer'], row['feature_id'])
        for _, row in harmful_token_analysis.head(4).iterrows()
    ]
    create_scenario_comparison_plot(
        exp6_data,
        top_features,
        results_dir / "scenario_comparison_top_harmful.png"
    )

    print("\n" + "="*80)
    print("PROTECTIVE FEATURES ANALYSIS")
    print("="*80)

    if len(exp5_features['protective_safe']) > 0:
        protective_token_analysis = analyze_harmful_feature_activations(
            exp5_features['protective_safe'],
            exp6_data,
            top_n=10
        )

        protective_token_analysis.to_csv(results_dir / "cross_analysis_protective_features.csv", index=False)
        print(f"\nSaved protective features analysis to {results_dir / 'cross_analysis_protective_features.csv'}")

        print("\nTop 10 Protective Features - Token Activation Patterns:")
        print(protective_token_analysis[display_cols].head(10).to_string(index=False))

    print("\n" + "="*80)
    print("CROSS-ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll results saved to: {results_dir}")

if __name__ == "__main__":
    main()
