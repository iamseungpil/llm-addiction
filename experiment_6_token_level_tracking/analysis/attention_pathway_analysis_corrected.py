#!/usr/bin/env python3
"""
Experiment 6: CORRECTED Attention Pathway + SAE Feature Analysis
Analyzes attention flow to the LAST TOKEN (actual decision point)
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import networkx as nx

def load_exp6_data(filepath: str) -> Dict:
    """Load Experiment 6 token-level tracking data"""
    print(f"Loading data from {filepath}...")
    with open(filepath, 'r') as f:
        data = json.load(f)
    print(f"Loaded {data['n_scenarios']} scenarios")
    return data

def analyze_token_pathway_corrected(
    scenario_data: Dict,
    layer: str,
    source_token_type: str,
    top_k_features: int = 20
) -> pd.DataFrame:
    """
    Analyze pathway from source token to LAST TOKEN (actual decision)

    Returns DataFrame with:
    - feature_id
    - source_activation (at information token)
    - target_activation (at last token where decision is made)
    - attention_weight (from information token to last token)
    - pathway_score (activation × attention)
    """
    key_positions = scenario_data['key_positions']
    layer_data = scenario_data['layers'][layer]

    features = np.array(layer_data['features'])  # [seq_len, 32768]
    attention = np.array(layer_data['attention'])  # [32, seq_len, seq_len]

    # Get source position (information token)
    source_pos = key_positions[source_token_type]
    if not source_pos:
        return pd.DataFrame()
    source_pos = source_pos[0]

    # Target is the LAST TOKEN (where decision is generated)
    seq_len = scenario_data['seq_len']
    target_pos = seq_len - 1

    # Get attention weight from source to target (average across heads)
    attention_weight = np.mean(attention[:, target_pos, source_pos])

    # Get feature activations at both positions
    source_features = features[source_pos]
    target_features = features[target_pos]

    # Calculate pathway scores
    results = []
    for feat_id in range(len(source_features)):
        if source_features[feat_id] > 0.01 or target_features[feat_id] > 0.01:
            pathway_score = (source_features[feat_id] + target_features[feat_id]) * attention_weight
            results.append({
                'feature_id': feat_id,
                'source_activation': source_features[feat_id],
                'target_activation': target_features[feat_id],
                'attention_weight': attention_weight,
                'pathway_score': pathway_score,
                'activation_change': target_features[feat_id] - source_features[feat_id]
            })

    df = pd.DataFrame(results)
    if len(df) > 0:
        df = df.sort_values('pathway_score', ascending=False).head(top_k_features)

    return df

def create_attention_heatmap_corrected(
    attention_matrix: np.ndarray,
    key_positions: Dict,
    tokens: List[str],
    scenario_name: str,
    output_path: Path
):
    """Create heatmap focusing on attention TO the last token (decision point)"""

    # Average across heads
    avg_attention = np.mean(attention_matrix, axis=0)

    seq_len = len(tokens)
    last_token_pos = seq_len - 1

    # Focus on attention TO the last token from all other tokens
    attention_to_last = avg_attention[last_token_pos, :]

    plt.figure(figsize=(14, 8))

    # Create bar plot showing attention from each position to last token
    positions = np.arange(seq_len)
    colors = ['gray'] * seq_len

    # Color code key positions
    color_map = {
        'balance': 'blue',
        'goal': 'green',
        'probability': 'purple'
    }

    for token_type, pos_list in key_positions.items():
        if token_type in color_map and pos_list:
            for pos in pos_list:
                colors[pos] = color_map[token_type]

    # Mark last token
    colors[last_token_pos] = 'red'

    plt.bar(positions, attention_to_last, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

    # Add labels for key positions
    for token_type, pos_list in key_positions.items():
        if token_type in color_map and pos_list:
            for pos in pos_list:
                if pos < len(tokens):
                    plt.text(pos, attention_to_last[pos], f"{token_type}\n{tokens[pos]}",
                            ha='center', va='bottom', fontsize=8, rotation=0)

    # Mark last token
    plt.text(last_token_pos, attention_to_last[last_token_pos],
            f"DECISION\n{tokens[last_token_pos]}",
            ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.xlabel('Token Position', fontsize=12)
    plt.ylabel('Attention Weight to Final Token', fontsize=12)
    plt.title(f'Attention Flow to Decision Token: {scenario_name}', fontsize=14, weight='bold')
    plt.grid(axis='y', alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', label='Balance'),
        Patch(facecolor='green', label='Goal'),
        Patch(facecolor='purple', label='Probability'),
        Patch(facecolor='red', label='Decision Token'),
        Patch(facecolor='gray', label='Other Tokens')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved corrected attention heatmap to {output_path}")

def compare_scenario_pathways_corrected(
    data: Dict,
    layer: str,
    risky_scenarios: List[str],
    safe_scenarios: List[str]
) -> pd.DataFrame:
    """
    Compare attention pathways to LAST TOKEN between risky and safe scenarios
    """

    def get_pathway_features(scenario_name):
        scenario_data = next(s for s in data['results'] if s['scenario'] == scenario_name)
        key_pos = scenario_data['key_positions']
        seq_len = scenario_data['seq_len']
        last_token_pos = seq_len - 1

        attention = np.array(scenario_data['layers'][layer]['attention'])
        avg_attn = np.mean(attention, axis=0)

        # Attention from key tokens TO the last token (decision)
        bal_to_decision = 0
        if key_pos['balance']:
            bal_pos = key_pos['balance'][0]
            bal_to_decision = avg_attn[last_token_pos, bal_pos]

        goal_to_decision = 0
        if key_pos['goal']:
            goal_pos = key_pos['goal'][0]
            goal_to_decision = avg_attn[last_token_pos, goal_pos]

        prob_to_decision = 0
        if key_pos['probability']:
            prob_pos = key_pos['probability'][0]
            prob_to_decision = avg_attn[last_token_pos, prob_pos]

        return {
            'scenario': scenario_name,
            'balance': scenario_data['balance'],
            'balance_to_decision': bal_to_decision,
            'goal_to_decision': goal_to_decision,
            'prob_to_decision': prob_to_decision,
            'total_attention': bal_to_decision + goal_to_decision + prob_to_decision
        }

    results = []

    for scenario in risky_scenarios:
        features = get_pathway_features(scenario)
        if features:
            features['category'] = 'risky'
            results.append(features)

    for scenario in safe_scenarios:
        features = get_pathway_features(scenario)
        if features:
            features['category'] = 'safe'
            results.append(features)

    return pd.DataFrame(results)

def main():
    # Paths
    data_path = "/data/llm_addiction/experiment_6_token_level/token_level_tracking_20251013_145433.json"
    output_dir = Path("/home/ubuntu/llm_addiction/experiment_6_token_level_tracking/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("EXPERIMENT 6: CORRECTED ATTENTION PATHWAY ANALYSIS")
    print("Target: LAST TOKEN (actual decision point)")
    print("="*80)

    # Load data
    data = load_exp6_data(data_path)

    # Categorize scenarios
    risky_scenarios = ['Desperate_10', 'Very_risky_25', 'Risky_40', 'Bankruptcy_90_all_in']
    safe_scenarios = ['Safe_130_one_win', 'Safe_140_near_goal', 'Goal_achieved_200']

    # Analyze each layer
    for layer in ['L8', 'L15', 'L31']:
        print(f"\n{'='*80}")
        print(f"ANALYZING LAYER {layer}")
        print(f"{'='*80}")

        # 1. Create corrected attention visualizations
        print(f"\n1. Creating corrected attention visualizations...")
        for scenario in ['Desperate_10', 'Safe_140_near_goal']:
            scenario_data = next(s for s in data['results'] if s['scenario'] == scenario)

            if layer in scenario_data['layers']:
                attention = np.array(scenario_data['layers'][layer]['attention'])
                output_path = output_dir / f"CORRECTED_attention_to_decision_{layer}_{scenario}.png"

                create_attention_heatmap_corrected(
                    attention,
                    scenario_data['key_positions'],
                    scenario_data['tokens'],
                    scenario,
                    output_path
                )

        # 2. Analyze specific pathways (Information → Decision)
        print(f"\n2. Analyzing Information → Decision pathways...")

        for source_type in ['balance', 'goal', 'probability']:
            pathway_results = []

            for scenario_data in data['results']:
                if layer in scenario_data['layers']:
                    df = analyze_token_pathway_corrected(
                        scenario_data,
                        layer,
                        source_type,
                        top_k_features=10
                    )

                    if len(df) > 0:
                        df['scenario'] = scenario_data['scenario']
                        df['balance'] = scenario_data['balance']
                        pathway_results.append(df)

            if pathway_results:
                combined_df = pd.concat(pathway_results, ignore_index=True)
                output_path = output_dir / f"CORRECTED_{source_type}_decision_pathway_{layer}.csv"
                combined_df.to_csv(output_path, index=False)
                print(f"Saved pathway analysis to {output_path.name}")

        # 3. Compare risky vs safe pathways
        print(f"\n3. Comparing risky vs safe scenario pathways...")
        comparison = compare_scenario_pathways_corrected(data, layer, risky_scenarios, safe_scenarios)

        if len(comparison) > 0:
            comparison.to_csv(output_dir / f"CORRECTED_pathway_comparison_{layer}.csv", index=False)

            print(f"\nRisky scenarios (mean attention to decision token):")
            risky_data = comparison[comparison['category'] == 'risky']
            print(f"  Balance → Decision: {risky_data['balance_to_decision'].mean():.6f}")
            print(f"  Goal → Decision: {risky_data['goal_to_decision'].mean():.6f}")
            print(f"  Prob → Decision: {risky_data['prob_to_decision'].mean():.6f}")
            print(f"  Total: {risky_data['total_attention'].mean():.6f}")

            print(f"\nSafe scenarios (mean attention to decision token):")
            safe_data = comparison[comparison['category'] == 'safe']
            print(f"  Balance → Decision: {safe_data['balance_to_decision'].mean():.6f}")
            print(f"  Goal → Decision: {safe_data['goal_to_decision'].mean():.6f}")
            print(f"  Prob → Decision: {safe_data['prob_to_decision'].mean():.6f}")
            print(f"  Total: {safe_data['total_attention'].mean():.6f}")

    print(f"\n{'='*80}")
    print("CORRECTED ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nAll results saved to: {output_dir}")

if __name__ == "__main__":
    main()
