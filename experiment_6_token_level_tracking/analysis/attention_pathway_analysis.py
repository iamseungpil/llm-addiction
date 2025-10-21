#!/usr/bin/env python3
"""
Experiment 6: Attention Pathway + SAE Feature Analysis
Analyzes attention flow patterns combined with SAE feature activations
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

def extract_attention_flow(attention_matrix: np.ndarray, threshold: float = 0.1) -> Dict:
    """
    Extract significant attention flows from attention matrix

    Args:
        attention_matrix: [n_heads, seq_len, seq_len]
        threshold: Minimum attention weight to consider

    Returns:
        Dict with attention flow statistics
    """
    n_heads, seq_len, _ = attention_matrix.shape

    # Average across heads
    avg_attention = np.mean(attention_matrix, axis=0)  # [seq_len, seq_len]

    # Find strong connections
    strong_connections = np.where(avg_attention > threshold)

    return {
        'avg_attention': avg_attention,
        'strong_connections': list(zip(strong_connections[0], strong_connections[1])),
        'max_attention_per_token': np.max(avg_attention, axis=1),
        'attention_entropy': -np.sum(avg_attention * np.log(avg_attention + 1e-10), axis=1)
    }

def analyze_token_pathway(
    scenario_data: Dict,
    layer: str,
    source_token_type: str,
    target_token_type: str,
    top_k_features: int = 20
) -> pd.DataFrame:
    """
    Analyze pathway from source token to target token

    Returns DataFrame with:
    - feature_id
    - source_activation
    - target_activation
    - attention_weight
    - pathway_score (activation × attention)
    """
    key_positions = scenario_data['key_positions']
    layer_data = scenario_data['layers'][layer]

    features = np.array(layer_data['features'])  # [seq_len, 32768]
    attention = np.array(layer_data['attention'])  # [32, seq_len, seq_len]

    # Get positions
    source_pos = key_positions[source_token_type]
    target_pos = key_positions[target_token_type]

    if not source_pos or not target_pos:
        return pd.DataFrame()

    source_pos = source_pos[0]
    target_pos = target_pos[0]

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

def create_attention_heatmap(
    attention_matrix: np.ndarray,
    key_positions: Dict,
    scenario_name: str,
    output_path: Path
):
    """Create heatmap of attention weights with key positions marked"""

    # Average across heads
    avg_attention = np.mean(attention_matrix, axis=0)

    plt.figure(figsize=(12, 10))

    # Create heatmap
    sns.heatmap(
        avg_attention,
        cmap='YlOrRd',
        cbar_kws={'label': 'Attention Weight'},
        vmin=0,
        vmax=0.2,
        square=True
    )

    # Mark key positions
    colors = {
        'balance': 'blue',
        'goal': 'green',
        'probability': 'purple',
        'choices': 'red'
    }

    for token_type, positions in key_positions.items():
        if positions and token_type in colors:
            for pos in positions:
                # Mark on y-axis (query tokens)
                plt.axhline(y=pos, color=colors[token_type], alpha=0.3, linewidth=2)
                # Mark on x-axis (key tokens)
                plt.axvline(x=pos, color=colors[token_type], alpha=0.3, linewidth=2)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=token_type)
                      for token_type, color in colors.items()]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.title(f'Attention Flow: {scenario_name}', fontsize=14, weight='bold')
    plt.xlabel('Key Tokens (Source)', fontsize=12)
    plt.ylabel('Query Tokens (Target)', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved attention heatmap to {output_path}")

def create_pathway_network(
    scenario_data: Dict,
    layer: str,
    output_path: Path,
    min_attention: float = 0.05
):
    """Create network graph of token pathways"""

    key_positions = scenario_data['key_positions']
    attention = np.array(scenario_data['layers'][layer]['attention'])
    avg_attention = np.mean(attention, axis=0)

    # Create network
    G = nx.DiGraph()

    # Add nodes for each key token type
    node_positions = {}
    for token_type, positions in key_positions.items():
        if positions:
            for i, pos in enumerate(positions):
                node_label = f"{token_type}_{i}" if len(positions) > 1 else token_type
                node_positions[pos] = node_label
                G.add_node(node_label, token_type=token_type)

    # Add edges based on attention
    for target_pos, target_label in node_positions.items():
        for source_pos, source_label in node_positions.items():
            weight = avg_attention[target_pos, source_pos]
            if weight > min_attention and target_pos != source_pos:
                G.add_edge(source_label, target_label, weight=weight)

    # Create visualization
    plt.figure(figsize=(14, 10))

    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50)

    # Node colors by type
    color_map = {
        'balance': '#3498db',
        'goal': '#2ecc71',
        'probability': '#9b59b6',
        'choices': '#e74c3c'
    }
    node_colors = [color_map.get(G.nodes[node]['token_type'], 'gray') for node in G.nodes()]

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=3000,
        alpha=0.9
    )

    # Draw edges with varying width based on attention weight
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    max_weight = max(weights) if weights else 1

    nx.draw_networkx_edges(
        G, pos,
        width=[w * 10 / max_weight for w in weights],
        alpha=0.6,
        edge_color='gray',
        arrows=True,
        arrowsize=20,
        connectionstyle='arc3,rad=0.1'
    )

    # Draw labels
    nx.draw_networkx_labels(
        G, pos,
        font_size=10,
        font_weight='bold',
        font_color='white'
    )

    # Add edge labels (attention weights)
    edge_labels = {(u, v): f"{G[u][v]['weight']:.3f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels,
        font_size=8
    )

    plt.title(f"Token Attention Network: {scenario_data['scenario']} ({layer})",
             fontsize=14, weight='bold')
    plt.axis('off')
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved pathway network to {output_path}")

def compare_scenario_pathways(
    data: Dict,
    layer: str,
    risky_scenarios: List[str],
    safe_scenarios: List[str]
) -> pd.DataFrame:
    """
    Compare attention pathways between risky and safe scenarios
    """

    def get_pathway_features(scenario_name):
        scenario_data = next(s for s in data['results'] if s['scenario'] == scenario_name)
        key_pos = scenario_data['key_positions']

        if not key_pos['balance'] or not key_pos['choices']:
            return None

        bal_pos = key_pos['balance'][0]
        choice_pos = key_pos['choices'][0]

        attention = np.array(scenario_data['layers'][layer]['attention'])
        avg_attn = np.mean(attention, axis=0)

        # Attention from balance to choice
        bal_to_choice = avg_attn[choice_pos, bal_pos]

        # Attention from goal to choice
        goal_to_choice = 0
        if key_pos['goal']:
            goal_pos = key_pos['goal'][0]
            goal_to_choice = avg_attn[choice_pos, goal_pos]

        # Attention from prob to choice
        prob_to_choice = 0
        if key_pos['probability']:
            prob_pos = key_pos['probability'][0]
            prob_to_choice = avg_attn[choice_pos, prob_pos]

        return {
            'scenario': scenario_name,
            'balance': scenario_data['balance'],
            'balance_to_choice': bal_to_choice,
            'goal_to_choice': goal_to_choice,
            'prob_to_choice': prob_to_choice,
            'total_attention': bal_to_choice + goal_to_choice + prob_to_choice
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
    print("EXPERIMENT 6: ATTENTION PATHWAY + SAE FEATURE ANALYSIS")
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

        # 1. Create attention heatmaps for representative scenarios
        print(f"\n1. Creating attention heatmaps...")
        for scenario in ['Desperate_10', 'Safe_140_near_goal']:
            scenario_data = next(s for s in data['results'] if s['scenario'] == scenario)

            if layer in scenario_data['layers']:
                attention = np.array(scenario_data['layers'][layer]['attention'])
                output_path = output_dir / f"attention_heatmap_{layer}_{scenario}.png"

                create_attention_heatmap(
                    attention,
                    scenario_data['key_positions'],
                    scenario,
                    output_path
                )

        # 2. Create pathway networks
        print(f"\n2. Creating pathway networks...")
        for scenario in ['Desperate_10', 'Safe_140_near_goal']:
            scenario_data = next(s for s in data['results'] if s['scenario'] == scenario)

            if layer in scenario_data['layers']:
                output_path = output_dir / f"pathway_network_{layer}_{scenario}.png"
                create_pathway_network(scenario_data, layer, output_path)

        # 3. Analyze specific pathways (Balance → Choice)
        print(f"\n3. Analyzing Balance → Choice pathway...")
        pathway_results = []

        for scenario_data in data['results']:
            if layer in scenario_data['layers']:
                df = analyze_token_pathway(
                    scenario_data,
                    layer,
                    'balance',
                    'choices',
                    top_k_features=10
                )

                if len(df) > 0:
                    df['scenario'] = scenario_data['scenario']
                    df['balance'] = scenario_data['balance']
                    pathway_results.append(df)

        if pathway_results:
            combined_df = pd.concat(pathway_results, ignore_index=True)
            combined_df.to_csv(output_dir / f"balance_choice_pathway_{layer}.csv", index=False)
            print(f"Saved pathway analysis to balance_choice_pathway_{layer}.csv")

        # 4. Compare risky vs safe pathways
        print(f"\n4. Comparing risky vs safe scenario pathways...")
        comparison = compare_scenario_pathways(data, layer, risky_scenarios, safe_scenarios)

        if len(comparison) > 0:
            comparison.to_csv(output_dir / f"pathway_comparison_{layer}.csv", index=False)

            print(f"\nRisky scenarios (mean attention to choice):")
            risky_data = comparison[comparison['category'] == 'risky']
            print(f"  Balance → Choice: {risky_data['balance_to_choice'].mean():.4f}")
            print(f"  Goal → Choice: {risky_data['goal_to_choice'].mean():.4f}")
            print(f"  Prob → Choice: {risky_data['prob_to_choice'].mean():.4f}")

            print(f"\nSafe scenarios (mean attention to choice):")
            safe_data = comparison[comparison['category'] == 'safe']
            print(f"  Balance → Choice: {safe_data['balance_to_choice'].mean():.4f}")
            print(f"  Goal → Choice: {safe_data['goal_to_choice'].mean():.4f}")
            print(f"  Prob → Choice: {safe_data['prob_to_choice'].mean():.4f}")

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nAll results saved to: {output_dir}")

if __name__ == "__main__":
    main()
