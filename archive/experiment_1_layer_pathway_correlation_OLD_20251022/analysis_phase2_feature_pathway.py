#!/usr/bin/env python3
"""
Phase 2 Analysis: Feature Pathway Tracing
Trace how features propagate from early layers to late layers
"""

import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from collections import defaultdict

def load_pathway_data():
    """Load pathway data"""
    print("Loading pathway data...")
    with open('/data/llm_addiction/experiment_1_pathway_L1_31/final_pathway_L1_31_20251001_165207.json', 'r') as f:
        data = json.load(f)
    return data['results']

def load_top_features():
    """Load top features from Phase 1 analysis"""
    print("Loading top features...")
    with open('/home/ubuntu/llm_addiction/experiment_1_layer_pathway_L1_31/feature_importance_results.json', 'r') as f:
        return json.load(f)

def compute_cross_layer_correlations(results, top_features):
    """
    Compute correlations between features in different layers
    Question: Does L8-2059 activation correlate with L31-10692 activation?
    """
    print("\nComputing cross-layer feature correlations...")

    # Focus on critical layers
    critical_layers = ['L8', 'L9', 'L10', 'L11', 'L31']

    # Collect activations for top features
    feature_activations = defaultdict(list)

    for game in results:
        if not game['round_data']:
            continue

        last_round = game['round_data'][-1]

        for layer_name in critical_layers:
            if layer_name not in last_round['features']:
                continue

            features = np.array(last_round['features'][layer_name])

            # Get top 5 features for this layer
            layer_top = top_features[layer_name][:5]

            for feat_info in layer_top:
                feat_id = feat_info['feature_id']
                key = f"{layer_name}-{feat_id}"
                feature_activations[key].append(features[feat_id])

    # Compute pairwise correlations
    correlations = []

    # L8 -> L31 pathways
    print("\n🔗 L8 → L31 Feature Pathways:")
    print("-"*100)

    for l8_feat in top_features['L8'][:5]:
        l8_key = f"L8-{l8_feat['feature_id']}"
        l8_activations = np.array(feature_activations[l8_key])

        for l31_feat in top_features['L31'][:5]:
            l31_key = f"L31-{l31_feat['feature_id']}"
            l31_activations = np.array(feature_activations[l31_key])

            # Correlation
            if len(l8_activations) > 0 and len(l31_activations) > 0:
                corr, p_value = stats.pearsonr(l8_activations, l31_activations)

                correlations.append({
                    'from_feature': l8_key,
                    'to_feature': l31_key,
                    'correlation': corr,
                    'p_value': p_value,
                    'from_direction': l8_feat['direction'],
                    'to_direction': l31_feat['direction']
                })

                if abs(corr) > 0.3 and p_value < 0.05:
                    print(f"  {l8_key} ({l8_feat['direction']}) → {l31_key} ({l31_feat['direction']}): "
                          f"r={corr:.3f}, p={p_value:.6f} {'✅' if p_value < 0.01 else '⚠️'}")

    # L8 -> L15 -> L31 pathways (check intermediate)
    print("\n🔗 L8 → L10 → L31 Three-Layer Pathways:")
    print("-"*100)

    for l8_feat in top_features['L8'][:3]:
        l8_key = f"L8-{l8_feat['feature_id']}"
        l8_activations = np.array(feature_activations[l8_key])

        for l10_feat in top_features['L10'][:3]:
            l10_key = f"L10-{l10_feat['feature_id']}"
            l10_activations = np.array(feature_activations[l10_key])

            corr_8_10, p_8_10 = stats.pearsonr(l8_activations, l10_activations)

            if abs(corr_8_10) > 0.3 and p_8_10 < 0.05:
                # Check L10 -> L31
                for l31_feat in top_features['L31'][:3]:
                    l31_key = f"L31-{l31_feat['feature_id']}"
                    l31_activations = np.array(feature_activations[l31_key])

                    corr_10_31, p_10_31 = stats.pearsonr(l10_activations, l31_activations)

                    if abs(corr_10_31) > 0.3 and p_10_31 < 0.05:
                        corr_8_31, p_8_31 = stats.pearsonr(l8_activations, l31_activations)

                        print(f"  {l8_key} → {l10_key} → {l31_key}")
                        print(f"    L8→L10: r={corr_8_10:.3f}, p={p_8_10:.6f}")
                        print(f"    L10→L31: r={corr_10_31:.3f}, p={p_10_31:.6f}")
                        print(f"    L8→L31: r={corr_8_31:.3f}, p={p_8_31:.6f}")
                        print()

    return correlations, feature_activations

def analyze_feature_progression(results, top_features):
    """
    Analyze how a decision's representation changes across layers
    Question: Do the same features activate across layers for the same decision?
    """
    print("\n📈 Feature Progression Analysis:")
    print("-"*100)

    # Group by outcome
    bankruptcy_games = [g for g in results if g['outcome'] == 'bankruptcy']
    safe_games = [g for g in results if g['outcome'] == 'voluntary_stop']

    # For each top feature, track activation across layers
    for layer_name in ['L8', 'L10', 'L31']:
        print(f"\n{layer_name} Top Feature Activation Patterns:")

        top_feat = top_features[layer_name][0]
        feat_id = top_feat['feature_id']

        b_activations = []
        s_activations = []

        for game in bankruptcy_games:
            if game['round_data']:
                features = np.array(game['round_data'][-1]['features'][layer_name])
                b_activations.append(features[feat_id])

        for game in safe_games:
            if game['round_data']:
                features = np.array(game['round_data'][-1]['features'][layer_name])
                s_activations.append(features[feat_id])

        b_mean = np.mean(b_activations)
        s_mean = np.mean(s_activations)

        print(f"  {layer_name}-{feat_id} ({top_feat['direction']}):")
        print(f"    Bankruptcy mean: {b_mean:.6f}")
        print(f"    Safe mean: {s_mean:.6f}")
        print(f"    Difference: {b_mean - s_mean:.6f}")
        print(f"    Cohen's d: {top_feat['cohen_d']:.3f}")

def visualize_feature_network(correlations):
    """
    Visualize feature network as a graph
    """
    print("\nCreating feature network visualization...")

    # Filter strong correlations
    strong_corrs = [c for c in correlations if abs(c['correlation']) > 0.4 and c['p_value'] < 0.01]

    if len(strong_corrs) == 0:
        print("⚠️ No strong correlations found (r > 0.4, p < 0.01)")
        return

    fig, ax = plt.subplots(figsize=(14, 10))

    # Node positions
    layer_x = {'L8': 0, 'L9': 1, 'L10': 2, 'L11': 3, 'L31': 6}
    node_positions = {}
    layer_counts = defaultdict(int)

    # Collect all nodes
    nodes = set()
    for corr in strong_corrs:
        nodes.add(corr['from_feature'])
        nodes.add(corr['to_feature'])

    # Position nodes
    for node in nodes:
        layer = node.split('-')[0]
        x = layer_x.get(layer, 0)
        y = layer_counts[layer]
        node_positions[node] = (x, y)
        layer_counts[layer] += 1

    # Draw edges
    for corr in strong_corrs:
        from_pos = node_positions[corr['from_feature']]
        to_pos = node_positions[corr['to_feature']]

        color = 'red' if corr['correlation'] > 0 else 'blue'
        alpha = min(abs(corr['correlation']), 1.0)
        width = abs(corr['correlation']) * 3

        ax.plot([from_pos[0], to_pos[0]], [from_pos[1], to_pos[1]],
               color=color, alpha=alpha, linewidth=width, zorder=1)

    # Draw nodes
    for node, (x, y) in node_positions.items():
        direction = 'risky' if 'risky' in [c['from_direction'] for c in strong_corrs if c['from_feature'] == node] else 'safe'
        color = '#C73E1D' if direction == 'risky' else '#2E86AB'

        ax.scatter(x, y, s=500, c=color, alpha=0.7, edgecolors='black', linewidth=2, zorder=2)
        ax.text(x, y, node.split('-')[1], ha='center', va='center', fontsize=8, fontweight='bold')

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Feature Index', fontsize=12)
    ax.set_title('Feature Network: Cross-Layer Correlations (r > 0.4, p < 0.01)',
                fontsize=14, fontweight='bold')
    ax.set_xticks(list(layer_x.values()))
    ax.set_xticklabels(list(layer_x.keys()))
    ax.grid(True, alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#C73E1D', label='Risky features'),
        Patch(facecolor='#2E86AB', label='Safe features'),
        Patch(facecolor='red', alpha=0.5, label='Positive correlation'),
        Patch(facecolor='blue', alpha=0.5, label='Negative correlation')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    output_path = '/home/ubuntu/llm_addiction/experiment_1_layer_pathway_L1_31/feature_network.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def create_correlation_heatmap(correlations):
    """Create heatmap of L8 -> L31 correlations"""

    print("\nCreating correlation heatmap...")

    # Extract L8 -> L31 correlations
    l8_l31 = [c for c in correlations if c['from_feature'].startswith('L8-') and
              c['to_feature'].startswith('L31-')]

    if len(l8_l31) == 0:
        print("⚠️ No L8 -> L31 correlations found")
        return

    # Create matrix
    l8_features = sorted(set([c['from_feature'] for c in l8_l31]))
    l31_features = sorted(set([c['to_feature'] for c in l8_l31]))

    matrix = np.zeros((len(l8_features), len(l31_features)))

    for i, l8_feat in enumerate(l8_features):
        for j, l31_feat in enumerate(l31_features):
            corr_list = [c['correlation'] for c in l8_l31
                        if c['from_feature'] == l8_feat and c['to_feature'] == l31_feat]
            if corr_list:
                matrix[i, j] = corr_list[0]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    ax.set_xticks(range(len(l31_features)))
    ax.set_yticks(range(len(l8_features)))
    ax.set_xticklabels([f.split('-')[1] for f in l31_features], rotation=45)
    ax.set_yticklabels([f.split('-')[1] for f in l8_features])

    ax.set_xlabel('L31 Features', fontsize=12)
    ax.set_ylabel('L8 Features', fontsize=12)
    ax.set_title('L8 → L31 Feature Correlations', fontsize=14, fontweight='bold')

    # Add correlation values
    for i in range(len(l8_features)):
        for j in range(len(l31_features)):
            text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)

    plt.colorbar(im, ax=ax, label='Correlation')
    plt.tight_layout()

    output_path = '/home/ubuntu/llm_addiction/experiment_1_layer_pathway_L1_31/correlation_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

if __name__ == '__main__':
    print("="*100)
    print("PHASE 2: FEATURE PATHWAY TRACING")
    print("Question: How do features propagate from early layers to output?")
    print("="*100)

    results = load_pathway_data()
    print(f"Loaded {len(results)} games")

    top_features = load_top_features()

    # Compute correlations
    correlations, feature_activations = compute_cross_layer_correlations(results, top_features)

    # Analyze progression
    analyze_feature_progression(results, top_features)

    # Visualizations
    visualize_feature_network(correlations)
    create_correlation_heatmap(correlations)

    # Save results
    output_file = '/home/ubuntu/llm_addiction/experiment_1_layer_pathway_L1_31/feature_pathway_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'correlations': correlations,
            'n_correlations': len(correlations),
            'strong_correlations': [c for c in correlations if abs(c['correlation']) > 0.4 and c['p_value'] < 0.01]
        }, f, indent=2)

    print(f"\nSaved results: {output_file}")
    print("\n✅ Phase 2 Feature Pathway Analysis Complete!")
