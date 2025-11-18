#!/usr/bin/env python3
"""
Image 3: Feature-Feature Correlation Network

Visualizes the network of correlations between features,
showing clusters and hub features.
"""

import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pathlib import Path
from collections import defaultdict

plt.style.use('seaborn-v0_8-white')

def load_phase2_data():
    """Load Phase 2 feature-feature correlation data"""
    print("Loading Phase 2 data...")

    all_correlations = []

    for gpu in [4, 5, 6, 7]:
        file_path = Path(f"/data/llm_addiction/experiment_pathway_token_analysis/results/phase2_feature_feature/feature_feature_correlation_gpu{gpu}.json")

        if not file_path.exists():
            print(f"⚠️  GPU {gpu} file not found, skipping...")
            continue

        with open(file_path, 'r') as f:
            data = json.load(f)

        all_correlations.extend(data['feature_correlations'])

    print(f"Loaded {len(all_correlations):,} feature-feature correlations")
    return all_correlations

def load_phase5_classifications():
    """Load Phase 5 risky/safe feature classifications"""
    print("Loading Phase 5 classifications...")

    risky_features = set()
    safe_features = set()

    for gpu in [4, 5, 6, 7]:
        file_path = Path(f"/data/llm_addiction/experiment_pathway_token_analysis/results/phase5_prompt_feature_full/prompt_feature_correlation_gpu{gpu}.json")

        with open(file_path, 'r') as f:
            data = json.load(f)

        for comp in data['feature_comparisons']:
            if comp['p_value'] < 0.05:
                if comp['cohens_d'] > 0.5:  # Strong risky threshold
                    risky_features.add(comp['feature'])
                elif comp['cohens_d'] < -0.5:  # Strong safe threshold
                    safe_features.add(comp['feature'])

    print(f"Strong risky features: {len(risky_features):,}")
    print(f"Strong safe features: {len(safe_features):,}")

    return risky_features, safe_features

def create_network_graph(correlations, risky_features, safe_features, top_n=500):
    """Create network graph from top correlations"""
    print(f"\nCreating network graph with top {top_n} correlations...")

    # Filter to top correlations
    sorted_corrs = sorted(correlations, key=lambda x: abs(x['correlation']), reverse=True)
    top_corrs = sorted_corrs[:top_n]

    # Create graph
    G = nx.Graph()

    for corr in top_corrs:
        f1 = corr['feature_1']
        f2 = corr['feature_2']
        weight = abs(corr['correlation'])

        G.add_edge(f1, f2, weight=weight)

    print(f"Network nodes: {G.number_of_nodes()}")
    print(f"Network edges: {G.number_of_edges()}")

    # Assign node colors based on risky/safe classification
    node_colors = []
    for node in G.nodes():
        if node in risky_features:
            node_colors.append('#e57373')  # Red for risky
        elif node in safe_features:
            node_colors.append('#64b5f6')  # Blue for safe
        else:
            node_colors.append('#bdbdbd')  # Gray for neutral

    return G, node_colors

def create_visualization(G, node_colors):
    """Create comprehensive network visualization"""

    # Create figure with 2 subplots
    fig = plt.figure(figsize=(20, 10))

    # Subplot 1: Full network with force-directed layout
    ax1 = plt.subplot(1, 2, 1)

    # Use spring layout for force-directed positioning
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

    # Draw network
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100,
                          alpha=0.7, ax=ax1)

    # Draw edges with varying thickness based on correlation strength
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(G, pos, width=[w*2 for w in weights],
                          alpha=0.3, edge_color='gray', ax=ax1)

    ax1.set_title('Feature-Feature Correlation Network\n(Top 500 Strongest Correlations)',
                 fontsize=14, fontweight='bold')
    ax1.axis('off')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e57373', label='Risky Features', alpha=0.7),
        Patch(facecolor='#64b5f6', label='Safe Features', alpha=0.7),
        Patch(facecolor='#bdbdbd', label='Neutral Features', alpha=0.7)
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=11)

    # Subplot 2: Hub features analysis
    ax2 = plt.subplot(1, 2, 2)
    ax2.axis('off')

    # Calculate degree centrality
    degree_centrality = nx.degree_centrality(G)
    top_hubs = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:20]

    # Create table
    table_data = []
    table_data.append(['Rank', 'Feature', 'Connections', 'Type'])

    for i, (feature, centrality) in enumerate(top_hubs, 1):
        connections = G.degree(feature)

        if feature in risky_features:
            ftype = 'Risky'
        elif feature in safe_features:
            ftype = 'Safe'
        else:
            ftype = 'Neutral'

        table_data.append([
            str(i),
            feature[:15],
            str(connections),
            ftype
        ])

    table = ax2.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.1, 0.4, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#e0e0e0')
        table[(0, i)].set_text_props(weight='bold')

    # Color code type column
    for i in range(1, min(21, len(table_data))):
        ftype = table_data[i][3]
        if ftype == 'Risky':
            table[(i, 3)].set_facecolor('#ffcdd2')
        elif ftype == 'Safe':
            table[(i, 3)].set_facecolor('#bbdefb')

    ax2.set_title('Top 20 Hub Features\n(Most Connected)', fontsize=14, fontweight='bold', pad=20)

    # Overall title
    fig.suptitle('Phase 2: Feature-Feature Correlation Network Analysis',
                fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig

def main():
    print("="*80)
    print("Image 3: Feature-Feature Correlation Network")
    print("="*80)
    print()

    # Load data
    correlations = load_phase2_data()
    risky_features, safe_features = load_phase5_classifications()

    # Create network
    G, node_colors = create_network_graph(correlations, risky_features, safe_features, top_n=500)

    # Network statistics
    print("\nNetwork Statistics:")
    print(f"  Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
    print(f"  Density: {nx.density(G):.4f}")
    print(f"  Number of connected components: {nx.number_connected_components(G)}")

    # Create visualization
    print("\nCreating network visualization...")
    fig = create_visualization(G, node_colors)

    # Save
    output_dir = Path("/home/ubuntu/llm_addiction/experiment_pathway_token_analysis/analysis/images")

    png_path = output_dir / "03_feature_correlation_network.png"
    pdf_path = output_dir / "03_feature_correlation_network.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')

    print(f"\n✅ Saved visualization:")
    print(f"   PNG: {png_path}")
    print(f"   PDF: {pdf_path}")
    print()

if __name__ == '__main__':
    main()
