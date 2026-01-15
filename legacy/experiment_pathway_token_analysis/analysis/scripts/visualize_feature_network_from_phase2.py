#!/usr/bin/env python3
"""
Image 3: Feature-Feature Correlation Network (FROM PHASE 2 DATA)

Creates network visualization from actual Phase 2 correlation data.
"""

import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pathlib import Path
from collections import defaultdict
import random

plt.style.use('seaborn-v0_8-white')

def load_phase2_correlations(data_dir, sample_size=5000):
    """Load Phase 2 correlation data from JSONL files"""
    print("Loading Phase 2 correlation data...")

    data_dir = Path(data_dir)
    files = list(data_dir.glob("correlations_*.jsonl"))

    print(f"Total correlation files: {len(files)}")

    # Sample files for manageable size
    sampled_files = random.sample(files, min(sample_size, len(files)))
    print(f"Sampling {len(sampled_files)} files...")

    all_correlations = []

    for i, file in enumerate(sampled_files, 1):
        if i % 500 == 0:
            print(f"  Loaded {i}/{len(sampled_files)} files...")

        with open(file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    all_correlations.append(data)

    print(f"Loaded {len(all_correlations):,} correlations")

    return all_correlations

def load_phase5_classifications():
    """Load Phase 5 risky/safe feature classifications"""
    print("Loading Phase 5 classifications...")

    risky_features = set()
    safe_features = set()

    for gpu in [4, 5, 6, 7]:
        file_path = Path(f"/data/llm_addiction/experiment_pathway_token_analysis/results/phase5_prompt_feature_full/prompt_feature_correlation_gpu{gpu}.json")

        if file_path.exists():
            with open(file_path, 'r') as f:
                data = json.load(f)

            for comp in data['feature_comparisons']:
                if comp['p_value'] < 0.05:
                    if comp['cohens_d'] > 0.5:  # Strong risky
                        risky_features.add(comp['feature'])
                    elif comp['cohens_d'] < -0.5:  # Strong safe
                        safe_features.add(comp['feature'])

    print(f"Risky features: {len(risky_features)}")
    print(f"Safe features: {len(safe_features)}")

    return risky_features, safe_features

def create_network_graph(correlations, risky_features, safe_features, top_n=500):
    """Create network graph from top correlations"""
    print(f"\nCreating network with top {top_n} correlations...")

    # Filter to strong correlations
    strong_corrs = [c for c in correlations if abs(c.get('pearson_r', 0)) > 0.9]

    print(f"Strong correlations (|r| > 0.9): {len(strong_corrs)}")

    # Sort by correlation strength
    sorted_corrs = sorted(strong_corrs, key=lambda x: abs(x.get('pearson_r', 0)), reverse=True)

    # Take top N
    top_corrs = sorted_corrs[:top_n]

    # Create graph
    G = nx.Graph()

    for corr in top_corrs:
        f1 = corr.get('feature_A') or corr.get('feature_1') or corr.get('feature1')
        f2 = corr.get('feature_B') or corr.get('feature_2') or corr.get('feature2')
        weight = abs(corr.get('pearson_r', 0))

        if f1 and f2:
            G.add_edge(f1, f2, weight=weight)

    print(f"Network nodes: {G.number_of_nodes()}")
    print(f"Network edges: {G.number_of_edges()}")

    # Assign node colors
    node_colors = []
    for node in G.nodes():
        if node in risky_features:
            node_colors.append('#e57373')  # Red for risky
        elif node in safe_features:
            node_colors.append('#64b5f6')  # Blue for safe
        else:
            node_colors.append('#bdbdbd')  # Gray for neutral

    return G, node_colors

def create_visualization(G, node_colors, correlations):
    """Create comprehensive network visualization"""

    fig = plt.figure(figsize=(20, 10))

    if G.number_of_nodes() == 0:
        # No network to show
        ax = fig.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, 'No strong correlations found (|r| > 0.9)\nwith current sampling',
               ha='center', va='center', fontsize=16)
        ax.axis('off')
        return fig

    # Subplot 1: Network graph
    ax1 = plt.subplot(1, 2, 1)

    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100,
                          alpha=0.7, ax=ax1)

    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(G, pos, width=[w*2 for w in weights],
                          alpha=0.3, edge_color='gray', ax=ax1)

    ax1.set_title('Feature-Feature Correlation Network\n(|r| > 0.9)',
                 fontsize=14, fontweight='bold')
    ax1.axis('off')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e57373', label='Risky Features', alpha=0.7),
        Patch(facecolor='#64b5f6', label='Safe Features', alpha=0.7),
        Patch(facecolor='#bdbdbd', label='Neutral Features', alpha=0.7)
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=11)

    # Subplot 2: Statistics
    ax2 = plt.subplot(1, 2, 2)
    ax2.axis('off')

    # Correlation distribution stats
    import pandas as pd
    df = pd.DataFrame(correlations)

    stats_text = []
    stats_text.append("PHASE 2: FEATURE-FEATURE CORRELATION STATISTICS\n")
    stats_text.append(f"Total correlations analyzed: {len(correlations):,}\n")

    if 'pearson_r' in df.columns:
        stats_text.append(f"\nPearson Correlation:")
        stats_text.append(f"  Mean: {df['pearson_r'].mean():+.4f}")
        stats_text.append(f"  Std:  {df['pearson_r'].std():.4f}")
        stats_text.append(f"  Min:  {df['pearson_r'].min():+.4f}")
        stats_text.append(f"  Max:  {df['pearson_r'].max():+.4f}")

    strong_pos = len(df[df['pearson_r'] > 0.7]) if 'pearson_r' in df.columns else 0
    strong_neg = len(df[df['pearson_r'] < -0.7]) if 'pearson_r' in df.columns else 0

    stats_text.append(f"\nStrong Correlations:")
    stats_text.append(f"  Positive (r > 0.7): {strong_pos:,} ({100*strong_pos/len(df):.1f}%)")
    stats_text.append(f"  Negative (r < -0.7): {strong_neg:,} ({100*strong_neg/len(df):.1f}%)")

    stats_text.append(f"\nNetwork Statistics:")
    stats_text.append(f"  Nodes: {G.number_of_nodes()}")
    stats_text.append(f"  Edges: {G.number_of_edges()}")

    if G.number_of_nodes() > 0:
        stats_text.append(f"  Avg degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
        stats_text.append(f"  Density: {nx.density(G):.4f}")

    # Top hub features
    if G.number_of_nodes() > 0:
        degree_centrality = nx.degree_centrality(G)
        top_hubs = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]

        stats_text.append(f"\nTop 10 Hub Features:")
        for i, (feature, centrality) in enumerate(top_hubs, 1):
            connections = G.degree(feature)
            stats_text.append(f"  {i:2d}. {feature}: {connections} connections")

    ax2.text(0.05, 0.95, '\n'.join(stats_text),
            transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    fig.suptitle('Phase 2: Feature-Feature Correlation Network Analysis',
                fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig

def main():
    print("="*80)
    print("Image 3: Feature-Feature Correlation Network (FROM PHASE 2 DATA)")
    print("="*80)
    print()

    # Load data
    data_dir = "/data/llm_addiction/experiment_pathway_token_analysis/results/phase2_correlations"
    correlations = load_phase2_correlations(data_dir, sample_size=1000)

    risky_features, safe_features = load_phase5_classifications()

    # Create network
    G, node_colors = create_network_graph(correlations, risky_features, safe_features, top_n=500)

    # Create visualization
    print("\nCreating network visualization...")
    fig = create_visualization(G, node_colors, correlations)

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
