#!/usr/bin/env python3
"""
Analyze token-level attribution from Experiment 6 data
Answer: Which input tokens influence the output decision?
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def load_token_level_data(filepath):
    """Load token-level tracking data"""
    with open(filepath, 'r') as f:
        return json.load(f)

def analyze_token_attribution(data):
    """
    Analyze which tokens contribute to output
    Method: Attention-weighted feature importance
    """

    print("="*100)
    print("TOKEN-LEVEL ATTRIBUTION ANALYSIS")
    print("="*100)

    for game_idx, game in enumerate(data['results']):
        print(f"\n{'='*100}")
        print(f"Game {game_idx + 1}")
        print('='*100)

        tokens = game['tokens']
        seq_len = game['seq_len']

        print(f"\nTokens ({seq_len}): {tokens[:10]} ... {tokens[-5:]}")

        # For each critical layer
        for layer_name in ['L8', 'L15', 'L31']:
            print(f"\n{layer_name} Token Attribution:")
            print("-"*100)

            layer_data = game['layers'][layer_name]
            features = np.array(layer_data['features'])  # (seq_len, 32768)
            attention = np.array(layer_data['attention'])  # (n_heads, seq_len, seq_len)

            # Average attention across heads
            avg_attention = attention.mean(axis=0)  # (seq_len, seq_len)

            # Attention TO last token (output position)
            attn_to_output = avg_attention[:, -1]  # (seq_len,)

            # Feature activation at each position
            feature_magnitudes = np.linalg.norm(features, axis=1)  # (seq_len,)

            # Token importance = attention × feature magnitude
            token_importance = attn_to_output * feature_magnitudes

            # Normalize
            token_importance = token_importance / (token_importance.sum() + 1e-10)

            # Top 10 important tokens
            top_indices = np.argsort(token_importance)[::-1][:10]

            print(f"Top 10 tokens contributing to output:")
            print(f"{'Rank':<6} {'Position':<10} {'Token':<20} {'Importance':<12} {'Attention':<12} {'||Features||':<15}")
            print("-"*100)

            for rank, idx in enumerate(top_indices, 1):
                token = tokens[idx] if idx < len(tokens) else 'N/A'
                print(f"{rank:<6} {idx:<10} {token:<20} {token_importance[idx]:<12.6f} "
                      f"{attn_to_output[idx]:<12.6f} {feature_magnitudes[idx]:<15.6f}")

            # Analyze balance tokens specifically
            balance_positions = [i for i, t in enumerate(tokens) if '$' in t and any(c.isdigit() for c in t)]

            if balance_positions:
                print(f"\n💰 Balance Token Analysis:")
                for pos in balance_positions:
                    print(f"  Position {pos}: '{tokens[pos]}'")
                    print(f"    Importance: {token_importance[pos]:.6f}")
                    print(f"    Attention to output: {attn_to_output[pos]:.6f}")
                    print(f"    Feature magnitude: {feature_magnitudes[pos]:.6f}")

                    # Top activated features at this position
                    pos_features = features[pos]
                    active_indices = np.where(pos_features > 0.1)[0]
                    if len(active_indices) > 0:
                        top_active = active_indices[np.argsort(pos_features[active_indices])[::-1]][:5]
                        print(f"    Top 5 active features: {top_active.tolist()}")
                        print(f"    Feature values: {pos_features[top_active].tolist()}")

def visualize_attention_flow(data, game_idx=0):
    """Visualize attention flow from input to output"""

    print(f"\n{'='*100}")
    print("ATTENTION FLOW VISUALIZATION")
    print('='*100)

    game = data['results'][game_idx]
    tokens = game['tokens']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, layer_name in enumerate(['L8', 'L15', 'L31']):
        ax = axes[i]

        attention = np.array(game['layers'][layer_name]['attention'])  # (n_heads, seq_len, seq_len)
        avg_attention = attention.mean(axis=0)  # (seq_len, seq_len)

        # Show attention TO last token (output)
        attn_to_output = avg_attention[:, -1]

        # Plot
        ax.bar(range(len(attn_to_output)), attn_to_output, color='#2E86AB', alpha=0.7)
        ax.set_xlabel('Token Position', fontsize=10)
        ax.set_ylabel('Attention to Output', fontsize=10)
        ax.set_title(f'{layer_name} Attention Flow', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Highlight top 5
        top5_indices = np.argsort(attn_to_output)[::-1][:5]
        for idx in top5_indices:
            ax.bar(idx, attn_to_output[idx], color='#C73E1D', alpha=0.8)

        # Add token labels for top 5
        for idx in top5_indices[:3]:
            if idx < len(tokens):
                ax.text(idx, attn_to_output[idx], tokens[idx],
                       ha='center', va='bottom', fontsize=7, rotation=45)

    plt.tight_layout()
    output_path = '/home/ubuntu/llm_addiction/experiment_6_token_level_tracking/attention_flow.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def visualize_feature_heatmap(data, game_idx=0, layer='L8'):
    """Visualize feature activations across token positions"""

    print(f"\n{'='*100}")
    print(f"FEATURE ACTIVATION HEATMAP ({layer})")
    print('='*100)

    game = data['results'][game_idx]
    tokens = game['tokens']
    features = np.array(game['layers'][layer]['features'])  # (seq_len, 32768)

    # Select top 50 features by variance across positions
    feature_vars = np.var(features, axis=0)
    top_feature_indices = np.argsort(feature_vars)[::-1][:50]

    # Extract these features
    selected_features = features[:, top_feature_indices]  # (seq_len, 50)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))

    im = ax.imshow(selected_features.T, cmap='RdYlBu_r', aspect='auto', interpolation='nearest')

    ax.set_xlabel('Token Position', fontsize=12)
    ax.set_ylabel('Feature Index (Top 50 by variance)', fontsize=12)
    ax.set_title(f'{layer} Feature Activations Across Tokens', fontsize=14, fontweight='bold')

    # Token labels (show every 5th)
    tick_positions = list(range(0, len(tokens), 5)) + [len(tokens)-1]
    tick_labels = [tokens[i] if i < len(tokens) else '' for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)

    plt.colorbar(im, ax=ax, label='Activation')
    plt.tight_layout()

    output_path = f'/home/ubuntu/llm_addiction/experiment_6_token_level_tracking/feature_heatmap_{layer}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def compare_with_last_token_only(data):
    """
    Compare full token-level data vs last-token-only
    Show what information is lost
    """

    print(f"\n{'='*100}")
    print("COMPARISON: Token-Level vs Last-Token-Only")
    print('='*100)

    for game_idx, game in enumerate(data['results'][:2]):  # First 2 games
        print(f"\nGame {game_idx + 1}:")
        print("-"*100)

        tokens = game['tokens']

        for layer_name in ['L8', 'L31']:
            features = np.array(game['layers'][layer_name]['features'])  # (seq_len, 32768)

            # Last token features (what Experiment 1 captured)
            last_token_features = features[-1, :]

            # All position features (what Experiment 6 captures)
            all_features = features

            # Information lost: positions with high activation not in last token
            for pos in range(len(tokens)):
                pos_features = features[pos, :]

                # Features active here but not in last token
                active_here = np.where((pos_features > 0.5) & (last_token_features < 0.1))[0]

                if len(active_here) > 5:  # Significant difference
                    print(f"  Position {pos}: '{tokens[pos]}'")
                    print(f"    {layer_name}: {len(active_here)} features active here but NOT in last token")
                    print(f"    Top features: {active_here[:5].tolist()}")

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python analyze_token_attribution.py <token_level_data.json>")
        sys.exit(1)

    filepath = sys.argv[1]

    print(f"Loading data from: {filepath}")
    data = load_token_level_data(filepath)

    # Analyses
    analyze_token_attribution(data)
    visualize_attention_flow(data, game_idx=0)
    visualize_feature_heatmap(data, game_idx=0, layer='L8')
    visualize_feature_heatmap(data, game_idx=0, layer='L31')
    compare_with_last_token_only(data)

    print("\n✅ Token attribution analysis complete!")
