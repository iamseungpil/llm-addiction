#!/usr/bin/env python3
"""
Phase 2 Analysis: Decision Signature
Identify multi-layer activation patterns that characterize "bet" vs "stop" decisions
"""

import json
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def load_pathway_data():
    """Load pathway data"""
    print("Loading pathway data...")
    with open('/data/llm_addiction/experiment_1_pathway_L1_31/final_pathway_L1_31_20251001_165207.json', 'r') as f:
        data = json.load(f)
    return data['results']

def load_top_features():
    """Load top features"""
    with open('/home/ubuntu/llm_addiction/experiment_1_layer_pathway_L1_31/feature_importance_results.json', 'r') as f:
        return json.load(f)

def create_decision_vectors(results, top_features, n_features_per_layer=5):
    """
    Create multi-layer decision vectors
    Each vector: [L8_feat1, L8_feat2, ..., L31_feat1, L31_feat2, ...]
    """
    print(f"\nCreating decision vectors (top {n_features_per_layer} features per critical layer)...")

    critical_layers = ['L8', 'L9', 'L10', 'L11', 'L31']

    # Get feature indices
    feature_indices = {}
    for layer in critical_layers:
        feature_indices[layer] = [
            f['feature_id'] for f in top_features[layer][:n_features_per_layer]
        ]

    # Extract vectors
    vectors = []
    labels = []
    outcomes = []

    for game in results:
        if not game['round_data']:
            continue

        last_round = game['round_data'][-1]
        decision = last_round['decision']['action']

        # Build multi-layer vector
        vector = []
        for layer in critical_layers:
            features = np.array(last_round['features'][layer])
            for feat_id in feature_indices[layer]:
                vector.append(features[feat_id])

        vectors.append(vector)
        labels.append(1 if decision == 'bet' else 0)
        outcomes.append(game['outcome'])

    vectors = np.array(vectors)
    labels = np.array(labels)

    print(f"Created {len(vectors)} decision vectors")
    print(f"Vector dimension: {vectors.shape[1]} (5 layers × {n_features_per_layer} features)")
    print(f"Bet decisions: {np.sum(labels == 1)}")
    print(f"Stop decisions: {np.sum(labels == 0)}")

    return vectors, labels, outcomes, feature_indices

def analyze_decision_signatures(vectors, labels, outcomes):
    """Analyze characteristic patterns of bet vs stop decisions"""

    print("\n" + "="*100)
    print("DECISION SIGNATURE ANALYSIS")
    print("="*100)

    bet_vectors = vectors[labels == 1]
    stop_vectors = vectors[labels == 0]

    # Compute statistics
    bet_mean = np.mean(bet_vectors, axis=0)
    stop_mean = np.mean(stop_vectors, axis=0)
    bet_std = np.std(bet_vectors, axis=0)
    stop_std = np.std(stop_vectors, axis=0)

    # Find most discriminative features in multi-layer vector
    pooled_std = np.sqrt((bet_std**2 + stop_std**2) / 2)
    cohen_d = (bet_mean - stop_mean) / (pooled_std + 1e-10)

    print("\n📊 Most Discriminative Features in Multi-Layer Signature:")
    print("-"*100)

    # Get indices of top discriminative features
    top_indices = np.argsort(np.abs(cohen_d))[::-1][:10]

    critical_layers = ['L8', 'L9', 'L10', 'L11', 'L31']
    features_per_layer = len(cohen_d) // len(critical_layers)

    for idx in top_indices:
        layer_idx = idx // features_per_layer
        feat_idx = idx % features_per_layer
        layer = critical_layers[layer_idx]

        print(f"  Position {idx}: {layer} Feature #{feat_idx}")
        print(f"    Cohen's d: {cohen_d[idx]:.3f}")
        print(f"    Bet mean: {bet_mean[idx]:.6f}, Stop mean: {stop_mean[idx]:.6f}")
        print()

    # Analyze outcome groups
    bankruptcy_mask = np.array([o == 'bankruptcy' for o in outcomes])
    safe_mask = np.array([o == 'voluntary_stop' for o in outcomes])

    bankruptcy_vectors = vectors[bankruptcy_mask]
    safe_vectors = vectors[safe_mask]

    print("\n📊 By Outcome (Bankruptcy vs Safe):")
    print("-"*100)

    if len(bankruptcy_vectors) > 0 and len(safe_vectors) > 0:
        b_mean = np.mean(bankruptcy_vectors, axis=0)
        s_mean = np.mean(safe_vectors, axis=0)
        b_std = np.std(bankruptcy_vectors, axis=0)
        s_std = np.std(safe_vectors, axis=0)

        pooled_std = np.sqrt((b_std**2 + s_std**2) / 2)
        outcome_cohen_d = (b_mean - s_mean) / (pooled_std + 1e-10)

        top_outcome_indices = np.argsort(np.abs(outcome_cohen_d))[::-1][:5]

        for idx in top_outcome_indices:
            layer_idx = idx // features_per_layer
            feat_idx = idx % features_per_layer
            layer = critical_layers[layer_idx]

            print(f"  {layer} Feature #{feat_idx}: Cohen's d = {outcome_cohen_d[idx]:.3f}")

def visualize_decision_space(vectors, labels, outcomes):
    """Visualize decision vectors in 2D using PCA and t-SNE"""

    print("\nCreating decision space visualizations...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # PCA
    pca = PCA(n_components=2)
    vectors_pca = pca.fit_transform(vectors)

    ax = axes[0]

    # Color by decision
    bet_mask = labels == 1
    stop_mask = labels == 0

    ax.scatter(vectors_pca[bet_mask, 0], vectors_pca[bet_mask, 1],
              c='red', alpha=0.6, s=50, label='Bet', edgecolors='black', linewidth=0.5)
    ax.scatter(vectors_pca[stop_mask, 0], vectors_pca[stop_mask, 1],
              c='blue', alpha=0.6, s=50, label='Stop', edgecolors='black', linewidth=0.5)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    ax.set_title('Decision Space (PCA)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # t-SNE
    print("  Computing t-SNE (may take a moment)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(vectors)//2))
    vectors_tsne = tsne.fit_transform(vectors)

    ax = axes[1]

    # Color by outcome
    bankruptcy_mask = np.array([o == 'bankruptcy' for o in outcomes])
    safe_mask = np.array([o == 'voluntary_stop' for o in outcomes])

    ax.scatter(vectors_tsne[bankruptcy_mask, 0], vectors_tsne[bankruptcy_mask, 1],
              c='#C73E1D', alpha=0.7, s=100, label='Bankruptcy', edgecolors='black', linewidth=1)
    ax.scatter(vectors_tsne[safe_mask, 0], vectors_tsne[safe_mask, 1],
              c='#2E86AB', alpha=0.7, s=100, label='Safe', edgecolors='black', linewidth=1)

    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.set_title('Decision Space (t-SNE)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = '/home/ubuntu/llm_addiction/experiment_1_layer_pathway_L1_31/decision_space.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def visualize_layer_contributions(vectors, labels, feature_indices):
    """Visualize contribution of each layer to decision"""

    print("\nAnalyzing layer contributions...")

    critical_layers = ['L8', 'L9', 'L10', 'L11', 'L31']
    features_per_layer = len(feature_indices[critical_layers[0]])

    bet_vectors = vectors[labels == 1]
    stop_vectors = vectors[labels == 0]

    # For each layer, compute average |activation|
    layer_activations = {layer: {'bet': [], 'stop': []} for layer in critical_layers}

    for i, layer in enumerate(critical_layers):
        start_idx = i * features_per_layer
        end_idx = start_idx + features_per_layer

        bet_layer_feats = bet_vectors[:, start_idx:end_idx]
        stop_layer_feats = stop_vectors[:, start_idx:end_idx]

        layer_activations[layer]['bet'] = np.mean(np.abs(bet_layer_feats), axis=1)
        layer_activations[layer]['stop'] = np.mean(np.abs(stop_layer_feats), axis=1)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(critical_layers))
    width = 0.35

    bet_means = [np.mean(layer_activations[layer]['bet']) for layer in critical_layers]
    stop_means = [np.mean(layer_activations[layer]['stop']) for layer in critical_layers]

    bet_stds = [np.std(layer_activations[layer]['bet']) for layer in critical_layers]
    stop_stds = [np.std(layer_activations[layer]['stop']) for layer in critical_layers]

    ax.bar(x - width/2, bet_means, width, yerr=bet_stds, label='Bet',
           color='#C73E1D', alpha=0.7, capsize=5)
    ax.bar(x + width/2, stop_means, width, yerr=stop_stds, label='Stop',
           color='#2E86AB', alpha=0.7, capsize=5)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Mean |Activation|', fontsize=12)
    ax.set_title('Layer Activation Levels by Decision Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(critical_layers)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = '/home/ubuntu/llm_addiction/experiment_1_layer_pathway_L1_31/layer_contributions.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

if __name__ == '__main__':
    print("="*100)
    print("PHASE 2: DECISION SIGNATURE ANALYSIS")
    print("Question: What is the multi-layer 'fingerprint' of bet vs stop decisions?")
    print("="*100)

    results = load_pathway_data()
    print(f"Loaded {len(results)} games")

    top_features = load_top_features()

    # Create decision vectors
    vectors, labels, outcomes, feature_indices = create_decision_vectors(results, top_features, n_features_per_layer=5)

    # Analyze signatures
    analyze_decision_signatures(vectors, labels, outcomes)

    # Visualizations
    visualize_decision_space(vectors, labels, outcomes)
    visualize_layer_contributions(vectors, labels, feature_indices)

    # Save
    output_file = '/home/ubuntu/llm_addiction/experiment_1_layer_pathway_L1_31/decision_signature_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'n_samples': len(vectors),
            'vector_dimension': vectors.shape[1],
            'n_bet': int(np.sum(labels == 1)),
            'n_stop': int(np.sum(labels == 0)),
            'feature_indices': feature_indices
        }, f, indent=2)

    print(f"\nSaved results: {output_file}")
    print("\n✅ Phase 2 Decision Signature Analysis Complete!")
