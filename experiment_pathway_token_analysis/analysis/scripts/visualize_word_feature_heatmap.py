#!/usr/bin/env python3
"""
Image 2: Word-Feature Association Heatmap

Shows which words (from model outputs) are most strongly associated with
risky vs safe features.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from collections import defaultdict

plt.style.use('seaborn-v0_8-darkgrid')

def load_phase4_data():
    """Load Phase 4 word-feature correlation data"""
    print("Loading Phase 4 data...")

    all_correlations = []

    for gpu in [4, 5, 6, 7]:
        file_path = Path(f"/data/llm_addiction/experiment_pathway_token_analysis/results/phase4_word_feature_FULL/word_feature_correlation_gpu{gpu}.json")

        with open(file_path, 'r') as f:
            data = json.load(f)

        all_correlations.extend(data['word_feature_correlations'])

    print(f"Loaded {len(all_correlations):,} word-feature correlations")
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
                if comp['cohens_d'] > 0.2:  # Risky threshold
                    risky_features.add(comp['feature'])
                elif comp['cohens_d'] < -0.2:  # Safe threshold
                    safe_features.add(comp['feature'])

    print(f"Risky features: {len(risky_features):,}")
    print(f"Safe features: {len(safe_features):,}")

    return risky_features, safe_features

def aggregate_word_feature_associations(correlations, risky_features, safe_features):
    """Aggregate word associations by risky/safe features"""
    print("\nAggregating word-feature associations...")

    word_risky_activations = defaultdict(list)
    word_safe_activations = defaultdict(list)

    for corr in correlations:
        word = corr['word']
        feature = corr['feature']
        activation = abs(corr['mean_activation'])

        if feature in risky_features:
            word_risky_activations[word].append(activation)
        elif feature in safe_features:
            word_safe_activations[word].append(activation)

    # Calculate mean activations
    word_risky_means = {word: np.mean(acts) for word, acts in word_risky_activations.items() if len(acts) >= 3}
    word_safe_means = {word: np.mean(acts) for word, acts in word_safe_activations.items() if len(acts) >= 3}

    print(f"Words with risky associations: {len(word_risky_means):,}")
    print(f"Words with safe associations: {len(word_safe_means):,}")

    return word_risky_means, word_safe_means

def get_top_words(word_risky_means, word_safe_means, n=30):
    """Get top N words for risky and safe"""

    # Top risky words
    top_risky = sorted(word_risky_means.items(), key=lambda x: x[1], reverse=True)[:n]
    top_risky_words = [w for w, _ in top_risky]

    # Top safe words
    top_safe = sorted(word_safe_means.items(), key=lambda x: x[1], reverse=True)[:n]
    top_safe_words = [w for w, _ in top_safe]

    return top_risky_words, top_safe_words

def create_heatmap(word_risky_means, word_safe_means, top_risky_words, top_safe_words):
    """Create comprehensive heatmap visualization"""

    # Create figure with 3 subplots
    fig = plt.figure(figsize=(20, 12))

    # Subplot 1: Top 30 Risky Words
    ax1 = plt.subplot(1, 3, 1)

    risky_data = []
    risky_labels = []
    for word in top_risky_words[:30]:
        risky_act = word_risky_means.get(word, 0)
        safe_act = word_safe_means.get(word, 0)
        risky_data.append([risky_act, safe_act])
        # Truncate long words
        display_word = word[:20] + '...' if len(word) > 20 else word
        risky_labels.append(display_word)

    risky_array = np.array(risky_data)
    sns.heatmap(risky_array, annot=True, fmt='.2f', cmap='Reds',
                xticklabels=['Risky\nFeatures', 'Safe\nFeatures'],
                yticklabels=risky_labels, ax=ax1, cbar_kws={'label': 'Mean Activation'})
    ax1.set_title('Top 30 Risky-Associated Words', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Word', fontsize=12, fontweight='bold')

    # Subplot 2: Top 30 Safe Words
    ax2 = plt.subplot(1, 3, 2)

    safe_data = []
    safe_labels = []
    for word in top_safe_words[:30]:
        risky_act = word_risky_means.get(word, 0)
        safe_act = word_safe_means.get(word, 0)
        safe_data.append([risky_act, safe_act])
        display_word = word[:20] + '...' if len(word) > 20 else word
        safe_labels.append(display_word)

    safe_array = np.array(safe_data)
    sns.heatmap(safe_array, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=['Risky\nFeatures', 'Safe\nFeatures'],
                yticklabels=safe_labels, ax=ax2, cbar_kws={'label': 'Mean Activation'})
    ax2.set_title('Top 30 Safe-Associated Words', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Word', fontsize=12, fontweight='bold')

    # Subplot 3: Differential activation (risky - safe)
    ax3 = plt.subplot(1, 3, 3)

    # Get words that appear in both
    common_words = set(top_risky_words[:15]) | set(top_safe_words[:15])

    diff_data = []
    diff_labels = []
    for word in sorted(common_words, key=lambda w: word_risky_means.get(w, 0) - word_safe_means.get(w, 0), reverse=True):
        risky_act = word_risky_means.get(word, 0)
        safe_act = word_safe_means.get(word, 0)
        diff = risky_act - safe_act
        diff_data.append([diff])
        display_word = word[:20] + '...' if len(word) > 20 else word
        diff_labels.append(display_word)

    diff_array = np.array(diff_data)
    sns.heatmap(diff_array, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                xticklabels=['Risky - Safe\nDifference'],
                yticklabels=diff_labels, ax=ax3, cbar_kws={'label': 'Activation Difference'})
    ax3.set_title('Differential Word Association\n(Risky - Safe)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Word', fontsize=12, fontweight='bold')

    # Overall title
    fig.suptitle('Phase 4: Word-Feature Association Analysis\n' +
                f'Risky Features vs Safe Features',
                fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig

def main():
    print("="*80)
    print("Image 2: Word-Feature Association Heatmap")
    print("="*80)
    print()

    # Load data
    correlations = load_phase4_data()
    risky_features, safe_features = load_phase5_classifications()

    # Aggregate associations
    word_risky_means, word_safe_means = aggregate_word_feature_associations(
        correlations, risky_features, safe_features
    )

    # Get top words
    top_risky_words, top_safe_words = get_top_words(word_risky_means, word_safe_means, n=30)

    print("\nTop 10 Risky Words:")
    for i, word in enumerate(top_risky_words[:10], 1):
        print(f"  {i:2d}. '{word}' (activation={word_risky_means[word]:.3f})")

    print("\nTop 10 Safe Words:")
    for i, word in enumerate(top_safe_words[:10], 1):
        print(f"  {i:2d}. '{word}' (activation={word_safe_means[word]:.3f})")

    # Create visualization
    print("\nCreating heatmap visualization...")
    fig = create_heatmap(word_risky_means, word_safe_means, top_risky_words, top_safe_words)

    # Save
    output_dir = Path("/home/ubuntu/llm_addiction/experiment_pathway_token_analysis/analysis/images")

    png_path = output_dir / "02_word_feature_association_heatmap.png"
    pdf_path = output_dir / "02_word_feature_association_heatmap.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')

    print(f"\n✅ Saved visualization:")
    print(f"   PNG: {png_path}")
    print(f"   PDF: {pdf_path}")
    print()

if __name__ == '__main__':
    main()
