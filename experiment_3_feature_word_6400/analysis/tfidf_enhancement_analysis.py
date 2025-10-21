#!/usr/bin/env python3
"""
Experiment 3: TF-IDF Enhancement for Feature-Word Analysis
Adds TF-IDF scoring to identify feature-characteristic words
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
import math

def load_feature_word_data(filepath: str) -> Dict:
    """Load existing feature-word analysis data"""
    print(f"Loading data from {filepath}...")
    with open(filepath, 'r') as f:
        data = json.load(f)
    print(f"Loaded {data['n_features']} features")
    return data

def compute_tf(word_counts: Dict[str, int]) -> Dict[str, float]:
    """
    Compute Term Frequency (TF)
    TF(word) = count(word) / total_words
    """
    total = sum(word_counts.values())
    if total == 0:
        return {}
    return {word: count / total for word, count in word_counts.items()}

def compute_idf(features_data: List[Dict], min_features: int = 2) -> Dict[str, float]:
    """
    Compute Inverse Document Frequency (IDF)
    IDF(word) = log(n_features / n_features_with_word)

    Args:
        features_data: List of feature results
        min_features: Minimum number of features a word must appear in

    Returns:
        Dict mapping words to IDF scores
    """
    # Count in how many features each word appears in high-activation set
    word_feature_count = Counter()

    for feature_data in features_data:
        # Get unique words from this feature's high activation words
        feature_words = set()
        for word_info in feature_data.get('high_activation_words', []):
            if word_info['diff'] > 0:  # Only positive associations
                feature_words.add(word_info['word'])

        for word in feature_words:
            word_feature_count[word] += 1

    # Compute IDF
    n_features = len(features_data)
    idf_scores = {}

    for word, count in word_feature_count.items():
        if count >= min_features:  # Filter rare words
            idf_scores[word] = math.log(n_features / count)

    return idf_scores

def compute_tfidf_scores(feature_data: Dict, idf_scores: Dict[str, float]) -> List[Dict]:
    """
    Compute TF-IDF scores for words in a feature

    Returns:
        List of word dicts with TF-IDF scores added
    """
    results = []

    for word_info in feature_data.get('high_activation_words', []):
        word = word_info['word']

        # Use high_freq as TF (already normalized frequency)
        tf = word_info['high_freq']

        # Get IDF score
        idf = idf_scores.get(word, 0.0)

        # Compute TF-IDF
        tfidf = tf * idf

        results.append({
            **word_info,  # Keep original fields
            'tf': tf,
            'idf': idf,
            'tfidf': tfidf
        })

    # Sort by TF-IDF score (descending)
    results.sort(key=lambda x: x['tfidf'], reverse=True)

    return results

def analyze_feature_specificity(features_data: List[Dict], idf_scores: Dict[str, float]) -> pd.DataFrame:
    """
    Analyze how specific each word is to its feature

    Returns DataFrame with:
    - word
    - n_features (how many features this word appears in)
    - idf (inverse document frequency)
    - top_features (features where this word has highest TF-IDF)
    """
    word_analysis = {}

    for feature_data in features_data:
        feature_name = feature_data['feature']

        for word_info in feature_data.get('high_activation_words', []):
            word = word_info['word']
            if word_info['diff'] <= 0:
                continue  # Skip negative associations

            if word not in word_analysis:
                word_analysis[word] = {
                    'word': word,
                    'n_features': 0,
                    'idf': idf_scores.get(word, 0.0),
                    'features': []
                }

            word_analysis[word]['n_features'] += 1
            word_analysis[word]['features'].append({
                'feature': feature_name,
                'freq': word_info['high_freq'],
                'diff': word_info['diff']
            })

    # Convert to DataFrame
    rows = []
    for word, data in word_analysis.items():
        # Get top 3 features by frequency
        top_features = sorted(data['features'], key=lambda x: x['freq'], reverse=True)[:3]
        top_features_str = ', '.join([f"{f['feature']}({f['freq']:.3f})" for f in top_features])

        rows.append({
            'word': word,
            'n_features': data['n_features'],
            'idf': data['idf'],
            'avg_freq': np.mean([f['freq'] for f in data['features']]),
            'max_freq': max([f['freq'] for f in data['features']]),
            'top_features': top_features_str
        })

    df = pd.DataFrame(rows)
    df = df.sort_values('idf', ascending=False)

    return df

def create_tfidf_comparison_plot(
    features_data: List[Dict],
    output_path: Path
):
    """Create visualization comparing frequency-based vs TF-IDF ranking"""
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Select a few representative features
    sample_features = features_data[:5]

    fig, axes = plt.subplots(len(sample_features), 2, figsize=(16, 4 * len(sample_features)))

    if len(sample_features) == 1:
        axes = axes.reshape(1, -1)

    for idx, feature_data in enumerate(sample_features):
        feature_name = feature_data['feature']

        # Get top words by diff and by TF-IDF
        words_by_diff = sorted(feature_data.get('high_activation_words', []),
                               key=lambda x: x['diff'], reverse=True)[:10]
        words_by_tfidf = sorted(feature_data.get('high_activation_words', []),
                               key=lambda x: x.get('tfidf', 0), reverse=True)[:10]

        # Plot frequency difference ranking
        if words_by_diff:
            words = [w['word'] for w in words_by_diff]
            diffs = [w['diff'] for w in words_by_diff]

            axes[idx, 0].barh(range(len(words)), diffs, color='skyblue', edgecolor='black')
            axes[idx, 0].set_yticks(range(len(words)))
            axes[idx, 0].set_yticklabels(words)
            axes[idx, 0].set_xlabel('Frequency Difference', fontsize=10)
            axes[idx, 0].set_title(f'{feature_name} - Frequency Ranking', fontsize=12, fontweight='bold')
            axes[idx, 0].invert_yaxis()
            axes[idx, 0].grid(axis='x', alpha=0.3)

        # Plot TF-IDF ranking
        if words_by_tfidf and any(w.get('tfidf', 0) > 0 for w in words_by_tfidf):
            words = [w['word'] for w in words_by_tfidf]
            tfidfs = [w.get('tfidf', 0) for w in words_by_tfidf]

            axes[idx, 1].barh(range(len(words)), tfidfs, color='coral', edgecolor='black')
            axes[idx, 1].set_yticks(range(len(words)))
            axes[idx, 1].set_yticklabels(words)
            axes[idx, 1].set_xlabel('TF-IDF Score', fontsize=10)
            axes[idx, 1].set_title(f'{feature_name} - TF-IDF Ranking', fontsize=12, fontweight='bold')
            axes[idx, 1].invert_yaxis()
            axes[idx, 1].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved TF-IDF comparison plot to {output_path}")

def main():
    # Paths
    data_path = "/data/llm_addiction/experiment_3_feature_word/final_feature_word_20251010_055835.json"
    output_dir = Path("/home/ubuntu/llm_addiction/experiment_3_feature_word_6400/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("EXPERIMENT 3: TF-IDF ENHANCEMENT ANALYSIS")
    print("="*80)

    # Load data
    data = load_feature_word_data(data_path)
    features_data = data['results']

    # Compute IDF scores across all features
    print(f"\n1. Computing IDF scores across {len(features_data)} features...")
    idf_scores = compute_idf(features_data, min_features=2)
    print(f"   Found {len(idf_scores)} words with IDF scores")

    # Save IDF scores
    idf_df = pd.DataFrame([
        {'word': word, 'idf': score, 'specificity': 'high' if score > 3 else 'medium' if score > 1.5 else 'low'}
        for word, score in sorted(idf_scores.items(), key=lambda x: x[1], reverse=True)
    ])
    idf_output = output_dir / "word_idf_scores.csv"
    idf_df.to_csv(idf_output, index=False)
    print(f"   Saved IDF scores to {idf_output.name}")

    # Compute TF-IDF for each feature
    print(f"\n2. Computing TF-IDF scores for each feature...")
    enhanced_features = []
    for feature_data in features_data:
        enhanced_words = compute_tfidf_scores(feature_data, idf_scores)
        enhanced_features.append({
            **feature_data,
            'high_activation_words_tfidf': enhanced_words
        })

    # Save enhanced data
    enhanced_output = output_dir / "feature_word_tfidf_enhanced.json"
    with open(enhanced_output, 'w') as f:
        json.dump({
            'timestamp': data['timestamp'],
            'n_features': data['n_features'],
            'idf_computed': True,
            'results': enhanced_features
        }, f, indent=2)
    print(f"   Saved enhanced data to {enhanced_output.name}")

    # Analyze feature specificity
    print(f"\n3. Analyzing word specificity across features...")
    specificity_df = analyze_feature_specificity(features_data, idf_scores)
    specificity_output = output_dir / "word_specificity_analysis.csv"
    specificity_df.to_csv(specificity_output, index=False)
    print(f"   Saved specificity analysis to {specificity_output.name}")

    # Print summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")

    print(f"\nMost feature-specific words (high IDF):")
    print(specificity_df.head(10).to_string(index=False))

    print(f"\nMost common words (low IDF, appear in many features):")
    print(specificity_df.tail(10).to_string(index=False))

    # Create visualization
    print(f"\n4. Creating TF-IDF comparison visualization...")
    plot_output = output_dir / "tfidf_vs_frequency_ranking.png"
    create_tfidf_comparison_plot(enhanced_features, plot_output)

    # Top features by TF-IDF
    print(f"\n5. Extracting top TF-IDF words per feature...")
    top_tfidf_rows = []
    for feature_data in enhanced_features:
        top_words = feature_data.get('high_activation_words_tfidf', [])[:5]
        if top_words:
            for rank, word_info in enumerate(top_words, 1):
                top_tfidf_rows.append({
                    'feature': feature_data['feature'],
                    'type': feature_data['type'],
                    'rank': rank,
                    'word': word_info['word'],
                    'tfidf': word_info.get('tfidf', 0),
                    'freq_diff': word_info['diff'],
                    'high_freq': word_info['high_freq']
                })

    top_tfidf_df = pd.DataFrame(top_tfidf_rows)
    top_tfidf_output = output_dir / "top_tfidf_words_per_feature.csv"
    top_tfidf_df.to_csv(top_tfidf_output, index=False)
    print(f"   Saved top TF-IDF words to {top_tfidf_output.name}")

    print(f"\n{'='*80}")
    print("TF-IDF ENHANCEMENT COMPLETE")
    print(f"{'='*80}")
    print(f"\nAll results saved to: {output_dir}")
    print(f"\nKey outputs:")
    print(f"  1. {idf_output.name} - IDF scores for all words")
    print(f"  2. {enhanced_output.name} - Full data with TF-IDF scores")
    print(f"  3. {specificity_output.name} - Word specificity analysis")
    print(f"  4. {plot_output.name} - Visual comparison")
    print(f"  5. {top_tfidf_output.name} - Top TF-IDF words per feature")

if __name__ == "__main__":
    main()
