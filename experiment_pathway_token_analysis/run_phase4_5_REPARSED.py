#!/usr/bin/env python3
"""
Run Phase 4 and Phase 5 on REPARSED Phase 1 data
Processes all GPU files and merges results
"""

import json
import argparse
import subprocess
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import logging
import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
LOGGER = logging.getLogger(__name__)

# Paths
PHASE1_DIR = Path("/data/llm_addiction/experiment_pathway_token_analysis/results/phase1_patching_REPARSED")
OUTPUT_DIR = Path("/data/llm_addiction/experiment_pathway_token_analysis/results")
SRC_DIR = Path("/home/ubuntu/llm_addiction/experiment_pathway_token_analysis/src")

GPU_FILES = [
    PHASE1_DIR / "phase1_patching_multifeature_gpu4.jsonl",
    PHASE1_DIR / "phase1_patching_multifeature_gpu5.jsonl",
    PHASE1_DIR / "phase1_patching_multifeature_gpu6.jsonl",
    PHASE1_DIR / "phase1_patching_multifeature_gpu7.jsonl",
]


def run_phase4_combined():
    """Run Phase 4 by processing all files and merging results"""
    LOGGER.info("=" * 60)
    LOGGER.info("PHASE 4: Word-Feature Correlation Analysis (REPARSED)")
    LOGGER.info("=" * 60)

    from collections import Counter

    # Aggregate data from all files
    word_feature_data = defaultdict(lambda: defaultdict(list))
    word_counts = Counter()
    all_feature_names = set()
    total_records = 0
    bpe_count = 0
    regex_count = 0

    import re
    def tokenize_regex(response: str) -> List[str]:
        response = ' '.join(response.split())
        return re.findall(r'\$?\d+|\b[a-zA-Z]+\b', response.lower())

    for gpu_file in GPU_FILES:
        LOGGER.info(f"Processing: {gpu_file.name}")

        with open(gpu_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue

                record = json.loads(line)
                all_features = record['all_features']

                if not all_feature_names:
                    all_feature_names = set(all_features.keys())

                # Get tokens
                if 'generated_tokens' in record and record['generated_tokens']:
                    tokens = [tok.strip().lower() for tok in record['generated_tokens'] if tok.strip()]
                    bpe_count += 1
                else:
                    tokens = tokenize_regex(record.get('response', ''))
                    regex_count += 1

                words = set(tokens)
                word_counts.update(words)

                for word in words:
                    for feature_name, activation_value in all_features.items():
                        word_feature_data[word][feature_name].append(activation_value)

                total_records += 1

                if total_records % 10000 == 0:
                    LOGGER.info(f"  Processed {total_records:,} records...")

    LOGGER.info(f"Total records: {total_records:,}")
    LOGGER.info(f"  BPE tokens: {bpe_count:,} ({100*bpe_count/total_records:.1f}%)")
    LOGGER.info(f"  Regex fallback: {regex_count:,} ({100*regex_count/total_records:.1f}%)")
    LOGGER.info(f"Unique tokens: {len(word_counts):,}")
    LOGGER.info(f"Total features: {len(all_feature_names):,}")

    # Compute correlations
    LOGGER.info("Computing word-feature correlations...")

    min_word_count = 10
    frequent_words = {word for word, count in word_counts.items() if count >= min_word_count}
    LOGGER.info(f"Analyzing {len(frequent_words):,} words (count >= {min_word_count})")

    all_correlations = []
    for word_idx, word in enumerate(sorted(frequent_words)):
        if (word_idx + 1) % 500 == 0:
            LOGGER.info(f"  Processing word {word_idx + 1}/{len(frequent_words)}...")

        for feature_name in all_feature_names:
            activations = word_feature_data[word][feature_name]
            if len(activations) < min_word_count:
                continue

            mean_activation = np.mean(activations)
            std_activation = np.std(activations)

            all_correlations.append({
                'word': word,
                'feature': feature_name,
                'word_count': len(activations),
                'mean_activation': float(mean_activation),
                'std_activation': float(std_activation)
            })

    # Sort by mean activation
    all_correlations.sort(key=lambda x: abs(x['mean_activation']), reverse=True)

    # Get top pairs
    top_positive = sorted(all_correlations, key=lambda x: x['mean_activation'], reverse=True)[:50]
    top_negative = sorted(all_correlations, key=lambda x: x['mean_activation'])[:50]

    # Feature-word map
    feature_words = defaultdict(list)
    for corr in all_correlations:
        feature_words[corr['feature']].append({
            'word': corr['word'],
            'mean_activation': corr['mean_activation'],
            'word_count': corr['word_count']
        })

    feature_map = {}
    for feature, words in feature_words.items():
        sorted_words = sorted(words, key=lambda x: abs(x['mean_activation']), reverse=True)[:20]
        feature_map[feature] = sorted_words

    results = {
        'total_words': len(word_counts),
        'total_features': len(all_feature_names),
        'total_records': total_records,
        'min_word_count': min_word_count,
        'frequent_words_analyzed': len(frequent_words),
        'total_correlations': len(all_correlations),
        'top_word_feature_pairs': {
            'top_positive_associations': top_positive,
            'top_negative_associations': top_negative
        },
        'feature_word_associations': feature_map,
        'word_feature_correlations': all_correlations[:10000]  # Top 10000 for file size
    }

    # Save
    output_file = OUTPUT_DIR / "phase4_word_feature_REPARSED" / "phase4_word_feature_combined.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    LOGGER.info(f"Phase 4 results saved to: {output_file}")
    LOGGER.info(f"Total word-feature correlations: {len(all_correlations):,}")

    return results


def run_phase5_combined():
    """Run Phase 5 by processing all files and merging results"""
    LOGGER.info("=" * 60)
    LOGGER.info("PHASE 5: Prompt-Feature Correlation Analysis (REPARSED)")
    LOGGER.info("=" * 60)

    # Aggregate data from all files
    prompt_features = defaultdict(lambda: defaultdict(list))
    total_records = 0

    for gpu_file in GPU_FILES:
        LOGGER.info(f"Processing: {gpu_file.name}")

        with open(gpu_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue

                record = json.loads(line)
                prompt_type = record['prompt_type']
                all_features = record['all_features']

                for feature_name, activation_value in all_features.items():
                    prompt_features[prompt_type][feature_name].append(activation_value)

                total_records += 1

                if total_records % 10000 == 0:
                    LOGGER.info(f"  Processed {total_records:,} records...")

    LOGGER.info(f"Total records: {total_records:,}")
    LOGGER.info(f"Prompt types: {list(prompt_features.keys())}")

    # Compute correlations
    LOGGER.info("Computing prompt-feature correlations...")

    all_features = set()
    for prompt_type, features in prompt_features.items():
        all_features.update(features.keys())

    LOGGER.info(f"Analyzing {len(all_features):,} features")

    feature_comparisons = []

    for feature_idx, feature_name in enumerate(sorted(all_features)):
        if (feature_idx + 1) % 500 == 0:
            LOGGER.info(f"  Processing feature {feature_idx + 1}/{len(all_features)}...")

        safe_activations = prompt_features['safe'][feature_name]
        risky_activations = prompt_features['risky'][feature_name]

        if len(safe_activations) < 10 or len(risky_activations) < 10:
            continue

        safe_mean = np.mean(safe_activations)
        safe_std = np.std(safe_activations)
        risky_mean = np.mean(risky_activations)
        risky_std = np.std(risky_activations)

        pooled_std = np.sqrt((safe_std**2 + risky_std**2) / 2)
        cohens_d = (risky_mean - safe_mean) / pooled_std if pooled_std > 0 else 0.0

        t_stat, p_value = stats.ttest_ind(risky_activations, safe_activations)

        feature_comparisons.append({
            'feature': feature_name,
            'safe_mean': float(safe_mean),
            'safe_std': float(safe_std),
            'safe_count': len(safe_activations),
            'risky_mean': float(risky_mean),
            'risky_std': float(risky_std),
            'risky_count': len(risky_activations),
            'mean_difference': float(risky_mean - safe_mean),
            'cohens_d': float(cohens_d),
            't_statistic': float(t_stat),
            'p_value': float(p_value)
        })

    # Sort by effect size
    feature_comparisons.sort(key=lambda x: abs(x['cohens_d']), reverse=True)

    # Compute summary
    significant = [f for f in feature_comparisons if f['p_value'] < 0.05]
    large_effect = [f for f in feature_comparisons if abs(f['cohens_d']) > 0.5]
    very_large_effect = [f for f in feature_comparisons if abs(f['cohens_d']) > 1.0]
    risky_biased = [f for f in significant if f['mean_difference'] > 0]
    safe_biased = [f for f in significant if f['mean_difference'] < 0]

    summary = {
        'total_features': len(feature_comparisons),
        'significant_features': len(significant),
        'large_effect_features': len(large_effect),
        'very_large_effect_features': len(very_large_effect),
        'risky_biased_features': len(risky_biased),
        'safe_biased_features': len(safe_biased),
        'top_risky_biased': risky_biased[:20],
        'top_safe_biased': safe_biased[:20],
        'top_effect_sizes': feature_comparisons[:50]
    }

    results = {
        'total_records': total_records,
        'prompt_types': list(prompt_features.keys()),
        'summary': summary,
        'feature_comparisons': feature_comparisons
    }

    # Save
    output_file = OUTPUT_DIR / "phase5_prompt_feature_REPARSED" / "phase5_prompt_feature_combined.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    LOGGER.info(f"Phase 5 results saved to: {output_file}")
    LOGGER.info(f"Summary: {len(significant)} significant, {len(large_effect)} large effect, {len(very_large_effect)} very large effect")

    return results


def main():
    parser = argparse.ArgumentParser(description='Run Phase 4 and 5 on REPARSED data')
    parser.add_argument('--phase', type=str, choices=['4', '5', 'both'], default='both')
    args = parser.parse_args()

    if args.phase in ['4', 'both']:
        run_phase4_combined()

    if args.phase in ['5', 'both']:
        run_phase5_combined()

    LOGGER.info("=" * 60)
    LOGGER.info("All phases complete!")
    LOGGER.info("=" * 60)


if __name__ == '__main__':
    main()
