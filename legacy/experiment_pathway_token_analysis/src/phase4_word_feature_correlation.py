#!/usr/bin/env python3
"""
Phase 4 (Redesigned): Output Word-Feature Correlation Analysis

Analyzes the correlation between:
- Output words (from model responses)
- SAE feature activations (all 2,787 features)

This reveals:
- Which features are activated when specific words appear
- Feature-word associations across the entire vocabulary
- Statistical significance of word-feature relationships
"""

import json
import argparse
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import logging
import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
LOGGER = logging.getLogger(__name__)


class WordFeatureCorrelationAnalyzer:
    def __init__(self, patching_file: Path, output_file: Path, min_word_count: int = 10):
        self.patching_file = patching_file
        self.output_file = output_file
        self.min_word_count = min_word_count

        # Store data
        self.word_feature_data = defaultdict(lambda: defaultdict(list))
        # word_feature_data[word][feature] = [activation values when word appears]

        self.word_counts = Counter()
        self.all_feature_names = set()

    def tokenize_response_regex(self, response: str) -> List[str]:
        """Fallback: Extract tokens using regex (for backward compatibility)"""
        response = ' '.join(response.split())
        tokens = re.findall(r'\$?\d+|\b[a-zA-Z]+\b', response.lower())
        return tokens

    def get_tokens_from_record(self, record: dict) -> List[str]:
        """Get tokens from record - prefer actual BPE tokens, fallback to regex"""
        # Prefer actual BPE tokens from Phase 1
        if 'generated_tokens' in record and record['generated_tokens']:
            # Clean and normalize tokens
            tokens = []
            for tok in record['generated_tokens']:
                # Strip whitespace and convert to lowercase for consistency
                cleaned = tok.strip().lower()
                if cleaned:
                    tokens.append(cleaned)
            return tokens

        # Fallback to regex tokenization for backward compatibility
        return self.tokenize_response_regex(record.get('response', ''))

    def analyze_patching_data(self):
        """Collect word-feature activation data"""
        LOGGER.info(f"Analyzing: {self.patching_file}")

        total_records = 0
        bpe_token_count = 0
        regex_fallback_count = 0

        with open(self.patching_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue

                record = json.loads(line)
                all_features = record['all_features']

                # Get feature names
                if not self.all_feature_names:
                    self.all_feature_names = set(all_features.keys())

                # Get tokens (prefer BPE, fallback to regex)
                if 'generated_tokens' in record and record['generated_tokens']:
                    bpe_token_count += 1
                else:
                    regex_fallback_count += 1

                words = set(self.get_tokens_from_record(record))  # Unique tokens in this response

                # Count word occurrences
                self.word_counts.update(words)

                # For each word in this response
                for word in words:
                    # Record all feature activations when this word appears
                    for feature_name, activation_value in all_features.items():
                        self.word_feature_data[word][feature_name].append(activation_value)

                total_records += 1

                if total_records % 1000 == 0:
                    LOGGER.info(f"Processed {total_records:,} records...")

        LOGGER.info(f"Total records analyzed: {total_records:,}")
        LOGGER.info(f"  - BPE tokens used: {bpe_token_count:,} ({100*bpe_token_count/max(1,total_records):.1f}%)")
        LOGGER.info(f"  - Regex fallback: {regex_fallback_count:,} ({100*regex_fallback_count/max(1,total_records):.1f}%)")
        LOGGER.info(f"Unique tokens found: {len(self.word_counts):,}")
        LOGGER.info(f"Total features tracked: {len(self.all_feature_names):,}")

    def compute_correlations(self) -> Dict:
        """Compute word-feature correlations"""
        LOGGER.info("Computing word-feature correlations...")

        results = {
            'total_words': len(self.word_counts),
            'total_features': len(self.all_feature_names),
            'min_word_count': self.min_word_count,
            'word_feature_correlations': []
        }

        # Filter words by minimum count
        frequent_words = {word for word, count in self.word_counts.items()
                         if count >= self.min_word_count}

        LOGGER.info(f"Analyzing {len(frequent_words):,} words (count >= {self.min_word_count})")

        all_correlations = []

        for word_idx, word in enumerate(sorted(frequent_words)):
            if (word_idx + 1) % 100 == 0:
                LOGGER.info(f"  Processing word {word_idx + 1}/{len(frequent_words)}...")

            for feature_name in self.all_feature_names:
                activations = self.word_feature_data[word][feature_name]

                if len(activations) < self.min_word_count:
                    continue

                # Compute statistics
                mean_activation = np.mean(activations)
                std_activation = np.std(activations)

                # Store correlation data
                all_correlations.append({
                    'word': word,
                    'feature': feature_name,
                    'word_count': len(activations),
                    'mean_activation': float(mean_activation),
                    'std_activation': float(std_activation)
                })

        # Sort by mean activation (descending)
        all_correlations.sort(key=lambda x: abs(x['mean_activation']), reverse=True)

        # Store ALL correlations (removed 10,000 limit for full coverage)
        results['word_feature_correlations'] = all_correlations

        # Compute summary statistics
        results['top_word_feature_pairs'] = self._get_top_pairs(all_correlations)
        results['feature_word_associations'] = self._get_feature_word_map(all_correlations)

        return results

    def _get_top_pairs(self, all_correlations: List[Dict]) -> Dict:
        """Get top word-feature pairs by activation strength"""
        top_positive = sorted(all_correlations, key=lambda x: x['mean_activation'], reverse=True)[:50]
        top_negative = sorted(all_correlations, key=lambda x: x['mean_activation'])[:50]

        return {
            'top_positive_associations': top_positive,
            'top_negative_associations': top_negative
        }

    def _get_feature_word_map(self, all_correlations: List[Dict]) -> Dict:
        """For each feature, find top associated words"""
        feature_words = defaultdict(list)

        for corr in all_correlations:
            feature_words[corr['feature']].append({
                'word': corr['word'],
                'mean_activation': corr['mean_activation'],
                'word_count': corr['word_count']
            })

        # For each feature, keep top 20 words by activation
        feature_map = {}
        for feature, words in feature_words.items():
            sorted_words = sorted(words, key=lambda x: abs(x['mean_activation']), reverse=True)[:20]
            feature_map[feature] = sorted_words

        return feature_map

    def save_results(self, results: Dict):
        """Save results to JSON file"""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_file, 'w') as f:
            json.dump(results, f, indent=2)

        LOGGER.info(f"Results saved to: {self.output_file}")


def main():
    parser = argparse.ArgumentParser(description='Phase 4: Output Word-Feature Correlation Analysis')
    parser.add_argument('--patching-file', type=str, required=True, help='Phase 1 patching JSONL file')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file')
    parser.add_argument('--min-word-count', type=int, default=1, help='Minimum word count threshold (1=all words)')

    args = parser.parse_args()

    patching_file = Path(args.patching_file)
    output_file = Path(args.output)

    if not patching_file.exists():
        LOGGER.error(f"Patching file not found: {patching_file}")
        return

    analyzer = WordFeatureCorrelationAnalyzer(patching_file, output_file, args.min_word_count)

    # Analyze
    analyzer.analyze_patching_data()

    # Compute correlations
    results = analyzer.compute_correlations()

    # Save
    analyzer.save_results(results)

    LOGGER.info("✅ Phase 4 word-feature correlation analysis complete!")


if __name__ == '__main__':
    main()
