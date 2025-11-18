#!/usr/bin/env python3
"""
Causal Word Patching Analysis for 2,787 Features

Method: Compare word frequencies BEFORE and AFTER feature patching
- This reveals which words a feature CAUSALLY adds or removes
- No activation extraction needed - direct causal analysis!

Key comparisons:
1. safe_baseline vs safe_with_risky_patch → Words added by risky feature
2. risky_baseline vs risky_with_safe_patch → Words removed by safe feature
3. Feature amplification: safe_baseline vs safe_with_safe_patch
4. Feature suppression: risky_baseline vs risky_with_risky_patch
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import re
from scipy import stats


class CausalWordPatchingAnalyzer:
    """
    Analyze causal word effects through feature patching

    Direct comparison of responses before/after patching reveals
    which words each feature causally controls.
    """

    def __init__(self):
        # Paths
        self.exp2_response_dir = Path("/data/llm_addiction/experiment_2_multilayer_patching/response_logs")
        self.results_dir = Path("/home/ubuntu/llm_addiction/experiment_3_feature_word_patching/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Load causal features
        print("Loading 2,787 causal features...")
        safe_df = pd.read_csv("/home/ubuntu/llm_addiction/analysis/CORRECT_consistent_safe_features.csv")
        risky_df = pd.read_csv("/home/ubuntu/llm_addiction/analysis/CORRECT_consistent_risky_features.csv")

        self.causal_features = set()
        for _, row in safe_df.iterrows():
            self.causal_features.add(row['feature'])
        for _, row in risky_df.iterrows():
            self.causal_features.add(row['feature'])

        print(f"  Total causal features: {len(self.causal_features)}")

        # Load Exp2 response logs
        print("\nLoading Exp2 response logs...")
        self.exp2_data = self._load_exp2_responses()
        print(f"  Total features in Exp2: {len(self.exp2_data)}")

    def _load_exp2_responses(self):
        """Load all Exp2 response logs"""
        feature_data = {}

        log_files = sorted(self.exp2_response_dir.glob("*.json"))

        for log_file in tqdm(log_files, desc="Loading response logs"):
            with open(log_file, 'r') as f:
                data = json.load(f)

            for record in data:
                feature = record['feature']

                # Only process causal features
                if feature not in self.causal_features:
                    continue

                if feature not in feature_data:
                    feature_data[feature] = {
                        'safe_baseline': [],
                        'safe_with_safe_patch': [],
                        'safe_with_risky_patch': [],
                        'risky_baseline': [],
                        'risky_with_safe_patch': [],
                        'risky_with_risky_patch': []
                    }

                condition = record['condition']
                response = record['response']

                feature_data[feature][condition].append(response)

        return feature_data

    def extract_words(self, text):
        """Extract words from response text"""
        # Convert to lowercase and extract words
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())

        # Remove very common stopwords
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
            'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
            'too', 'very', 'can', 'will', 'just', 'should', 'now'
        }

        # Filter words
        words = [w for w in words if w not in stopwords and len(w) > 2]

        return words

    def compare_word_frequencies(self, baseline_responses, patched_responses):
        """
        Compare word frequencies between baseline and patched responses

        Returns:
            added_words: Words significantly more frequent after patching
            removed_words: Words significantly less frequent after patching
        """
        # Extract all words
        baseline_words = []
        for response in baseline_responses:
            baseline_words.extend(self.extract_words(response))

        patched_words = []
        for response in patched_responses:
            patched_words.extend(self.extract_words(response))

        # Count frequencies
        baseline_counts = Counter(baseline_words)
        patched_counts = Counter(patched_words)

        # Get all unique words
        all_words = set(baseline_counts.keys()) | set(patched_counts.keys())

        # Calculate frequency changes
        word_effects = []

        for word in all_words:
            baseline_freq = baseline_counts[word] / max(len(baseline_words), 1)
            patched_freq = patched_counts[word] / max(len(patched_words), 1)

            # Log odds ratio (effect size)
            # Add smoothing to avoid division by zero
            odds_ratio = np.log((patched_freq + 0.001) / (baseline_freq + 0.001))

            # Chi-square test for significance
            contingency = np.array([
                [baseline_counts[word], len(baseline_words) - baseline_counts[word]],
                [patched_counts[word], len(patched_words) - patched_counts[word]]
            ])

            try:
                chi2, p_value = stats.chi2_contingency(contingency)[:2]
            except:
                p_value = 1.0

            word_effects.append({
                'word': word,
                'baseline_freq': baseline_freq,
                'patched_freq': patched_freq,
                'log_odds_ratio': odds_ratio,
                'p_value': p_value,
                'baseline_count': baseline_counts[word],
                'patched_count': patched_counts[word]
            })

        # Sort by absolute effect size
        word_effects.sort(key=lambda x: abs(x['log_odds_ratio']), reverse=True)

        # Filter significant words (p < 0.05 and effect size > 0.5)
        added_words = [w for w in word_effects
                      if w['log_odds_ratio'] > 0.5 and w['p_value'] < 0.05]
        removed_words = [w for w in word_effects
                        if w['log_odds_ratio'] < -0.5 and w['p_value'] < 0.05]

        return added_words, removed_words, word_effects

    def analyze_feature(self, feature):
        """
        Analyze causal word effects for a single feature

        Returns:
            dict with 4 causal comparisons
        """
        if feature not in self.exp2_data:
            return None

        data = self.exp2_data[feature]

        # Check if we have enough data
        if len(data['safe_baseline']) < 10 or len(data['risky_baseline']) < 10:
            return None

        result = {'feature': feature}

        # 1. Risky feature effect on safe prompt
        # (What words does the risky feature ADD?)
        added, removed, all_effects = self.compare_word_frequencies(
            data['safe_baseline'],
            data['safe_with_risky_patch']
        )
        result['risky_feature_adds'] = added[:20]  # Top 20
        result['risky_feature_removes'] = removed[:20]
        result['safe_risky_patch_all'] = all_effects[:50]

        # 2. Safe feature effect on risky prompt
        # (What words does the safe feature REMOVE?)
        added, removed, all_effects = self.compare_word_frequencies(
            data['risky_baseline'],
            data['risky_with_safe_patch']
        )
        result['safe_feature_adds'] = added[:20]
        result['safe_feature_removes'] = removed[:20]
        result['risky_safe_patch_all'] = all_effects[:50]

        # 3. Feature amplification (safe prompt + safe feature)
        added, removed, all_effects = self.compare_word_frequencies(
            data['safe_baseline'],
            data['safe_with_safe_patch']
        )
        result['safe_amplification_adds'] = added[:20]
        result['safe_amplification_removes'] = removed[:20]

        # 4. Feature suppression (risky prompt + risky feature)
        added, removed, all_effects = self.compare_word_frequencies(
            data['risky_baseline'],
            data['risky_with_risky_patch']
        )
        result['risky_amplification_adds'] = added[:20]
        result['risky_amplification_removes'] = removed[:20]

        return result

    def run_analysis(self):
        """Analyze all causal features"""

        print(f"\n{'='*80}")
        print(f"Causal Word Patching Analysis")
        print(f"Features: {len(self.causal_features)}")
        print(f"{'='*80}\n")

        results = []

        for feature in tqdm(sorted(self.causal_features), desc="Analyzing features"):
            result = self.analyze_feature(feature)
            if result:
                results.append(result)

        # Save results
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.results_dir / f'causal_word_effects_{timestamp}.json'

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*80}")
        print(f"✅ Analysis complete!")
        print(f"Results saved: {output_file}")
        print(f"Features analyzed: {len(results)}/{len(self.causal_features)}")
        print(f"{'='*80}\n")

        # Quick statistics
        print("=== Sample Results ===\n")
        if results:
            sample = results[0]
            print(f"Feature: {sample['feature']}")
            print(f"\nTop words ADDED by risky feature (to safe prompt):")
            for w in sample['risky_feature_adds'][:5]:
                print(f"  '{w['word']}': {w['baseline_freq']:.3f} → {w['patched_freq']:.3f} (p={w['p_value']:.4f})")

            print(f"\nTop words REMOVED by safe feature (from risky prompt):")
            for w in sample['safe_feature_removes'][:5]:
                print(f"  '{w['word']}': {w['baseline_freq']:.3f} → {w['patched_freq']:.3f} (p={w['p_value']:.4f})")

        return results, output_file


def main():
    analyzer = CausalWordPatchingAnalyzer()
    results, output_file = analyzer.run_analysis()


if __name__ == "__main__":
    main()
