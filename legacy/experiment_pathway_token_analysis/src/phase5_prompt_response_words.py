#!/usr/bin/env python3
"""
Phase 5: Prompt-Response Word Correlation Analysis

Analyzes the relationship between:
1. Input prompt type (safe vs risky)
2. Feature patching condition (safe_mean vs risky_mean)
3. Output word patterns

This reveals:
- How prompt wording affects response language
- How feature manipulation affects language generation
- Interaction effects between prompts and features
"""

import json
import argparse
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
LOGGER = logging.getLogger(__name__)


class PromptResponseWordAnalyzer:
    def __init__(self, patching_file: Path, output_file: Path):
        self.patching_file = patching_file
        self.output_file = output_file

        # Store word counts by prompt_type and patch_condition
        self.words_by_condition = defaultdict(lambda: Counter())

        # Prompt type names from CLAUDE.md
        self.prompt_components = {
            'safe': 'Conservative prompts (less risky language)',
            'risky': 'Aggressive prompts (goal-oriented, maximize rewards)'
        }

    def tokenize_response(self, response: str) -> List[str]:
        """Extract meaningful tokens from response"""
        # Remove newlines and extra spaces
        response = ' '.join(response.split())

        # Extract all words and numbers
        tokens = re.findall(r'\$?\d+|\b[a-zA-Z]+\b', response.lower())

        return tokens

    def analyze_patching_data(self):
        """Analyze word patterns across prompt types and patching conditions"""
        LOGGER.info(f"Analyzing: {self.patching_file}")

        total_records = 0
        with open(self.patching_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue

                record = json.loads(line)
                prompt_type = record['prompt_type']  # 'safe' or 'risky'
                patch_condition = record['patch_condition']  # 'safe_mean' or 'risky_mean'
                response = record['response']

                # Tokenize response
                tokens = self.tokenize_response(response)

                # Create combined condition key
                condition_key = f"{prompt_type}_prompt_{patch_condition}_patch"

                # Count words
                self.words_by_condition[condition_key].update(tokens)

                total_records += 1

                if total_records % 1000 == 0:
                    LOGGER.info(f"Processed {total_records:,} records...")

        LOGGER.info(f"Total records analyzed: {total_records:,}")

    def compute_statistics(self) -> Dict:
        """Compute word frequency statistics and comparisons"""
        results = {
            'total_records_analyzed': sum(sum(counter.values()) for counter in self.words_by_condition.values()),
            'conditions': {}
        }

        # For each condition, get top words
        for condition, counter in self.words_by_condition.items():
            total_tokens = sum(counter.values())
            top_words = counter.most_common(20)

            results['conditions'][condition] = {
                'total_tokens': total_tokens,
                'unique_words': len(counter),
                'top_words': [
                    {'word': word, 'count': count, 'frequency': count / total_tokens}
                    for word, count in top_words
                ]
            }

        # Compute comparisons
        results['comparisons'] = self.compute_comparisons()

        return results

    def compute_comparisons(self) -> Dict:
        """Compare word usage between different conditions"""
        comparisons = {}

        # 1. Prompt effect (safe vs risky prompts with same patching)
        for patch in ['safe_mean', 'risky_mean']:
            safe_key = f"safe_prompt_{patch}_patch"
            risky_key = f"risky_prompt_{patch}_patch"

            if safe_key in self.words_by_condition and risky_key in self.words_by_condition:
                comparisons[f'prompt_effect_{patch}'] = self._compare_word_distributions(
                    self.words_by_condition[safe_key],
                    self.words_by_condition[risky_key],
                    f"safe_prompt vs risky_prompt (with {patch})"
                )

        # 2. Patching effect (safe vs risky patching with same prompt)
        for prompt in ['safe', 'risky']:
            safe_key = f"{prompt}_prompt_safe_mean_patch"
            risky_key = f"{prompt}_prompt_risky_mean_patch"

            if safe_key in self.words_by_condition and risky_key in self.words_by_condition:
                comparisons[f'patching_effect_{prompt}'] = self._compare_word_distributions(
                    self.words_by_condition[safe_key],
                    self.words_by_condition[risky_key],
                    f"safe_mean vs risky_mean (with {prompt} prompt)"
                )

        # 3. Interaction: All four conditions
        all_keys = [
            'safe_prompt_safe_mean_patch',
            'safe_prompt_risky_mean_patch',
            'risky_prompt_safe_mean_patch',
            'risky_prompt_risky_mean_patch'
        ]

        if all(k in self.words_by_condition for k in all_keys):
            comparisons['interaction'] = self._analyze_interaction(all_keys)

        return comparisons

    def _compare_word_distributions(self, counter1: Counter, counter2: Counter, description: str) -> Dict:
        """Compare two word distributions"""
        # Get top words from each
        top1 = set(word for word, _ in counter1.most_common(50))
        top2 = set(word for word, _ in counter2.most_common(50))

        # Find distinctive words (appear in one but not other's top 50)
        distinctive_1 = top1 - top2
        distinctive_2 = top2 - top1
        common = top1 & top2

        return {
            'description': description,
            'common_words': len(common),
            'distinctive_in_first': list(distinctive_1)[:10],
            'distinctive_in_second': list(distinctive_2)[:10],
            'enrichment': self._compute_enrichment(counter1, counter2)
        }

    def _compute_enrichment(self, counter1: Counter, counter2: Counter) -> List[Dict]:
        """Find words enriched in counter1 vs counter2"""
        total1 = sum(counter1.values())
        total2 = sum(counter2.values())

        enrichments = []
        for word in set(counter1.keys()) | set(counter2.keys()):
            freq1 = counter1.get(word, 0) / total1 if total1 > 0 else 0
            freq2 = counter2.get(word, 0) / total2 if total2 > 0 else 0

            # Skip very rare words
            if freq1 < 0.001 and freq2 < 0.001:
                continue

            # Compute log2 fold change
            if freq2 > 0:
                fold_change = freq1 / freq2
                log2fc = np.log2(fold_change) if freq1 > 0 else -10
            else:
                log2fc = 10 if freq1 > 0 else 0

            enrichments.append({
                'word': word,
                'freq1': freq1,
                'freq2': freq2,
                'log2_fold_change': log2fc
            })

        # Sort by absolute log2 fold change
        enrichments.sort(key=lambda x: abs(x['log2_fold_change']), reverse=True)

        return enrichments[:20]

    def _analyze_interaction(self, all_keys: List[str]) -> Dict:
        """Analyze interaction between prompt type and patching condition"""
        # Get word counts for all four conditions
        counters = {key: self.words_by_condition[key] for key in all_keys}

        # Find words that show interaction pattern
        # (different in both safe/risky prompts AND safe/risky patches)

        safe_safe = counters['safe_prompt_safe_mean_patch']
        safe_risky = counters['safe_prompt_risky_mean_patch']
        risky_safe = counters['risky_prompt_safe_mean_patch']
        risky_risky = counters['risky_prompt_risky_mean_patch']

        # Get totals
        totals = {key: sum(counter.values()) for key, counter in counters.items()}

        # Find words with interesting patterns
        interaction_words = []
        all_words = set()
        for counter in counters.values():
            all_words.update(counter.keys())

        for word in all_words:
            freqs = {
                'safe_safe': safe_safe.get(word, 0) / totals['safe_prompt_safe_mean_patch'],
                'safe_risky': safe_risky.get(word, 0) / totals['safe_prompt_risky_mean_patch'],
                'risky_safe': risky_safe.get(word, 0) / totals['risky_prompt_safe_mean_patch'],
                'risky_risky': risky_risky.get(word, 0) / totals['risky_prompt_risky_mean_patch']
            }

            # Skip very rare words
            if max(freqs.values()) < 0.001:
                continue

            # Compute variance (higher variance = more interaction)
            mean_freq = sum(freqs.values()) / 4
            variance = sum((f - mean_freq) ** 2 for f in freqs.values()) / 4

            interaction_words.append({
                'word': word,
                'frequencies': freqs,
                'variance': variance
            })

        # Sort by variance (highest interaction)
        interaction_words.sort(key=lambda x: x['variance'], reverse=True)

        return {
            'top_interaction_words': interaction_words[:20]
        }

    def save_results(self, results: Dict):
        """Save results to JSON file"""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_file, 'w') as f:
            json.dump(results, f, indent=2)

        LOGGER.info(f"Results saved to: {self.output_file}")


# Add numpy import for log2 computation
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Phase 5: Prompt-Response Word Correlation Analysis')
    parser.add_argument('--patching-file', type=str, required=True, help='Phase 1 patching JSONL file')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file')

    args = parser.parse_args()

    patching_file = Path(args.patching_file)
    output_file = Path(args.output)

    if not patching_file.exists():
        LOGGER.error(f"Patching file not found: {patching_file}")
        return

    analyzer = PromptResponseWordAnalyzer(patching_file, output_file)

    # Analyze
    analyzer.analyze_patching_data()

    # Compute statistics
    results = analyzer.compute_statistics()

    # Save
    analyzer.save_results(results)

    LOGGER.info("✅ Phase 5 analysis complete!")


if __name__ == '__main__':
    main()
