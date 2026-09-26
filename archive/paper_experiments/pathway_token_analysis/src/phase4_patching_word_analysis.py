#!/usr/bin/env python3
"""
Phase 4: Feature-Word Association Analysis (for Patching Data)

Input: Phase 1 patching data (JSONL with response texts)
Output: Feature-word associations (JSON)

Analyzes word patterns in responses for each feature/condition combination.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("phase4")

# Token pattern: numbers (with currency symbols), Korean, English, special chars
TOKEN_PATTERN = re.compile(
    r"(?:[$₩￦]?\d+(?:\.\d+)?)|[\uac00-\ud7a3]+|[a-zA-Z]+|[%$]"
)


def tokenize(text: str) -> list[str]:
    """Tokenize text into words/numbers"""
    tokens = TOKEN_PATTERN.findall(text)
    normalised = []
    for token in tokens:
        if token.isalpha():
            normalised.append(token.lower())
        else:
            normalised.append(token)
    return normalised


def laplace_log_ratio(count_a, total_a, count_b, total_b, vocab_size, alpha=1.0):
    """Compute Laplace-smoothed log-ratio (PMI-style)"""
    p_a = (count_a + alpha) / (total_a + alpha * vocab_size)
    p_b = (count_b + alpha) / (total_b + alpha * vocab_size)
    return math.log(p_a / p_b)


def analyse_feature_condition(feature: str, condition: str, token_counts, token_totals):
    """Analyze word associations for specific feature/condition"""
    condition_stats = token_counts[feature][condition]
    total_tokens = token_totals[feature][condition]

    if total_tokens == 0:
        return None

    # Top tokens for this condition
    top_tokens = Counter(condition_stats).most_common(20)

    return {
        "feature": feature,
        "condition": condition,
        "total_tokens": total_tokens,
        "top_tokens": [(word, count) for word, count in top_tokens],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 word analysis for patching data")
    parser.add_argument(
        "--patching-file",
        type=Path,
        required=True,
        help="Phase 1 patching data (JSONL)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON containing word statistics",
    )
    args = parser.parse_args()

    token_counts = defaultdict(lambda: defaultdict(Counter))
    token_totals = defaultdict(lambda: defaultdict(int))

    # Process patching data
    LOGGER.info("Reading patching data from %s", args.patching_file)
    with open(args.patching_file, "r") as f:
        for line in tqdm(f, desc="Processing responses"):
            if not line.strip():
                continue

            record = json.loads(line)
            target_feature = record.get("target_feature")
            patch_condition = record.get("patch_condition")
            prompt_type = record.get("prompt_type")
            response = record.get("response", "")

            if not target_feature or not response:
                continue

            # Combine patch_condition and prompt_type as condition
            condition = f"{patch_condition}_{prompt_type}"

            # Tokenize response
            tokens = tokenize(response)
            if not tokens:
                continue

            # Update counts
            counter = token_counts[target_feature][condition]
            counter.update(tokens)
            token_totals[target_feature][condition] += len(tokens)

    LOGGER.info("Collected token statistics for %d features", len(token_counts))

    # Analyze each feature/condition
    analyses = []
    for feature in tqdm(token_counts.keys(), desc="Analyzing features"):
        for condition in token_counts[feature].keys():
            result = analyse_feature_condition(feature, condition, token_counts, token_totals)
            if result:
                analyses.append(result)

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(analyses, handle, indent=2, ensure_ascii=False)

    LOGGER.info("✅ Phase 4 complete: wrote %d feature/condition analyses to %s", len(analyses), args.output)


if __name__ == "__main__":
    main()
