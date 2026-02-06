#!/usr/bin/env python3
"""
Phase 4: Improved word association analysis

This script consumes the Phase 1 activation stream (which includes response texts)
and computes condition-aware word statistics for each feature.  The output includes:
  • per-condition top tokens
  • Laplace-smoothed log-ratio (PMI-style) comparisons for all 6 condition pairs
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

import numpy as np
from tqdm.auto import tqdm

CONDITIONS = [
    "safe_baseline",
    "safe_with_safe_patch",
    "safe_with_risky_patch",
    "risky_baseline",
    "risky_with_risky_patch",
    "risky_with_safe_patch",
]

TOKEN_PATTERN = re.compile(
    r"(?:[$₩￦]?\d+(?:\.\d+)?)|[\uac00-\ud7a3]+|[a-zA-Z]+|[%$]"
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("phase4")


def tokenize(text: str) -> list[str]:
    tokens = TOKEN_PATTERN.findall(text)
    normalised = []
    for token in tokens:
        if token.isalpha():
            normalised.append(token.lower())
        else:
            normalised.append(token)
    return normalised


def update_counts(record: dict, token_counts, token_totals):
    feature = record["feature"]
    condition = record["condition"]
    response = record.get("response", "")
    if not response or condition not in CONDITIONS:
        return

    tokens = tokenize(response)
    if not tokens:
        return

    counter = token_counts[feature][condition]
    counter.update(tokens)
    token_totals[feature][condition] += len(tokens)


def laplace_log_ratio(count_a, total_a, count_b, total_b, vocab_size, alpha=1.0):
    # Apply Laplace smoothing to avoid division by zero
    p_a = (count_a + alpha) / (total_a + alpha * vocab_size)
    p_b = (count_b + alpha) / (total_b + alpha * vocab_size)
    return math.log(p_a / p_b)


def analyse_feature(feature: str, token_counts, token_totals):
    condition_stats = token_counts[feature]
    totals = token_totals[feature]

    # per-condition top tokens
    top_tokens = {
        cond: Counter(condition_stats[cond]).most_common(20)
        for cond in CONDITIONS
        if condition_stats[cond]
    }

    pairwise_results = {}
    for cond_a, cond_b in combinations(CONDITIONS, 2):
        counts_a = condition_stats[cond_a]
        counts_b = condition_stats[cond_b]
        total_a = totals.get(cond_a, 0)
        total_b = totals.get(cond_b, 0)
        if total_a == 0 or total_b == 0:
            continue

        vocab = set(counts_a.keys()) | set(counts_b.keys())
        vocab_size = len(vocab) if vocab else 1

        log_ratios = {}
        for token in vocab:
            count_a = counts_a.get(token, 0)
            count_b = counts_b.get(token, 0)
            log_ratios[token] = laplace_log_ratio(count_a, total_a, count_b, total_b, vocab_size)

        if not log_ratios:
            continue

        sorted_tokens = sorted(log_ratios.items(), key=lambda x: x[1], reverse=True)
        positive = [dict(word=w, score=float(s)) for w, s in sorted_tokens[:10] if s > 0]
        negative = [dict(word=w, score=float(s)) for w, s in sorted_tokens[-10:] if s < 0]

        pairwise_results[f"{cond_a}_vs_{cond_b}"] = {
            "top_positive": positive,
            "top_negative": negative,
        }

    return {
        "feature": feature,
        "layer": int(feature.split('-')[0][1:]) if '-' in feature else None,
        "condition_totals": totals,
        "top_tokens": top_tokens,
        "pairwise_log_ratio": pairwise_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 word analysis")
    parser.add_argument(
        "--phase1-dir",
        type=Path,
        default=Path("/data/llm_addiction/experiment_pathway_token_analysis/results/phase1_activations"),
        help="Directory containing Phase 1 activation files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/data/llm_addiction/experiment_pathway_token_analysis/results/phase4_word_analysis.json"),
        help="Output JSON containing PMI/word statistics",
    )
    args = parser.parse_args()

    token_counts = defaultdict(lambda: defaultdict(Counter))
    token_totals = defaultdict(lambda: defaultdict(int))

    layer_files = sorted(args.phase1_dir.glob("feature_activations_L*.jsonl"))
    if not layer_files:
        raise FileNotFoundError(f"No activation files found in {args.phase1_dir}")

    for layer_file in tqdm(layer_files, desc="Reading activations"):
        with open(layer_file, "r") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                update_counts(record, token_counts, token_totals)

    LOGGER.info("Collected token statistics for %d features", len(token_counts))

    analyses = []
    for feature in tqdm(token_counts.keys(), desc="Analysing features"):
        analyses.append(analyse_feature(feature, token_counts, token_totals))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(analyses, handle, indent=2, ensure_ascii=False)

    LOGGER.info("Phase 4 complete: wrote %d feature analyses to %s", len(analyses), args.output)


if __name__ == "__main__":
    main()
