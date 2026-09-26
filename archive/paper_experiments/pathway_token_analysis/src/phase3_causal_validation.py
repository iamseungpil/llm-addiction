#!/usr/bin/env python3
"""
Phase 3: Selective causal validation

Given the high-correlation feature pairs produced in Phase 2, this script runs a
light-weight causal direction analysis that uses linear regression on aligned
activation sequences.  While this does not replace full gradient tracing +
path patching, it provides a reproducible automatic triage signal and records
the statistics required for manual follow-up.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from scipy import stats
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("phase3")


@dataclass
class ActivationSeries:
    feature: str
    layer: int
    trials: Dict[int, float]

    @staticmethod
    def from_records(records: Iterable[dict]) -> "ActivationSeries":
        records = list(records)
        if not records:
            raise ValueError("ActivationSeries requires at least one record")
        meta = records[0]
        trials = {rec["trial"]: rec["activation"] for rec in records if rec["trial"] is not None}
        return ActivationSeries(feature=meta["feature"], layer=meta["layer"], trials=trials)


def load_activation_series(activation_dir: Path, feature: str, condition: str) -> ActivationSeries | None:
    """
    Load activation records for the given feature/condition across all layer files.
    """
    records = []
    for layer_file in activation_dir.glob("feature_activations_L*.jsonl"):
        with open(layer_file, "r") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("feature") != feature or rec.get("condition") != condition:
                    continue
                records.append(rec)
    if not records:
        return None
    return ActivationSeries.from_records(records)


def align_trials(series_a: ActivationSeries, series_b: ActivationSeries, min_samples: int = 10) -> Tuple[np.ndarray, np.ndarray, list[int]]:
    shared_trials = sorted(set(series_a.trials.keys()) & set(series_b.trials.keys()))
    if len(shared_trials) < min_samples:
        return np.array([]), np.array([]), []
    a_vals = np.array([series_a.trials[t] for t in shared_trials])
    b_vals = np.array([series_b.trials[t] for t in shared_trials])
    return a_vals, b_vals, shared_trials


def regression_direction(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Return slope, rvalue, and stderr from linear regression."""
    result = stats.linregress(x, y)
    return result.slope, result.rvalue, result.stderr


def classify_direction(r2_ab: float, r2_ba: float, threshold: float = 0.05) -> str:
    delta = r2_ab - r2_ba
    if abs(delta) < threshold:
        return "bidirectional"
    return "A→B" if delta > 0 else "B→A"


def evaluate_pair(series_a: ActivationSeries, series_b: ActivationSeries, condition: str) -> dict | None:
    samples_a, samples_b, shared = align_trials(series_a, series_b)
    if not shared:
        return None

    # Normalize (z-score) for comparability
    norm_a = stats.zscore(samples_a)
    norm_b = stats.zscore(samples_b)

    slope_ab, r_ab, stderr_ab = regression_direction(norm_a, norm_b)
    slope_ba, r_ba, stderr_ba = regression_direction(norm_b, norm_a)

    r2_ab = r_ab**2
    r2_ba = r_ba**2
    direction = classify_direction(r2_ab, r2_ba)

    return {
        "feature_A": series_a.feature,
        "feature_B": series_b.feature,
        "layer_A": series_a.layer,
        "layer_B": series_b.layer,
        "condition": condition,
        "shared_trials": len(shared),
        "slope_A_to_B": float(slope_ab),
        "slope_B_to_A": float(slope_ba),
        "r_value_A_to_B": float(r_ab),
        "r_value_B_to_A": float(r_ba),
        "stderr_A_to_B": float(stderr_ab),
        "stderr_B_to_A": float(stderr_ba),
        "r_squared_A_to_B": float(r2_ab),
        "r_squared_B_to_A": float(r2_ba),
        "direction": direction,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 causal validation")
    parser.add_argument(
        "--correlation-file",
        type=Path,
        required=True,
        help="Path to high-correlation pairs generated in Phase 2 (JSONL)",
    )
    parser.add_argument(
        "--activation-dir",
        type=Path,
        default=Path("/data/llm_addiction/experiment_pathway_token_analysis/results/phase1_activations"),
        help="Directory containing Phase 1 activation JSONL files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/data/llm_addiction/experiment_pathway_token_analysis/results/phase3_causal_validation.jsonl"),
        help="Output JSONL for causal validation results",
    )
    parser.add_argument("--max-pairs", type=int, default=None, help="Optional cap for debugging")
    args = parser.parse_args()

    if not args.correlation_file.exists():
        raise FileNotFoundError(f"Correlation file not found: {args.correlation_file}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    written = 0
    with open(args.correlation_file, "r") as corr_handle, open(args.output, "w") as out_handle:
        for line in tqdm(corr_handle, desc="Validating pairs"):
            if not line.strip():
                continue
            pair = json.loads(line)
            condition = pair["condition"]
            feature_a = pair["feature_A"]
            feature_b = pair["feature_B"]

            series_a = load_activation_series(args.activation_dir, feature_a, condition)
            series_b = load_activation_series(args.activation_dir, feature_b, condition)

            if not series_a or not series_b:
                LOGGER.warning("Missing activation series for %s or %s", feature_a, feature_b)
                continue

            result = evaluate_pair(series_a, series_b, condition)
            if result is None:
                continue

            out_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            written += 1
            processed += 1

            if args.max_pairs and processed >= args.max_pairs:
                LOGGER.info("Reached max_pairs=%d limit", args.max_pairs)
                break

    LOGGER.info("Phase 3 complete: %d records written", written)


if __name__ == "__main__":
    main()
