#!/usr/bin/env python3
"""
Phase 3: Causal Direction Validation (for Patching Data)

Input: Phase 2 correlation results (JSONL)
Output: Causal direction analysis (JSONL)

Uses regression-based approach to determine directionality of feature relationships.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from scipy import stats
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("phase3")


@dataclass
class PatchingActivationSeries:
    feature: str
    target_feature: str
    condition: str
    trials: dict[int, float]

    @staticmethod
    def from_patching_file(
        patching_file: Path, target_feature: str, condition: str, feature: str
    ) -> "PatchingActivationSeries | None":
        """Load activation series for specific feature from patching data"""
        trials = {}

        with open(patching_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue

                record = json.loads(line)

                # Filter by target and condition
                if record.get('target_feature') != target_feature:
                    continue

                record_cond = f"{record.get('patch_condition')}_{record.get('prompt_type')}"
                if record_cond != condition:
                    continue

                trial = record.get('trial')
                if trial is None:
                    continue

                # Get activation for this feature
                all_features = record.get('all_features', {})
                if feature in all_features:
                    trials[trial] = all_features[feature]

        if not trials:
            return None

        return PatchingActivationSeries(
            feature=feature,
            target_feature=target_feature,
            condition=condition,
            trials=trials
        )


def align_trials(
    series_a: PatchingActivationSeries, series_b: PatchingActivationSeries, min_samples: int = 10
) -> Tuple[np.ndarray, np.ndarray, list[int]]:
    """Align trial data for two features"""
    shared_trials = sorted(set(series_a.trials.keys()) & set(series_b.trials.keys()))
    if len(shared_trials) < min_samples:
        return np.array([]), np.array([]), []

    a_vals = np.array([series_a.trials[t] for t in shared_trials])
    b_vals = np.array([series_b.trials[t] for t in shared_trials])
    return a_vals, b_vals, shared_trials


def regression_direction(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Return slope, rvalue, and stderr from linear regression"""
    result = stats.linregress(x, y)
    return result.slope, result.rvalue, result.stderr


def classify_direction(r2_ab: float, r2_ba: float, threshold: float = 0.05) -> str:
    """Classify directionality based on R² difference"""
    delta = r2_ab - r2_ba
    if abs(delta) < threshold:
        return "bidirectional"
    return "A→B" if delta > 0 else "B→A"


def evaluate_pair(
    series_a: PatchingActivationSeries, series_b: PatchingActivationSeries
) -> dict | None:
    """Evaluate causal direction for feature pair"""
    samples_a, samples_b, shared = align_trials(series_a, series_b)
    if not shared:
        return None

    # Normalize (z-score) for comparability
    try:
        norm_a = stats.zscore(samples_a)
        norm_b = stats.zscore(samples_b)
    except Exception:
        return None

    slope_ab, r_ab, stderr_ab = regression_direction(norm_a, norm_b)
    slope_ba, r_ba, stderr_ba = regression_direction(norm_b, norm_a)

    r2_ab = r_ab**2
    r2_ba = r_ba**2
    direction = classify_direction(r2_ab, r2_ba)

    return {
        "target_feature": series_a.target_feature,
        "condition": series_a.condition,
        "feature_A": series_a.feature,
        "feature_B": series_b.feature,
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
    parser = argparse.ArgumentParser(description="Phase 3 causal validation for patching data")
    parser.add_argument(
        "--correlation-file",
        type=Path,
        required=True,
        help="Phase 2 correlation results (JSONL)",
    )
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
        help="Output JSONL for causal validation results",
    )
    parser.add_argument("--max-pairs", type=int, default=None, help="Optional cap for debugging")
    args = parser.parse_args()

    if not args.correlation_file.exists():
        raise FileNotFoundError(f"Correlation file not found: {args.correlation_file}")

    if not args.patching_file.exists():
        raise FileNotFoundError(f"Patching file not found: {args.patching_file}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    written = 0

    with open(args.correlation_file, "r") as corr_handle, open(args.output, "w") as out_handle:
        for line in tqdm(corr_handle, desc="Validating pairs"):
            if not line.strip():
                continue

            pair = json.loads(line)
            target_feature = pair["target_feature"]
            condition = pair["condition"]
            feature_a = pair["feature_A"]
            feature_b = pair["feature_B"]

            # Load activation series from patching file
            series_a = PatchingActivationSeries.from_patching_file(
                args.patching_file, target_feature, condition, feature_a
            )
            series_b = PatchingActivationSeries.from_patching_file(
                args.patching_file, target_feature, condition, feature_b
            )

            if not series_a or not series_b:
                LOGGER.warning(f"Missing activation series for {feature_a} or {feature_b}")
                continue

            result = evaluate_pair(series_a, series_b)
            if result is None:
                continue

            out_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            written += 1
            processed += 1

            if args.max_pairs and processed >= args.max_pairs:
                LOGGER.info("Reached max_pairs=%d limit", args.max_pairs)
                break

    LOGGER.info("✅ Phase 3 complete: %d records written to %s", written, args.output)


if __name__ == "__main__":
    main()
