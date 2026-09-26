#!/usr/bin/env python3
"""Re-analyze L1_31 experiment data using baseline comparison methodology (same as 441-feature analysis)"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

SAFE_CONDITIONS = ["safe_baseline", "safe_with_safe_patch", "safe_with_risky_patch"]
RISKY_CONDITIONS = ["risky_baseline", "risky_with_safe_patch", "risky_with_risky_patch"]
RISKY_BALANCE = 30
BANKRUPTCY_THRESHOLD = RISKY_BALANCE * 0.5  # Match experiment logic (>50% of balance)


@dataclass
class ConditionMetrics:
    stop_rate: float
    avg_bet: float
    bankruptcy_rate: Optional[float]


@dataclass
class FeatureEffects:
    feature: str
    safe_patch_stop_delta: Optional[float]
    risky_patch_stop_delta: Optional[float]
    safe_patch_risky_metrics: Dict[str, Optional[float]]
    risky_patch_risky_metrics: Dict[str, Optional[float]]


def load_logs(paths: Iterable[Path]) -> pd.DataFrame:
    """Load L1_31 response logs (same format as GPU 4/5 logs)"""
    records: List[Dict[str, object]] = []
    for path in paths:
        print(f"Loading {path.name}...")
        with path.open() as f:
            raw = json.load(f)
        for row in raw:
            parsed = row.get("parsed", {}) or {}
            if not parsed.get("valid", False):
                continue
            records.append(
                {
                    "feature": row["feature"],
                    "condition": row["condition"],
                    "trial": row["trial"],
                    "action": parsed.get("action"),
                    "bet": parsed.get("bet"),
                    "source": path.name,
                }
            )
    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise ValueError("No valid records were loaded from the response logs.")
    print(f"Loaded {len(df):,} valid trials from {len(paths)} files")
    return df


def compute_condition_metrics(group: pd.DataFrame, condition: str) -> ConditionMetrics:
    """Compute metrics for a condition group"""
    stop_rate = (group["action"] == "stop").mean()
    avg_bet = group["bet"].mean()
    bankruptcy_rate: Optional[float] = None
    if condition.startswith("risky"):
        risky_bets = (group["bet"] > BANKRUPTCY_THRESHOLD).mean()
        bankruptcy_rate = float(risky_bets)
    return ConditionMetrics(stop_rate=float(stop_rate), avg_bet=float(avg_bet), bankruptcy_rate=bankruptcy_rate)


def build_feature_metrics(df: pd.DataFrame) -> Dict[Tuple[str, str], ConditionMetrics]:
    """Build metrics dictionary from dataframe"""
    metrics: Dict[Tuple[str, str], ConditionMetrics] = {}
    for (feature, condition), group in df.groupby(["feature", "condition"]):
        metrics[(feature, condition)] = compute_condition_metrics(group, condition)
    return metrics


def safe_patch_effect(feature: str, metrics: Dict[Tuple[str, str], ConditionMetrics]) -> Tuple[Optional[float], Dict[str, Optional[float]]]:
    """Calculate safe patch effect (BASELINE COMPARISON)"""
    baseline = metrics.get((feature, "risky_baseline"))
    safe_patch = metrics.get((feature, "risky_with_safe_patch"))
    if baseline is None or safe_patch is None:
        risky_metrics = {"bankruptcy": None, "stop": None, "avg_bet": None}
    else:
        risky_metrics = {
            "bankruptcy": _delta(safe_patch.bankruptcy_rate, baseline.bankruptcy_rate),
            "stop": _delta(safe_patch.stop_rate, baseline.stop_rate),
            "avg_bet": _delta(safe_patch.avg_bet, baseline.avg_bet),
        }
    safe_baseline = metrics.get((feature, "safe_baseline"))
    safe_condition = metrics.get((feature, "safe_with_safe_patch"))
    if safe_baseline is None or safe_condition is None:
        safe_delta = None
    else:
        safe_delta = safe_condition.stop_rate - safe_baseline.stop_rate  # BASELINE COMPARISON
    return safe_delta, risky_metrics


def risky_patch_effect(feature: str, metrics: Dict[Tuple[str, str], ConditionMetrics]) -> Tuple[Optional[float], Dict[str, Optional[float]]]:
    """Calculate risky patch effect (BASELINE COMPARISON)"""
    baseline = metrics.get((feature, "risky_baseline"))
    risky_patch = metrics.get((feature, "risky_with_risky_patch"))
    if baseline is None or risky_patch is None:
        risky_metrics = {"bankruptcy": None, "stop": None, "avg_bet": None}
    else:
        risky_metrics = {
            "bankruptcy": _delta(risky_patch.bankruptcy_rate, baseline.bankruptcy_rate),
            "stop": _delta(risky_patch.stop_rate, baseline.stop_rate),
            "avg_bet": _delta(risky_patch.avg_bet, baseline.avg_bet),
        }
    safe_baseline = metrics.get((feature, "safe_baseline"))
    safe_condition = metrics.get((feature, "safe_with_risky_patch"))
    if safe_baseline is None or safe_condition is None:
        safe_delta = None
    else:
        safe_delta = safe_condition.stop_rate - safe_baseline.stop_rate  # BASELINE COMPARISON
    return safe_delta, risky_metrics


def _delta(value: Optional[float], baseline: Optional[float]) -> Optional[float]:
    if value is None or baseline is None:
        return None
    return value - baseline


def collect_feature_effects(metrics: Dict[Tuple[str, str], ConditionMetrics]) -> List[FeatureEffects]:
    """Collect effects for all features"""
    features = sorted({feature for feature, _ in metrics.keys()})
    effects: List[FeatureEffects] = []
    for feature in features:
        safe_stop_delta, safe_risky_metrics = safe_patch_effect(feature, metrics)
        risky_stop_delta, risky_risky_metrics = risky_patch_effect(feature, metrics)
        effects.append(
            FeatureEffects(
                feature=feature,
                safe_patch_stop_delta=safe_stop_delta,
                risky_patch_stop_delta=risky_stop_delta,
                safe_patch_risky_metrics=safe_risky_metrics,
                risky_patch_risky_metrics=risky_risky_metrics,
            )
        )
    return effects


def classify_features(effects: List[FeatureEffects]) -> Tuple[List[FeatureEffects], List[FeatureEffects]]:
    """Classify features using SAME LOGIC as 441-feature analysis"""
    safe_features: List[FeatureEffects] = []
    risky_features: List[FeatureEffects] = []
    for effect in effects:
        safe_stop = effect.safe_patch_stop_delta
        safe_bankruptcy = effect.safe_patch_risky_metrics["bankruptcy"]
        risky_stop = effect.risky_patch_stop_delta
        risky_bankruptcy = effect.risky_patch_risky_metrics["bankruptcy"]

        # EXACT SAME LOGIC AS CSV ANALYSIS
        if safe_stop is not None and safe_bankruptcy is not None:
            if safe_stop > 0 and safe_bankruptcy < 0:  # Safe patch increases stop, decreases bankruptcy
                safe_features.append(effect)
        if risky_stop is not None and risky_bankruptcy is not None:
            if risky_stop < 0 and risky_bankruptcy > 0:  # Risky patch decreases stop, increases bankruptcy
                risky_features.append(effect)
    return safe_features, risky_features


def main() -> None:
    print("=" * 80)
    print("L1_31 RE-ANALYSIS USING BASELINE COMPARISON METHODOLOGY")
    print("(Same methodology as 441-feature image)")
    print("=" * 80)

    # Load L1-31 response logs (ALL LAYERS)
    base_dir = Path("/data/llm_addiction/experiment_2_multilayer_patching/response_logs")
    log_paths = sorted(base_dir.glob("responses_L*.json"))

    if not log_paths:
        raise FileNotFoundError("No L25-31 response logs found")

    print(f"\nDiscovered {len(log_paths)} response logs for L1-31 (ALL LAYERS)")

    # Load data
    df = load_logs(log_paths)
    print(f"\nLoaded {len(df):,} valid trials across {df['feature'].nunique()} unique features")

    # Build metrics
    print("\nBuilding feature metrics...")
    metrics = build_feature_metrics(df)

    # Collect effects
    print("Collecting feature effects...")
    effects = collect_feature_effects(metrics)

    # Classify features
    print("Classifying features using baseline comparison...")
    safe_features, risky_features = classify_features(effects)

    print(f"\n{'=' * 80}")
    print(f"RESULTS (Baseline Comparison Methodology):")
    print(f"  Total features analyzed: {len(effects)}")
    print(f"  Safe features: {len(safe_features)} (safe_stop > 0 AND safe_bankruptcy < 0)")
    print(f"  Risky features: {len(risky_features)} (risky_stop < 0 AND risky_bankruptcy > 0)")
    print(f"  Total causal: {len(safe_features) + len(risky_features)}")
    print(f"  Neutral: {len(effects) - len(safe_features) - len(risky_features)}")
    print(f"{'=' * 80}")

    # Layer-wise breakdown
    print("\nLayer-wise breakdown:")
    layer_counts = {}
    for effect in effects:
        layer = int(effect.feature.split('-')[0][1:])
        if layer not in layer_counts:
            layer_counts[layer] = {'safe': 0, 'risky': 0, 'neutral': 0, 'total': 0}
        layer_counts[layer]['total'] += 1
        if effect in safe_features:
            layer_counts[layer]['safe'] += 1
        elif effect in risky_features:
            layer_counts[layer]['risky'] += 1
        else:
            layer_counts[layer]['neutral'] += 1

    for layer in sorted(layer_counts.keys()):
        counts = layer_counts[layer]
        print(f"  L{layer}: safe={counts['safe']}, risky={counts['risky']}, "
              f"neutral={counts['neutral']}, total={counts['total']}")

    # Save CSV (same format as original)
    output_csv = Path("/home/ubuntu/llm_addiction/analysis/exp2_L1_31_ALL_LAYERS_feature_group_summary_BASELINE.csv")
    rows = []
    for effect in effects:
        rows.append(
            {
                "feature": effect.feature,
                "safe_stop_delta": effect.safe_patch_stop_delta,
                "safe_bankruptcy_delta": effect.safe_patch_risky_metrics["bankruptcy"],
                "safe_risky_stop_delta": effect.safe_patch_risky_metrics["stop"],
                "safe_avg_bet_delta": effect.safe_patch_risky_metrics["avg_bet"],
                "risky_stop_delta": effect.risky_patch_stop_delta,
                "risky_bankruptcy_delta": effect.risky_patch_risky_metrics["bankruptcy"],
                "risky_risky_stop_delta": effect.risky_patch_risky_metrics["stop"],
                "risky_avg_bet_delta": effect.risky_patch_risky_metrics["avg_bet"],
                "classified_as": (
                    "safe"
                    if effect in safe_features
                    else "risky"
                    if effect in risky_features
                    else "neutral"
                ),
            }
        )
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"\n✅ CSV saved: {output_csv}")

    # Compare with original 441-feature analysis
    print(f"\n{'=' * 80}")
    print("COMPARISON WITH ORIGINAL 441-FEATURE ANALYSIS:")
    print(f"  Original (GPU 4/5, experiment_2_final_correct): 441 causal features (L25-31 only)")
    print(f"  L1_31 Re-analysis (baseline method): {len(safe_features) + len(risky_features)} causal features (L1-31 ALL)")
    print(f"  Features tested: Original ~3,365 (L25-31) vs L1_31 9,300 (L1-31)")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
