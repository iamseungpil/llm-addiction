#!/usr/bin/env python3
"""Aggregate Experiment 2 response logs to compare safe vs risky feature effects."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
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
    records: List[Dict[str, object]] = []
    for path in paths:
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
    return df


def compute_condition_metrics(group: pd.DataFrame, condition: str) -> ConditionMetrics:
    stop_rate = (group["action"] == "stop").mean()
    avg_bet = group["bet"].mean()
    bankruptcy_rate: Optional[float] = None
    if condition.startswith("risky"):
        risky_bets = (group["bet"] > BANKRUPTCY_THRESHOLD).mean()
        bankruptcy_rate = float(risky_bets)
    return ConditionMetrics(stop_rate= float(stop_rate), avg_bet=float(avg_bet), bankruptcy_rate=bankruptcy_rate)


def build_feature_metrics(df: pd.DataFrame) -> Dict[Tuple[str, str], ConditionMetrics]:
    metrics: Dict[Tuple[str, str], ConditionMetrics] = {}
    for (feature, condition), group in df.groupby(["feature", "condition"]):
        metrics[(feature, condition)] = compute_condition_metrics(group, condition)
    return metrics


def safe_patch_effect(feature: str, metrics: Dict[Tuple[str, str], ConditionMetrics]) -> Tuple[Optional[float], Dict[str, Optional[float]]]:
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
        safe_delta = safe_condition.stop_rate - safe_baseline.stop_rate
    return safe_delta, risky_metrics


def risky_patch_effect(feature: str, metrics: Dict[Tuple[str, str], ConditionMetrics]) -> Tuple[Optional[float], Dict[str, Optional[float]]]:
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
        safe_delta = safe_condition.stop_rate - safe_baseline.stop_rate
    return safe_delta, risky_metrics


def _delta(value: Optional[float], baseline: Optional[float]) -> Optional[float]:
    if value is None or baseline is None:
        return None
    return value - baseline


def collect_feature_effects(metrics: Dict[Tuple[str, str], ConditionMetrics]) -> List[FeatureEffects]:
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
    safe_features: List[FeatureEffects] = []
    risky_features: List[FeatureEffects] = []
    for effect in effects:
        safe_stop = effect.safe_patch_stop_delta
        safe_bankruptcy = effect.safe_patch_risky_metrics["bankruptcy"]
        risky_stop = effect.risky_patch_stop_delta
        risky_bankruptcy = effect.risky_patch_risky_metrics["bankruptcy"]
        if safe_stop is not None and safe_bankruptcy is not None:
            if safe_stop > 0 and safe_bankruptcy < 0:
                safe_features.append(effect)
        if risky_stop is not None and risky_bankruptcy is not None:
            if risky_stop < 0 and risky_bankruptcy > 0:
                risky_features.append(effect)
    return safe_features, risky_features


def summarize_group(effects: List[FeatureEffects], use_safe_patch: bool) -> Dict[str, Dict[str, float]]:
    if not effects:
        return {}
    stop_deltas: List[float] = []
    bankruptcy_deltas: List[float] = []
    risky_stop_deltas: List[float] = []
    avg_bet_deltas: List[float] = []
    for effect in effects:
        if use_safe_patch:
            stop_delta = effect.safe_patch_stop_delta
            risky_metrics = effect.safe_patch_risky_metrics
        else:
            stop_delta = effect.risky_patch_stop_delta
            risky_metrics = effect.risky_patch_risky_metrics
        if stop_delta is not None:
            stop_deltas.append(stop_delta)
        if risky_metrics["bankruptcy"] is not None:
            bankruptcy_deltas.append(risky_metrics["bankruptcy"])
        if risky_metrics["stop"] is not None:
            risky_stop_deltas.append(risky_metrics["stop"])
        if risky_metrics["avg_bet"] is not None:
            avg_bet_deltas.append(risky_metrics["avg_bet"])
    def _mean_se(values: List[float]) -> Tuple[float, float]:
        arr = np.array(values, dtype=float)
        return float(arr.mean()), float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else (float(arr.mean()), 0.0)
    summary: Dict[str, Dict[str, float]] = {}
    if stop_deltas:
        mean, se = _mean_se(stop_deltas)
        summary["safe_stop_delta"] = {"mean": mean, "se": se}
    if bankruptcy_deltas:
        mean, se = _mean_se(bankruptcy_deltas)
        summary["risky_bankruptcy_delta"] = {"mean": mean, "se": se}
    if risky_stop_deltas:
        mean, se = _mean_se(risky_stop_deltas)
        summary["risky_stop_delta"] = {"mean": mean, "se": se}
    if avg_bet_deltas:
        mean, se = _mean_se(avg_bet_deltas)
        summary["risky_avg_bet_delta"] = {"mean": mean, "se": se}
    summary["count"] = {"mean": float(len(effects)), "se": 0.0}
    return summary


def format_percentage(value: float) -> str:
    return f"{value * 100:+.1f}%"


def render_figure(
    safe_summary: Dict[str, Dict[str, float]],
    risky_summary: Dict[str, Dict[str, float]],
    output_path: Path,
) -> None:
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(8, 5))

    categories = [
        "Stop Rate (Safe Context)",
        "Stop Rate (Risky Context)",
        "Bankruptcy Rate (Risky Context)",
    ]
    safe_values = [
        safe_summary.get("safe_stop_delta", {"mean": 0.0})["mean"],
        safe_summary.get("risky_stop_delta", {"mean": 0.0})["mean"],
        safe_summary.get("risky_bankruptcy_delta", {"mean": 0.0})["mean"],
    ]
    risky_values = [
        risky_summary.get("safe_stop_delta", {"mean": 0.0})["mean"],
        risky_summary.get("risky_stop_delta", {"mean": 0.0})["mean"],
        risky_summary.get("risky_bankruptcy_delta", {"mean": 0.0})["mean"],
    ]
    safe_errors = [
        safe_summary.get("safe_stop_delta", {"se": 0.0})["se"],
        safe_summary.get("risky_stop_delta", {"se": 0.0})["se"],
        safe_summary.get("risky_bankruptcy_delta", {"se": 0.0})["se"],
    ]
    risky_errors = [
        risky_summary.get("safe_stop_delta", {"se": 0.0})["se"],
        risky_summary.get("risky_stop_delta", {"se": 0.0})["se"],
        risky_summary.get("risky_bankruptcy_delta", {"se": 0.0})["se"],
    ]

    x = np.arange(len(categories))
    width = 0.32
    safe_bars = ax.bar(x - width / 2, safe_values, width, yerr=safe_errors,
                       color="#6ac36a", edgecolor="black", label="Safe Features")
    risky_bars = ax.bar(x + width / 2, risky_values, width, yerr=risky_errors,
                        color="#f57d7c", edgecolor="black", label="Risky Features")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=15, ha="right")
    ax.set_ylabel("Effect Size (Δ)")
    ax.set_title("Experiment 2 Feature Effects (Safe vs Risky Groups)")
    ax.axhline(0, color="black", linewidth=1)

    for bars, values in [(safe_bars, safe_values), (risky_bars, risky_values)]:
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.01 if value >= 0 else -0.01),
                format_percentage(value),
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=9,
            )

    ax.legend(loc="upper right")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_path.with_suffix('.png')
    pdf_path = output_path.with_suffix('.pdf')
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path, dpi=300)
    plt.close(fig)


def main() -> None:
    base_dir = Path("/data/llm_addiction/results")
    log_paths = sorted(base_dir.glob("exp2_response_log_4_*.json"))
    log_paths += sorted(base_dir.glob("exp2_response_log_5_*.json"))
    if not log_paths:
        raise FileNotFoundError("No response logs found matching exp2_response_log_[45]_*.json")
    print(f"Discovered {len(log_paths)} response logs.")
    df = load_logs(log_paths)

    print(f"Loaded {len(df):,} valid trials across {df['feature'].nunique()} unique features.")

    metrics = build_feature_metrics(df)
    effects = collect_feature_effects(metrics)
    safe_features, risky_features = classify_features(effects)
    risky_bankruptcy_increase = [
        effect.feature
        for effect in effects
        if effect.risky_patch_risky_metrics["bankruptcy"] is not None
        and effect.risky_patch_risky_metrics["bankruptcy"] > 0
    ]

    safe_summary = summarize_group(safe_features, use_safe_patch=True)
    risky_summary = summarize_group(risky_features, use_safe_patch=False)

    output_csv = Path("analysis/exp2_feature_group_summary.csv")
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

    summary_txt = Path("analysis/exp2_feature_group_summary.txt")
    with summary_txt.open("w") as f:
        f.write("Safe features (safe patch helpful):\n")
        for effect in safe_features:
            f.write(f"  - {effect.feature}\n")
        f.write("\nRisky features (risky patch harmful):\n")
        if risky_features:
            for effect in risky_features:
                f.write(f"  - {effect.feature}\n")
        else:
            f.write("  (none met: safe stop decreased AND risky bankruptcy increased)\n")
        f.write("\nFeatures with increased risky-context bankruptcy (regardless of stop change):\n")
        if risky_bankruptcy_increase:
            for feature_id in risky_bankruptcy_increase:
                f.write(f"  - {feature_id}\n")
        else:
            f.write("  (none)\n")
        f.write("\nSafe feature summary:\n")
        for key, stats in safe_summary.items():
            f.write(f"  {key}: mean={stats['mean']:.4f}, se={stats['se']:.4f}\n")
        f.write("\nRisky feature summary:\n")
        if risky_summary:
            for key, stats in risky_summary.items():
                f.write(f"  {key}: mean={stats['mean']:.4f}, se={stats['se']:.4f}\n")
        else:
            f.write("  (no features in group)\n")

    figure_base = Path("/home/ubuntu/llm_addiction/writing/figures/causal_patching_comparison")
    render_figure(safe_summary, risky_summary, figure_base)

    print("Safe features:", [effect.feature for effect in safe_features])
    print("Risky features:", [effect.feature for effect in risky_features])
    print("Bankruptcy rate increases (risky context):", risky_bankruptcy_increase)
    print("Summary CSV:", output_csv)
    print("Summary TXT:", summary_txt)
    print("Figure saved to:", f"{figure_base}.png and {figure_base}.pdf")


if __name__ == "__main__":
    main()
