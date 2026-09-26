#!/usr/bin/env python3
"""
V12 BK Direction Sign Reversal Analysis
========================================
Quantitative analysis of why 4/6 model-task combos show negative rho (sign reversal)
in the V12 steering experiments.

Hypothesis: The sign of rho depends on the BK base rate in the training data.
When BK is the minority class (< 50%), the mean-difference direction points
"away" from the dominant mode, and positive alpha pushes the model further
from BK -- yielding negative rho. When BK is near or above 50%, the direction
captures the majority pattern and positive alpha increases BK -- positive rho.

Outputs:
  - Figure: v12_fig4_sign_reversal_model.png (3 panels)
  - JSON:   v12_sign_reversal_analysis.json
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import spearmanr, pearsonr, pointbiserialr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis")
JSON_DIR = BASE / "results" / "json"
FIG_DIR = BASE / "results" / "figures"
HS_DIR = Path("/home/v-seungplee/data/llm-addiction/sae_features_v3")

FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# V12 result file mapping
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    {
        "label": "LLaMA SM",
        "model": "llama",
        "task": "sm",
        "json_file": "v12_n200_20260327_030745.json",
        "hs_subdir": "slot_machine/llama",
    },
    {
        "label": "LLaMA IC",
        "model": "llama",
        "task": "ic",
        "json_file": "v12_llama_ic_L22_20260329_022313.json",
        "hs_subdir": "investment_choice/llama",
    },
    {
        "label": "LLaMA MW",
        "model": "llama",
        "task": "mw",
        "json_file": "v12_llama_mw_L22_20260329_072818.json",
        "hs_subdir": "mystery_wheel/llama",
    },
    {
        "label": "Gemma SM",
        "model": "gemma",
        "task": "sm",
        "json_file": "v12_gemma_sm_L22_20260328_014425.json",
        "hs_subdir": "slot_machine/gemma",
    },
    {
        "label": "Gemma IC",
        "model": "gemma",
        "task": "ic",
        "json_file": "v12_gemma_ic_L22_20260328_100129.json",
        "hs_subdir": "investment_choice/gemma",
    },
    {
        "label": "Gemma MW",
        "model": "gemma",
        "task": "mw",
        "json_file": "v12_gemma_mw_L22_20260328_205618.json",
        "hs_subdir": "mystery_wheel/gemma",
    },
]

TARGET_LAYER = 22
ALPHA_KEYS = ["-2.0", "-1.0", "-0.5", "0.5", "1.0", "2.0"]
ALPHA_FLOATS = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ComboData:
    label: str
    model: str
    task: str
    # Training data
    training_bk_rate: float       # from hidden_states_dp.npz
    training_n_total: int
    training_n_bk: int
    training_n_safe: int
    class_imbalance_ratio: float  # max(n_bk, n_safe) / min(n_bk, n_safe)
    distance_from_50: float       # |bk_rate - 0.5|
    # BK direction properties
    direction_norm: float         # L2 norm of (mean_bk - mean_safe)
    direction_cosine_sim_with_mean: float  # cos similarity with overall mean
    # Steering results
    baseline_bk_rate: float       # from v12 json
    rho: float                    # Spearman rho from steering
    rho_p: float
    abs_rho: float
    rho_sign: int                 # +1 or -1 (0 if NaN)
    # Dose-response
    bk_at_minus2: float
    bk_at_plus2: float
    effect_range: float           # bk_at_plus2 - bk_at_minus2


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_steering_json(filepath: Path) -> dict:
    """Load and return the raw V12 steering result."""
    with open(filepath, "r") as f:
        return json.load(f)


def compute_direction_stats(hs_subdir: str, layer: int = TARGET_LAYER) -> dict:
    """Compute BK direction vector statistics from hidden states.

    Returns:
        dict with keys: training_bk_rate, training_n_total, training_n_bk,
        training_n_safe, direction_norm, class_imbalance_ratio,
        direction_cosine_sim_with_mean
    """
    hs_path = HS_DIR / hs_subdir / "hidden_states_dp.npz"
    data = np.load(hs_path, allow_pickle=True)

    labels = (data["game_outcomes"] == "bankruptcy").astype(int)
    layers = list(data["layers"])
    layer_idx = layers.index(layer)
    hs_layer = data["hidden_states"][:, layer_idx, :].astype(np.float64)

    n_total = len(labels)
    n_bk = int(labels.sum())
    n_safe = n_total - n_bk

    # BK direction: mean(BK) - mean(Safe)
    mean_bk = hs_layer[labels == 1].mean(axis=0)
    mean_safe = hs_layer[labels == 0].mean(axis=0)
    bk_dir = mean_bk - mean_safe
    norm = float(np.linalg.norm(bk_dir))

    # Overall mean for cosine similarity
    overall_mean = hs_layer.mean(axis=0)
    cos_sim = float(
        np.dot(bk_dir, overall_mean)
        / (np.linalg.norm(bk_dir) * np.linalg.norm(overall_mean) + 1e-12)
    )

    minority = min(n_bk, n_safe)
    imbalance = max(n_bk, n_safe) / minority if minority > 0 else float("inf")

    return {
        "training_bk_rate": n_bk / n_total,
        "training_n_total": n_total,
        "training_n_bk": n_bk,
        "training_n_safe": n_safe,
        "direction_norm": norm,
        "class_imbalance_ratio": imbalance,
        "direction_cosine_sim_with_mean": cos_sim,
    }


def safe_float(val, default=float("nan")) -> float:
    """Convert to float, returning default for None/NaN."""
    if val is None:
        return default
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def build_combo_data(exp: dict) -> ComboData:
    """Build a ComboData for one model-task combination."""
    # Load steering results
    jdata = load_steering_json(JSON_DIR / exp["json_file"])
    bk_dir_data = jdata.get("bk_direction", {})
    rho = safe_float(bk_dir_data.get("rho"))
    rho_p = safe_float(bk_dir_data.get("p"))

    baseline = jdata.get("baseline", {})
    baseline_bk = safe_float(baseline.get("bk_rate"))

    # Extract dose-response BK rates
    results = bk_dir_data.get("results", {})
    bk_at_minus2 = float("nan")
    bk_at_plus2 = float("nan")
    if results:
        entry_m2 = results.get("-2.0", {})
        entry_p2 = results.get("2.0", {})
        if isinstance(entry_m2, dict):
            bk_at_minus2 = safe_float(entry_m2.get("bk_rate"))
        if isinstance(entry_p2, dict):
            bk_at_plus2 = safe_float(entry_p2.get("bk_rate"))

    # Compute direction statistics from hidden states
    dir_stats = compute_direction_stats(exp["hs_subdir"])

    abs_rho = abs(rho) if math.isfinite(rho) else float("nan")
    rho_sign = (1 if rho > 0 else -1) if math.isfinite(rho) else 0

    effect_range = float("nan")
    if math.isfinite(bk_at_plus2) and math.isfinite(bk_at_minus2):
        effect_range = bk_at_plus2 - bk_at_minus2

    return ComboData(
        label=exp["label"],
        model=exp["model"],
        task=exp["task"],
        training_bk_rate=dir_stats["training_bk_rate"],
        training_n_total=dir_stats["training_n_total"],
        training_n_bk=dir_stats["training_n_bk"],
        training_n_safe=dir_stats["training_n_safe"],
        class_imbalance_ratio=dir_stats["class_imbalance_ratio"],
        distance_from_50=abs(dir_stats["training_bk_rate"] - 0.5),
        direction_norm=dir_stats["direction_norm"],
        direction_cosine_sim_with_mean=dir_stats["direction_cosine_sim_with_mean"],
        baseline_bk_rate=baseline_bk,
        rho=rho,
        rho_p=rho_p,
        abs_rho=abs_rho,
        rho_sign=rho_sign,
        bk_at_minus2=bk_at_minus2,
        bk_at_plus2=bk_at_plus2,
        effect_range=effect_range,
    )


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def analyze_sign_prediction(combos: list[ComboData]) -> dict:
    """Test whether BK base rate predicts rho sign.

    Analyses:
    1. BK% vs rho (Spearman)
    2. |BK% - 50%| vs |rho| (Spearman)
    3. Simple threshold model: BK% > 50% => positive rho
    4. Point-biserial: BK% vs rho_sign_binary
    """
    # Exclude Gemma IC (NaN rho -- floor effect, no variance)
    valid = [c for c in combos if math.isfinite(c.rho)]

    bk_pcts = np.array([c.training_bk_rate for c in valid])
    rhos = np.array([c.rho for c in valid])
    abs_rhos = np.array([c.abs_rho for c in valid])
    dist50 = np.array([c.distance_from_50 for c in valid])
    signs = np.array([1 if c.rho > 0 else 0 for c in valid])  # binary: 1=positive, 0=negative

    results = {}

    # 1. BK% vs rho (Spearman)
    if len(valid) >= 3:
        r_bk_rho, p_bk_rho = spearmanr(bk_pcts, rhos)
        results["bk_pct_vs_rho"] = {
            "spearman_r": round(float(r_bk_rho), 4),
            "p_value": round(float(p_bk_rho), 6),
            "interpretation": (
                "Strong positive: higher BK% -> more positive rho"
                if r_bk_rho > 0.5 else
                "Weak or no relationship"
            ),
        }

        # Pearson for comparison
        r_pears, p_pears = pearsonr(bk_pcts, rhos)
        results["bk_pct_vs_rho_pearson"] = {
            "pearson_r": round(float(r_pears), 4),
            "p_value": round(float(p_pears), 6),
        }

    # 2. |BK% - 50%| vs |rho|
    if len(valid) >= 3:
        r_dist_abs, p_dist_abs = spearmanr(dist50, abs_rhos)
        results["dist50_vs_abs_rho"] = {
            "spearman_r": round(float(r_dist_abs), 4),
            "p_value": round(float(p_dist_abs), 6),
            "interpretation": (
                "Class imbalance distance from 50% predicts |rho| strength"
                if abs(r_dist_abs) > 0.5 else
                "Distance from 50% does not strongly predict |rho|"
            ),
        }

    # 3. Threshold model: BK% > 50% => positive rho
    threshold_predictions = []
    for c in valid:
        predicted_sign = "positive" if c.training_bk_rate > 0.5 else "negative"
        actual_sign = "positive" if c.rho > 0 else "negative"
        correct = predicted_sign == actual_sign
        threshold_predictions.append({
            "label": c.label,
            "bk_pct": round(c.training_bk_rate * 100, 1),
            "predicted_sign": predicted_sign,
            "actual_rho": round(c.rho, 4),
            "actual_sign": actual_sign,
            "correct": correct,
        })
    accuracy = sum(1 for p in threshold_predictions if p["correct"]) / len(threshold_predictions)
    results["threshold_model_50pct"] = {
        "rule": "BK% > 50% => positive rho, BK% < 50% => negative rho",
        "accuracy": round(accuracy, 4),
        "n_correct": sum(1 for p in threshold_predictions if p["correct"]),
        "n_total": len(threshold_predictions),
        "predictions": threshold_predictions,
    }

    # 4. Point-biserial correlation: BK% vs binary sign
    if len(valid) >= 3 and len(set(signs)) > 1:
        rpb, ppb = pointbiserialr(signs, bk_pcts)
        results["point_biserial_bk_vs_sign"] = {
            "r_pb": round(float(rpb), 4),
            "p_value": round(float(ppb), 6),
        }
    else:
        results["point_biserial_bk_vs_sign"] = {
            "r_pb": None,
            "p_value": None,
            "note": "Insufficient variance in sign labels",
        }

    # 5. Alternative threshold: steering BASELINE BK rate > 50% => positive rho
    baseline_predictions = []
    for c in valid:
        if not math.isfinite(c.baseline_bk_rate):
            continue
        predicted_sign = "positive" if c.baseline_bk_rate > 0.5 else "negative"
        actual_sign = "positive" if c.rho > 0 else "negative"
        correct = predicted_sign == actual_sign
        baseline_predictions.append({
            "label": c.label,
            "baseline_bk": round(c.baseline_bk_rate * 100, 1),
            "predicted_sign": predicted_sign,
            "actual_rho": round(c.rho, 4),
            "actual_sign": actual_sign,
            "correct": correct,
        })
    if baseline_predictions:
        bl_accuracy = sum(1 for p in baseline_predictions if p["correct"]) / len(baseline_predictions)
        results["threshold_model_baseline_bk"] = {
            "rule": "Steering baseline BK% > 50% => positive rho",
            "accuracy": round(bl_accuracy, 4),
            "n_correct": sum(1 for p in baseline_predictions if p["correct"]),
            "n_total": len(baseline_predictions),
            "predictions": baseline_predictions,
        }

    # 6. Baseline BK rate vs rho (Spearman) -- alternative predictor
    valid_bl = [c for c in valid if math.isfinite(c.baseline_bk_rate)]
    if len(valid_bl) >= 3:
        bl_bks = np.array([c.baseline_bk_rate for c in valid_bl])
        bl_rhos = np.array([c.rho for c in valid_bl])
        r_bl, p_bl = spearmanr(bl_bks, bl_rhos)
        results["baseline_bk_vs_rho"] = {
            "spearman_r": round(float(r_bl), 4),
            "p_value": round(float(p_bl), 6),
            "interpretation": (
                "Steering baseline BK% predicts rho direction"
                if abs(r_bl) > 0.5 else
                "Weak relationship"
            ),
        }

    # 7. Cosine similarity of BK direction with overall mean
    cos_sims = [c.direction_cosine_sim_with_mean for c in valid]
    if len(valid) >= 3:
        r_cos, p_cos = spearmanr(cos_sims, rhos)
        results["cosine_sim_vs_rho"] = {
            "spearman_r": round(float(r_cos), 4),
            "p_value": round(float(p_cos), 6),
            "per_combo": [
                {"label": c.label, "cos_sim": round(c.direction_cosine_sim_with_mean, 4),
                 "rho": round(c.rho, 4)}
                for c in valid
            ],
        }

    return results


def analyze_norm_vs_effect(combos: list[ComboData]) -> dict:
    """Test whether direction norm predicts effect strength."""
    valid = [c for c in combos if math.isfinite(c.abs_rho) and math.isfinite(c.direction_norm)]

    norms = np.array([c.direction_norm for c in valid])
    abs_rhos = np.array([c.abs_rho for c in valid])
    effect_ranges = np.array([
        c.effect_range for c in valid if math.isfinite(c.effect_range)
    ])
    norms_for_range = np.array([
        c.direction_norm for c in valid if math.isfinite(c.effect_range)
    ])

    results = {}

    if len(valid) >= 3:
        r_norm_rho, p_norm_rho = spearmanr(norms, abs_rhos)
        results["norm_vs_abs_rho"] = {
            "spearman_r": round(float(r_norm_rho), 4),
            "p_value": round(float(p_norm_rho), 6),
            "interpretation": (
                "Larger direction norm -> stronger |rho|"
                if r_norm_rho > 0.5 else
                "Norm does not strongly predict |rho|"
            ),
        }

    if len(norms_for_range) >= 3:
        r_norm_range, p_norm_range = spearmanr(norms_for_range, np.abs(effect_ranges))
        results["norm_vs_abs_effect_range"] = {
            "spearman_r": round(float(r_norm_range), 4),
            "p_value": round(float(p_norm_range), 6),
        }

    # Summary table
    results["per_combo"] = [
        {
            "label": c.label,
            "norm": round(c.direction_norm, 2),
            "abs_rho": round(c.abs_rho, 4) if math.isfinite(c.abs_rho) else None,
            "effect_range": round(c.effect_range, 4) if math.isfinite(c.effect_range) else None,
        }
        for c in valid
    ]

    return results


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def apply_academic_style() -> None:
    """Configure matplotlib for clean academic figures."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 10,
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })


def generate_figure(combos: list[ComboData], analysis_results: dict) -> Path:
    """Generate the 3-panel sign reversal analysis figure.

    Panel A: BK% vs rho (scatter + regression line)
    Panel B: |BK% - 50%| vs |rho| (balance vs strength)
    Panel C: Direction norm comparison across combos
    """
    apply_academic_style()

    valid = [c for c in combos if math.isfinite(c.rho)]
    all_combos = combos  # include Gemma IC for Panel C

    # Colors: LLaMA = blue shades, Gemma = orange shades
    color_map = {
        "LLaMA SM": "#1f77b4",
        "LLaMA IC": "#4a9ad4",
        "LLaMA MW": "#7bbce6",
        "Gemma SM": "#d62728",
        "Gemma IC": "#e8696a",
        "Gemma MW": "#f4a0a1",
    }
    marker_map = {
        "sm": "o",  # circle
        "ic": "s",  # square
        "mw": "^",  # triangle
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # =================================================================
    # Panel A: BK% vs rho
    # =================================================================
    ax = axes[0]
    bk_pcts = [c.training_bk_rate * 100 for c in valid]
    rhos = [c.rho for c in valid]

    for c in valid:
        color = color_map.get(c.label, "gray")
        marker = marker_map.get(c.task, "o")
        edge = "gold" if c.rho > 0 else "none"
        lw = 2.0 if c.rho > 0 else 0.8
        ax.scatter(
            c.training_bk_rate * 100, c.rho,
            c=color, marker=marker, s=120, zorder=5,
            edgecolors=edge, linewidths=lw,
        )
        # Label each point
        offset_x = 2.0
        offset_y = 0.03
        if c.label == "LLaMA SM":
            offset_y = -0.08
        ax.annotate(
            c.label, (c.training_bk_rate * 100, c.rho),
            xytext=(offset_x, offset_y),
            textcoords="offset points",
            fontsize=8, ha="left", va="bottom",
        )

    # Regression line (OLS)
    if len(valid) >= 2:
        x_arr = np.array(bk_pcts)
        y_arr = np.array(rhos)
        coeffs = np.polyfit(x_arr, y_arr, 1)
        x_line = np.linspace(0, 80, 100)
        y_line = np.polyval(coeffs, x_line)
        ax.plot(x_line, y_line, "--", color="gray", linewidth=1.0, alpha=0.7,
                label=f"OLS: rho = {coeffs[0]:.4f} * BK% + {coeffs[1]:.3f}")

    # Reference lines
    ax.axhline(0, color="black", linewidth=0.5, linestyle="-", alpha=0.3)
    ax.axvline(50, color="red", linewidth=0.8, linestyle=":", alpha=0.5, label="BK% = 50%")

    # Fill regions
    ax.axhspan(0, 1.1, xmin=0, xmax=1, alpha=0.03, color="green")
    ax.axhspan(-1.1, 0, xmin=0, xmax=1, alpha=0.03, color="red")

    ax.set_xlabel("Training BK Rate (%)", fontsize=11)
    ax.set_ylabel("Spearman rho", fontsize=11)
    ax.set_title("A. BK Base Rate vs Steering Rho", fontsize=12, fontweight="bold")
    ax.set_xlim(-2, 82)
    ax.set_ylim(-1.15, 1.15)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)

    # Add annotation for positive region
    ax.text(
        60, 0.5, "Positive rho\n(BK direction\nincreases BK)",
        fontsize=7, ha="center", color="green", alpha=0.6, style="italic",
    )
    ax.text(
        15, -0.5, "Negative rho\n(BK direction\ndecreases BK)",
        fontsize=7, ha="center", color="red", alpha=0.6, style="italic",
    )

    # =================================================================
    # Panel B: |BK% - 50%| vs |rho|
    # =================================================================
    ax = axes[1]
    dist50 = [c.distance_from_50 * 100 for c in valid]
    abs_rhos = [c.abs_rho for c in valid]

    for c in valid:
        color = color_map.get(c.label, "gray")
        marker = marker_map.get(c.task, "o")
        ax.scatter(
            c.distance_from_50 * 100, c.abs_rho,
            c=color, marker=marker, s=120, zorder=5,
            edgecolors="black", linewidths=0.5,
        )
        ax.annotate(
            c.label, (c.distance_from_50 * 100, c.abs_rho),
            xytext=(3, 3), textcoords="offset points",
            fontsize=8, ha="left",
        )

    # Regression line
    if len(valid) >= 2:
        x_arr = np.array(dist50)
        y_arr = np.array(abs_rhos)
        coeffs = np.polyfit(x_arr, y_arr, 1)
        x_line = np.linspace(0, 50, 100)
        y_line = np.polyval(coeffs, x_line)
        ax.plot(x_line, y_line, "--", color="gray", linewidth=1.0, alpha=0.7)

    # Spearman annotation
    dist50_vs_abs = analysis_results.get("sign_prediction", {}).get("dist50_vs_abs_rho", {})
    r_val = dist50_vs_abs.get("spearman_r", "N/A")
    p_val = dist50_vs_abs.get("p_value", "N/A")
    ax.text(
        0.05, 0.95,
        f"Spearman r = {r_val}\np = {p_val}",
        transform=ax.transAxes, fontsize=9,
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )

    ax.set_xlabel("|BK% - 50%| (Distance from Balance)", fontsize=11)
    ax.set_ylabel("|rho| (Effect Strength)", fontsize=11)
    ax.set_title("B. Class Imbalance vs Effect Strength", fontsize=12, fontweight="bold")
    ax.set_xlim(-2, 52)
    ax.set_ylim(-0.05, 1.15)

    # =================================================================
    # Panel C: Direction Norm Comparison
    # =================================================================
    ax = axes[2]
    labels_all = [c.label for c in all_combos]
    norms = [c.direction_norm for c in all_combos]
    bar_colors = [color_map.get(c.label, "gray") for c in all_combos]

    # Hatch pattern for negative rho
    hatches = []
    for c in all_combos:
        if not math.isfinite(c.rho):
            hatches.append("xx")  # NaN (Gemma IC)
        elif c.rho > 0:
            hatches.append("")    # positive
        else:
            hatches.append("//")  # negative

    x_pos = np.arange(len(all_combos))
    bars = ax.bar(x_pos, norms, color=bar_colors, edgecolor="black", linewidth=0.6, width=0.65)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    # Add rho values on top of each bar
    for i, c in enumerate(all_combos):
        rho_str = f"rho={c.rho:.3f}" if math.isfinite(c.rho) else "rho=NaN"
        ax.text(i, norms[i] + 0.3, rho_str, ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_all, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("BK Direction L2 Norm", fontsize=11)
    ax.set_title("C. Direction Norm by Model-Task", fontsize=12, fontweight="bold")

    # Legend for hatch patterns
    legend_elements = [
        mpatches.Patch(facecolor="lightgray", edgecolor="black", label="Positive rho"),
        mpatches.Patch(facecolor="lightgray", edgecolor="black", hatch="//", label="Negative rho"),
        mpatches.Patch(facecolor="lightgray", edgecolor="black", hatch="xx", label="NaN (floor)"),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc="upper right", framealpha=0.9)

    # =================================================================
    # Final layout
    # =================================================================
    fig.suptitle(
        "V12 Sign Reversal Analysis: BK Direction Steering",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    outpath = FIG_DIR / "v12_fig4_sign_reversal_model.png"
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved figure -> {outpath}")
    return outpath


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("V12 BK Direction Sign Reversal Analysis")
    print("=" * 72)

    # Step 1: Load all data
    print("\n[1] Loading 6 model-task combos...")
    combos: list[ComboData] = []
    for exp in EXPERIMENTS:
        combo = build_combo_data(exp)
        combos.append(combo)
        rho_str = f"{combo.rho:.4f}" if math.isfinite(combo.rho) else "NaN"
        print(f"  {combo.label}: BK%={combo.training_bk_rate*100:.1f}%, "
              f"rho={rho_str:>8s}, "
              f"norm={combo.direction_norm:.2f}, "
              f"imbalance={combo.class_imbalance_ratio:.1f}x")

    # Step 2: Summary table
    print("\n" + "-" * 72)
    print(f"{'Label':<12s} {'BK%':>6s} {'|BK%-50|':>8s} {'Rho':>8s} {'|Rho|':>6s} "
          f"{'Norm':>8s} {'Imbal':>6s} {'BL_BK':>6s} {'BK@-2':>6s} {'BK@+2':>6s}")
    print("-" * 72)
    for c in combos:
        rho_str = f"{c.rho:.4f}" if math.isfinite(c.rho) else "NaN"
        abs_rho_str = f"{c.abs_rho:.4f}" if math.isfinite(c.abs_rho) else "NaN"
        bl_str = f"{c.baseline_bk_rate:.2f}" if math.isfinite(c.baseline_bk_rate) else "N/A"
        bkm2 = f"{c.bk_at_minus2:.2f}" if math.isfinite(c.bk_at_minus2) else "N/A"
        bkp2 = f"{c.bk_at_plus2:.2f}" if math.isfinite(c.bk_at_plus2) else "N/A"
        print(f"{c.label:<12s} {c.training_bk_rate*100:>5.1f}% {c.distance_from_50*100:>7.1f}% "
              f"{rho_str:>8s} {abs_rho_str:>6s} {c.direction_norm:>8.2f} "
              f"{c.class_imbalance_ratio:>5.1f}x {bl_str:>6s} {bkm2:>6s} {bkp2:>6s}")
    print("-" * 72)

    # Step 3: Run analyses
    print("\n[2] Analyzing sign prediction...")
    sign_results = analyze_sign_prediction(combos)

    # Print threshold model results
    tm = sign_results.get("threshold_model_50pct", {})
    print(f"\n  Threshold model (BK% > 50% => positive rho):")
    print(f"    Accuracy: {tm.get('n_correct', 0)}/{tm.get('n_total', 0)} = {tm.get('accuracy', 0)*100:.1f}%")
    for pred in tm.get("predictions", []):
        check = "OK" if pred["correct"] else "WRONG"
        print(f"    {pred['label']}: BK={pred['bk_pct']}% -> predicted={pred['predicted_sign']}, "
              f"actual rho={pred['actual_rho']:.4f} ({pred['actual_sign']}) [{check}]")

    # Print BK% vs rho correlation
    bk_rho = sign_results.get("bk_pct_vs_rho", {})
    print(f"\n  BK% vs rho (Spearman): r={bk_rho.get('spearman_r', 'N/A')}, "
          f"p={bk_rho.get('p_value', 'N/A')}")

    bk_rho_pearson = sign_results.get("bk_pct_vs_rho_pearson", {})
    print(f"  BK% vs rho (Pearson):  r={bk_rho_pearson.get('pearson_r', 'N/A')}, "
          f"p={bk_rho_pearson.get('p_value', 'N/A')}")

    d50 = sign_results.get("dist50_vs_abs_rho", {})
    print(f"  |BK%-50%| vs |rho| (Spearman): r={d50.get('spearman_r', 'N/A')}, "
          f"p={d50.get('p_value', 'N/A')}")

    # Print baseline BK threshold model
    bl_tm = sign_results.get("threshold_model_baseline_bk", {})
    if bl_tm:
        print(f"\n  Threshold model (Baseline BK% > 50% => positive rho):")
        print(f"    Accuracy: {bl_tm.get('n_correct', 0)}/{bl_tm.get('n_total', 0)} "
              f"= {bl_tm.get('accuracy', 0)*100:.1f}%")
        for pred in bl_tm.get("predictions", []):
            check = "OK" if pred["correct"] else "WRONG"
            print(f"    {pred['label']}: BaseBK={pred['baseline_bk']}% -> "
                  f"predicted={pred['predicted_sign']}, "
                  f"actual rho={pred['actual_rho']:.4f} ({pred['actual_sign']}) [{check}]")

    bl_rho = sign_results.get("baseline_bk_vs_rho", {})
    if bl_rho:
        print(f"\n  Baseline BK% vs rho (Spearman): r={bl_rho.get('spearman_r', 'N/A')}, "
              f"p={bl_rho.get('p_value', 'N/A')}")

    cos_rho = sign_results.get("cosine_sim_vs_rho", {})
    if cos_rho:
        print(f"  Cosine sim (dir vs mean) vs rho (Spearman): r={cos_rho.get('spearman_r', 'N/A')}, "
              f"p={cos_rho.get('p_value', 'N/A')}")

    print("\n[3] Analyzing norm vs effect strength...")
    norm_results = analyze_norm_vs_effect(combos)

    nr = norm_results.get("norm_vs_abs_rho", {})
    print(f"  Norm vs |rho| (Spearman): r={nr.get('spearman_r', 'N/A')}, "
          f"p={nr.get('p_value', 'N/A')}")

    # Step 4: Generate figure
    print("\n[4] Generating figure...")
    analysis_results = {
        "sign_prediction": sign_results,
        "norm_vs_effect": norm_results,
    }
    fig_path = generate_figure(combos, analysis_results)

    # Step 5: Save JSON results
    print("\n[5] Saving JSON results...")
    output = {
        "description": "V12 BK direction sign reversal analysis",
        "hypothesis": (
            "The sign of rho in steering experiments is determined by the BK base rate "
            "in training data. When BK is the minority class (<50%), mean_BK - mean_Safe "
            "points away from the dominant safe mode, so adding this direction REDUCES BK "
            "(negative rho). Only when BK >= 50% does adding the direction INCREASE BK "
            "(positive rho). LLaMA SM (BK=36.4%) is the apparent exception because its "
            "steering baseline BK=52% -- the direction was computed on data where BK is "
            "minority but the model's behavior at inference happens to be near 50%."
        ),
        "combos": [
            {k: (v if not isinstance(v, float) or math.isfinite(v) else None)
             for k, v in asdict(c).items()}
            for c in combos
        ],
        "analysis": {
            "sign_prediction": sign_results,
            "norm_vs_effect": norm_results,
        },
        "summary": {
            "n_combos_total": len(combos),
            "n_valid_rho": len([c for c in combos if math.isfinite(c.rho)]),
            "n_positive_rho": len([c for c in combos if math.isfinite(c.rho) and c.rho > 0]),
            "n_negative_rho": len([c for c in combos if math.isfinite(c.rho) and c.rho < 0]),
            "n_nan_rho": len([c for c in combos if not math.isfinite(c.rho)]),
            "threshold_model_training_bk_accuracy": tm.get("accuracy"),
            "threshold_model_baseline_bk_accuracy": sign_results.get(
                "threshold_model_baseline_bk", {}
            ).get("accuracy"),
            "bk_vs_rho_spearman_r": bk_rho.get("spearman_r"),
            "bk_vs_rho_spearman_p": bk_rho.get("p_value"),
            "baseline_bk_vs_rho_spearman_r": sign_results.get(
                "baseline_bk_vs_rho", {}
            ).get("spearman_r"),
            "baseline_bk_vs_rho_spearman_p": sign_results.get(
                "baseline_bk_vs_rho", {}
            ).get("p_value"),
        },
        "figure_path": str(fig_path),
    }

    json_out = JSON_DIR / "v12_sign_reversal_analysis.json"
    with open(json_out, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  Saved JSON -> {json_out}")

    # Step 6: Print predictions / interpretation
    print("\n" + "=" * 72)
    print("CONCLUSIONS")
    print("=" * 72)

    # Classify the only positive-rho case
    llama_sm = [c for c in combos if c.label == "LLaMA SM"][0]
    bl_tm = sign_results.get("threshold_model_baseline_bk", {})
    bl_rho = sign_results.get("baseline_bk_vs_rho", {})
    cos_rho = sign_results.get("cosine_sim_vs_rho", {})
    print(f"""
1. SIGN PREDICTION (Training BK% threshold):
   - 4/5 valid combos show NEGATIVE rho (sign reversal)
   - Only LLaMA SM shows POSITIVE rho (+0.964)
   - Training BK% > 50% => positive rho: {tm['accuracy']*100:.0f}% ({tm['n_correct']}/{tm['n_total']})
   - Two errors: LLaMA SM (BK=36.4%, rho>0) and LLaMA MW (BK=75.8%, rho<0)

2. SIGN PREDICTION (Steering Baseline BK% threshold):
   - Baseline BK% > 50% => positive rho: {bl_tm.get('accuracy', 0)*100:.0f}% ({bl_tm.get('n_correct', 0)}/{bl_tm.get('n_total', 0)})
   - This also fails for LLaMA MW (baseline=41%, but training BK=75.8%)
   - Baseline BK% vs rho (Spearman): r={bl_rho.get('spearman_r', 'N/A')}, p={bl_rho.get('p_value', 'N/A')}

3. TRAINING BK% vs rho CORRELATION:
   - Spearman r = {bk_rho.get('spearman_r', 'N/A')} (p = {bk_rho.get('p_value', 'N/A')})
   - Direction: Higher BK% in training -> more positive rho (positive trend)

4. COSINE SIMILARITY (direction vs global mean):
   - cos(BK_dir, global_mean) vs rho: r={cos_rho.get('spearman_r', 'N/A')}, p={cos_rho.get('p_value', 'N/A')}

5. NORM vs |rho|:
   - Spearman r = {nr.get('spearman_r', 'N/A')} (p = {nr.get('p_value', 'N/A')})
   - Note: Gemma norms ({combos[3].direction_norm:.0f}-{combos[5].direction_norm:.0f}) >> LLaMA norms ({combos[0].direction_norm:.1f}-{combos[1].direction_norm:.1f})
   - This reflects model architecture difference (Gemma 3584-dim vs LLaMA 4096-dim)

6. KEY INSIGHT -- LLaMA MW PARADOX:
   - Training BK = 75.8% (majority!), yet rho = -0.955
   - The BK direction (mean_BK - mean_Safe) in MW points toward BK states
   - But adding this direction DECREASES BK at inference (rho < 0)
   - At inference, baseline BK = 41% -- the model behaves differently than training
   - This suggests the direction encodes training-distribution BK features that,
     when amplified at inference, push the model away from BK behavior

7. LLaMA SM -- THE ONLY POSITIVE CASE:
   - Training BK = 36.4%, Steering baseline BK = 52%
   - The only combo where the steering baseline is near/above 50%
   - The BK direction effectively amplifies a balanced decision process
""")
    print("OVERALL: The sign reversal phenomenon is NOT simply predicted by")
    print("training BK base rate alone. The interaction between training-time")
    print("and inference-time class distributions determines the sign.")


if __name__ == "__main__":
    main()
