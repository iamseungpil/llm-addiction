"""
V12 Steering Experiment Cross-Analysis
=======================================
Loads all v12_*.json result files, normalizes schema differences across
experiment generations (standard, n200, S1a/S1b/S1c, random_steering),
and produces:
  1. Summary table (printed + JSON)
  2. Dose-response comparison figure (multi-panel)
  3. Layer comparison figure (LLaMA SM: L22 vs L25 vs L30 vs Combined)
  4. Verdict heatmap (models x tasks)

Usage:
    python analyze_v12_results.py [--json-dir PATH] [--fig-dir PATH]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALPHA_VALUES = ["-2.0", "-1.0", "-0.5", "0.5", "1.0", "2.0"]
ALPHA_FLOATS = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]

DEFAULT_JSON_DIR = Path(__file__).resolve().parent.parent / "results" / "json"
DEFAULT_FIG_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExperimentRecord:
    """Normalized representation of a single V12 steering experiment."""
    filename: str
    experiment: str
    model: str
    task: str
    layers: list[int]
    layer_label: str
    n_trials: int
    baseline_bk: float
    baseline_stop: float
    bk_rho: float
    bk_p: float
    bk_rates: dict[str, float]       # alpha -> bk_rate
    stop_rates: dict[str, float]     # alpha -> stop_rate
    random_rhos: list[float]
    random_ps: list[float]
    verification: dict[str, bool]     # T1-T6 -> pass/fail
    verdict: str
    effect_bk_plus2: float            # bk_rate at alpha=+2 minus baseline
    effect_stop_minus2: float         # stop_rate at alpha=-2 minus baseline
    sort_key: str = ""                # for ordering


# ---------------------------------------------------------------------------
# Schema normalisation helpers
# ---------------------------------------------------------------------------

def _safe_float(val: Any, default: float = float("nan")) -> float:
    """Convert value to float, returning default for None/NaN/non-numeric."""
    if val is None:
        return default
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _extract_bk_rates_from_results(results: dict) -> tuple[dict[str, float], dict[str, float]]:
    """Extract bk_rates and stop_rates dicts from a results dict with full trial data."""
    bk_rates = {}
    stop_rates = {}
    for alpha in ALPHA_VALUES:
        entry = results.get(alpha, {})
        if isinstance(entry, dict):
            bk_rates[alpha] = _safe_float(entry.get("bk_rate"))
            stop_rates[alpha] = _safe_float(entry.get("stop_rate"))
        else:
            bk_rates[alpha] = float("nan")
            stop_rates[alpha] = float("nan")
    return bk_rates, stop_rates


def _extract_bk_rates_flat(bk_rates_dict: dict) -> tuple[dict[str, float], dict[str, float]]:
    """When bk_direction stores bk_rates directly (flat dict alpha -> rate)."""
    bk_rates = {}
    stop_rates = {}
    for alpha in ALPHA_VALUES:
        bk = _safe_float(bk_rates_dict.get(alpha))
        bk_rates[alpha] = bk
        stop_rates[alpha] = 1.0 - bk if math.isfinite(bk) else float("nan")
    return bk_rates, stop_rates


def _infer_model(data: dict, filename: str) -> str:
    """Infer model name from data or filename."""
    if "model" in data:
        return str(data["model"]).lower()
    fname = os.path.basename(filename).lower()
    if "gemma" in fname:
        return "gemma"
    if "llama" in fname:
        return "llama"
    # S1 files: experiment name may contain model
    exp = data.get("experiment", "")
    if "llama" in exp.lower():
        return "llama"
    if "gemma" in exp.lower():
        return "gemma"
    # n200 and random_steering were LLaMA L22 (default)
    return "llama"


def _infer_task(data: dict, filename: str) -> str:
    """Infer task from data or filename."""
    if "task" in data:
        return str(data["task"]).lower()
    if "game" in data:
        return str(data["game"]).lower()
    fname = os.path.basename(filename).lower()
    for t in ["sm", "ic", "mw"]:
        if f"_{t}_" in fname or f"_{t}." in fname:
            return t
    return "sm"  # default: slot machine


def _infer_layers(data: dict, filename: str) -> list[int]:
    """Infer layers from data or filename."""
    if "layers" in data:
        return [int(l) for l in data["layers"]]
    if "layer" in data:
        return [int(data["layer"])]
    fname = os.path.basename(filename).lower()
    # Try to parse L## patterns
    import re
    matches = re.findall(r"l(\d+)", fname)
    if matches:
        return [int(m) for m in matches]
    return [22]  # default layer


def _parse_verification(data: dict) -> dict[str, bool]:
    """Normalise the various verification formats to T1-T6 -> bool."""
    verif = data.get("verification", {})
    result = {}
    for key in ["T1", "T2", "T3", "T4", "T5", "T6"]:
        val = verif.get(key)
        if val is None:
            result[key] = False
        elif isinstance(val, dict):
            # S1 format: {"pass": true/false/"True", "value": "..."}
            p = val.get("pass", False)
            result[key] = _to_bool(p)
        else:
            result[key] = _to_bool(val)
    return result


def _to_bool(val: Any) -> bool:
    """Coerce mixed bool/string to Python bool."""
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes", "pass")
    return bool(val)


def _infer_verdict(data: dict) -> str:
    """Extract verdict, handling both top-level and nested formats.

    Priority:
      1. data["verdict"]              -- standard/n200 files
      2. data["analysis"]["verdict"]  -- random_steering files
      3. data["verification"]["T6"]["value"]  -- S1 files where verdict is
         embedded inside the T6 verification entry
      4. "UNKNOWN"
    """
    if "verdict" in data:
        return str(data["verdict"])
    analysis = data.get("analysis", {})
    if "verdict" in analysis:
        return str(analysis["verdict"])
    # S1 files: verdict stored inside verification.T6.value
    verif = data.get("verification", {})
    t6 = verif.get("T6")
    if isinstance(t6, dict) and "value" in t6:
        val = str(t6["value"]).strip()
        # Only accept known verdict strings
        if val in ("BK_SPECIFIC_CONFIRMED", "NOT_CONFIRMED", "NOT_SIGNIFICANT",
                    "PARTIALLY_SPECIFIC"):
            return val
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_experiment(filepath: str) -> Optional[ExperimentRecord]:
    """Load and normalize a single V12 result JSON file."""
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  [SKIP] Cannot parse {os.path.basename(filepath)}: {exc}")
        return None

    filename = os.path.basename(filepath)
    model = _infer_model(data, filepath)
    task = _infer_task(data, filepath)
    layers = _infer_layers(data, filepath)
    n_trials = int(data.get("n_trials", 0))

    # Baseline
    baseline = data.get("baseline", {})
    baseline_bk = _safe_float(baseline.get("bk_rate"))
    baseline_stop = _safe_float(baseline.get("stop_rate"))

    # BK direction
    bk_dir = data.get("bk_direction", {})
    bk_rho = _safe_float(bk_dir.get("rho"))
    bk_p = _safe_float(bk_dir.get("p"))

    # BK rates: two formats -- "results" dict with full entries, or flat "bk_rates"
    if "results" in bk_dir:
        bk_rates, stop_rates = _extract_bk_rates_from_results(bk_dir["results"])
    elif "bk_rates" in bk_dir:
        bk_rates, stop_rates = _extract_bk_rates_flat(bk_dir["bk_rates"])
    else:
        bk_rates = {a: float("nan") for a in ALPHA_VALUES}
        stop_rates = {a: float("nan") for a in ALPHA_VALUES}

    # Random directions
    random_dirs = data.get("random_directions", [])
    random_rhos = [_safe_float(rd.get("rho")) for rd in random_dirs]
    random_ps = [_safe_float(rd.get("p")) for rd in random_dirs]

    # Verification
    verification = _parse_verification(data)

    # Verdict
    verdict = _infer_verdict(data)

    # Layer label
    layer_label = "+".join(f"L{l}" for l in layers)

    # Effect sizes
    bk_at_plus2 = _safe_float(bk_rates.get("2.0"))
    stop_at_minus2 = _safe_float(stop_rates.get("-2.0"))
    effect_bk_plus2 = bk_at_plus2 - baseline_bk if (
        math.isfinite(bk_at_plus2) and math.isfinite(baseline_bk)
    ) else float("nan")
    effect_stop_minus2 = stop_at_minus2 - baseline_stop if (
        math.isfinite(stop_at_minus2) and math.isfinite(baseline_stop)
    ) else float("nan")

    # Experiment label
    experiment = data.get("experiment", filename.replace(".json", ""))

    # Sort key for deterministic ordering
    sort_key = f"{model}_{task}_{layer_label}_{n_trials:05d}_{filename}"

    return ExperimentRecord(
        filename=filename,
        experiment=experiment,
        model=model,
        task=task,
        layers=layers,
        layer_label=layer_label,
        n_trials=n_trials,
        baseline_bk=baseline_bk,
        baseline_stop=baseline_stop,
        bk_rho=bk_rho,
        bk_p=bk_p,
        bk_rates=bk_rates,
        stop_rates=stop_rates,
        random_rhos=random_rhos,
        random_ps=random_ps,
        verification=verification,
        verdict=verdict,
        effect_bk_plus2=effect_bk_plus2,
        effect_stop_minus2=effect_stop_minus2,
        sort_key=sort_key,
    )


def load_all_experiments(json_dir: str | Path) -> list[ExperimentRecord]:
    """Load all v12_*.json files from the given directory.

    Excludes any files produced by this script (e.g., cross_analysis_summary).
    """
    pattern = os.path.join(str(json_dir), "v12_*.json")
    files = sorted(glob.glob(pattern))
    # Exclude outputs generated by this analysis script
    exclude_substrings = ("cross_analysis_summary",)
    files = [f for f in files if not any(ex in os.path.basename(f) for ex in exclude_substrings)]

    if not files:
        print(f"[WARNING] No v12_*.json files found in {json_dir}")
        return []

    print(f"Found {len(files)} V12 result files in {json_dir}")
    records = []
    for fp in files:
        rec = load_experiment(fp)
        if rec is not None:
            records.append(rec)
    records.sort(key=lambda r: r.sort_key)
    print(f"Successfully loaded {len(records)} experiments\n")
    return records


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _fmt(val: float, precision: int = 3) -> str:
    """Format a float, showing 'N/A' for NaN."""
    if math.isfinite(val):
        return f"{val:.{precision}f}"
    return "N/A"


def _bool_mark(val: bool) -> str:
    return "P" if val else "F"


def print_summary_table(records: list[ExperimentRecord]) -> list[dict]:
    """Print a formatted summary table and return serializable rows."""
    rows = []
    header = (
        f"{'Experiment':<32s} {'Model':<6s} {'Task':<4s} {'Layer(s)':<14s} "
        f"{'N':>4s} {'BL_BK':>6s} {'rho':>7s} {'p':>8s} "
        f"{'T1':>2s} {'T2':>2s} {'T3':>2s} {'T4':>2s} {'T5':>2s} {'T6':>2s} "
        f"{'dBK+2':>7s} {'dSt-2':>7s} {'Verdict':<26s}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for rec in records:
        v = rec.verification
        line = (
            f"{rec.experiment:<32s} {rec.model:<6s} {rec.task:<4s} {rec.layer_label:<14s} "
            f"{rec.n_trials:>4d} {_fmt(rec.baseline_bk):>6s} {_fmt(rec.bk_rho):>7s} {_fmt(rec.bk_p):>8s} "
            f"{_bool_mark(v.get('T1', False)):>2s} {_bool_mark(v.get('T2', False)):>2s} "
            f"{_bool_mark(v.get('T3', False)):>2s} {_bool_mark(v.get('T4', False)):>2s} "
            f"{_bool_mark(v.get('T5', False)):>2s} {_bool_mark(v.get('T6', False)):>2s} "
            f"{_fmt(rec.effect_bk_plus2):>7s} {_fmt(rec.effect_stop_minus2):>7s} {rec.verdict:<26s}"
        )
        print(line)

        # Build JSON-serializable row
        row = {
            "experiment": rec.experiment,
            "filename": rec.filename,
            "model": rec.model,
            "task": rec.task,
            "layers": rec.layers,
            "layer_label": rec.layer_label,
            "n_trials": rec.n_trials,
            "baseline_bk_rate": rec.baseline_bk if math.isfinite(rec.baseline_bk) else None,
            "baseline_stop_rate": rec.baseline_stop if math.isfinite(rec.baseline_stop) else None,
            "bk_rho": rec.bk_rho if math.isfinite(rec.bk_rho) else None,
            "bk_p": rec.bk_p if math.isfinite(rec.bk_p) else None,
            "verification": {k: v for k, v in rec.verification.items()},
            "effect_bk_at_plus2": rec.effect_bk_plus2 if math.isfinite(rec.effect_bk_plus2) else None,
            "effect_stop_at_minus2": rec.effect_stop_minus2 if math.isfinite(rec.effect_stop_minus2) else None,
            "verdict": rec.verdict,
        }
        rows.append(row)

    print(sep)
    return rows


# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------

def _apply_academic_style() -> None:
    """Configure matplotlib for clean academic-style figures."""
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


# ---------------------------------------------------------------------------
# Fig 1: Dose-response comparison
# ---------------------------------------------------------------------------

def _get_dose_response_data(rec: ExperimentRecord) -> tuple[list[float], list[float]]:
    """Return (alphas, bk_rates) lists for plotting, skipping NaN entries."""
    alphas = []
    rates = []
    for a_str, a_float in zip(ALPHA_VALUES, ALPHA_FLOATS):
        val = rec.bk_rates.get(a_str, float("nan"))
        if math.isfinite(val):
            alphas.append(a_float)
            rates.append(val)
    return alphas, rates


def _plot_dose_response_panel(
    ax: plt.Axes, rec: ExperimentRecord, color: str, label: str
) -> None:
    """Plot a single dose-response curve on an axes."""
    alphas, rates = _get_dose_response_data(rec)
    if not alphas:
        return
    ax.plot(alphas, rates, "o-", color=color, linewidth=1.5, markersize=5, label=label)
    # Baseline
    if math.isfinite(rec.baseline_bk):
        ax.axhline(rec.baseline_bk, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)


def fig1_dose_response(records: list[ExperimentRecord], fig_dir: Path) -> None:
    """Multi-panel dose-response comparison across all experiments."""
    # Filter to records with meaningful data (n >= 10)
    valid = [r for r in records if r.n_trials >= 10]
    if not valid:
        print("[Fig 1] No valid experiments with n >= 10. Skipping.")
        return

    # Group by (model, task) for panels
    groups: dict[tuple[str, str], list[ExperimentRecord]] = {}
    for rec in valid:
        key = (rec.model, rec.task)
        groups.setdefault(key, []).append(rec)

    n_panels = len(groups)
    if n_panels == 0:
        return

    ncols = min(n_panels, 3)
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.2 * nrows), squeeze=False)

    palette = plt.cm.tab10
    panel_idx = 0
    for (model, task), group_recs in sorted(groups.items()):
        row, col = divmod(panel_idx, ncols)
        ax = axes[row][col]

        for i, rec in enumerate(group_recs):
            color = palette(i % 10)
            label = f"{rec.layer_label} n={rec.n_trials}"
            if rec.verdict == "BK_SPECIFIC_CONFIRMED":
                label += " *"
            _plot_dose_response_panel(ax, rec, color, label)

        ax.set_title(f"{model.upper()} - {task.upper()}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Steering strength (alpha)")
        ax.set_ylabel("Bankruptcy rate")
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
        panel_idx += 1

    # Hide unused panels
    for idx in range(panel_idx, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle("V12 Steering: Dose-Response Curves", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    outpath = fig_dir / "v12_fig1_dose_response.png"
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved Fig 1 -> {outpath}")


# ---------------------------------------------------------------------------
# Fig 2: Layer comparison (LLaMA SM)
# ---------------------------------------------------------------------------

def fig2_layer_comparison(records: list[ExperimentRecord], fig_dir: Path) -> None:
    """Compare layers for LLaMA SM experiments."""
    candidates = [
        r for r in records
        if r.model == "llama" and r.task == "sm" and r.n_trials >= 50
    ]
    if not candidates:
        print("[Fig 2] No LLaMA SM experiments with n >= 50. Skipping.")
        return

    fig, (ax_bk, ax_rho) = plt.subplots(1, 2, figsize=(11, 4.5))

    palette = {
        "L22": "#1f77b4",
        "L25": "#ff7f0e",
        "L30": "#2ca02c",
        "L22+L25+L30": "#d62728",
    }
    markers = {"L22": "o", "L25": "s", "L30": "^", "L22+L25+L30": "D"}

    # Panel A: dose-response per layer
    for rec in candidates:
        lbl = rec.layer_label
        color = palette.get(lbl, "gray")
        marker = markers.get(lbl, "o")
        alphas, rates = _get_dose_response_data(rec)
        if alphas:
            ax_bk.plot(
                alphas, rates,
                marker=marker, linestyle="-", color=color,
                linewidth=1.5, markersize=6,
                label=f"{lbl} (n={rec.n_trials})",
            )
        if math.isfinite(rec.baseline_bk):
            ax_bk.axhline(rec.baseline_bk, color=color, linestyle=":", linewidth=0.6, alpha=0.4)

    ax_bk.set_title("A. Dose-Response by Layer", fontsize=11, fontweight="bold")
    ax_bk.set_xlabel("Steering strength (alpha)")
    ax_bk.set_ylabel("Bankruptcy rate")
    ax_bk.set_xlim(-2.5, 2.5)
    ax_bk.set_ylim(-0.02, 1.02)
    ax_bk.legend(fontsize=8, framealpha=0.9)

    # Panel B: rho comparison (bar chart)
    labels = []
    rho_vals = []
    colors = []
    for rec in candidates:
        lbl = rec.layer_label
        labels.append(f"{lbl}\nn={rec.n_trials}")
        rho_vals.append(rec.bk_rho if math.isfinite(rec.bk_rho) else 0.0)
        colors.append(palette.get(lbl, "gray"))

    x = np.arange(len(labels))
    bars = ax_rho.bar(x, rho_vals, color=colors, edgecolor="black", linewidth=0.6, width=0.6)
    ax_rho.set_xticks(x)
    ax_rho.set_xticklabels(labels, fontsize=8)
    ax_rho.set_ylabel("Spearman rho (BK direction)")
    ax_rho.set_title("B. Correlation Strength by Layer", fontsize=11, fontweight="bold")
    ax_rho.axhline(0, color="gray", linewidth=0.5)

    # Add p-value annotations
    for i, rec in enumerate(candidates):
        p_str = _fmt(rec.bk_p, 4) if math.isfinite(rec.bk_p) else "N/A"
        ypos = rho_vals[i] + 0.03 if rho_vals[i] >= 0 else rho_vals[i] - 0.06
        ax_rho.text(i, ypos, f"p={p_str}", ha="center", fontsize=7, style="italic")

    fig.suptitle("V12: LLaMA SM Layer Comparison", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    outpath = fig_dir / "v12_fig2_layer_comparison.png"
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved Fig 2 -> {outpath}")


# ---------------------------------------------------------------------------
# Fig 3: Verdict heatmap
# ---------------------------------------------------------------------------

def fig3_verdict_heatmap(records: list[ExperimentRecord], fig_dir: Path) -> None:
    """Heatmap: models x tasks x verdict (best result per cell)."""
    # Use only records with reasonable sample size
    valid = [r for r in records if r.n_trials >= 50]
    if not valid:
        print("[Fig 3] No valid experiments with n >= 50. Skipping.")
        return

    # Collect unique models and tasks
    models = sorted(set(r.model for r in valid))
    tasks = sorted(set(r.task for r in valid))

    if not models or not tasks:
        print("[Fig 3] Insufficient model/task diversity. Skipping.")
        return

    verdict_map = {
        "BK_SPECIFIC_CONFIRMED": 2,
        "PARTIALLY_SPECIFIC": 1,
        "NOT_SIGNIFICANT": 0,
        "UNKNOWN": -1,
    }
    verdict_labels = {2: "Confirmed", 1: "Partial", 0: "Not sig.", -1: "Unknown"}
    verdict_colors = {2: "#2ca02c", 1: "#ffcc00", 0: "#d62728", -1: "#cccccc"}

    # Best verdict per (model, task) -- prefer best n_trials among confirmed
    best: dict[tuple[str, str], ExperimentRecord] = {}
    for rec in valid:
        key = (rec.model, rec.task)
        score = verdict_map.get(rec.verdict, -1)
        existing = best.get(key)
        if existing is None:
            best[key] = rec
        else:
            existing_score = verdict_map.get(existing.verdict, -1)
            if score > existing_score or (score == existing_score and rec.n_trials > existing.n_trials):
                best[key] = rec

    # Build matrix
    n_models = len(models)
    n_tasks = len(tasks)
    matrix = np.full((n_models, n_tasks), -1, dtype=int)
    annot = [[" " for _ in range(n_tasks)] for _ in range(n_models)]

    for i, m in enumerate(models):
        for j, t in enumerate(tasks):
            rec = best.get((m, t))
            if rec is not None:
                v = verdict_map.get(rec.verdict, -1)
                matrix[i, j] = v
                rho_str = _fmt(rec.bk_rho) if math.isfinite(rec.bk_rho) else "N/A"
                annot[i][j] = f"{verdict_labels[v]}\nrho={rho_str}\nn={rec.n_trials}"

    # Build colormap
    cmap_list = [verdict_colors[-1], verdict_colors[0], verdict_colors[1], verdict_colors[2]]
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5]
    cmap = mcolors.ListedColormap(cmap_list)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(3.2 * n_tasks + 1, 2.5 * n_models + 0.5))
    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    # Annotations
    for i in range(n_models):
        for j in range(n_tasks):
            text_color = "white" if matrix[i, j] in (0, -1) else "black"
            ax.text(j, i, annot[i][j], ha="center", va="center",
                    fontsize=9, fontweight="bold", color=text_color)

    ax.set_xticks(range(n_tasks))
    ax.set_xticklabels([t.upper() for t in tasks], fontsize=11)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels([m.upper() for m in models], fontsize=11)
    ax.set_xlabel("Task", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)
    ax.set_title("V12 Steering Verdict: Model x Task", fontsize=13, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=verdict_colors[2], edgecolor="black", label="Confirmed"),
        Patch(facecolor=verdict_colors[1], edgecolor="black", label="Partial"),
        Patch(facecolor=verdict_colors[0], edgecolor="black", label="Not sig."),
        Patch(facecolor=verdict_colors[-1], edgecolor="black", label="No data"),
    ]
    ax.legend(
        handles=legend_elements, loc="upper left",
        bbox_to_anchor=(1.02, 1.0), fontsize=9, framealpha=0.9,
    )

    plt.tight_layout()
    outpath = fig_dir / "v12_fig3_verdict_heatmap.png"
    fig.savefig(outpath)
    plt.close(fig)
    print(f"  Saved Fig 3 -> {outpath}")


# ---------------------------------------------------------------------------
# Save summary JSON
# ---------------------------------------------------------------------------

def save_summary_json(rows: list[dict], json_dir: Path) -> None:
    """Save the cross-analysis summary to a JSON file."""
    outpath = json_dir / "v12_cross_analysis_summary.json"
    summary = {
        "description": "V12 steering experiment cross-comparison summary",
        "n_experiments": len(rows),
        "experiments": rows,
    }
    with open(outpath, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Saved summary JSON -> {outpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="V12 steering cross-analysis")
    parser.add_argument(
        "--json-dir", type=str, default=str(DEFAULT_JSON_DIR),
        help="Directory containing v12_*.json result files",
    )
    parser.add_argument(
        "--fig-dir", type=str, default=str(DEFAULT_FIG_DIR),
        help="Directory to save figures",
    )
    args = parser.parse_args()

    json_dir = Path(args.json_dir)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    _apply_academic_style()

    # Load data
    records = load_all_experiments(json_dir)
    if not records:
        print("No experiments loaded. Exiting.")
        sys.exit(1)

    # Summary table
    print("=" * 80)
    print("V12 STEERING EXPERIMENT CROSS-ANALYSIS SUMMARY")
    print("=" * 80)
    rows = print_summary_table(records)
    print()

    # Save summary JSON
    save_summary_json(rows, json_dir)

    # Generate figures
    print("\nGenerating figures...")
    fig1_dose_response(records, fig_dir)
    fig2_layer_comparison(records, fig_dir)
    fig3_verdict_heatmap(records, fig_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
