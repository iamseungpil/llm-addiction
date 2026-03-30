"""
V13 Publication-Quality Figure Generator
=========================================
Generates 8 figures for the V13 report, each conveying ONE clear message.

Figures:
  0: Experimental Framework Overview (schematic)
  1: BK Classification Summary (RQ1)
  2: Universal BK Neurons (RQ1)
  3: Cross-Domain Transfer Heatmap (RQ2)
  4: Cross-Domain Steering Transfer (RQ2 + Causal)
  7: Direction Steering Dose-Response (Causal - KEY)
  8: Multi-Layer Amplification (Causal)
  9: Overall Verdict Matrix (Summary)

Usage:
    conda run -n llm-addiction python sae_v3_analysis/src/generate_v13_figures.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
JSON_DIR = BASE_DIR / "results" / "json"
FIG_DIR = BASE_DIR / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------

# Colorblind-friendly palette
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PINK = "#CC79A7"
GRAY = "#999999"
LIGHT_GRAY = "#CCCCCC"
DARK_GRAY = "#666666"

DPI = 200

# Academic style
STYLE_RC = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Georgia"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
}

plt.rcParams.update(STYLE_RC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    """Load a JSON file, handling NaN values."""
    with open(path) as f:
        text = f.read()
    text = text.replace("NaN", "null").replace("Infinity", "null")
    return json.loads(text)


def _save_fig(fig: plt.Figure, name: str) -> Path:
    """Save figure and close."""
    path = FIG_DIR / name
    fig.savefig(path, dpi=DPI, facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def _add_text_box(ax, text: str, xy: tuple, fontsize: int = 9,
                  color: str = "black", bg: str = "#FFFDE7",
                  ha: str = "center", va: str = "center"):
    """Add an annotated text box."""
    ax.annotate(
        text, xy=xy, fontsize=fontsize, ha=ha, va=va, color=color,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=bg,
                  edgecolor=DARK_GRAY, alpha=0.9),
        xycoords="axes fraction",
    )


# ===========================================================================
# FIGURE 0: Experimental Framework Overview
# ===========================================================================

def fig0_experimental_framework() -> Path:
    """Schematic: 2 Models x 3 Tasks -> Analysis Pipeline -> 4 RQs."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    fig.suptitle("Figure 0: Experimental Framework Overview",
                 fontsize=14, fontweight="bold", y=0.98)

    # Helper to draw a rounded box
    def draw_box(x, y, w, h, text, color, fontsize=9, textcolor="white",
                 bold=False):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="black", linewidth=1.2
        )
        ax.add_patch(box)
        weight = "bold" if bold else "normal"
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, color=textcolor, fontweight=weight)

    def draw_arrow(x1, y1, x2, y2, color="black"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

    # Column 1: Models
    ax.text(0.8, 4.7, "Models", ha="center", fontsize=11, fontweight="bold")
    draw_box(0.1, 3.8, 1.4, 0.6, "Gemma-2-9B\n(SAE features)", BLUE,
             fontsize=8)
    draw_box(0.1, 2.9, 1.4, 0.6, "LLaMA-3.1-8B\n(hidden states)", ORANGE,
             fontsize=8)

    # Column 2: Tasks
    ax.text(2.95, 4.7, "Gambling Tasks", ha="center", fontsize=11,
            fontweight="bold")
    task_colors = [GREEN, PINK, BLUE]
    task_names = ["Slot Machine\n(SM)", "Investment\nChoice (IC)",
                  "Mystery\nWheel (MW)"]
    for i, (name, color) in enumerate(zip(task_names, task_colors)):
        draw_box(2.2, 3.8 - i * 0.85, 1.5, 0.6, name, color, fontsize=8)

    # Arrows: models -> tasks
    for my in [4.1, 3.2]:
        for ty in [4.1, 3.25, 2.4]:
            draw_arrow(1.5, my, 2.2, ty, color=GRAY)

    # Column 3: Hidden states
    ax.text(5.1, 4.7, "Neural\nRepresentations", ha="center", fontsize=10,
            fontweight="bold")
    draw_box(4.3, 3.2, 1.6, 1.1, "Hidden States\n(32 layers\n"
             "x 3072/4096 dims)", DARK_GRAY, fontsize=8)

    # Arrows: tasks -> hidden states
    for ty in [4.1, 3.25, 2.4]:
        draw_arrow(3.7, ty, 4.3, 3.75, color=GRAY)

    # Column 4: Analyses (RQs)
    ax.text(7.8, 4.7, "Research Questions", ha="center", fontsize=11,
            fontweight="bold")
    rqs = [
        ("RQ1: BK Prediction\n(DP + R1 classification)", BLUE),
        ("RQ2: Cross-Domain\n(transfer across tasks)", GREEN),
        ("RQ3: Conditions\n(bet type, prompt)", PINK),
        ("Causal: Steering\n(direction injection)", ORANGE),
    ]
    for i, (label, color) in enumerate(rqs):
        draw_box(6.5, 4.0 - i * 0.85, 2.5, 0.6, label, color, fontsize=8)

    # Arrows: hidden states -> RQs
    for ry in [4.3, 3.45, 2.6, 1.75]:
        draw_arrow(5.9, 3.75, 6.5, ry, color=GRAY)

    # Bottom annotation
    ax.text(5.0, 0.5,
            "6 model-task combinations  |  "
            "n = 50-200 games per condition  |  "
            "6 steering strengths per experiment",
            ha="center", va="center", fontsize=9, style="italic",
            color=DARK_GRAY)

    return _save_fig(fig, "v13_fig0_experimental_framework.png")


# ===========================================================================
# FIGURE 1: BK Classification Summary (RQ1)
# ===========================================================================

def fig1_bk_classification() -> Path:
    """2-panel: DP AUC and R1 AUC for Gemma vs LLaMA across 3 tasks."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle("Figure 1: Bankruptcy Is Predictable from Neural Activations",
                 fontsize=13, fontweight="bold", y=1.02)

    tasks = ["SM", "IC", "MW"]
    x = np.arange(len(tasks))
    width = 0.3

    # Data from V10 analyses
    gemma_dp = [0.976, 0.960, 0.966]
    llama_dp = [0.974, 0.954, 0.963]
    gemma_r1 = [0.909, 0.866, 0.864]
    llama_r1 = [0.897, 0.849, 0.855]

    # Panel (a): DP AUC
    bars1 = ax1.bar(x - width / 2, gemma_dp, width, label="Gemma-2-9B",
                    color=BLUE, edgecolor="white", linewidth=0.5)
    bars2 = ax1.bar(x + width / 2, llama_dp, width, label="LLaMA-3.1-8B",
                    color=ORANGE, edgecolor="white", linewidth=0.5)

    ax1.set_ylabel("Direction Probe AUC")
    ax1.set_title("(a) Direction Probe (DP)", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(tasks)
    ax1.set_ylim(0.90, 1.0)
    ax1.axhline(y=0.5, color=GRAY, linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.legend(loc="lower left", framealpha=0.9)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, height + 0.002,
                     f"{height:.3f}", ha="center", va="bottom", fontsize=8)

    # Panel (b): R1 AUC
    bars3 = ax2.bar(x - width / 2, gemma_r1, width, label="Gemma-2-9B",
                    color=BLUE, edgecolor="white", linewidth=0.5)
    bars4 = ax2.bar(x + width / 2, llama_r1, width, label="LLaMA-3.1-8B",
                    color=ORANGE, edgecolor="white", linewidth=0.5)

    ax2.set_ylabel("Round-1 Probe AUC")
    ax2.set_title("(b) Round-1 Probe (R1)", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(tasks)
    ax2.set_ylim(0.80, 0.95)
    ax2.axhline(y=0.5, color=GRAY, linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.legend(loc="lower left", framealpha=0.9)

    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, height + 0.002,
                     f"{height:.3f}", ha="center", va="bottom", fontsize=8)

    # Annotation highlighting key finding
    ax1.annotate(
        "All AUCs > 0.95\n(near-perfect)",
        xy=(1, 0.955), xytext=(1.5, 0.925),
        fontsize=8, ha="center", color=BLUE,
        arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.0),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F4FD",
                  edgecolor=BLUE, alpha=0.8),
    )

    fig.tight_layout()
    return _save_fig(fig, "v13_fig1_bk_classification.png")


# ===========================================================================
# FIGURE 2: Universal BK Neurons
# ===========================================================================

def fig2_universal_neurons() -> Path:
    """Stacked bar: Gemma 600 vs LLaMA 1334, showing promoting/inhibiting."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    fig.suptitle("Figure 2: Balanced Push-Pull BK Neuron Structure",
                 fontsize=13, fontweight="bold", y=1.02)

    models = ["Gemma-2-9B\n(SAE features)", "LLaMA-3.1-8B\n(hidden dims)"]
    promoting = [302, 672]
    inhibiting = [298, 662]
    totals = [600, 1334]

    x = np.arange(len(models))
    width = 0.45

    bars1 = ax.bar(x, promoting, width, label="BK-Promoting",
                   color=ORANGE, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x, inhibiting, width, bottom=promoting,
                   label="BK-Inhibiting", color=BLUE,
                   edgecolor="white", linewidth=0.5)

    # Value labels
    for i in range(len(models)):
        # Promoting count
        ax.text(x[i], promoting[i] / 2, f"{promoting[i]}",
                ha="center", va="center", fontsize=11, fontweight="bold",
                color="white")
        # Inhibiting count
        ax.text(x[i], promoting[i] + inhibiting[i] / 2, f"{inhibiting[i]}",
                ha="center", va="center", fontsize=11, fontweight="bold",
                color="white")
        # Total on top
        ax.text(x[i], totals[i] + 30, f"Total: {totals[i]}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Number of Significant BK Neurons")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 1550)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=10)

    # Annotation: ratio
    for i in range(len(models)):
        ratio = promoting[i] / inhibiting[i]
        ax.text(x[i], totals[i] + 80,
                f"Ratio: {ratio:.2f}",
                ha="center", va="bottom", fontsize=9, color=DARK_GRAY,
                style="italic")

    # Key message annotation
    _add_text_box(ax, "Near 1:1 promoting:inhibiting\n"
                  "ratio in both models", (0.7, 0.85), fontsize=9,
                  bg="#E8F4FD")

    fig.tight_layout()
    return _save_fig(fig, "v13_fig2_universal_neurons.png")


# ===========================================================================
# FIGURE 3: Cross-Domain Transfer Heatmap (RQ2)
# ===========================================================================

def fig3_crossdomain_transfer() -> Path:
    """2-panel heatmap: Gemma SAE transfer vs LLaMA hidden transfer."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Figure 3: BK Signal Transfers Across Gambling Domains",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.subplots_adjust(right=0.85, wspace=0.35)

    tasks = ["IC", "SM", "MW"]

    # Data from V10 cross-domain analysis
    # Gemma SAE transfer AUC (train row, test col)
    gemma_data = np.array([
        [np.nan, 0.913, 0.932],  # IC -> SM, MW
        [0.646, np.nan, 0.867],  # SM -> IC, MW
        [0.853, np.nan, np.nan],  # MW -> IC only
    ])

    # LLaMA hidden transfer AUC
    llama_data = np.array([
        [np.nan, 0.577, 0.680],  # IC -> SM, MW
        [0.749, np.nan, 0.561],  # SM -> IC, MW
        [0.805, 0.682, np.nan],  # MW -> IC, SM
    ])

    # Custom colormap: green high, orange low
    cmap = plt.cm.YlGn

    for ax, data, title in [(ax1, gemma_data, "(a) Gemma SAE Features"),
                            (ax2, llama_data, "(b) LLaMA Hidden States")]:
        # Mask diagonal (NaN)
        masked = np.ma.masked_invalid(data)

        im = ax.imshow(masked, cmap=cmap, vmin=0.4, vmax=1.0, aspect="auto")

        # Annotate cells
        for i in range(3):
            for j in range(3):
                if np.isnan(data[i, j]):
                    ax.text(j, i, "--", ha="center", va="center",
                            fontsize=10, color=GRAY)
                else:
                    val = data[i, j]
                    text_color = "white" if val > 0.8 else "black"
                    fontw = "bold" if val > 0.8 else "normal"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=11, color=text_color, fontweight=fontw)

        ax.set_xticks(range(3))
        ax.set_xticklabels(tasks)
        ax.set_yticks(range(3))
        ax.set_yticklabels(tasks)
        ax.set_xlabel("Test Domain")
        ax.set_ylabel("Train Domain")
        ax.set_title(title, fontweight="bold", fontsize=11)

    # Shared colorbar -- use dedicated axis to avoid overlap
    cbar_ax = fig.add_axes([0.88, 0.18, 0.02, 0.65])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Transfer AUC", fontsize=10)

    return _save_fig(fig, "v13_fig3_crossdomain_transfer.png")


# ===========================================================================
# FIGURE 4: Cross-Domain Steering Transfer (RQ2 + Causal)
# ===========================================================================

def fig4_crossdomain_steering() -> Path:
    """3x3 heatmap of cross-domain steering |rho| with p-values."""
    data = _load_json(JSON_DIR / "v12_crossdomain_steering.json")

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle(
        "Figure 4: BK Direction From One Task Causally Steers Another",
        fontsize=13, fontweight="bold", y=1.02)

    tasks = ["SM", "IC", "MW"]
    task_order = {"sm": 0, "ic": 1, "mw": 2}

    # Build matrix
    rho_matrix = np.full((3, 3), np.nan)
    p_matrix = np.full((3, 3), np.nan)

    # Diagonal: within-domain from summary
    within = data["within_domain_rho"]
    for task_key, idx in task_order.items():
        rho_matrix[idx, idx] = within[task_key]
        p_matrix[idx, idx] = 0.001  # all significant

    # Off-diagonal: cross-domain
    for result in data["cross_domain_results"]:
        src = task_order[result["source_task"]]
        tgt = task_order[result["target_task"]]
        rho_matrix[src, tgt] = result["abs_rho"]
        p_matrix[src, tgt] = result["p_value"]

    # Color map
    cmap = plt.cm.Blues
    im = ax.imshow(rho_matrix, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    # Annotate cells
    for i in range(3):
        for j in range(3):
            rho_val = rho_matrix[i, j]
            p_val = p_matrix[i, j]

            if np.isnan(rho_val):
                continue

            # Significance marker
            if p_val < 0.01:
                sig = "**"
            elif p_val < 0.05:
                sig = "*"
            else:
                sig = ""

            text_color = "white" if rho_val > 0.7 else "black"

            # Main value
            if i == j:
                label = f"{rho_val:.3f}\n(within)"
            else:
                label = f"{rho_val:.3f}{sig}\n(p={p_val:.3f})"

            ax.text(j, i, label, ha="center", va="center",
                    fontsize=10, color=text_color,
                    fontweight="bold" if rho_val > 0.8 else "normal")

    # Red border on diagonal
    for k in range(3):
        rect = plt.Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False,
                              edgecolor="red", linewidth=2.5)
        ax.add_patch(rect)

    ax.set_xticks(range(3))
    ax.set_xticklabels(tasks, fontsize=11)
    ax.set_yticks(range(3))
    ax.set_yticklabels(tasks, fontsize=11)
    ax.set_xlabel("Target Task (steered game)", fontsize=12)
    ax.set_ylabel("Source Task (BK direction from)", fontsize=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("|Spearman rho|", fontsize=10)

    # Bottom note
    ax.text(1, 3.3,
            "Red border = within-domain.  * p<0.05  ** p<0.01\n"
            "3/6 cross-domain transfers are significant (p<0.05)",
            ha="center", va="top", fontsize=9, color=DARK_GRAY,
            style="italic", transform=ax.transData)

    fig.tight_layout()
    return _save_fig(fig, "v13_fig4_crossdomain_steering.png")


# ===========================================================================
# FIGURE 7: Direction Steering Dose-Response (KEY figure)
# ===========================================================================

def fig7_dose_response() -> Path:
    """2-panel: Left = LLaMA SM L22 n=200, Right = All 6 combos."""
    n200 = _load_json(JSON_DIR / "v12_n200_20260327_030745.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        "Figure 7: BK Direction Specifically and Causally Controls "
        "Gambling Behavior",
        fontsize=13, fontweight="bold", y=1.02)
    fig.subplots_adjust(wspace=0.3)

    # ---- Panel (a): LLaMA SM L22 n=200 ----
    bk_results = n200["bk_direction"]["results"]
    alphas = sorted(bk_results.keys(), key=float)
    alpha_vals = [float(a) for a in alphas]
    bk_rates = [bk_results[a]["bk_rate"] for a in alphas]

    ax1.plot(alpha_vals, bk_rates, "o-", color=BLUE, linewidth=2.5,
             markersize=8, label="BK direction", zorder=5)

    # Random directions (gray)
    for i, rand in enumerate(n200["random_directions"]):
        r_results = rand["results"]
        r_alphas = sorted(r_results.keys(), key=float)
        r_vals = [float(a) for a in r_alphas]
        r_bk = [r_results[a]["bk_rate"] for a in r_alphas]
        label = "Random directions" if i == 0 else None
        ax1.plot(r_vals, r_bk, "s--", color=LIGHT_GRAY, linewidth=1.0,
                 markersize=4, alpha=0.7, label=label, zorder=2)

    # Baseline
    baseline_bk = n200["baseline"]["bk_rate"]
    ax1.axhline(y=baseline_bk, color=GRAY, linestyle=":", linewidth=1,
                alpha=0.6)
    ax1.text(-2.3, baseline_bk, f"Baseline\n{baseline_bk:.0%}",
             fontsize=8, va="center", color=GRAY)

    ax1.set_xlabel("Steering Strength (alpha)")
    ax1.set_ylabel("Bankruptcy Rate")
    ax1.set_title("(a) LLaMA SM, Layer 22, n=200", fontweight="bold")
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.set_xlim(-2.5, 2.5)
    ax1.set_ylim(0.2, 0.75)

    # Annotation: rho and p
    ax1.annotate(
        f"rho = {n200['bk_direction']['rho']:.3f}\n"
        f"p = {n200['bk_direction']['p']:.4f}",
        xy=(2.0, bk_rates[-1]), xytext=(0.5, 0.7),
        fontsize=9, ha="center", color=BLUE, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F4FD",
                  edgecolor=BLUE, alpha=0.9),
    )

    # Annotation: random control
    ax1.annotate(
        "Random: all p > 0.45",
        xy=(0, 0.5), xytext=(0.72, 0.15),
        fontsize=8, ha="center", color=DARK_GRAY, style="italic",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#F5F5F5",
                  edgecolor=GRAY, alpha=0.8),
        xycoords="axes fraction",
    )

    # ---- Panel (b): All 6 model x task combos ----
    combo_files = {
        "LLaMA SM": "v12_llama_sm_L22_20260328_091923.json",
        "LLaMA IC": "v12_llama_ic_L22_20260329_022313.json",
        "LLaMA MW": "v12_llama_mw_L22_20260329_072818.json",
        "Gemma SM": "v12_gemma_sm_L22_20260328_014425.json",
        "Gemma IC": "v12_gemma_ic_L22_20260328_100129.json",
        "Gemma MW": "v12_gemma_mw_L22_20260328_205618.json",
    }

    combo_colors = {
        "LLaMA SM": BLUE,
        "LLaMA IC": GREEN,
        "LLaMA MW": PINK,
        "Gemma SM": BLUE,
        "Gemma IC": GREEN,
        "Gemma MW": PINK,
    }
    combo_styles = {
        "LLaMA SM": "-",
        "LLaMA IC": "-",
        "LLaMA MW": "-",
        "Gemma SM": "--",
        "Gemma IC": "--",
        "Gemma MW": "--",
    }
    combo_markers = {
        "LLaMA SM": "o",
        "LLaMA IC": "s",
        "LLaMA MW": "^",
        "Gemma SM": "o",
        "Gemma IC": "s",
        "Gemma MW": "^",
    }

    for label, fname in combo_files.items():
        d = _load_json(JSON_DIR / fname)
        bk_dir = d["bk_direction"]["results"]
        a_keys = sorted(bk_dir.keys(), key=float)
        a_vals = [float(a) for a in a_keys]
        bk_vals = [bk_dir[a]["bk_rate"] for a in a_keys]
        rho = d["bk_direction"].get("rho")
        p = d["bk_direction"].get("p")

        # Build label with rho
        if rho is not None and not (isinstance(rho, float) and math.isnan(rho)):
            sig = "*" if (p is not None and p < 0.05) else ""
            lbl = f"{label} (rho={rho:.2f}{sig})"
        else:
            lbl = f"{label} (n/a)"

        ax2.plot(a_vals, bk_vals, combo_styles[label],
                 color=combo_colors[label], marker=combo_markers[label],
                 linewidth=1.5, markersize=5, label=lbl, alpha=0.85)

    ax2.set_xlabel("Steering Strength (alpha)")
    ax2.set_ylabel("Bankruptcy Rate")
    ax2.set_title("(b) All 6 Model x Task Combinations", fontweight="bold")
    ax2.legend(loc="upper left", framealpha=0.9, fontsize=8.5, ncol=1)
    ax2.set_xlim(-2.5, 2.5)
    ax2.set_ylim(-0.05, 1.05)

    # Annotation
    ax2.annotate(
        "Solid = LLaMA\nDashed = Gemma",
        xy=(0.03, 0.97), xytext=(0.03, 0.97),
        fontsize=8, ha="left", va="top", color=DARK_GRAY,
        xycoords="axes fraction",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#F5F5F5",
                  edgecolor=GRAY, alpha=0.8),
    )

    fig.tight_layout()
    return _save_fig(fig, "v13_fig7_dose_response.png")


# ===========================================================================
# FIGURE 8: Multi-Layer Amplification
# ===========================================================================

def fig8_multilayer() -> Path:
    """2-panel: dose-response curves for L22/L25/L30/Combined + bar chart."""
    # Load data
    l22_data = _load_json(JSON_DIR / "v12_llama_sm_L22_20260328_091923.json")
    l25_data = _load_json(
        JSON_DIR / "v12_s1a_llama_sm_l25_20260327_140500.json")
    l30_data = _load_json(
        JSON_DIR / "v12_s1b_llama_sm_l30_20260327_194712.json")
    combined = _load_json(
        JSON_DIR / "v12_s1c_llama_sm_l22+l25+l30_20260328_013858.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(
        "Figure 8: Multi-Layer Steering Amplifies Effect 2.5x",
        fontsize=13, fontweight="bold", y=1.02)

    # ---- Panel (a): Dose-response curves ----
    layer_data = [
        ("L22", l22_data, BLUE, "o"),
        ("L25", l25_data, GREEN, "s"),
        ("L30", l30_data, PINK, "^"),
        ("L22+L25+L30", combined, ORANGE, "D"),
    ]

    for label, d, color, marker in layer_data:
        bk_dir = d["bk_direction"]["results"]
        a_keys = sorted(bk_dir.keys(), key=float)
        a_vals = [float(a) for a in a_keys]
        bk_vals = [bk_dir[a]["bk_rate"] for a in a_keys]
        rho = d["bk_direction"]["rho"]
        p = d["bk_direction"]["p"]
        lw = 2.5 if "+" in label else 1.5
        lbl = f"{label} (rho={rho:.2f}, p={p:.3f})"

        ax1.plot(a_vals, bk_vals, f"-{marker}", color=color,
                 linewidth=lw, markersize=6, label=lbl, alpha=0.9)

    ax1.set_xlabel("Steering Strength (alpha)")
    ax1.set_ylabel("Bankruptcy Rate")
    ax1.set_title("(a) Dose-Response by Layer", fontweight="bold")
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=8)
    ax1.set_xlim(-2.5, 2.5)
    ax1.set_ylim(0.1, 0.9)

    # ---- Panel (b): Effect sizes bar chart ----
    # Effect = BK at alpha=+2 minus baseline
    labels = ["L22\n(single)", "L25\n(single)", "L30\n(single)",
              "L22+L25+L30\n(combined)"]
    effects = []
    baselines = []
    for d in [l22_data, l25_data, l30_data, combined]:
        bl = d["baseline"]["bk_rate"]
        bk_plus2 = d["bk_direction"]["results"]["2.0"]["bk_rate"]
        effects.append(bk_plus2 - bl)
        baselines.append(bl)

    colors = [BLUE, GREEN, PINK, ORANGE]
    x = np.arange(len(labels))
    bars = ax2.bar(x, effects, 0.55, color=colors, edgecolor="white",
                   linewidth=0.5)

    # Value labels
    for i, (bar, eff) in enumerate(zip(bars, effects)):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"+{eff:.2f}" if eff > 0 else f"{eff:.2f}",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax2.set_ylabel("Effect Size\n(BK rate at alpha=+2 minus baseline)")
    ax2.set_title("(b) Effect Size Comparison", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.axhline(y=0, color="black", linewidth=0.5)

    # Annotation: amplification
    if effects[0] != 0:
        ratio = effects[3] / effects[0]
        ax2.annotate(
            f"Combined effect\n{ratio:.1f}x single layer",
            xy=(3, effects[3]), xytext=(1.5, effects[3] + 0.12),
            fontsize=9, ha="center", color=ORANGE, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3E0",
                      edgecolor=ORANGE, alpha=0.9),
        )

    fig.tight_layout()
    return _save_fig(fig, "v13_fig8_multilayer.png")


# ===========================================================================
# FIGURE 9: Overall Verdict Matrix
# ===========================================================================

def fig9_verdict_matrix() -> Path:
    """Model x Task x Evidence matrix with color coding."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.suptitle(
        "Figure 9: Convergent Evidence Across Methods, "
        "with Clear Boundary Conditions",
        fontsize=12, fontweight="bold", y=1.02)
    fig.subplots_adjust(left=0.15)

    # Evidence types (columns)
    evidence_types = [
        "DP\nClassification\n(AUC)",
        "R1\nClassification\n(AUC)",
        "Cross-Domain\nTransfer\n(AUC)",
        "BK Neuron\nCount",
        "Causal\nSteering\n(|rho|)",
        "Cross-Domain\nSteering\n(|rho|)",
    ]

    # Models x Tasks (rows)
    row_labels = [
        "Gemma SM", "Gemma IC", "Gemma MW",
        "LLaMA SM", "LLaMA IC", "LLaMA MW",
    ]

    # Data matrix: (value_text, status)
    # status: "confirmed" (green), "partial" (yellow), "failed" (red),
    #         "not_tested" (gray)
    data = [
        # Gemma SM
        [("0.976", "confirmed"), ("0.909", "confirmed"),
         ("--", "not_tested"),  ("302/298", "confirmed"),
         ("0.512", "failed"), ("--", "not_tested")],
        # Gemma IC
        [("0.960", "confirmed"), ("0.866", "confirmed"),
         ("0.932", "confirmed"), ("--", "not_tested"),
         ("n/a", "failed"), ("--", "not_tested")],
        # Gemma MW
        [("0.966", "confirmed"), ("0.864", "confirmed"),
         ("0.867", "confirmed"), ("--", "not_tested"),
         ("1.000", "confirmed"), ("--", "not_tested")],
        # LLaMA SM
        [("0.974", "confirmed"), ("0.897", "confirmed"),
         ("--", "not_tested"), ("672/662", "confirmed"),
         ("0.964", "confirmed"), ("0.964", "confirmed")],
        # LLaMA IC
        [("0.954", "confirmed"), ("0.849", "confirmed"),
         ("0.680", "partial"), ("--", "not_tested"),
         ("0.991", "confirmed"), ("0.900", "confirmed")],
        # LLaMA MW
        [("0.963", "confirmed"), ("0.855", "confirmed"),
         ("0.805", "confirmed"), ("--", "not_tested"),
         ("0.955", "confirmed"), ("0.975", "confirmed")],
    ]

    n_rows = len(row_labels)
    n_cols = len(evidence_types)

    # Color mapping
    status_colors = {
        "confirmed": "#A5D6A7",   # green
        "partial": "#FFF59D",     # yellow
        "failed": "#EF9A9A",     # red
        "not_tested": "#E0E0E0",  # gray
    }

    # Draw the matrix
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(-0.5, n_rows - 0.5)
    ax.invert_yaxis()

    for i in range(n_rows):
        for j in range(n_cols):
            text, status = data[i][j]
            color = status_colors[status]

            rect = plt.Rectangle((j - 0.48, i - 0.45), 0.96, 0.9,
                                 facecolor=color, edgecolor="white",
                                 linewidth=1.5)
            ax.add_patch(rect)

            text_color = "black" if status != "not_tested" else GRAY
            fontw = "bold" if status == "confirmed" else "normal"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=9, color=text_color, fontweight=fontw)

    # Divider between Gemma and LLaMA
    ax.axhline(y=2.5, color="black", linewidth=2, xmin=-0.05, xmax=1.05)

    # Model labels on left
    ax.text(-1.0, 1.0, "Gemma\n-2-9B", ha="center", va="center",
            fontsize=11, fontweight="bold", color=BLUE)
    ax.text(-1.0, 4.0, "LLaMA\n-3.1-8B", ha="center", va="center",
            fontsize=11, fontweight="bold", color=ORANGE)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(evidence_types, fontsize=8, ha="center")
    ax.set_yticks(range(n_rows))
    task_short = ["SM", "IC", "MW", "SM", "IC", "MW"]
    ax.set_yticklabels(task_short, fontsize=10)
    ax.tick_params(axis="both", length=0)

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Legend
    legend_items = [
        mpatches.Patch(facecolor="#A5D6A7", edgecolor="black",
                       label="Confirmed (p<0.05 or AUC>0.85)"),
        mpatches.Patch(facecolor="#FFF59D", edgecolor="black",
                       label="Partial (mixed evidence)"),
        mpatches.Patch(facecolor="#EF9A9A", edgecolor="black",
                       label="Failed / Not significant"),
        mpatches.Patch(facecolor="#E0E0E0", edgecolor="black",
                       label="Not tested"),
    ]
    ax.legend(handles=legend_items, loc="lower center",
              bbox_to_anchor=(0.5, -0.25), ncol=4, fontsize=8,
              framealpha=0.9)

    return _save_fig(fig, "v13_fig9_verdict_matrix.png")


# ===========================================================================
# Main
# ===========================================================================

def main():
    """Generate all V13 figures."""
    print("=" * 60)
    print("V13 Figure Generator")
    print("=" * 60)

    generated = []

    print("\n[Fig 0] Experimental Framework Overview...")
    generated.append(fig0_experimental_framework())

    print("\n[Fig 1] BK Classification Summary...")
    generated.append(fig1_bk_classification())

    print("\n[Fig 2] Universal BK Neurons...")
    generated.append(fig2_universal_neurons())

    print("\n[Fig 3] Cross-Domain Transfer Heatmap...")
    generated.append(fig3_crossdomain_transfer())

    print("\n[Fig 4] Cross-Domain Steering Transfer...")
    generated.append(fig4_crossdomain_steering())

    print("\n[Fig 7] Direction Steering Dose-Response...")
    generated.append(fig7_dose_response())

    print("\n[Fig 8] Multi-Layer Amplification...")
    generated.append(fig8_multilayer())

    print("\n[Fig 9] Overall Verdict Matrix...")
    generated.append(fig9_verdict_matrix())

    print("\n" + "=" * 60)
    print(f"Generated {len(generated)} figures:")
    for p in generated:
        print(f"  {p}")
    print("=" * 60)


if __name__ == "__main__":
    main()
