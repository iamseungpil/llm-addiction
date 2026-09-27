#!/usr/bin/env python3
"""Build the paper's Figure 1 (overview) from scratch using matplotlib.

Insight-driven narrative (vs data-listing):
- LEFT:  ONE autonomy knob flips behavior from "safe stop" to "bankruptcy cascade"
         across six LLMs. The knob is the story, not the game counts.
- RIGHT: Three gambling tasks CONVERGE into one shared hidden-state risk axis,
         then FAN OUT into task-specific SAE readouts. The shape is the story.
- BOTTOM: single thesis band connecting both halves.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


# The unreleased ``paper_figure_style`` helper lives in the separate analysis
# tree, not in this repository and not on the HuggingFace dataset.  Point
# ``LLM_ADDICTION_ANALYSIS`` at that checkout to re-run this generator; a
# five-name subset is vendored at ``scripts/figures/paper_style_vendored.py``
# for the generators that were ported to it.
ANALYSIS_ROOT = Path(os.environ.get("LLM_ADDICTION_ANALYSIS", Path.home() / "llm-addiction"))
STYLE_SRC = ANALYSIS_ROOT / "experiments" / "07_sae_readout" / "src"
if str(STYLE_SRC) not in sys.path:
    sys.path.insert(0, str(STYLE_SRC))

from paper_figure_style import COLORS, save_pdf_png, use_paper_style  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "images"


C_FIXED = COLORS["fixed"]          # green  #59A14F
C_VAR = COLORS["variable"]         # red    #E15759
C_GEMMA = COLORS["gemma"]          # orange #F28E2B
C_LLAMA = COLORS["llama"]          # blue   #4E79A7
C_NEUTRAL = COLORS["neutral"]      # gray   #9D9D9D
C_PANEL = "#F4F4F4"
C_INK = "#1F1F1F"
C_MUTED = "#6B6B6B"


def _rounded_box(ax, x, y, w, h, *, fc, ec=C_INK, lw=0.9, alpha=1.0, rounding=0.015):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={rounding}",
        facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha,
    )
    ax.add_patch(box)
    return box


def _arrow(ax, x1, y1, x2, y2, *, color=C_INK, lw=1.4, mutation_scale=14, style="->"):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=mutation_scale,
        color=color, linewidth=lw, shrinkA=2, shrinkB=2,
    )
    ax.add_patch(arr)
    return arr


def _text(ax, x, y, text, *, size=10, color=C_INK, weight="normal", ha="center", va="center", style="normal"):
    ax.text(x, y, text, fontsize=size, color=color, fontweight=weight,
            ha=ha, va=va, fontstyle=style)


def _draw_left_panel(ax):
    """Phase 1: autonomy knob + divergent behavior."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Panel title
    _text(ax, 5, 5.65, "Phase 1 · Behavior", size=12, weight="bold")
    _text(ax, 5, 5.25, "One autonomy knob flips safe stopping into bankruptcy cascade",
          size=8.8, color=C_MUTED, style="italic")

    # Subtitle row (ABOVE the chip strip)
    _text(ax, 5, 4.95, "Six LLMs  ·  slot machine  ·  investment choice",
          size=8.0, color=C_MUTED)

    # Six LLMs as a single strip
    llm_names = ["GPT-4o-mini", "GPT-4.1-mini", "Gemini-2.5-Flash",
                 "Claude-3.5-Haiku", "LLaMA-3.1-8B", "Gemma-2-9B"]
    strip_y = 4.45
    strip_h = 0.38
    left = 0.25
    right = 9.75
    cell_w = (right - left) / 6
    for i, name in enumerate(llm_names):
        cx = left + (i + 0.5) * cell_w
        _rounded_box(ax, left + i * cell_w + 0.06, strip_y, cell_w - 0.12, strip_h,
                     fc="#FFFFFF", ec="#BDBDBD", lw=0.7, rounding=0.04)
        _text(ax, cx, strip_y + strip_h / 2, name, size=7.2, color=C_INK)

    # Autonomy knob — a rounded box with two pill toggles
    knob_y = 3.3
    knob_h = 0.7
    _rounded_box(ax, 2.2, knob_y, 5.6, knob_h, fc=C_PANEL, ec="#BDBDBD", lw=0.7, rounding=0.06)
    _text(ax, 5, knob_y + knob_h + 0.16, "Autonomy knob", size=9.2, weight="bold")

    # Left pill (green, Fixed)
    _rounded_box(ax, 2.45, knob_y + 0.15, 2.3, knob_h - 0.3,
                 fc=C_FIXED, ec=C_INK, lw=0.9, rounding=0.08)
    _text(ax, 3.6, knob_y + knob_h / 2 + 0.05, "Low autonomy",
          size=8.6, weight="bold", color="white")
    _text(ax, 3.6, knob_y + knob_h / 2 - 0.18, "Fixed bet · External goal",
          size=7.4, color="white")

    # Right pill (red, Variable)
    _rounded_box(ax, 5.25, knob_y + 0.15, 2.3, knob_h - 0.3,
                 fc=C_VAR, ec=C_INK, lw=0.9, rounding=0.08)
    _text(ax, 6.4, knob_y + knob_h / 2 + 0.05, "High autonomy",
          size=8.6, weight="bold", color="white")
    _text(ax, 6.4, knob_y + knob_h / 2 - 0.18, "Variable bet · Self-set goal",
          size=7.4, color="white")

    # Arrow from low knob to safe outcome
    _arrow(ax, 3.6, knob_y, 2.0, 2.0, color=C_FIXED, lw=1.6)
    _arrow(ax, 6.4, knob_y, 8.0, 2.0, color=C_VAR, lw=1.6)

    # Outcome boxes
    out_y = 0.6
    out_h = 1.35
    # Low-autonomy outcome (green)
    _rounded_box(ax, 0.4, out_y, 3.2, out_h, fc="#E9F3E3", ec=C_FIXED, lw=1.3, rounding=0.06)
    _text(ax, 2.0, out_y + out_h - 0.28, "Safe stop", size=10.4, weight="bold", color=C_FIXED)
    # Mini bankruptcy bar (low)
    _draw_bank_bar(ax, cx=2.0, cy=out_y + 0.4, value=0.04, color=C_FIXED,
                   label="≈ 0% bankruptcy")

    # High-autonomy outcome (red)
    _rounded_box(ax, 6.4, out_y, 3.2, out_h, fc="#FBEAEA", ec=C_VAR, lw=1.3, rounding=0.06)
    _text(ax, 8.0, out_y + out_h - 0.28, "Bankruptcy cascade", size=10.4, weight="bold", color=C_VAR)
    # Mini bankruptcy bar (high)
    _draw_bank_bar(ax, cx=8.0, cy=out_y + 0.4, value=0.72, color=C_VAR,
                   label="up to 72% bankruptcy (LLaMA)")


def _draw_bank_bar(ax, *, cx, cy, value, color, label):
    bar_w = 2.4
    bar_h = 0.16
    fill_w = bar_w * value
    # Track
    ax.add_patch(mpatches.Rectangle(
        (cx - bar_w / 2, cy - bar_h / 2), bar_w, bar_h,
        facecolor="#FFFFFF", edgecolor="#BDBDBD", linewidth=0.7,
    ))
    # Fill
    ax.add_patch(mpatches.Rectangle(
        (cx - bar_w / 2, cy - bar_h / 2), fill_w, bar_h,
        facecolor=color, edgecolor="none",
    ))
    _text(ax, cx, cy - 0.36, label, size=7.8, color=C_INK)


def _draw_right_panel(ax):
    """Phase 2: 3 tasks converge into shared axis, fan out to readouts."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    _text(ax, 5, 5.65, "Phase 2 · Internal Representation", size=12, weight="bold")
    _text(ax, 5, 5.25, "Three tasks share one risk axis, but each task reads it differently",
          size=8.8, color=C_MUTED, style="italic")

    # Two model chips at the top
    _rounded_box(ax, 1.3, 4.45, 3.0, 0.52, fc="#FFFFFF", ec=C_GEMMA, lw=1.5, rounding=0.06)
    _text(ax, 2.8, 4.71, "Gemma-2-9B · L24", size=9.2, weight="bold", color=C_GEMMA)
    _rounded_box(ax, 5.7, 4.45, 3.0, 0.52, fc="#FFFFFF", ec=C_LLAMA, lw=1.5, rounding=0.06)
    _text(ax, 7.2, 4.71, "LLaMA-3.1-8B · L16", size=9.2, weight="bold", color=C_LLAMA)

    # Three task chips on the left (column)
    task_x = 0.3
    task_w = 1.5
    task_h = 0.46
    tasks = [("Slot Machine", "SM"), ("Investment", "IC"), ("Mystery Wheel", "MW")]
    task_centers = []
    for i, (full, short) in enumerate(tasks):
        y = 3.3 - i * 0.7
        _rounded_box(ax, task_x, y, task_w, task_h, fc="#FFFFFF", ec="#BDBDBD", lw=0.8, rounding=0.06)
        _text(ax, task_x + task_w / 2, y + task_h / 2 + 0.07, short, size=9.2, weight="bold")
        _text(ax, task_x + task_w / 2, y + task_h / 2 - 0.12, full, size=7.0, color=C_MUTED)
        task_centers.append((task_x + task_w, y + task_h / 2))

    # Converge arrows into central shared axis
    axis_x = 4.3
    axis_y = 2.55
    axis_w = 1.4
    axis_h = 1.0
    # Shared axis box with a single bold horizontal axis line inside (the rank-1 direction)
    _rounded_box(ax, axis_x, axis_y, axis_w, axis_h, fc="#F7F7F7", ec=C_INK, lw=1.1, rounding=0.05)
    # Scatter dots clustered along a single direction (cartoon of rank-1 structure)
    rng = np.random.default_rng(7)
    n = 22
    t = rng.uniform(-0.48, 0.48, n)
    jitter = rng.normal(0, 0.07, n)
    cx_a = axis_x + axis_w / 2
    cy_a = axis_y + axis_h / 2
    ax.scatter(cx_a + t, cy_a + jitter, s=8, color=C_INK, alpha=0.55, linewidths=0, zorder=3)
    # Rank-1 axis line
    ax.plot([axis_x + 0.15, axis_x + axis_w - 0.15],
            [cy_a, cy_a], color=C_VAR, linewidth=1.6, alpha=0.9, zorder=4)
    _text(ax, cx_a, axis_y + axis_h + 0.18,
          "Shared hidden-state risk axis", size=8.6, weight="bold")
    _text(ax, cx_a, axis_y - 0.18, "(rank 1 direction)", size=7.6, color=C_MUTED, style="italic")

    # Converging arrows
    for (tx, ty) in task_centers:
        _arrow(ax, tx + 0.1, ty, axis_x, axis_y + axis_h / 2 + (ty - axis_y - axis_h / 2) * 0.1,
               color=C_MUTED, lw=1.2)

    # Fan-out readouts on the right
    read_x = 7.3
    read_w = 2.3
    read_h = 0.46
    readouts = ["SAE readout · SM", "SAE readout · IC", "SAE readout · MW"]
    read_centers = []
    for i, lbl in enumerate(readouts):
        y = 3.3 - i * 0.7
        _rounded_box(ax, read_x, y, read_w, read_h, fc="#FFFFFF", ec=C_INK, lw=0.9, rounding=0.06)
        _text(ax, read_x + read_w / 2, y + read_h / 2, lbl, size=8.6)
        read_centers.append((read_x, y + read_h / 2))

    # Diverging arrows out of axis
    axis_right = axis_x + axis_w
    for (rx, ry) in read_centers:
        _arrow(ax, axis_right, axis_y + axis_h / 2 + (ry - axis_y - axis_h / 2) * 0.1,
               rx - 0.05, ry, color=C_MUTED, lw=1.2)

    # Small caption below the converge-diverge diagram
    _text(ax, 5, 0.85, "Partial convergence  →  task-specific readouts",
          size=8.8, color=C_INK, style="italic")
    _text(ax, 5, 0.55, "(R² = 0.24–0.78 across 6 model×task pairs)",
          size=7.6, color=C_MUTED)


def _draw_thesis_band(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _rounded_box(ax, 0.2, 0.15, 9.6, 0.7, fc="#EDEDED", ec="#BDBDBD", lw=0.7, rounding=0.08)
    _text(ax, 5, 0.5,
          "Thesis:  Autonomy organizes risk — both in behavior (outside) and in internal representation (inside).",
          size=10.4, weight="bold", color=C_INK)


def build_overview() -> Path:
    use_paper_style(10.0)
    fig = plt.figure(figsize=(13.5, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[5.2, 1.2], width_ratios=[1, 1],
                          hspace=0.05, wspace=0.07)

    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])
    ax_band = fig.add_subplot(gs[1, :])

    _draw_left_panel(ax_left)
    _draw_right_panel(ax_right)
    _draw_thesis_band(ax_band)

    # Subtle vertical separator between the two top panels
    fig.canvas.draw()
    sep = fig.add_axes([0.499, 0.28, 0.002, 0.6])
    sep.axvline(0.5, color="#CFCFCF", linewidth=0.7)
    sep.set_xlim(0, 1); sep.set_ylim(0, 1); sep.axis("off")

    save_pdf_png(fig, OUT_DIR, "representative_flow_diagram")
    plt.close(fig)
    return OUT_DIR / "representative_flow_diagram.pdf"


if __name__ == "__main__":
    out = build_overview()
    print(f"Wrote {out}")
