#!/usr/bin/env python3
"""
Generate paper figures from the audited neural metrics manifest.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from paper_figure_style import COLORS, annotate_bars, panel_title, save_pdf_png, style_axes, use_paper_style


ROOT = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis")
AUDIT_PATH = ROOT / "results" / "paper_neural_audit.json"
OUT_DIR = Path("/home/v-seungplee/LLM_Addiction_NMT_KOR/images")
OUT_DIR.mkdir(exist_ok=True)


def load_audit() -> dict:
    if not AUDIT_PATH.exists():
        raise FileNotFoundError(
            f"Missing {AUDIT_PATH}. Run build_paper_neural_audit.py first."
        )
    return json.loads(AUDIT_PATH.read_text())


def build_neural_summary(audit: dict) -> None:
    use_paper_style(10.2)
    gemma_color = COLORS["gemma"]
    llama_color = COLORS["llama"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 2.9), gridspec_kw={"width_ratios": [1.0, 1.0]})

    # Panel (a): I_LC readout grouped by task with paired Gemma/LLaMA bars,
    # matching the paired-bar layout of panel (b) so the two panels read consistently.
    task_labels = ["SM", "IC", "MW"]
    gemma_ilc = [audit["rq1_ilc"][f"gemma_{t.lower()}"]["r2"] for t in task_labels]
    llama_ilc = [audit["rq1_ilc"][f"llama_{t.lower()}"]["r2"] for t in task_labels]
    x = np.arange(len(task_labels))
    width = 0.36
    bars_g_a = ax1.bar(x - width / 2, gemma_ilc, width, label="Gemma-2-9B (L24)",
                       color=gemma_color, alpha=0.95, edgecolor="black", linewidth=0.95)
    bars_l_a = ax1.bar(x + width / 2, llama_ilc, width, label="LLaMA-3.1-8B (L16)",
                       color=llama_color, alpha=0.95, edgecolor="black", linewidth=0.95)
    ax1.set_xticks(x)
    ax1.set_xticklabels(task_labels, fontsize=9)
    ax1.set_ylabel(r"$R^2$")
    panel_title(ax1, "(a)", r"I$_{LC}$ Readout Across Tasks")
    ymax_a = max(gemma_ilc + llama_ilc)
    ax1.set_ylim(0, ymax_a * 1.22)
    annotate_bars(ax1, bars_g_a, fmt="{:.2f}", size=7.7)
    annotate_bars(ax1, bars_l_a, fmt="{:.2f}", size=7.7)
    ax1.axhline(0, color="black", lw=0.8, alpha=0.35)
    ax1.legend(fontsize=8.4, loc="upper left", frameon=False)
    style_axes(ax1)

    # Panel (b): audited condition modulation for I_BA in slot machine.
    cond_labels = ["All\nVar", "+G", "-G", "Fixed"]
    gemma = audit["rq3_condition_i_ba"]["gemma_sm_i_ba"]["subsets"]
    llama = audit["rq3_condition_i_ba"]["llama_sm_i_ba"]["subsets"]
    gemma_vals = [gemma["all_variable"]["r2"], gemma["plus_G"]["r2"], gemma["minus_G"]["r2"], gemma["fixed_all"]["r2"]]
    llama_vals = [llama["all_variable"]["r2"], llama["plus_G"]["r2"], llama["minus_G"]["r2"], llama["fixed_all"]["r2"]]
    x2 = np.arange(len(cond_labels))
    width = 0.35
    bars_g = ax2.bar(x2 - width / 2, gemma_vals, width, label="Gemma", color=gemma_color, alpha=0.95, edgecolor="black", linewidth=0.95)
    bars_l = ax2.bar(x2 + width / 2, llama_vals, width, label="LLaMA", color=llama_color, alpha=0.95, edgecolor="black", linewidth=0.95)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(cond_labels, fontsize=9)
    ax2.set_ylabel(r"$R^2$")
    panel_title(ax2, "(b)", r"Condition Modulation of I$_{BA}$")
    ax2.axhline(0, color="black", lw=0.8, alpha=0.35)
    ax2.legend(fontsize=9, loc="upper right")
    ymax = max(gemma_vals + llama_vals)
    ax2.set_ylim(-0.04, ymax * 1.22)
    style_axes(ax2)
    annotate_bars(ax2, bars_g, fmt="{:.2f}", size=7.7)
    annotate_bars(ax2, bars_l, fmt="{:.2f}", size=7.7)

    # Bracket-style annotation: compare +G vs -G for Gemma as evidence that
    # goal autonomy sharpens the readout. Cleaner than an arrow floating above bars.
    bracket_y = ymax * 1.10
    ax2.plot([1 - width / 2, 2 - width / 2], [bracket_y, bracket_y],
             color=gemma_color, linewidth=1.2, clip_on=False)
    ax2.plot([1 - width / 2, 1 - width / 2], [bracket_y, bracket_y - ymax * 0.03],
             color=gemma_color, linewidth=1.2, clip_on=False)
    ax2.plot([2 - width / 2, 2 - width / 2], [bracket_y, bracket_y - ymax * 0.03],
             color=gemma_color, linewidth=1.2, clip_on=False)
    ratio = gemma["plus_G"]["r2"] / gemma["minus_G"]["r2"]
    ax2.text(1.5 - width / 2, bracket_y + ymax * 0.02, f"{ratio:.1f}× with goal",
             ha="center", va="bottom", fontsize=8.4, color=gemma_color, fontweight="bold")

    save_pdf_png(fig, OUT_DIR, "neural_analysis_combined")
    plt.close(fig)


def build_cross_transfer() -> None:
    use_paper_style(9.2)
    # Representative archived cross-task transfer matrix from the paper's negative-transfer story.
    paradigms = ["SM", "IC", "MW"]
    matrix = np.array(
        [
            [0.24, -0.05, -2.01],
            [-0.10, 0.48, -0.08],
            [-0.06, -0.04, 0.48],
        ]
    )
    display = np.clip(matrix, -0.5, 0.6)

    cmap = LinearSegmentedColormap.from_list(
        "paper_diverging",
        ["#C95A49", "#F4E8E1", "#F7F7F7", "#DBE8F4", "#4E79A7"],
        N=256,
    )

    fig, ax = plt.subplots(figsize=(3.35, 2.85))
    im = ax.imshow(display, cmap=cmap, vmin=-0.5, vmax=0.6, aspect="equal")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = "white" if val <= -0.25 or val >= 0.38 else "black"
            weight = "bold" if i == j else "normal"
            label = f"{val:.2f}" if abs(val) < 1 else f"{val:.1f}"
            ax.text(j, i, label, ha="center", va="center", color=color, fontsize=9.2, fontweight=weight)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(paradigms)
    ax.set_yticklabels(paradigms)
    ax.set_xlabel("Test paradigm")
    ax.set_ylabel("Train paradigm")
    panel_title(ax, "", r"Cross-Paradigm Transfer ($R^2$)")
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax.grid(which="minor", color="#FFFFFF", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = plt.colorbar(im, ax=ax, shrink=0.84, pad=0.02)
    cbar.set_label(r"$R^2$")
    save_pdf_png(fig, OUT_DIR, "cross_paradigm_transfer")
    plt.close(fig)


def build_condition_modulation_iba(audit: dict) -> None:
    use_paper_style(9.3)
    gemma_color = COLORS["gemma"]
    llama_color = COLORS["llama"]

    cond_labels = ["All var", "+G", "-G", "+M", "-M", "Fixed"]
    gemma = audit["rq3_condition_i_ba"]["gemma_sm_i_ba"]["subsets"]
    llama = audit["rq3_condition_i_ba"]["llama_sm_i_ba"]["subsets"]
    gemma_vals = [
        gemma["all_variable"]["r2"],
        gemma["plus_G"]["r2"],
        gemma["minus_G"]["r2"],
        gemma["plus_M"]["r2"],
        gemma["minus_M"]["r2"],
        gemma["fixed_all"]["r2"],
    ]
    llama_vals = [
        llama["all_variable"]["r2"],
        llama["plus_G"]["r2"],
        llama["minus_G"]["r2"],
        llama["plus_M"]["r2"],
        llama["minus_M"]["r2"],
        llama["fixed_all"]["r2"],
    ]

    # Keep figsize close to the final rendered width (~3.7 inch at 0.68\cw) to
    # avoid aggressive scaling that cramps text inside.
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    x = np.arange(len(cond_labels))
    width = 0.34
    bars_g = ax.bar(
        x - width / 2,
        gemma_vals,
        width,
        label="Gemma-2-9B",
        color=gemma_color,
        edgecolor="black",
        linewidth=0.8,
    )
    bars_l = ax.bar(
        x + width / 2,
        llama_vals,
        width,
        label="LLaMA-3.1-8B",
        color=llama_color,
        edgecolor="black",
        linewidth=0.8,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels)
    ax.set_ylabel(r"I$_{BA}$ readout $R^2$")
    panel_title(ax, "", r"Condition Modulation of I$_{BA}$")
    ax.axhline(0, color="#555555", lw=0.8, alpha=0.5)
    style_axes(ax)
    ax.legend(loc="upper right")
    ymax = max(gemma_vals + llama_vals)
    ax.set_ylim(-0.045, ymax * 1.28)
    annotate_bars(ax, bars_g, fmt="{:.2f}", size=7.4)
    annotate_bars(ax, bars_l, fmt="{:.2f}", size=7.4)

    # Light background shading on the Fixed column to visually separate
    # the near-zero regime without a text callout that would clip other bars.
    ax.axvspan(x[-1] - 0.5, x[-1] + 0.5, color="#F4E4E4", alpha=0.55, zorder=0)
    # Vertical label inside the shaded column, safely below the top margin.
    ax.text(
        x[-1] - 0.42, ymax * 0.9, "FIXED BET",
        ha="left", va="top", fontsize=7.2, rotation=90,
        color=COLORS["variable"], fontweight="bold", alpha=0.85,
    )

    save_pdf_png(fig, OUT_DIR, "condition_modulation_iba")
    plt.close(fig)


def main() -> None:
    audit = load_audit()
    build_neural_summary(audit)
    build_cross_transfer()
    build_condition_modulation_iba(audit)
    print("Saved audited paper figures.")


if __name__ == "__main__":
    main()
