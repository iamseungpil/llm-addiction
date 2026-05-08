#!/usr/bin/env python3
"""Build a single full-width 1x3 neural-readout figure that replaces the
former Fig 5 (cross-paradigm) + Fig 6 (1x2) split.

Panels:
  (a) I_LC readout R^2 across tasks (Gemma L24 vs LLaMA L16, paired bars)
  (b) Cross-paradigm transfer matrix (Gemma I_LC, L24)
  (c) I_BA condition modulation in slot machine (Gemma vs LLaMA)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).parent))
from paper_figure_style import COLORS, annotate_bars, panel_title, save_pdf_png, style_axes, use_paper_style


ROOT = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis")
AUDIT_PATH = ROOT / "results" / "paper_neural_audit.json"
OUT_DIR = Path("/home/v-seungplee/LLM_Addiction_NMT_KOR/images")


def main() -> None:
    audit = json.loads(AUDIT_PATH.read_text())
    use_paper_style(11.8)
    gemma = COLORS["gemma"]
    llama = COLORS["llama"]

    fig, axes = plt.subplots(
        1, 3,
        figsize=(15.0, 3.7),
        gridspec_kw={"width_ratios": [1.05, 0.85, 1.10]},
    )
    ax_ilc, ax_cross, ax_iba = axes

    # -----------------------------------------------------------------------
    # (a) I_LC readout R^2 across tasks
    # -----------------------------------------------------------------------
    tasks = ["SM", "IC", "MW"]
    g_ilc = [audit["rq1_ilc"][f"gemma_{t.lower()}"]["r2"] for t in tasks]
    l_ilc = [audit["rq1_ilc"][f"llama_{t.lower()}"]["r2"] for t in tasks]
    x = np.arange(len(tasks))
    w = 0.42
    bars_g = ax_ilc.bar(x - w / 2, g_ilc, w, label="Gemma-2-9B (L24)",
                        color=gemma, edgecolor="black", linewidth=0.9)
    bars_l = ax_ilc.bar(x + w / 2, l_ilc, w, label="LLaMA-3.1-8B (L16)",
                        color=llama, edgecolor="black", linewidth=0.9)
    ax_ilc.set_xticks(x)
    ax_ilc.set_xticklabels(tasks)
    ax_ilc.set_ylabel(r"$R^2$")
    panel_title(ax_ilc, "(a)", r"I$_{LC}$ readout across tasks")
    ymax_a = max(g_ilc + l_ilc)
    ax_ilc.set_ylim(0, ymax_a * 1.22)
    annotate_bars(ax_ilc, bars_g, fmt="{:.2f}", size=10.0)
    annotate_bars(ax_ilc, bars_l, fmt="{:.2f}", size=10.0)
    ax_ilc.axhline(0, color="black", lw=0.8, alpha=0.35)
    ax_ilc.legend(fontsize=10, loc="upper left", frameon=False)
    style_axes(ax_ilc)

    # -----------------------------------------------------------------------
    # (b) Cross-paradigm transfer matrix (Gemma I_LC, L24)
    # -----------------------------------------------------------------------
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
    im = ax_cross.imshow(display, cmap=cmap, vmin=-0.5, vmax=0.6, aspect="equal")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = "white" if val <= -0.25 or val >= 0.38 else "black"
            weight = "bold" if i == j else "normal"
            label = f"{val:.2f}" if abs(val) < 1 else f"{val:.1f}"
            ax_cross.text(j, i, label, ha="center", va="center",
                          color=color, fontsize=11.5, fontweight=weight)
    ax_cross.set_xticks(range(3))
    ax_cross.set_yticks(range(3))
    ax_cross.set_xticklabels(paradigms)
    ax_cross.set_yticklabels(paradigms)
    ax_cross.set_xlabel("Test paradigm")
    ax_cross.set_ylabel("Train paradigm")
    panel_title(ax_cross, "(b)", "Cross-paradigm transfer (Gemma)")
    ax_cross.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax_cross.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax_cross.grid(which="minor", color="#FFFFFF", linestyle="-", linewidth=1.2)
    ax_cross.tick_params(which="minor", bottom=False, left=False)
    cbar = plt.colorbar(im, ax=ax_cross, shrink=0.78, pad=0.02)
    cbar.set_label(r"$R^2$", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # -----------------------------------------------------------------------
    # (c) I_BA condition modulation in slot machine
    # -----------------------------------------------------------------------
    cond_labels = ["All\nVar", "+G", "-G", "Fixed"]
    g_cond = audit["rq3_condition_i_ba"]["gemma_sm_i_ba"]["subsets"]
    l_cond = audit["rq3_condition_i_ba"]["llama_sm_i_ba"]["subsets"]
    g_vals = [g_cond["all_variable"]["r2"], g_cond["plus_G"]["r2"],
              g_cond["minus_G"]["r2"], g_cond["fixed_all"]["r2"]]
    l_vals = [l_cond["all_variable"]["r2"], l_cond["plus_G"]["r2"],
              l_cond["minus_G"]["r2"], l_cond["fixed_all"]["r2"]]
    x2 = np.arange(len(cond_labels))
    w2 = 0.42
    bars_gm = ax_iba.bar(x2 - w2 / 2, g_vals, w2, label="Gemma",
                         color=gemma, edgecolor="black", linewidth=0.9)
    bars_lm = ax_iba.bar(x2 + w2 / 2, l_vals, w2, label="LLaMA",
                         color=llama, edgecolor="black", linewidth=0.9)
    ax_iba.set_xticks(x2)
    ax_iba.set_xticklabels(cond_labels)
    ax_iba.set_ylabel(r"$R^2$")
    panel_title(ax_iba, "(c)", r"Condition modulation of I$_{BA}$")
    ax_iba.axhline(0, color="black", lw=0.8, alpha=0.35)
    ax_iba.legend(fontsize=10, loc="upper right", frameon=False)
    ymax_c = max(g_vals + l_vals)
    ax_iba.set_ylim(-0.045, ymax_c * 1.30)
    annotate_bars(ax_iba, bars_gm, fmt="{:.2f}", size=10.0)
    annotate_bars(ax_iba, bars_lm, fmt="{:.2f}", size=10.0)
    style_axes(ax_iba)

    # +G vs -G bracket annotation on Gemma
    bracket_y = ymax_c * 1.12
    ax_iba.plot([1 - w2 / 2, 2 - w2 / 2], [bracket_y, bracket_y],
                color=gemma, linewidth=1.2, clip_on=False)
    ax_iba.plot([1 - w2 / 2, 1 - w2 / 2], [bracket_y, bracket_y - ymax_c * 0.03],
                color=gemma, linewidth=1.2, clip_on=False)
    ax_iba.plot([2 - w2 / 2, 2 - w2 / 2], [bracket_y, bracket_y - ymax_c * 0.03],
                color=gemma, linewidth=1.2, clip_on=False)
    ratio = g_cond["plus_G"]["r2"] / g_cond["minus_G"]["r2"]
    ax_iba.text(1.5 - w2 / 2, bracket_y + ymax_c * 0.02,
                f"{ratio:.1f}× with goal",
                ha="center", va="bottom", fontsize=10, color=gemma,
                fontweight="bold")

    save_pdf_png(fig, OUT_DIR, "neural_readout_panel")
    plt.close(fig)
    print("Saved neural_readout_panel.{pdf,png}")


if __name__ == "__main__":
    main()
