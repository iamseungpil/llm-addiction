"""Body Figure 4 in the house style of Figures 2 and 3.

Draws ``images/fig04_causal_battery.pdf`` / ``.png`` from the sidecar
``paper_data/fig04_causal_battery.json`` that ``fig04_causal_battery.py``
writes; no number is recomputed here.  What changes against that generator is
only the drawing:

* The canvas is the Figure 2 canvas (1274.435 pt wide, the same axes height; 300 pt tall, since no rotated names sit under the axes),
  so the figure prints at the same type sizes as Figures 2 and 3 when it is
  included at ``\\textwidth``: 17 pt bold titles, 15 pt labels and ticks, 14 pt
  legend, 13 pt annotations.
* The colours come from the same Tableau set: the behaviour-built direction,
  which raises betting, is drawn in the red of the risk-raising arm; the
  balance direction, which restrains betting on Gemma, in the green of the
  restrained arm; the readout direction in the grey Figure 3 uses for its
  neutral option; the random-direction band in light grey.  Model identity is
  carried by the panel titles, not by colour.
* The parse-rate strips are dropped.  A dose whose share of readable wagers
  falls below the parse gate is still drawn, hollow, and still left out of the
  slope, exactly as before; the per-dose parse rates stay in the sidecar and in
  the appendix removal figure's generator.

Run from ``paper/`` after ``./link_paper_repo.sh``::

    python scripts/figures/fig04_causal_battery_paperstyle.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paper_style_vendored import COLORS, save_pdf_png, style_axes, use_paper_style  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REPO_ROOT = HERE.parents[1]
SIDECAR = REPO_ROOT / "paper_data" / "fig04_causal_battery.json"
IMAGES = REPO_ROOT / "images"

# Figure 2's canvas and type ladder (see fig02_slot_machine.py).
FIG_W_PT = 1274.435
FIG_H_PT = 300.0
AXES_BOT_PT = 52.0
AXES_H_PT = 221.58
PANELS_PT = {"a": (86.0, 520.0), "b": (740.0, 520.0)}
FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND, FS_NOTE = 17.0, 15.0, 15.0, 14.0, 13.0

STYLE = {
    "behavioural": dict(color=COLORS["variable"], marker="o", ls="-", label="Behaviour-built"),
    "confound": dict(color=COLORS["fixed"], marker="^", ls="--", label="Balance"),
    "readout": dict(color=COLORS["option3"], marker="s", ls="--", label="Readout"),
}
ORDER = ("behavioural", "confound", "readout")
BAND = "#D9D9D9"


def ladder(axis_block):
    doses = sorted(axis_block["doses"], key=float)
    x = np.array([float(d) for d in doses])
    m = np.array([axis_block["doses"][d]["mean_bet_ratio"] for d in doses])
    lo = np.array([axis_block["doses"][d]["ci95"][0] for d in doses])
    hi = np.array([axis_block["doses"][d]["ci95"][1] for d in doses])
    ok = np.array([axis_block["doses"][d]["passes_parse_gate"] for d in doses])
    return x, m, lo, hi, ok


def draw_panel(ax, panel, model):
    for dose, band in panel["null_band_by_dose"].items():
        d = float(dose)
        ax.add_patch(plt.Rectangle((d - 0.22, band["lo"]), 0.44, band["hi"] - band["lo"],
                                   facecolor=BAND, edgecolor="none", zorder=1))
        ax.plot([d - 0.22, d + 0.22], [band["mean"]] * 2, color="#9A9A9A", lw=1.4, zorder=2)
    ax.axvline(0.0, color="#BBBBBB", lw=1.0, zorder=0)
    ends = []
    for axis in ORDER:
        st = STYLE[axis]
        x, m, lo, hi, ok = ladder(panel["ladder"][axis])
        ax.plot(x, m, color=st["color"], ls=st["ls"], lw=2.2, zorder=3)
        ax.errorbar(x, m, yerr=[m - lo, hi - m], fmt="none", ecolor=st["color"],
                    elinewidth=1.2, capsize=2.5, zorder=4)
        for xi, mi, oki in zip(x, m, ok):
            ax.plot([xi], [mi], marker=st["marker"], ms=8.5, color=st["color"],
                    mfc=st["color"] if oki else "white", mew=1.6, zorder=5)
        blk = panel["ladder"][axis]
        if model == "gemma":
            note = f"{blk['slope_parse_gated']:+.3f}, z {blk['z_vs_null_slope_band']:+.2f}"
        else:
            note = f"{blk['slope_parse_gated']:+.3f}, z {blk['z_vs_null_level_at_plus3']:+.2f}"
        ends.append([m[-1], note, st["color"]])
    # annotations at the right end of each line, nudged apart
    ends.sort(key=lambda e: e[0])
    y0, y1 = ax.get_ylim()
    gap = 0.075 * (y1 - y0)
    for i in range(1, len(ends)):
        if ends[i][0] - ends[i - 1][0] < gap:
            ends[i][0] = ends[i - 1][0] + gap
    for y, note, c in ends:
        ax.text(3.32, y, note, color=c, fontsize=FS_NOTE, va="center", ha="left", zorder=6)
    ax.set_xlim(-3.45, 4.95)
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_xlabel("Steering dose", fontsize=FS_LABEL, labelpad=2.0)
    style_axes(ax)
    ax.spines["bottom"].set_bounds(-3.45, 3.25)


def main() -> None:
    d = json.loads(SIDECAR.read_text())
    use_paper_style(base=FS_LABEL)
    plt.rcParams.update({"axes.titlesize": FS_TITLE, "legend.fontsize": FS_LEGEND,
                         "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
                         "xtick.major.pad": 2.0, "ytick.major.pad": 2.0})
    fig = plt.figure(figsize=(FIG_W_PT / 72.0, FIG_H_PT / 72.0), dpi=72.0)
    axes = {}
    for k, (x0, w) in PANELS_PT.items():
        axes[k] = fig.add_axes([x0 / FIG_W_PT, AXES_BOT_PT / FIG_H_PT,
                                w / FIG_W_PT, AXES_H_PT / FIG_H_PT])
    axes["a"].set_ylim(-0.005, 0.30)
    axes["b"].set_ylim(0.08, 0.34)
    draw_panel(axes["a"], d["panel_a_gemma"], "gemma")
    draw_panel(axes["b"], d["panel_b_llama"], "llama")
    axes["a"].set_ylabel("Mean bet ratio", fontsize=FS_LABEL)
    axes["b"].set_ylabel("Mean bet ratio", fontsize=FS_LABEL)
    axes["a"].set_title("(a) Gemma, layers 16–21", fontweight="bold", fontsize=FS_TITLE, pad=4.0)
    axes["b"].set_title("(b) LLaMA, layers 14–19", fontweight="bold", fontsize=FS_TITLE, pad=4.0)
    handles = [Line2D([], [], color=STYLE[a]["color"], ls=STYLE[a]["ls"], lw=2.2,
                      marker=STYLE[a]["marker"], ms=7.5, label=STYLE[a]["label"]) for a in ORDER]
    handles += [Patch(facecolor=BAND, label="Random directions, mean ± 2 SD"),
                Line2D([], [], ls="none", marker="o", ms=7.5, mfc="white",
                       mec="#666666", mew=1.4, label="Too few readable wagers")]
    axes["a"].legend(handles=handles, loc="upper left", fontsize=FS_LEGEND, frameon=True,
                     framealpha=0.92, edgecolor="#CCCCCC", fancybox=False,
                     handlelength=2.2, borderpad=0.4, labelspacing=0.3).set_zorder(7)
    save_pdf_png(fig, IMAGES, "fig04_causal_battery")
    print(f"wrote {IMAGES / 'fig04_causal_battery.pdf'}")


if __name__ == "__main__":
    main()
