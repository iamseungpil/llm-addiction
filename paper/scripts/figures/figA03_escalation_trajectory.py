"""Figure A3 -- within-game bet escalation trajectory (images/escalation_trajectory.pdf).

SUPERSEDED -- DO NOT RERUN WITHOUT RE-DECIDING.
``images/escalation_trajectory.pdf`` in the working tree is the *submitted*
artwork, restored byte for byte from commit dd2d229.  This print-width re-
authoring was compared against it panel by panel and carried nothing the
submission did not already have (its extracted text is identical to the
submission's, character for character), so it lost on the standing rule that
figures which are not rebuttal experiments must match the submitted version.
Running this script overwrites the submitted file.

Included at ``width=\\paperfigmid`` = 0.65\\textwidth.  Measured off the compiled
neurips_en.pdf the placement scale is 0.53212 on a 483.73 pt canvas, i.e. the
figure is printed 257.40 pt = 3.575 in wide (\\textwidth is exactly 396 pt = 5.5 in).
That shrank the submitted 10.5 pt ticks to 5.6 pt on paper.  This regeneration
authors the canvas at 3.575 x 2.30 in, i.e. scale 1.0.

Composition is unchanged: one axes, Fixed (green) and Variable (red) mean
bet/balance ratio against normalised round, +/- 1 SEM error bars, mean per-game
Spearman rho quoted in the legend.

Data: ``data/figA03_escalation_trajectory.json``.  The camera-ready art on HF
(``paper_neurips_2026/camera_ready/figures/escalation_trajectory.pdf``) has no
released generator -- the HF script ``figA03_escalation/code/run_escalation_analysis.py``
plots an earlier, differently styled version and needs a raw LLaMA slot-machine
export that is not in this repository.  The ten bin means and SEMs per series were
therefore read back out of the submitted PDF's vector geometry (exact to the
0.0001 the y-axis can resolve) and are stored in the sidecar.

Run:  python3 scripts/figures/figA03_escalation_trajectory.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from paper_style_appendix import COLORS, panel_title, save_pdf_png, style_axes, use_paper_style

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DATA = HERE / "data" / "figA03_escalation_trajectory.json"
OUT_DIR = REPO / "images"

# 0.65 * 396 pt = 257.4 pt = 3.575 in (\\paperfigmid).
#
# The height is NOT chosen freely.  The submitted artwork is 483.731 x 225.332 pt,
# an aspect of 2.14675, and it printed at 257.4 x 119.90 pt.  Choosing a height
# for the content instead inflated the float to 165.6 pt and changed the shape of
# the figure on the page, so the height is pinned to width / submitted aspect.
SUBMITTED_ASPECT = 483.731 / 225.332          # = 2.14675
FIG_W = 3.575
FIG_H = FIG_W / SUBMITTED_ASPECT              # = 1.6653 in = 119.90 pt

# Margins are budgeted in printed points, not in fractions of the canvas: the
# chrome (title, axis labels, tick labels) is a fixed number of points at a fixed
# type size, so only the plotting rectangle may absorb the shorter canvas.
_H_PT = FIG_H * 72.0
_M_TOP = 15.5      # 8.6 pt bold title + 6 pt pad
_M_BOTTOM = 28.5   # 8 pt tick labels + tick + 9 pt axis label with parens + pads


def main() -> None:
    d = json.loads(DATA.read_text())
    x = d["bin_centers"]

    use_paper_style(8.0)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    for key, color, label in (("fixed", COLORS["fixed"], "Fixed"),
                              ("variable", COLORS["variable"], "Variable")):
        series = d[key]
        ax.errorbar(
            x, series["mean"], yerr=series["sem"],
            color=color, marker="o", markersize=3.6, markerfacecolor="white",
            markeredgewidth=1.1, linewidth=1.5, elinewidth=0.9, capsize=0,
            label=rf"{label}  ($\bar{{\rho}} = {series['mean_rho']:.3f}$)",
            zorder=3,
        )

    ax.set_xlabel("Normalised round (0 = start, 1 = end)")
    ax.set_ylabel("Bet / balance ratio")
    panel_title(ax, "", "Within-Game Bet Escalation Trajectory", size=8.6)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.085, 0.355)
    ax.set_yticks([0.10, 0.15, 0.20, 0.25, 0.30, 0.35])
    ax.legend(loc="upper left", fontsize=7.6, borderaxespad=0.4,
              handlelength=1.8, handletextpad=0.5)
    style_axes(ax)

    fig.subplots_adjust(left=0.152, right=0.982,
                        top=1.0 - _M_TOP / _H_PT, bottom=_M_BOTTOM / _H_PT)
    save_pdf_png(fig, OUT_DIR, "escalation_trajectory")
    print(f"wrote {OUT_DIR / 'escalation_trajectory.pdf'}  ({FIG_W:.4f}x{FIG_H:.4f} in, aspect {FIG_W / FIG_H:.3f})")


if __name__ == "__main__":
    main()
