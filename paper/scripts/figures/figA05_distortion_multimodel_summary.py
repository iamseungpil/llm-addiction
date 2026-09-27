"""Figure A5 -- multimodel cognitive-distortion summary
(images/distortion_multimodel_summary.pdf).

This generator re-authors the **submitted** artwork: same canvas
(794.344 x 353.904 pt), same axes rectangles, same tick sets, same horizontal
value labels, same per-panel legend, same point sizes.  Every geometric constant
below was measured out of the submitted PDF's content stream (axes patch rects,
bar rects, glyph bounding boxes, stroke widths), not guessed, so the output is
the submitted figure plus exactly one change.

The figure is included at ``width=\\paperfigfull`` = 0.78\\textwidth = 308.88 pt,
so the canvas is scaled by 0.3888 on the page and the point sizes below print at
4.3-5.7 pt.  That is what the submission chose.  An earlier revision re-authored
this figure at print width to lift every label to a 7 pt floor; no reviewer asked
for that, and at the submitted aspect it cost the y tick set, the horizontal
value labels and the per-panel key.  The submitted presentation wins.

Content defect fixed here (the one change): in the submitted figure the
GPT-4o-mini x Loss chasing cell is pure white and carries no number, while every
other cell is tinted and labelled.  It is a masked NaN, not a zero.
``run_multimodel_distortion_analysis.py`` scores Loss chasing on the post-loss
decision window only (``PRIMARY_WINDOWS['loss_chasing']['scope'] ==
'post_loss_only'``) and infers the round outcome from ``round_details.game_result``.
The GPT-4o-mini export
(``analysis/gpt_results_fixed_parsing/gpt_fixed_parsing_complete_20250919_151240.json``)
carries no ``game_result``; it records each round's outcome in ``game_history``, as
the submitted generator's loader already knew.  Reading that field gives 7,739
post-loss decisions, 2,002 fixed and 5,737 variable, and a delta of +5.5 pp.  The
same run reproduces the two outcome-independent cells exactly (+5.5 and -11.9),
which validates the loader.  The cell is now drawn and labelled like every other.

Data: ``data/figA05_distortion_multimodel_summary.json``.

Run:  python3 scripts/figures/figA05_distortion_multimodel_summary.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from paper_style_appendix import COLORS, HEATMAP_CMAP, save_pdf_png, use_paper_style

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DATA = HERE / "data" / "figA05_distortion_multimodel_summary.json"
OUT_DIR = REPO / "images"

# ---------------------------------------------------------------- canvas ----
# The submitted page box, to the digit.
W_PT = 794.34375
H_PT = 353.9040222167969
FIG_W, FIG_H = W_PT / 72.0, H_PT / 72.0

# Axes patch rectangles read out of the submitted content stream, in PDF points
# with the origin at the *top* left; converted to figure fractions below.
#   panel (a) heatmap : 117.54, 26.59 -> 393.39, 281.61
#   colourbar         : 403.49, 45.72 -> 414.32, 262.49
#   panel (b) bars    : 514.82, 26.59 -> 787.14, 281.61
def _rect(x0, y0, x1, y1):
    return [x0 / W_PT, (H_PT - y1) / H_PT, (x1 - x0) / W_PT, (y1 - y0) / H_PT]


RECT_A = _rect(117.54, 26.59, 393.39, 281.61)
RECT_C = _rect(403.49, 45.72, 414.32, 262.49)
RECT_B = _rect(514.82, 26.59, 787.14, 281.61)

# The panel titles sit on a 14.5 pt bold baseline 18.58 pt below the page top,
# left-aligned on their own axes' left edge.
TITLE_BASELINE_Y = (H_PT - 18.58) / H_PT

# ------------------------------------------------------------ typography ----
# Submitted point sizes, on the submitted canvas.  At the 0.3888 include scale
# these print at 4.29-5.66 pt.
TICK_FS = 12.0        # model names, rotated distortion labels, panel (b) ticks
CELL_FS = 13.0        # heatmap cell values
TITLE_FS = 14.5       # panel titles
YLAB_B_FS = 12.5      # panel (b) y axis label
BAR_FS = 11.5         # panel (b) bar value labels
LEG_FS = 11.5         # panel (b) key
CBAR_TICK_FS = 11.0
CBAR_LAB_FS = 12.0

# Panel (b) axis geometry, measured: bars 0.40 wide and touching, x from -0.54
# to 2.54, y from 0 to 60 with a tick every 10.
BAR_W = 0.40
XLIM_B = (-0.54, 2.54)
YLIM_B = (0.0, 60.0)
YTICKS_B = [0, 10, 20, 30, 40, 50, 60]
# The submitted value labels clear their bar tops by 2.82 pt = 0.66 pp.
BAR_LABEL_PAD_PP = 0.66
# White text once the cell is dark enough to need it; 10.0 stayed black in the
# submission and 11.2 went white, so the threshold sits between them.
CELL_WHITE_ABOVE = 10.5


def main() -> None:
    d = json.loads(DATA.read_text())
    labels = d["distortions"]
    models = d["models"]
    vlim = d["vlim"]
    delta = np.array([[np.nan if v is None else v for v in row] for row in d["delta_pp"]],
                     dtype=float)

    use_paper_style(TICK_FS)
    # The submitted chrome: 0.8 pt spines and ticks in #444444, 0.6 pt #DDDDDD
    # grid, 1.0 pt #CCCCCC legend frame -- all already in the vendored style --
    # with matplotlib's default 3.5 pt tick length and 3.5 pt pad, which is what
    # the submitted tick-label offsets measure out to.
    plt.rcParams.update({
        "xtick.major.size": 3.5, "ytick.major.size": 3.5,
        "xtick.major.pad": 3.5, "ytick.major.pad": 3.5,
        "legend.labelspacing": 0.5, "legend.handletextpad": 0.8,
        "legend.borderpad": 0.4,
    })
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    # ------------------------------------------------ (a) delta heatmap ----
    ax = fig.add_axes(RECT_A)
    # One Rectangle per cell rather than ``imshow``: same geometry, same colormap
    # lookup, same appearance, but vector instead of an embedded raster.  The
    # colourbar is left rasterised, as the submission had it -- a vector colour
    # ramp shows white seams between its mesh quads at this width.
    norm = Normalize(vmin=-vlim, vmax=vlim)
    cmap = HEATMAP_CMAP
    smap = ScalarMappable(norm=norm, cmap=cmap)
    for i in range(len(models)):
        for j in range(len(labels)):
            if np.isnan(delta[i, j]):
                continue
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                   facecolor=cmap(norm(delta[i, j])),
                                   edgecolor="none", zorder=1))
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_ylim(len(models) - 0.5, -0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=TICK_FS)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=TICK_FS)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    for i in range(len(models)):
        for j in range(len(labels)):
            value = delta[i, j]
            if np.isnan(value):
                # Explicit missing cell: hatched and labelled, not silently white.
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#F2F2F2",
                                       edgecolor="#9A9A9A", linewidth=0.5,
                                       hatch="///", zorder=2))
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=CELL_FS,
                        color="#555555", fontstyle="italic", zorder=3)
            else:
                ax.text(j, i, f"{value:+.1f}", ha="center", va="center",
                        fontsize=CELL_FS,
                        color="white" if abs(value) > CELL_WHITE_ABOVE else "#111111",
                        zorder=3)

    cax = fig.add_axes(RECT_C)
    cbar = fig.colorbar(smap, cax=cax)
    cbar.set_label("V − F (pp)", fontsize=CBAR_LAB_FS, labelpad=3.0)
    cbar.set_ticks(range(-vlim, vlim + 1, 5))
    cbar.ax.tick_params(labelsize=CBAR_TICK_FS, length=3.5, width=0.8, pad=3.5)
    cbar.outline.set_linewidth(0.8)
    cbar.outline.set_edgecolor("#BBBBBB")

    # ------------------------------------------- (b) pooled primary rates ----
    ax = fig.add_axes(RECT_B)
    x = np.arange(len(labels))
    fixed = d["pooled_fixed_pct"]
    variable = d["pooled_variable_pct"]
    ax.bar(x - BAR_W / 2, fixed, BAR_W, label="Fixed", color=COLORS["fixed"], zorder=3)
    ax.bar(x + BAR_W / 2, variable, BAR_W, label="Variable", color=COLORS["variable"],
           zorder=3)
    # Horizontal, centred on their own bar.  At this canvas the two labels of a
    # pair are 9.8 pt apart, so neither has to be rotated or nudged.
    for xi, (f, v) in enumerate(zip(fixed, variable)):
        ax.text(xi - BAR_W / 2, f + BAR_LABEL_PAD_PP, f"{f:.1f}", ha="center",
                va="bottom", fontsize=BAR_FS, zorder=4)
        ax.text(xi + BAR_W / 2, v + BAR_LABEL_PAD_PP, f"{v:.1f}", ha="center",
                va="bottom", fontsize=BAR_FS, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=TICK_FS)
    ax.set_ylabel("% of games with ≥1 keyword hit", fontsize=YLAB_B_FS, labelpad=4.1)
    ax.set_xlim(*XLIM_B)
    ax.set_ylim(*YLIM_B)
    ax.set_yticks(YTICKS_B)
    ax.tick_params(labelsize=TICK_FS)
    # Per-panel key, framed, inside the axes at the upper right -- where the
    # submission put it.
    ax.legend(loc="upper right", fontsize=LEG_FS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color=COLORS["grid"], linewidth=0.6)

    # Panel titles: 14.5 pt bold, left-aligned on each panel's own left edge.
    fig.text(RECT_A[0], TITLE_BASELINE_Y, "(a) Decision-Level Delta", fontsize=TITLE_FS,
             fontweight="bold", color="#000000", ha="left", va="baseline")
    fig.text(RECT_B[0], TITLE_BASELINE_Y, "(b) Pooled Primary Rates", fontsize=TITLE_FS,
             fontweight="bold", color="#000000", ha="left", va="baseline")

    save_pdf_png(fig, OUT_DIR, "distortion_multimodel_summary")
    print(f"wrote {OUT_DIR / 'distortion_multimodel_summary.pdf'}  "
          f"({W_PT:.3f}x{H_PT:.3f} pt, aspect {W_PT / H_PT:.5f})")


if __name__ == "__main__":
    main()
