"""Figure A9 -- CoT investment-choice distributions
(images/investment_choice_distributions_cot.pdf).

SUPERSEDED -- DO NOT RERUN WITHOUT RE-DECIDING.
``images/investment_choice_distributions_cot.pdf`` in the working tree is the
*submitted* artwork, restored byte for byte from commit dd2d229.  This print-
width re-authoring was compared against it panel by panel and carried nothing
the submission did not already have (it adds only axis tick labels to panels
the submission left unticked), so it lost on the standing rule that figures
which are not rebuttal experiments must match the submitted version.  Running
this script overwrites the submitted file.

Included at ``width=0.95\\textwidth``.  Measured off the compiled neurips_en.pdf the
placement scale is 0.40990 on a 917.79 pt canvas, i.e. the figure is printed
376.20 pt = 5.225 in wide (\\textwidth is exactly 396 pt = 5.5 in).  That shrank the
submitted 22 pt tick labels to 9 pt but the 20 pt y-ticks to 8.2 pt, and the whole
figure to 41% of the size it was drawn at.  This regeneration authors the canvas at
5.225 x 2.95 in, i.e. scale 1.0.

Composition is unchanged: a 2 x 4 grid of stacked bars, top row Fixed betting,
bottom row Variable betting, columns GPT-4o-mini / GPT-4.1-mini / Gemini-2.5-Flash
/ Claude-3.5-Haiku, four prompt conditions (BASE, G, M, GM) per panel, Options 1-4
stacked bottom-to-top in the submitted colours, every tick label kept.  The only
thing that moved is the Option legend: at print width there is no room for a
legend column beside the panels without squeezing the panels below legibility, so
it now runs as a single row under the grid.

Data: ``data/figA09_investment_choice_distributions_cot.json``.  The camera-ready
art on HF has no released generator (the HF script
``figA09_ic_distributions_cot/code/create_choice_distribution_cot.py`` draws an
earlier, differently ordered and differently coloured version, and reads a results
tree that is not in this repository), so the 32 stacked distributions were read
back out of the submitted PDF's vector geometry -- each of the 32 stacks recovers
to 100.000%.

Run:  python3 scripts/figures/figA09_investment_choice_distributions_cot.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from paper_style_appendix import (CHOICE_COLORS, COLORS, save_pdf_png, style_axes,
                                  use_paper_style)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DATA = HERE / "data" / "figA09_investment_choice_distributions_cot.json"
OUT_DIR = REPO / "images"

# 0.95 * 396 pt = 376.2 pt = 5.225 in.
#
# The submitted artwork is 917.791 x 406.865 pt -- aspect 2.25576 -- and printed
# at 376.2 x 166.77 pt.  The height is pinned to width / that aspect so the float
# keeps its submitted shape; the chrome below is budgeted in printed points, so
# only the eight plotting rectangles absorb the difference.
SUBMITTED_ASPECT = 917.791 / 406.865            # = 2.25576
FIG_W = 5.225
FIG_H = FIG_W / SUBMITTED_ASPECT                # = 2.3163 in = 166.77 pt
_H_PT = FIG_H * 72.0

_TOP = 26.0        # 9.5 pt suptitle + 8 pt row-0 panel titles + their pads
_BOTTOM = 37.0     # x tick labels + 7.5 pt x axis label + the 7.5 pt key row
_ROW_GAP = 11.0    # row-0 x tick labels between the two rows
_ROW_H = (_H_PT - _TOP - _BOTTOM - _ROW_GAP) / 2.0


def main() -> None:
    d = json.loads(DATA.read_text())
    models = d["models"]
    names = d["model_names"]
    conds = d["conditions"]
    dist = d["distribution_pct"]

    use_paper_style(8.0)
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs = fig.add_gridspec(2, 4, left=0.105, right=0.960,
                          top=1.0 - _TOP / _H_PT, bottom=_BOTTOM / _H_PT,
                          wspace=0.42, hspace=_ROW_GAP / _ROW_H)

    x = np.arange(len(conds))
    row_labels = {"fixed": "Fixed betting", "variable": "Variable betting"}

    for r, bet_type in enumerate(("fixed", "variable")):
        for c, model in enumerate(models):
            ax = fig.add_subplot(gs[r, c])
            bottoms = np.zeros(len(conds))
            for opt in range(4):
                vals = np.array([dist[bet_type][model][cond][opt] for cond in conds])
                ax.bar(x, vals, width=0.65, bottom=bottoms, color=CHOICE_COLORS[opt],
                       edgecolor="white", linewidth=0.5, zorder=3)
                bottoms += vals

            ax.set_xticks(x)
            ax.set_xticklabels(conds, fontsize=7.0)
            ax.set_ylim(0, 100)
            ax.set_yticks([0, 20, 40, 60, 80, 100])
            ax.tick_params(axis="y", labelsize=7.0, length=2.0)
            ax.tick_params(axis="x", length=2.0)
            style_axes(ax)

            if r == 0:
                ax.set_title(names[model], fontsize=8.0, fontweight="bold",
                             color="#000000", pad=4.0)
            if c == 0:
                # Each panel is 46 pt tall at the submitted aspect while the
                # two-line row label is ~57 pt long, so a centred label runs
                # 5 pt past both ends of its own panel and the two rows' labels
                # meet in the middle.  Each is pushed toward its own outer end
                # (the gutter is empty there), which opens 11 pt between them.
                # Same words, same 7.5 pt type.
                ax.set_ylabel(f"{row_labels[bet_type]}\nDistribution (%)",
                              fontsize=7.5, labelpad=2.0,
                              y=0.66 if r == 0 else 0.34)
            if r == 1:
                ax.set_xlabel("Prompt condition", fontsize=7.5, labelpad=2.0)

    fig.suptitle("Investment Choice Distribution by Model", fontsize=9.5,
                 fontweight="bold", color="#000000", y=1.0 - 2.5 / _H_PT)

    handles = [Patch(facecolor=CHOICE_COLORS[i], edgecolor="white", linewidth=0.5,
                     label=f"Option {i + 1}") for i in range(4)]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, 1.5 / _H_PT),
               ncol=4, fontsize=7.5, columnspacing=1.6, handlelength=1.5,
               handletextpad=0.5, borderaxespad=0.0)

    save_pdf_png(fig, OUT_DIR, "investment_choice_distributions_cot")
    print(f"wrote {OUT_DIR / 'investment_choice_distributions_cot.pdf'}  ({FIG_W:.4f}x{FIG_H:.4f} in, aspect {FIG_W / FIG_H:.3f})")


if __name__ == "__main__":
    main()
