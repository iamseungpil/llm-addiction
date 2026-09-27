"""Figure A4 -- temperature robustness (images/temperature_robustness.pdf).

SUPERSEDED -- DO NOT RERUN WITHOUT RE-DECIDING.
``images/temperature_robustness.pdf`` in the working tree is the *submitted*
artwork, restored byte for byte from commit dd2d229.  This print-width re-
authoring was compared against it panel by panel and carried nothing the
submission did not already have (the shaded min-max band it explains is
already drawn in the submitted figure, and the four prompt conditions it names
are already stated in the caption, so the additions are annotation rather than
data), so it lost on the standing rule that figures which are not rebuttal
experiments must match the submitted version.  Running this script overwrites
the submitted file.

Included at ``width=\\paperfigfull`` = 0.78\\textwidth.  Measured off the compiled
neurips_en.pdf the placement scale is 0.61469 on a 502.50 pt canvas, i.e. the
figure is printed 308.88 pt = 4.29 in wide (\\textwidth is exactly 396 pt = 5.5 in).
That shrank the submitted 10.5 pt ticks to 6.5 pt on paper.  This regeneration
authors the canvas at 4.29 x 2.70 in, i.e. scale 1.0.

Composition is unchanged: one axes, mean bankruptcy rate against sampling
temperature for Fixed (green) and Variable (red), each with a shaded band.

Two content defects in the submitted version are fixed here, without changing
the chart:

1. The shaded band was undefined -- it appears in no caption and in no legend.
   It is the min-max range of the bankruptcy rate across the four prompt
   conditions (BASE, G, H, GMHW) at that temperature.  It now has its own legend
   entries, and the axis note spells the definition out.
2. The band looked as if it were drawn on the Variable series only.  It is in
   fact drawn on both, but the Fixed band collapses onto the Fixed line because
   Fixed bankruptcy is 20% in every prompt condition at t = 0.0/0.5/0.7 and
   18-20% at t = 1.0.  Both bands now carry a hairline boundary so a collapsed
   band is visible as such, and the note states why it collapses.

Data: ``data/figA04_temperature_robustness.json``, extracted verbatim from the HF
file ``sae_v3_analysis/results/temperature_control/temperature_control_full_20260406_065510.json``
(LLaMA-3.1-8B, 4 temperatures x 4 prompt conditions x 2 bet types x 50 games).
The means and the band limits recomputed from it reproduce the submitted PDF's
plotted geometry exactly.

The HF script ``figA04_temperature_robustness/code/plot_temperature_robustness.py``
draws an earlier grouped-bar version of this figure and imports ``paper_figure_style``,
which is not in this repository; the constants it needs are vendored in
``paper_style_appendix.py``.

Run:  python3 scripts/figures/figA04_temperature_robustness.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from paper_style_appendix import COLORS, panel_title, save_pdf_png, style_axes, use_paper_style

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DATA = HERE / "data" / "figA04_temperature_robustness.json"
OUT_DIR = REPO / "images"

# 0.78 * 396 pt = 308.88 pt = 4.29 in (\\paperfigfull).
#
# The submitted artwork is 502.503 x 241.964 pt -- aspect 2.07677 -- and printed
# at 308.88 x 148.73 pt.  The height is pinned to width / that aspect so the
# float keeps the shape it had on the submitted page; only the plotting
# rectangle absorbs the difference, never the type size.
SUBMITTED_ASPECT = 502.503 / 241.964           # = 2.07677
FIG_W = 4.29
FIG_H = FIG_W / SUBMITTED_ASPECT               # = 2.0657 in = 148.73 pt

# Chrome budgeted in printed points (fixed type size => fixed point cost).
_H_PT = FIG_H * 72.0
_M_TOP = 15.5      # 8.6 pt bold title + 6 pt pad
_M_BOTTOM = 28.5   # 8 pt tick labels + tick + 9 pt axis label + pads


def main() -> None:
    d = json.loads(DATA.read_text())
    temps = d["temperatures"]
    prompts = d["prompts"]
    rates = d["bankruptcy_pct"]

    use_paper_style(8.0)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    line_handles, band_handles = [], []
    for key, color, label in (("fixed", COLORS["fixed"], "Fixed"),
                              ("variable", COLORS["variable"], "Variable")):
        block = np.array([rates[key][p] for p in prompts])  # (n_prompts, n_temps)
        mean = block.mean(axis=0)
        lo, hi = block.min(axis=0), block.max(axis=0)

        ax.fill_between(temps, lo, hi, color=color, alpha=0.16, linewidth=0, zorder=2)
        # hairline boundary: makes a collapsed (zero-width) band legible as one
        ax.plot(temps, lo, color=color, alpha=0.55, linewidth=0.5, zorder=2.5)
        ax.plot(temps, hi, color=color, alpha=0.55, linewidth=0.5, zorder=2.5)
        ax.plot(temps, mean, color=color, linewidth=1.6, marker="o", markersize=3.6,
                markerfacecolor="white", markeredgewidth=1.1, zorder=4)

        line_handles.append(Line2D([], [], color=color, linewidth=1.6, marker="o",
                                   markersize=3.6, markerfacecolor="white",
                                   markeredgewidth=1.1, label=label))
        band_handles.append(Patch(facecolor=color, alpha=0.30, edgecolor=color,
                                  linewidth=0.5))

    ax.set_xlabel("Temperature")
    ax.set_ylabel("Bankruptcy rate (%)")
    panel_title(ax, "", "Temperature Robustness", size=8.6)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 105)
    ax.set_xticks(temps)
    ax.set_xticklabels([f"{t:.1f}" for t in temps])
    ax.set_yticks([0, 20, 40, 60, 80, 100])

    handles = line_handles + [tuple(band_handles)]
    labels = ["Fixed — mean over prompts", "Variable — mean over prompts",
              "Shaded: min–max over " + "/".join(prompts) + " (both series)"]
    # The key and the note sit in the empty band between the Fixed line (20%)
    # and the lower edge of the Variable min-max band (74% at t = 0).  That band
    # is 56 pp tall, which is 60 pt of the shorter plotting rectangle, so the
    # key's row spacing is tightened from 0.40 to 0.30 to keep both clear of the
    # data.  Nothing is dropped and no type shrinks.
    ax.legend(handles, labels, loc="center left", bbox_to_anchor=(0.015, 0.545),
              fontsize=7.0, ncol=1, borderaxespad=0.0, handlelength=1.9,
              handletextpad=0.5, labelspacing=0.30,
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0.4)})

    note = ("The Fixed band collapses onto the Fixed line: Fixed bankruptcy\n"
            "is 20% in every prompt condition (18–20% at t = 1.0).")
    ax.text(0.015, 0.385, note, transform=ax.transAxes, fontsize=7.0,
            color=COLORS["text"], va="top", ha="left", linespacing=1.4, zorder=5)

    style_axes(ax)
    fig.subplots_adjust(left=0.128, right=0.982,
                        top=1.0 - _M_TOP / _H_PT, bottom=_M_BOTTOM / _H_PT)
    save_pdf_png(fig, OUT_DIR, "temperature_robustness")
    print(f"wrote {OUT_DIR / 'temperature_robustness.pdf'}  ({FIG_W:.4f}x{FIG_H:.4f} in, aspect {FIG_W / FIG_H:.3f})")


if __name__ == "__main__":
    main()
