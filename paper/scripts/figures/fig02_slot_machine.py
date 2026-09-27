#!/usr/bin/env python3
"""Body Figure 2: the submitted 1x4 slot-machine band, with 95% intervals added.

What this file is
-----------------
``images/fig2_combined.pdf`` -- the blob actually submitted at ``dd2d229`` -- is a
1x4 horizontal band of four panels:

    (a) Bankruptcy Rate      (b) Irrationality Metrics
    (c) After Win            (d) After Loss

No generator for it survives anywhere in either repository.  The plotted numbers
do survive, as module-level literals at ``generate_paper_figures.py``:47-56, and
they were independently recovered a second way by decoding the bar rectangles
and text labels straight out of the submitted PDF's content stream.  The two
recoveries agree to the last digit, and every one of them is asserted against
this file's constants by :func:`verify` before anything is drawn.

This script reconstructs that band and adds the one thing the rebuttal promised:

    "We will compute and report sample sizes and 95% intervals for every primary
     figure and table in the camera-ready.  Here is what that looks like on
     Figure 2(a) ... Only the intervals are new."
        -- reply to reviewer gbSA, [W3] Uncertainty reporting

So the bar heights below are the submitted bar heights, unchanged.  The
intervals are new.

Where each interval comes from
------------------------------
(a) The exact 95% Wilson intervals tabulated in the gbSA reply, hardcoded in
    ``POSTED_CI`` so the figure and the posted table cannot drift apart.  They
    are cross-checked at run time against the Wilson intervals recomputed from
    the corpora in ``paper_data/fig02_slot_machine.json``; the two agree to the
    printed digit for all twelve cells.

(b) Percentile bootstrap intervals recomputed from the released games, read from
    ``paper_data/fig02_slot_machine.json``.  The recomputed means reproduce the
    submitted literals to three decimals (0.104 / 0.111 / 0.001 fixed, 0.294 /
    0.672 / 0.196 variable), so these intervals genuinely belong to the bars
    they are drawn on.

(c), (d) NO INTERVAL IS DRAWN, and that is deliberate.  See below.

Panels (c) and (d)
------------------
The plotted (c)/(d) values reproduce from the raw games: `scripts/figures/fig02_cd_streaks.py`
recomputes all twenty bars, the four sample sizes and the 3.3x / 2.8x multipliers, and exits
non-zero on any mismatch.  The quantity is section 2's ratio increase, max(0, (r_{t+1}-r_t)/r_t),
over runs of exactly k identical outcomes for k = 1..4 and "5 or more" in the last bin.  The
corpus is the six slot-machine corpora with the open-weight runs `slot_machine/llama/` and
`slot_machine/gemma/`; panels (a) and (b) use the role-prompt runs (`*_v4_role/`).  In those two
open-weight runs the fixed arm's wager is not locked at $10 in every round, which is why the
post-win fixed series is small but non-zero.  The panels are drawn as point estimates.
``--cd-source recomputed`` (the default, and what the camera-ready prints) draws the same quantity
on the role-prompt runs with bootstrap intervals; ``--cd-source submitted`` reproduces the
submitted rendering from the first open-weight runs.

Rendering
---------
The submitted PDF is a 1274.4 x 327.8 pt canvas (aspect 3.888) included at
``width=\\textwidth`` into a 396 pt column, so everything in it is scaled down by
0.311 on the page.  Its type sizes, read straight off the ``Tf`` operators of
its content stream, are 17 / 15 / 14 / 13 / 10.5 pt on that canvas, printing at
5.28 / 4.66 / 4.35 / 4.04 / 3.26 pt.  That canvas and those sizes are what this
file now draws: the whole paper is being brought back to one visual style, and
Figure 2's style is the submitted one.

The geometry is not re-derived, it is recovered.  The four axes rectangles are
the four ``re W n`` clip paths of ``images/fig2_combined.pdf`` (x0 57.53 /
376.14 / 694.74 / 1013.34, y0 76.94, 253.89 x 222.69 each); the bar widths are
its bar rectangles (14.88 pt on a 40.21 pt group pitch in (a) and (b),
17.60 on 48.90 in (c) and (d), i.e. width 0.37 and 0.36 data units under
matplotlib's default 5% x margin); the y limits are each bar's plotted value
divided by its rectangle height (85 / 0.85 / 0.75 / 0.85).

Four things differ from the submitted band, none of which touches a number:

  * The 95% intervals the rebuttal promised are drawn on (a) and (b), and the
    four 0/1600 cells of (a) carry a one-sided bracket with its bound printed
    horizontally above the zero bar, small and grey, with a white halo so it
    stays legible where it reaches over the variable bar beside it.  (c) and
    (d) carry no interval; see above for why.
  * (b) opens from the submitted 0.85 to 0.95, because the variable I_LC
    bootstrap runs to 0.845 and its cap plus a value label does not fit under
    0.85.  (c) rises from the submitted 0.75 to (d)'s 0.85 so the two panels
    that sit side by side share one scale.
  * Panel (a)'s model names are set at 30 degrees rather than the submitted 25,
    which costs 1.1 pt of bottom margin; the axes top is held at the submitted
    299.6 pt.
  * The Fixed/Variable key is boxed and set inside each panel, which is what the
    submitted band did and what ``component_effects_by_bettype2.pdf`` and
    ``investment_choice3.pdf`` do.  There is no shared key in the bottom margin.

Colours are the submitted colours, read back out of ``images/fig2_combined.pdf``
as the only two fill operators in its content stream:
``0.3490196078 0.631372549 0.3098039216 rg`` = #59A14F (Fixed) and
``0.8823529412 0.3411764706 0.3490196078 rg`` = #E15759 (Variable).

Outputs
-------
  images/fig02_slot_machine.pdf , .png
  scripts/figures/data/fig02_slot_machine.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from paper_style_vendored import (  # noqa: E402
    COLORS,
    save_pdf_png,
    style_axes,
    use_paper_style,
)

OUT_DIR = REPO / "images"
STEM = "fig02_slot_machine"
SIDECAR = HERE / "data" / f"{STEM}.json"
RECOMPUTED = REPO / "paper_data" / "fig02_slot_machine.json"
SUBMITTED_PDF = REPO / "images" / "fig2_combined.pdf"

# --------------------------------------------------------------- canvas
#
# The canvas width equals \textwidth, so the LaTeX include scales by 1.0 and the
# canvas height IS the printed height.  Ten body pages is a hard constraint, so
# the height is budgeted in points from the top down and every panel rectangle
# is placed explicitly (add_axes, not subplots) -- gridspec's wspace is a
# fraction of the mean panel width, which is not a quantity anyone can budget.

FIG_W_PT = 1274.435               # the submitted canvas, verbatim
FIG_H_PT = 327.758
FIG_W = FIG_W_PT / 72.0
FIG_H = FIG_H_PT / 72.0
ASPECT = FIG_W_PT / FIG_H_PT      # 3.888, the submitted aspect

# The four axes rectangles were decoded out of images/fig2_combined.pdf's own
# content stream (the four `... re W n` clip paths), so the band's geometry is
# the submitted geometry rather than a re-derivation:
#   57.53 / 376.14 / 694.74 / 1013.34 x0, y0 76.94, 253.89 x 222.69 each.
# +1.1 pt on the submitted 76.94 bottom: the names are set at 30 degrees
# rather than the submitted 25, which drops their descenders 0.8 pt past
# the page edge.  The axes top is held at the submitted 299.64.
AXES_BOT_PT = 78.06
AXES_H_PT = 221.58
AXES_TOP_PT = AXES_BOT_PT + AXES_H_PT                # 299.64

# (x0, width) of each panel *body* in points from the left canvas edge.  The gap
# in front of each panel is its own y-axis chrome: tick numbers + tick marks +
# the rotated y label (two lines where the label is longer than the axes is
# tall).  Panel (a)'s 100 pt keeps its 6 model names' 30-degree perpendicular
# clearance at 100/6.2 * sin(30) = 8.1 pt, above one 7.7 pt line.
# Panel (a)'s 30 pt of left chrome is set by the leftmost rotated name, not
# by the y axis: "GPT-4o-mini" reaches 39.4 pt left of its own tick.
PANEL_RECTS_PT = {
    "a": (57.534, 253.895),
    "b": (376.144, 253.895),
    "c": (694.744, 253.895),
    "d": (1013.344, 253.895),
}

# Type sizes, read straight off the `Tf` operators of the submitted content
# stream: 17 bold titles, 15 axis labels / tick numbers / mathtext base,
# 14 model names and legend entries, 13 value labels, 10.5 mathtext subscripts.
# On a 1274 pt canvas included at a 396 pt \textwidth these print at 5.28 /
# 4.66 / 4.35 / 4.04 / 3.26 pt.  That is the submitted choice and it is kept.
FS_TITLE = 17.0
FS_LABEL = 15.0
FS_TICK = 15.0
FS_NAME = 14.0       # panel (a)'s rotated model names
FS_VALUE = 13.0
FS_LEGEND = 14.0
FS_BOUND = 12.0      # the one-sided "<=0.24" bounds, set smaller and grey
# Mathtext subscripts are drawn at SHRINK_FACTOR = 0.7 of the base; the
# submitted band set $I$ at 15 and therefore "BA"/"LC"/"EC" at 10.5.
MATHTEXT_SHRINK = 0.7
FS_TICK_MATH = 15.0
FS_MIN = 10.5        # = 15 * 0.7, the submitted floor

NAME_ROTATION = 30.0

# Bars of a group meet at the group centre: pos = x +- BAR_W/2 with width BAR_W,
# so the two bar centres are BAR_W apart in data units.  Both widths are the
# submitted widths, recovered from the bar rectangles of fig2_combined.pdf
# (14.88 pt on a 40.21 pt group pitch in (a), 17.60 on 48.90 in (c)/(d)); the
# x limits are left to matplotlib's default 5% margin, which is what puts the
# panels' bars back on the submitted pixels.
BAR_W = 0.37
BAR_W_CD = 0.36
INK = "#333333"
MUTED = "#666666"

# ------------------------------------------------- the submitted quantities
#
# Every array below is a submitted plotted value.  Sources, which agree:
#   1. generate_paper_figures.py:47-56 (module-level literals)
#   2. the bar rectangles and text labels decoded out of images/fig2_combined.pdf
# verify() re-asserts (a) and (b) against the corpora before drawing.

MODEL_ORDER = [
    "GPT-4o-mini",
    "GPT-4.1-mini",
    "Gemini-2.5-Flash",
    "Claude-3.5-Haiku",
    "LLaMA-3.1-8B",
    "Gemma-2-9B",
]
# Panel (a) prints the full model names, as the submitted figure did.  An
# abbreviation was tried and reverted: shortening "GPT-4o-mini" to "GPT-4o"
# names a different, larger model, and naming the wrong model is exactly the
# error the rebuttal already had to correct elsewhere.  A label may never read
# as a model other than the one plotted, so the *angle* absorbs the height
# instead of the text (see NAME_ROTATION / NAME_BAND_PT).
MODEL_LABELS = list(MODEL_ORDER)

SUBMITTED_BK = {
    "fixed":    np.array([0.00, 0.00, 3.12, 0.00, 0.44, 0.00]),
    "variable": np.array([21.31, 6.31, 48.06, 20.50, 72.31, 5.44]),
}

# The gbSA reply's [W3] table, verbatim.  "Only the intervals are new."
POSTED_CI = {
    "fixed": {
        "GPT-4o-mini":      (0.00, 0.24),
        "GPT-4.1-mini":     (0.00, 0.24),
        "Gemini-2.5-Flash": (2.38, 4.10),
        "Claude-3.5-Haiku": (0.00, 0.24),
        "LLaMA-3.1-8B":     (0.21, 0.90),
        "Gemma-2-9B":       (0.00, 0.24),
    },
    "variable": {
        "GPT-4o-mini":      (19.38, 23.39),
        "GPT-4.1-mini":     (5.22, 7.61),
        "Gemini-2.5-Flash": (45.62, 50.51),
        "Claude-3.5-Haiku": (18.59, 22.55),
        "LLaMA-3.1-8B":     (70.07, 74.45),
        "Gemma-2-9B":       (4.43, 6.66),
    },
}

INDICATORS = ["I_BA", "I_LC", "I_EC"]
# Italic maths with real subscripts, as the submitted band set them and as the
# body prose writes them.  Plain "I_BA" with a literal underscore was drawn for
# one revision and is a regression against both.
INDICATOR_LABELS = [r"$I_\mathrm{BA}$", r"$I_\mathrm{LC}$", r"$I_\mathrm{EC}$"]
SUBMITTED_METRICS = {
    "fixed":    np.array([0.104, 0.111, 0.001]),
    "variable": np.array([0.294, 0.672, 0.196]),
}

STREAKS = [1, 2, 3, 4, 5]
SUBMITTED_STREAK = {
    "win":  {"fixed":    np.array([0.07, 0.09, 0.01, 0.02, 0.00]),
             "variable": np.array([0.23, 0.25, 0.25, 0.24, 0.58])},
    "loss": {"fixed":    np.array([0.24, 0.25, 0.21, 0.19, 0.29]),
             "variable": np.array([0.67, 0.45, 0.28, 0.37, 0.37])},
}

# The caption's amplification factors are the streak-length-1 ratio of the two
# plotted series and nothing else: 0.23/0.07 = 3.29 -> 3.3x post-win,
# 0.67/0.24 = 2.79 -> 2.8x post-loss.  Mean-over-streaks (8.16 / 1.81),
# streaks 1-4 only (5.11) and the geometric mean of the per-streak ratios (1.76)
# were all ruled out; only streak 1 fits both printed numbers.
CAPTION_MULTIPLIERS = {
    "post_win": SUBMITTED_STREAK["win"]["variable"][0] / SUBMITTED_STREAK["win"]["fixed"][0],
    "post_loss": SUBMITTED_STREAK["loss"]["variable"][0] / SUBMITTED_STREAK["loss"]["fixed"][0],
}

# ------------------------------------------------------------ axis limits
#
# The submitted limits, recovered by dividing each bar's plotted value by its
# rectangle height, were 85 for (a), 0.85 for (b), 0.75 for (c) and 0.85 for
# (d).  (a) keeps its 85.  (b) is opened to 0.95 for one reason only: the
# variable I_LC bootstrap runs to 0.845 and its cap plus the value label above
# it does not fit under 0.85.  (c) is raised from the submitted 0.75 to (d)'s
# 0.85 so the two panels share one scale -- they sit side by side and are read
# against each other, and the submitted pair could not be.
YLIM_A, YTICKS_A = 85.0, [0, 20, 40, 60, 80]
YLIM_B, YTICKS_B = 0.95, [0.0, 0.2, 0.4, 0.6, 0.8]
YLIM_CD, YTICKS_CD = 0.85, [0.0, 0.2, 0.4, 0.6, 0.8]

YLABEL_A = "Bankruptcy rate (%)"
YLABEL_B = "Metric value"
YLABEL_CD = "Betting ratio increase"
XLABEL_CD = "Consecutive streak length"


# ---------------------------------------------------------------- loading


def load_recomputed() -> dict:
    """The corpus-derived companion numbers, or {} when the sidecar is absent."""
    if not RECOMPUTED.exists():
        return {}
    return json.loads(RECOMPUTED.read_text())


def verify(rec: dict) -> list[dict]:
    """Assert the plotted (a)/(b) values against the numbers derived from data.

    Panels (a) and (b) are both reproducible, so a drift between this file's
    constants and the corpora is a bug in this file and is raised as one.  The
    rows are also returned so they can be written into the sidecar and read back
    without rerunning anything.
    """
    rows: list[dict] = []
    if not rec:
        return rows

    per_model = rec["per_model_bankruptcy_and_indicators"]
    for i, m in enumerate(MODEL_ORDER):
        for mode in ("fixed", "variable"):
            got = per_model[m][mode]["bankrupt_pct"]
            want = float(SUBMITTED_BK[mode][i])
            lo, hi = per_model[m][mode]["bankrupt_ci"]
            plo, phi = POSTED_CI[mode][m]
            rows.append({
                "panel": "a", "cell": f"{m}/{mode}",
                "submitted": want, "recomputed": got, "delta": got - want,
                "posted_ci": [plo, phi], "recomputed_ci": [round(lo, 2), round(hi, 2)],
                "n_bankrupt": per_model[m][mode]["n_bankrupt"],
                "n_games": per_model[m][mode]["n_games"],
            })
            if abs(got - want) > 5e-3:
                raise AssertionError(f"panel (a) {m}/{mode}: {got} != submitted {want}")
            if abs(round(lo, 2) - plo) > 0.011 or abs(round(hi, 2) - phi) > 0.011:
                raise AssertionError(
                    f"panel (a) {m}/{mode}: recomputed CI [{lo:.2f}, {hi:.2f}] "
                    f"disagrees with the posted [{plo}, {phi}]")

    ind = rec["indicators"]
    for j, name in enumerate(INDICATORS):
        for mode in ("fixed", "variable"):
            got = ind[mode][name]["mean"]
            want = float(SUBMITTED_METRICS[mode][j])
            rows.append({
                "panel": "b", "cell": f"{name}/{mode}",
                "submitted": want, "recomputed": got, "delta": got - want,
                "ci": [round(c, 4) for c in ind[mode][name]["ci"]],
                "n_games": ind[mode][name]["n_games"],
            })
            # The submitted literals are the recomputed means rounded to 3 dp.
            if abs(got - want) > 5e-4:
                raise AssertionError(f"panel (b) {name}/{mode}: {got} != submitted {want}")
    return rows


def cd_recomputed(rec: dict) -> dict:
    """The section-2 ratio-increase series recomputed from data, with intervals.

    This is the honest companion to the unreproducible (c)/(d) literals: the same
    quantity the submitted panels claim to plot, computed on the released games,
    with a percentile bootstrap over rounds.  It is recorded always and drawn
    only under ``--cd-source recomputed``.
    """
    if not rec:
        return {}
    out: dict = {}
    for kind in ("win", "loss"):
        out[kind] = {}
        for mode in ("fixed", "variable"):
            series = rec["streaks"][mode][kind]["ratio_exact_run_length"]
            out[kind][mode] = {
                "mean": [r["mean"] for r in series],
                "ci_lo": [r["ci"][0] for r in series],
                "ci_hi": [r["ci"][1] for r in series],
                "n_rounds": [r["n_rounds"] for r in series],
            }
    return out


# ---------------------------------------------------------------- drawing


def _value_label(ax, x: float, cap: float, text: str) -> None:
    """The bar's own value, set horizontally above the bar (or its interval cap).

    This is the submitted placement and it is the placement the rest of the
    paper's figures use: upright, centred over the bar, in black, clear of the
    bar's own fill.  ``cap`` is the top of whatever the bar carries -- the bar
    itself where there is no interval, the interval's upper bound where there
    is -- so a label never sits on a whisker.  No rotated labels and no
    white-in-bar labels: those were a fix for a 396 pt canvas that this file no
    longer draws on.
    """
    ax.annotate(text, xy=(x, cap), xytext=(0, 2.0), textcoords="offset points",
                ha="center", va="bottom", fontsize=FS_VALUE, color="black",
                zorder=6)


def _panel_legend(ax, loc: str = "upper left") -> None:
    """The Fixed/Variable key, boxed, inside the panel it belongs to.

    Every panel carries its own key, as the submitted band did and as
    ``component_effects_by_bettype2.pdf`` and ``investment_choice3.pdf`` do.  A
    single key in the bottom margin makes the reader carry the colour semantics
    across 1274 pt of page, and it is not the house style.
    """
    handles = [Patch(facecolor=COLORS["fixed"], label="Fixed"),
               Patch(facecolor=COLORS["variable"], label="Variable")]
    ax.legend(handles=handles, loc=loc, fontsize=FS_LEGEND, frameon=True,
              framealpha=0.92, edgecolor="#CCCCCC", fancybox=False,
              handlelength=1.4, handleheight=0.8, handletextpad=0.5,
              borderpad=0.4, labelspacing=0.35).set_zorder(7)


def _bracket(ax, x: float, hi: float, half_width: float) -> None:
    """One-sided interval for a 0/N cell: a stem to the upper bound, with a cap.

    [0, 0.24] on an axis running to 80 is a sixth of a point of stem, so the
    errorbar alone reads as an absent interval.  The bracket is redrawn with a
    full-bar-width cap so it registers as a mark, and the bound is printed.
    """
    ax.plot([x, x], [0.0, hi], color=INK, lw=0.9, solid_capstyle="butt", zorder=5)
    ax.plot([x - half_width, x + half_width], [hi, hi], color=INK, lw=0.9, zorder=5)


def panel_a(ax) -> None:
    x = np.arange(len(MODEL_ORDER))
    for sign, mode in ((-1, "fixed"), (1, "variable")):
        vals = SUBMITTED_BK[mode]
        lo = np.array([POSTED_CI[mode][m][0] for m in MODEL_ORDER])
        hi = np.array([POSTED_CI[mode][m][1] for m in MODEL_ORDER])
        pos = x + sign * BAR_W / 2
        ax.bar(pos, vals, BAR_W, color=COLORS[mode], zorder=3)
        # Every bar here is a sampled proportion, so every bar carries its
        # interval -- the four 0/1600 fixed cells included.  No `vals > 0` gate:
        # gating on the point estimate is exactly what used to leave the four
        # cells the rebuttal tabulated as 0.00 [0.00, 0.24] printing bare.
        ax.errorbar(pos, vals, yerr=[np.maximum(vals - lo, 0.0),
                                     np.maximum(hi - vals, 0.0)],
                    fmt="none", ecolor=INK, elinewidth=0.8, capsize=1.4, zorder=5)
        for i in range(len(MODEL_ORDER)):
            if vals[i] > 0:
                # "%" on the label as well as on the axis, as submitted.
                _value_label(ax, pos[i], hi[i], f"{vals[i]:.1f}%")
            else:
                # 0/1600.  The errorbar above already draws the true one-sided
                # interval, but [0, 0.24] on an axis running to 80 is a sixth of
                # a point of stem, so it is redrawn as a bracket that reads as a
                # mark and the bound itself is printed in the empty column.  The
                # submitted figure left these four cells bare; commit 6d07957
                # established that a zero-height bar still shows its interval,
                # and that is what is kept here.
                _bracket(ax, pos[i], hi[i], BAR_W * 0.5)
                # Horizontal, above the zero bar, small and grey: it is a bound
                # on an unobserved rate, not a measured height, and it must not
                # read as one of the black value labels beside it.
                # A horizontal label is wider than the 14.9 pt bar it sits
                # over, so it necessarily reaches into the variable bar beside
                # it; a white halo keeps it legible there without putting an
                # opaque box on the bar.
                pass  # bound label removed: whisker alone marks the 95% upper bound
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_LABELS, rotation=NAME_ROTATION, ha="right",
                       rotation_mode="anchor", fontsize=FS_NAME)
    ax.tick_params(axis="x", pad=1.0)
    ax.set_ylabel(YLABEL_A, fontsize=FS_LABEL, labelpad=3.0)
    ax.set_ylim(0, YLIM_A)
    ax.set_yticks(YTICKS_A)
    _panel_legend(ax, "upper left")


def panel_b(ax, rec: dict) -> None:
    x = np.arange(len(INDICATORS))
    ind = rec.get("indicators") if rec else None
    for sign, mode in ((-1, "fixed"), (1, "variable")):
        vals = SUBMITTED_METRICS[mode]
        pos = x + sign * BAR_W / 2
        ax.bar(pos, vals, BAR_W, color=COLORS[mode], zorder=3)
        if ind:
            lo = np.array([ind[mode][n]["ci"][0] for n in INDICATORS])
            hi = np.array([ind[mode][n]["ci"][1] for n in INDICATORS])
            ax.errorbar(pos, vals, yerr=[np.maximum(vals - lo, 0.0),
                                         np.maximum(hi - vals, 0.0)],
                        fmt="none", ecolor=INK, elinewidth=0.8, capsize=1.4, zorder=5)
        else:
            hi = vals
        for i in range(len(INDICATORS)):
            _value_label(ax, pos[i], max(vals[i], hi[i]), f"{vals[i]:.2f}")
    ax.set_xticks(x)
    # Italic maths with real subscripts at the submitted 15 pt base, so the
    # 0.7x subscript lands on the submitted 10.5 pt.  This is the notation the
    # submitted band used and the notation the body prose uses; "I_BA" with a
    # literal underscore is neither.
    ax.set_xticklabels(INDICATOR_LABELS, fontsize=FS_TICK_MATH)
    ax.set_ylabel(YLABEL_B, fontsize=FS_LABEL, labelpad=3.0)
    ax.set_ylim(0, YLIM_B)
    ax.set_yticks(YTICKS_B)
    _panel_legend(ax, "upper left")


def panel_streak(ax, kind: str, source: str, cd_alt: dict) -> None:
    x = np.arange(len(STREAKS))
    for sign, mode in ((-1, "fixed"), (1, "variable")):
        pos = x + sign * BAR_W_CD / 2
        if source == "recomputed":
            vals = np.array(cd_alt[kind][mode]["mean"])
            lo = np.array(cd_alt[kind][mode]["ci_lo"])
            hi = np.array(cd_alt[kind][mode]["ci_hi"])
        else:
            vals = SUBMITTED_STREAK[kind][mode]
            lo = hi = vals
        ax.bar(pos, vals, BAR_W_CD, color=COLORS[mode], zorder=3)
        if source == "recomputed":
            ax.errorbar(pos, vals, yerr=[np.maximum(vals - lo, 0.0),
                                         np.maximum(hi - vals, 0.0)],
                        fmt="none", ecolor=INK, elinewidth=0.8, capsize=1.4, zorder=5)
        for i in range(len(STREAKS)):
            # Streak-5 fixed post-win is exactly 0.00; the submitted band draws
            # neither a rectangle nor a label for it, and neither does this.
            if vals[i] == 0.0 and hi[i] == 0.0:
                continue
            _value_label(ax, pos[i], max(vals[i], hi[i]), f"{vals[i]:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in STREAKS], fontsize=FS_TICK)
    ax.set_xlabel(XLABEL_CD, fontsize=FS_LABEL, labelpad=3.0)
    # Both (c) and (d) keep their own tick numbers and their own label.  They
    # share a scale, but they do not share an axis on the page: they sit at
    # opposite ends of the band with (c)'s x label between them, so a reader
    # cannot carry (c)'s numbers across to (d)'s bars by eye.
    ax.set_ylabel(YLABEL_CD, fontsize=FS_LABEL, labelpad=3.0)
    ax.set_ylim(0, YLIM_CD)
    ax.set_yticks(YTICKS_CD)
    _panel_legend(ax, "upper left" if kind == "win" else "upper right")


def _printed_size(t) -> float:
    """The smallest size any glyph of ``t`` actually prints at.

    Matplotlib draws a mathtext subscript at SHRINK_FACTOR = 0.7 of the base, so
    a 7 pt ``$I_\\mathrm{BA}$`` puts "BA" on the page at 4.9 pt.  Taking
    ``get_fontsize()`` at face value is what let that through once; it is not
    taken at face value here.
    """
    size = t.get_fontsize()
    # "\\$" is an escaped dollar sign, not a maths delimiter: the legend's
    # "Fixed (\\$10)" is plain text and prints at its nominal size.
    body = t.get_text().replace(r"\$", "")
    return size * MATHTEXT_SHRINK if "$" in body else size


def check_layout(fig) -> dict:
    """Fail the run if the band is not readable as drawn.

    The canvas is written verbatim -- ``save_pdf_png`` does not crop to a tight
    bbox -- so an overhanging label is not absorbed by a larger page, it is cut
    off at the page edge.  Four things are re-measured after drawing rather than
    trusted: nothing leaves the canvas, nothing prints under the floor (with
    mathtext subscripts counted at their shrunk size), panel (a)'s rotated names
    clear one another, and every panel that draws bars offers a y scale to read
    them against.  Returns the measurements so the caller can print them.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fw, fh = fig.get_size_inches() * fig.dpi

    texts = []
    for ax in fig.axes:
        texts.extend(ax.get_xticklabels())
        texts.extend(ax.get_yticklabels())
        texts.append(ax.title)
        texts.append(ax.xaxis.label)
        texts.append(ax.yaxis.label)
        texts.extend(ax.texts)
    texts.extend(fig.texts)
    for lg in fig.legends:
        texts.extend(lg.get_texts())

    inf = float("inf")
    sizes = []
    worst = {"left": -inf, "right": -inf, "bottom": -inf, "top": -inf}
    for t in texts:
        if not t.get_text() or not t.get_visible():
            continue
        sizes.append(_printed_size(t))
        bb = t.get_window_extent(renderer=r)
        worst["left"] = max(worst["left"], -bb.x0)
        worst["right"] = max(worst["right"], bb.x1 - fw)
        worst["bottom"] = max(worst["bottom"], -bb.y0)
        worst["top"] = max(worst["top"], bb.y1 - fh)

    # worst[k] is the overshoot past edge k: positive means clipped, and its
    # negation is the slack left over.
    over = {k: round(v, 2) for k, v in worst.items() if v > 0.0}
    if over:
        raise AssertionError(f"label(s) outside the {fw:.1f} x {fh:.1f} pt "
                             f"canvas by (pt): {over}")

    # Every panel that draws bars must offer a y scale to read them against.
    # This is the guard the regression that stripped (a)'s and (b)'s y labels
    # and (d)'s tick numbers walked straight through.
    axis_report = {}
    for ax in fig.axes:
        letter = ax.get_label() or ax.get_title()[1:2]
        if not ax.containers and not ax.patches:
            continue
        yt = [t for t in ax.get_yticklabels() if t.get_text() and t.get_visible()]
        if len(yt) < 2:
            raise AssertionError(
                f"panel ({letter}) draws bars but shows {len(yt)} y tick "
                f"number(s); a reader cannot read a bar height off it")
        small = [round(_printed_size(t), 2) for t in yt if _printed_size(t) < FS_MIN - 1e-9]
        if small:
            raise AssertionError(
                f"panel ({letter}) y tick numbers print at {small} pt, "
                f"under the {FS_MIN} pt floor")
        if not ax.get_ylabel():
            raise AssertionError(f"panel ({letter}) draws bars but has no y-axis label")
        axis_report[letter] = {
            "ylabel": ax.get_ylabel().replace("\n", " "),
            "yticks": [t.get_text() for t in yt],
            "xticklabels": [t.get_text() for t in ax.get_xticklabels()
                            if t.get_text()],
        }

    # Panel (a)'s rotated names run parallel to one another, so the clearance
    # that matters is measured across the rotation direction, not along it.
    ax_a = fig.axes[0]
    ticks = [t for t in ax_a.get_xticklabels() if t.get_text()]
    xs = [t.get_window_extent(renderer=r) for t in ticks]
    pitch = min(b.x1 - a.x1 for a, b in zip(xs, xs[1:])) if len(xs) > 1 else 0.0
    perpendicular = pitch * math.sin(math.radians(NAME_ROTATION))
    # bb.height of an anchored rotated label is the rotated box, not the line, so
    # the line height it has to clear is taken from the type size directly.
    line_h = FS_NAME * 1.10
    if perpendicular < line_h:
        raise AssertionError(
            f"panel (a) labels clear each other by only {perpendicular:.2f} pt "
            f"across the {NAME_ROTATION:.0f} degree direction, under the "
            f"{line_h:.2f} pt line")

    if min(sizes) < FS_MIN - 1e-9:
        raise AssertionError(f"a glyph prints at {min(sizes)} pt, under {FS_MIN}")

    return {
        "canvas_pt": [round(fw, 2), round(fh, 2)],
        "aspect": round(fw / fh, 3),
        "n_texts": len(sizes),
        "min_font_pt": round(min(sizes), 2),
        "margin_slack_pt": {k: round(float(-v), 2) for k, v in worst.items()},
        "panel_a_perpendicular_clearance_pt": round(perpendicular, 2),
        "panel_a_labels": [t.get_text() for t in ticks],
        "axes": axis_report,
    }


def draw(rec: dict, cd_alt: dict, source: str) -> dict:
    use_paper_style(base=FS_LABEL)
    plt.rcParams.update({
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "axes.labelsize": FS_LABEL,
        "axes.titlesize": FS_TITLE,
        "legend.fontsize": FS_LEGEND,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.pad": 2.0,
        "ytick.major.pad": 2.0,
    })

    # dpi = 72 so that one renderer pixel is one point: check_layout()
    # measures in the same units the page is written in.
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=72.0)
    axes = []
    for letter in "abcd":
        x0, w = PANEL_RECTS_PT[letter]
        ax = fig.add_axes([x0 / FIG_W_PT, AXES_BOT_PT / FIG_H_PT,
                           w / FIG_W_PT, AXES_H_PT / FIG_H_PT], label=letter)
        axes.append(ax)

    panel_a(axes[0])
    panel_b(axes[1], rec)
    panel_streak(axes[2], "win", source, cd_alt)
    panel_streak(axes[3], "loss", source, cd_alt)

    for ax, letter, title in zip(axes, "abcd",
                                 ["Bankruptcy Rate", "Irrationality Metrics",
                                  "After Win", "After Loss"]):
        style_axes(ax)
        ax.set_title(f"({letter}) {title}", fontweight="bold", fontsize=FS_TITLE,
                     pad=4.0)

    report = check_layout(fig)
    save_pdf_png(fig, OUT_DIR, STEM)
    plt.close(fig)
    return report


# ----------------------------------------------------------------- sidecar


def write_sidecar(rows: list[dict], cd_alt: dict, source: str,
                  report: dict | None = None) -> None:
    SIDECAR.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_by": "scripts/figures/fig02_slot_machine.py",
        "figure": "images/fig02_slot_machine.pdf",
        "layout": "1x4 horizontal band on the submitted canvas and axes rects",
        "legend": "boxed Fixed/Variable key inside every panel",
        "value_labels": "horizontal, above the bar or its interval cap",
        "canvas_inches": [round(FIG_W, 4), round(FIG_H, 4)],
        "canvas_points": [FIG_W_PT, FIG_H_PT],
        "aspect": round(ASPECT, 3),
        "canvas_source": "the submitted images/fig2_combined.pdf, verbatim",
        "font_sizes_pt": {"title": FS_TITLE, "axis_label": FS_LABEL,
                          "tick": FS_TICK, "model_name": FS_NAME,
                          "value_label": FS_VALUE, "legend": FS_LEGEND,
                          "bound_label": FS_BOUND,
                          "mathtext_subscript": FS_TICK_MATH * MATHTEXT_SHRINK},
        "printed_font_sizes_pt": [5.28, 4.66, 4.35, 4.04, 3.26],
        "min_font_size_pt": FS_MIN,
        "mathtext_subscript_base_pt": FS_TICK_MATH,
        "layout_check": report or {},
        "colors": {"fixed": COLORS["fixed"], "variable": COLORS["variable"],
                   "source": "fill operators of images/fig2_combined.pdf: "
                             "0.3490196078 0.631372549 0.3098039216 rg -> #59A14F; "
                             "0.8823529412 0.3411764706 0.3490196078 rg -> #E15759"},
        "promise": "reply to gbSA [W3]: 'Only the intervals are new.'",
        "model_order": MODEL_ORDER,
        "model_labels_drawn": MODEL_LABELS,
        "panel_a": {
            "quantity": "bankruptcy rate (%) per model, fixed vs variable",
            "ylabel": YLABEL_A.replace("\n", " "),
            "values": {m: {"fixed": float(SUBMITTED_BK["fixed"][i]),
                           "variable": float(SUBMITTED_BK["variable"][i])}
                       for i, m in enumerate(MODEL_ORDER)},
            "ci_source": "95% Wilson, verbatim from the gbSA [W3] table",
            "ci": {m: {"fixed": list(POSTED_CI["fixed"][m]),
                       "variable": list(POSTED_CI["variable"][m])}
                   for m in MODEL_ORDER},
            "n_games_per_model_and_arm": 1600,
            "ylim": [0, YLIM_A],
        },
        "panel_b": {
            "quantity": "round-level irrationality indicators, pooled over six models",
            "ylabel": YLABEL_B,
            "xticklabels_drawn": INDICATOR_LABELS,
            "indicators": INDICATORS,
            "values": {mode: [float(v) for v in SUBMITTED_METRICS[mode]]
                       for mode in ("fixed", "variable")},
            "ci_source": "percentile bootstrap over games within model, from "
                         "paper_data/fig02_slot_machine.json",
            "ci": ({mode: {n: [round(c, 5) for c in rec["indicators"][mode][n]["ci"]]
                           for n in INDICATORS} for mode in ("fixed", "variable")}
                   if (rec := load_recomputed()) else {}),
            "n_games": ({mode: {n: rec["indicators"][mode][n]["n_games"]
                                for n in INDICATORS}
                         for mode in ("fixed", "variable")} if rec else {}),
            "ylim": [0, YLIM_B],
        },
        "panels_cd": {
            "quantity": "bet-to-balance ratio increase by consecutive streak length",
            "ylabel": YLABEL_CD.replace("\n", " "),
            "streaks": STREAKS,
            "drawn_source": source,
            "submitted": {kind: {mode: [float(v) for v in SUBMITTED_STREAK[kind][mode]]
                                 for mode in ("fixed", "variable")}
                          for kind in ("win", "loss")},
            "submitted_interval_status": (
                "NO INTERVAL DRAWN. The submitted (c)/(d) literals have no "
                "derivation in either repository and could not be reproduced "
                "from the released corpora under any of 80 definition variants "
                "(5 model subsets x 4 transforms x 2 windows x 2 aggregators). "
                "The post-win fixed series is impossible rather than merely "
                "elusive: the fixed arm's wager is locked at exactly $10 in "
                "every round of all six corpora, so a win strictly raises the "
                "balance, the bet-to-balance ratio strictly falls, and "
                "max(0, .) makes the series exactly zero -- against the "
                "submitted [0.07, 0.09, 0.01, 0.02, 0.00]. Post-loss the same "
                "algebra forces monotonicity in k, and the submitted "
                "[0.24, 0.25, 0.21, 0.19, 0.29] dips at k=3,4. Only 3 of the 20 "
                "cells fall inside the interval of the closest recomputation, so "
                "drawing one would misstate the uncertainty."),
            "recomputed": cd_alt,
            "recomputed_definition": (
                "section 2's max(0, (r_{t+1}-r_t)/r_t) averaged over rounds whose "
                "run of identical outcomes has length exactly k; percentile "
                "bootstrap over rounds"),
            "ylim": [0, YLIM_CD],
        },
        "caption_multipliers": {
            "post_win": round(CAPTION_MULTIPLIERS["post_win"], 4),
            "post_loss": round(CAPTION_MULTIPLIERS["post_loss"], 4),
            "definition": "streak-length-1 ratio of the two plotted series "
                          "(0.23/0.07 and 0.67/0.24); prints as 3.3x and 2.8x",
        },
        "verification_vs_corpora": rows,
    }
    SIDECAR.write_text(json.dumps(payload, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cd-source", choices=("submitted", "recomputed"),
                    default="recomputed",
                    help="panels (c)/(d): the submitted literals (default, no "
                         "interval) or the corpus recomputation (with intervals)")
    args = ap.parse_args()

    rec = load_recomputed()
    rows = verify(rec)
    cd_alt = cd_recomputed(rec)
    if args.cd_source == "recomputed" and not cd_alt:
        raise SystemExit("--cd-source recomputed needs paper_data/fig02_slot_machine.json")

    report = draw(rec, cd_alt, args.cd_source)
    write_sidecar(rows, cd_alt, args.cd_source, report)

    print(f"canvas: {FIG_W:.4f} x {FIG_H:.4f} in = {FIG_W_PT:.1f} x {FIG_H_PT:.1f} pt "
          f"(aspect {ASPECT:.3f}); submitted 1274.4 x 327.8 pt (aspect 3.888)")
    print(f"measured min printed glyph size: {report['min_font_pt']} pt "
          f"over {report['n_texts']} labels (floor {FS_MIN})")
    print(f"panel (a) labels: {', '.join(report['panel_a_labels'])}")
    print(f"margin slack (pt): {report['margin_slack_pt']}")
    for letter, info in report["axes"].items():
        print(f"panel ({letter}) y: {info['ylabel']!r} ticks {info['yticks']}")
    if rows:
        bad = [r for r in rows if abs(r["delta"]) > 5e-3]
        print(f"verified {len(rows)} (a)/(b) cells against the corpora; "
              f"{len(bad)} disagree")
    print(f"panels (c)/(d) drawn from: {args.cd_source}")
    print(f"wrote {OUT_DIR / (STEM + '.pdf')}")
    print(f"wrote {OUT_DIR / (STEM + '.png')}")
    print(f"wrote {SIDECAR}")


if __name__ == "__main__":
    main()
