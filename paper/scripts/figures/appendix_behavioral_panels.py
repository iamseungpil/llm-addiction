#!/usr/bin/env python3
"""Regenerate appendix figures A01, A02, A06, A07, A08 at print size.

\\textwidth in this document is 396.0 pt (5.50 in) exactly -- measured from the
built paper, where a 0.95\\textwidth float prints 376.2 pt wide.

    A01  images/component_effects_by_bettype2.pdf        \\paperfigfull  -> 4.290 in
    A02  images/4model_complexity_trend_average3.pdf     \\paperfigfull  -> 4.290 in
    A06  images/component_effects_all_models_3x4_2.pdf   0.95\\textwidth -> 5.225 in
    A07  images/4model_complexity_trend_3x4.pdf          0.95\\textwidth -> 5.225 in
    A08  images/individual_model_streak_analysis.pdf     0.95\\textwidth -> 5.225 in

A01 and A02 are included with \\paperfigfull, which shared/paper_core.tex defines
as 0.78\\textwidth -- NOT 1.00\\textwidth.  Both the English and the Korean
appendix use it, and neurips_en.pdf confirms it: those two floats print 308.9 pt
wide, i.e. the submitted artwork was being squeezed to 0.394, not 0.51.

WHY THIS FILE EXISTS
--------------------
The generator published on HuggingFace under
``paper_neurips_2026/figures/appendix/figA01_component_effects_bettype/code/
generate_behavioral_figures.py`` (byte-identical copies also sit under figA02,
figA06, figA07 and figA08) does **not** draw any of these five figures.  It draws
``behavioral_gemma_vs_llama.png``, ``bet_type_asymmetry.png`` and
``prompt_component_effects.png`` from a collaborator's data mount.
The attachment is mislabelled.  No generator for the five figures above exists in
this repo or in the HF upload, and no sidecar records the plotted values
(see ``paper_index/appendix/fig_appendix_component_effects.md``).

The series below were therefore recovered from the submitted vector PDFs by
mapping every bar rectangle / marker centre back through each axes' gridline
calibration.  The recovery is exact to plotting precision: every value label
printed on the A01 artwork and every r-coefficient printed on A02 and A07
round-trips from these numbers.  Nothing has been re-derived from raw logs, so
the corpus-vintage caveats in ``paper_index/appendix/`` still stand unchanged.

NOTHING VISUAL WAS RE-DESIGNED.  Same panels, same order, same chart types, same
data, same colour semantics.  Canvas size, font sizes and spacing changed so the
figures are legible at the width they are actually printed.  Three placement
changes were forced by the narrower canvas, and nothing else:
  * A01 value labels are set vertically (see FIX A01 in fig_a01).
  * A02 panel titles are centred over their axes rather than left-anchored,
    because "(c) Average Total Bet" no longer fits to the right of a 69 pt panel.
  * A02 and A07 x-axis labels wrap onto two lines.  As a side effect the A07
    artwork no longer loses the closing bracket of its fourth x-label, which the
    submitted version truncated at the page edge.
No panel was reordered, no label was dropped, and every number that the
submitted artwork printed is still printed.

VENDORED CONSTANTS
------------------
None from ``paper_figure_style`` -- that module is not imported by the original
generator either.  The palette, grid colour and ink colour below were read
straight out of the submitted PDFs' fill/stroke operators, so they reproduce the
submitted colours exactly.

Usage:  python3 scripts/figures/appendix_behavioral_panels.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

OUT = Path(__file__).resolve().parents[2] / "images"

TEXTWIDTH_PT = 396.0                       # \\textwidth of the NeurIPS class here
W_FULL = 0.78 * TEXTWIDTH_PT / 72.0        # \\paperfigfull  = 4.290 in
W_95 = 0.95 * TEXTWIDTH_PT / 72.0          # 0.95\\textwidth = 5.225 in

# --------------------------------------------------------------------------
# Canvas heights.  The width is the printed width; the height is NOT free.
# Each submitted PDF has an aspect (its own MediaBox, w/h) and printed at that
# aspect, so the height here is printed width / submitted aspect.  Choosing
# heights for the content instead made every one of these floats taller than the
# submitted one and changed the shape of the page.
# --------------------------------------------------------------------------
ASPECT = {                                 # submitted MediaBox w / h
    "component_effects_by_bettype2": 784.847 / 264.140,      # 2.97133
    "4model_complexity_trend_average3": 774.816 / 258.596,   # 2.99624
    "component_effects_all_models_3x4_2": 816.794 / 533.585, # 1.53077
    "4model_complexity_trend_3x4": 808.419 / 539.920,        # 1.49729
    "individual_model_streak_analysis": 805.247 / 419.536,   # 1.91938
}


def canvas(name, width_in):
    """(width, height) in inches at the submitted aspect."""
    return width_in, width_in / ASPECT[name]

# --------------------------------------------------------------------------
# Palette, read out of the submitted PDFs (device RGB of the fill operators)
# --------------------------------------------------------------------------
GREEN = "#59A14F"   # (0.349, 0.631, 0.310)  "Fixed" / "Win streak"
RED = "#E15759"     # (0.882, 0.341, 0.349)  "Variable" / "Loss streak"
BLUE = "#4C72B0"    # (0.298, 0.447, 0.690)
ORANGE = "#DD8452"  # (0.867, 0.518, 0.322)
INK = "#444444"     # (0.267, 0.267, 0.267)  spines, ticks, text
GRIDC = "#DDDDDD"   # (0.867, 0.867, 0.867)  horizontal gridlines

# --------------------------------------------------------------------------
# Type scale.  The canvas is now the printed size, so 1 pt here == 1 pt on the
# page.  Nothing below 7 pt.
# --------------------------------------------------------------------------
FS_SUPTITLE = 10.0
FS_TITLE = 8.5
FS_COLTITLE = 7.5   # 4-model column headers: "Claude-3.5-Haiku" must fit its column
FS_AXLABEL = 8.0
FS_TICK = 7.0
FS_LEGEND = 7.0
FS_VALUE = 7.0
FS_RBOX = 7.5

BASE_RC = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": FS_TICK,
    "axes.labelsize": FS_AXLABEL,
    "axes.titlesize": FS_TITLE,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_LEGEND,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRIDC,
    "grid.linewidth": 0.6,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "pdf.fonttype": 42,
}

# AXIS LABELS ARE WRAPPED, not shortened.  At the submitted aspects a panel row
# is 59-85 pt tall while "Bankruptcy effect (pp)" is 89 pt long at 8 pt, so a
# one-line y label overran its own axes at both ends: on A07 and A08 the labels
# of neighbouring rows printed on top of each other, and on A01 and A02 the top
# one ran off the canvas.  Every word is kept; only the line break is new.

COMPONENTS = ["G", "M", "P", "H", "W"]
MODELS = ["GPT-4o-mini", "GPT-4.1-mini", "Gemini-2.5-Flash", "Claude-3.5-Haiku"]
MODEL_COLORS = [BLUE, ORANGE, GREEN, RED]

# ==========================================================================
# DATA -- recovered from the submitted vector PDFs (see module docstring)
# ==========================================================================

# --- A01 / component_effects_by_bettype2 ----------------------------------
# marginal effect of each prompt component, split by bet type, pooled models
A01 = {
    "Bankruptcy": {  # percentage points
        "ylabel": "Bankruptcy\neffect (pp)",
        "F": [0.896, 1.021, 0.146, -0.438, 0.438],
        "V": [14.729, 19.729, 2.479, 1.354, 11.646],
        "fmt": "{:+.1f}",
    },
    "Total Bet": {  # dollars
        "ylabel": "Total bet\neffect ($)",
        "F": [39.423, 14.297, -0.028, -9.829, 16.190],
        "V": [171.668, 89.040, 47.920, -41.560, 77.523],
        "fmt": "{:+.0f}",
    },
    "Rounds": {
        "ylabel": "Rounds effect",
        "F": [3.943, 1.430, -0.002, -0.983, 1.619],
        "V": [5.428, 0.186, 2.823, -1.759, 1.531],
        "fmt": "{:+.1f}",
    },
}

# --- A02 / 4model_complexity_trend_average3 -------------------------------
# pooled mean per prompt complexity level 0..5
A02 = [
    {
        "title": "(a) Bankruptcy",
        "ylabel": "Bankruptcy\nrate (%)",
        "color": RED,
        "y": [3.834, 7.0672, 11.833, 17.3676, 22.500, 29.6673],
    },
    {
        "title": "(b) Average Rounds",
        "ylabel": "Avg. game\nrounds",
        "color": BLUE,
        "y": [4.4883, 6.8986, 8.3925, 10.0448, 11.0619, 11.4351],
    },
    {
        "title": "(c) Average Total Bet",
        "ylabel": "Avg. total\nbet ($)",
        "color": GREEN,
        "y": [57.5127, 100.9799, 134.9746, 179.2161, 226.6634, 239.4145],
    },
]

# --- A06 / component_effects_all_models_3x4_2 -----------------------------
# rows: bankruptcy (pp), total bet ($), rounds; cols: the four API models
A06_ROWS = ["Bankruptcy\neffect (pp)", "Total bet\neffect ($)", "Rounds effect"]
A06 = {
    (0, 0): {"F": [0.0, 0.0, 0.0, 0.0, 0.0],
             "V": [7.628, 8.125, 1.379, -2.876, 17.376]},
    (0, 1): {"F": [0.0, 0.0, 0.0, 0.0, 0.0],
             "V": [-1.127, 8.377, -3.625, 6.879, 11.875]},
    (0, 2): {"F": [4.752, 5.500, 0.504, -2.246, 1.749],
             "V": [23.128, 58.621, 6.375, -6.375, 11.623]},
    (0, 3): {"F": [0.0, 0.0, 0.0, 0.0, 0.0],
             "V": [32.750, 20.000, 4.248, 6.004, 15.248]},
    (1, 0): {"F": [19.086, 3.482, -0.187, -17.905, 8.517],
             "V": [127.199, 6.030, 18.962, -38.483, 30.090]},
    (1, 1): {"F": [27.230, 8.082, 3.295, -3.606, 16.848],
             "V": [79.391, 72.241, -2.176, -17.097, 76.904]},
    (1, 2): {"F": [65.776, 37.613, -3.171, -42.711, 16.724],
             "V": [165.682, 165.496, 77.899, -127.821, 74.479]},
    (1, 3): {"F": [38.545, 7.709, 1.368, -13.615, 31.644],
             "V": [417.221, 195.399, 68.387, -66.584, 164.377]},
    (2, 0): {"F": [1.909, 0.347, -0.020, -1.788, 0.852],
             "V": [6.954, -0.243, 0.828, -3.175, -0.899]},
    (2, 1): {"F": [2.722, 0.804, 0.330, -0.363, 1.683],
             "V": [10.517, 3.690, 1.501, -3.642, 3.321]},
    (2, 2): {"F": [6.580, 3.763, -0.321, -4.273, 1.673],
             "V": [3.721, -1.542, 2.078, -2.262, 1.047]},
    (2, 3): {"F": [3.851, 0.774, 0.136, -1.361, 3.161],
             "V": [7.808, 5.686, 8.025, -1.702, 5.402]},
}
# y limits of the submitted panels, kept so the row scales stay shared
A06_YLIM = {0: (-10.0, 68.0), 1: (-215.0, 470.0), 2: (-6.4, 12.0)}
A06_YTICKS = {0: [0, 20, 40, 60], 1: [-200, 0, 200, 400], 2: [-5, 0, 5, 10]}

# --- A07 / 4model_complexity_trend_3x4 ------------------------------------
A07_ROWS = ["Bankruptcy\nrate (%)", "Avg. game\nrounds", "Avg. total\nbet ($)"]
A07 = {
    (0, 0): [0.999, 6.199, 9.100, 12.800, 13.600, 22.001],
    (0, 1): [0.000, 0.000, 1.300, 4.900, 5.201, 13.001],
    (0, 2): [0.000, 9.803, 19.602, 32.099, 41.400, 45.997],
    (0, 3): [0.001, 0.799, 4.500, 11.601, 24.799, 39.001],
    (1, 0): [1.280, 2.682, 3.722, 3.922, 4.192, 3.980],
    (1, 1): [0.720, 1.740, 4.071, 6.055, 8.506, 9.360],
    (1, 2): [1.440, 2.332, 4.860, 6.016, 5.918, 5.110],
    (1, 3): [5.110, 10.766, 15.592, 18.317, 20.888, 20.140],
    (2, 0): [24.749, 45.828, 68.962, 83.390, 88.947, 117.451],
    (2, 1): [5.349, 13.283, 37.187, 69.494, 94.494, 114.626],
    (2, 2): [14.460, 40.823, 99.666, 141.293, 196.161, 152.944],
    (2, 3): [47.686, 133.902, 231.763, 301.479, 421.681, 395.812],
}

# --- A08 / individual_model_streak_analysis -------------------------------
A08_ROWS = ["Bet increase\nrate", "Continuation\nrate"]
A08 = {
    (0, 0): {"W": [0.3305, 0.3678, 0.4065, 0.3333, 0.2500],
             "L": [0.1647, 0.1415, 0.0797, 0.1057, 0.1189]},
    (0, 1): {"W": [0.2710, 0.2954, 0.3529, 0.2857, 0.5000],
             "L": [0.1166, 0.1459, 0.1092, 0.1633, 0.2355]},
    (0, 2): {"W": [0.2908, 0.4867, 0.4348, 0.2941, 0.8333],
             "L": [0.2980, 0.3730, 0.4312, 0.4151, 0.2959]},
    (0, 3): {"W": [0.5262, 0.5427, 0.6425, 0.6574, 0.6964],
             "L": [0.2251, 0.2347, 0.0709, 0.1673, 0.1119]},
    (1, 0): {"W": [0.9517, 0.8891, 0.9337, 0.9575, 0.8000],
             "L": [0.9146, 0.8534, 0.7664, 0.7653, 0.7371]},
    (1, 1): {"W": [0.9348, 0.9154, 0.8901, 0.9655, 1.0000],
             "L": [0.9383, 0.8828, 0.8874, 0.8673, 0.7667]},
    (1, 2): {"W": [0.8741, 0.8982, 0.8118, 1.0000, 1.0000],
             "L": [0.9501, 0.8108, 0.7637, 0.6418, 0.5868]},
    (1, 3): {"W": [0.9911, 0.9885, 0.9680, 0.9643, 0.9180],
             "L": [0.9930, 0.9791, 0.9436, 0.9437, 0.8773]},
}


# ==========================================================================
# helpers
# ==========================================================================

def _pt_to_data(ax, pts: float) -> float:
    """Convert a vertical distance in points to data units on ``ax``."""
    h_in = ax.get_window_extent().height / ax.figure.dpi
    lo, hi = ax.get_ylim()
    return pts / 72.0 / h_in * (hi - lo)


def _fit_rotated_labels(fig, ax, labels, lo, hi, gap_pt=2.5, pad_pt=2.0):
    """Nudge each rotated value label clear of its bar end and grow the y-limits
    until every label fits inside the axes.  Iterates because the point->data
    conversion changes as the limits change."""
    for _ in range(6):
        gap = _pt_to_data(ax, gap_pt)
        for txt, val, up in labels:
            txt.set_position((txt.get_position()[0], val + (gap if up else -gap)))
        fig.canvas.draw()
        inv = ax.transData.inverted()
        tops, bots = [], []
        for txt, _v, _u in labels:
            bb = txt.get_window_extent(fig.canvas.get_renderer())
            bots.append(inv.transform((0, bb.y0))[1])
            tops.append(inv.transform((0, bb.y1))[1])
        y0, y1 = ax.get_ylim()
        pad = _pt_to_data(ax, pad_pt)
        need0 = min(min(bots) - pad, lo - _pt_to_data(ax, 1.0))
        need1 = max(max(tops) + pad, hi)
        if need0 > y0 - 1e-9 and need1 < y1 + 1e-9 and \
                abs(need0 - y0) < 0.02 * (y1 - y0) and \
                abs(need1 - y1) < 0.02 * (y1 - y0):
            break
        ax.set_ylim(need0, need1)
        fig.canvas.draw()


def _trend(y):
    x = np.arange(len(y), dtype=float)
    m, b = np.polyfit(x, y, 1)
    return x, m * x + b, float(np.corrcoef(x, y)[0, 1])


def _rbox(ax, r):
    ax.text(0.04, 0.95, f"r = {r:.3f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=FS_RBOX,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#BBBBBB", linewidth=0.7))


def _zero_line(ax):
    ax.axhline(0.0, color=INK, linewidth=0.8, zorder=2)


def _save(fig, name, width_in):
    """Save at exactly the canvas size and report anything that fell off it.

    The overflow is measured with matplotlib's own tight bbox, not by reading the
    PDF back: MuPDF/pypdf clip extraction to the page box, so text that lands
    entirely outside simply disappears from the readback and would look clean.
    """
    fig.canvas.draw()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    fw, fh = fig.get_size_inches()
    over = (round(max(0.0, -tb.x0) * 72, 2), round(max(0.0, tb.x1 - fw) * 72, 2),
            round(max(0.0, tb.y1 - fh) * 72, 2), round(max(0.0, -tb.y0) * 72, 2))
    path = OUT / f"{name}.pdf"
    fig.savefig(path)                      # NO bbox_inches -- page == canvas
    plt.close(fig)
    import pypdf
    box = pypdf.PdfReader(str(path)).pages[0].mediabox
    w, h = float(box.width), float(box.height)
    flag = "" if max(over) <= 0.05 else "   <-- CLIPPED"
    print(f"  {path.name:<44s} page {w/72:.3f} x {h/72:.3f} in"
          f"   scale = {width_in * 72 / w:.3f}"
          f"   overflow(l,r,t,b)pt = {over}{flag}")


# ==========================================================================
# A01 -- component effects by betting type (1 x 3)
# ==========================================================================

def fig_a01():
    """FIX A01: in the submitted artwork the horizontal value labels collide --
    +1.6/+1.5 overlapped at W, -0.0 sat on the zero line, -1.0/-1.8 fell inside
    their own bars.  At half the canvas width a horizontal label is ~20 pt wide
    while a bar group is ~20 pt, so horizontal placement cannot hold ten labels
    per panel at >=7 pt.  Every label is kept, set vertically, and pushed clear
    of the bar end (above for positive, below for negative)."""
    with plt.rc_context(BASE_RC):
        fig, axes = plt.subplots(1, 3, figsize=canvas("component_effects_by_bettype2", W_FULL), layout="constrained")
        fig.get_layout_engine().set(w_pad=0.03, h_pad=0.02, wspace=0.06,
                                    rect=(0.0, 0.0, 0.974, 1.0))

        x = np.arange(len(COMPONENTS))
        w = 0.38
        for ax, (key, d) in zip(axes, A01.items()):
            f, v = np.array(d["F"]), np.array(d["V"])
            ax.bar(x - w / 2, f, w, color=GREEN, label="Fixed", zorder=3)
            ax.bar(x + w / 2, v, w, color=RED, label="Variable", zorder=3)
            _zero_line(ax)
            ax.set_xticks(x)
            ax.set_xticklabels(COMPONENTS)
            ax.set_xlabel("Prompt component")
            ax.set_ylabel(d["ylabel"])
            ax.set_title(f"({'abc'[list(A01).index(key)]}) {key}",
                         fontweight="bold", loc="left", pad=3)
            ax.set_xlim(-0.62, len(COMPONENTS) - 0.38)

            lo = min(0.0, float(min(f.min(), v.min())))
            hi = max(0.0, float(max(f.max(), v.max())))
            span = hi - lo or 1.0
            ax.set_ylim(lo - 0.10 * span, hi + 0.42 * span)

            labels = []
            fig.canvas.draw()
            # The label x positions are NOT the bar centres.  Rotated, a 7 pt
            # label is ~7 pt wide, and at the submitted aspect a panel is 72 pt
            # for five groups: bar centres are 5.5 pt apart within a pair, so
            # pair-mates printed on top of each other.  Spreading each pair to
            # +/-0.25 of the group centre makes every gap -- inside a pair and
            # between pairs -- 7.2 pt, and each label still sits over its own
            # bar (the bars span +/-0.0 to +/-0.38).
            lab_dx = 0.25
            for xi, val in (list(zip(x - lab_dx, f)) + list(zip(x + lab_dx, v))):
                up = val >= 0
                labels.append((ax.text(xi, val, d["fmt"].format(val),
                                       rotation=90, ha="center",
                                       va="bottom" if up else "top",
                                       fontsize=FS_VALUE, zorder=4), val, up))
            _fit_rotated_labels(fig, ax, labels, lo, hi)

        # FIX A01 (key): the per-panel key used to stand inside the plotting
        # rectangle, and _clear_legend bought it room by pushing the y limit up
        # -- 22 pt of a 67 pt panel, taken straight out of the bars.  It is the
        # same two entries in all three panels, so it is now one key under the
        # figure, exactly as A08 does.
        handles = [Patch(facecolor=GREEN, label="Fixed"),
                   Patch(facecolor=RED, label="Variable")]
        fig.legend(handles=handles, loc="outside lower center", ncol=2,
                   frameon=False, fontsize=FS_LEGEND, handlelength=1.2,
                   handleheight=0.8, handletextpad=0.4, columnspacing=2.0,
                   borderaxespad=0.15)
        _save(fig, "component_effects_by_bettype2", W_FULL)


# ==========================================================================
# A02 -- pooled complexity trend (1 x 3)
# ==========================================================================

def fig_a02():
    with plt.rc_context(BASE_RC):
        fig, axes = plt.subplots(1, 3, figsize=canvas("4model_complexity_trend_average3", W_FULL), layout="constrained")
        fig.get_layout_engine().set(w_pad=0.03, h_pad=0.02, wspace=0.06,
                                    rect=(0.0, 0.0, 0.940, 1.0))
        for ax, d in zip(axes, A02):
            y = np.array(d["y"])
            xs, tl, r = _trend(y)
            ax.plot(xs, tl, ls="--", lw=1.1, color=d["color"], alpha=0.35, zorder=2)
            ax.plot(xs, y, "-o", lw=1.6, color=d["color"], ms=4.0, mfc="white",
                    mew=1.2, zorder=3)
            ax.set_xticks(xs)
            ax.set_xlabel("Prompt complexity\n(# components)")
            ax.set_ylabel(d["ylabel"])
            # centred, not axes-left as in the submitted artwork: at 4.29 in
            # "(c) Average Total Bet" is 92 pt wide against a 69 pt panel, so a
            # left-anchored title runs off the page.  Centring halves the
            # overhang and keeps the whole title.
            ax.set_title(d["title"], fontweight="bold", loc="center", pad=3,
                         fontsize=FS_COLTITLE)
            ax.margins(x=0.06)
            ax.set_ylim(top=ax.get_ylim()[1] + 0.22 * (np.ptp(ax.get_ylim())))
            _rbox(ax, r)
        _save(fig, "4model_complexity_trend_average3", W_FULL)


# ==========================================================================
# A06 -- component effects, one column per model (3 x 4)
# ==========================================================================

def fig_a06():
    with plt.rc_context(BASE_RC):
        fig, axes = plt.subplots(3, 4, figsize=canvas("component_effects_all_models_3x4_2", W_95), layout="constrained")
        # rect leaves a right/top strip so the widest column header
        # ("Claude-3.5-Haiku") and the last x-label stay on the page
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.05,
                                    hspace=0.05, rect=(0.0, 0.0, 0.972, 0.994))
        fig.suptitle("Component Effects by Betting Type",
                     fontsize=FS_SUPTITLE, fontweight="bold")
        x = np.arange(len(COMPONENTS))
        w = 0.38
        for r in range(3):
            for c in range(4):
                ax = axes[r, c]
                d = A06[(r, c)]
                ax.bar(x - w / 2, d["F"], w, color=GREEN, label="Fixed", zorder=3)
                ax.bar(x + w / 2, d["V"], w, color=RED, label="Variable", zorder=3)
                _zero_line(ax)
                ax.set_xticks(x)
                ax.set_xlim(-0.62, len(COMPONENTS) - 0.38)
                ax.set_ylim(*A06_YLIM[r])
                ax.set_yticks(A06_YTICKS[r])
                if r == 0:
                    ax.set_title(MODELS[c], fontweight="bold", pad=4, fontsize=FS_COLTITLE)
                if r == 2:
                    ax.set_xticklabels(COMPONENTS)
                    ax.set_xlabel("Prompt component")
                else:
                    ax.set_xticklabels([])
                if c == 0:
                    ax.set_ylabel(A06_ROWS[r])
                if (r, c) == (0, 3):
                    ax.legend(loc="upper right", frameon=True, framealpha=0.95,
                              edgecolor="#CCCCCC", borderpad=0.3,
                              handlelength=1.0, handletextpad=0.4,
                              labelspacing=0.25, borderaxespad=0.2)
        _save(fig, "component_effects_all_models_3x4_2", W_95)


# ==========================================================================
# A07 -- complexity trend, one column per model (3 x 4)
# ==========================================================================

def fig_a07():
    with plt.rc_context(BASE_RC):
        fig, axes = plt.subplots(3, 4, figsize=canvas("4model_complexity_trend_3x4", W_95), layout="constrained")
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.05,
                                    hspace=0.05, rect=(0.0, 0.0, 0.982, 0.994))
        fig.suptitle("Gambling Behavior vs. Prompt Complexity",
                     fontsize=FS_SUPTITLE, fontweight="bold")
        for r in range(3):
            for c in range(4):
                ax = axes[r, c]
                y = np.array(A07[(r, c)])
                col = MODEL_COLORS[c]
                xs, tl, rr = _trend(y)
                ax.plot(xs, tl, ls="--", lw=1.0, color=col, alpha=0.35, zorder=2)
                ax.plot(xs, y, "-o", lw=1.5, color=col, ms=3.4, mfc="white",
                        mew=1.1, zorder=3)
                ax.set_xticks(xs)
                ax.margins(x=0.07)
                ax.set_ylim(top=ax.get_ylim()[1] + 0.26 * np.ptp(ax.get_ylim()))
                ax.yaxis.set_major_locator(plt.MaxNLocator(4))
                if r == 0:
                    ax.set_title(MODELS[c], fontweight="bold", pad=4, fontsize=FS_COLTITLE)
                if r == 2:
                    ax.set_xlabel("Prompt complexity\n(# components)", fontsize=FS_COLTITLE)
                else:
                    ax.set_xticklabels([])
                if c == 0:
                    ax.set_ylabel(A07_ROWS[r])
                _rbox(ax, rr)
        _save(fig, "4model_complexity_trend_3x4", W_95)


# ==========================================================================
# A08 -- streak analysis, one column per model (2 x 4)
# ==========================================================================

def fig_a08():
    with plt.rc_context(BASE_RC):
        fig, axes = plt.subplots(2, 4, figsize=canvas("individual_model_streak_analysis", W_95), layout="constrained")
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.05,
                                    hspace=0.05, rect=(0.0, 0.0, 0.982, 0.994))
        fig.suptitle("Individual Model Streak Analysis",
                     fontsize=FS_SUPTITLE, fontweight="bold")
        x = np.arange(1, 6)
        w = 0.38
        for r in range(2):
            for c in range(4):
                ax = axes[r, c]
                d = A08[(r, c)]
                ax.bar(x - w / 2, d["W"], w, color=GREEN, label="Win streak", zorder=3)
                ax.bar(x + w / 2, d["L"], w, color=RED, label="Loss streak", zorder=3)
                ax.set_xticks(x)
                ax.set_xlim(0.4, 5.6)
                if r == 0:
                    ax.set_ylim(0, 1.0)
                    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
                    ax.set_title(MODELS[c], fontweight="bold", pad=4, fontsize=FS_COLTITLE)
                    ax.set_xticklabels([])
                else:
                    ax.set_ylim(0, 1.12)
                    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                    ax.set_xlabel("Streak length")
                if c == 0:
                    ax.set_ylabel(A08_ROWS[r])
        # FIX A08: the key used to sit inside the Claude-3.5-Haiku panel, on top
        # of its plot area, and its swatches are the same shape and colour as the
        # bars beside them -- which is how one reader took the green swatch for a
        # sixth win-streak bar.  It is one key for all eight panels, so it now
        # runs outside the axes, under the grid, where nothing can be mistaken
        # for data.  The layout engine reserves the strip, so no panel is
        # overdrawn and no value moves.
        handles = [Patch(facecolor=GREEN, label="Win streak"),
                   Patch(facecolor=RED, label="Loss streak")]
        fig.legend(handles=handles, loc="outside lower center", ncol=2,
                   frameon=False, fontsize=FS_LEGEND, handlelength=1.2,
                   handleheight=0.8, handletextpad=0.4, columnspacing=2.0,
                   borderaxespad=0.15)
        _save(fig, "individual_model_streak_analysis", W_95)


def main():
    print(f"Writing to {OUT}")
    fig_a01()
    fig_a02()
    fig_a06()
    fig_a07()
    fig_a08()


if __name__ == "__main__":
    main()
