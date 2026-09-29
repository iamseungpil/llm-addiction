#!/usr/bin/env python3
r"""Appendix figures in one house style, authored at the width they print at.

Writes ten NEW files under images/ (prefix ``appx_``); the submitted artwork in
images/ is never touched.  Every canvas is the printed size, so LaTeX must include
each file at scale 1.0:

  FULL = 5.5 in = 396 pt  (\textwidth)          -> \includegraphics[width=\textwidth]
  HALF = 2.7 in = 194.4 pt (~0.49\textwidth)    -> \includegraphics[width=0.49\textwidth]
                                                   (0.49\textwidth = 194.04 pt; scale 0.998)

  appx_component_effects.pdf        A01  <- component_effects_by_bettype2
  appx_complexity_average.pdf       A02  <- 4model_complexity_trend_average3
  appx_escalation.pdf               A03  <- escalation_trajectory             (HALF)
  appx_temperature.pdf              A04  <- temperature_robustness            (HALF)
  appx_language_markers.pdf         A05  <- distortion_multimodel_summary
  appx_component_effects_models.pdf A06  <- component_effects_all_models_3x4_2
  appx_complexity_models.pdf        A07  <- 4model_complexity_trend_3x4
  appx_streak_models.pdf            A08  <- individual_model_streak_analysis
  appx_choice_distribution.pdf      A09  <- investment_choice_distributions_cot
  appx_condition_ladders.pdf        xctx <- fig_xctx_ladders_solo

DATA (nothing is recomputed; every value comes from an existing source):
  A01 A02 A06 A07 A08  the recovered series in appendix_behavioral_panels.py
  A03 A04 A09          data/figA0{3,4,9}_*.json
  A05                  the values printed on the CURRENT images/distortion_multimodel_summary.pdf
                       (decision-level version, commit ec5945c), recovered from its text and bar
                       geometry -- NOT data/figA05_*.json, which holds the older, submitted
                       game-level numbers.  GPT-4o-mini x Loss chasing is the re-derived +5.5,
                       i.e. the NaN fix of figA05 is kept.
  ladders              images/fig_cross_context_write_values.json (per-condition summaries)

STYLE (one block, shared by all ten):
  palette   fixed #59A14F, variable #E15759, variable_light #E8A0A0, neutral #C7C7C7,
            ink #444444, grid #DDDDDD 0.6 pt, spines 0.8 pt (top/right hidden);
            extra series: blue #4E79A7, orange #F28E2B, purple #B07AA1
            (A02/A07 metrics: bankruptcy blue, rounds orange, total bet purple, so green/red
            keep meaning Fixed/Variable everywhere in the appendix)
  type      DejaVu Sans throughout (mathtext dejavusans); ticks 7, axis labels 7.5,
            panel titles 8 bold left-aligned, legends 7, value labels >= 6.5
  layout    no suptitles; one shared legend per figure in a single row above the panels
            (A05: in the empty upper-right of panel b); a shared x label under grids whose
            panels share the x variable; savefig WITHOUT bbox_inches so page == canvas

Run:  cd llm-addiction/paper && python3 scripts/figures/appx_unified.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import appendix_behavioral_panels as ABP  # noqa: E402  (data only; importing draws nothing)

OUT = HERE.parents[1] / "images"
DATA = HERE / "data"

# =============================================================== style =====
FULL = 5.5
HALF = 2.7

FIXED = "#59A14F"
VARIABLE = "#E15759"
VARIABLE_LIGHT = "#E8A0A0"
NEUTRAL = "#C7C7C7"
INK = "#444444"
GRID = "#DDDDDD"
BLUE = "#4E79A7"
ORANGE = "#F28E2B"
PURPLE = "#B07AA1"
TITLE_INK = "#000000"

FS_TICK = 7.0
FS_LABEL = 7.5
FS_TITLE = 8.0
FS_LEGEND = 7.0
FS_VALUE = 7.0
FS_VALUE_SMALL = 6.5   # rotated bar labels of A01 only
MIN_PT = 6.5

METRIC_COLORS = {"bankruptcy": BLUE, "rounds": ORANGE, "totalbet": PURPLE}
CHOICE_COLORS = [FIXED, NEUTRAL, VARIABLE_LIGHT, VARIABLE]
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "paper_diverging", ["#3B6A99", "#6F95B9", "#A6C2DB", "#D2E0ED", "#FFFEFE",
                        "#F4D0D0", "#E8A0A0", "#CD6D6D", "#B23A3A"])

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": FS_TICK,
    "axes.titlesize": FS_TITLE,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 3.0,
    "axes.titlecolor": TITLE_INK,
    "axes.labelsize": FS_LABEL,
    "axes.labelpad": 2.0,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_LEGEND,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.pad": 2.0,
    "ytick.major.pad": 2.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "legend.handlelength": 1.4,
    "legend.handleheight": 0.8,
    "legend.handletextpad": 0.4,
    "legend.columnspacing": 1.4,
    "legend.borderaxespad": 0.2,
    "legend.borderpad": 0.2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 150,
    "savefig.dpi": 300,
}
plt.rcParams.update(RC)


def new_fig(width, height, **kw):
    fig = plt.figure(figsize=(width, height), layout="constrained", **kw)
    fig.get_layout_engine().set(w_pad=2.0 / 72, h_pad=2.0 / 72, wspace=0.03, hspace=0.03)
    return fig


def top_legend(fig, handles, labels=None, ncol=None):
    kw = dict(handles=handles, loc="outside upper center", ncol=ncol or len(handles),
              fontsize=FS_LEGEND, frameon=False)
    if labels is not None:
        kw["labels"] = labels
    return fig.legend(**kw)


def patch(color, label):
    return Patch(facecolor=color, edgecolor="none", label=label)


def zero_line(ax):
    ax.axhline(0.0, color=INK, linewidth=0.8, zorder=2)


def rbox(ax, r):
    ax.text(0.04, 0.96, f"r = {r:.3f}", transform=ax.transAxes, ha="left", va="top",
            fontsize=FS_VALUE, zorder=5,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#BBBBBB",
                      linewidth=0.6))


def trend(y):
    x = np.arange(len(y), dtype=float)
    m, b = np.polyfit(x, y, 1)
    return x, m * x + b, float(np.corrcoef(x, y)[0, 1])


def pt_to_data(ax, pts):
    h_in = ax.get_window_extent().height / ax.figure.dpi
    lo, hi = ax.get_ylim()
    return pts / 72.0 / h_in * (hi - lo)


REPORT = []


def save(fig, name):
    """Save at exactly the canvas size; report overflow and the smallest font drawn."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tb = fig.get_tightbbox(renderer)
    fw, fh = fig.get_size_inches()
    over = (max(0.0, -tb.x0) * 72, max(0.0, tb.x1 - fw) * 72,
            max(0.0, tb.y1 - fh) * 72, max(0.0, -tb.y0) * 72)
    sizes = [t.get_fontsize() for t in fig.findobj(matplotlib.text.Text)
             if t.get_visible() and t.get_text().strip()]
    path = OUT / f"{name}.pdf"
    fig.savefig(path)                           # NO bbox_inches: page == canvas
    plt.close(fig)
    flag = "" if max(over) <= 0.05 else "   <-- CLIPPED"
    line = (f"{path.name:<36s} {fw * 72:6.1f} x {fh * 72:6.1f} pt  min font {min(sizes):.1f} pt"
            f"  overflow(l,r,t,b) = {tuple(round(v, 2) for v in over)}{flag}")
    REPORT.append(line)
    print(line)
    assert min(sizes) >= MIN_PT - 1e-9, (name, min(sizes))


# ======================================================== A01 (1 x 3) =====
def fig_component_effects():
    comps = ABP.COMPONENTS
    fig = new_fig(FULL, 1.9)
    axes = fig.subplots(1, 3)
    x = np.arange(len(comps))
    w = 0.38
    all_labels = []
    for k, (ax, (key, d)) in enumerate(zip(axes, ABP.A01.items())):
        f, v = np.array(d["F"]), np.array(d["V"])
        ax.bar(x - w / 2, f, w, color=FIXED, zorder=3)
        ax.bar(x + w / 2, v, w, color=VARIABLE, zorder=3)
        zero_line(ax)
        ax.set_xticks(x, comps)
        ax.set_xlim(-0.62, len(comps) - 0.38)
        ax.set_ylabel(d["ylabel"].replace("\n", " "))
        ax.set_title(f"({'abc'[k]}) {key}")
        lo = min(0.0, float(min(f.min(), v.min())))
        hi = max(0.0, float(max(f.max(), v.max())))
        span = hi - lo
        ax.set_ylim(lo - 0.25 * span, hi + 0.30 * span)
        labs = []
        for xi, val in list(zip(x - 0.22, f)) + list(zip(x + 0.22, v)):
            up = val >= 0
            labs.append((ax.text(xi, val, d["fmt"].format(val), rotation=90, ha="center",
                                 va="bottom" if up else "top", fontsize=FS_VALUE_SMALL,
                                 zorder=4), val, up))
        all_labels.append((ax, labs, lo, hi))
    fig.supxlabel("Prompt component", fontsize=FS_LABEL)
    top_legend(fig, [patch(FIXED, "Fixed"), patch(VARIABLE, "Variable")])
    # push every rotated label clear of its bar end and grow the y-limits until they fit
    for _ in range(8):
        fig.canvas.draw()
        changed = False
        for ax, labs, lo, hi in all_labels:
            gap = pt_to_data(ax, 1.5)
            for t, val, up in labs:
                t.set_y(val + (gap if up else -gap))
            fig.canvas.draw()
            inv = ax.transData.inverted()
            r = fig.canvas.get_renderer()
            tops = [inv.transform((0, t.get_window_extent(r).y1))[1] for t, _, _ in labs]
            bots = [inv.transform((0, t.get_window_extent(r).y0))[1] for t, _, _ in labs]
            pad = pt_to_data(ax, 1.5)
            y0, y1 = ax.get_ylim()
            n0 = min(min(bots) - pad, lo - pt_to_data(ax, 2.0))
            n1 = max(max(tops) + pad, hi)
            if abs(n0 - y0) > 0.005 * (y1 - y0) or abs(n1 - y1) > 0.005 * (y1 - y0):
                ax.set_ylim(n0, n1)
                changed = True
        if not changed:
            break
    save(fig, "appx_component_effects")


# ======================================================== A02 (1 x 3) =====
A02_KEYS = ["bankruptcy", "rounds", "totalbet"]


def fig_complexity_average():
    fig = new_fig(FULL, 1.7)
    axes = fig.subplots(1, 3)
    for ax, d, key in zip(axes, ABP.A02, A02_KEYS):
        col = METRIC_COLORS[key]
        y = np.array(d["y"])
        xs, tl, r = trend(y)
        ax.plot(xs, tl, ls="--", lw=1.0, color=col, alpha=0.45, zorder=2)
        ax.plot(xs, y, "-o", lw=1.5, color=col, ms=3.6, mfc="white", mew=1.1, zorder=3)
        ax.set_xticks(xs)
        ax.set_ylabel(d["ylabel"].replace("\n", " "))
        ax.set_title(d["title"])
        ax.margins(x=0.06)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.28 * (hi - lo))
        rbox(ax, r)
    fig.supxlabel("Prompt complexity (# components)", fontsize=FS_LABEL)
    save(fig, "appx_complexity_average")


# ============================================================ A03 =========
def fig_escalation():
    d = json.loads((DATA / "figA03_escalation_trajectory.json").read_text())
    x = d["bin_centers"]
    fig = new_fig(HALF, 1.85)
    ax = fig.subplots()
    handles = []
    for key, color, label in (("fixed", FIXED, "Fixed"), ("variable", VARIABLE, "Variable")):
        s = d[key]
        ax.errorbar(x, s["mean"], yerr=s["sem"], color=color, marker="o", ms=3.2, mfc="white",
                    mew=1.0, lw=1.4, elinewidth=0.8, capsize=0, zorder=3)
        handles.append(Line2D([], [], color=color, marker="o", ms=3.2, mfc="white", mew=1.0,
                              lw=1.4, label=rf"{label} ($\bar{{\rho}} = {s['mean_rho']:.3f}$)"))
    ax.set_xlabel("Normalised round (0 = start, 1 = end)")
    ax.set_ylabel("Bet / balance ratio")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.085, 0.355)
    ax.set_yticks([0.10, 0.15, 0.20, 0.25, 0.30, 0.35])
    lg = top_legend(fig, handles)
    lg.set_in_layout(True)
    save(fig, "appx_escalation")


# ============================================================ A04 =========
def fig_temperature():
    d = json.loads((DATA / "figA04_temperature_robustness.json").read_text())
    temps, prompts, rates = d["temperatures"], d["prompts"], d["bankruptcy_pct"]
    fig = new_fig(HALF, 1.85)
    ax = fig.subplots()
    handles = []
    for key, color, label in (("fixed", FIXED, "Fixed"), ("variable", VARIABLE, "Variable")):
        block = np.array([rates[key][p] for p in prompts])
        mean, lo, hi = block.mean(0), block.min(0), block.max(0)
        ax.fill_between(temps, lo, hi, color=color, alpha=0.18, linewidth=0, zorder=2)
        ax.plot(temps, lo, color=color, alpha=0.6, lw=0.5, zorder=2.5)
        ax.plot(temps, hi, color=color, alpha=0.6, lw=0.5, zorder=2.5)
        ax.plot(temps, mean, color=color, lw=1.4, marker="o", ms=3.2, mfc="white", mew=1.0,
                zorder=4)
        handles.append(Line2D([], [], color=color, lw=1.4, marker="o", ms=3.2, mfc="white",
                              mew=1.0, label=label))
    handles.append(Patch(facecolor=VARIABLE, alpha=0.30, edgecolor=VARIABLE, linewidth=0.5,
                         label="min–max, 4 prompts"))
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Bankruptcy rate (%)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 102)
    ax.set_xticks(temps, [f"{t:.1f}" for t in temps])
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    top_legend(fig, handles).set_in_layout(True)
    save(fig, "appx_temperature")


# ============================================================ A05 =========
# Values printed on the current images/distortion_multimodel_summary.pdf (see docstring).
# Heatmap cells are the printed one-decimal deltas; the bars were read back from their
# rectangles (y 0..60 over 255.02 pt) and land exactly on the printed labels.
A05 = {
    "distortions": ["Pattern belief", "Loss chasing", "Probability misestimation"],
    "tick_labels": ["Pattern\nbelief", "Loss\nchasing", "Probability\nmisestimation"],
    "models": ["Claude-3.5-Haiku", "GPT-4.1-mini", "GPT-4o-mini", "Gemini-2.5-Flash",
               "Gemma-2-9B", "LLaMA-3.1-8B"],
    "delta_pp": [[2.2, 17.7, 2.4],
                 [7.9, 8.6, -0.0],
                 [3.6, 5.5, -11.9],
                 [-2.3, -3.2, -1.2],
                 [0.8, 11.2, -0.2],
                 [-6.0, 4.8, -1.9]],
    "pooled_fixed_pct": [15.9, 32.2, 6.9],
    "pooled_variable_pct": [19.9, 38.2, 5.9],
    "vlim": 20,
}


def fig_language_markers():
    d = A05
    delta = np.array(d["delta_pp"], dtype=float)
    models, ticks = d["models"], d["tick_labels"]
    vlim = d["vlim"]
    fig = new_fig(FULL, 2.2)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.045, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    axb = fig.add_subplot(gs[0, 2])

    norm = Normalize(vmin=-vlim, vmax=vlim)
    smap = ScalarMappable(norm=norm, cmap=HEATMAP_CMAP)
    for i in range(len(models)):
        for j in range(len(ticks)):
            v = delta[i, j]
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=HEATMAP_CMAP(norm(v)),
                                   edgecolor="white", linewidth=0.6, zorder=1))
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=FS_VALUE, zorder=3,
                    color="white" if abs(v) > 10.5 else "#111111")
    ax.set_xlim(-0.5, len(ticks) - 0.5)
    ax.set_ylim(len(models) - 0.5, -0.5)
    ax.set_xticks(range(len(ticks)), ticks, linespacing=1.0)
    ax.set_yticks(range(len(models)), models)
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("(a) Decision-Level Delta")

    cb = fig.colorbar(smap, cax=cax)
    # the ramp stays a raster (as submitted): as vector quads it shows white seams in print
    cb.set_label("V − F (pp)", fontsize=FS_LABEL, labelpad=2.0)
    cb.set_ticks(range(-vlim, vlim + 1, 5))
    cb.ax.tick_params(labelsize=FS_TICK, length=2.5, width=0.8, pad=2.0)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#BBBBBB")

    x = np.arange(len(ticks))
    w = 0.38
    fx, vr = d["pooled_fixed_pct"], d["pooled_variable_pct"]
    axb.bar(x - w / 2, fx, w, color=FIXED, zorder=3)
    axb.bar(x + w / 2, vr, w, color=VARIABLE, zorder=3)
    for xi, f, v in zip(x, fx, vr):
        axb.annotate(f"{f:.1f}", (xi - w / 2, f), xytext=(0, 1.5), textcoords="offset points",
                     ha="center", va="bottom", fontsize=FS_VALUE, zorder=4)
        axb.annotate(f"{v:.1f}", (xi + w / 2, v), xytext=(0, 1.5), textcoords="offset points",
                     ha="center", va="bottom", fontsize=FS_VALUE, zorder=4)
    axb.set_xticks(x, ticks, linespacing=1.0)
    axb.set_xlim(-0.55, len(ticks) - 0.45)
    axb.set_ylim(0, 45)
    axb.set_yticks([0, 10, 20, 30, 40])
    axb.set_ylabel("% of decisions with\na keyword hit", linespacing=1.0)
    axb.set_title("(b) Pooled Primary Rates")
    axb.legend(handles=[patch(FIXED, "Fixed"), patch(VARIABLE, "Variable")], loc="upper right",
               ncol=1, frameon=False, borderaxespad=0.1)
    save(fig, "appx_language_markers")


# ======================================================== A06 (3 x 4) =====
def fig_component_effects_models():
    comps = ABP.COMPONENTS
    fig = new_fig(FULL, 3.55)
    axes = fig.subplots(3, 4, sharex=True)
    x = np.arange(len(comps))
    w = 0.38
    for r in range(3):
        for c in range(4):
            ax = axes[r, c]
            d = ABP.A06[(r, c)]
            ax.bar(x - w / 2, d["F"], w, color=FIXED, zorder=3)
            ax.bar(x + w / 2, d["V"], w, color=VARIABLE, zorder=3)
            zero_line(ax)
            ax.set_xticks(x, comps)
            ax.set_xlim(-0.62, len(comps) - 0.38)
            ax.set_ylim(*ABP.A06_YLIM[r])
            ax.set_yticks(ABP.A06_YTICKS[r])
            if c:
                ax.tick_params(labelleft=False)
            if r == 0:
                ax.set_title(ABP.MODELS[c])
            if c == 0:
                ax.set_ylabel(ABP.A06_ROWS[r], linespacing=1.0)
    fig.supxlabel("Prompt component", fontsize=FS_LABEL)
    top_legend(fig, [patch(FIXED, "Fixed"), patch(VARIABLE, "Variable")])
    save(fig, "appx_component_effects_models")


# ======================================================== A07 (3 x 4) =====
def fig_complexity_models():
    fig = new_fig(FULL, 3.55)
    axes = fig.subplots(3, 4, sharex=True)
    for r, key in enumerate(A02_KEYS):
        col = METRIC_COLORS[key]
        for c in range(4):
            ax = axes[r, c]
            y = np.array(ABP.A07[(r, c)])
            xs, tl, rr = trend(y)
            ax.plot(xs, tl, ls="--", lw=0.9, color=col, alpha=0.45, zorder=2)
            ax.plot(xs, y, "-o", lw=1.3, color=col, ms=3.0, mfc="white", mew=1.0, zorder=3)
            ax.set_xticks(xs)
            ax.margins(x=0.07)
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi + 0.36 * (hi - lo))
            ax.yaxis.set_major_locator(plt.MaxNLocator(4, steps=[1, 2, 2.5, 5, 10]))
            if r == 0:
                # per-panel y tick labels leave a 72 pt axes, narrower than
                # "Claude-3.5-Haiku" at 8 pt bold: start the title over the tick labels
                ax.set_title(ABP.MODELS[c], x=-0.14)
            if c == 0:
                ax.set_ylabel(ABP.A07_ROWS[r], linespacing=1.0)
            rbox(ax, rr)
    fig.supxlabel("Prompt complexity (# components)", fontsize=FS_LABEL)
    save(fig, "appx_complexity_models")


# ======================================================== A08 (2 x 4) =====
def fig_streak_models():
    fig = new_fig(FULL, 2.65)
    axes = fig.subplots(2, 4, sharex=True)
    x = np.arange(1, 6)
    w = 0.38
    for r in range(2):
        for c in range(4):
            ax = axes[r, c]
            d = ABP.A08[(r, c)]
            ax.bar(x - w / 2, d["W"], w, color=FIXED, zorder=3)
            ax.bar(x + w / 2, d["L"], w, color=VARIABLE, zorder=3)
            ax.set_xticks(x)
            ax.set_xlim(0.4, 5.6)
            if r == 0:
                ax.set_ylim(0, 0.88)
                ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
                ax.set_title(ABP.MODELS[c])
            else:
                ax.set_ylim(0, 1.04)
                ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            if c:
                ax.tick_params(labelleft=False)
            else:
                ax.set_ylabel(ABP.A08_ROWS[r], linespacing=1.0)
    fig.supxlabel("Streak length", fontsize=FS_LABEL)
    top_legend(fig, [patch(FIXED, "Win streak"), patch(VARIABLE, "Loss streak")])
    save(fig, "appx_streak_models")


# ======================================================== A09 (2 x 4) =====
def fig_choice_distribution():
    d = json.loads((DATA / "figA09_investment_choice_distributions_cot.json").read_text())
    models, names, conds, dist = d["models"], d["model_names"], d["conditions"], d["distribution_pct"]
    fig = new_fig(FULL, 2.55)
    axes = fig.subplots(2, 4, sharex=True, sharey=True)
    x = np.arange(len(conds))
    row_labels = {"fixed": "Fixed betting\nDistribution (%)",
                  "variable": "Variable betting\nDistribution (%)"}
    for r, bet in enumerate(("fixed", "variable")):
        for c, m in enumerate(models):
            ax = axes[r, c]
            bottom = np.zeros(len(conds))
            for opt in range(4):
                vals = np.array([dist[bet][m][cond][opt] for cond in conds])
                ax.bar(x, vals, 0.65, bottom=bottom, color=CHOICE_COLORS[opt], edgecolor="white",
                       linewidth=0.4, zorder=3)
                bottom += vals
            ax.set_xticks(x, conds)
            ax.set_ylim(0, 100)
            ax.set_yticks([0, 20, 40, 60, 80, 100])
            if r == 0:
                ax.set_title(names[m])
            if c == 0:
                ax.set_ylabel(row_labels[bet], linespacing=1.0)
    fig.supxlabel("Prompt condition", fontsize=FS_LABEL)
    top_legend(fig, [Patch(facecolor=CHOICE_COLORS[i], edgecolor="none", label=f"Option {i + 1}")
                     for i in range(4)])
    save(fig, "appx_choice_distribution")


# ================================================ condition ladders ========
LADDER_COLORS = {"minusG": FIXED, "plusG": VARIABLE, "plusM": BLUE}
LADDER_NAMES = {"minusG": r"$-G$", "plusG": r"$+G^{\mathrm{twin}}$", "plusM": r"$+M^{\mathrm{twin}}$"}
CONDS = ["minusG", "plusG", "plusM"]


def draw_ladder(ax, summ, jitter, ms=3.0, elw=0.7, capsize=1.5):
    offsets = {"minusG": -jitter, "plusG": 0.0, "plusM": jitter}
    for cond in CONDS:
        s = summ[cond]
        c = LADDER_COLORS[cond]
        a = np.array(s["alphas"])
        m = np.array(s["means"])
        yerr = np.vstack([m - np.array(s["ci_lo"]), np.array(s["ci_hi"]) - m])
        ax.errorbar(a + offsets[cond], m, yerr=yerr, fmt="o", ms=ms, color=c, ecolor=c,
                    elinewidth=elw, capsize=capsize, capthick=elw, zorder=3)
        xs = np.linspace(a.min(), a.max(), 50)
        ax.plot(xs, s["intercept"] + s["slope"] * xs, "-", color=c, lw=1.0, alpha=0.8, zorder=2)


def fig_condition_ladders():
    vals = json.loads((OUT / "fig_cross_context_write_values.json").read_text())["ladders"]
    gemma, llama = vals["gemma"], vals["llama"]
    fig = new_fig(FULL, 1.8)
    axg, axl = fig.subplots(1, 2, sharey=True, width_ratios=[2.0, 1.0])
    draw_ladder(axg, gemma, jitter=0.12)
    draw_ladder(axl, llama, jitter=0.22)
    # y range that contains every point, error bar and fitted line of both models
    ys = []
    for summ in (gemma, llama):
        for s in summ.values():
            ys += s["ci_lo"] + s["ci_hi"]
            a = np.array(s["alphas"])
            ys += list(s["intercept"] + s["slope"] * np.array([a.min(), a.max()]))
    lo, hi = min(ys), max(ys)
    pad = 0.05 * (hi - lo)
    axg.set_ylim(lo - pad, hi + pad)
    axg.set_yticks([0.0, 0.1, 0.2, 0.3])
    for ax, title, xt in ((axg, "(a) Gemma-2-9B", [-3, -2, -1, 0, 1, 2, 3]),
                          (axl, "(b) LLaMA-3.1-8B", [-3, 0, 3])):
        ax.set_xticks(xt)
        ax.set_xlim(-3.5, 3.5)
        ax.set_title(title)
        ax.set_xlabel("Steering dose")
    axg.set_ylabel("Mixed bet ratio")
    handles = [Line2D([], [], color=LADDER_COLORS[c], marker="o", ms=3.0, lw=1.0,
                      label=f"{LADDER_NAMES[c]} (Gemma: {gemma[c]['slope']:+.3f} per dose)")
               for c in CONDS]
    fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False,
               fontsize=FS_LEGEND, handlelength=1.1, handletextpad=0.3, columnspacing=0.8, borderpad=0.0)
    save(fig, "appx_condition_ladders")
    return lo, hi


def main():
    print(f"writing to {OUT}")
    fig_component_effects()
    fig_complexity_average()
    fig_escalation()
    fig_temperature()
    fig_language_markers()
    fig_component_effects_models()
    fig_complexity_models()
    fig_streak_models()
    fig_choice_distribution()
    fig_condition_ladders()


if __name__ == "__main__":
    main()
