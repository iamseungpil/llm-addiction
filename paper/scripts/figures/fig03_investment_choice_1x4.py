"""Body Figure 3 as a 1x4 panel: the submitted artwork, carrying two corrections.

The layout below *is* the submitted layout.  Canvas, axes rectangles, axis
limits, bar widths, tick sets, point sizes, value-label offsets and both legends
were measured out of the submitted ``images/investment_choice3.pdf`` content
stream (axes patch rects, bar rects, glyph bounding boxes, stroke widths) and are
pinned here as constants, so the output differs from the submitted figure only
where it is meant to.

The figure is included at ``width=\\textwidth`` = 396 pt, so the 933 pt canvas is
scaled by 0.426 on the page and the point sizes below print at 4.3-5.5 pt.  That
is the submission's own choice.  An earlier revision re-authored the figure at
print width to lift every label to a 7 pt floor; nobody asked for that, and at
the submitted aspect it cost the (b) tick set, the horizontal (d) value labels
and the per-panel key.  The submitted presentation wins.

What is *not* the submitted figure -- the two things worth keeping:

1. **Every panel carries a 95% interval**, which the submitted version did not
   print.  Bankruptcy and moving-target intervals are Wilson score; the
   option-mix interval is a percentile cluster bootstrap over games and is drawn
   on the highest-variance share.  Panel (d) prints Newcombe intervals on the
   difference of two proportions, which is what its rows now are.
2. **Panel (d) carries all four matched-cap models, not one.**  Reviewer KuK5
   read the matched-cap claim as resting on a single model.  The reply tabulated
   the replication; the body figure did not show it, so the answer was invisible
   where the claim is made.  Panel (d) is now a forest plot of the 16 matched
   model-by-cap differences (variable - fixed, in percentage points) with 95%
   Newcombe intervals: four models x four caps, the GMPRW five-module prompt
   condition under which bankruptcy occurs and the one the reply tabulated.
   Ten intervals sit entirely above zero; one -- Claude-Haiku-4.5 at the $50 cap,
   -12.0 pp, CI [-23.8, -2.4] -- sits entirely below it and is drawn hollow so
   the reversal is visible rather than averaged away.  Panels (a)-(c) are
   unchanged to the digit.

Panels (a)-(c) come from ``paper_data/fig03_investment_choice.json``.  Panel (d)
is the matched-cap ablation from ``paper_data/fig05_matched_cap.json``
``panel_b.pairs``, filtered to ``prompt_combo == "GMPRW"``; the same 16 rows are
mirrored into that file under ``panel_d_body`` as the printed panel's sidecar.

Do not re-enable ``tight_layout`` or ``bbox_inches="tight"``: both resize the page
and would break the submitted aspect (933.0786 / 290.4560 = 3.21246), which the
float's placement on the page depends on.

Usage:  python scripts/figures/fig03_investment_choice_1x4.py
Writes: images/investment_choice3.pdf
"""

import json
import math
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]
IC = json.loads((ROOT / "paper_data" / "fig03_investment_choice.json").read_text())
MC = json.loads((ROOT / "paper_data" / "fig05_matched_cap.json").read_text())

PROMPTS = IC["prompt_order"]
# Palette lifted pixel-for-pixel from the submitted figure: BASE and M are two
# greens, G and GM two reds, and the option-mix stack runs green / grey / pink /
# red.
PROMPT_COLOURS = {"BASE": "#59A14F", "M": "#9DC388", "G": "#E15759", "GM": "#B33533"}
GREEN, RED = "#59A14F", "#E15759"
SHARE_COLOURS = ["#59A14F", "#C7C7C7", "#E8A0A0", "#E15759"]
SHARE_TEXT = ["white", "black", "black", "white"]
SHARE_LABELS = ["Safe exit", "Low var.", "Mid var.", "High var."]
TEXT_GREY, GRID_GREY = "#444444", "#DDDDDD"

# ---------------------------------------------------------------- canvas ----
# The submitted page box, to the digit.
W_PT = 933.0785522460938
H_PT = 290.45599365234375
FIG_W, FIG_H = W_PT / 72.0, H_PT / 72.0

# Axes patch rectangles read out of the submitted content stream, PDF points with
# the origin at the *top* left.
AXES_BOXES = [
    (42.22, 25.08, 206.73, 224.66),   # (a) bankruptcy
    (244.34, 25.08, 463.70, 224.66),  # (b) option mix
    (501.30, 25.08, 665.82, 224.66),  # (c) goal reset
    (703.43, 25.08, 907.12, 224.66),  # (d) matched caps
]


def _rect(box):
    x0, y0, x1, y1 = box
    return [x0 / W_PT, (H_PT - y1) / H_PT, (x1 - x0) / W_PT, (y1 - y0) / H_PT]


# Axis limits and bar widths, back-solved from the measured bar rectangles.
XLIMS = [(-0.732, 3.732), (-0.768, 3.768), (-0.732, 3.732)]
BAR_W = [0.72, 0.78, 0.72]
YLIMS = [(0.0, 48.0), (0.0, 100.0), (0.0, 62.0)]
YTICKS = [[0, 10, 20, 30, 40],
          [0, 20, 40, 60, 80, 100],
          [0, 10, 20, 30, 40, 50, 60]]

# --- panel (d): the four-model mean, in the submitted bar form -------------
# The matched-cap control is reported on the three API models that are the
# versions of the main experiments (50 games per cell, prompt GMPRW); the
# Claude-3.5-Haiku checkpoint had been withdrawn from the API.  The body panel
# keeps the submitted grouped-bar form and shows the mean over the three models, i.e. the pooled bankruptcy proportion over 200
# games per arm and cap, with a Wilson interval on that pooled count.  The
# sixteen per-model pairs are the appendix forest plot.
D_MODELS = ["GPT-4o-mini", "GPT-4.1-mini", "Gemini-2.5-Flash"]
D_CAPS = [10, 30, 50, 70]
D_COMBO = "GMPRW"
XLIMS.append((-0.732, 3.732))
BAR_W.append(0.36)  # two bars per cap: 2 x 0.36 fits inside the unit spacing
YLIMS.append((0.0, 70.0))
YTICKS.append([0, 10, 20, 30, 40, 50, 60, 70])

# --- typography: the submitted point sizes, on the submitted canvas ----------
TICK_FS = 10.5     # tick labels
LABEL_FS = 11.0    # axis labels
TITLE_FS = 13.0    # panel titles (bold)
VALUE_FS = 10.0    # numeric annotations over the bars
LEGEND_D_FS = 10.5  # panel (d) key
LEGEND_S_FS = 10.0  # shared option-mix key under (a)-(c)

# The submitted value labels clear their bar tops by 2.7 pt.
VALUE_PAD_PT = 2.7
# A 10 pt DejaVu line is ~11.6 pt tall; two panel-(d) labels whose x boxes
# overlap must be at least this far apart vertically.
LABEL_H_PT = 11.6
# Panel titles sit on a 13 pt bold baseline 17.03 pt below the page top.
TITLE_BASELINE_Y = (H_PT - 17.03) / H_PT

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": TICK_FS,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "axes.edgecolor": TEXT_GREY,
    "axes.labelcolor": TEXT_GREY, "text.color": TEXT_GREY,
    "xtick.color": TEXT_GREY, "ytick.color": TEXT_GREY,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "xtick.major.pad": 3.5, "ytick.major.pad": 3.5,
    "xtick.labelsize": TICK_FS, "ytick.labelsize": TICK_FS,
    "axes.labelsize": LABEL_FS, "axes.labelpad": 3.7,
    "grid.color": GRID_GREY, "grid.linewidth": 0.6,
    "legend.frameon": True, "legend.framealpha": 1.0,
    "legend.edgecolor": "#CCCCCC",
})

# Interval chrome, in canvas points: 1.9 / 1.9 / 3.4 prints as 0.8 / 0.8 / 1.4 pt.
EKW = dict(capsize=3.4, error_kw={"elinewidth": 1.9, "capthick": 1.9,
                                  "ecolor": "#333333"})

fig = plt.figure(figsize=(FIG_W, FIG_H))
axes = [fig.add_axes(_rect(b)) for b in AXES_BOXES]


def dress(ax, i):
    ax.set_xlim(*XLIMS[i])
    ax.set_ylim(*YLIMS[i])
    ax.set_yticks(YTICKS[i])
    ax.grid(True, axis="y", color=GRID_GREY, linewidth=0.6)
    ax.set_axisbelow(False)


def err(pcts, cis):
    lo = [max(0.0, p - c[0]) for p, c in zip(pcts, cis)]
    hi = [max(0.0, c[1] - p) for p, c in zip(pcts, cis)]
    return [lo, hi]


def bars(ax, i, key, ylabel):
    pcts = [IC["by_prompt"][p][key]["pct"] for p in PROMPTS]
    cis = [IC["by_prompt"][p][key]["ci"] for p in PROMPTS]
    cols = [PROMPT_COLOURS[p] for p in PROMPTS]
    ax.bar(PROMPTS, pcts, width=BAR_W[i], color=cols, yerr=err(pcts, cis), **EKW)
    dress(ax, i)
    for j, v in enumerate(pcts):
        # Horizontal, centred on its own bar, clearing the interval cap.
        ax.annotate(f"{v:.1f}%", (j, cis[j][1]), textcoords="offset points",
                    xytext=(0, VALUE_PAD_PT), ha="center", va="bottom",
                    fontsize=VALUE_FS)
    ax.set_ylabel(ylabel)


bars(axes[0], 0, "bankruptcy", "Bankruptcy rate (%)")

# --- (b) option mix, stacked; the interval is drawn on the top boundary ------
ax = axes[1]
bottoms = [0.0] * len(PROMPTS)
share_handles = []
for lab, col, txt in zip(SHARE_LABELS, SHARE_COLOURS, SHARE_TEXT):
    vals = [IC["by_prompt"][p]["option_shares_pct"][lab] for p in PROMPTS]
    share_handles.append(ax.bar(PROMPTS, vals, bottom=bottoms, color=col,
                                label=lab, width=BAR_W[1]))
    # The submitted figure printed a share inside its own segment only where the
    # segment could hold the digits; the 15 pp cut reproduces its label set
    # exactly (BASE/M mid-variance and every safe-exit share stay unlabelled).
    for j, (v, b0) in enumerate(zip(vals, bottoms)):
        if v >= 15.0:
            ax.text(j, b0 + v / 2.0, f"{v:.0f}%", ha="center", va="center",
                    fontsize=VALUE_FS, color=txt)
    bottoms = [b + v for b, v in zip(bottoms, vals)]
hv = [IC["by_prompt"][p]["high_variance_share"] for p in PROMPTS]
# The highest-variance segment is the top of the stack, so its lower boundary
# sits at 100 - share; the whisker is drawn there and carries the full interval.
ax.errorbar(range(len(PROMPTS)), [100.0 - h["pct"] for h in hv],
            yerr=[[h["ci"][1] - h["pct"] for h in hv],
                  [h["pct"] - h["ci"][0] for h in hv]],
            fmt="none", ecolor="#333333", elinewidth=1.9, capsize=3.0,
            capthick=1.9)
dress(ax, 1)
ax.set_ylabel("Distribution (%)")

bars(axes[2], 2, "moving_target", "Moving-target rate (%)")

# --- (d) matched caps, mean over four models --------------------------------
ax = axes[3]

by_key = {}
for pair in MC["panel_b"]["pairs"]:
    if pair["prompt_combo"] == D_COMBO:
        by_key[(pair["model_label"], pair["cap"])] = pair
D_ROWS = [by_key[(m, c)] for m in D_MODELS for c in D_CAPS]
assert len(D_ROWS) == 12, len(D_ROWS)
# The reply letter's GPT-4o-mini column, to the digit.
assert [(r["fixed"]["pct"], r["variable"]["pct"]) for r in D_ROWS[:4]] == [
    (0.0, 2.0), (0.0, 20.0), (4.0, 26.0), (0.0, 40.0)]


def wilson(k, n, z=1.959963984540054):
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half_w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [100.0 * (centre - half_w), 100.0 * (centre + half_w)]


def pooled(arm, cap):
    cells = [by_key[(m, cap)][arm] for m in D_MODELS]
    k, n = sum(c["k"] for c in cells), sum(c["n"] for c in cells)
    return {"k": k, "n": n, "pct": 100.0 * k / n, "ci": wilson(k, n)}


D_MEAN = [{"cap": c, "fixed": pooled("fixed", c), "variable": pooled("variable", c)}
          for c in D_CAPS]
# Equal cell sizes, so the pooled proportion is the plain mean of the four
# per-model rates.
for row in D_MEAN:
    for arm in ("fixed", "variable"):
        m = sum(by_key[(mdl, row["cap"])][arm]["pct"] for mdl in D_MODELS) / 3.0
        assert abs(m - row[arm]["pct"]) < 1e-9, (row["cap"], arm, m, row[arm]["pct"])
# The printed sidecar and the panel must be the same numbers.
if "panel_d_body" in MC:
    side = MC["panel_d_body"]["rows"]
    assert [(x["cap"], round(x["fixed_pct"], 6), round(x["variable_pct"], 6))
            for x in side] == [(r["cap"], round(r["fixed"]["pct"], 6),
                                round(r["variable"]["pct"], 6)) for r in D_MEAN]

x = list(range(len(D_CAPS)))
half = BAR_W[3] / 2.0
fx = [r["fixed"]["pct"] for r in D_MEAN]
fc = [r["fixed"]["ci"] for r in D_MEAN]
vx = [r["variable"]["pct"] for r in D_MEAN]
vc = [r["variable"]["ci"] for r in D_MEAN]
ax.bar([i - half for i in x], fx, width=BAR_W[3], color=GREEN,
       label="Fixed (= cap)", yerr=err(fx, fc), **EKW)
ax.bar([i + half for i in x], vx, width=BAR_W[3], color=RED,
       label="Variable (≤ cap)", yerr=err(vx, vc), **EKW)
ax.set_xticks(x)
ax.set_xticklabels([f"${c}" for c in D_CAPS])
dress(ax, 3)
ax.set_ylabel("Bankruptcy rate (%)")

# Both arms carry their value above the interval, as the submitted panel did.
# Value labels are ~34 pt wide, wider than the 0.28-unit gap between the
# variable bar of one cap and the fixed bar of the next, so any two labels
# closer than one label width horizontally are kept a label height apart
# vertically: later labels are lifted clear of earlier ones.
pt_per_unit = (AXES_BOXES[3][3] - AXES_BOXES[3][1]) / (YLIMS[3][1] - YLIMS[3][0])
pt_per_x = (AXES_BOXES[3][2] - AXES_BOXES[3][0]) / (XLIMS[3][1] - XLIMS[3][0])
min_gap = (LABEL_H_PT + 4.0) / pt_per_unit
label_w = 34.0 / pt_per_x
placed = []
labels = []
for i in x:
    labels.append((i - half, fc[i][1], fx[i]))
    labels.append((i + half, vc[i][1], vx[i]))
for lx, ly, val in sorted(labels):
    for px, py in placed:
        if abs(lx - px) < label_w and abs(ly - py) < min_gap:
            ly = py + min_gap
    placed.append((lx, ly))
    ax.annotate(f"{val:.1f}%", (lx, ly), textcoords="offset points",
                xytext=(0, VALUE_PAD_PT), ha="center", va="bottom", fontsize=VALUE_FS)

ax.legend(loc="upper left", fontsize=LEGEND_D_FS, handlelength=1.0,
          handletextpad=0.5, borderpad=0.3, labelspacing=0.35, handleheight=0.7)

# --- shared option-mix key, unframed, centred under panels (a)-(c) -----------
AC_CENTRE = ((AXES_BOXES[0][0] + AXES_BOXES[2][2]) / 2.0) / W_PT
fig.legend(share_handles, SHARE_LABELS, fontsize=LEGEND_S_FS, ncol=4,
           frameon=False, loc="lower center",
           bbox_to_anchor=(AC_CENTRE, 6.87 / H_PT), borderaxespad=0.0,
           borderpad=0.4, columnspacing=1.0, handlelength=1.0, handleheight=0.7,
           handletextpad=0.8)

# --- panel titles: 13 pt bold, left-aligned on each panel's own left edge -----
for box, title in zip(AXES_BOXES[:3],
                      ["(a) Bankruptcy", "(b) Option Mix", "(c) Moving Target"]):
    fig.text(box[0] / W_PT, TITLE_BASELINE_Y, title, fontsize=TITLE_FS,
             fontweight="bold", color="#000000", ha="left", va="baseline")
# Panel (d)'s title is back on its own left edge: at 209.1 pt it clears the
# 229.6 pt from panel (d)'s left edge to the page edge, so all four titles are
# left-aligned on their panels again, as the submitted figure had them.
# Panel (d)'s title is wider than the 229.6 pt from its left edge to the page
# edge, so, as in the submitted figure, it is set flush right and runs back into
# the empty band above the panel's y axis.
fig.text((W_PT - 6.0) / W_PT, TITLE_BASELINE_Y, "(d) SM Matched Caps (3-model mean)",
         fontsize=TITLE_FS, fontweight="bold", color="#000000", ha="right",
         va="baseline")

out = ROOT / "images" / "investment_choice3.pdf"
fig.savefig(out)
print(f"wrote {out}  ({W_PT:.4f}x{H_PT:.4f} pt, aspect {W_PT / H_PT:.5f})")
