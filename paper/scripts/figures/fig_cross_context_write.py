#!/usr/bin/env python
r"""Generate the §4.2/§4.3 cross-context write panels + the appendix alignment bars.

Outputs (images/):
  fig_xctx_signmap.pdf   panel (a): sec4_w7 12-cell sign-transfer heatmap
  fig_xctx_ladders.pdf   panel (b): sec4_w14 condition dose ladders (Gemma + LLaMA inset)
  fig_xctx_ladders_solo.pdf  the same ladders with no panel letter, for the English
    appendix, where the ladders no longer share a float with the sign map
  fig_axis_alignment.pdf appendix: behavioural-axis vs endpoint-direction cosines (bars)

PRINT-SIZE CONTRACT
-------------------
\textwidth is 5.5in.  matplotlib writes 72 PDF points to the inch, so the text
block is 396.0bp wide -- not 397.5, which is 5.5in measured in *TeX* points and
is the wrong unit for a canvas size.  Each figure is drawn on a canvas whose
width equals the width it is actually printed at, so \includegraphics scales it
by exactly 1.0 and every font size below is the size that reaches the page.

  fig_xctx_signmap   \includegraphics[width=0.49\textwidth]  ->  194.04bp = 2.6950in
  fig_xctx_ladders   \includegraphics[width=0.49\textwidth]  ->  194.04bp = 2.6950in
  fig_axis_alignment \includegraphics[width=\paperfigmid]    ->  0.65\textwidth
                                                             =  257.40bp = 3.5750in

(\paperfigmid is defined in shared/paper_core.tex as 0.65\textwidth.)
Heights are scaled by the same 396.0/397.5 factor, so each float's printed
footprint is unchanged from the previous canvas; only the scale factor moves,
from 0.99623 to 1.00000, which lifts the 7.0pt labels off 6.97pt printed.
Nothing on these canvases is set below MIN_PT.  savefig() is called WITHOUT
bbox_inches="tight" so the PDF MediaBox equals figsize exactly.

Data sources (all pulled from the gated HF dataset, no local scratch dirs):
  Heatmap: the frozen W7 adjudication record (W7_CELLS below) - pre-registered
    sign + observed z per cell, identical to appendix tab:causal-transfer-matrix.
  Ladders: experiments/sec4_causal/checkpoints/sec4_w14/*.jsonl
  Bars:    experiments/sec4_causal/assets/gemma_*_i_ba_behavioural.npz

  Needs HF auth:  set -a; source ~/.config/secrets/tokens.env; set +a

Derived from scripts/gen_fig_cross_context_write.py (same data, same contrasts,
same panel order); canvas sizes and type sizes rebuilt for print legibility,
and the LLaMA inset moved out of the Gemma data.

PALETTE
-------
The ladders and the alignment bars used Okabe-Ito blue/orange/grey, a third
unrelated pair on top of the paper's Fixed/Variable green-red and Figure 4's
model colours.  They now share Figure 4's palette exactly: Dark2 teal #1B9E77
for the behaviour-built direction and the +G graft, purple #7570B3 for the
endpoint (BK) directions, grey #666666 for the bare -G control, and Dark2
amber #E6AB02 for the +M graft.  Legends are boxed and gridlines are the
house hairline, as in the submitted body figures.
"""
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from huggingface_hub import hf_hub_download

REPO = "llm-addiction-research/llm-addiction"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "images")

# --- print geometry -------------------------------------------------------
# 5.5in of text block in PDF points, which is what matplotlib emits: 72bp/in.
TEXTWIDTH_BP = 396.0                 # 5.5in NeurIPS text block
BP_FIX = 396.0 / 397.5               # undo the old TeX-point canvas width
W_HALF_IN = 0.49 * TEXTWIDTH_BP / 72.0   # 2.6950in = 194.04bp  (0.49\textwidth)
W_MID_IN = 0.65 * TEXTWIDTH_BP / 72.0    # 3.5750in = 257.40bp  (\paperfigmid)
H_HALF_IN = 2.62 * BP_FIX            # printed heights unchanged by the fix
H_MID_IN = 2.35 * BP_FIX
MIN_PT = 7.0                         # nothing smaller than this may be drawn

LAYERS = list(range(16, 22))
RNG = np.random.default_rng(0)
N_BOOT = 1000

# One palette for the whole causal family (fig04, fig04b, and the two panels
# below).  Green and red are the betting condition everywhere else in the paper
# and are not used here; the Okabe-Ito blue/orange these panels used to carry
# were a third, unrelated pair.  Dark2 teal is the behaviour-built direction,
# Dark2 purple its endpoint counterpart, grey the bare control and Dark2 amber
# the module graft -- the same roles those colours carry on Figure 4.
TEAL = "#1B9E77"      # COLORS["gemma"]   behaviour-built / +G
PURPLE = "#7570B3"    # COLORS["llama"]   endpoint (BK) directions
GRAY = "#666666"      # COLORS["readout"] the bare -G control
AMBER = "#E6AB02"     # COLORS["balance"] the +M graft
GRID = "#EBEBEB"


def dl(path):
    return hf_hub_download(REPO, path, repo_type="dataset",
                           token=os.environ.get("HF_TOKEN"))


def base_style():
    """Base rcParams. Every size here is a *printed* size (scale is 1.0)."""
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "axes.titleweight": "bold",
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        # The house grid: a hairline on one axis, behind the data.
        "grid.color": GRID, "grid.linewidth": 0.7,
        # Boxed legends, as every other figure in the paper sets them.
        "legend.frameon": True, "legend.framealpha": 1.0,
        "legend.edgecolor": "#CCCCCC", "legend.fancybox": False,
    })


# ---------------- panel (a): sign-transfer heatmap ----------------
# Frozen W7 adjudication record, cell -> (pre-registered sign, observed z,
# low-confidence flag). Transcribed from the pre-registered sign table and
# adjudication (appendix tab:causal-transfer-matrix; llm-addiction repo
# multilayer_causal/experiments/sec4_causal/INDEX.md, W7). The read of record
# is the dose-slope z against the target null band, NOT the raw alpha=+3 -
# sham mean contrast: the mean can flip on outliers where slope + sign test
# agree (e.g. icrc->sm, sh3c->sm are confirmed negative at z=-2.9/-3.7).
W7_CELLS = {
    ("smiba", "sm"): ("+", +6.0, False),
    ("smiba", "ic"): ("-", -2.2, False),
    ("smiba", "mw"): ("+", +3.4, True),
    ("icrc", "sm"): ("-", -2.9, False),
    ("icrc", "ic"): ("+", +3.3, False),
    ("icrc", "mw"): ("+", +0.26, False),
    ("mwrc", "sm"): ("+", +4.2, True),
    ("mwrc", "ic"): ("+", +3.0, False),
    ("mwrc", "mw"): ("+", +0.32, False),
    ("sh3c", "sm"): ("-", -3.7, False),
    ("sh3c", "ic"): ("+", +3.5, False),
    ("sh3c", "mw"): ("+", -0.34, False),
}


def fig_signmap():
    axes_order = ["smiba", "icrc", "mwrc", "sh3c"]
    axis_labels = {"smiba": "SM I_BA", "icrc": "IC RC", "mwrc": "MW RC",
                   "sh3c": "shared-3C"}
    tasks_order = ["sm", "ic", "mw"]
    task_labels = {"sm": "SM\n(bet)", "ic": "IC\n(risky)", "mw": "MW\n(spin)"}

    Z = np.array([[W7_CELLS[(ax, t)][1] for t in tasks_order]
                  for ax in axes_order])

    base_style()
    fig, axB = plt.subplots(figsize=(W_HALF_IN, H_HALF_IN))
    fig.subplots_adjust(left=0.255, right=0.78, top=0.795, bottom=0.175)
    vmax = 6.0
    # Vector cells, not an image.  ``imshow`` wrote the twelve cells as one
    # embedded raster and the colourbar as a second, so the only coloured blocks
    # on this canvas blurred under magnification while the rest of the figure is
    # pure vector.  One Rectangle per cell with the same colormap lookup
    # reproduces the submitted appearance and scales.
    norm = Normalize(vmin=-vmax, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r")
    smap = ScalarMappable(norm=norm, cmap=cmap)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            axB.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                    facecolor=cmap(norm(Z[i, j])),
                                    edgecolor="none", zorder=1))
    # imshow's own limits, set by hand now that there is no image: origin at the
    # top row, half a cell of margin on every side.
    axB.set_xlim(-0.5, Z.shape[1] - 0.5)
    axB.set_ylim(Z.shape[0] - 0.5, -0.5)
    mw_j = tasks_order.index("mw")
    for i, ax in enumerate(axes_order):
        for j, t in enumerate(tasks_order):
            pred, z, lowconf = W7_CELLS[(ax, t)]
            grey = (j == mw_j)
            if grey:
                axB.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                        facecolor="#cccccc", edgecolor="none",
                                        zorder=2))
            col = "#666666" if grey else ("w" if abs(z) >= 4.0 else "k")
            ztxt = f"{z:+.1f}" if abs(z) >= 1 else f"{z:+.2f}"
            axB.text(j, i - 0.15, ztxt, ha="center", va="center",
                     fontsize=7.5, color=col, zorder=3)
            ptxt = f"pred ${pred}$" + ("\u2020" if lowconf else "")
            axB.text(j, i + 0.24, ptxt, ha="center", va="center",
                     fontsize=7.0, color=col, zorder=3)
    axB.text(mw_j, -0.80, "ceiling\n(spin 0.82)", ha="center", va="center",
             fontsize=7.0, color="#666666", linespacing=1.05)
    axB.set_xticks(range(len(tasks_order)))
    axB.set_xticklabels([task_labels[t] for t in tasks_order], fontsize=7.0,
                        linespacing=1.05)
    axB.set_yticks(range(len(axes_order)))
    axB.set_yticklabels([axis_labels[a] for a in axes_order], fontsize=7.0)
    axB.set_xlabel("scored task", fontsize=8)
    axB.set_ylabel("steered axis", labelpad=1, fontsize=8)
    for s in axB.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(smap, ax=axB, fraction=0.055, pad=0.05)
    # Colorbar.solids is a QuadMesh matplotlib rasterises by default; the bar is
    # 8 pt wide on the page, so that raster was the second embedded image.
    cb.solids.set_rasterized(False)
    cb.set_label("observed $z$ (dose slope vs null)", fontsize=7.0, labelpad=2)
    cb.ax.tick_params(labelsize=7.0)
    cb.outline.set_linewidth(0.5)
    axB.text(-0.36, 1.045, "(a)", transform=axB.transAxes, fontweight="bold",
             fontsize=9)
    fig.savefig(f"{OUT}/fig_xctx_signmap.pdf")
    plt.close(fig)
    print("saved fig_xctx_signmap.pdf")
    return Z


# ---------------- panel (b): condition dose ladders ----------------
COLORS = {"minusG": GRAY, "plusG": TEAL, "plusM": AMBER}
CONDS = ["minusG", "plusG", "plusM"]
# alpha suffixes present on HF for each model x condition
LADDER_FILES = {
    ("gemma", "minusG"): ["a0", "am3", "ap3"],
    ("gemma", "plusG"): ["a0", "am1", "am2", "am3", "ap1", "ap2", "ap3"],
    ("gemma", "plusM"): ["a0", "am1", "am2", "am3", "ap1", "ap2", "ap3"],
    ("llama", "minusG"): ["a0", "am3", "ap3"],
    ("llama", "plusG"): ["a0", "am3", "ap3"],
    ("llama", "plusM"): ["a0", "am3", "ap3"],
}


def load_condition(model, cond):
    by_alpha = {}
    for suf in LADDER_FILES[(model, cond)]:
        path = dl("experiments/sec4_causal/checkpoints/sec4_w14/"
                  f"sec4_w14_{model}_{cond}_{suf}.jsonl")
        for line in open(path):
            d = json.loads(line)
            if not d.get("parse_ok"):
                continue
            by_alpha.setdefault(float(d["alpha"]), []).append(
                float(d["bet_ratio"]))
    return {a: np.asarray(v) for a, v in sorted(by_alpha.items())}


def boot_ci(x, n_boot=N_BOOT):
    idx = RNG.integers(0, len(x), size=(n_boot, len(x)))
    return np.percentile(x[idx].mean(axis=1), [2.5, 97.5])


def summarize(model):
    out = {}
    for cond in CONDS:
        data = load_condition(model, cond)
        alphas = np.array(sorted(data))
        means = np.array([data[a].mean() for a in alphas])
        los, his = zip(*[boot_ci(data[a]) for a in alphas])
        at = np.concatenate([np.full(len(data[a]), a) for a in alphas])
        yt = np.concatenate([data[a] for a in alphas])
        A = np.column_stack([np.ones_like(at), at])
        (b0, b1), *_ = np.linalg.lstsq(A, yt, rcond=None)
        out[cond] = {"alphas": alphas.tolist(), "means": means.tolist(),
                     "ci_lo": list(los), "ci_hi": list(his),
                     "n_per_dose": [int(len(data[a])) for a in alphas],
                     "intercept": float(b0), "slope": float(b1)}
    return out


def draw(ax, summ, labels, jitter=0.06, ms=2.8, lw_fit=0.9, capsize=1.4,
         elw=0.7, xfit=None):
    offsets = {"minusG": -jitter, "plusG": 0.0, "plusM": jitter}
    for cond in CONDS:
        s = summ[cond]
        a = np.array(s["alphas"]) + offsets[cond]
        m = np.array(s["means"])
        yerr = np.vstack([m - np.array(s["ci_lo"]), np.array(s["ci_hi"]) - m])
        c = COLORS[cond]
        ax.errorbar(a, m, yerr=yerr, fmt="o", ms=ms, color=c, ecolor=c,
                    elinewidth=elw, capsize=capsize, capthick=elw, zorder=3,
                    label=labels.get(cond))
        lo, hi = (min(s["alphas"]), max(s["alphas"])) if xfit is None else xfit
        xs = np.linspace(lo, hi, 50)
        ax.plot(xs, s["intercept"] + s["slope"] * xs, "-", color=c,
                lw=lw_fit, alpha=0.8, zorder=2)


def fig_ladders(panel_letter="(b)", outname="fig_xctx_ladders.pdf"):
    base_style()
    gemma = summarize("gemma")
    llama = summarize("llama")

    # slopes go into the legend so the plot area stays free of annotation
    labels = {c: f"${'-' if c == 'minusG' else '+'}${c[-1]} "
                 f"({gemma[c]['slope']:+.3f} per dose)" for c in CONDS}

    fig, ax = plt.subplots(figsize=(W_HALF_IN, H_HALF_IN))
    fig.subplots_adjust(left=0.175, right=0.995, top=0.90, bottom=0.135)
    draw(ax, gemma, labels)
    ax.set_xlabel("steering dose", fontsize=8)
    ax.set_ylabel("mixed bet ratio", labelpad=2, fontsize=8)
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])

    # The LLaMA inset used to sit at [0.64, 0.06, .34, .36] -- on top of the
    # Gemma plusM/minusG points and fits at alpha = +1..+3. Give it its own
    # column by extending the x limit past the data and clipping the spines
    # back to the measured dose range, so the inset never overlaps data.
    ax.set_xlim(-3.6, 7.45)
    ax.set_ylim(-0.035, 0.34)
    ax.spines["bottom"].set_bounds(-3.5, 3.5)
    ax.spines["left"].set_bounds(0.0, 0.30)
    ax.set_yticks([0.0, 0.1, 0.2, 0.3])
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.legend(loc="upper left", frameon=True, fontsize=7.0,
              edgecolor="#CCCCCC", framealpha=1.0, fancybox=False,
              handletextpad=0.35, borderaxespad=0.15, borderpad=0.4,
              labelspacing=0.30, handlelength=1.1).set_zorder(6)

    axin = ax.inset_axes([0.685, 0.055, 0.305, 0.42])
    draw(axin, llama, {}, jitter=0.12, ms=2.0, lw_fit=0.7, capsize=1.0,
         elw=0.55)
    axin.set_xticks([-3, 0, 3])
    axin.set_yticks([0.1, 0.2])
    axin.tick_params(labelsize=7.0, width=0.5, length=2, pad=1.2)
    axin.set_title("LLaMA", fontsize=7.5, pad=2.0)
    for sp in axin.spines.values():
        sp.set_linewidth(0.5)
    axin.spines["top"].set_visible(False)
    axin.spines["right"].set_visible(False)

    # The canvas geometry is identical with and without the letter, so the
    # solo file prints at the same size and the type sizes stay at scale 1.0.
    if panel_letter:
        ax.text(-0.20, 1.045, panel_letter, transform=ax.transAxes,
                fontweight="bold", fontsize=9)
    fig.savefig(f"{OUT}/{outname}")
    plt.close(fig)
    print(f"saved {outname}")
    return gemma, llama


# ---------------- appendix: axis-alignment bars ----------------
def fig_alignment_bars():
    axis_files = {
        "SM": "experiments/sec4_causal/assets/gemma_slot_machine_i_ba_behavioural.npz",
        "IC": "experiments/sec4_causal/assets/gemma_investment_choice_i_ba_behavioural.npz",
        "MW": "experiments/sec4_causal/assets/gemma_mystery_wheel_i_ba_behavioural.npz",
    }
    dirs = {k: np.load(dl(f))["directions"] for k, f in axis_files.items()}
    pairs = [("IC", "SM"), ("SM", "MW"), ("IC", "MW")]
    beh_cos = {}
    for a, b in pairs:
        cs = [float(dirs[a][l] @ dirs[b][l]
                    / (np.linalg.norm(dirs[a][l]) * np.linalg.norm(dirs[b][l])))
              for l in LAYERS]
        beh_cos[f"{a}-{b}"] = float(np.mean(cs))
    bk_cos = {"IC-SM": 0.042, "SM-MW": -0.026, "IC-MW": -0.026}

    base_style()
    fig, axA = plt.subplots(figsize=(W_MID_IN, H_MID_IN))
    fig.subplots_adjust(left=0.165, right=0.985, top=0.97, bottom=0.115)
    x = np.arange(len(pairs))
    w = 0.38
    beh = [beh_cos[f"{a}-{b}"] for a, b in pairs]
    bk = [bk_cos[f"{a}-{b}"] for a, b in pairs]
    axA.bar(x - w / 2, beh, w, color=TEAL, label="behaviour-built directions", zorder=3)
    axA.bar(x + w / 2, bk, w, color=PURPLE, label="endpoint (BK) directions",
            zorder=3)
    # Every bar carries its own value, horizontally, clear of the bar: the
    # house convention, and the only way to read the two near-zero endpoint
    # bars against a scale that has to reach 0.67.
    for xs, vals in ((x - w / 2, beh), (x + w / 2, bk)):
        for xi, v in zip(xs, vals):
            axA.annotate(f"{v:+.3f}", xy=(xi, v),
                         xytext=(0, 2.0 if v >= 0 else -2.0),
                         textcoords="offset points", ha="center",
                         va="bottom" if v >= 0 else "top",
                         fontsize=7.0, color="black", zorder=5)
    axA.axhline(0, color="k", lw=0.6, zorder=2)
    axA.set_xticks(x)
    axA.set_xticklabels([f"{a}–{b}" for a, b in pairs], fontsize=8)
    axA.set_ylabel("cosine similarity (L16–21 mean)", fontsize=8)
    # The key is one row of two, across the top: stacked in a corner it sat on
    # the 0.572 bar's own value label.  The limit is opened to 0.95 to give
    # that row a band of its own above the tallest bar and its label.
    axA.set_ylim(-0.13, 0.95)
    axA.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    axA.set_axisbelow(True)
    axA.grid(True, axis="y", color=GRID, linewidth=0.7, zorder=0)
    axA.tick_params(axis="y", labelsize=7.5)
    axA.legend(frameon=True, fontsize=7.0, loc="upper left", ncol=2,
               edgecolor="#CCCCCC", framealpha=1.0, fancybox=False,
               handlelength=1.2, borderpad=0.4, labelspacing=0.35,
               columnspacing=1.2, borderaxespad=0.2)
    fig.savefig(f"{OUT}/fig_axis_alignment.pdf")
    plt.close(fig)
    print("saved fig_axis_alignment.pdf")
    return beh_cos, bk_cos


if __name__ == "__main__":
    z = fig_signmap()
    gemma, llama = fig_ladders()
    fig_ladders(panel_letter=None, outname="fig_xctx_ladders_solo.pdf")
    beh_cos, bk_cos = fig_alignment_bars()
    vals = {
        "signmap_observed_z": z.tolist(),
        "signmap_prereg_cells": {f"{a}->{t}": {"pred": p, "z": zz,
                                               "low_conf": lc}
                                 for (a, t), (p, zz, lc) in W7_CELLS.items()},
        "ladders": {"gemma": gemma, "llama": llama},
        "alignment": {"behavioural_L16_21_mean": beh_cos,
                      "endpoint_BK_from_paper": bk_cos},
    }
    with open(f"{OUT}/fig_cross_context_write_values.json", "w") as f:
        json.dump(vals, f, indent=1)
    print("saved values json")
