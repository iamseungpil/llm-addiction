#!/usr/bin/env python
"""Generate the §4.2/§4.3 cross-context write figure panels + appendix bars.

Outputs (images/):
  fig_xctx_signmap.pdf   panel (a): sec4_w7 12-cell sign-transfer heatmap
  fig_xctx_ladders.pdf   panel (b): sec4_w14 condition dose ladders (Gemma + LLaMA inset)
  fig_axis_alignment.pdf appendix: behavioural-axis vs endpoint-direction cosines (bars)

Data sources:
  Heatmap: the frozen W7 adjudication record (W7_CELLS below) — pre-registered
    sign + observed z per cell, identical to appendix tab:causal-transfer-matrix.
  Bars: HF dataset llm-addiction-research/llm-addiction
    (gemma i_ba behavioural axis npz). Needs HF auth:
    set -a; source ~/.env; set +a
  Ladders: HF experiments/sec4_causal/checkpoints/sec4_w14/*.jsonl

Derived from the fig_sign_transfer / fig_condition_ladders generators of the
0709 session (same data, same contrasts); panels re-lettered for the body
figure fig:cross-context-write and the bars split out for the appendix.
"""
import glob
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from huggingface_hub import hf_hub_download

REPO = "llm-addiction-research/llm-addiction"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images")
# The local ladder checkpoints live in the separate analysis tree; the same
# files are on the dataset under experiments/sec4_causal/checkpoints/sec4_w14.
ANALYSIS_ROOT = os.environ.get("LLM_ADDICTION_ANALYSIS",
                               os.path.expanduser("~/llm-addiction"))
RESULTS_DIR = os.path.join(ANALYSIS_ROOT, "multilayer_causal", "results", "sec4_w14")
LAYERS = list(range(16, 22))
RNG = np.random.default_rng(0)
N_BOOT = 1000
BLUE, ORANGE, GRAY = "#0072B2", "#E69F00", "#999999"  # Okabe-Ito


def dl(path):
    return hf_hub_download(REPO, path, repo_type="dataset")


def base_style():
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
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
    axis_labels = {"smiba": "SM $I_{BA}$", "icrc": "IC RC", "mwrc": "MW RC",
                   "sh3c": "shared-3C"}
    tasks_order = ["sm", "ic", "mw"]
    task_labels = {"sm": "SM\n(bet)", "ic": "IC\n(risky)", "mw": "MW\n(spin)"}

    Z = np.array([[W7_CELLS[(ax, t)][1] for t in tasks_order]
                  for ax in axes_order])

    base_style()
    fig, axB = plt.subplots(figsize=(2.1, 2.2))
    fig.subplots_adjust(left=0.30, right=0.80, top=0.82, bottom=0.20)
    vmax = 6.0
    im = axB.imshow(Z, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
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
            axB.text(j, i - 0.12, ztxt, ha="center", va="center",
                     fontsize=6.5, color=col, zorder=3)
            ptxt = f"pred ${pred}$" + ("$^\\dagger$" if lowconf else "")
            axB.text(j, i + 0.26, ptxt, ha="center", va="center",
                     fontsize=5.2, color=col, zorder=3)
    axB.text(mw_j, -0.78, "ceiling\n(spin 0.82)", ha="center", va="center",
             fontsize=5.5, color="#666666")
    axB.set_xticks(range(len(tasks_order)))
    axB.set_xticklabels([task_labels[t] for t in tasks_order], fontsize=6)
    axB.set_yticks(range(len(axes_order)))
    axB.set_yticklabels([axis_labels[a] for a in axes_order], fontsize=6)
    axB.set_xlabel("scored task")
    axB.set_ylabel("steered axis", labelpad=1)
    for s in axB.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=axB, fraction=0.055, pad=0.05)
    cb.set_label("observed $z$ (dose slope vs null)", fontsize=6, labelpad=1)
    cb.ax.tick_params(labelsize=6)
    axB.text(-0.44, 1.06, "(a)", transform=axB.transAxes, fontweight="bold")
    fig.savefig(f"{OUT}/fig_xctx_signmap.pdf")
    plt.close(fig)
    print("saved fig_xctx_signmap.pdf")
    return Z


# ---------------- panel (b): condition dose ladders ----------------
COLORS = {"minusG": "#999999", "plusG": "#0072B2", "plusM": "#D55E00"}
CONDS = ["minusG", "plusG", "plusM"]


def load_condition(model, cond):
    by_alpha = {}
    for path in sorted(glob.glob(f"{RESULTS_DIR}/sec4_w14_{model}_{cond}_a*.jsonl")):
        for line in open(path):
            d = json.loads(line)
            if not d.get("parse_ok"):
                continue
            by_alpha.setdefault(float(d["alpha"]), []).append(float(d["bet_ratio"]))
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


def draw(ax, summ, labels, jitter=0.06, ms=2.6, lw_fit=0.8, capsize=1.3, elw=0.7):
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
        xs = np.linspace(min(s["alphas"]), max(s["alphas"]), 50)
        ax.plot(xs, s["intercept"] + s["slope"] * xs, "-", color=c,
                lw=lw_fit, alpha=0.8, zorder=2)


def fig_ladders():
    base_style()
    gemma = summarize("gemma")
    llama = summarize("llama")

    # slopes go into the legend so the right margin stays free at this width
    labels = {c: f"${'-' if c == 'minusG' else '+'}${c[-1]} "
                 f"({gemma[c]['slope']:+.3f}/$\\alpha$)" for c in CONDS}

    fig, ax = plt.subplots(figsize=(2.55, 2.2))
    fig.subplots_adjust(left=0.185, right=0.97, top=0.90, bottom=0.20)
    draw(ax, gemma, labels)
    ax.set_xlabel(r"steering dose $\alpha$")
    ax.set_ylabel("mixed bet ratio", labelpad=1)
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_xlim(-3.5, 3.5)
    ax.legend(loc="upper left", frameon=False, fontsize=5.8,
              handletextpad=0.35, borderaxespad=0.15, labelspacing=0.28)

    axin = ax.inset_axes([0.64, 0.06, 0.34, 0.36])
    draw(axin, llama, {}, jitter=0.12, ms=1.8, lw_fit=0.6, capsize=0.9, elw=0.5)
    axin.set_xticks([-3, 0, 3])
    axin.tick_params(labelsize=5.0, width=0.5, length=2, pad=1.2)
    axin.set_title("LLaMA", fontsize=5.8, pad=1.5)
    for sp in axin.spines.values():
        sp.set_linewidth(0.5)
    axin.spines["top"].set_visible(False)
    axin.spines["right"].set_visible(False)

    # -0.30 put the label outside the figure bbox (only ")" survived the crop)
    ax.text(-0.22, 1.06, "(b)", transform=ax.transAxes, fontweight="bold")
    fig.savefig(f"{OUT}/fig_xctx_ladders.pdf")
    plt.close(fig)
    print("saved fig_xctx_ladders.pdf")
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
    fig, axA = plt.subplots(figsize=(2.4, 2.1))
    fig.subplots_adjust(left=0.22, right=0.97, top=0.95, bottom=0.14)
    x = np.arange(len(pairs))
    w = 0.38
    axA.bar(x - w / 2, [beh_cos[f"{a}-{b}"] for a, b in pairs], w,
            color=BLUE, label="behavioural axes")
    axA.bar(x + w / 2, [bk_cos[f"{a}-{b}"] for a, b in pairs], w,
            color=ORANGE, label="endpoint (BK)\ndirections")
    axA.axhline(0, color="k", lw=0.6)
    axA.set_xticks(x)
    axA.set_xticklabels([f"{a}–{b}" for a, b in pairs], fontsize=6.3)
    axA.set_ylabel("cosine similarity (L16–21 mean)")
    axA.set_ylim(-0.1, 0.75)
    axA.legend(frameon=False, fontsize=6.5, loc="upper left",
               handlelength=1.2, borderpad=0.2, labelspacing=0.3)
    fig.savefig(f"{OUT}/fig_axis_alignment.pdf")
    plt.close(fig)
    print("saved fig_axis_alignment.pdf")
    return beh_cos, bk_cos


if __name__ == "__main__":
    z = fig_signmap()
    gemma, llama = fig_ladders()
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
