#!/usr/bin/env python
r"""Appendix figure: 3-panel LOTO PCA scatter (Gemma L22).

Outputs (images/):
  fig5b_pca_appendix.pdf

For each held-out task (SM, IC, MW):
  X-axis = projection onto LOTO rank-1 SHARED direction
           (built from the other two tasks' BK contrasts).
  Y-axis = projection onto the held-out task's OWN BK direction
           after orthogonalising vs the shared axis (residual).

Voluntary-stop is downsampled to 3x the bankruptcy count for visual balance
(class ratio is otherwise 30-50:1, washing out the BK cluster).

PRINT-SIZE CONTRACT
-------------------
\textwidth is 5.5in.  matplotlib writes 72 PDF points to the inch, so the text
block is 396.0bp wide -- not 397.5, which is 5.5in measured in *TeX* points and
is the wrong unit for a canvas size.  The appendix float includes this figure at
\paperfigfull, defined in shared/paper_core.tex as 0.78\textwidth, so the
printed width is 0.78 x 396.0 = 308.88bp = 4.2900in.  The canvas is drawn at
exactly that width, so \includegraphics scales it by 1.0 and every font size
below is the size that reaches the page.  Nothing is set below 7pt.

The height is scaled by the same 396.0/397.5 factor, so the printed footprint of
the float is unchanged from the previous canvas; only the scale factor moves,
from 0.99623 to 1.00000, which lifts the 7.0pt labels off 6.97pt printed.

savefig() is called WITHOUT bbox_inches="tight" (the previous version used it,
which is why the committed PDF was 631pt wide and printed at scale 0.49);
margins are set explicitly instead so the MediaBox equals figsize exactly.

Data source: HF dataset llm-addiction-research/llm-addiction
  sae_features_v3/{slot_machine,investment_choice,mystery_wheel}/gemma/
      hidden_states_dp.npz      (~575MB total, cached by huggingface_hub)
  Needs HF auth:  set -a; source ~/.config/secrets/tokens.env; set +a

Rebuilt from scripts/gen_fig5b_pca.py.  That script is stale -- it does not
produce the committed figure (it has no panel titles, uses IC/MW/SM order, and
puts a per-panel legend with n= inside each panel).  This file reproduces the
COMMITTED composition: SM/IC/MW order, bold "(a) SM (Slot Machine)" titles, an
AUC + BK/VS count box top-right of each panel, one shared legend along the
bottom, and "LOTO shared axis" as every panel's x-label.  Verified against the
committed PDF: AUC_shared = 0.69 / 0.59 / 0.53 and BK = 87 / 172 / 54 with
VS = 261 / 516 / 162.
"""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score
from huggingface_hub import hf_hub_download

REPO = "llm-addiction-research/llm-addiction"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "images")

# --- print geometry -------------------------------------------------------
# 5.5in of text block in PDF points, which is what matplotlib emits: 72bp/in.
TEXTWIDTH_BP = 396.0                      # 5.5in NeurIPS text block
W_FULL_IN = 0.78 * TEXTWIDTH_BP / 72.0    # 4.2900in = 308.88bp  (\paperfigfull)
# The submitted artwork is 631.181 x 211.461 bp -- aspect 2.98486 -- and printed
# at 308.88 x 103.48 bp.  The previous canvas kept the old *height* and so made
# the float 38 bp taller than the submitted one; the height is now pinned to
# width / submitted aspect.  Margins below are budgeted in printed points, so the
# chrome costs what it costs and only the scatter rectangles get shorter.
SUBMITTED_ASPECT = 631.181 / 211.461      # = 2.98486
H_IN = W_FULL_IN / SUBMITTED_ASPECT       # 1.4372in = 103.48bp
_H_BP = H_IN * 72.0
_M_TOP = 21.0     # two-line bold panel titles (2 x 7.5 pt) + 3 pt pad
_M_BOTTOM = 34.0  # x tick labels + 7.5 pt x label + the shared key row

RNG = np.random.default_rng(42)
LAYER = 22
# Panel order as committed: SM, IC, MW
PANELS = ["sm", "ic", "mw"]
TASK_DIRS = {"sm": "slot_machine", "ic": "investment_choice",
             "mw": "mystery_wheel"}
# Panel tag and task name, split so the shared-axis AUC can ride on the tag
# line.  At the submitted aspect a panel is 48 bp tall; a two-line AUC/count box
# inside it would need 21 bp of that, leaving 26 bp of scatter.  The AUC moves
# up into the title, the box keeps one line, and the scatter gets 35 bp.
TITLES = {"sm": ("(a) SM", "(Slot Machine)"),
          "ic": ("(b) IC", "(Investment Choice)"),
          "mw": ("(c) MW", "(Mystery Wheel)")}
# The bankruptcy red is the paper-wide #E15759 (the submitted Tableau pair).
# This file alone carried #C44E52, a near-miss of the same red in the same role.
C_STOP, C_BK = "#3b6db5", "#E15759"


def load_hs_bk(task):
    path = hf_hub_download(
        REPO, f"sae_features_v3/{TASK_DIRS[task]}/gemma/hidden_states_dp.npz",
        repo_type="dataset", token=os.environ.get("HF_TOKEN"))
    d = np.load(path, allow_pickle=False)
    li = list(d["layers"]).index(LAYER)
    H = d["hidden_states"][:, li, :]
    out = d["game_outcomes"]
    valid = (out == "bankruptcy") | (out == "voluntary_stop")
    return H[valid], (out[valid] == "bankruptcy").astype(int)


def bk_contrast(H, bk):
    v = H[bk == 1].mean(0) - H[bk == 0].mean(0)
    return v / max(np.linalg.norm(v), 1e-12)


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.5,
        "axes.labelsize": 7.5, "axes.titlesize": 7.5,
        "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
        "legend.fontsize": 7.5, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": 0.7,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.major.size": 2.4, "ytick.major.size": 2.4,
        "xtick.major.pad": 1.8, "ytick.major.pad": 1.8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    data = {t: load_hs_bk(t) for t in PANELS}

    fig, axes = plt.subplots(1, 3, figsize=(W_FULL_IN, H_IN), sharey=False)
    # 0.135 of the width (41.7 bp) is the left gutter: "-100" tick labels plus
    # the two-line y label.  It was 0.105 when the y label was one line.
    fig.subplots_adjust(left=0.145, right=0.990,
                        top=1.0 - _M_TOP / _H_BP, bottom=_M_BOTTOM / _H_BP,
                        wspace=0.42)

    for ax, held in zip(axes, PANELS):
        others = [t for t in PANELS if t != held]
        # Shared rank-1 axis from the other two tasks' BK contrasts
        contrasts = np.stack([bk_contrast(*data[t]) for t in others], axis=0)
        _, _, Vt = np.linalg.svd(contrasts, full_matrices=False)
        shared = Vt[0]

        H_h, bk_h = data[held]
        # Held-out task's own BK direction, orthogonalised against `shared`
        own = bk_contrast(H_h, bk_h)
        own_orth = own - shared * (shared @ own)
        own_orth = own_orth / max(np.linalg.norm(own_orth), 1e-12)

        x = H_h @ shared
        y = H_h @ own_orth

        # Orient so bankruptcy mean > stop mean on each axis
        if x[bk_h == 1].mean() < x[bk_h == 0].mean():
            x = -x
        if y[bk_h == 1].mean() < y[bk_h == 0].mean():
            y = -y

        auc_shared = roc_auc_score(bk_h, x)

        # Downsample voluntary stop to 3x BK count for visual balance
        n_bk = int((bk_h == 1).sum())
        idx_st = np.where(bk_h == 0)[0]
        if len(idx_st) > 3 * n_bk:
            idx_st = RNG.choice(idx_st, size=3 * n_bk, replace=False)

        ax.axhline(0, color="0.75", linewidth=0.5, zorder=0)
        ax.axvline(0, color="0.75", linewidth=0.5, zorder=0)
        ax.scatter(x[idx_st], y[idx_st], s=3.2, c=C_STOP, alpha=0.45,
                   edgecolors="none", zorder=2)
        ax.scatter(x[bk_h == 1], y[bk_h == 1], s=6.0, c=C_BK, alpha=0.85,
                   edgecolors="white", linewidths=0.25, zorder=3)

        # Headroom so the AUC/count box never sits on top of a marker.
        y_used = np.concatenate([y[idx_st], y[bk_h == 1]])
        y_lo, y_hi = float(y_used.min()), float(y_used.max())
        span = max(y_hi - y_lo, 1e-9)
        # 13 bp of dead band at the top for the one-line count box and its
        # clearance.  No point moves -- only the view limit does.
        ax.set_ylim(y_lo - 0.06 * span, y_hi + 0.40 * span)

        # "AUC (shared) = 0.69" is 75 bp wide with its frame against a 68 bp
        # panel, so the box used to hang over the panel edge.  The AUC is now on
        # the title line -- where "(shared)" is redundant with the x-axis label
        # -- and the box keeps the two counts on one line.
        ax.text(0.96, 0.96, f"BK={n_bk}, VS={len(idx_st)}",
                transform=ax.transAxes, fontsize=7.0, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="0.7",
                          lw=0.4))
        tag, task = TITLES[held]
        ax.set_title(f"{tag}  AUC {auc_shared:.2f}\n{task}", fontsize=7.5,
                     fontweight="bold", linespacing=1.2, pad=3.0)
        ax.set_xlabel("LOTO shared axis", labelpad=1.5)
        ax.locator_params(axis="x", nbins=4)
        ax.locator_params(axis="y", nbins=5)

    # One line of "Own BK axis (residual)" is 84 bp long against a 48 bp panel,
    # so it overran its own axes at both ends.  Same words, wrapped.
    axes[0].set_ylabel("Own BK axis\n(residual)", labelpad=1.5)

    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=2.6,
               markerfacecolor=C_STOP, markeredgecolor="none", alpha=0.7,
               label="voluntary stop"),
        Line2D([], [], marker="o", linestyle="none", markersize=3.4,
               markerfacecolor=C_BK, markeredgecolor="white",
               markeredgewidth=0.3, label="bankruptcy"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=7.5, handletextpad=0.4, columnspacing=2.2,
               bbox_to_anchor=(0.5, 1.0 / _H_BP))

    out = os.path.join(OUT, "fig5b_pca_appendix.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
