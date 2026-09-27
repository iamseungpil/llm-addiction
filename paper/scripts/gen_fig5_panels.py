"""Generate two single-panel figures for §4.1 and §4.3.

§4.1 (fig5a): I_LC R^2 across 6 (model, task) cells, with control band.
§4.3 (fig5c): I_BA condition modulation R^2 (Gemma + LLaMA).

Numbers are pinned from neurips_content/4.neural.tex Table 1 / Table 3
(sources: paper_neural_audit.json + v17_nonlinear_deconfound.txt).
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parents[1] / "images"
OUT.mkdir(parents=True, exist_ok=True)


def _common_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.0,
        "axes.labelsize": 10.0,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
    })


def fig5a_loss_chasing_r2():
    """Two-panel: I_LC and I_BA R^2 across (model, task) cells with control band.

    Panel 1: I_LC at peak layer (Gemma L18, LLaMA L22) — readable in all 6 cells
    Panel 2: I_BA at peak layer (Gemma L24, LLaMA L16) — readable in SM and MW only
    I_EC SM-only values noted in caption.
    """
    _common_style()

    # I_LC values (rq1_ilc.* in paper_neural_audit.json)
    # Strict within-fold RF deconfound (run_perm_null_ilc.py); see appendix
    # tab:appendix-sweep-verification for the leaky-pipeline comparison.
    cells_lc = [
        ("Gemma\nSM", 0.237, "L18"),
        ("Gemma\nIC", 0.243, "L18"),
        ("Gemma\nMW", 0.108, "L18"),
        ("LLaMA\nSM", 0.323, "L22"),
        ("LLaMA\nIC", 0.157, "L22"),
        ("LLaMA\nMW", 0.293, "L22"),
    ]
    # I_BA values (rq1_direct.*_i_ba in paper_neural_audit.json) — IC = n/a (label var collapsed)
    cells_ba = [
        ("Gemma\nSM", 0.161, "L24"),
        ("Gemma\nIC", None,  "L24"),   # n/a
        ("Gemma\nMW", 0.056, "L24"),
        ("LLaMA\nSM", 0.121, "L16"),
        ("LLaMA\nIC", None,  "L16"),   # n/a
        ("LLaMA\nMW", 0.068, "L16"),
    ]
    colors = ["#3b6db5"] * 3 + ["#c44e52"] * 3

    fig, axes = plt.subplots(1, 2, figsize=(8.7, 3.5), sharey=False)

    for ax, cells, ylabel, ymax in [
        (axes[0], cells_lc, r"$I_\mathrm{LC}$ readout $R^2$", 0.85),
        (axes[1], cells_ba, r"$I_\mathrm{BA}$ readout $R^2$", 0.20),
    ]:
        labels = [c[0] for c in cells]
        vals = [c[1] for c in cells]
        layer_tags = [c[2] for c in cells]
        x = np.arange(len(cells))
        plot_vals = [v if v is not None else 0.0 for v in vals]
        bars = ax.bar(x, plot_vals, color=colors, width=0.7,
                      edgecolor="white", linewidth=0.6)
        ax.axhspan(-0.01, 0.025, color="#bdbdbd", alpha=0.35, zorder=0,
                   label="control band")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.04, ymax)
        ax.axhline(0, color="0.6", linewidth=0.6)
        for xi, b, tag, v in zip(x, bars, layer_tags, vals):
            if v is None:
                ax.text(xi, 0.005, "n/a", ha="center", va="bottom",
                        fontsize=8.5, color="0.4", style="italic")
            else:
                ax.text(xi, b.get_height() + ymax * 0.025, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=9.0)
            ax.text(xi, -0.035 if ymax < 0.3 else -0.07,
                    tag, ha="center", va="top", fontsize=8.0, color="0.4")

    axes[0].legend(loc="upper left", frameon=False, fontsize=8.5)
    fig.tight_layout(pad=0.5)
    out = OUT / "fig5a_loss_chasing_r2.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def fig5c_condition_modulation():
    """Grouped bar: I_BA condition modulation R^2 across conditions, two models."""
    _common_style()

    conditions = [r"$-G$", r"$+G$", r"$-M$", r"$+M$", "Fixed"]
    gemma = [0.078, 0.152, 0.127, 0.149, -0.016]
    llama = [0.087, 0.122, 0.073, 0.099, 0.002]

    x = np.arange(len(conditions))
    width = 0.34

    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    b1 = ax.bar(x - width / 2, gemma, width=width, color="#3b6db5",
                edgecolor="white", linewidth=0.6, label="Gemma-2-9B-IT (L24)")
    b2 = ax.bar(x + width / 2, llama, width=width, color="#c44e52",
                edgecolor="white", linewidth=0.6, label="LLaMA-3.1-8B (L16)")

    # Highlight the matched-cap +G vs -G comparison band
    ax.axvspan(-0.5, 1.5, color="#fff7d6", alpha=0.45, zorder=0,
               label="variance-matched: $\\pm G$")

    ax.axhline(0, color="0.6", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylabel(r"$I_\mathrm{BA}$ readout $R^2$")
    ax.set_ylim(-0.05, 0.20)

    for bars, vals in [(b1, gemma), (b2, llama)]:
        for bar, v in zip(bars, vals):
            offset = 0.005 if v >= 0 else -0.012
            ax.text(bar.get_x() + bar.get_width() / 2, v + offset,
                    f"{v:+.2f}" if v < 0 else f"{v:.2f}",
                    ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=8.5)

    ax.legend(loc="upper right", frameon=False, ncol=1)
    fig.tight_layout(pad=0.4)
    out = OUT / "fig5c_condition_modulation.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    fig5a_loss_chasing_r2()
    fig5c_condition_modulation()
