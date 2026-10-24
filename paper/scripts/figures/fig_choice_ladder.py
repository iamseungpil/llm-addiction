#!/usr/bin/env python3
"""Body figure for Finding 5: the choice ladder on LLaMA-3.1-8B and Gemma-2-9B.

Drawn on Figure 3's canvas and typography (``fig03_investment_choice_1x4.py``): a 933 x 290 pt
page included at \\textwidth, DejaVu Sans, the same point sizes, colours, value labels, Wilson
intervals and boxed key, so the two figures print with identical type and line weights.

Data: the E8 rollouts on the HF dataset,
``rebuttal_neurips_2026/policy_choice_ladder_e8/e8_{llama,gemma}_*.json`` (BASE prompt,
100 games per arm, 200 for the one-time choice arm). ``--rebuild`` downloads them and
recomputes ``data/fig_choice_ladder.json``; without it the script draws from that sidecar.
Bankruptcy = share of included games ending bankrupt; participation = share of games with at
least one executed wager. Two Gemma one-time-choice games whose stake could not be parsed are
dropped (the files' own ``denominators``).

    python3 scripts/figures/fig_choice_ladder.py [--rebuild]
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "images" / "fig_choice_ladder.pdf"
SIDE = HERE / "data" / "fig_choice_ladder.json"
ARMS = ["forced_fixed_cap70", "choose_fixed", "variable_cap70", "variable_open"]
LABELS = ["env. sets\n$70", "names it\nonce", "revises each\nround, $70", "revises each\nround, $100"]

# Figure 3's constants (fig03_investment_choice_1x4.py).
GREEN, RED = "#59A14F", "#E15759"
NEUTRAL = "#C7C7C7"   # Figure 3's low-variance grey
TEXT_GREY, GRID_GREY = "#444444", "#DDDDDD"
W_PT, H_PT = 933.0785522460938, 312.0
TICK_FS, LABEL_FS, TITLE_FS, VALUE_FS, LEGEND_FS = 14.0, 14.0, 16.0, 13.5, 14.0
VALUE_PAD_PT = 3.0
EKW = dict(capsize=3.4, error_kw={"elinewidth": 1.9, "capthick": 1.9, "ecolor": "#333333"})

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
    "legend.frameon": True, "legend.framealpha": 1.0, "legend.edgecolor": "#CCCCCC",
})


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0.0, c - h), 100 * min(1.0, c + h))


def rebuild() -> dict:
    from huggingface_hub import hf_hub_download
    pre = "rebuttal_neurips_2026/policy_choice_ladder_e8/"
    names = {
        "llama": ["e8_llama_forced_fixed_cap70_20260730_101408.json", "e8_llama_choose_fixed_choose_fixed_20260730_115040.json",
                  "e8_llama_variable_cap70_variable_cap70_20260730_151025.json", "e8_llama_variable_open_variable_open_20260730_181252.json"],
        "gemma": ["e8_gemma_forced_fixed_cap70_20260729_171958.json", "e8_gemma_choose_fixed_choose_fixed_20260729_185737.json",
                  "e8_gemma_variable_cap70_variable_cap70_20260729_191649.json", "e8_gemma_variable_open_variable_open_20260729_193705.json"],
    }
    out = {}
    for model, files in names.items():
        rows = []
        for arm, fn in zip(ARMS, files):
            p = hf_hub_download("llm-addiction-research/llm-addiction", pre + fn, repo_type="dataset",
                                token=os.environ.get("HF_TOKEN"))
            res = [g for g in json.load(open(p))["results"] if "bankrupt" in g]
            rows.append({"arm": arm, "n": len(res),
                         "bankrupt": sum(g["bankrupt"] for g in res),
                         "played": sum(g["total_rounds"] > 0 for g in res),
                         "rounds": sum(g["total_rounds"] for g in res) / len(res)})
        out[model] = rows
    SIDE.write_text(json.dumps(out, indent=1))
    return out


def bars(ax, x, rows, colours):
    pct = [100 * r["bankrupt"] / r["n"] for r in rows]
    ci = [wilson(r["bankrupt"], r["n"]) for r in rows]
    err = [[max(0.0, p - lo) for p, (lo, hi) in zip(pct, ci)], [max(0.0, hi - p) for p, (lo, hi) in zip(pct, ci)]]
    ax.bar(x, pct, width=0.56, color=colours, yerr=err, zorder=3, **EKW)
    for xi, p, (lo, hi) in zip(x, pct, ci):
        ax.annotate(f"{p:.1f}%", (xi, hi), xytext=(0, VALUE_PAD_PT), textcoords="offset points",
                    ha="center", va="bottom", fontsize=VALUE_FS, zorder=5)


def dress(ax, title):
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.yaxis.grid(True, zorder=0)
    ax.set_ylabel("% of games")
    ax.set_title(title, loc="left", fontsize=TITLE_FS, fontweight="bold", color="#000000", pad=6)


def main() -> None:
    data = rebuild() if "--rebuild" in sys.argv else json.loads(SIDE.read_text())
    fig = plt.figure(figsize=(W_PT / 72, H_PT / 72))
    # (left, top, width, height) in pt from the top-left corner
    boxes = [(64, 34, 410, 150), (556, 34, 366, 150)]
    axes = [fig.add_axes([l / W_PT, 1 - (t + h) / H_PT, w / W_PT, h / H_PT]) for l, t, w, h in boxes]

    def grey(ax, x, rows):
        """Share of games in which the model placed at least one bet (drawn behind the bankruptcy bars)."""
        ax.bar(x, [100 * r["played"] / r["n"] for r in rows], width=0.84, color=NEUTRAL, zorder=2)

    # (a) the four-step ladder on LLaMA-3.1-8B (no research framing)
    ax = axes[0]
    rows = data["llama"]
    grey(ax, [0, 1, 2, 3], rows)
    bars(ax, [0, 1, 2, 3], rows, [GREEN, GREEN, RED, RED])
    ax.set_xlim(-0.6, 3.6)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["Environment\nsets $70", "Model picks\nonce,\n$70 cap",
                        "Model picks\nevery round,\n$70 cap", "Model picks\nevery round,\nup to $100"], linespacing=1.1)
    dress(ax, "(a) Who sets the bet? LLaMA-3.1-8B")
    ax.set_ylabel("% of games")

    # (b) fixed versus re-chosen bet at the $70 cap, both open-weight models, research framing (E7 cells)
    ax = axes[1]
    xs = [0, 1.15, 2.95, 4.1]
    rows_b = data["role_cap70"]["llama"] + data["role_cap70"]["gemma"]
    grey(ax, xs, rows_b)
    bars(ax, xs, rows_b, [GREEN, RED, GREEN, RED])
    ax.set_xlim(-0.7, 4.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(["Env. sets\n$70", "Model picks\nevery\nround"] * 2, linespacing=1.1, fontsize=TICK_FS - 1)
    dress(ax, "(b) With research framing, $70 cap")
    axes[1].set_ylabel("")
    for cx, name in [(0.575, "LLaMA-3.1-8B"), (3.525, "Gemma-2-9B")]:
        ax.annotate(name, (cx, 0), xycoords=("data", "axes fraction"), xytext=(0, -62), textcoords="offset points",
                    ha="center", va="top", fontsize=TICK_FS, fontweight="bold", color="#000000")

    handles = [Patch(color=GREEN, label="Bankrupt, bet fixed for the whole game"),
               Patch(color=RED, label="Bankrupt, bet re-chosen every round"),
               Patch(color=NEUTRAL, label="Placed at least one bet")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0),
               fontsize=LEGEND_FS, handlelength=1.6, columnspacing=2.2, borderpad=0.5)
    fig.savefig(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
