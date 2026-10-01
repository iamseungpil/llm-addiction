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
W_PT, H_PT = 933.0785522460938, 236.0
TICK_FS, LABEL_FS, TITLE_FS, VALUE_FS, LEGEND_FS = 10.5, 11.0, 13.0, 10.0, 10.5
VALUE_PAD_PT = 2.7
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
    play = [100 * r["played"] / r["n"] for r in rows]
    ax.bar(x, play, width=0.72, color=NEUTRAL, zorder=2)
    ax.bar(x, pct, width=0.40, color=colours, yerr=err, zorder=3, **EKW)
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
    boxes = [(62, 28, 300, 146), (420, 28, 300, 146)]   # (left, top, width, height) in pt from top-left
    axes = [fig.add_axes([l / W_PT, 1 - (t + h) / H_PT, w / W_PT, h / H_PT]) for l, t, w, h in boxes]

    # (a) the full ladder on LLaMA
    ax = axes[0]
    bars(ax, [0, 1, 2, 3], data["llama"], [GREEN, GREEN, RED, RED])
    ax.set_xlim(-0.6, 3.6)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(LABELS, linespacing=1.05)
    dress(ax, "(a) Choice Ladder (LLaMA-3.1-8B)")

    # (b) forced versus revisable at the $70 cap, both open-weight models
    ax = axes[1]
    rc = data["role_cap70"]
    xs = [0, 1, 2.5, 3.5]
    bars(ax, xs, rc["llama"] + rc["gemma"], [GREEN, RED, GREEN, RED])
    ax.set_xlim(-0.6, 4.1)
    ax.set_xticks(xs)
    ax.set_xticklabels(["env. sets\n$70", "revises each\nround, $70"] * 2, linespacing=1.05)
    for xc, name in [(0.5, "LLaMA-3.1-8B"), (3.0, "Gemma-2-9B")]:
        ax.annotate(name, (xc, 0), xycoords=("data", "axes fraction"), xytext=(0, -33),
                    textcoords="offset points", ha="center", va="top", fontsize=LABEL_FS, fontweight="bold")
    dress(ax, "(b) Fixed vs. Revisable at $70")

    handles = [Patch(color=GREEN, label="Bankrupt, stake set\nfor the game"),
               Patch(color=RED, label="Bankrupt, stake\nrevisable"),
               Patch(color=NEUTRAL, label="Played at least\none round")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(742 / W_PT, 1 - 40 / H_PT),
               fontsize=LEGEND_FS, handlelength=1.6, labelspacing=0.9, borderpad=0.6)
    fig.savefig(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
