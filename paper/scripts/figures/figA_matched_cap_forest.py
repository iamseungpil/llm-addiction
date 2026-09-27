"""Appendix figure: the twelve matched-cap pairs as a forest plot.

Body Figure 3(d) shows the four-model mean in the submitted bar form.  This
figure keeps every model and cap: variable minus fixed bankruptcy in percentage
points with the 95% Newcombe interval each pair carries, prompt GMPRW, 50 games
per cell.  Data: ``paper_data/fig05_matched_cap.json`` -> ``panel_b`` filtered to
GMPRW (mirrored under ``panel_d_forest_appendix``).  The chrome (ink, line
weights, marker) is the one the body figures use.
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]
MC = json.loads((ROOT / "paper_data" / "fig05_matched_cap.json").read_text())

MODELS = ["GPT-4o-mini", "GPT-4.1-mini", "Gemini-2.5-Flash"]
CAPS = [10, 30, 50, 70]
COMBO = "GMPRW"
GROUP_GAP = 0.6
XLIM = (-30.0, 74.0)
XTICKS = [-20, 0, 20, 40, 60]
LABEL_X = 73.0
INK, TEXT_GREY, GRID_GREY = "#333333", "#444444", "#DDDDDD"
MS = 5.5
TICK_FS, LABEL_FS, LEGEND_FS = 10.5, 11.0, 10.5

plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})

by_key = {(p["model_label"], p["cap"]): p
          for p in MC["panel_b"]["pairs"] if p["prompt_combo"] == COMBO}
ROWS = [by_key[(m, c)] for m in MODELS for c in CAPS]
assert len(ROWS) == 12
side = MC["panel_d_forest_appendix"]["rows"]
assert [(x["model"], x["cap"], x["delta_pp"], x["ci"]) for x in side] == [
    (r["model_label"], r["cap"], r["delta_pp"], r["delta_newcombe_ci"]) for r in ROWS]

fig, ax = plt.subplots(figsize=(5.6, 2.8))
fig.subplots_adjust(left=0.10, right=0.98, top=0.97, bottom=0.20)

ys, y = [], 0.0
for _ in MODELS:
    for _ in CAPS:
        ys.append(y)
        y += 1.0
    y += GROUP_GAP
deltas = [r["delta_pp"] for r in ROWS]
cis = [r["delta_newcombe_ci"] for r in ROWS]

ax.set_xlim(*XLIM)
ax.set_ylim(max(ys) + 0.4, min(ys) - 0.4)
ax.set_xticks(XTICKS)
ax.set_yticks(ys)
ax.set_yticklabels([f"${r['cap']}" for r in ROWS], fontsize=TICK_FS)
ax.tick_params(axis="x", labelsize=TICK_FS, colors=TEXT_GREY)
ax.tick_params(axis="y", colors=TEXT_GREY)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    ax.spines[sp].set_color(TEXT_GREY)
ax.grid(True, axis="x", color=GRID_GREY, linewidth=0.6)
ax.set_axisbelow(True)
ax.axvline(0.0, color=TEXT_GREY, linewidth=0.8, zorder=2.2)

reversal = [i for i, c in enumerate(cis) if c[1] < 0.0]
assert reversal == [], reversal        # no interval lies entirely below zero
ax.errorbar(deltas, ys,
            xerr=[[d - c[0] for d, c in zip(deltas, cis)],
                  [c[1] - d for d, c in zip(deltas, cis)]],
            fmt="none", ecolor=INK, elinewidth=1.9, capsize=3.4, capthick=1.9, zorder=3)
plain = [i for i in range(len(ROWS)) if i not in reversal]
ax.plot([deltas[i] for i in plain], [ys[i] for i in plain], linestyle="none",
        marker="o", markersize=MS, color=INK, markeredgecolor=INK,
        markeredgewidth=0.8, zorder=3.5)
for gi, model in enumerate(MODELS):
    ax.text(LABEL_X, ys[gi * len(CAPS)], model, fontsize=TICK_FS, ha="right",
            va="center", color="#000000", zorder=4)
ax.set_xlabel("variable − fixed bankruptcy (pp)", fontsize=LABEL_FS, color=TEXT_GREY)
out = ROOT / "images" / "figA_matched_cap_forest.pdf"
fig.savefig(out)
print("wrote", out)
