"""
Generate Balance Confound Analysis Figure for paper.
One figure, one key idea: "Classification captures internal representations
beyond balance information."
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np

OUT_PATH = "/home/v-seungplee/LLM_Addiction_NMT_KOR/images/balance_confound_analysis.pdf"

# Data from analysis results
analyses = {
    "Baseline\n(L22 full)": {"auc": 0.971, "std": 0.008, "color": "#4C72B0"},
    "Balance\nonly": {"auc": 0.895, "std": 0.013, "color": "#DD8452", "hatch": "//"},
    "$95-105\nmatched": {"auc": 0.994, "std": 0.010, "color": "#55A868"},
    "$85-95\nmatched": {"auc": 0.990, "std": 0.011, "color": "#55A868"},
    "$70-90\nmatched": {"auc": 0.960, "std": 0.039, "color": "#55A868"},
    "Variable\nonly": {"auc": 0.887, "std": 0.024, "color": "#8172B3"},
    "Residual\n(bal removed)": {"auc": 0.937, "std": 0.011, "color": "#937860"},
}

fig, ax = plt.subplots(figsize=(8, 4.5))

names = list(analyses.keys())
aucs = [analyses[n]["auc"] for n in names]
stds = [analyses[n]["std"] for n in names]
colors = [analyses[n]["color"] for n in names]

bars = ax.bar(range(len(names)), aucs, yerr=stds, capsize=4,
              color=colors, edgecolor="black", linewidth=0.8, alpha=0.85)

# Add hatch for balance-only
bars[1].set_hatch("//")
bars[1].set_alpha(0.6)

# Annotate key comparison
ax.annotate("", xy=(0, 0.975), xytext=(2, 0.975),
            arrowprops=dict(arrowstyle="<->", color="red", lw=1.5))
ax.text(1, 0.980, "Balance matched\n→ AUC increases", ha="center", fontsize=8,
        color="red", fontweight="bold")

# Add AUC values on bars
for i, (a, s) in enumerate(zip(aucs, stds)):
    ax.text(i, a + s + 0.008, f"{a:.3f}", ha="center", fontsize=8, fontweight="bold")

# Reference lines
ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3, label="Chance (0.5)")
ax.axhline(y=0.895, color="#DD8452", linestyle=":", alpha=0.5, label="Balance-only ceiling")

ax.set_ylabel("AUC (5-fold CV)", fontsize=11)
ax.set_title("BK Classification: Beyond Balance Information", fontsize=12, fontweight="bold")
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, fontsize=8.5)
ax.set_ylim(0.8, 1.02)
ax.legend(loc="lower left", fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
print(f"Saved to {OUT_PATH}")
