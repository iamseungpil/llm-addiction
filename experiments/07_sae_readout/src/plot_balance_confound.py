"""
Balance Confound Figure (v2) - 색상 일관성 개선
빨강/초록 기반 컬러 팔레트로 통일 + 메시지 강화
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_PATH = "/home/v-seungplee/LLM_Addiction_NMT_KOR/images/balance_confound_analysis.pdf"

# Data
labels = [
    "Baseline\n(L22)",
    "Balance\nonly",
    "$95-105\nmatched",
    "$85-95\nmatched",
    "$70-90\nmatched",
    "Variable\nonly",
    "Residual",
]
aucs =  [0.971, 0.895, 0.994, 0.990, 0.960, 0.887, 0.937]
stds =  [0.008, 0.013, 0.010, 0.011, 0.039, 0.024, 0.011]

# Color scheme: green-based for controls, red accent for confound ceiling
colors = [
    "#4a7c59",  # baseline - dark green
    "#d9534f",  # balance only - red (confound)
    "#2ecc71",  # $95-105 - bright green (best)
    "#27ae60",  # $85-95 - medium green
    "#1e8449",  # $70-90 - darker green
    "#7dcea0",  # variable only - light green
    "#a9dfbf",  # residual - very light green
]
hatches = ["", "//", "", "", "", "", ""]

fig, ax = plt.subplots(figsize=(8, 4.2))

bars = ax.bar(range(len(labels)), aucs, yerr=stds, capsize=4,
              color=colors, edgecolor="black", linewidth=0.7, alpha=0.9)

# Apply hatch to balance-only
bars[1].set_hatch("//")

# Annotate key finding
ax.annotate("", xy=(0, 0.978), xytext=(2, 0.978),
            arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=2))
ax.text(1, 0.982, "Balance controlled\n→ AUC increases", ha="center", fontsize=8.5,
        color="#c0392b", fontweight="bold")

# Value labels
for i, (a, s) in enumerate(zip(aucs, stds)):
    ax.text(i, a + s + 0.007, f"{a:.3f}", ha="center", fontsize=8.5, fontweight="bold")

# Reference line
ax.axhline(y=0.895, color="#d9534f", linestyle=":", alpha=0.5, linewidth=1,
           label="Balance-only ceiling (0.895)")

ax.set_ylabel("AUC (5-fold CV)", fontsize=11)
ax.set_title("BK Classification: Beyond Balance Information", fontsize=12, fontweight="bold")
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylim(0.82, 1.02)
ax.legend(loc="lower left", fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
print(f"Saved to {OUT_PATH}")
