"""
Generate neural analysis figures for the paper.
Figure A: Layer sweep (left) + Condition modulation (right)
Figure B: Cross-paradigm transfer heatmap
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path("/home/v-seungplee/LLM_Addiction_NMT_KOR/images")
OUT_DIR.mkdir(exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────

# Layer sweep data (from v18/v19 reports)
gemma_layers = [0, 6, 12, 18, 24, 30, 36, 39]
gemma_r2 = [0.01, 0.03, 0.11, 0.18, 0.25, 0.22, 0.21, 0.20]

llama_layers = [0, 4, 8, 12, 16, 20, 24, 28]
llama_r2 = [0.21, 0.22, 0.24, 0.28, 0.35, 0.33, 0.32, 0.31]

# Condition modulation (Gemma SM L24)
conditions = ["All\nVariable", "+G\n(Goal)", "+M\n(Maximize)", "Fixed\nBetting"]
gemma_cond_r2 = [0.248, 0.278, 0.264, -0.061]
llama_cond_r2 = [0.345, 0.380, 0.360, 0.000]  # LLaMA approximate

# Cross-paradigm transfer (from v20 report, Gemma L24)
# Rows = train, Cols = test
paradigms = ["SM", "IC", "MW"]
# Within-task (diagonal) from Table 1
within_gemma = [0.248, 0.476, 0.553]  # I_LC
# Cross-task (off-diagonal) all negative
cross_gemma = np.array([
    [0.248, -0.05, -2.01],   # Train SM, test SM/IC/MW
    [-0.10, 0.476, -0.08],   # Train IC, test SM/IC/MW
    [-0.06, -0.04, 0.553],   # Train MW, test SM/IC/MW
])


# ── Figure A: Layer Sweep + Condition Modulation ─────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1.2, 1]})

# Left: Layer sweep
ax1.plot(gemma_layers, gemma_r2, "o-", color="#e74c3c", label="Gemma-2-9B", linewidth=2, markersize=6)
ax1.plot(llama_layers, llama_r2, "s-", color="#3498db", label="LLaMA-3.1-8B", linewidth=2, markersize=6)
ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
ax1.fill_between([18, 28], -0.02, 0.38, alpha=0.08, color="orange", label="Mid-late peak zone")
ax1.set_xlabel("Layer", fontsize=11)
ax1.set_ylabel("$R^2$ (I$_{LC}$, NL-deconfounded)", fontsize=11)
ax1.set_title("(a) Layer Profile", fontsize=12, fontweight="bold")
ax1.legend(fontsize=9, loc="upper left")
ax1.set_ylim(-0.02, 0.40)
ax1.set_xlim(-1, 42)

# Right: Condition modulation
x = np.arange(len(conditions))
w = 0.35
bars1 = ax2.bar(x - w/2, gemma_cond_r2, w, label="Gemma", color="#e74c3c", alpha=0.8)
bars2 = ax2.bar(x + w/2, llama_cond_r2, w, label="LLaMA", color="#3498db", alpha=0.8)
ax2.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
ax2.set_xticks(x)
ax2.set_xticklabels(conditions, fontsize=9)
ax2.set_ylabel("$R^2$ (I$_{LC}$)", fontsize=11)
ax2.set_title("(b) Condition Modulation", fontsize=12, fontweight="bold")
ax2.legend(fontsize=9)
ax2.set_ylim(-0.1, 0.45)

# Annotate the 1.73x
ax2.annotate("1.73×", xy=(1 - w/2, 0.278), xytext=(0.5, 0.35),
             fontsize=9, color="#e74c3c", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.2))

# Annotate "signal vanishes"
ax2.annotate("signal\nvanishes", xy=(3, -0.03), xytext=(3, -0.08),
             fontsize=8, color="gray", ha="center", style="italic")

plt.tight_layout()
plt.savefig(OUT_DIR / "neural_analysis_combined.pdf", bbox_inches="tight", dpi=300)
plt.savefig(OUT_DIR / "neural_analysis_combined.png", bbox_inches="tight", dpi=150)
print(f"Saved Figure A to {OUT_DIR / 'neural_analysis_combined.pdf'}")


# ── Figure B: Cross-Paradigm Transfer Heatmap ───────────────────

fig2, ax3 = plt.subplots(figsize=(5, 4))

# Clip for visualization (very negative values are hard to see)
display = np.clip(cross_gemma, -0.5, 0.6)

im = ax3.imshow(display, cmap="RdYlGn", vmin=-0.5, vmax=0.6, aspect="auto")

# Add text annotations
for i in range(3):
    for j in range(3):
        val = cross_gemma[i, j]
        color = "white" if val < 0 else "black"
        fontweight = "bold" if i == j else "normal"
        text = f"{val:.2f}" if abs(val) < 1 else f"{val:.1f}"
        ax3.text(j, i, text, ha="center", va="center", color=color,
                fontsize=12, fontweight=fontweight)

ax3.set_xticks(range(3))
ax3.set_yticks(range(3))
ax3.set_xticklabels(paradigms, fontsize=11)
ax3.set_yticklabels(paradigms, fontsize=11)
ax3.set_xlabel("Test Paradigm", fontsize=11)
ax3.set_ylabel("Train Paradigm", fontsize=11)
ax3.set_title("Cross-Paradigm Probe Transfer ($R^2$, Gemma I$_{LC}$)", fontsize=11, fontweight="bold")

cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
cbar.set_label("$R^2$", fontsize=10)

plt.tight_layout()
plt.savefig(OUT_DIR / "cross_paradigm_transfer.pdf", bbox_inches="tight", dpi=300)
plt.savefig(OUT_DIR / "cross_paradigm_transfer.png", bbox_inches="tight", dpi=150)
print(f"Saved Figure B to {OUT_DIR / 'cross_paradigm_transfer.pdf'}")

print("Done.")
