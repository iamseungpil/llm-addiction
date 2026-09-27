"""§4.2 sharing figure: shared/residual decomposition + cross-task transfer R²."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "images"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.0,
    "axes.labelsize": 10.0, "axes.titlesize": 10.5,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "legend.fontsize": 9.0, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.8,
})

fig, axes = plt.subplots(1, 2, figsize=(8.7, 3.2))

# Panel A: Shared/Residual AUC decomposition (Gemma L22)
tasks = ["IC", "MW", "SM"]
shared = [0.74, 0.52, 0.80]
residual = [0.93, 0.95, 0.64]
combined = [0.96, 0.95, 0.97]
x = np.arange(3)
w = 0.27
ax = axes[0]
b1 = ax.bar(x - w, shared, w, color="#3b6db5", edgecolor="white", linewidth=0.6, label="Shared-only")
b2 = ax.bar(x, residual, w, color="#c44e52", edgecolor="white", linewidth=0.6, label="Residual-only")
b3 = ax.bar(x + w, combined, w, color="#7f7f7f", edgecolor="white", linewidth=0.6, label="Combined")
ax.axhline(0.5, color="0.4", linestyle="--", linewidth=0.8, alpha=0.7, zorder=0)
ax.text(2.55, 0.51, "chance", fontsize=8.5, color="0.35", ha="right")
ax.set_xticks(x); ax.set_xticklabels(tasks)
ax.set_ylabel("BK separability AUC")
ax.set_ylim(0.0, 1.05)
ax.set_title("Shared / Residual decomposition (Gemma L22)", fontsize=10.5)
ax.legend(loc="lower right", frameon=False, fontsize=8.5)
for bars, vals in [(b1, shared), (b2, residual), (b3, combined)]:
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.012, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8.5)

# Panel B: Cross-task transfer R² (Gemma L24, LLaMA L16) — I_BA pairs SM↔MW
# Source: iba_cross_task_transfer.json
ax = axes[1]
pairs = ["within-SM", "within-MW", "SM→MW", "MW→SM"]
gemma = [0.011, 0.059, -2.006, -0.060]
llama = [0.029, 0.069, -7.698, -0.077]
# clip very negative for visualization, mark with arrow
gemma_disp = [v if v > -2.5 else -2.0 for v in gemma]
llama_disp = [v if v > -2.5 else -2.0 for v in llama]
x = np.arange(4)
w = 0.36
b1 = ax.bar(x - w/2, gemma_disp, w, color="#3b6db5", edgecolor="white", linewidth=0.6, label="Gemma (L24)")
b2 = ax.bar(x + w/2, llama_disp, w, color="#c44e52", edgecolor="white", linewidth=0.6, label="LLaMA (L16)")
ax.axhline(0, color="0.5", linewidth=0.8)
ax.set_xticks(x); ax.set_xticklabels(pairs, fontsize=9.0)
ax.set_ylabel(r"$I_\mathrm{BA}$ readout $R^2$")
ax.set_ylim(-2.4, 0.4)
ax.set_title("Cross-task readout transfer", fontsize=10.5)
ax.legend(loc="lower left", frameon=False, fontsize=8.5)
# Annotate values
for i, (gv, lv) in enumerate(zip(gemma, llama)):
    if gv > 0:
        ax.text(i - w/2, gv + 0.04, f"{gv:+.2f}", ha="center", va="bottom", fontsize=8.5)
    else:
        clip = -1.85 if gv < -2 else gv
        ax.text(i - w/2, clip - 0.10, f"{gv:+.2f}", ha="center", va="top", fontsize=8.5, color="white" if gv < -1 else "black")
    if lv > 0:
        ax.text(i + w/2, lv + 0.04, f"{lv:+.2f}", ha="center", va="bottom", fontsize=8.5)
    else:
        clip = -1.85 if lv < -2 else lv
        ax.text(i + w/2, clip - 0.10, f"{lv:+.2f}", ha="center", va="top", fontsize=8.5, color="white" if lv < -1 else "black")

fig.tight_layout(pad=0.5)
out = OUT / "fig5b_sharing.pdf"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"saved {out}")
