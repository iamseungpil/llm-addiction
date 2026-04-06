"""
Temperature Robustness Figure (v3 - simple, reliable)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path

OUT_PATH = "/home/v-seungplee/LLM_Addiction_NMT_KOR/images/temperature_robustness.pdf"

# Parse data from log
data = {}
log_path = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis/results/temperature_control/full_run_restart.log")
for line in log_path.read_text().split("\n"):
    m = re.search(r"temp=(\d+\.\d+)\s+(\w+)\s+(fixed|variable)\s*:\s*BK=(\d+\.\d+)%", line)
    if m:
        temp, prompt, bt, bk = float(m.group(1)), m.group(2), m.group(3), float(m.group(4))
        key = (temp, prompt, bt)
        data[key] = bk

temps = sorted(set(k[0] for k in data.keys()))
prompts = ["BASE", "G", "H", "GMHW"]

print(f"Available: {len(data)} conditions, temps={temps}")

# ── Single clean figure ──
fig, ax = plt.subplots(figsize=(8, 4.5))

x = np.arange(len(prompts))
n_temps = len(temps)
total_width = 0.7
bar_w = total_width / (n_temps * 2)  # 2 bars (fixed/var) per temp

colors_f = ["#27ae60", "#1e8449", "#145a32", "#0d3b1f"]
colors_v = ["#e74c3c", "#c0392b", "#922b21", "#6c1d1d"]

for i, temp in enumerate(temps):
    fixed_vals = [data.get((temp, p, "fixed"), np.nan) for p in prompts]
    var_vals = [data.get((temp, p, "variable"), np.nan) for p in prompts]

    offset = (i - (n_temps - 1) / 2) * bar_w * 2.2

    # Fixed bars
    f_bars = ax.bar(x + offset - bar_w/2, fixed_vals, bar_w,
                    color=colors_f[i], alpha=0.8, edgecolor="black", linewidth=0.5)
    # Variable bars
    v_bars = ax.bar(x + offset + bar_w/2, var_vals, bar_w,
                    color=colors_v[i], alpha=0.85, edgecolor="black", linewidth=0.5)

    # Value labels on variable bars only (fixed is always 20)
    for j, v in enumerate(var_vals):
        if not np.isnan(v):
            ax.text(x[j] + offset + bar_w/2, v + 1.5, f"{v:.0f}%",
                    ha="center", fontsize=7, fontweight="bold", color=colors_v[i])

# Fixed baseline annotation
ax.axhline(y=20, color="#27ae60", linewidth=1.5, linestyle="--", alpha=0.6)
ax.text(3.45, 23, "Fixed ≈ 20%\n(all temps)", fontsize=8, color="#1a8a4a",
        fontstyle="italic", ha="right", va="bottom")

# Legend - manual
from matplotlib.patches import Patch
legend_elements = []
for i, temp in enumerate(temps):
    legend_elements.append(Patch(facecolor=colors_f[i], edgecolor="black", linewidth=0.5,
                                 label=f"Fixed (t={temp})"))
    legend_elements.append(Patch(facecolor=colors_v[i], edgecolor="black", linewidth=0.5,
                                 label=f"Variable (t={temp})"))
ax.legend(handles=legend_elements, fontsize=7, ncol=2, loc="upper left",
          framealpha=0.9)

ax.set_xticks(x)
ax.set_xticklabels(prompts, fontsize=11)
ax.set_ylabel("Bankruptcy Rate (%)", fontsize=11)
ax.set_xlabel("Prompt Condition", fontsize=11)
ax.set_title("Temperature Robustness: Variable > Fixed at All Temperatures",
             fontsize=12, fontweight="bold")
ax.set_ylim(0, 105)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
print(f"Saved to {OUT_PATH}")
