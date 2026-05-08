"""
Temperature Robustness Figure (v3 - simple, reliable)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path

from paper_figure_style import COLORS, panel_title, save_pdf_png, style_axes, use_paper_style

OUT_DIR = Path("/home/v-seungplee/LLM_Addiction_NMT_KOR/images")

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
use_paper_style(9.0)
fig, ax = plt.subplots(figsize=(6.9, 2.95))

x = np.arange(len(prompts))
n_temps = len(temps)
total_width = 0.7
bar_w = total_width / (n_temps * 2)  # 2 bars (fixed/var) per temp

colors_f = ["#7BC77A", "#59A14F", "#3D7D3B", "#255628"]
colors_v = ["#F28B82", COLORS["variable"], "#C94A48", "#8B2C2A"]

for i, temp in enumerate(temps):
    fixed_vals = [data.get((temp, p, "fixed"), np.nan) for p in prompts]
    var_vals = [data.get((temp, p, "variable"), np.nan) for p in prompts]

    offset = (i - (n_temps - 1) / 2) * bar_w * 2.2

    # Fixed bars
    f_bars = ax.bar(x + offset - bar_w/2, fixed_vals, bar_w,
                    color=colors_f[i], alpha=0.92, edgecolor="white", linewidth=0.5)
    # Variable bars
    v_bars = ax.bar(x + offset + bar_w/2, var_vals, bar_w,
                    color=colors_v[i], alpha=0.92, edgecolor="white", linewidth=0.5)

    # Value labels on variable bars only (fixed is always 20)
    for j, v in enumerate(var_vals):
        if not np.isnan(v):
            ax.text(x[j] + offset + bar_w/2, v + 1.5, f"{v:.0f}%",
                    ha="center", fontsize=6.8, fontweight="bold", color=colors_v[i])

# Fixed baseline annotation
ax.axhline(y=20, color=COLORS["fixed"], linewidth=1.2, linestyle="--", alpha=0.75)
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
ax.legend(handles=legend_elements, fontsize=6.8, ncol=2, loc="upper left",
          framealpha=0.9)

ax.set_xticks(x)
ax.set_xticklabels(prompts)
ax.set_ylabel("Bankruptcy rate (%)")
ax.set_xlabel("Prompt condition")
panel_title(ax, "", "Temperature Robustness")
ax.set_ylim(0, 105)
style_axes(ax)

save_pdf_png(fig, OUT_DIR, "temperature_robustness")
print(f"Saved to {OUT_DIR / 'temperature_robustness.pdf'}")
