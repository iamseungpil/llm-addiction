#!/usr/bin/env python3
"""
Create bankruptcy rate comparison bar chart across 6 LLMs.
Fixed (blue) vs Variable (red) betting comparison.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("/home/ubuntu/llm_addiction/rebuttal_analysis/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Data from Table 1 in section3_revised.tex
models = [
    "GPT\n4o-mini",
    "GPT\n4.1-mini",
    "Gemini\n2.5-Flash",
    "Claude\n3.5-Haiku",
    "LLaMA\n3.1-8B",
    "Gemma\n2-9B"
]

fixed_rates = [0.00, 0.00, 3.12, 0.00, 0.11, 12.81]
variable_rates = [21.31, 6.31, 48.06, 20.50, 7.14, 29.06]

# Create figure
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(14, 7))

x = np.arange(len(models))
width = 0.35

# Colors: Fixed = green, Variable = red (matching streak analysis figure)
bars_fixed = ax.bar(x - width/2, fixed_rates, width,
                    label='Fixed ($10)',
                    color='#2ecc71',  # Green
                    edgecolor='black',
                    linewidth=1.8,
                    alpha=0.85)

bars_variable = ax.bar(x + width/2, variable_rates, width,
                       label='Variable ($5-$100)',
                       color='#e74c3c',  # Red
                       edgecolor='black',
                       linewidth=1.8,
                       alpha=0.85)

# Styling
ax.set_ylabel('Bankruptcy Rate (%)', fontsize=20, fontweight='bold')
ax.set_xlabel('Model', fontsize=20, fontweight='bold')
ax.set_title('Bankruptcy Rates: Fixed vs Variable Betting\nAcross Six LLMs (Slot Machine Experiment)',
             fontsize=22, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.set_ylim(0, max(variable_rates) * 1.15)
ax.grid(axis='y', alpha=0.3, linewidth=1.2)
ax.legend(fontsize=16, loc='upper right')

# Add value labels on bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.8,
                   f'{height:.2f}%', ha='center', va='bottom',
                   fontsize=13, fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.8,
                   f'0%', ha='center', va='bottom',
                   fontsize=13, fontweight='bold')

add_labels(bars_fixed)
add_labels(bars_variable)

plt.tight_layout()

# Save files
png_path = OUTPUT_DIR / "bankruptcy_fixed_vs_variable_comparison.png"
pdf_path = OUTPUT_DIR / "bankruptcy_fixed_vs_variable_comparison.pdf"

fig.savefig(png_path, dpi=300, bbox_inches="tight")
fig.savefig(pdf_path, format='pdf', bbox_inches="tight")

print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")

plt.close()
