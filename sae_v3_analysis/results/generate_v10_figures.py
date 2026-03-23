#!/usr/bin/env python3
"""Generate V10 report figures from analysis JSON results."""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'figure.dpi': 150, 'font.size': 11, 'font.family': 'serif',
    'axes.grid': True, 'grid.alpha': 0.3, 'lines.linewidth': 2.0,
    'figure.constrained_layout.use': True,
})
COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9']
FIG_DIR = "figures"

# Load data
with open('json/llama_v10_symmetric_20260319_174351.json') as f:
    v10 = json.load(f)
with open('json/f1_cross_bettype_transfer_20260320_144522.json') as f:
    f1 = json.load(f)
with open('json/llama_3paradigm_20260322_132400.json') as f:
    p3 = json.load(f)

import os; os.makedirs(FIG_DIR, exist_ok=True)

# ═══════════════════════════════════════════════
# Fig 1: Cross-Model BK Classification AUC
# ═══════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# (a) AUC comparison bar chart
paradigms = ['SM', 'IC', 'MW']
gemma_auc = [0.976, 0.960, np.nan]
llama_auc = [0.974, 0.954, 0.963]

x = np.arange(len(paradigms))
w = 0.35
bars1 = axes[0].bar(x - w/2, gemma_auc, w, label='Gemma-2-9B', color=COLORS[0], alpha=0.85)
bars2 = axes[0].bar(x + w/2, llama_auc, w, label='LLaMA-3.1-8B', color=COLORS[1], alpha=0.85)
axes[0].set_ylabel('BK Classification AUC')
axes[0].set_xticks(x)
axes[0].set_xticklabels(paradigms)
axes[0].set_ylim(0.90, 1.0)
axes[0].axhline(y=0.95, color='gray', linestyle='--', alpha=0.5, label='AUC = 0.95')
axes[0].legend(loc='lower right', fontsize=9)
axes[0].set_title('(a) Cross-Model BK Prediction')
for bar, val in zip(bars1, gemma_auc):
    if not np.isnan(val):
        axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.002, f'{val:.3f}', ha='center', va='bottom', fontsize=8)
for bar, val in zip(bars2, llama_auc):
    if not np.isnan(val):
        axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.002, f'{val:.3f}', ha='center', va='bottom', fontsize=8)

# (b) Universal BK neurons
models = ['Gemma\n(3-par)', 'LLaMA\n(2-par)']
promoting = [302, 672]
inhibiting = [298, 662]
x2 = np.arange(len(models))
bars_p = axes[1].bar(x2 - w/2, promoting, w, label='BK-promoting', color=COLORS[2], alpha=0.85)
bars_i = axes[1].bar(x2 + w/2, inhibiting, w, label='BK-inhibiting', color=COLORS[3], alpha=0.85)
axes[1].set_ylabel('Number of Universal Neurons (L22)')
axes[1].set_xticks(x2)
axes[1].set_xticklabels(models)
axes[1].legend(fontsize=9)
axes[1].set_title('(b) Universal BK Neurons: Balanced Ratio')
for bar, val in zip(bars_p, promoting):
    axes[1].text(bar.get_x() + bar.get_width()/2, val + 10, str(val), ha='center', va='bottom', fontsize=9)
for bar, val in zip(bars_i, inhibiting):
    axes[1].text(bar.get_x() + bar.get_width()/2, val + 10, str(val), ha='center', va='bottom', fontsize=9)

plt.savefig(f'{FIG_DIR}/v10_fig1_cross_model_bk.png', bbox_inches='tight')
plt.close()
print("Fig 1: Cross-model BK classification + universal neurons")

# ═══════════════════════════════════════════════
# Fig 2: Cross-Domain Transfer Heatmap (Gemma + LLaMA)
# ═══════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Gemma SAE transfer
gemma_matrix = np.array([
    [np.nan, 0.913, 0.932],
    [0.646, np.nan, 0.867],
    [0.853, np.nan, np.nan],
])
paradigm_labels = ['IC', 'SM', 'MW']
im1 = axes[0].imshow(gemma_matrix, cmap='RdYlGn', vmin=0.4, vmax=1.0, aspect='auto')
for i in range(3):
    for j in range(3):
        val = gemma_matrix[i, j]
        if not np.isnan(val):
            color = 'white' if val < 0.7 else 'black'
            axes[0].text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=11, fontweight='bold', color=color)
        else:
            axes[0].text(j, i, '—', ha='center', va='center', fontsize=11, color='gray')
axes[0].set_xticks(range(3)); axes[0].set_xticklabels(paradigm_labels)
axes[0].set_yticks(range(3)); axes[0].set_yticklabels(paradigm_labels)
axes[0].set_xlabel('Test Paradigm'); axes[0].set_ylabel('Train Paradigm')
axes[0].set_title('(a) Gemma SAE Transfer (best layer)')

# LLaMA 3-paradigm transfer
llama_matrix = np.array([
    [np.nan, 0.577, 0.680],
    [0.749, np.nan, 0.561],
    [0.805, 0.682, np.nan],
])
im2 = axes[1].imshow(llama_matrix, cmap='RdYlGn', vmin=0.4, vmax=1.0, aspect='auto')
for i in range(3):
    for j in range(3):
        val = llama_matrix[i, j]
        if not np.isnan(val):
            color = 'white' if val < 0.7 else 'black'
            axes[1].text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=11, fontweight='bold', color=color)
        else:
            axes[1].text(j, i, '—', ha='center', va='center', fontsize=11, color='gray')
axes[1].set_xticks(range(3)); axes[1].set_xticklabels(paradigm_labels)
axes[1].set_yticks(range(3)); axes[1].set_yticklabels(paradigm_labels)
axes[1].set_xlabel('Test Paradigm'); axes[1].set_ylabel('Train Paradigm')
axes[1].set_title('(b) LLaMA HS Transfer (best layer)')

fig.colorbar(im2, ax=axes, label='Transfer AUC', shrink=0.8)
plt.savefig(f'{FIG_DIR}/v10_fig2_cross_domain_transfer.png', bbox_inches='tight')
plt.close()
print("Fig 2: Cross-domain transfer heatmaps")

# ═══════════════════════════════════════════════
# Fig 3: Cross-Bet-Type Transfer (F1) — KEY FIGURE
# ═══════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# LLaMA IC Hidden States
layers_l = [8, 12, 22, 25, 30]
f2v_llama = [0.872, 0.772, 0.736, 0.735, 0.819]
v2f_llama = [0.842, 0.927, 0.912, 0.884, 0.911]

axes[0].plot(layers_l, f2v_llama, 'o-', color=COLORS[0], label='Fix→Var', markersize=8)
axes[0].plot(layers_l, v2f_llama, 's-', color=COLORS[1], label='Var→Fix', markersize=8)
axes[0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Chance')
axes[0].fill_between(layers_l, 0.5, [min(a, b) for a, b in zip(f2v_llama, v2f_llama)], alpha=0.1, color='green')
axes[0].set_xlabel('Layer')
axes[0].set_ylabel('Transfer AUC')
axes[0].set_ylim(0.4, 1.0)
axes[0].set_title('(a) LLaMA IC Hidden States')
axes[0].legend(fontsize=9)
axes[0].text(20, 0.45, 'all p = 0.000', fontsize=9, fontstyle='italic', color='green')

# Gemma IC SAE
layers_g = [10, 18, 22, 26, 30]
f2v_gemma = [0.518, 0.808, 0.703, 0.716, 0.902]
v2f_gemma = [0.472, 0.726, 0.553, 0.637, 0.696]

axes[1].plot(layers_g, f2v_gemma, 'o-', color=COLORS[0], label='Fix→Var', markersize=8)
axes[1].plot(layers_g, v2f_gemma, 's-', color=COLORS[1], label='Var→Fix', markersize=8)
axes[1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Chance')
axes[1].set_xlabel('Layer')
axes[1].set_ylabel('Transfer AUC')
axes[1].set_ylim(0.4, 1.0)
axes[1].set_title('(b) Gemma IC SAE')
axes[1].legend(fontsize=9)
# Mark NS point
axes[1].annotate('NS', xy=(10, 0.518), fontsize=8, color='red', ha='center', va='bottom')

plt.savefig(f'{FIG_DIR}/v10_fig3_cross_bettype_transfer.png', bbox_inches='tight')
plt.close()
print("Fig 3: Cross-bet-type transfer (F1)")

# ═══════════════════════════════════════════════
# Fig 4: Factor Decomposition + Bet Constraint Mapping
# ═══════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# (a) Factor decomposition
models = ['Gemma\n(3-par)', 'LLaMA\n(2-par)', 'LLaMA\n(3-par)']
outcome_pct = [65.2, 69.5, 75.8]
bars = axes[0].bar(models, outcome_pct, color=[COLORS[0], COLORS[1], COLORS[1]], alpha=0.85)
axes[0].axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Permutation null (~1%)')
axes[0].set_ylabel('Outcome-Significant Features (%)')
axes[0].set_ylim(0, 85)
axes[0].set_title('(a) Factor Decomposition')
axes[0].legend(fontsize=9)
for bar, val in zip(bars, outcome_pct):
    axes[0].text(bar.get_x() + bar.get_width()/2, val + 1, f'{val}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# (b) Bet constraint linear mapping
constraints = [10, 30, 50, 70]
gemma_prob = [0.000, 0.056, 0.212, 0.270]
llama_prob = [0.057, 0.113, 0.233, 0.360]

axes[1].plot(constraints, gemma_prob, 'o-', color=COLORS[0], label=f'Gemma (r=0.979)', markersize=8)
axes[1].plot(constraints, llama_prob, 's-', color=COLORS[1], label=f'LLaMA (r=0.987)', markersize=8)
axes[1].set_xlabel('Bet Constraint ($)')
axes[1].set_ylabel('Mean BK Probability')
axes[1].set_title('(b) Bet Constraint → BK Probability')
axes[1].legend(fontsize=9)

plt.savefig(f'{FIG_DIR}/v10_fig4_factor_constraint.png', bbox_inches='tight')
plt.close()
print("Fig 4: Factor decomposition + bet constraint")

print("\nAll 4 figures generated in figures/")
