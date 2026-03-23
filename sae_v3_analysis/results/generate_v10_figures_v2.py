#!/usr/bin/env python3
"""V10 figures v2: balanced Gemma+LLaMA, publication-quality style."""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Publication style (matching NMT paper)
plt.rcParams.update({
    'figure.dpi': 200,
    'font.size': 10,
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'axes.linewidth': 0.8,
    'axes.grid': False,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'lines.linewidth': 1.8,
    'figure.constrained_layout.use': True,
})

C_GEMMA = '#2196F3'   # blue
C_LLAMA = '#FF7043'   # orange
C_PRO = '#66BB6A'     # green (promoting)
C_INH = '#AB47BC'     # purple (inhibiting)
C_FIX2VAR = '#1976D2' # dark blue
C_VAR2FIX = '#E64A19' # dark orange

FIG = 'figures'
import os; os.makedirs(FIG, exist_ok=True)

# ═══════════════════════════════════════════════════════
# Fig 1: RQ1 — Cross-Model Comparison (3 panels)
# ═══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))

# (a) BK Classification AUC
paradigms = ['SM', 'IC', 'MW']
gemma = [0.976, 0.960, np.nan]
llama = [0.974, 0.954, 0.963]
x = np.arange(3); w = 0.32
b1 = axes[0].bar(x - w/2, gemma, w, label='Gemma', color=C_GEMMA, alpha=0.8, edgecolor='white')
b2 = axes[0].bar(x + w/2, llama, w, label='LLaMA', color=C_LLAMA, alpha=0.8, edgecolor='white')
axes[0].set_ylim(0.92, 1.0); axes[0].set_ylabel('BK Classification AUC')
axes[0].set_xticks(x); axes[0].set_xticklabels(paradigms)
axes[0].legend(fontsize=8, loc='lower right')
for b, v in zip(b1, gemma):
    if not np.isnan(v): axes[0].text(b.get_x()+b.get_width()/2, v+0.001, f'{v:.3f}', ha='center', fontsize=7, va='bottom')
for b, v in zip(b2, llama):
    if not np.isnan(v): axes[0].text(b.get_x()+b.get_width()/2, v+0.001, f'{v:.3f}', ha='center', fontsize=7, va='bottom')
axes[0].axhline(0.95, color='gray', ls='--', lw=0.7, alpha=0.5)
axes[0].set_title('(a) BK Prediction AUC', fontsize=10, fontweight='bold')

# (b) Universal Neurons — balanced
models = ['Gemma\n(3-par)', 'LLaMA\n(2-par)']
pro = [302, 672]; inh = [298, 662]
x2 = np.arange(2)
axes[1].bar(x2-w/2, pro, w, label='Promoting', color=C_PRO, alpha=0.8, edgecolor='white')
axes[1].bar(x2+w/2, inh, w, label='Inhibiting', color=C_INH, alpha=0.8, edgecolor='white')
axes[1].set_ylabel('Universal Neurons (L22)')
axes[1].set_xticks(x2); axes[1].set_xticklabels(models)
axes[1].legend(fontsize=8)
for i, (p, n) in enumerate(zip(pro, inh)):
    axes[1].text(i-w/2, p+15, str(p), ha='center', fontsize=8)
    axes[1].text(i+w/2, n+15, str(n), ha='center', fontsize=8)
axes[1].set_title('(b) Balanced Pro/Inh Ratio', fontsize=10, fontweight='bold')

# (c) Factor Decomposition
labels_fd = ['Gemma\n3-par', 'LLaMA\n2-par', 'LLaMA\n3-par']
vals = [65.2, 69.5, 75.8]
colors = [C_GEMMA, C_LLAMA, C_LLAMA]
alphas = [0.8, 0.5, 0.8]
for i, (v, c, a) in enumerate(zip(vals, colors, alphas)):
    axes[2].bar(i, v, color=c, alpha=a, edgecolor='white', width=0.6)
    axes[2].text(i, v+1.5, f'{v}%', ha='center', fontsize=9, fontweight='bold')
axes[2].axhline(1.0, color='red', ls='--', lw=0.7, alpha=0.6)
axes[2].text(2.3, 2.5, 'null\n~1%', fontsize=7, color='red', ha='center')
axes[2].set_ylabel('Outcome-Significant (%)')
axes[2].set_xticks(range(3)); axes[2].set_xticklabels(labels_fd)
axes[2].set_ylim(0, 85)
axes[2].set_title('(c) Factor Decomposition', fontsize=10, fontweight='bold')

plt.savefig(f'{FIG}/v10_fig1_rq1_overview.png', bbox_inches='tight')
plt.close()
print("Fig 1: RQ1 overview (3 panels)")

# ═══════════════════════════════════════════════════════
# Fig 2: RQ2 — Cross-Domain Transfer (2 heatmaps)
# ═══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

labels_p = ['IC', 'SM', 'MW']
gemma_m = np.array([[np.nan, 0.913, 0.932], [0.646, np.nan, 0.867], [0.853, np.nan, np.nan]])
llama_m = np.array([[np.nan, 0.577, 0.680], [0.749, np.nan, 0.561], [0.805, 0.682, np.nan]])

for ax, mat, title in [(axes[0], gemma_m, '(a) Gemma SAE'), (axes[1], llama_m, '(b) LLaMA Hidden States')]:
    im = ax.imshow(mat, cmap='RdYlGn', vmin=0.4, vmax=1.0, aspect='equal')
    for i in range(3):
        for j in range(3):
            v = mat[i,j]
            if np.isnan(v):
                ax.text(j, i, '—', ha='center', va='center', fontsize=10, color='gray')
            else:
                c = 'white' if v < 0.65 else 'black'
                ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=10, fontweight='bold', color=c)
    ax.set_xticks(range(3)); ax.set_xticklabels(labels_p)
    ax.set_yticks(range(3)); ax.set_yticklabels(labels_p)
    ax.set_xlabel('Test'); ax.set_ylabel('Train')
    ax.set_title(title, fontsize=10, fontweight='bold')

fig.colorbar(im, ax=axes, label='Transfer AUC', shrink=0.85, pad=0.02)
plt.savefig(f'{FIG}/v10_fig2_rq2_transfer.png', bbox_inches='tight')
plt.close()
print("Fig 2: RQ2 transfer heatmaps")

# ═══════════════════════════════════════════════════════
# Fig 3: RQ3 — Cross-Bet-Type Transfer F1 (BALANCED: both models)
# ═══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# (a) LLaMA HS
layers_l = [8, 12, 22, 25, 30]
f2v_l = [0.872, 0.772, 0.736, 0.735, 0.819]
v2f_l = [0.842, 0.927, 0.912, 0.884, 0.911]
axes[0].plot(layers_l, f2v_l, 'o-', color=C_FIX2VAR, label='Fix→Var', ms=7, zorder=3)
axes[0].plot(layers_l, v2f_l, 's-', color=C_VAR2FIX, label='Var→Fix', ms=7, zorder=3)
axes[0].axhline(0.5, color='red', ls='--', lw=0.7, alpha=0.5)
axes[0].fill_between(layers_l, 0.5, [min(a,b) for a,b in zip(f2v_l, v2f_l)], alpha=0.08, color='green')
axes[0].set_ylim(0.4, 1.0); axes[0].set_xlabel('Layer'); axes[0].set_ylabel('Transfer AUC')
axes[0].legend(fontsize=8); axes[0].set_title('(a) LLaMA IC (Hidden States)', fontsize=10, fontweight='bold')
axes[0].text(18, 0.44, 'all p = 0.000', fontsize=8, fontstyle='italic', color='#2E7D32')

# (b) Gemma SAE — NOW SYMMETRIC
layers_g = [10, 18, 22, 26, 30]
f2v_g = [0.518, 0.808, 0.703, 0.716, 0.902]
v2f_g = [0.472, 0.726, 0.553, 0.637, 0.696]
axes[1].plot(layers_g, f2v_g, 'o-', color=C_FIX2VAR, label='Fix→Var', ms=7, zorder=3)
axes[1].plot(layers_g, v2f_g, 's-', color=C_VAR2FIX, label='Var→Fix', ms=7, zorder=3)
axes[1].axhline(0.5, color='red', ls='--', lw=0.7, alpha=0.5)
axes[1].set_ylim(0.4, 1.0); axes[1].set_xlabel('Layer'); axes[1].set_ylabel('Transfer AUC')
axes[1].legend(fontsize=8); axes[1].set_title('(b) Gemma IC (SAE Features)', fontsize=10, fontweight='bold')
axes[1].annotate('NS', xy=(10, 0.518), fontsize=7, color='red', ha='center', va='bottom')
axes[1].annotate('NS', xy=(10, 0.472), fontsize=7, color='red', ha='center', va='top')
axes[1].text(10, 0.42, 'Var BK=14\n(low power)', fontsize=6, color='gray', ha='center')

plt.savefig(f'{FIG}/v10_fig3_rq3_bettype_transfer.png', bbox_inches='tight')
plt.close()
print("Fig 3: RQ3 cross-bet-type transfer (balanced)")

# ═══════════════════════════════════════════════════════
# Fig 4: RQ3 — Var/Fix Direction Cosine (BALANCED: both models)
# ═══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# (a) Hidden State cosine
# Gemma IC HS (V9 data)
layers_gh = [10, 18, 22, 26, 30]
cos_gh = [-0.195, -0.082, 0.330, 0.443, 0.401]
# LLaMA IC HS
layers_lh = [8, 12, 22, 25, 30]
cos_lh = [0.882, 0.814, 0.835, 0.837, 0.819]

axes[0].plot(layers_gh, cos_gh, 'o-', color=C_GEMMA, label='Gemma (Var BK=14)', ms=7)
axes[0].plot(layers_lh, cos_lh, 's-', color=C_LLAMA, label='LLaMA (Var BK=77)', ms=7)
axes[0].axhline(0, color='gray', ls='-', lw=0.5, alpha=0.5)
axes[0].set_xlabel('Layer'); axes[0].set_ylabel('cos(Var BK dir, Fix BK dir)')
axes[0].set_ylim(-0.4, 1.0); axes[0].legend(fontsize=8)
axes[0].set_title('(a) Hidden State Level', fontsize=10, fontweight='bold')

# (b) SAE cosine — NOW SYMMETRIC
# Gemma IC SAE (just computed)
layers_gs = [10, 18, 22, 26, 30]
cos_gs = [-0.012, -0.125, 0.185, 0.317, 0.390]
# LLaMA IC SAE (V10 data)
layers_ls = [8, 12, 22, 25, 30]
cos_ls = [0.851, 0.767, 0.789, 0.789, 0.797]

axes[1].plot(layers_gs, cos_gs, 'o-', color=C_GEMMA, label='Gemma (Var BK=14)', ms=7)
axes[1].plot(layers_ls, cos_ls, 's-', color=C_LLAMA, label='LLaMA (Var BK=77)', ms=7)
axes[1].axhline(0, color='gray', ls='-', lw=0.5, alpha=0.5)
axes[1].set_xlabel('Layer'); axes[1].set_ylabel('cos(Var BK dir, Fix BK dir)')
axes[1].set_ylim(-0.4, 1.0); axes[1].legend(fontsize=8)
axes[1].set_title('(b) SAE Feature Level', fontsize=10, fontweight='bold')

plt.savefig(f'{FIG}/v10_fig4_rq3_direction_cosine.png', bbox_inches='tight')
plt.close()
print("Fig 4: RQ3 Var/Fix direction cosine (balanced, both levels)")

print("\n4 figures generated.")
