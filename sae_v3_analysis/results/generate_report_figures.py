#!/usr/bin/env python3
"""Generate publication-quality figures for the V3 SAE analysis report."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2.0,
})

COLORS = {'ic': '#2ecc71', 'sm': '#e74c3c', 'mw': '#3498db'}
LABELS = {'ic': 'Investment Choice', 'sm': 'Slot Machine', 'mw': 'Mystery Wheel'}
FIGDIR = "/home/jovyan/llm-addiction/sae_v3_analysis/results/figures"

# ============================================================
# Data from log (source: run_all_20260306_091055.log)
# ============================================================

# Goal A: SAE key-layer AUC (all layers available but we use key layers for clarity)
goal_a_sae = {
    'ic': {
        'layers': [0, 5, 10, 15, 18, 20, 22, 25, 28, 30, 33, 35, 38, 40],
        'auc':    [0.9245, 0.9385, 0.9506, 0.9540, 0.9574, 0.9523, 0.9637, 0.9624, 0.9538, 0.9479, 0.9489, 0.9469, 0.9432, 0.9557],
        'f1':     [0.574, 0.691, 0.724, 0.714, 0.724, 0.725, 0.722, 0.733, 0.724, 0.707, 0.711, 0.713, 0.719, 0.720],
    },
    'sm': {
        'layers': [0, 5, 10, 15, 18, 20, 22, 25, 28, 30, 33, 35, 38, 40],
        'auc':    [0.9432, 0.9695, 0.9772, 0.9764, 0.9804, 0.9695, 0.9608, 0.9529, 0.9496, 0.9384, 0.9471, 0.9567, 0.9373, 0.9577],
        'f1':     [0.357, 0.489, 0.518, 0.534, 0.544, 0.521, 0.535, 0.437, 0.422, 0.482, 0.456, 0.452, 0.476, 0.504],
    },
    'mw': {
        'layers': [0, 5, 10, 15, 18, 20, 22, 25, 28, 30, 33, 35, 38, 40],
        'auc':    [0.9409, 0.9308, 0.9451, 0.9388, 0.9373, 0.9385, 0.9410, 0.9508, 0.9215, 0.9133, 0.9658, 0.9287, 0.9294, 0.9541],
        'f1':     [0.294, 0.303, 0.321, 0.284, 0.271, 0.286, 0.207, 0.252, 0.218, 0.260, 0.254, 0.293, 0.197, 0.209],
    },
}

goal_a_hidden = {
    'ic': {
        'layers': [0, 5, 10, 15, 18, 20, 22, 25, 28, 30, 33, 35, 38, 40],
        'auc':    [0.9539, 0.9551, 0.9580, 0.9610, 0.9610, 0.9624, 0.9599, 0.9628, 0.9596, 0.9567, 0.9535, 0.9573, 0.9572, 0.9581],
    },
    'sm': {
        'layers': [0, 5, 10, 15, 18, 20, 22, 25, 28, 30, 33, 35, 38, 40],
        'auc':    [0.9770, 0.9724, 0.9816, 0.9792, 0.9779, 0.9770, 0.9779, 0.9783, 0.9761, 0.9695, 0.9688, 0.9654, 0.9691, 0.9644],
    },
    'mw': {
        'layers': [0, 5, 10, 15, 18, 20, 22, 25, 28, 30, 33, 35, 38, 40],
        'auc':    [0.9570, 0.9586, 0.9588, 0.9666, 0.9667, 0.9678, 0.9637, 0.9581, 0.9532, 0.9551, 0.9560, 0.9533, 0.9386, 0.9435],
    },
}

# Goal B: Early prediction
goal_b = {
    'ic': {
        'rounds': [1, 2, 3, 4, 5, 10, 15],
        'auc':    [0.8522, 0.8731, 0.8403, 0.7809, 0.7232, 0.4571, 0.3406],
        'n':      [1600, 1442, 1114, 957, 765, 252, 98],
        'n_bk':   [172, 172, 88, 65, 42, 8, 4],
    },
    'sm': {
        'rounds': [1, 2, 3, 4, 5, 10, 15, 20, 25, 30],
        'auc':    [0.8953, 0.8812, 0.8765, 0.8084, 0.8130, 0.8235, 0.7352, 0.6739, 0.8750, 0.9333],
        'n':      [3200, 3044, 2744, 2205, 1797, 641, 272, 111, 50, 25],
        'n_bk':   [87, 87, 87, 87, 87, 74, 48, 22, 12, 8],
    },
    'mw': {
        'rounds': [1, 2, 3, 4, 5, 10, 15],
        'auc':    [0.7662, 0.8353, 0.8508, 0.8395, 0.6536, 0.7465, 0.9067],
        'n':      [3200, 2879, 2608, 2031, 1404, 268, 78],
        'n_bk':   [54, 54, 54, 54, 18, 3, 3],
    },
}

# Goal C: Cross-domain transfer (partial - what's available)
goal_c = {
    'ic_to_sm': {'layer': 16, 'auc': 0.8944},
    'ic_to_mw': {'layer': 40, 'auc': 0.9162},
    'sm_to_ic': {'layer': 40, 'auc': 0.8199},
    'sm_to_mw': {'layer': 16, 'auc': 0.8116},
    'mw_to_ic': {'layer': 38, 'auc': 0.8326},
    'mw_to_sm': {'layer': 36, 'auc': 0.8823},
}


# ============================================================
# Figure 1: Goal A — SAE vs Hidden State AUC across layers (3 panels)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for col, p in enumerate(['ic', 'sm', 'mw']):
    ax = axes[col]
    # SAE
    ax.plot(goal_a_sae[p]['layers'], goal_a_sae[p]['auc'],
            '-o', color=COLORS[p], markersize=5, lw=2, label='SAE Feature', alpha=0.9)
    # Hidden
    ax.plot(goal_a_hidden[p]['layers'], goal_a_hidden[p]['auc'],
            '-s', color='#555555', markersize=5, lw=2, label='Hidden State', alpha=0.7)

    # Best markers
    sae_best_idx = np.argmax(goal_a_sae[p]['auc'])
    hid_best_idx = np.argmax(goal_a_hidden[p]['auc'])
    ax.plot(goal_a_sae[p]['layers'][sae_best_idx], goal_a_sae[p]['auc'][sae_best_idx],
            '*', color=COLORS[p], markersize=16, zorder=5)
    ax.plot(goal_a_hidden[p]['layers'][hid_best_idx], goal_a_hidden[p]['auc'][hid_best_idx],
            '*', color='#555555', markersize=16, zorder=5)

    # Annotations
    best_sae = max(goal_a_sae[p]['auc'])
    best_hid = max(goal_a_hidden[p]['auc'])
    ax.annotate(f'SAE: {best_sae:.3f}',
                xy=(goal_a_sae[p]['layers'][sae_best_idx], best_sae),
                xytext=(5, 10), textcoords='offset points', fontsize=9, color=COLORS[p], fontweight='bold')
    ax.annotate(f'Hidden: {best_hid:.3f}',
                xy=(goal_a_hidden[p]['layers'][hid_best_idx], best_hid),
                xytext=(5, -15), textcoords='offset points', fontsize=9, color='#555555', fontweight='bold')

    ax.axhline(y=0.5, color='gray', ls='--', alpha=0.4)
    ax.set_xlabel('Layer')
    ax.set_ylabel('AUC (5-fold CV)')
    n_info = {'ic': '1600 games, 172 BK', 'sm': '3200 games, 87 BK', 'mw': '3200 games, 54 BK'}
    ax.set_title(f'{LABELS[p]}\n({n_info[p]})', fontweight='bold')
    ax.legend(fontsize=9, loc='lower left')
    ax.set_ylim(0.88, 1.0)
    ax.set_xlim(-1, 42)

fig.suptitle('Goal A: Bankruptcy Classification — SAE Features vs Hidden States',
             fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(f'{FIGDIR}/report_goal_a_sae_vs_hidden.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print("Figure 1: Goal A saved")


# ============================================================
# Figure 2: Goal A — F1 comparison (bar chart, best layers)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))

paradigms = ['ic', 'sm', 'mw']
x = np.arange(len(paradigms))
width = 0.35

# Best SAE F1
sae_f1 = []
sae_labels = []
for p in paradigms:
    best_idx = np.argmax(goal_a_sae[p]['auc'])
    sae_f1.append(goal_a_sae[p]['f1'][best_idx])
    sae_labels.append(f"L{goal_a_sae[p]['layers'][best_idx]}")

# Best Hidden F1 (from log)
hid_f1 = [0.740, 0.531, 0.270]  # IC L25, SM L10, MW L20
hid_labels = ['L25', 'L10', 'L20']

bars1 = ax.bar(x - width/2, sae_f1, width, label='SAE Feature', color=[COLORS[p] for p in paradigms], alpha=0.8)
bars2 = ax.bar(x + width/2, hid_f1, width, label='Hidden State', color='#555555', alpha=0.6)

for i, (b1, b2) in enumerate(zip(bars1, bars2)):
    ax.text(b1.get_x() + b1.get_width()/2, b1.get_height() + 0.01,
            f'{sae_f1[i]:.3f}\n({sae_labels[i]})', ha='center', va='bottom', fontsize=9)
    ax.text(b2.get_x() + b2.get_width()/2, b2.get_height() + 0.01,
            f'{hid_f1[i]:.3f}\n({hid_labels[i]})', ha='center', va='bottom', fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels([LABELS[p] for p in paradigms])
ax.set_ylabel('F1 Score (Best Layer)')
ax.set_title('Goal A: F1 Score Comparison at Best AUC Layer', fontweight='bold')
ax.legend()
ax.set_ylim(0, 0.95)

fig.tight_layout()
fig.savefig(f'{FIGDIR}/report_goal_a_f1_comparison.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print("Figure 2: F1 comparison saved")


# ============================================================
# Figure 3: Goal B — Early Prediction
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5.5))

for p in ['ic', 'sm', 'mw']:
    data = goal_b[p]
    ax.plot(data['rounds'], data['auc'], '-o', color=COLORS[p],
            markersize=6, lw=2.5, label=LABELS[p])
    # Annotate R1
    ax.annotate(f"R1: {data['auc'][0]:.3f}", xy=(data['rounds'][0], data['auc'][0]),
                xytext=(10, -5), textcoords='offset points', fontsize=9, color=COLORS[p])

ax.axhline(y=0.5, color='gray', ls='--', alpha=0.5, label='Chance')
ax.set_xlabel('Round Number', fontsize=12)
ax.set_ylabel('BK Prediction AUC', fontsize=12)
ax.set_title('Goal B: Early Bankruptcy Prediction — Round-by-Round (L22)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.set_ylim(0.3, 1.05)
ax.set_xlim(0, 32)

# Add secondary info
ax.text(0.98, 0.02, 'Higher AUC = earlier detection of eventual bankruptcy',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=9, style='italic', color='gray')

fig.tight_layout()
fig.savefig(f'{FIGDIR}/report_goal_b_early_prediction.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print("Figure 3: Early prediction saved")


# ============================================================
# Figure 4: Goal C — Cross-domain transfer matrix (partial)
# ============================================================
fig, ax = plt.subplots(figsize=(7, 6))

paradigms_list = ['ic', 'sm', 'mw']
n = len(paradigms_list)
matrix = np.full((n, n), np.nan)

# Fill available results
transfer_data = {
    (0, 1): 0.8944,  # ic -> sm
    (0, 2): 0.9162,  # ic -> mw
    (1, 0): 0.8199,  # sm -> ic
    (1, 2): 0.8116,  # sm -> mw
    (2, 0): 0.8326,  # mw -> ic
    (2, 1): 0.8823,  # mw -> sm
}
for (i, j), v in transfer_data.items():
    matrix[i, j] = v

# Within-domain (best SAE AUC) for diagonal
within = [0.9637, 0.9811, 0.9658]  # IC L22, SM L12, MW L33
for i in range(n):
    matrix[i, i] = within[i]

im = ax.imshow(matrix, cmap='RdYlGn', vmin=0.5, vmax=1.0, aspect='auto')

labels = [LABELS[p] for p in paradigms_list]
ax.set_xticks(range(n))
ax.set_xticklabels(labels, fontsize=11)
ax.set_yticks(range(n))
ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel('Target (Test)', fontsize=12)
ax.set_ylabel('Source (Train)', fontsize=12)

for i in range(n):
    for j in range(n):
        if not np.isnan(matrix[i, j]):
            color = 'white' if matrix[i, j] < 0.7 else 'black'
            suffix = '\n(within)' if i == j else ''
            text = f'{matrix[i, j]:.3f}{suffix}'
            ax.text(j, i, text, ha='center', va='center',
                    fontsize=13, color=color, fontweight='bold')
        else:
            ax.text(j, i, '--', ha='center', va='center',
                    fontsize=10, color='gray')

plt.colorbar(im, ax=ax, label='AUC', shrink=0.8)
ax.set_title('Goal C: Cross-Domain BK Prediction Transfer\n(SAE Features, Best Layer)',
             fontsize=13, fontweight='bold')

fig.tight_layout()
fig.savefig(f'{FIGDIR}/report_goal_c_transfer_matrix.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print("Figure 4: Transfer matrix saved")


# ============================================================
# Figure 5: Goal D — SAE vs Hidden summary (grouped bar)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(3)
width = 0.35

sae_best_auc = [0.9637, 0.9811, 0.9658]
hid_best_auc = [0.9628, 0.9816, 0.9678]

bars1 = ax.bar(x - width/2, sae_best_auc, width, label='SAE Feature (131K sparse)',
               color=[COLORS[p] for p in paradigms_list], alpha=0.85, edgecolor='black', lw=0.5)
bars2 = ax.bar(x + width/2, hid_best_auc, width, label='Hidden State (3584 dense)',
               color='#555555', alpha=0.6, edgecolor='black', lw=0.5)

for i, (b1, b2) in enumerate(zip(bars1, bars2)):
    ax.text(b1.get_x() + b1.get_width()/2, b1.get_height() + 0.002,
            f'{sae_best_auc[i]:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.text(b2.get_x() + b2.get_width()/2, b2.get_height() + 0.002,
            f'{hid_best_auc[i]:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels([f'{LABELS[p]}\n({["172","87","54"][i]} BK)' for i, p in enumerate(paradigms_list)])
ax.set_ylabel('Best AUC (5-fold CV)')
ax.set_title('Goal D: SAE Feature vs Hidden State — Best Layer AUC Comparison',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.set_ylim(0.92, 1.0)
ax.axhline(y=0.5, color='gray', ls='--', alpha=0.3)

fig.tight_layout()
fig.savefig(f'{FIGDIR}/report_goal_d_feature_vs_hidden.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print("Figure 5: Feature vs Hidden saved")

print("\nAll report figures generated successfully!")
