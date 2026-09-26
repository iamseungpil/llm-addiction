"""V5 Study Document Figure Generator"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import os

plt.rcParams.update({
    'figure.figsize': (12, 7),
    'figure.dpi': 150,
    'font.size': 12,
    'font.family': 'DejaVu Sans',
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2.0,
    'figure.constrained_layout.use': True,
})

COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9']
FIG_DIR = "/home/jovyan/llm-addiction/sae_v3_analysis/results/figures"
R = "/home/jovyan/llm-addiction/sae_v3_analysis/results/json"


def fig1_comprehensive_bk_matrix():
    """Figure 1: 2x3 matrix of BK classification AUC (SAE vs Hidden × IC/SM/MW)"""
    data = {
        'IC': {'SAE_DP': 0.964, 'Hidden_DP': 0.964, 'SAE_R1': 0.854, 'Hidden_R1': 0.856,
               'SAE_BM': 0.744, 'Hidden_BM': 0.724},
        'SM': {'SAE_DP': 0.981, 'Hidden_DP': 0.982, 'SAE_R1': 0.901, 'Hidden_R1': 0.900,
               'SAE_BM': 0.689, 'Hidden_BM': 0.630},
        'MW': {'SAE_DP': 0.966, 'Hidden_DP': 0.968, 'SAE_R1': 0.766, 'Hidden_R1': 0.764,
               'SAE_BM': 0.702, 'Hidden_BM': 0.835},
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    modes = ['DP', 'R1', 'BM']
    mode_labels = ['Decision Point\n(All Rounds)', 'Round 1\n(Balance=$100)', 'Balance-Matched']
    paradigms = ['IC', 'SM', 'MW']

    for i, (mode, label) in enumerate(zip(modes, mode_labels)):
        ax = axes[i]
        sae_vals = [data[p][f'SAE_{mode}'] for p in paradigms]
        hid_vals = [data[p][f'Hidden_{mode}'] for p in paradigms]

        x = np.arange(len(paradigms))
        w = 0.35
        bars1 = ax.bar(x - w/2, sae_vals, w, label='SAE Features', color=COLORS[0], alpha=0.85)
        bars2 = ax.bar(x + w/2, hid_vals, w, label='Hidden States', color=COLORS[1], alpha=0.85)

        ax.set_xlabel('Paradigm')
        ax.set_ylabel('AUC')
        ax.set_title(label)
        ax.set_xticks(x)
        ax.set_xticklabels(paradigms)
        ax.set_ylim(0.5, 1.05)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
        ax.legend(fontsize=9)

        # Add value labels
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                   f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                   f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

    fig.suptitle('Gemma-2-9B: Bankruptcy Prediction (SAE vs Hidden States)', fontsize=15, y=1.02)
    plt.savefig(f'{FIG_DIR}/v5_fig1_bk_classification_matrix.png', bbox_inches='tight')
    plt.close()
    print("  Figure 1: BK classification matrix saved")


def fig2_cross_domain_transfer():
    """Figure 2: Cross-domain transfer heatmap with bootstrap CI"""
    transfer = {
        ('IC', 'SM'): (0.893, 0.867, 0.921),
        ('IC', 'MW'): (0.908, 0.874, 0.944),
        ('SM', 'IC'): (0.616, 0.574, 0.679),
        ('SM', 'MW'): (0.657, 0.583, 0.706),
        ('MW', 'IC'): (0.631, 0.586, 0.680),
        ('MW', 'SM'): (0.877, 0.846, 0.906),
    }

    paradigms = ['IC', 'SM', 'MW']
    matrix = np.zeros((3, 3))
    ci_text = np.empty((3, 3), dtype=object)

    for i, src in enumerate(paradigms):
        for j, tgt in enumerate(paradigms):
            if i == j:
                # Within-domain (use balance-matched R1)
                within = {'IC': 0.854, 'SM': 0.901, 'MW': 0.766}
                matrix[i, j] = within[src]
                ci_text[i, j] = f'{within[src]:.3f}\n(within)'
            else:
                auc, lo, hi = transfer[(src, tgt)]
                matrix[i, j] = auc
                ci_text[i, j] = f'{auc:.3f}\n[{lo:.3f},{hi:.3f}]'

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0.5, vmax=1.0, aspect='auto')

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(paradigms)
    ax.set_yticklabels(paradigms)
    ax.set_xlabel('Test Domain')
    ax.set_ylabel('Train Domain')
    ax.set_title('Cross-Domain SAE Feature Transfer (Bootstrap 95% CI)')

    for i in range(3):
        for j in range(3):
            color = 'white' if matrix[i, j] < 0.7 else 'black'
            ax.text(j, i, ci_text[i, j], ha='center', va='center', fontsize=10, color=color)

    plt.colorbar(im, ax=ax, label='AUC', shrink=0.8)
    plt.savefig(f'{FIG_DIR}/v5_fig2_cross_domain_transfer.png', bbox_inches='tight')
    plt.close()
    print("  Figure 2: Cross-domain transfer heatmap saved")


def fig3_llama_vs_gemma():
    """Figure 3: LLaMA vs Gemma IC comparison"""
    # LLaMA IC results
    with open(f"{R}/llama_ic_analyses_20260308_202635.json") as f:
        lic = json.load(f)

    llama_dp = [(x['layer'], x['auc']) for x in lic['bk_classification']['dp']]
    llama_dp.sort()

    # Gemma IC results from V3
    with open(f"{R}/all_analyses_20260306_091055.json") as f:
        v3 = json.load(f)

    gemma_ic = v3['goal_a_classification']['ic']['sae']
    gemma_dp = [(x['layer'], x['auc']) for x in gemma_ic]
    gemma_dp.sort()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: BK classification by layer
    ax1.plot([x[0] for x in llama_dp], [x[1] for x in llama_dp],
             'o-', color=COLORS[0], label=f'LLaMA (best={max(x[1] for x in llama_dp):.3f})', markersize=4)
    ax1.plot([x[0] for x in gemma_dp], [x[1] for x in gemma_dp],
             's-', color=COLORS[1], label=f'Gemma (best={max(x[1] for x in gemma_dp):.3f})', markersize=4)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('AUC')
    ax1.set_title('(A) BK Classification by Layer (SAE)')
    ax1.legend()
    ax1.set_ylim(0.85, 1.0)

    # Panel B: Constraint encoding
    constraint_data = {
        'LLaMA': {'c10': 0, 'c30': 1.2, 'c50': 13.0, 'c70': 21.2},
        'Gemma': {'c10': 0, 'c30': 5.25, 'c50': 16.75, 'c70': 21.0},
    }

    x = np.arange(4)
    w = 0.35
    llama_rates = [constraint_data['LLaMA'][c] for c in ['c10', 'c30', 'c50', 'c70']]
    gemma_rates = [constraint_data['Gemma'][c] for c in ['c10', 'c30', 'c50', 'c70']]

    ax2.bar(x - w/2, llama_rates, w, label='LLaMA', color=COLORS[0], alpha=0.85)
    ax2.bar(x + w/2, gemma_rates, w, label='Gemma', color=COLORS[1], alpha=0.85)
    ax2.set_xlabel('Bet Constraint')
    ax2.set_ylabel('Bankruptcy Rate (%)')
    ax2.set_title('(B) BK Rate by Constraint')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['c10', 'c30', 'c50', 'c70'])
    ax2.legend()

    fig.suptitle('Cross-Architecture Validation: LLaMA-3.1-8B vs Gemma-2-9B (IC)', fontsize=14, y=1.02)
    plt.savefig(f'{FIG_DIR}/v5_fig3_llama_vs_gemma.png', bbox_inches='tight')
    plt.close()
    print("  Figure 3: LLaMA vs Gemma comparison saved")


def fig4_condition_encoding():
    """Figure 4: Condition-level encoding (bet type, constraint, prompt)"""
    data = {
        'Bet Type\n(2-class)': {'IC': 1.000, 'SM': 1.000, 'MW': 1.000},
        'Bet Constraint\n(4-class, IC)': {'Gemma': 0.966, 'LLaMA': 0.994},
        'Prompt Condition\n(32-class, IC)': {'Gemma': 1.000},
        'Bet Magnitude\n(round-level)': {'SM': 0.908, 'MW': 0.826},
        'Risk Choice\n(round-level)': {'IC': 0.681},
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    categories = list(data.keys())
    all_bars = []
    all_labels = []
    all_colors = []
    color_map = {'IC': COLORS[0], 'SM': COLORS[1], 'MW': COLORS[2],
                 'Gemma': COLORS[3], 'LLaMA': COLORS[4]}

    y_pos = 0
    y_positions = []
    y_labels = []

    for cat in categories:
        items = data[cat]
        for label, val in items.items():
            bar = ax.barh(y_pos, val, height=0.6, color=color_map.get(label, COLORS[5]), alpha=0.85)
            ax.text(val + 0.01, y_pos, f'{val:.3f}', va='center', fontsize=10)
            y_positions.append(y_pos)
            y_labels.append(f'{cat}\n({label})')
            y_pos += 1
        y_pos += 0.5

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel('AUC')
    ax.set_title('SAE Feature Encoding of Experimental Conditions')
    ax.set_xlim(0, 1.15)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.invert_yaxis()

    plt.savefig(f'{FIG_DIR}/v5_fig4_condition_encoding.png', bbox_inches='tight')
    plt.close()
    print("  Figure 4: Condition encoding saved")


def fig5_per_condition_hidden_bk():
    """Figure 5: Per-condition hidden BK classification heatmap"""
    # Data from comprehensive_gemma per-condition results
    data = {
        'IC': {
            'fixed': 0.936, 'variable': 0.941,
            'c30': 0.986, 'c50': 0.951, 'c70': 0.917,
            'with_G': 0.963, 'without_G': 0.960,
            'with_M': 0.956, 'without_M': 0.970,
        },
        'SM': {
            'variable': 0.962,
            'with_G': 0.966, 'without_G': 0.997,
            'with_H': 0.976, 'without_H': 0.973,
            'with_M': 0.973, 'without_M': 0.989,
            'with_P': 0.984, 'without_P': 0.982,
            'with_W': 0.982, 'without_W': 0.988,
        },
        'MW': {
            'fixed': 0.944, 'variable': 0.999,
            'with_G': 0.950, 'without_G': 0.979,
            'with_H': 0.971, 'without_H': 0.968,
            'with_M': 0.966, 'without_M': 0.982,
            'with_P': 0.968, 'without_P': 0.969,
            'with_W': 0.960, 'without_W': 0.966,
        }
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    for idx, (paradigm, conditions) in enumerate(data.items()):
        ax = axes[idx]
        conds = list(conditions.keys())
        vals = list(conditions.values())

        colors = [COLORS[0] if v >= 0.95 else COLORS[1] if v >= 0.90 else COLORS[3] for v in vals]
        bars = ax.barh(range(len(conds)), vals, color=colors, alpha=0.85)

        for i, (bar, v) in enumerate(zip(bars, vals)):
            ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)

        ax.set_yticks(range(len(conds)))
        ax.set_yticklabels(conds, fontsize=9)
        ax.set_xlabel('AUC')
        ax.set_title(f'{paradigm}')
        ax.set_xlim(0.85, 1.05)
        ax.axvline(x=0.95, color='gray', linestyle=':', alpha=0.5)
        ax.invert_yaxis()

    fig.suptitle('Per-Condition Hidden State BK Classification (Gemma, Decision Point)', fontsize=14, y=1.02)
    plt.savefig(f'{FIG_DIR}/v5_fig5_per_condition_hidden_bk.png', bbox_inches='tight')
    plt.close()
    print("  Figure 5: Per-condition hidden BK saved")


def fig6_progress_matrix():
    """Figure 6: Experiment/Analysis progress matrix"""
    # Analysis types (rows)
    analyses = [
        'Data Collected',
        'SAE Extracted',
        'Hidden States',
        'BK DP (SAE)',
        'BK DP (Hidden)',
        'BK R1 (SAE)',
        'BK R1 (Hidden)',
        'Balance-Matched (SAE)',
        'Balance-Matched (Hidden)',
        'Risk (SAE)',
        'Risk (Hidden)',
        'Per-Cond BK',
        'Per-Cond Risk',
        'Cross-Domain',
        'R1 Permutation',
        'Constraint Encoding',
    ]

    # Columns: Gemma IC/SM/MW, LLaMA IC/SM/MW
    columns = ['Gemma\nIC', 'Gemma\nSM', 'Gemma\nMW', 'LLaMA\nIC', 'LLaMA\nSM', 'LLaMA\nMW']

    # Status: 2=done, 1=running, 0=pending, -1=blocked
    status = np.array([
        [2, 2, 2, 2, 1, 1],  # Data
        [2, 2, 2, 2, -1, -1],  # SAE
        [2, 2, 2, 2, -1, -1],  # Hidden
        [2, 2, 2, 2, -1, -1],  # BK DP SAE
        [2, 2, 2, 0, -1, -1],  # BK DP Hidden
        [2, 2, 2, 2, -1, -1],  # BK R1 SAE
        [2, 2, 2, 0, -1, -1],  # BK R1 Hidden
        [2, 2, 2, 0, -1, -1],  # Bal-matched SAE
        [1, 1, 1, 0, -1, -1],  # Bal-matched Hidden
        [2, 2, 2, 0, -1, -1],  # Risk SAE
        [2, 1, 1, 0, -1, -1],  # Risk Hidden
        [2, 2, 2, 2, -1, -1],  # Per-cond BK
        [2, 2, 1, 0, -1, -1],  # Per-cond Risk
        [2, 2, 2, 0, -1, -1],  # Cross-domain
        [2, 2, 2, 2, -1, -1],  # Permutation
        [2, 2, 2, 2, -1, -1],  # Constraint
    ])

    fig, ax = plt.subplots(figsize=(10, 10))

    cmap = plt.cm.colors.ListedColormap(['#FFB3B3', '#FFEB99', '#99FF99', '#B3D9FF'])
    # -1=blocked(red), 0=pending(yellow), 1=running(green-light), 2=done(blue)
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(status, cmap=cmap, norm=norm, aspect='auto')

    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, fontsize=10)
    ax.set_yticks(range(len(analyses)))
    ax.set_yticklabels(analyses, fontsize=10)

    labels = {2: 'Done', 1: 'Running', 0: 'Pending', -1: 'Blocked'}
    for i in range(len(analyses)):
        for j in range(len(columns)):
            ax.text(j, i, labels[status[i, j]], ha='center', va='center', fontsize=8,
                   fontweight='bold' if status[i, j] == 2 else 'normal')

    ax.set_title('V5 Analysis Progress Matrix', fontsize=14)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#B3D9FF', label='Done'),
        Patch(facecolor='#99FF99', label='Running'),
        Patch(facecolor='#FFEB99', label='Pending'),
        Patch(facecolor='#FFB3B3', label='Blocked (needs data)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.savefig(f'{FIG_DIR}/v5_fig6_progress_matrix.png', bbox_inches='tight')
    plt.close()
    print("  Figure 6: Progress matrix saved")


if __name__ == '__main__':
    os.makedirs(FIG_DIR, exist_ok=True)
    print("Generating V5 figures...")
    fig1_comprehensive_bk_matrix()
    fig2_cross_domain_transfer()
    fig3_llama_vs_gemma()
    fig4_condition_encoding()
    fig5_per_condition_hidden_bk()
    fig6_progress_matrix()
    print("Done! 6 figures generated.")
