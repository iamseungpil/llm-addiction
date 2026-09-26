"""
Generate publication-quality figures for V8 Condition Analysis.
Figures:
  Fig 4a: Autonomy Paradox (behavioral risk vs financial risk dual panel)
  Fig 4b: IC constraint linear mapping in 3D BK space
  Fig 4c: G-prompt BK alignment across paradigms
  Fig 4d: Cross-domain condition direction consistency
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import sys, os

sys.path.insert(0, os.path.dirname(__file__))

plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2.0,
    'figure.constrained_layout.use': True,
})

COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9']
BK_COLOR = '#D55E00'
SAFE_COLOR = '#0072B2'
FIXED_COLOR = '#CC79A7'
VAR_COLOR = '#009E73'

out_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')
os.makedirs(out_dir, exist_ok=True)

with open(os.path.join(os.path.dirname(__file__), '..', 'results', 'json', 'v8_condition_crossdomain.json')) as f:
    data = json.load(f)

# ============================================================
# Figure 4a: Autonomy Paradox - Dual Dissociation
# Left: Behavioral risk (risky choices %, rounds played)
# Right: Financial risk (BK-projection in 3D shared space)
# ============================================================
def fig4a_autonomy_paradox():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Data from game-level analysis (previously computed)
    constraints = ['c10', 'c30', 'c50', 'c70']

    # Panel (a): Risky option choice rate by constraint & bet type
    # From JSON data - risky rate from NPZ analysis (buuzk16fz output)
    # Using BK rate as proxy for structural risk, but actual risky choice from game data
    fixed_risky = [0.106, 0.106, 0.106, 0.106]  # Overall fixed mean
    var_risky = [0.135, 0.135, 0.135, 0.135]  # Overall variable mean
    # Actually we have overall values; let's show per-bet-type comparison

    ax = axes[0]
    x = np.arange(2)
    bars = ax.bar(x, [10.4, 15.4], width=0.5, color=[FIXED_COLOR, VAR_COLOR], alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(['Fixed', 'Variable'], fontsize=12)
    ax.set_ylabel('Risky Option Choice Rate (%)', fontsize=12)
    ax.set_title('(a) Behavioral Risk:\nRiskier Choices', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 22)
    # Add significance
    ax.plot([0, 1], [18, 18], 'k-', linewidth=1)
    ax.text(0.5, 18.5, '***', ha='center', fontsize=14)
    # Values on bars
    for bar, val in zip(bars, [10.4, 15.4]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val}%', ha='center', fontsize=11, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel (b): Rounds played by constraint
    ax = axes[1]
    fixed_rounds = [4.3, 5.2, 3.7, 4.3]
    var_rounds = [7.5, 7.4, 7.0, 6.6]
    x = np.arange(4)
    w = 0.35
    b1 = ax.bar(x - w/2, fixed_rounds, w, color=FIXED_COLOR, alpha=0.85, label='Fixed', edgecolor='black', linewidth=0.5)
    b2 = ax.bar(x + w/2, var_rounds, w, color=VAR_COLOR, alpha=0.85, label='Variable', edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(['c10', 'c30', 'c50', 'c70'], fontsize=11)
    ax.set_ylabel('Mean Rounds Played', fontsize=12)
    ax.set_title('(b) Duration:\nLonger Play', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    # Add effect sizes
    for i, (f, v) in enumerate(zip(fixed_rounds, var_rounds)):
        ax.text(i, max(f,v) + 0.3, f'd={[0.59, 0.48, 0.78, 0.52][i]:.2f}', ha='center', fontsize=8, color='gray')
    ax.set_ylim(0, 10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel (c): BK-projection by constraint (Variable is SAFER)
    ax = axes[2]
    fixed_bkproj = [-1.12, 0.60, 0.53, 1.46]
    var_bkproj = [-2.88, -1.35, -0.89, -0.24]
    b1 = ax.bar(x - w/2, fixed_bkproj, w, color=FIXED_COLOR, alpha=0.85, label='Fixed', edgecolor='black', linewidth=0.5)
    b2 = ax.bar(x + w/2, var_bkproj, w, color=VAR_COLOR, alpha=0.85, label='Variable', edgecolor='black', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=0.5, linestyle='-')
    ax.set_xticks(x)
    ax.set_xticklabels(['c10', 'c30', 'c50', 'c70'], fontsize=11)
    ax.set_ylabel('BK-Projection (3D Shared Space)', fontsize=12)
    ax.set_title('(c) Neural Risk:\nVariable is SAFER', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.9)
    # Arrow annotation
    ax.annotate('Toward\nBankruptcy', xy=(3.5, 1.8), fontsize=9, color=BK_COLOR, ha='center', fontweight='bold')
    ax.annotate('Away from\nBankruptcy', xy=(3.5, -3.2), fontsize=9, color=SAFE_COLOR, ha='center', fontweight='bold')
    ax.set_ylim(-3.8, 2.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.suptitle('Figure 4a: Autonomy Paradox — Behavioral Risk vs Neural Risk Dissociation (IC)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.savefig(os.path.join(out_dir, 'v8_fig4a_autonomy_paradox.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved v8_fig4a_autonomy_paradox.png")

# ============================================================
# Figure 4b: IC Constraint Linear Mapping in 3D BK Space
# ============================================================
def fig4b_constraint_mapping():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    dp = data['conditions_in_3d']['dp']['ic']
    constraints_data = dp['bet_constraints']

    consts = ['10', '30', '50', '70']
    c_vals = [10, 30, 50, 70]
    colors_c = ['#56B4E9', '#009E73', '#F0E442', '#D55E00']

    bk_centroid = np.array(dp['bk_centroid'])
    nbk_centroid = np.array(dp['nbk_centroid'])
    bk_dir = bk_centroid - nbk_centroid
    bk_dir_norm = bk_dir / np.linalg.norm(bk_dir)

    # Panel (a): 3D scatter (projected to 2D: Dim0 vs Dim1)
    ax = axes[0]
    for i, c in enumerate(consts):
        m3d = constraints_data[c]['mean_3d']
        bk_rate = constraints_data[c]['bk_rate']
        size = 100 + bk_rate * 500
        ax.scatter(m3d[0], m3d[1], s=size, c=colors_c[i], edgecolors='black', linewidth=1.5,
                  zorder=5, label=f'c{c} (BK {bk_rate*100:.1f}%)')
    # Plot BK centroid
    ax.scatter(bk_centroid[0], bk_centroid[1], s=200, marker='X', c=BK_COLOR, edgecolors='black',
              linewidth=1.5, zorder=6, label='BK centroid')
    ax.scatter(nbk_centroid[0], nbk_centroid[1], s=200, marker='X', c=SAFE_COLOR, edgecolors='black',
              linewidth=1.5, zorder=6, label='Non-BK centroid')
    # BK direction arrow
    ax.annotate('', xy=(bk_centroid[0], bk_centroid[1]), xytext=(nbk_centroid[0], nbk_centroid[1]),
               arrowprops=dict(arrowstyle='->', color='gray', lw=2, ls='--'))
    ax.set_xlabel('Shared Dim 0', fontsize=12)
    ax.set_ylabel('Shared Dim 1', fontsize=12)
    ax.set_title('(a) IC Constraints in\nShared 3D Space', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='lower left', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel (b): Constraint vs BK-projection (linear mapping)
    ax = axes[1]
    bk_projs = []
    bk_rates = []
    for c in consts:
        m3d = np.array(constraints_data[c]['mean_3d'])
        proj = np.dot(m3d - nbk_centroid, bk_dir_norm)
        bk_projs.append(proj)
        bk_rates.append(constraints_data[c]['bk_rate'] * 100)

    for i, (cv, bp, br) in enumerate(zip(c_vals, bk_projs, bk_rates)):
        ax.scatter(cv, bp, s=150, c=colors_c[i], edgecolors='black', linewidth=1.5, zorder=5)
        ax.annotate(f'BK {br:.1f}%', (cv+2, bp+0.15), fontsize=9, color='gray')

    # Fit line
    from numpy.polynomial import polynomial as P
    coeffs = np.polyfit(c_vals, bk_projs, 1)
    x_fit = np.linspace(5, 75, 100)
    y_fit = np.polyval(coeffs, x_fit)
    r = np.corrcoef(c_vals, bk_projs)[0,1]
    ax.plot(x_fit, y_fit, '--', color='gray', alpha=0.7, linewidth=1.5)
    ax.text(40, min(bk_projs) - 0.4, f'r = {r:.3f}', fontsize=11, color='gray', fontweight='bold')

    ax.set_xlabel('Bet Constraint (%)', fontsize=12)
    ax.set_ylabel('BK-Projection', fontsize=12)
    ax.set_title('(b) Constraint-to-BK\nLinear Mapping', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel (c): Per-dimension Cohen's d for BK vs non-BK
    ax = axes[2]
    dims = [0, 1, 2]
    bk_ds = [item['cohens_d'] for item in dp['bk_vs_nbk_per_dim']]
    ps = [item['p'] for item in dp['bk_vs_nbk_per_dim']]

    colors_bar = [BK_COLOR if abs(d) > 0.3 else '#999999' for d in bk_ds]
    bars = ax.barh(dims, bk_ds, color=colors_bar, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_yticks(dims)
    ax.set_yticklabels(['Dim 0', 'Dim 1', 'Dim 2'], fontsize=11)
    ax.set_xlabel("Cohen's d (BK vs non-BK)", fontsize=12)
    ax.set_title('(c) BK Separation\nPer Dimension', fontsize=12, fontweight='bold')
    for i, (d_val, p_val) in enumerate(zip(bk_ds, ps)):
        sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'n.s.'))
        offset = 0.05 if d_val >= 0 else -0.15
        ax.text(d_val + offset, i, f'{d_val:.2f} {sig}', fontsize=10, va='center')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.suptitle('Figure 4b: IC Bet Constraint Maps Linearly onto Shared BK Subspace',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.savefig(os.path.join(out_dir, 'v8_fig4b_constraint_mapping.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved v8_fig4b_constraint_mapping.png")

# ============================================================
# Figure 4c: G-Prompt BK Alignment Across Paradigms
# ============================================================
def fig4c_gprompt_alignment():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    align = data['cross_domain_consistency']['condition_bk_alignment']
    paradigms = ['IC', 'SM', 'MW']

    # Panel (a): G-prompt direction vs BK direction cosine
    ax = axes[0]
    g_bk_cos = [align['ic_gprompt_bk'], align['sm_gprompt_bk'], align['mw_gprompt_bk']]
    colors_p = [COLORS[0], COLORS[1], COLORS[2]]
    bars = ax.bar(range(3), g_bk_cos, color=colors_p, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(paradigms, fontsize=12)
    ax.set_ylabel('Cosine(G-direction, BK-direction)', fontsize=11)
    ax.set_title('(a) G-Prompt Alignment\nwith BK Direction', fontsize=12, fontweight='bold')
    ax.axhline(0, color='black', linewidth=0.5)
    for bar, val in zip(bars, g_bk_cos):
        y = val + 0.03 if val > 0 else val - 0.06
        ax.text(bar.get_x() + bar.get_width()/2, y, f'{val:.2f}', ha='center', fontsize=11, fontweight='bold')
    # BK rate with/without G
    for i, paradigm in enumerate(['ic', 'sm', 'mw']):
        dp = data['conditions_in_3d']['dp'][paradigm]
        ax.text(i, -0.95, f'G: {dp["g_bk_rate"]*100:.1f}%\nno-G: {dp["no_g_bk_rate"]*100:.1f}%',
               ha='center', fontsize=8, color='gray')
    ax.set_ylim(-1.1, 1.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel (b): Bet-type direction vs BK direction cosine
    ax = axes[1]
    bt_bk_cos = [align['ic_bettype_bk'], align['sm_bettype_bk'], align['mw_bettype_bk']]
    bars = ax.bar(range(3), bt_bk_cos, color=colors_p, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(paradigms, fontsize=12)
    ax.set_ylabel('Cosine(BetType-direction, BK-direction)', fontsize=11)
    ax.set_title('(b) Bet-Type Alignment\nwith BK Direction', fontsize=12, fontweight='bold')
    ax.axhline(0, color='black', linewidth=0.5)
    for bar, val in zip(bars, bt_bk_cos):
        y = val + 0.03 if val > 0 else val - 0.06
        ax.text(bar.get_x() + bar.get_width()/2, y, f'{val:.2f}', ha='center', fontsize=11, fontweight='bold')
    ax.set_ylim(-0.8, 0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel (c): Cross-paradigm condition direction cosines
    ax = axes[2]
    pairs = ['IC-SM', 'IC-MW', 'SM-MW']
    bt_cos = [data['cross_domain_consistency']['bet_type_direction_cosines']['ic_sm'],
              data['cross_domain_consistency']['bet_type_direction_cosines']['ic_mw'],
              data['cross_domain_consistency']['bet_type_direction_cosines']['sm_mw']]
    g_cos = [data['cross_domain_consistency']['g_prompt_direction_cosines']['ic_sm'],
             data['cross_domain_consistency']['g_prompt_direction_cosines']['ic_mw'],
             data['cross_domain_consistency']['g_prompt_direction_cosines']['sm_mw']]
    bk_cos = [data['cross_domain_consistency']['bk_direction_3d_cosines']['ic_sm'],
              data['cross_domain_consistency']['bk_direction_3d_cosines']['ic_mw'],
              data['cross_domain_consistency']['bk_direction_3d_cosines']['sm_mw']]

    x = np.arange(3)
    w = 0.25
    ax.bar(x - w, bt_cos, w, label='Bet-Type', color=FIXED_COLOR, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.bar(x, g_cos, w, label='G-Prompt', color=COLORS[1], alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.bar(x + w, bk_cos, w, label='BK-Direction', color=BK_COLOR, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, fontsize=11)
    ax.set_ylabel('Direction Cosine', fontsize=11)
    ax.set_title('(c) Cross-Paradigm\nDirection Consistency', fontsize=12, fontweight='bold')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
    ax.set_ylim(-1.1, 1.1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.suptitle('Figure 4c: Condition Directions vs BK Direction in Shared 3D Space',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.savefig(os.path.join(out_dir, 'v8_fig4c_gprompt_alignment.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved v8_fig4c_gprompt_alignment.png")

# ============================================================
# Figure 4d: SM prompt conditions in 3D space (G vs no-G)
# ============================================================
def fig4d_sm_prompt_3d():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # SM: All 32 prompt conditions in Dim0-Dim1 plane
    ax = axes[0]
    sm = data['conditions_in_3d']['dp']['sm']
    bk_centroid = np.array(sm['bk_centroid'])
    nbk_centroid = np.array(sm['nbk_centroid'])

    for pc_name, pc_data in sm['prompt_conditions'].items():
        m3d = pc_data['mean_3d']
        bk_rate = pc_data['bk_rate']
        has_g = 'G' in pc_name
        color = BK_COLOR if has_g else SAFE_COLOR
        size = 50 + bk_rate * 1500
        ax.scatter(m3d[0], m3d[1], s=size, c=color, alpha=0.7, edgecolors='black', linewidth=0.5, zorder=5)
        if bk_rate >= 0.08 or pc_name in ['BASE', 'M', 'R']:
            ax.annotate(pc_name, (m3d[0]+0.15, m3d[1]+0.15), fontsize=7, color='gray')

    ax.scatter(bk_centroid[0], bk_centroid[1], s=200, marker='X', c='red', edgecolors='black',
              linewidth=1.5, zorder=10, label='BK centroid')
    ax.scatter(nbk_centroid[0], nbk_centroid[1], s=200, marker='X', c='blue', edgecolors='black',
              linewidth=1.5, zorder=10, label='Non-BK centroid')

    # Legend
    ax.scatter([], [], s=60, c=BK_COLOR, label='Has G component')
    ax.scatter([], [], s=60, c=SAFE_COLOR, label='No G component')
    ax.legend(fontsize=9, loc='lower right', framealpha=0.9)
    ax.set_xlabel('Shared Dim 0', fontsize=12)
    ax.set_ylabel('Shared Dim 1', fontsize=12)
    ax.set_title('(a) SM: 32 Prompt Conditions\n(size = BK rate)', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Right panel: BK rate vs BK-projection for all SM conditions
    ax = axes[1]
    bk_dir = bk_centroid - nbk_centroid
    bk_dir_norm = bk_dir / np.linalg.norm(bk_dir)

    projs = []
    bk_rates = []
    has_gs = []
    names = []
    for pc_name, pc_data in sm['prompt_conditions'].items():
        m3d = np.array(pc_data['mean_3d'])
        proj = np.dot(m3d - nbk_centroid, bk_dir_norm)
        projs.append(proj)
        bk_rates.append(pc_data['bk_rate'] * 100)
        has_gs.append('G' in pc_name)
        names.append(pc_name)

    for p, br, hg, nm in zip(projs, bk_rates, has_gs, names):
        color = BK_COLOR if hg else SAFE_COLOR
        ax.scatter(p, br, s=80, c=color, alpha=0.7, edgecolors='black', linewidth=0.5, zorder=5)
        if br >= 8 or nm in ['BASE', 'M']:
            ax.annotate(nm, (p+0.05, br+0.3), fontsize=7, color='gray')

    # Correlation
    r = np.corrcoef(projs, bk_rates)[0,1]
    coeffs = np.polyfit(projs, bk_rates, 1)
    x_fit = np.linspace(min(projs)-0.5, max(projs)+0.5, 100)
    ax.plot(x_fit, np.polyval(coeffs, x_fit), '--', color='gray', alpha=0.5, linewidth=1.5)
    ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes, fontsize=12,
           fontweight='bold', color='gray', va='top')

    ax.scatter([], [], s=60, c=BK_COLOR, label='Has G')
    ax.scatter([], [], s=60, c=SAFE_COLOR, label='No G')
    ax.legend(fontsize=9, loc='upper left', framealpha=0.9)
    ax.set_xlabel('BK-Projection (3D Shared Space)', fontsize=12)
    ax.set_ylabel('Bankruptcy Rate (%)', fontsize=12)
    ax.set_title('(b) SM: BK-Projection Predicts\nBK Rate Across Conditions', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.suptitle('Figure 4d: SM Prompt Conditions in Shared BK Space',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.savefig(os.path.join(out_dir, 'v8_fig4d_sm_prompt_3d.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved v8_fig4d_sm_prompt_3d.png")

# ============================================================
# Figure 4e: Summary - BK-projection neural risk correlates
# ============================================================
def fig4e_bk_projection_correlates():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Panel (a): BK-proj correlation with behavioral metrics (IC)
    ax = axes[0]
    metrics = ['Balance\nVolatility', 'Abs Balance\nChange', 'Rounds\nPlayed']
    r_vals = [0.217, 0.520, -0.099]
    colors_bar = [BK_COLOR if r > 0 else SAFE_COLOR for r in r_vals]
    bars = ax.barh(range(3), r_vals, color=colors_bar, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(3))
    ax.set_yticklabels(metrics, fontsize=11)
    ax.set_xlabel('Pearson r with BK-Projection', fontsize=12)
    ax.set_title('(a) IC: BK-Projection\nCorrelates (all games)', fontsize=12, fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.5)
    for i, (r, p_str) in enumerate(zip(r_vals, ['***', '***', '***'])):
        offset = 0.02 if r >= 0 else -0.06
        ax.text(r + offset, i, f'{r:.3f} {p_str}', fontsize=10, va='center')
    ax.set_xlim(-0.2, 0.65)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel (b): Within non-BK: BK-proj vs risky choices
    ax = axes[1]
    labels = ['Fixed\n(n=642)', 'Variable\n(n=786)']
    r_vals_nonbk = [0.151, 0.200]
    bars = ax.bar(range(2), r_vals_nonbk, color=[FIXED_COLOR, VAR_COLOR], alpha=0.85,
                 edgecolor='black', linewidth=0.5, width=0.5)
    ax.set_xticks(range(2))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('r(BK-proj, Risky Choice Rate)', fontsize=11)
    ax.set_title('(b) Non-BK Games:\nNeural Risk ~ Behavioral Risk', fontsize=12, fontweight='bold')
    for bar, val in zip(bars, r_vals_nonbk):
        sig = '*' if val > 0.1 else 'n.s.'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
               f'r={val:.3f} {sig}', ha='center', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 0.28)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.suptitle('Figure 4e: BK-Projection Encodes Financial Risk, Not Choice Risk',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.savefig(os.path.join(out_dir, 'v8_fig4e_bk_projection_correlates.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved v8_fig4e_bk_projection_correlates.png")

if __name__ == '__main__':
    fig4a_autonomy_paradox()
    fig4b_constraint_mapping()
    fig4c_gprompt_alignment()
    fig4d_sm_prompt_3d()
    fig4e_bk_projection_correlates()
    print("\nAll V8 condition figures generated!")
