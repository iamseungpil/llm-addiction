#!/usr/bin/env python3
"""Generate publication-quality figures for V9 cross-model study."""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'figure.dpi': 150, 'font.size': 11, 'font.family': 'serif',
    'axes.labelsize': 12, 'axes.titlesize': 13, 'legend.fontsize': 10,
    'axes.grid': True, 'grid.alpha': 0.3, 'lines.linewidth': 2.0,
    'figure.constrained_layout.use': True,
})
COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9']
JSON_DIR = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis/results/json")
FIG_DIR = Path("/home/v-seungplee/llm-addiction/sae_v3_analysis/results/figures")

def load(name):
    with open(JSON_DIR / name) as f: return json.load(f)

# ============================================================
# Figure 1: Cross-Domain SAE Consistency — Gemma 3-paradigm
# ============================================================
def fig1_crossdomain_consistency():
    b1 = load("b1_b2_results_20260317_125620.json")
    layers = [10, 12, 18, 22, 26, 30, 33]
    pcts = [float(b1["b1a_multilayer"][str(l)]["sign_consistent_pct"].rstrip('%')) for l in layers]
    strong = [b1["b1a_multilayer"][str(l)]["n_strong_d02"] for l in layers]
    active = [b1["b1a_multilayer"][str(l)]["n_active"] for l in layers]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: sign-consistency rate with chance line
    ax1.bar([f'L{l}' for l in layers], pcts, color=COLORS[0], alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.axhline(y=25, color='red', linestyle='--', linewidth=1.5, label='Chance (25%)')
    # Mark significant layers
    binomial_p = [0.062, 0.080, 0.005, 2.4e-7, 6.5e-13, 2.5e-19, 2.5e-15]
    for i, (l, p) in enumerate(zip(layers, binomial_p)):
        if p < 0.01:
            ax1.text(i, pcts[i]+1, '***', ha='center', fontsize=10, fontweight='bold')
        elif p < 0.05:
            ax1.text(i, pcts[i]+1, '*', ha='center', fontsize=10)
        else:
            ax1.text(i, pcts[i]+1, 'NS', ha='center', fontsize=9, color='gray')
    ax1.set_ylabel('Sign-Consistency Rate (%)')
    ax1.set_xlabel('Layer')
    ax1.set_title('(a) Gemma 3-Paradigm Sign-Consistency vs Chance')
    ax1.legend()
    ax1.set_ylim(0, 55)

    # Right: strong feature counts
    ax2.bar([f'L{l}' for l in layers], strong, color=COLORS[2], alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('Strong Features (geo_mean d ≥ 0.2)')
    ax2.set_xlabel('Layer')
    ax2.set_title('(b) Cross-Domain Strong Features per Layer')

    plt.savefig(FIG_DIR / 'fig1_crossdomain_consistency.png', bbox_inches='tight')
    plt.close()
    print("  Fig 1: Cross-domain consistency")


# ============================================================
# Figure 2: BK Direction Cosine — Gemma vs LLaMA IC
# ============================================================
def fig2_bk_direction():
    b1 = load("b1_b2_results_20260317_125620.json")
    llama = load("llama_hidden_analyses_20260318_172609.json")

    gemma_layers = [10, 18, 22, 26, 30]
    gemma_cos = [b1["b2a_bk_only"][f"ic_L{l}"]["cos_bk_direction"] for l in gemma_layers]

    llama_layers = [8, 12, 22, 25, 30]
    llama_cos = [llama["analysis2_bk_direction"][f"L{l}"]["cos_bk_direction"] for l in llama_layers]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gemma_layers, gemma_cos, 'o-', color=COLORS[0], markersize=8, label='Gemma IC (Var BK=14, Fix BK=158)')
    ax.plot(llama_layers, llama_cos, 's-', color=COLORS[1], markersize=8, label='LLaMA IC (Var BK=77, Fix BK=65)')
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Cosine Similarity (Variable BK dir, Fixed BK dir)')
    ax.set_title('BK Direction Convergence: Variable vs Fixed')
    ax.legend()
    ax.set_ylim(-0.4, 1.0)

    plt.savefig(FIG_DIR / 'fig2_bk_direction_crossmodel.png', bbox_inches='tight')
    plt.close()
    print("  Fig 2: BK direction cosine cross-model")


# ============================================================
# Figure 3: Factor Decomposition — Cross-Model
# ============================================================
def fig3_factor_decomposition():
    sc = load("selfcritique_20260318_114430.json")
    gemma = sc["test1_factor_decomp_perm"]["gemma"]
    llama = sc["test1_factor_decomp_perm"]["llama"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(2)
    width = 0.5
    bars = ax.bar(x, [gemma["actual_pct"], llama["actual_pct"]], width,
                  color=[COLORS[0], COLORS[1]], alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.axhline(y=gemma["perm_mean_pct"], color='red', linestyle='--', linewidth=1.5,
               label=f'Permutation null (~{gemma["perm_mean_pct"]:.1f}%)')

    for i, (bar, pct) in enumerate(zip(bars, [gemma["actual_pct"], llama["actual_pct"]])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{pct:.1f}%\np=0.000', ha='center', fontsize=11, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(['Gemma\n(IC+SM+MW, 8000 games)', 'LLaMA\n(IC+SM, 4800 games)'])
    ax.set_ylabel('Outcome-Significant Features (%)')
    ax.set_title('Factor Decomposition: BK Signal Independence (Permutation Validated)')
    ax.legend()
    ax.set_ylim(0, 85)

    plt.savefig(FIG_DIR / 'fig3_factor_decomposition.png', bbox_inches='tight')
    plt.close()
    print("  Fig 3: Factor decomposition cross-model")


# ============================================================
# Figure 4: Cross-Domain Transfer Heatmap — Gemma
# ============================================================
def fig4_transfer_heatmap():
    fv = load("final_verification_20260317_201024.json")
    transfer = fv["gemma_crossdomain_transfer"]

    layers = [18, 22, 26, 30]
    pairs = [("IC", "SM"), ("IC", "MW"), ("SM", "MW")]

    matrix = np.zeros((len(layers), len(pairs)))
    for i, l in enumerate(layers):
        for j, (tr, te) in enumerate(pairs):
            key = f"L{l}_{tr}_{te}"
            if key in transfer:
                matrix[i, j] = transfer[key]["auc"]
            else:
                matrix[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0.4, vmax=1.0, aspect='auto')

    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels([f'{tr}→{te}' for tr, te in pairs])
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f'L{l}' for l in layers])
    ax.set_xlabel('Transfer Direction')
    ax.set_ylabel('Layer')
    ax.set_title('Gemma Cross-Domain Transfer AUC')

    for i in range(len(layers)):
        for j in range(len(pairs)):
            v = matrix[i, j]
            if not np.isnan(v):
                key = f"L{layers[i]}_{pairs[j][0]}_{pairs[j][1]}"
                p = transfer.get(key, {}).get("perm_p", 1)
                sig = '***' if p < 0.001 else ('NS' if p > 0.05 else '*')
                ax.text(j, i, f'{v:.3f}\n{sig}', ha='center', va='center',
                        fontsize=10, fontweight='bold' if p < 0.05 else 'normal',
                        color='white' if v < 0.6 else 'black')

    plt.colorbar(im, ax=ax, label='AUC')
    plt.savefig(FIG_DIR / 'fig4_transfer_heatmap.png', bbox_inches='tight')
    plt.close()
    print("  Fig 4: Transfer heatmap")


# ============================================================
# Figure 5: Prompt Component BK Rate — Cross-Model
# ============================================================
def fig5_prompt_components():
    gf = load("gap_filling_20260317_194117.json")
    ls = load("llama_symmetric_20260318.json")

    gemma_components = ['G', 'M', 'W', 'P']  # R is Gemma-only
    gemma_ratios = [gf["task10_prompt_components"]["L22"][c]["bk_rate_ratio"] for c in gemma_components]

    llama_components = ['G', 'M', 'W', 'P']
    llama_ratios = [ls["analysis2_prompt_component"]["L20"][c]["bk_ratio"] for c in llama_components]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(gemma_components))
    w = 0.35
    ax.bar(x - w/2, gemma_ratios, w, color=COLORS[0], alpha=0.8, label='Gemma SM', edgecolor='black', linewidth=0.5)
    ax.bar(x + w/2, llama_ratios, w, color=COLORS[1], alpha=0.8, label='LLaMA SM', edgecolor='black', linewidth=0.5)

    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(['G (Goal)', 'M (Mood/Money)', 'W (Warning)', 'P (Persona)'])
    ax.set_ylabel('BK Rate Ratio (with/without component)')
    ax.set_title('Prompt Component BK Effect: Cross-Model Comparison')
    ax.legend()

    # Annotate G's extreme value
    ax.annotate(f'{gemma_ratios[0]:.1f}x', xy=(0 - w/2, gemma_ratios[0]),
                xytext=(-0.5, gemma_ratios[0] + 2), fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='black'))

    plt.savefig(FIG_DIR / 'fig5_prompt_components.png', bbox_inches='tight')
    plt.close()
    print("  Fig 5: Prompt components cross-model")


# ============================================================
# Figure 6: Classification AUC — Cross-Model Cross-Paradigm
# ============================================================
def fig6_classification_auc():
    fv = load("final_verification_20260317_201024.json")
    ls = load("llama_symmetric_20260318.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # SM AUC
    gemma_sm_layers = [18, 20, 22, 26, 30]
    gemma_sm_auc = [0.978, 0.976, 0.970, 0.972, 0.953]  # from Gemma SM verification

    llama_sm = fv["llama_sm_classification"]
    llama_sm_layers = sorted([int(k) for k in llama_sm.keys()])
    llama_sm_auc = [llama_sm[str(l)]["auc"] for l in llama_sm_layers]

    ax1.plot(gemma_sm_layers, gemma_sm_auc, 'o-', color=COLORS[0], markersize=6, label='Gemma SM')
    ax1.plot(llama_sm_layers, llama_sm_auc, 's-', color=COLORS[1], markersize=6, label='LLaMA SM')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('BK Classification AUC')
    ax1.set_title('(a) SM: BK Classification AUC across Layers')
    ax1.legend()
    ax1.set_ylim(0.93, 0.99)

    # IC AUC
    gemma_ic = ls["analysis3_gemma_ic_auc"]
    gemma_ic_layers = sorted([int(k.replace('L','')) for k in gemma_ic.keys()])
    gemma_ic_auc = [gemma_ic[f"L{l}"]["mean_auc"] for l in gemma_ic_layers]

    llama_ic_data = load("llama_ic_results_20260317_130655.json")
    llama_ic_layers = sorted([int(k) for k in llama_ic_data["b1_classification"].keys()])
    llama_ic_auc = [llama_ic_data["b1_classification"][str(l)]["auc"] for l in llama_ic_layers]

    ax2.plot(gemma_ic_layers, gemma_ic_auc, 'o-', color=COLORS[0], markersize=6, label='Gemma IC')
    ax2.plot(llama_ic_layers, llama_ic_auc, 's-', color=COLORS[1], markersize=4, label='LLaMA IC')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('BK Classification AUC')
    ax2.set_title('(b) IC: BK Classification AUC across Layers')
    ax2.legend()
    ax2.set_ylim(0.91, 0.97)

    plt.savefig(FIG_DIR / 'fig6_classification_auc.png', bbox_inches='tight')
    plt.close()
    print("  Fig 6: Classification AUC cross-model")


# ============================================================
# Figure 7: Shared BK Neurons — Cross-Model Comparison
# ============================================================
def fig7_shared_bk():
    b1 = load("b1_b2_results_20260317_125620.json")
    llama = load("llama_hidden_analyses_20260318_172609.json")

    models = ['Gemma IC', 'LLaMA IC', 'LLaMA SM']
    shared = [
        b1["b2b_interaction"]["ic"]["shared_bk_neurons"],
        llama["analysis3_interaction"]["ic"]["shared_bk_neurons"],
        llama["analysis3_interaction"]["sm"]["shared_bk_neurons"],
    ]
    interaction = [
        b1["b2b_interaction"]["ic"]["interaction_sig_p01"],
        llama["analysis3_interaction"]["ic"]["interaction_sig_p01"],
        llama["analysis3_interaction"]["sm"]["interaction_sig_p01"],
    ]
    n_neurons = [3584, 4096, 4096]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(models))
    w = 0.35
    bars1 = ax.bar(x - w/2, [s/n*100 for s, n in zip(shared, n_neurons)], w,
                   color=COLORS[2], alpha=0.8, label='Shared BK Neurons (%)', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + w/2, [i/n*100 for i, n in zip(interaction, n_neurons)], w,
                   color=COLORS[3], alpha=0.8, label='Interaction Neurons (%)', edgecolor='black', linewidth=0.5)

    for bar, val, n in zip(bars1, shared, n_neurons):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val}\n({val/n*100:.1f}%)', ha='center', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel('Percentage of Total Neurons')
    ax.set_title('Shared vs Interaction BK Neurons (L22)')
    ax.legend()

    plt.savefig(FIG_DIR / 'fig7_shared_bk_neurons.png', bbox_inches='tight')
    plt.close()
    print("  Fig 7: Shared BK neurons")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating publication figures...")
    fig1_crossdomain_consistency()
    fig2_bk_direction()
    fig3_factor_decomposition()
    fig4_transfer_heatmap()
    fig5_prompt_components()
    fig6_classification_auc()
    fig7_shared_bk()
    print(f"\nDone! {len(list(FIG_DIR.glob('*.png')))} figures in {FIG_DIR}")

if __name__ == "__main__":
    main()
