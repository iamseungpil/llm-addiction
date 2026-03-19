#!/usr/bin/env python3
"""
Improved IC SAE Visualization — Publication-Quality Figures
===========================================================

Mirrors visualize_improved_figures.py style for IC data:
- Fig A: 1-AUC log scale + active feature count (IC 5 subsets)
- Fig B: Enhanced heatmap with pattern annotations + η²
- Fig D: η² interaction with SM overlay + Cohen's benchmarks
- Fig E: Interaction % with SM overlay + IC-specific annotations
- Fig F: Constraint effect (c30 vs c50) improved
- CP Fig: Cross-paradigm AUC comparison (improved)

Usage:
    python visualize_ic_improved_figures.py
    python visualize_ic_improved_figures.py --only fig_a fig_b
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import seaborn as sns


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
IC_RESULTS_DIR = Path("/home/jovyan/beomi/llm-addiction-data/sae_condition_comparison/ic_cross_paradigm/results")
SM_RESULTS_DIR = Path("/home/jovyan/beomi/llm-addiction-data/sae_condition_comparison/within_model/results")
OUTPUT_DIR = Path("/home/jovyan/llm-addiction/exploratory_experiments/additional_experiments/"
                  "sae_feature_analysis/results/ic_improved")

MODEL_LABELS = {"ic": "Investment Choice (Gemma-2-9B)", "sm": "Slot Machine (Gemma-2-9B)"}

COLORS = {
    "ic": "#27ae60",
    "sm": "#c0392b",
    "all": "#2c3e50",
    "variable": "#e74c3c",
    "fixed": "#3498db",
    "c30": "#f39c12",
    "c50": "#8e44ad",
}

ETA_SQ_SMALL = 0.01
ETA_SQ_MEDIUM = 0.06
ETA_SQ_LARGE = 0.14


def load_latest(results_dir: Path, prefix: str) -> Optional[Dict]:
    matches = sorted(results_dir.glob(f"{prefix}_*.json"))
    if not matches:
        return None
    with open(matches[-1]) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Fig A: 1-AUC log scale + Active Feature Count (IC 5 subsets)
# ---------------------------------------------------------------------------
def fig_a_improved(ic_clf: Dict, save_path: Path):
    """IC classification error on log scale with 5 subsets + active feature bar."""
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 1, hspace=0.35)

    # --- Top: 1-AUC log scale ---
    ax1 = fig.add_subplot(gs[0, 0])

    subset_styles = {
        'all_games':     {'color': COLORS['all'],      'lw': 2.5, 'ls': '-',  'marker': 'o', 'ms': 4, 'label': 'All games'},
        'variable_only': {'color': COLORS['variable'],  'lw': 1.0, 'ls': '--', 'marker': None, 'ms': 0, 'label': 'Variable'},
        'fixed_only':    {'color': COLORS['fixed'],     'lw': 1.0, 'ls': '--', 'marker': None, 'ms': 0, 'label': 'Fixed'},
        'c30_only':      {'color': COLORS['c30'],       'lw': 1.0, 'ls': '-.', 'marker': None, 'ms': 0, 'label': 'c30 (30% constraint)'},
        'c50_only':      {'color': COLORS['c50'],       'lw': 1.0, 'ls': '-.', 'marker': None, 'ms': 0, 'label': 'c50 (50% constraint)'},
    }

    for subset_name, style in subset_styles.items():
        results = [r for r in ic_clf.get(subset_name, []) if not r.get('skipped', False)]
        if not results:
            continue
        layers = [r['layer'] for r in results]
        error = [max(1 - r['auc_mean'], 1e-6) for r in results]

        kwargs = dict(color=style['color'], lw=style['lw'], linestyle=style['ls'],
                      alpha=0.7 if subset_name != 'all_games' else 1.0, label=style['label'])
        if style['marker']:
            kwargs['marker'] = style['marker']
            kwargs['markersize'] = style['ms']
        ax1.semilogy(layers, error, **kwargs)

    ax1.axhline(y=0.01, color='gray', linestyle=':', alpha=0.5, lw=0.8)
    ax1.axhline(y=0.05, color='gray', linestyle=':', alpha=0.3, lw=0.8)
    ax1.text(41.5, 0.012, '1% error', fontsize=7, color='gray', va='bottom')
    ax1.text(41.5, 0.055, '5% error', fontsize=7, color='gray', va='bottom')

    ax1.set_ylabel('Classification Error (1 - AUC)', fontsize=11)
    ax1.set_title('IC: SAE Features Predict Bankruptcy Outcome', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=8, loc='upper right', ncol=2)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_ylim(1e-2, 2e-1)
    ax1.set_xlim(-0.5, 42)

    # Annotate best layer
    all_results = [r for r in ic_clf.get('all_games', []) if not r.get('skipped', False)]
    if all_results:
        best = min(all_results, key=lambda r: 1 - r['auc_mean'])
        best_err = max(1 - best['auc_mean'], 1e-6)
        ax1.annotate(f"Best: L{best['layer']}\nAUC={best['auc_mean']:.4f}",
                     xy=(best['layer'], best_err),
                     xytext=(15, 25), textcoords='offset points',
                     fontsize=8, fontweight='bold', color=COLORS['all'],
                     arrowprops=dict(arrowstyle='->', color=COLORS['all'], lw=1.0))

    # --- Bottom: Active feature count ---
    ax2 = fig.add_subplot(gs[1, 0])
    if all_results:
        layers = [r['layer'] for r in all_results]
        n_features = [r.get('n_features', 0) for r in all_results]

        ax2.bar(layers, n_features, color=COLORS['ic'], alpha=0.6, width=0.8)

        max_idx = int(np.argmax(n_features))
        ax2.annotate(f"{n_features[max_idx]:,}",
                     xy=(layers[max_idx], n_features[max_idx]),
                     xytext=(0, 5), textcoords='offset points',
                     fontsize=7, ha='center', fontweight='bold', color=COLORS['ic'])

        avg_pct = np.mean([100 * nf / 131072 for nf in n_features])
        ax2.text(0.98, 0.95,
                 f"Avg: {avg_pct:.1f}% of\n131K total features\npass activation filter",
                 transform=ax2.transAxes, ha='right', va='top', fontsize=8,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    ax2.set_xlabel('Layer', fontsize=11)
    ax2.set_ylabel('Active SAE Features\n(>1% activation rate)', fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


# ---------------------------------------------------------------------------
# Fig B: Enhanced Heatmap with Pattern Annotations
# ---------------------------------------------------------------------------
def fig_b_improved(ic_anova: Dict, save_path: Path):
    """IC heatmap with pattern type labels + η² annotations."""
    layer_key = 'layer_results' if 'layer_results' in ic_anova else 'layer_summary'
    layer_data = ic_anova[layer_key]

    all_sig = []
    for lr in layer_data:
        feat_key = 'significant_features' if 'significant_features' in lr else 'top_features'
        for feat in lr.get(feat_key, []):
            if feat.get('interaction_significant', True):
                feat_copy = dict(feat)
                feat_copy['layer'] = lr['layer']
                all_sig.append(feat_copy)

    if not all_sig:
        print("  No significant interactions for heatmap")
        return

    all_sig.sort(key=lambda x: x['interaction_eta_sq'], reverse=True)
    top_n = min(10, len(all_sig))
    top_feats = all_sig[:top_n]

    # IC ANOVA uses encoded 0/1 keys
    cells = ['0_0', '0_1', '1_0', '1_1']
    cell_labels = ['Var-BK', 'Var-VS', 'Fix-BK', 'Fix-VS']
    matrix = np.zeros((top_n, 4))

    for i, feat in enumerate(top_feats):
        gm = feat['group_means']
        for j, cell in enumerate(cells):
            matrix[i, j] = gm.get(cell, 0.0)

    row_means = matrix.mean(axis=1, keepdims=True)
    row_stds = matrix.std(axis=1, keepdims=True)
    row_stds[row_stds == 0] = 1.0
    matrix_z = (matrix - row_means) / row_stds

    # Classify pattern
    pattern_labels = []
    for i in range(top_n):
        z = matrix_z[i]
        max_col = int(np.argmax(z))
        pattern_labels.append(['Var-BK', 'Var-VS', 'Fix-BK', 'Fix-VS'][max_col])

    feat_labels = [f"L{f['layer']}/F{f['feature_idx']}" for f in top_feats]
    eta_values = [f"{f['interaction_eta_sq']:.3f}" for f in top_feats]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(matrix_z, cmap='RdBu_r', aspect='auto', vmin=-2.2, vmax=2.2)

    ax.set_xticks(range(4))
    ax.set_xticklabels(cell_labels, fontsize=10)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(feat_labels, fontsize=9)

    # η² on the right
    ax2 = ax.secondary_yaxis('right')
    ax2.set_yticks(range(top_n))
    ax2.set_yticklabels([f"η²={v}" for v in eta_values], fontsize=7.5, color='gray')

    # Pattern marker
    for i, pat in enumerate(pattern_labels):
        if 'Var' in pat:
            marker_color = COLORS['variable']
        else:
            marker_color = COLORS['fixed']
        ax.plot(-0.6, i, 's', color=marker_color, markersize=6,
                transform=ax.get_yaxis_transform(), clip_on=False)

    ax.set_title('IC: Top-10 Interaction Features (bet_type × outcome)',
                 fontsize=12, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Z-score', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


# ---------------------------------------------------------------------------
# Fig D: η² with SM overlay + Cohen's benchmarks
# ---------------------------------------------------------------------------
def fig_d_improved(ic_anova: Dict, sm_anova: Optional[Dict], save_path: Path):
    """η² interaction — IC vs SM on same plot with Cohen's benchmarks."""
    fig, ax = plt.subplots(figsize=(12, 5))

    datasets = [('ic', ic_anova)]
    if sm_anova:
        datasets.append(('sm', sm_anova))

    max_layer = 0
    for paradigm, anova_data in datasets:
        layer_key = 'layer_results' if 'layer_results' in anova_data else 'layer_summary'
        layer_data = anova_data[layer_key]

        layers, medians, p75s, maxes = [], [], [], []
        for lr in layer_data:
            feat_key = 'significant_features' if 'significant_features' in lr else 'top_features'
            results_to_use = lr.get('all_results', lr.get(feat_key, []))
            if not results_to_use:
                continue
            eta_sq = [r['interaction_eta_sq'] for r in results_to_use]
            if not eta_sq:
                continue
            layers.append(lr['layer'])
            medians.append(np.median(eta_sq))
            p75s.append(np.percentile(eta_sq, 75))
            maxes.append(np.max(eta_sq))

        if not layers:
            continue

        max_layer = max(max_layer, max(layers))
        color = COLORS[paradigm]
        label = MODEL_LABELS[paradigm]

        ax.plot(layers, medians, '-', color=color, lw=2.0, label=f'{label} (median)')
        ax.plot(layers, maxes, ':', color=color, lw=1.2, alpha=0.7, label=f'{label} (max)')
        ax.fill_between(layers, medians, p75s, alpha=0.15, color=color)

    # Cohen's benchmarks
    for val, label_text in [(ETA_SQ_LARGE, 'Large'), (ETA_SQ_MEDIUM, 'Medium'), (ETA_SQ_SMALL, 'Small')]:
        ax.axhline(y=val, color='gray', linestyle='--', alpha=0.4, lw=0.8)
        ax.text(max_layer + 0.5, val, label_text, fontsize=7.5, color='gray', va='bottom')

    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('η² (Interaction Effect Size)', fontsize=11)
    ax.set_title('Betting Condition × Outcome Interaction: IC vs SM',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left', ncol=2)
    ax.grid(axis='y', alpha=0.3)

    ax.text(0.98, 0.02,
            "η² = fraction of variance in SAE feature activation\n"
            "explained by the interaction of betting type and outcome.\n"
            "Higher η² → the feature responds differently to bankruptcy\n"
            "depending on whether betting was variable or fixed.",
            transform=ax.transAxes, fontsize=7.5, ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.85))

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


# ---------------------------------------------------------------------------
# Fig E: Interaction % — IC vs SM overlay
# ---------------------------------------------------------------------------
def fig_e_improved(ic_anova: Dict, sm_anova: Optional[Dict], save_path: Path):
    """Interaction % per layer — IC vs SM with annotations."""
    fig, ax = plt.subplots(figsize=(12, 5))

    datasets = [('ic', ic_anova)]
    if sm_anova:
        datasets.append(('sm', sm_anova))

    for paradigm, anova_data in datasets:
        layer_key = 'layer_results' if 'layer_results' in anova_data else 'layer_summary'
        layer_data = anova_data[layer_key]

        layers, pcts, n_tested_list = [], [], []
        for lr in layer_data:
            n_tested = lr['n_tested']
            n_sig = lr['n_significant']
            if n_tested > 0:
                layers.append(lr['layer'])
                pcts.append(100.0 * n_sig / n_tested)
                n_tested_list.append(n_tested)

        if not layers:
            continue

        color = COLORS[paradigm]
        ax.plot(layers, pcts, '-o', color=color, lw=1.8, markersize=4,
                label=MODEL_LABELS[paradigm])

        # Peak (exclude artifact layers with n < 10)
        valid_pcts = list(pcts)
        for i, n in enumerate(n_tested_list):
            if n < 10:
                valid_pcts[i] = 0

        peak_idx = int(np.argmax(valid_pcts))
        ax.plot(layers[peak_idx], pcts[peak_idx], 'o', color=color,
                markersize=8, markeredgecolor='white', markeredgewidth=1.5, zorder=5)
        ax.annotate(f"L{layers[peak_idx]} ({pcts[peak_idx]:.1f}%)",
                    xy=(layers[peak_idx], pcts[peak_idx]),
                    xytext=(5, 8), textcoords='offset points',
                    fontsize=8, color=color, fontweight='bold')

    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('Features with Significant\nBetting × Outcome Interaction (%)', fontsize=10)
    ax.set_title('What Fraction of SAE Features Encode Betting × Outcome Interaction?\n'
                 'Investment Choice vs Slot Machine',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    ax.text(0.98, 0.55,
            "A significant interaction means:\n"
            "the feature's response to bankruptcy vs. safe\n"
            "outcomes differs depending on whether the\n"
            "model had betting autonomy (variable) or not (fixed).\n\n"
            "Higher % → more features in this layer jointly\n"
            "encode both betting condition and outcome.",
            transform=ax.transAxes, fontsize=7.5, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.85))

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


# ---------------------------------------------------------------------------
# Fig F: Constraint Effect Improved (c30 vs c50 with layer profile)
# ---------------------------------------------------------------------------
def fig_f_improved(welch_data: Dict, save_path: Path):
    """Constraint effect: c30 vs c50 significant feature count with context."""
    layer_results = welch_data.get('layer_results', [])
    if not layer_results:
        return

    fig = plt.figure(figsize=(12, 7))
    gs = GridSpec(2, 1, hspace=0.35, height_ratios=[2, 1])

    # Top: stacked bar
    ax1 = fig.add_subplot(gs[0, 0])
    layers = [lr['layer'] for lr in layer_results]
    c30_up = [lr['n_c30_up'] for lr in layer_results]
    c50_up = [lr['n_c50_up'] for lr in layer_results]
    total_sig = [lr['n_sig_with_d'] for lr in layer_results]

    x = np.arange(len(layers))
    ax1.bar(x, c30_up, 0.7, label='c30 > c50', color=COLORS['c30'], alpha=0.8)
    ax1.bar(x, c50_up, 0.7, bottom=c30_up, label='c50 > c30', color=COLORS['c50'], alpha=0.8)

    # Peak annotation
    peak_idx = int(np.argmax(total_sig))
    ax1.annotate(f"L{layers[peak_idx]}: {total_sig[peak_idx]} features",
                 xy=(peak_idx, total_sig[peak_idx]),
                 xytext=(10, 10), textcoords='offset points',
                 fontsize=8, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', lw=0.8))

    ax1.set_xticks(x[::2])
    ax1.set_xticklabels([layers[i] for i in range(0, len(layers), 2)], fontsize=8)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('# Significant Features\n(FDR + |d| >= 0.3)', fontsize=10)
    ax1.set_title('IC: Which Layers Encode Constraint Level (c30 vs c50)?',
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3)

    # Bottom: proportion (c30_up fraction)
    ax2 = fig.add_subplot(gs[1, 0])
    c30_frac = []
    for c3, c5 in zip(c30_up, c50_up):
        total = c3 + c5
        c30_frac.append(c3 / total * 100 if total > 0 else 50)

    ax2.bar(x, c30_frac, 0.7, color=COLORS['c30'], alpha=0.6)
    ax2.axhline(y=50, color='gray', ls='--', alpha=0.5, lw=1)
    ax2.text(len(layers) - 1, 52, 'Balanced', fontsize=7, color='gray')

    ax2.set_xticks(x[::2])
    ax2.set_xticklabels([layers[i] for i in range(0, len(layers), 2)], fontsize=8)
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('% c30-up\n(of significant)', fontsize=10)
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', alpha=0.3)

    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


# ---------------------------------------------------------------------------
# CP Fig: Cross-Paradigm AUC Comparison (improved)
# ---------------------------------------------------------------------------
def cp_fig_improved(ic_clf: Dict, sm_clf: Dict, save_path: Path):
    """SM vs IC AUC comparison — improved with error-rate view."""
    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 2, wspace=0.3)

    # Panel 1: AUC overlay
    ax1 = fig.add_subplot(gs[0, 0])
    for paradigm, clf, color in [('sm', sm_clf, COLORS['sm']), ('ic', ic_clf, COLORS['ic'])]:
        results = [r for r in clf.get('all_games', []) if not r.get('skipped', False)]
        if not results:
            continue
        layers = [r['layer'] for r in results]
        aucs = np.array([r['auc_mean'] for r in results])
        stds = np.array([r['auc_std'] for r in results])

        ax1.plot(layers, aucs, '-', color=color, lw=2.0, label=MODEL_LABELS[paradigm])
        ax1.fill_between(layers, aucs - stds, np.minimum(aucs + stds, 1.0),
                         alpha=0.15, color=color)

    ax1.axhline(y=0.5, color='gray', ls=':', alpha=0.5)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('AUC-ROC')
    ax1.set_title('Classification AUC', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3)

    # Panel 2: 1-AUC log scale
    ax2 = fig.add_subplot(gs[0, 1])
    for paradigm, clf, color in [('sm', sm_clf, COLORS['sm']), ('ic', ic_clf, COLORS['ic'])]:
        results = [r for r in clf.get('all_games', []) if not r.get('skipped', False)]
        if not results:
            continue
        layers = [r['layer'] for r in results]
        error = [max(1 - r['auc_mean'], 1e-6) for r in results]

        ax2.semilogy(layers, error, '-o', color=color, lw=1.8, markersize=3,
                     label=MODEL_LABELS[paradigm])

    ax2.axhline(y=0.01, color='gray', ls=':', alpha=0.5, lw=0.8)
    ax2.text(41.5, 0.012, '1%', fontsize=7, color='gray')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Classification Error (1 - AUC)')
    ax2.set_title('Error Rate (log scale)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_ylim(1e-5, 2e-1)

    fig.suptitle('Cross-Paradigm: SAE Classification Comparison (Gemma-2-9B)',
                 fontsize=14, fontweight='bold', y=1.02)

    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Generate improved IC SAE figures')
    parser.add_argument('--only', nargs='+',
                        choices=['fig_a', 'fig_b', 'fig_d', 'fig_e', 'fig_f', 'cp', 'all'],
                        default=['all'])
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = set(args.only)
    do_all = 'all' in targets

    print("Loading JSON results...")

    ic_clf = load_latest(IC_RESULTS_DIR, 'ic_classification')
    ic_anova = load_latest(IC_RESULTS_DIR, 'ic_anova')
    welch_data = load_latest(IC_RESULTS_DIR, 'ic_constraint_welch')
    sm_clf = load_latest(SM_RESULTS_DIR, 'classification_gemma')
    sm_anova = load_latest(SM_RESULTS_DIR, 'anova_gemma')

    for name, data in [('ic_clf', ic_clf), ('ic_anova', ic_anova), ('welch', welch_data),
                       ('sm_clf', sm_clf), ('sm_anova', sm_anova)]:
        print(f"  {name}: {'loaded' if data else 'NOT FOUND'}")

    if do_all or 'fig_a' in targets:
        if ic_clf:
            print("\nFig A: Error rate log scale + active features...")
            fig_a_improved(ic_clf, OUTPUT_DIR / 'ic_fig_a_error_rate_improved.png')

    if do_all or 'fig_b' in targets:
        if ic_anova:
            print("\nFig B: Enhanced heatmap with pattern labels...")
            fig_b_improved(ic_anova, OUTPUT_DIR / 'ic_fig_b_heatmap_improved.png')

    if do_all or 'fig_d' in targets:
        if ic_anova:
            print("\nFig D: η² with SM overlay + Cohen's benchmarks...")
            fig_d_improved(ic_anova, sm_anova, OUTPUT_DIR / 'ic_fig_d_eta_improved.png')

    if do_all or 'fig_e' in targets:
        if ic_anova:
            print("\nFig E: Interaction % IC vs SM...")
            fig_e_improved(ic_anova, sm_anova, OUTPUT_DIR / 'ic_fig_e_interaction_improved.png')

    if do_all or 'fig_f' in targets:
        if welch_data:
            print("\nFig F: Constraint effect improved...")
            fig_f_improved(welch_data, OUTPUT_DIR / 'ic_fig_f_constraint_improved.png')

    if do_all or 'cp' in targets:
        if ic_clf and sm_clf:
            print("\nCP: Cross-paradigm AUC comparison improved...")
            cp_fig_improved(ic_clf, sm_clf, OUTPUT_DIR / 'cp_auc_comparison_improved.png')

    print(f"\nAll figures saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
