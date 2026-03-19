#!/usr/bin/env python3
"""
Comprehensive SAE Feature Analysis & Visualization
===================================================
Runs all 6 claim analyses on pre-extracted NPZ features and generates
publication-quality figures.

Claims:
  1. Variable vs Fixed neural representation differences
  2. Prompt component causal effects on risk features
  3. Dose-response (prompt complexity → risk feature activation)
  4. Mechanistic pathway (Variable → risk feature → bankruptcy)
  5. LLaMA vs Gemma encoding strategy comparison
  6. Cross-paradigm generalization (placeholder)

Usage:
    python run_all_analyses.py
    python run_all_analyses.py --models llama
    python run_all_analyses.py --models gemma --layers 25 30 35 40
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_ROOT = Path("/home/jovyan/beomi/llm-addiction-data/hf-dataset")
SAE_DIR = DATA_ROOT / "sae_patching" / "corrected_sae_analysis"
GAME_DIR = DATA_ROOT / "slot_machine"

OUTPUT_DIR = Path("/home/jovyan/beomi/llm-addiction-data/sae_condition_comparison")
FIGURE_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"

LLAMA_LAYERS = list(range(1, 32))  # 1-31
GEMMA_LAYERS = [5, 10, 15, 20, 25, 30, 35, 40]  # sampled layers

GAME_FILES = {
    "llama": GAME_DIR / "llama" / "final_llama_20251004_021106.json",
    "gemma": GAME_DIR / "gemma" / "final_gemma_20251004_172426.json",
}

# Prompt component definitions (5 binary variables, 32 combos)
COMPONENTS = ['G', 'M', 'H', 'W', 'P']
COMPONENT_NAMES = {
    'G': 'Goal-setting', 'M': 'Maximize reward',
    'H': 'Hidden patterns', 'W': 'Win multiplier', 'P': 'Win rate'
}


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_game_results(model: str) -> List[Dict]:
    """Load game results JSON."""
    with open(GAME_FILES[model]) as f:
        data = json.load(f)
    return data['results']


def parse_prompt_combo(combo_str: str) -> Dict[str, bool]:
    """Parse prompt combo string to component booleans."""
    if combo_str == 'BASE' or combo_str == '':
        return {c: False for c in COMPONENTS}
    parts = set(combo_str.split('_')) if '_' in combo_str else set(combo_str)
    return {c: (c in parts) for c in COMPONENTS}


def get_game_metadata(games: List[Dict]) -> Dict[str, np.ndarray]:
    """Extract metadata arrays aligned with NPZ game order."""
    n = len(games)
    bet_types = np.array([g['bet_type'] for g in games])
    outcomes = np.array([g['outcome'] for g in games])

    prompt_combos = []
    for g in games:
        combo = g.get('prompt_combo', 'BASE')
        prompt_combos.append(combo)
    prompt_combos = np.array(prompt_combos)

    # Parse components
    component_flags = {c: np.zeros(n, dtype=bool) for c in COMPONENTS}
    complexity = np.zeros(n, dtype=int)
    for i, combo in enumerate(prompt_combos):
        parsed = parse_prompt_combo(combo)
        for c in COMPONENTS:
            component_flags[c][i] = parsed[c]
        complexity[i] = sum(parsed.values())

    return {
        'bet_types': bet_types,
        'outcomes': outcomes,
        'prompt_combos': prompt_combos,
        'component_flags': component_flags,
        'complexity': complexity,
    }


def load_npz_features(model: str, layer: int) -> Optional[np.ndarray]:
    """Load SAE features for a given model and layer."""
    npz_path = SAE_DIR / model / f"layer_{layer}_features.npz"
    if not npz_path.exists():
        return None
    data = np.load(npz_path, allow_pickle=True)
    return data['features']  # (3200, n_features)


# ---------------------------------------------------------------------------
# Statistical Utilities
# ---------------------------------------------------------------------------
def welch_ttest(a, b):
    if len(a) < 2 or len(b) < 2:
        return 0.0, 1.0
    if np.std(a) == 0 and np.std(b) == 0:
        return 0.0, 1.0
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(t), float(p) if not np.isnan(p) else 1.0


def cohens_d(a, b):
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0
    v1, v2 = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2))
    if pooled == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled


def fdr_correction(p_values, alpha=0.05):
    if len(p_values) == 0:
        return np.array([]), np.array([])
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
    return reject, pvals_corrected


# ---------------------------------------------------------------------------
# Analysis 1: Variable vs Fixed (Claim 1)
# ---------------------------------------------------------------------------
def analyze_variable_vs_fixed(model: str, meta: Dict, layers: List[int],
                               min_d: float = 0.3) -> Dict:
    """Analyze per-feature Variable vs Fixed differences across layers."""
    print(f"\n{'='*60}")
    print(f"[Claim 1] Variable vs Fixed — {model.upper()}")
    print(f"{'='*60}")

    var_mask = meta['bet_types'] == 'variable'
    fix_mask = meta['bet_types'] == 'fixed'

    all_results = []

    for layer in layers:
        features = load_npz_features(model, layer)
        if features is None:
            continue

        n_feat = features.shape[1]
        # Subsample features for speed — test every feature but only store stats
        p_values = []
        d_values = []

        var_feats = features[var_mask]
        fix_feats = features[fix_mask]

        for fid in range(n_feat):
            v = var_feats[:, fid]
            f = fix_feats[:, fid]
            if np.std(v) == 0 and np.std(f) == 0:
                p_values.append(1.0)
                d_values.append(0.0)
                continue
            _, p = welch_ttest(v, f)
            d = cohens_d(v, f)
            p_values.append(p)
            d_values.append(d)

        # FDR
        p_arr = np.array(p_values)
        d_arr = np.array(d_values)
        reject, p_fdr = fdr_correction(p_arr)

        sig_mask = reject & (np.abs(d_arr) >= min_d)
        n_sig = sig_mask.sum()
        n_var_higher = (sig_mask & (d_arr > 0)).sum()
        n_fix_higher = (sig_mask & (d_arr < 0)).sum()

        print(f"  Layer {layer:2d}: {n_feat:6d} features, "
              f"{n_sig:4d} sig (|d|≥{min_d}), "
              f"Var↑={n_var_higher}, Fix↑={n_fix_higher}")

        all_results.append({
            'layer': layer,
            'n_features': n_feat,
            'n_significant': int(n_sig),
            'n_var_higher': int(n_var_higher),
            'n_fix_higher': int(n_fix_higher),
            'p_values': p_arr,
            'd_values': d_arr,
            'reject': reject,
            'p_fdr': p_fdr,
        })

    return {'model': model, 'results': all_results, 'min_d': min_d}


# ---------------------------------------------------------------------------
# Analysis 2: Outcome (Bankrupt vs Safe) — for cross-reference (Claim 4)
# ---------------------------------------------------------------------------
def analyze_bankrupt_vs_safe(model: str, meta: Dict, layers: List[int],
                              min_d: float = 0.3) -> Dict:
    """Analyze per-feature Bankrupt vs Safe differences."""
    print(f"\n{'='*60}")
    print(f"[Claim 4 prep] Bankrupt vs Safe — {model.upper()}")
    print(f"{'='*60}")

    bk_mask = meta['outcomes'] == 'bankruptcy'
    safe_mask = meta['outcomes'] == 'voluntary_stop'
    print(f"  Bankrupt: {bk_mask.sum()}, Safe: {safe_mask.sum()}")

    all_results = []

    for layer in layers:
        features = load_npz_features(model, layer)
        if features is None:
            continue

        n_feat = features.shape[1]
        bk_feats = features[bk_mask]
        safe_feats = features[safe_mask]

        p_values, d_values = [], []
        for fid in range(n_feat):
            b = bk_feats[:, fid]
            s = safe_feats[:, fid]
            if np.std(b) == 0 and np.std(s) == 0:
                p_values.append(1.0)
                d_values.append(0.0)
                continue
            _, p = welch_ttest(b, s)
            d = cohens_d(b, s)
            p_values.append(p)
            d_values.append(d)

        p_arr = np.array(p_values)
        d_arr = np.array(d_values)
        reject, p_fdr = fdr_correction(p_arr)

        sig_mask = reject & (np.abs(d_arr) >= min_d)
        n_risky = (sig_mask & (d_arr > 0)).sum()
        n_safe = (sig_mask & (d_arr < 0)).sum()

        print(f"  Layer {layer:2d}: {sig_mask.sum():4d} sig — "
              f"Risky(BK↑)={n_risky}, Safe(BK↓)={n_safe}")

        all_results.append({
            'layer': layer,
            'n_features': n_feat,
            'p_values': p_arr,
            'd_values': d_arr,
            'reject': reject,
            'p_fdr': p_fdr,
        })

    return {'model': model, 'results': all_results}


# ---------------------------------------------------------------------------
# Analysis for Claim 2 & 3: Prompt Component & Complexity
# ---------------------------------------------------------------------------
def analyze_prompt_effects(model: str, meta: Dict, layers: List[int]) -> Dict:
    """Analyze prompt component effects and complexity dose-response."""
    print(f"\n{'='*60}")
    print(f"[Claim 2&3] Prompt Component & Complexity — {model.upper()}")
    print(f"{'='*60}")

    bk_mask = meta['outcomes'] == 'bankruptcy'
    component_results = {c: [] for c in COMPONENTS}
    complexity_results = []

    for layer in layers:
        features = load_npz_features(model, layer)
        if features is None:
            continue

        n_feat = features.shape[1]

        # --- Claim 2: Component effects ---
        for comp in COMPONENTS:
            has_comp = meta['component_flags'][comp]
            # Mean activation of top-activating features for games with/without component
            # Split further by outcome
            groups = {
                'with_bk': features[has_comp & bk_mask],
                'with_safe': features[has_comp & ~bk_mask],
                'without_bk': features[~has_comp & bk_mask],
                'without_safe': features[~has_comp & ~bk_mask],
            }

            # For efficiency: compute mean across all features per group
            means = {k: v.mean(axis=0) if len(v) > 0 else np.zeros(n_feat)
                     for k, v in groups.items()}

            # Interaction metric: (with_bk - with_safe) - (without_bk - without_safe)
            interaction = (means['with_bk'] - means['with_safe']) - \
                          (means['without_bk'] - means['without_safe'])

            # Top interaction features
            top_idx = np.argsort(np.abs(interaction))[-100:]
            mean_abs_interaction = np.mean(np.abs(interaction[top_idx]))

            component_results[comp].append({
                'layer': layer,
                'mean_abs_interaction_top100': float(mean_abs_interaction),
                'group_means_global': {k: float(v.mean()) for k, v in means.items()},
                'n_with': int(has_comp.sum()),
                'n_without': int((~has_comp).sum()),
            })

        # --- Claim 3: Complexity dose-response ---
        complexity = meta['complexity']
        levels = sorted(set(complexity))

        level_data = {}
        for lev in levels:
            lev_mask = complexity == lev
            lev_feats = features[lev_mask]
            # Mean activation across top-variance features
            if len(lev_feats) > 0:
                # Use mean of all feature means as a summary
                mean_act = float(lev_feats.mean())
                # Also bankruptcy rate for this level
                bk_rate = float(bk_mask[lev_mask].sum() / lev_mask.sum()) if lev_mask.sum() > 0 else 0.0
                level_data[int(lev)] = {
                    'mean_activation': mean_act,
                    'bankruptcy_rate': bk_rate,
                    'n_games': int(lev_mask.sum()),
                    'n_bankrupt': int(bk_mask[lev_mask].sum()),
                }

        complexity_results.append({
            'layer': layer,
            'levels': level_data,
        })

        print(f"  Layer {layer:2d}: components analyzed, complexity levels={len(levels)}")

    return {
        'model': model,
        'component_results': component_results,
        'complexity_results': complexity_results,
    }


# ---------------------------------------------------------------------------
# Visualization Functions
# ---------------------------------------------------------------------------
def setup_style():
    """Set publication-quality plot style."""
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def fig1_volcano_plot(vf_data: Dict, save_path: Path):
    """
    Fig 1: Volcano Plot — Variable vs Fixed per layer.
    X = Cohen's d, Y = -log10(p_FDR), colored by layer.
    """
    model = vf_data['model']
    results = vf_data['results']

    fig, ax = plt.subplots(figsize=(10, 7))

    cmap = plt.cm.viridis
    layers = [r['layer'] for r in results]
    norm = plt.Normalize(min(layers), max(layers))

    for r in results:
        layer = r['layer']
        d = r['d_values']
        p_fdr = r['p_fdr']

        # Only plot features with some effect
        mask = np.abs(d) > 0.05
        if mask.sum() == 0:
            continue

        neg_log_p = -np.log10(np.clip(p_fdr[mask], 1e-300, 1.0))
        ax.scatter(d[mask], neg_log_p, s=3, alpha=0.15,
                   c=[cmap(norm(layer))]*mask.sum(), rasterized=True)

    # Reference lines
    ax.axhline(-np.log10(0.05), color='gray', ls='--', lw=0.8, label='FDR=0.05')
    ax.axvline(-0.3, color='red', ls=':', lw=0.8, alpha=0.5)
    ax.axvline(0.3, color='red', ls=':', lw=0.8, alpha=0.5, label='|d|=0.3')
    ax.axvline(0, color='gray', ls='-', lw=0.5, alpha=0.3)

    # Shade significant regions
    ylim = ax.get_ylim()
    ax.fill_betweenx([(-np.log10(0.05)), ylim[1]], 0.3, ax.get_xlim()[1],
                      alpha=0.05, color='red', label='Sig. Variable↑')
    ax.fill_betweenx([(-np.log10(0.05)), ylim[1]], ax.get_xlim()[0], -0.3,
                      alpha=0.05, color='blue', label='Sig. Fixed↑')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Layer', shrink=0.8)

    ax.set_xlabel("Cohen's d (positive = higher in Variable)")
    ax.set_ylabel("-log₁₀(p_FDR)")
    ax.set_title(f"Volcano Plot: Variable vs Fixed — {model.upper()}")
    ax.legend(loc='upper left', fontsize=9, framealpha=0.8)

    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def fig1b_feature_count_bar(vf_data: Dict, save_path: Path):
    """
    Fig 1b: Significant feature count per layer, stacked by direction.
    """
    model = vf_data['model']
    results = vf_data['results']

    layers = [r['layer'] for r in results]
    var_higher = [r['n_var_higher'] for r in results]
    fix_higher = [r['n_fix_higher'] for r in results]

    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(layers))
    w = 0.7

    ax.bar(x, var_higher, w, label='Variable↑', color='#e74c3c', alpha=0.8)
    ax.bar(x, fix_higher, w, bottom=var_higher, label='Fixed↑', color='#3498db', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([f'L{l}' for l in layers], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Significant Features (FDR < 0.05, |d| ≥ 0.3)')
    ax.set_title(f'Significant Feature Count by Layer — {model.upper()}')
    ax.legend()

    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def fig2_pathway_scatter(vf_data: Dict, bs_data: Dict, save_path: Path):
    """
    Fig 2: Mechanistic Pathway Scatter.
    X = Cohen's d (Bankrupt vs Safe), Y = Cohen's d (Variable vs Fixed).
    Q1 (++): Risk Amplification, Q3 (--): Protective.
    """
    model = vf_data['model']

    # Collect all (d_condition, d_outcome) pairs across matching layers
    all_d_cond = []
    all_d_outc = []
    all_layers = []

    vf_by_layer = {r['layer']: r for r in vf_data['results']}
    bs_by_layer = {r['layer']: r for r in bs_data['results']}

    for layer in vf_by_layer:
        if layer not in bs_by_layer:
            continue
        vf = vf_by_layer[layer]
        bs = bs_by_layer[layer]

        # Both must have same n_features
        n = min(len(vf['d_values']), len(bs['d_values']))
        d_cond = vf['d_values'][:n]
        d_outc = bs['d_values'][:n]

        # Only plot features significant in at least one analysis
        sig_either = vf['reject'][:n] | bs['reject'][:n]
        meaningful = sig_either & ((np.abs(d_cond) > 0.2) | (np.abs(d_outc) > 0.2))

        all_d_cond.append(d_cond[meaningful])
        all_d_outc.append(d_outc[meaningful])
        all_layers.append(np.full(meaningful.sum(), layer))

    if not all_d_cond:
        print("  No data for pathway scatter")
        return

    d_cond = np.concatenate(all_d_cond)
    d_outc = np.concatenate(all_d_outc)
    layer_arr = np.concatenate(all_layers)

    fig, ax = plt.subplots(figsize=(9, 9))

    # Shade quadrants
    lim = max(np.percentile(np.abs(d_cond), 99), np.percentile(np.abs(d_outc), 99)) * 1.1
    ax.fill_between([0, lim], 0, lim, alpha=0.06, color='red', label='Q1: Risk Amplification')
    ax.fill_between([-lim, 0], -lim, 0, alpha=0.06, color='blue', label='Q3: Protective')
    ax.fill_between([0, lim], -lim, 0, alpha=0.03, color='gray')
    ax.fill_between([-lim, 0], 0, lim, alpha=0.03, color='gray')

    scatter = ax.scatter(d_outc, d_cond, s=5, alpha=0.25, c=layer_arr,
                         cmap='viridis', rasterized=True)

    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)

    # Count quadrants
    q1 = ((d_cond > 0.3) & (d_outc > 0.3)).sum()
    q3 = ((d_cond < -0.3) & (d_outc < -0.3)).sum()
    total = len(d_cond)

    ax.text(0.95, 0.95, f'Q1 (Risk Amp.): {q1}\n({100*q1/total:.1f}%)',
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.8))
    ax.text(0.05, 0.05, f'Q3 (Protective): {q3}\n({100*q3/total:.1f}%)',
            transform=ax.transAxes, ha='left', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='#cce5ff', alpha=0.8))

    cbar = plt.colorbar(scatter, ax=ax, label='Layer', shrink=0.8)

    ax.set_xlabel("Cohen's d (Bankrupt vs Safe)\n→ Risky features")
    ax.set_ylabel("Cohen's d (Variable vs Fixed)\n→ Variable-higher features")
    ax.set_title(f"Mechanistic Pathway: Condition → Feature → Outcome — {model.upper()}")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=9)

    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def fig3_component_interaction(prompt_data: Dict, save_path: Path):
    """
    Fig 3: Prompt Component × Outcome Interaction Plot (2×3 grid).
    """
    model = prompt_data['model']
    comp_results = prompt_data['component_results']

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes_flat = axes.flatten()

    for idx, comp in enumerate(COMPONENTS):
        ax = axes_flat[idx]
        layers_data = comp_results[comp]

        if not layers_data:
            ax.set_visible(False)
            continue

        # Aggregate across layers
        with_bk = np.mean([d['group_means_global']['with_bk'] for d in layers_data])
        with_safe = np.mean([d['group_means_global']['with_safe'] for d in layers_data])
        without_bk = np.mean([d['group_means_global']['without_bk'] for d in layers_data])
        without_safe = np.mean([d['group_means_global']['without_safe'] for d in layers_data])

        # Plot interaction
        x = [0, 1]
        ax.plot(x, [with_bk, with_safe], 'o-', color='#e74c3c', lw=2.5,
                markersize=8, label=f'With {comp}', zorder=3)
        ax.plot(x, [without_bk, without_safe], 's--', color='#3498db', lw=2.5,
                markersize=8, label=f'Without {comp}', zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels(['Bankrupt', 'Safe'])
        ax.set_ylabel('Mean Feature Activation')
        ax.set_title(f'{comp}: {COMPONENT_NAMES[comp]}', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

        # Annotate interaction strength
        interaction = (with_bk - with_safe) - (without_bk - without_safe)
        ax.text(0.5, 0.02, f'Interaction: {interaction:.4f}',
                transform=ax.transAxes, ha='center', fontsize=8,
                style='italic', color='gray')

    # Hide 6th subplot
    axes_flat[5].set_visible(False)

    fig.suptitle(f'Prompt Component × Outcome Interaction — {model.upper()}',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def fig4_dose_response(prompt_data: Dict, meta: Dict, save_path: Path):
    """
    Fig 4: Dose-Response — Complexity vs Neural Activation + Bankruptcy Rate.
    """
    model = prompt_data['model']
    complexity_results = prompt_data['complexity_results']

    # Aggregate across layers
    level_acts = defaultdict(list)
    level_bk = {}

    for lr in complexity_results:
        for lev_str, lev_data in lr['levels'].items():
            lev = int(lev_str)
            level_acts[lev].append(lev_data['mean_activation'])
            level_bk[lev] = lev_data['bankruptcy_rate']

    levels = sorted(level_acts.keys())
    mean_acts = [np.mean(level_acts[l]) for l in levels]
    std_acts = [np.std(level_acts[l]) for l in levels]
    bk_rates = [level_bk.get(l, 0) * 100 for l in levels]

    # Compute behavioral bankruptcy rates from metadata
    complexity = meta['complexity']
    outcomes = meta['outcomes']
    bk_rates_actual = []
    n_games_per_level = []
    for lev in levels:
        mask = complexity == lev
        n = mask.sum()
        bk = (outcomes[mask] == 'bankruptcy').sum()
        bk_rates_actual.append(100 * bk / n if n > 0 else 0)
        n_games_per_level.append(n)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Neural activation (left axis)
    color1 = '#2c3e50'
    ax1.errorbar(levels, mean_acts, yerr=std_acts, fmt='o-', color=color1,
                 lw=2.5, markersize=8, capsize=4, label='Neural Activation (mean)')
    ax1.set_xlabel('Prompt Complexity (number of components)')
    ax1.set_ylabel('Mean SAE Feature Activation', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    # Bankruptcy rate (right axis)
    ax2 = ax1.twinx()
    color2 = '#e74c3c'
    ax2.plot(levels, bk_rates_actual, 's--', color=color2, lw=2.5, markersize=8,
             label='Bankruptcy Rate (%)')
    ax2.set_ylabel('Bankruptcy Rate (%)', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    # Sample size annotations
    for i, lev in enumerate(levels):
        ax1.annotate(f'n={n_games_per_level[i]}', (lev, mean_acts[i]),
                     textcoords="offset points", xytext=(0, 12),
                     fontsize=8, ha='center', color='gray')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

    ax1.set_title(f'Dose-Response: Prompt Complexity → Neural & Behavioral — {model.upper()}')
    ax1.set_xticks(levels)

    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def fig5_model_comparison(llama_vf: Dict, gemma_vf: Dict, save_path: Path):
    """
    Fig 5: Mirror Bar Chart — LLaMA vs Gemma encoding comparison.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=False)

    for ax, data, label, color in [
        (ax1, llama_vf, 'LLaMA', '#2c3e50'),
        (ax2, gemma_vf, 'Gemma', '#e67e22'),
    ]:
        if data is None:
            ax.text(0.5, 0.5, f'{label}: No data', transform=ax.transAxes, ha='center')
            continue

        results = data['results']
        layers = [r['layer'] for r in results]
        var_h = [r['n_var_higher'] for r in results]
        fix_h = [r['n_fix_higher'] for r in results]

        x = np.arange(len(layers))
        w = 0.35

        ax.bar(x - w/2, var_h, w, label='Variable↑', color='#e74c3c', alpha=0.8)
        ax.bar(x + w/2, fix_h, w, label='Fixed↑', color='#3498db', alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([f'L{l}' for l in layers], fontsize=7, rotation=45, ha='right')
        ax.set_ylabel('# Sig. Features')
        ax.set_title(f'{label} — Significant Feature Distribution', fontweight='bold')
        ax.legend(fontsize=8)

    fig.suptitle('LLaMA vs Gemma: Layer-wise Encoding Strategy',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def fig6_summary_table(all_data: Dict, save_path: Path):
    """
    Fig 6: Summary statistics table across all claims.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')

    rows = []
    for model in ['llama', 'gemma']:
        if model not in all_data:
            continue
        d = all_data[model]

        # Claim 1
        vf = d.get('vf')
        if vf:
            total_sig = sum(r['n_significant'] for r in vf['results'])
            total_var = sum(r['n_var_higher'] for r in vf['results'])
            total_fix = sum(r['n_fix_higher'] for r in vf['results'])
            rows.append([model.upper(), 'Claim 1: Var vs Fix',
                        f'{total_sig:,}', f'{total_var:,}', f'{total_fix:,}', '—'])

        # Claim 4
        bs = d.get('bs')
        if bs:
            total_sig_bs = sum((r['reject'] & (np.abs(r['d_values']) >= 0.3)).sum()
                               for r in bs['results'])
            n_risky = sum(((r['reject']) & (r['d_values'] > 0.3)).sum() for r in bs['results'])
            n_safe = sum(((r['reject']) & (r['d_values'] < -0.3)).sum() for r in bs['results'])
            rows.append([model.upper(), 'Claim 4: BK vs Safe',
                        f'{total_sig_bs:,}', f'{n_risky:,} risky', f'{n_safe:,} safe', '—'])

        # Claim 2
        prompt = d.get('prompt')
        if prompt:
            for comp in COMPONENTS:
                comp_data = prompt['component_results'][comp]
                if comp_data:
                    mean_int = np.mean([c['mean_abs_interaction_top100'] for c in comp_data])
                    rows.append([model.upper(), f'Claim 2: {comp}({COMPONENT_NAMES[comp]})',
                                '—', '—', '—', f'{mean_int:.5f}'])

    col_labels = ['Model', 'Analysis', 'Sig. Features', 'Direction 1', 'Direction 2', 'Interaction']

    table = ax.table(cellText=rows, colLabels=col_labels,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#2c3e50')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Alternate row colors
    for i in range(1, len(rows) + 1):
        color = '#f0f0f0' if i % 2 == 0 else 'white'
        for j in range(len(col_labels)):
            table[i, j].set_facecolor(color)

    ax.set_title('Summary: SAE Feature Analysis Across Claims', fontsize=14,
                 fontweight='bold', pad=20)

    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='SAE Feature Analysis & Visualization')
    parser.add_argument('--models', nargs='+', default=['llama', 'gemma'],
                        choices=['llama', 'gemma'])
    args = parser.parse_args()

    setup_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_data = {}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for model in args.models:
        print(f"\n{'#'*70}")
        print(f"# Processing {model.upper()}")
        print(f"{'#'*70}")

        layers = LLAMA_LAYERS if model == 'llama' else GEMMA_LAYERS

        # Load game metadata
        games = load_game_results(model)
        meta = get_game_metadata(games)

        print(f"  Games: {len(games)}")
        print(f"  Bet types: {dict(Counter(meta['bet_types']))}")
        print(f"  Outcomes: {dict(Counter(meta['outcomes']))}")
        print(f"  Complexity dist: {dict(Counter(meta['complexity']))}")

        # --- Run analyses ---
        vf = analyze_variable_vs_fixed(model, meta, layers)
        bs = analyze_bankrupt_vs_safe(model, meta, layers)
        prompt = analyze_prompt_effects(model, meta, layers)

        all_data[model] = {'vf': vf, 'bs': bs, 'prompt': prompt, 'meta': meta}

        # --- Generate figures ---
        print(f"\n  Generating figures for {model.upper()}...")

        fig1_volcano_plot(vf, FIGURE_DIR / f'fig1_volcano_{model}.png')
        fig1b_feature_count_bar(vf, FIGURE_DIR / f'fig1b_feature_count_{model}.png')
        fig2_pathway_scatter(vf, bs, FIGURE_DIR / f'fig2_pathway_{model}.png')
        fig3_component_interaction(prompt, FIGURE_DIR / f'fig3_component_{model}.png')
        fig4_dose_response(prompt, meta, FIGURE_DIR / f'fig4_dose_response_{model}.png')

    # --- Cross-model figures ---
    if 'llama' in all_data and 'gemma' in all_data:
        print(f"\n  Generating cross-model comparison...")
        fig5_model_comparison(
            all_data['llama']['vf'], all_data['gemma']['vf'],
            FIGURE_DIR / 'fig5_model_comparison.png'
        )
    elif len(all_data) == 1:
        model = list(all_data.keys())[0]
        fig5_model_comparison(
            all_data[model]['vf'], None,
            FIGURE_DIR / 'fig5_model_comparison.png'
        )

    # --- Summary table ---
    fig6_summary_table(all_data, FIGURE_DIR / 'fig6_summary_table.png')

    # --- Save numerical results ---
    summary = {}
    for model, data in all_data.items():
        vf = data['vf']
        bs = data['bs']
        model_summary = {
            'variable_vs_fixed': [{
                'layer': r['layer'],
                'n_significant': r['n_significant'],
                'n_var_higher': r['n_var_higher'],
                'n_fix_higher': r['n_fix_higher'],
            } for r in vf['results']],
            'bankrupt_vs_safe': [{
                'layer': r['layer'],
                'n_risky': int((r['reject'] & (r['d_values'] > 0.3)).sum()),
                'n_safe': int((r['reject'] & (r['d_values'] < -0.3)).sum()),
            } for r in bs['results']],
        }
        summary[model] = model_summary

    results_file = RESULTS_DIR / f'analysis_summary_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved: {results_file}")

    print(f"\n{'='*70}")
    print(f"ALL ANALYSES COMPLETE")
    print(f"{'='*70}")
    print(f"Figures: {FIGURE_DIR}")
    print(f"Results: {RESULTS_DIR}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
