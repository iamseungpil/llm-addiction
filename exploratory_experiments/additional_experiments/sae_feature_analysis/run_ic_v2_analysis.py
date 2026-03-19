#!/usr/bin/env python3
"""
IC V2 Within-Model Analysis + Improved Figures
================================================
Mirrors run_v2_within_model_analysis.py but for Investment Choice (Gemma only).

V2 data: /home/jovyan/beomi/llm-addiction-data/sae_features_v2/investment_choice/gemma/
Output:  results/ic_v2/

Analyses:
1. Classification: 5-fold LogReg, bankruptcy vs safe (all, variable, fixed, c10, c30, c50)
2. Two-Way ANOVA: bet_type × outcome interaction per feature
3. Constraint effect: c10 vs c30 vs c50 comparison
4. Figures: Fig A (AUC), B (heatmap raw), D (eta²), E (interaction ratio), F (constraint)
"""

import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter

from scipy.stats import f_oneway, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# ===========================================================================
# Configuration
# ===========================================================================
IC_V2_DIR = Path("/home/jovyan/beomi/llm-addiction-data/sae_features_v2/investment_choice/gemma")
SM_V2_RESULTS = Path("/home/jovyan/llm-addiction/exploratory_experiments/additional_experiments/"
                     "sae_feature_analysis/results/within_model_v2/json")

OUTPUT_DIR = Path("/home/jovyan/llm-addiction/exploratory_experiments/additional_experiments/"
                  "sae_feature_analysis/results/ic_v2")
FIG_DIR = OUTPUT_DIR / "figures"
JSON_DIR = OUTPUT_DIR / "json"

LAYERS = list(range(42))
MIN_ACTIVATION_RATE = 0.01
CV_FOLDS = 5
C_REG = 1.0
FDR_ALPHA = 0.05
N_PERMUTATIONS = 1000

COLORS = {"ic": "#27ae60", "sm": "#c0392b", "all": "#2c3e50",
          "variable": "#e74c3c", "fixed": "#3498db",
          "c10": "#2ecc71", "c30": "#f39c12", "c50": "#8e44ad"}

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


def setup_logging():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)


def load_layer(layer):
    path = IC_V2_DIR / f"layer_{layer}_features.npz"
    if not path.exists():
        return None
    return np.load(path, allow_pickle=True)


def filter_sparse(features, min_rate=0.01):
    """Filter features with >min_rate activation rate."""
    active_rate = (features > 0).mean(axis=0)
    active_mask = active_rate > min_rate
    active_idx = np.where(active_mask)[0]
    return features[:, active_mask], active_idx


def run_classification_subset(features, labels, n_folds=5, C=1.0):
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos < n_folds or n_neg < n_folds:
        return {'auc_mean': 0.5, 'auc_std': 0.0, 'accuracy_mean': 0.5, 'f1_mean': 0.0,
                'n_pos': int(n_pos), 'n_neg': int(n_neg), 'skipped': True}

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    aucs, accs, f1s = [], [], []
    coefs = np.zeros(features.shape[1])

    for train_idx, test_idx in skf.split(features, labels):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(features[train_idx])
        X_test = scaler.transform(features[test_idx])
        clf = LogisticRegression(C=C, penalty='l2', solver='lbfgs', max_iter=1000,
                                 class_weight='balanced', random_state=42)
        clf.fit(X_train, labels[train_idx])
        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred = clf.predict(X_test)
        aucs.append(roc_auc_score(labels[test_idx], y_prob))
        accs.append(accuracy_score(labels[test_idx], y_pred))
        f1s.append(f1_score(labels[test_idx], y_pred, zero_division=0))
        coefs += clf.coef_[0]

    coefs /= n_folds
    top_k = min(50, features.shape[1])
    top_idx = np.argsort(np.abs(coefs))[-top_k:][::-1]

    return {
        'auc_mean': float(np.mean(aucs)), 'auc_std': float(np.std(aucs)),
        'auc_folds': [float(a) for a in aucs],
        'accuracy_mean': float(np.mean(accs)), 'f1_mean': float(np.mean(f1s)),
        'n_pos': int(n_pos), 'n_neg': int(n_neg), 'n_features': features.shape[1],
        'skipped': False,
        'top_features': [{'feature_idx': int(top_idx[i]), 'coef': float(coefs[top_idx[i]])} for i in range(top_k)],
    }


def run_permutation_test(features, labels, observed_auc, n_perm=1000):
    perm_aucs = []
    for i in range(n_perm):
        perm_labels = np.random.permutation(labels)
        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=i)
        fold_aucs = []
        for train_idx, test_idx in skf.split(features, perm_labels):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(features[train_idx])
            X_test = scaler.transform(features[test_idx])
            clf = LogisticRegression(C=C_REG, penalty='l2', solver='lbfgs', max_iter=500,
                                     class_weight='balanced', random_state=42)
            clf.fit(X_train, perm_labels[train_idx])
            fold_aucs.append(roc_auc_score(perm_labels[test_idx], clf.predict_proba(X_test)[:, 1]))
        perm_aucs.append(np.mean(fold_aucs))
    perm_aucs = np.array(perm_aucs)
    return {
        'p_value': float((perm_aucs >= observed_auc).mean()),
        'observed_auc': float(observed_auc),
        'perm_mean': float(perm_aucs.mean()),
        'perm_std': float(perm_aucs.std()),
        'perm_95': float(np.percentile(perm_aucs, 95)),
    }


# ===========================================================================
# Main Analysis
# ===========================================================================
def main():
    logger = setup_logging()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Load metadata from layer 0
    d0 = load_layer(0)
    outcomes = d0['outcomes']
    bet_types = d0['bet_types']
    bet_constraints = d0['bet_constraints']
    labels = (outcomes == 'bankruptcy').astype(int)

    var_mask = bet_types == 'variable'
    fix_mask = bet_types == 'fixed'
    c10_mask = bet_constraints == '10'
    c30_mask = bet_constraints == '30'
    c50_mask = bet_constraints == '50'

    n_total = len(labels)
    n_bk = labels.sum()
    logger.info(f"IC V2 Data: {n_total} games, {n_bk} BK ({100*n_bk/n_total:.1f}%)")
    logger.info(f"  Variable: {var_mask.sum()} (BK={labels[var_mask].sum()})")
    logger.info(f"  Fixed: {fix_mask.sum()} (BK={labels[fix_mask].sum()})")
    logger.info(f"  c10: {c10_mask.sum()} (BK={labels[c10_mask].sum()})")
    logger.info(f"  c30: {c30_mask.sum()} (BK={labels[c30_mask].sum()})")
    logger.info(f"  c50: {c50_mask.sum()} (BK={labels[c50_mask].sum()})")

    # ==========================
    # Part 1: Classification
    # ==========================
    logger.info("=" * 60)
    logger.info("Part 1: Classification")
    logger.info("=" * 60)

    clf_results = {'all': [], 'variable': [], 'fixed': [], 'c10': [], 'c30': [], 'c50': []}
    best_layer, best_auc, best_features = None, 0.0, None

    for layer in LAYERS:
        d = load_layer(layer)
        if d is None:
            continue
        raw_features = d['features']
        features, active_idx = filter_sparse(raw_features, MIN_ACTIVATION_RATE)
        n_active = features.shape[1]

        # All games
        r = run_classification_subset(features, labels)
        r['layer'] = layer
        r['n_active'] = n_active
        clf_results['all'].append(r)

        if not r['skipped'] and r['auc_mean'] > best_auc:
            best_auc, best_layer = r['auc_mean'], layer
            best_features = features.copy()

        # Subsets
        for name, mask in [('variable', var_mask), ('fixed', fix_mask),
                           ('c10', c10_mask), ('c30', c30_mask), ('c50', c50_mask)]:
            if labels[mask].sum() >= CV_FOLDS:
                rs = run_classification_subset(features[mask], labels[mask])
                rs['layer'] = layer
                clf_results[name].append(rs)

        logger.info(f"  L{layer:2d}: {raw_features.shape[1]}→{n_active} feat, AUC={r['auc_mean']:.4f}±{r['auc_std']:.4f}")

    # Permutation test on best layer
    perm_result = None
    if best_layer is not None and best_features is not None:
        logger.info(f"Permutation test on best layer L{best_layer} (AUC={best_auc:.4f})...")
        perm_result = run_permutation_test(best_features, labels, best_auc, N_PERMUTATIONS)
        perm_result['layer'] = best_layer
        logger.info(f"  p={perm_result['p_value']}, perm_mean={perm_result['perm_mean']:.4f}")

    clf_output = {
        'model': 'gemma_ic',
        'best_layer': best_layer,
        'best_auc': best_auc,
        'permutation_test': perm_result,
        'all_games': clf_results['all'],
        'variable_only': clf_results['variable'],
        'fixed_only': clf_results['fixed'],
        'c10_only': clf_results['c10'],
        'c30_only': clf_results['c30'],
        'c50_only': clf_results['c50'],
    }
    with open(JSON_DIR / f"ic_classification_{ts}.json", 'w') as f:
        json.dump(clf_output, f, cls=NumpyEncoder, indent=2)
    logger.info(f"Classification saved. Best: L{best_layer} AUC={best_auc:.4f}")

    # ==========================
    # Part 2: Two-Way ANOVA
    # ==========================
    logger.info("=" * 60)
    logger.info("Part 2: Two-Way ANOVA (bet_type × outcome)")
    logger.info("=" * 60)

    anova_results = {'layer_summary': []}

    for layer in LAYERS:
        d = load_layer(layer)
        if d is None:
            continue
        raw_features = d['features']
        features, active_idx = filter_sparse(raw_features, MIN_ACTIVATION_RATE)
        n_active = features.shape[1]

        df = pd.DataFrame({
            'bet_type': bet_types,
            'outcome': outcomes,
        })

        tested, significant = 0, 0
        top_features = []

        for j in range(n_active):
            feat_vals = features[:, j]
            if feat_vals.std() == 0:
                continue
            tested += 1

            df['feat'] = feat_vals
            try:
                model = ols('feat ~ C(bet_type) * C(outcome)', data=df).fit()
                anova_table = anova_lm(model, typ=2)

                interaction_row = None
                for idx_name in anova_table.index:
                    if 'bet_type' in idx_name and 'outcome' in idx_name:
                        interaction_row = anova_table.loc[idx_name]
                        break

                if interaction_row is None:
                    continue

                p_val = interaction_row['PR(>F)']
                ss_interaction = interaction_row['sum_sq']
                ss_total = anova_table['sum_sq'].sum()
                eta_sq = ss_interaction / ss_total if ss_total > 0 else 0

                # Group means
                group_means = {}
                for bt in ['variable', 'fixed']:
                    for oc in ['bankruptcy', 'voluntary_stop']:
                        mask = (bet_types == bt) & (outcomes == oc)
                        if mask.sum() > 0:
                            group_means[f"{bt}_{oc}"] = float(feat_vals[mask].mean())

                top_features.append({
                    'feature_idx': int(active_idx[j]),
                    'interaction_p': float(p_val),
                    'interaction_eta_sq': float(eta_sq),
                    'interaction_significant': False,  # will be set after FDR
                    'group_means': group_means,
                })
            except Exception:
                continue

        # FDR correction
        if top_features:
            p_vals = [f['interaction_p'] for f in top_features]
            reject, _, _, _ = multipletests(p_vals, alpha=FDR_ALPHA, method='fdr_bh')
            for i, r in enumerate(reject):
                top_features[i]['interaction_significant'] = bool(r)
            significant = int(reject.sum())

        # Sort by eta_sq, keep top 50
        top_features.sort(key=lambda x: x['interaction_eta_sq'], reverse=True)

        anova_results['layer_summary'].append({
            'layer': layer,
            'n_tested': tested,
            'n_significant': significant,
            'top_features': top_features[:50],
        })

        pct = 100 * significant / tested if tested > 0 else 0
        logger.info(f"  L{layer:2d}: tested={tested}, sig={significant} ({pct:.1f}%)")

    with open(JSON_DIR / f"ic_anova_{ts}.json", 'w') as f:
        json.dump(anova_results, f, cls=NumpyEncoder, indent=2)
    logger.info("ANOVA saved.")

    # ==========================
    # Part 3: Constraint effect (c10 vs c30 vs c50)
    # ==========================
    logger.info("=" * 60)
    logger.info("Part 3: Constraint Effect (c10 vs c30 vs c50)")
    logger.info("=" * 60)

    constraint_results = []

    for layer in LAYERS:
        d = load_layer(layer)
        if d is None:
            continue
        raw_features = d['features']
        features, active_idx = filter_sparse(raw_features, MIN_ACTIVATION_RATE)

        sig_features = []
        for j in range(features.shape[1]):
            vals_c10 = features[c10_mask, j]
            vals_c30 = features[c30_mask, j]
            vals_c50 = features[c50_mask, j]

            if vals_c10.std() == 0 and vals_c30.std() == 0 and vals_c50.std() == 0:
                continue

            # Kruskal-Wallis for 3 groups
            try:
                stat, p = f_oneway(vals_c10, vals_c30, vals_c50)
            except Exception:
                continue

            if p < 0.05:
                means = {'c10': float(vals_c10.mean()), 'c30': float(vals_c30.mean()), 'c50': float(vals_c50.mean())}
                # Cohen's d for c30 vs c50
                pooled_std = np.sqrt((vals_c30.var() + vals_c50.var()) / 2)
                d_val = (vals_c30.mean() - vals_c50.mean()) / pooled_std if pooled_std > 0 else 0
                sig_features.append({
                    'feature_idx': int(active_idx[j]),
                    'p': float(p),
                    'cohens_d_c30_c50': float(d_val),
                    'means': means,
                })

        # FDR
        if sig_features:
            p_vals = [f['p'] for f in sig_features]
            reject, _, _, _ = multipletests(p_vals, alpha=FDR_ALPHA, method='fdr_bh')
            sig_features = [f for f, r in zip(sig_features, reject) if r and abs(f['cohens_d_c30_c50']) >= 0.3]

        constraint_results.append({
            'layer': layer,
            'n_significant': len(sig_features),
            'features': sig_features[:20],
        })
        if sig_features:
            logger.info(f"  L{layer:2d}: {len(sig_features)} constraint-sensitive features")

    with open(JSON_DIR / f"ic_constraint_{ts}.json", 'w') as f:
        json.dump(constraint_results, f, cls=NumpyEncoder, indent=2)

    # ==========================
    # Part 4: Figures
    # ==========================
    logger.info("=" * 60)
    logger.info("Part 4: Generating Figures")
    logger.info("=" * 60)

    # Load SM V2 results for comparison
    sm_clf, sm_anova = None, None
    try:
        sm_clf_files = sorted(SM_V2_RESULTS.glob("classification_gemma_*.json"))
        sm_anova_files = sorted(SM_V2_RESULTS.glob("anova_gemma_*.json"))
        if sm_clf_files:
            with open(sm_clf_files[-1]) as f:
                sm_clf = json.load(f)
        if sm_anova_files:
            with open(sm_anova_files[-1]) as f:
                sm_anova = json.load(f)
    except Exception as e:
        logger.warning(f"Could not load SM V2 results: {e}")

    # --- Fig A: Classification AUC (IC, with subsets) ---
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={'height_ratios': [2, 1]})

    ax1 = axes[0]
    # All games
    layers_all = [r['layer'] for r in clf_results['all'] if not r.get('skipped')]
    aucs_all = [r['auc_mean'] for r in clf_results['all'] if not r.get('skipped')]
    stds_all = [r['auc_std'] for r in clf_results['all'] if not r.get('skipped')]
    ax1.plot(layers_all, aucs_all, '-o', color=COLORS['all'], lw=2.0, markersize=4, label='All games')
    ax1.fill_between(layers_all, [a-s for a,s in zip(aucs_all, stds_all)],
                     [a+s for a,s in zip(aucs_all, stds_all)], alpha=0.15, color=COLORS['all'])

    # Subsets
    for name, style in [('variable', {'color': COLORS['variable'], 'ls': '--', 'label': 'Variable'}),
                        ('fixed', {'color': COLORS['fixed'], 'ls': '--', 'label': 'Fixed'}),
                        ('c10', {'color': COLORS['c10'], 'ls': ':', 'label': 'c10'}),
                        ('c30', {'color': COLORS['c30'], 'ls': ':', 'label': 'c30'}),
                        ('c50', {'color': COLORS['c50'], 'ls': ':', 'label': 'c50'})]:
        sub = [r for r in clf_results[name] if not r.get('skipped')]
        if sub:
            ax1.plot([r['layer'] for r in sub], [r['auc_mean'] for r in sub],
                     ls=style['ls'], color=style['color'], lw=1.0, label=style['label'], alpha=0.7)

    # Best annotation
    if best_layer is not None:
        ax1.annotate(f"Best: L{best_layer}\nAUC={best_auc:.4f}",
                     xy=(best_layer, best_auc), xytext=(10, -15), textcoords='offset points',
                     fontsize=9, fontweight='bold', color=COLORS['all'],
                     arrowprops=dict(arrowstyle='->', color=COLORS['all']))

    ax1.axhline(y=0.5, color='gray', ls=':', alpha=0.5)
    ax1.set_ylabel('Classification AUC', fontsize=11)
    ax1.set_title('IC V2: SAE Features Predict Bankruptcy Outcome (Gemma)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=8, ncol=3, loc='lower right')
    ax1.grid(axis='y', alpha=0.2)
    y_min = min(aucs_all) - 0.05
    y_max = max(aucs_all) + 0.05
    ax1.set_ylim(max(0.4, y_min), min(1.0, y_max))

    # Active features bar
    ax2 = axes[1]
    n_actives = [r.get('n_active', r.get('n_features', 0)) for r in clf_results['all']]
    ax2.bar(layers_all, n_actives, color=COLORS['ic'], alpha=0.5)
    ax2.set_xlabel('Layer', fontsize=11)
    ax2.set_ylabel('Active SAE Features\n(>1% activation rate)', fontsize=9)
    ax2.annotate(f"Avg: {np.mean(n_actives):.0f} of 131K total\npass activation filter",
                 xy=(0.98, 0.95), xycoords='axes fraction', ha='right', va='top', fontsize=8,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.85))

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'ic_fig_a_classification.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info("  Saved: ic_fig_a_classification.png")

    # --- Fig B: Top-10 Interaction Heatmap (raw values, shared scale with SM) ---
    cells = ["variable_bankruptcy", "variable_voluntary_stop", "fixed_bankruptcy", "fixed_voluntary_stop"]
    cell_labels = ["Var-BK", "Var-Safe", "Fix-BK", "Fix-Safe"]

    all_sig = []
    for lr in anova_results['layer_summary']:
        for feat in lr.get('top_features', []):
            if feat.get('interaction_significant'):
                fc = dict(feat)
                fc['layer'] = lr['layer']
                all_sig.append(fc)
    all_sig.sort(key=lambda x: x['interaction_eta_sq'], reverse=True)
    top10 = all_sig[:10]

    if top10:
        fig, ax = plt.subplots(figsize=(8, 6))
        matrix = np.zeros((len(top10), 4))
        for i, feat in enumerate(top10):
            gm = feat['group_means']
            for j, c in enumerate(cells):
                matrix[i, j] = gm.get(c, 0.0)

        global_max = matrix.max() * 1.1 if matrix.max() > 0 else 1.0

        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=global_max)
        ax.set_xticks(range(4))
        ax.set_xticklabels(cell_labels, fontsize=11, fontweight='bold')
        feat_labels = [f"L{f['layer']}/F{f['feature_idx']}" for f in top10]
        ax.set_yticks(range(len(top10)))
        ax.set_yticklabels(feat_labels, fontsize=9)

        for i in range(len(top10)):
            for j in range(4):
                val = matrix[i, j]
                color = 'white' if val > global_max * 0.6 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=7, color=color, fontweight='bold')

        for i, f in enumerate(top10):
            ax.text(4.3, i, f"η²={f['interaction_eta_sq']:.4f}", fontsize=7.5, va='center', color='gray')

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Raw SAE Activation', fontsize=10)
        ax.set_title('IC V2: Top-10 Interaction Features (raw activation)', fontsize=12, fontweight='bold')
        fig.tight_layout()
        fig.savefig(FIG_DIR / 'ic_fig_b_heatmap_raw.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info("  Saved: ic_fig_b_heatmap_raw.png")

    # --- Fig D: eta² comparison IC vs SM ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=False)

    for ax, (data, name, color) in zip(axes, [(anova_results, 'IC', COLORS['ic']),
                                               (sm_anova, 'SM', COLORS['sm'])]):
        if data is None:
            ax.text(0.5, 0.5, f"No {name} data", transform=ax.transAxes, ha='center')
            continue
        lk = 'layer_summary'
        layers_, medians_, maxes_ = [], [], []
        for lr in data[lk]:
            etas = [f['interaction_eta_sq'] for f in lr.get('top_features', []) if f.get('interaction_significant')]
            layers_.append(lr['layer'])
            medians_.append(np.median(etas) if etas else 0)
            maxes_.append(np.max(etas) if etas else 0)

        ax.plot(layers_, medians_, '-', color=color, lw=2.0, label='Median')
        ax.plot(layers_, maxes_, ':', color=color, lw=1.2, alpha=0.7, label='Max')
        ax.fill_between(layers_, medians_, maxes_, alpha=0.15, color=color)

        y_max = max(maxes_) * 1.2 if max(maxes_) > 0 else 0.02
        for y, label in [(0.01, 'Small'), (0.06, 'Medium'), (0.14, 'Large')]:
            if y < y_max:
                ax.axhline(y=y, color='gray', ls='--', alpha=0.4, lw=0.8)
                ax.text(max(layers_) + 0.5, y, label, fontsize=7, color='gray')

        peak_idx = int(np.argmax(maxes_))
        if maxes_[peak_idx] > 0:
            ax.annotate(f"L{layers_[peak_idx]}\n{maxes_[peak_idx]:.4f}",
                        xy=(layers_[peak_idx], maxes_[peak_idx]),
                        xytext=(10, -10), textcoords='offset points',
                        fontsize=8, fontweight='bold', color=color,
                        arrowprops=dict(arrowstyle='->', color=color))

        ax.set_xlabel('Layer', fontsize=11)
        ax.set_ylabel('η² (Interaction Effect Size)', fontsize=10)
        ax.set_title(f'{name} (Gemma-2-9B)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(bottom=0, top=y_max)

    fig.suptitle('Betting × Outcome Interaction Effect (V2 Decision-Point Fix)', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'ic_fig_d_eta.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info("  Saved: ic_fig_d_eta.png")

    # --- Fig E: Interaction Ratio — separate panels ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    for ax, (data, name, color) in zip(axes, [(anova_results, 'IC', COLORS['ic']),
                                               (sm_anova, 'SM', COLORS['sm'])]):
        if data is None:
            continue
        layers_, pcts_, n_tested_ = [], [], []
        for lr in data['layer_summary']:
            nt = lr['n_tested']
            ns = lr['n_significant']
            if nt > 0:
                layers_.append(lr['layer'])
                pcts_.append(100 * ns / nt)
                n_tested_.append(nt)

        ax.bar(layers_, pcts_, color=color, alpha=0.5, width=0.8)
        ax.plot(layers_, pcts_, '-o', color=color, lw=1.5, markersize=3, zorder=5)

        # Peak (exclude n<10 artifacts)
        valid = [p if n >= 10 else -1 for p, n in zip(pcts_, n_tested_)]
        peak = int(np.argmax(valid))
        ax.annotate(f"L{layers_[peak]}: {pcts_[peak]:.1f}%",
                    xy=(layers_[peak], pcts_[peak]), xytext=(10, 10), textcoords='offset points',
                    fontsize=10, color=color, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=color))

        total_tested = sum(n_tested_)
        total_sig = sum(lr['n_significant'] for lr in data['layer_summary'])
        total_pct = 100 * total_sig / total_tested if total_tested > 0 else 0
        ax.text(0.02, 0.95, f"Total: {total_sig:,}/{total_tested:,} ({total_pct:.1f}%)",
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        ax.set_xlabel('Layer', fontsize=11)
        ax.set_ylabel('Interaction ratio (%)', fontsize=10)
        ax.set_title(f'{name} (Gemma-2-9B)', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.2)
        ax.set_ylim(0, max(pcts_) * 1.15)

    fig.suptitle('Interaction Feature Ratio by Layer (V2)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'ic_fig_e_interaction_ratio.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info("  Saved: ic_fig_e_interaction_ratio.png")

    # --- Fig F: Constraint effect ---
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), gridspec_kw={'height_ratios': [2, 1]})

    layers_c = [r['layer'] for r in constraint_results]
    n_sig_c = [r['n_significant'] for r in constraint_results]

    # Direction breakdown
    n_c30_up = []
    n_c50_up = []
    for r in constraint_results:
        c30_up = sum(1 for f in r['features'] if f['cohens_d_c30_c50'] > 0)
        c50_up = sum(1 for f in r['features'] if f['cohens_d_c30_c50'] < 0)
        n_c30_up.append(c30_up)
        n_c50_up.append(c50_up)

    ax = axes[0]
    ax.bar(layers_c, n_c30_up, color=COLORS['c30'], alpha=0.7, label='c30 > c50')
    ax.bar(layers_c, n_c50_up, bottom=n_c30_up, color=COLORS['c50'], alpha=0.7, label='c50 > c30')
    ax.set_ylabel('# Significant Features\n(FDR + |d| >= 0.3)', fontsize=10)
    ax.set_title('IC V2: Which Layers Encode Constraint Level?', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)

    # Percentage c30-up
    ax2 = axes[1]
    pct_c30 = [100 * c30 / (c30 + c50) if (c30 + c50) > 0 else 50 for c30, c50 in zip(n_c30_up, n_c50_up)]
    ax2.bar(layers_c, pct_c30, color=COLORS['c30'], alpha=0.5)
    ax2.axhline(y=50, color='gray', ls='--', alpha=0.5, lw=0.8)
    ax2.text(max(layers_c) + 0.5, 50, 'Balanced', fontsize=7, color='gray')
    ax2.set_xlabel('Layer', fontsize=11)
    ax2.set_ylabel('% c30-up\n(of significant)', fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'ic_fig_f_constraint.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info("  Saved: ic_fig_f_constraint.png")

    # --- CP Fig: AUC comparison IC vs SM (both V2) ---
    if sm_clf:
        fig, ax = plt.subplots(figsize=(13, 5))

        sm_layers = [r['layer'] for r in sm_clf['all_games'] if not r.get('skipped')]
        sm_aucs = [r['auc_mean'] for r in sm_clf['all_games'] if not r.get('skipped')]
        sm_stds = [r['auc_std'] for r in sm_clf['all_games'] if not r.get('skipped')]

        ax.plot(sm_layers, sm_aucs, '-o', color=COLORS['sm'], lw=2.0, markersize=4, label='Slot Machine (Gemma)')
        ax.fill_between(sm_layers, [a-s for a,s in zip(sm_aucs, sm_stds)],
                        [a+s for a,s in zip(sm_aucs, sm_stds)], alpha=0.1, color=COLORS['sm'])

        ax.plot(layers_all, aucs_all, '-o', color=COLORS['ic'], lw=2.0, markersize=4, label='Investment Choice (Gemma)')
        ax.fill_between(layers_all, [a-s for a,s in zip(aucs_all, stds_all)],
                        [a+s for a,s in zip(aucs_all, stds_all)], alpha=0.1, color=COLORS['ic'])

        ax.axhline(y=0.5, color='gray', ls=':', alpha=0.5)
        ax.set_xlabel('Layer', fontsize=11)
        ax.set_ylabel('Classification AUC', fontsize=11)
        ax.set_title('V2 Cross-Paradigm: SAE Classification AUC (Gemma, Decision-Point Fix)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.2)

        fig.tight_layout()
        fig.savefig(FIG_DIR / 'cp_auc_comparison_v2.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info("  Saved: cp_auc_comparison_v2.png")

    logger.info("=" * 60)
    logger.info("All done!")
    logger.info(f"Results: {JSON_DIR}")
    logger.info(f"Figures: {FIG_DIR}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
