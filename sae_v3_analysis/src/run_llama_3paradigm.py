#!/usr/bin/env python3
"""
LLaMA 3-paradigm analyses (IC + SM + MW).
Now that MW SAE is extracted, run the analyses that require all 3 paradigms.

Analyses:
  3a. 3-paradigm SAE sign-consistency (chance=25%, comparable to Gemma)
  3b. 6-direction cross-domain transfer matrix (IC↔SM↔MW)
  3c. LLaMA BK classification AUC across paradigms (MW)
  3d. LLaMA MW factor decomposition (3-paradigm)
"""
import sys, json, numpy as np, time
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).parent))
from config import LLAMA_PARADIGMS

np.random.seed(42)
OUT = Path(__file__).parent.parent / "results" / "json"
OUT.mkdir(parents=True, exist_ok=True)
N_PERM = 200


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_sae_dp(paradigm_key, layer):
    """Load decision-point SAE features for a LLaMA paradigm."""
    cfg = LLAMA_PARADIGMS[paradigm_key]
    npz_path = Path(cfg['sae_dir']) / f"sae_features_L{layer}.npz"
    if not npz_path.exists():
        return None, None, None

    data = np.load(npz_path, allow_pickle=False)
    is_last = data['is_last_round'].astype(bool)
    dp_idx = np.where(is_last)[0]
    shape = tuple(data['shape'])

    row, col, vals = data['row_indices'], data['col_indices'], data['values']
    dp_set = set(dp_idx.tolist())
    mask = np.isin(row, dp_idx)
    row_dp, col_dp, vals_dp = row[mask], col[mask], vals[mask]
    row_map = {old: new for new, old in enumerate(dp_idx)}
    row_new = np.array([row_map[r] for r in row_dp])

    dense = np.zeros((len(dp_idx), shape[1]), dtype=np.float32)
    dense[row_new, col_dp] = vals_dp

    # Active features (>=1% activation rate)
    active = (dense > 0).mean(axis=0) >= 0.01
    dense = dense[:, active]
    active_indices = np.where(active)[0]

    meta = {
        'game_outcomes': data['game_outcomes'][dp_idx],
        'bet_types': data['bet_types'][dp_idx],
        'prompt_conditions': data['prompt_conditions'][dp_idx] if 'prompt_conditions' in data else None,
    }
    labels = (meta['game_outcomes'] == 'bankruptcy').astype(int)
    return dense, labels, active_indices


# ══════════════════════════════════════════════
# 3a. 3-paradigm SAE sign-consistency
# ══════════════════════════════════════════════
def analysis_3a():
    log("=" * 60)
    log("3a. LLaMA 3-PARADIGM SAE SIGN-CONSISTENCY (chance=25%)")
    log("=" * 60)

    results = {}
    for layer in [0, 4, 8, 12, 16, 20, 22, 26, 30]:
        # Load all 3 paradigms
        data = {}
        for p in ['ic', 'sm', 'mw']:
            dense, labels, active_idx = load_sae_dp(p, layer)
            if dense is None:
                log(f"  L{layer} {p}: not found")
                break
            data[p] = (dense, labels, active_idx)

        if len(data) < 3:
            continue

        # Find union-active features (active in at least 1 paradigm)
        all_active = set()
        for p in data:
            all_active.update(data[p][2].tolist())

        # For each feature in union, compute Cohen's d per paradigm
        from scipy.stats import norm
        n_sign_consistent = 0
        n_strong = 0
        n_total = 0

        for feat_global_idx in all_active:
            ds = {}
            for p in ['ic', 'sm', 'mw']:
                dense, labels, active_idx = data[p]
                local_idx = np.where(active_idx == feat_global_idx)[0]
                if len(local_idx) == 0:
                    ds[p] = 0.0
                    continue
                local_idx = local_idx[0]
                bk_vals = dense[labels == 1, local_idx]
                safe_vals = dense[labels == 0, local_idx]
                if len(bk_vals) < 5 or len(safe_vals) < 5:
                    ds[p] = 0.0
                    continue
                pooled_std = np.sqrt((bk_vals.var() * len(bk_vals) + safe_vals.var() * len(safe_vals)) /
                                    (len(bk_vals) + len(safe_vals)))
                if pooled_std < 1e-10:
                    ds[p] = 0.0
                else:
                    ds[p] = (bk_vals.mean() - safe_vals.mean()) / pooled_std

            n_total += 1
            signs = [np.sign(ds[p]) for p in ['ic', 'sm', 'mw'] if ds[p] != 0]
            if len(signs) == 3 and all(s == signs[0] for s in signs):
                n_sign_consistent += 1
                geo_mean_d = (abs(ds['ic']) * abs(ds['sm']) * abs(ds['mw'])) ** (1/3)
                if geo_mean_d >= 0.2:
                    n_strong += 1

        pct = n_sign_consistent / n_total * 100 if n_total > 0 else 0
        # Binomial test vs 25% chance
        try:
            p_val = binomtest(n_sign_consistent, n_total, 0.25, alternative='greater').pvalue
        except Exception:
            p_val = 1.0

        results[f'L{layer}'] = {
            'n_active': n_total,
            'n_sign_consistent': n_sign_consistent,
            'n_strong': n_strong,
            'pct_consistent': round(pct, 1),
            'binomial_p': round(p_val, 6),
        }
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "NS"
        log(f"  L{layer}: {n_sign_consistent}/{n_total} ({pct:.1f}%) sign-consistent, "
            f"strong={n_strong}, p={p_val:.4e} {sig}")

    return results


# ══════════════════════════════════════════════
# 3b. 6-direction cross-domain transfer
# ══════════════════════════════════════════════
def load_sae_dp_raw(paradigm_key, layer):
    """Load full dense DP features (all 32K columns, no active filter)."""
    cfg = LLAMA_PARADIGMS[paradigm_key]
    npz_path = Path(cfg['sae_dir']) / f"sae_features_L{layer}.npz"
    if not npz_path.exists():
        return None, None

    data = np.load(npz_path, allow_pickle=False)
    is_last = data['is_last_round'].astype(bool)
    dp_idx = np.where(is_last)[0]
    shape = tuple(data['shape'])

    row, col, vals = data['row_indices'], data['col_indices'], data['values']
    mask = np.isin(row, dp_idx)
    row_dp, col_dp, vals_dp = row[mask], col[mask], vals[mask]
    row_map = {old: new for new, old in enumerate(dp_idx)}
    row_new = np.array([row_map[r] for r in row_dp])

    dense = np.zeros((len(dp_idx), shape[1]), dtype=np.float32)
    dense[row_new, col_dp] = vals_dp

    labels = (data['game_outcomes'][dp_idx] == 'bankruptcy').astype(int)
    return dense, labels


def analysis_3b():
    log("=" * 60)
    log("3b. LLaMA 6-DIRECTION CROSS-DOMAIN TRANSFER (IC↔SM↔MW)")
    log("=" * 60)

    results = {}
    for layer in [8, 12, 22, 25, 30]:
        # Load raw (all 32K features) for consistent dimensions
        data = {}
        for p in ['ic', 'sm', 'mw']:
            dense, labels = load_sae_dp_raw(p, layer)
            if dense is not None:
                data[p] = (dense, labels)

        if len(data) < 3:
            log(f"  L{layer}: insufficient paradigms ({len(data)})")
            continue

        # Common active features (>=1% in at least 2 paradigms)
        active_masks = []
        for p in ['ic', 'sm', 'mw']:
            active_masks.append((data[p][0] > 0).mean(axis=0) >= 0.01)
        common_active = np.sum(active_masks, axis=0) >= 2
        common_idx = np.where(common_active)[0]
        log(f"  L{layer}: {len(common_idx)} common active features")

        layer_results = {}
        for src in ['ic', 'sm', 'mw']:
            for tgt in ['ic', 'sm', 'mw']:
                if src == tgt:
                    continue

                X_src = data[src][0][:, common_idx]
                y_src = data[src][1]
                X_tgt = data[tgt][0][:, common_idx]
                y_tgt = data[tgt][1]

                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_src)
                X_te = scaler.transform(X_tgt)

                n_comp = min(50, X_tr.shape[1], X_tr.shape[0] - 1)
                pca = PCA(n_components=n_comp)
                X_tr = pca.fit_transform(X_tr)
                X_te = pca.transform(X_te)

                clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000, random_state=42)
                clf.fit(X_tr, y_src)

                if len(np.unique(y_tgt)) < 2:
                    layer_results[f'{src}_to_{tgt}'] = {'auc': float('nan'), 'perm_p': float('nan')}
                    continue

                proba = clf.predict_proba(X_te)[:, 1]
                auc = roc_auc_score(y_tgt, proba)

                null_aucs = []
                for _ in range(N_PERM):
                    y_shuf = np.random.permutation(y_tgt)
                    if len(np.unique(y_shuf)) < 2:
                        continue
                    null_aucs.append(roc_auc_score(y_shuf, proba))
                perm_p = np.mean([n >= auc for n in null_aucs]) if null_aucs else 1.0

                layer_results[f'{src}_to_{tgt}'] = {
                    'auc': round(auc, 4),
                    'perm_p': round(perm_p, 4),
                }
                sig = "***" if perm_p < 0.001 else "NS"
                log(f"  L{layer} {src.upper()}→{tgt.upper()}: AUC={auc:.3f} (p={perm_p:.3f}) {sig}")

        results[f'L{layer}'] = layer_results

    return results


# ══════════════════════════════════════════════
# 3c. MW BK classification AUC
# ══════════════════════════════════════════════
def analysis_3c():
    log("=" * 60)
    log("3c. LLaMA MW BK CLASSIFICATION AUC (per layer)")
    log("=" * 60)

    results = {}
    for layer in [0, 4, 8, 12, 16, 20, 22, 26, 30]:
        dense, labels, _ = load_sae_dp('mw', layer)
        if dense is None:
            continue

        scaler = StandardScaler()
        X = scaler.fit_transform(dense)
        n_comp = min(50, X.shape[1], X.shape[0] - 1)
        pca = PCA(n_components=n_comp)
        X = pca.fit_transform(X)

        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs = []
        for train_idx, test_idx in skf.split(X, labels):
            clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000, random_state=42)
            clf.fit(X[train_idx], labels[train_idx])
            proba = clf.predict_proba(X[test_idx])[:, 1]
            if len(np.unique(labels[test_idx])) >= 2:
                aucs.append(roc_auc_score(labels[test_idx], proba))

        mean_auc = np.mean(aucs) if aucs else 0
        results[f'L{layer}'] = {
            'auc': round(mean_auc, 4),
            'n_features': dense.shape[1],
            'n_bk': int(labels.sum()),
        }
        log(f"  L{layer}: AUC={mean_auc:.4f} [{dense.shape[1]} features, {labels.sum()} BK]")

    return results


# ══════════════════════════════════════════════
# 3d. 3-paradigm factor decomposition
# ══════════════════════════════════════════════
def analysis_3d():
    log("=" * 60)
    log("3d. LLaMA 3-PARADIGM FACTOR DECOMPOSITION (IC+SM+MW)")
    log("=" * 60)

    layer = 22
    all_features = []
    all_outcome = []
    all_bettype = []
    all_paradigm = []

    for p_key, p_label in [('ic', 0), ('sm', 1), ('mw', 2)]:
        dense, labels, _ = load_sae_dp(p_key, layer)
        if dense is None:
            log(f"  {p_key}: not found")
            return {}
        all_features.append(dense)
        all_outcome.append(labels)
        all_bettype.append((load_sae_dp(p_key, layer)[0] is not None))  # placeholder
        all_paradigm.append(np.full(len(labels), p_label))

    # Reload bet_types properly
    all_bt = []
    for p_key in ['ic', 'sm', 'mw']:
        cfg = LLAMA_PARADIGMS[p_key]
        npz_path = Path(cfg['sae_dir']) / f"sae_features_L{layer}.npz"
        data = np.load(npz_path, allow_pickle=False)
        is_last = data['is_last_round'].astype(bool)
        bt = data['bet_types'][is_last]
        all_bt.append((bt == 'variable').astype(int))

    # Align to minimum feature count across paradigms
    min_feat = min(f.shape[1] for f in all_features)
    all_features = [f[:, :min_feat] for f in all_features]
    X = np.vstack(all_features)
    y_outcome = np.concatenate(all_outcome)
    y_bettype = np.concatenate(all_bt)
    y_paradigm = np.concatenate(all_paradigm)

    # Find common active features (intersection)
    # X columns may differ per paradigm due to different active features
    # Use union approach: keep all columns, some may be zero for some paradigms
    n_features = X.shape[1]

    # Per-feature OLS: feature ~ outcome + bettype + paradigm_SM + paradigm_MW
    from numpy.linalg import lstsq
    design = np.column_stack([
        np.ones(len(y_outcome)),
        y_outcome,
        y_bettype,
        (y_paradigm == 1).astype(float),
        (y_paradigm == 2).astype(float),
    ])

    n_outcome_sig = 0
    n_bettype_sig = 0
    n_paradigm_sig = 0

    for fi in range(n_features):
        y = X[:, fi]
        beta, residuals, rank, sv = lstsq(design, y, rcond=None)
        y_hat = design @ beta
        resid = y - y_hat
        n = len(y)
        p = design.shape[1]
        mse = np.sum(resid**2) / (n - p)
        if mse < 1e-20:
            continue
        cov = mse * np.linalg.inv(design.T @ design + 1e-10 * np.eye(p))
        se = np.sqrt(np.diag(cov))
        t_stats = beta / (se + 1e-20)

        from scipy.stats import t as t_dist
        p_vals = 2 * (1 - t_dist.cdf(np.abs(t_stats), df=n-p))

        if p_vals[1] < 0.01:  # outcome
            n_outcome_sig += 1
        if p_vals[2] < 0.01:  # bettype
            n_bettype_sig += 1
        if p_vals[3] < 0.01 or p_vals[4] < 0.01:  # paradigm
            n_paradigm_sig += 1

    results = {
        'layer': layer,
        'n_features': n_features,
        'n_games': len(y_outcome),
        'outcome_sig': n_outcome_sig,
        'outcome_pct': round(n_outcome_sig / n_features * 100, 1),
        'bettype_sig': n_bettype_sig,
        'bettype_pct': round(n_bettype_sig / n_features * 100, 1),
        'paradigm_sig': n_paradigm_sig,
        'paradigm_pct': round(n_paradigm_sig / n_features * 100, 1),
    }

    log(f"  L{layer}: {n_features} features, {len(y_outcome)} games")
    log(f"  Outcome-sig: {n_outcome_sig} ({results['outcome_pct']}%)")
    log(f"  Bettype-sig: {n_bettype_sig} ({results['bettype_pct']}%)")
    log(f"  Paradigm-sig: {n_paradigm_sig} ({results['paradigm_pct']}%)")

    return results


# ══════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════
if __name__ == '__main__':
    start = time.time()

    all_results = {
        'metadata': {
            'script': 'run_llama_3paradigm.py',
            'timestamp': datetime.now().isoformat(),
            'paradigms': ['ic', 'sm', 'mw'],
            'n_permutations': N_PERM,
        },
        '3a_sign_consistency': analysis_3a(),
        '3b_cross_domain_transfer': analysis_3b(),
        '3c_mw_classification': analysis_3c(),
        '3d_factor_decomposition': analysis_3d(),
    }

    elapsed = time.time() - start
    log(f"\nAll 3-paradigm analyses complete in {elapsed:.0f}s")

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = OUT / f"llama_3paradigm_{ts}.json"
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    log(f"Saved to {out_file}")
