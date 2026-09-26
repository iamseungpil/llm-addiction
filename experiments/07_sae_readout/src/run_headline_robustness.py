"""§4.3 headline robustness: fold-level SD + permutation null on key cells.

Compute for the 4 SM headline cells (Gemma + LLaMA, ±G and all_variable for I_BA at L22):
  - Fold-level R² across 5 folds (mean ± std)
  - Block-permutation null (game-level shuffle) p-value
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
sys.path.insert(0, '/home/v-seungplee/llm-addiction/sae_v3_analysis/src')
from run_perm_null_ilc import (
    load_sae_and_meta, nl_deconfound_split, TOP_K, RIDGE_ALPHA,
)
from run_comprehensive_robustness import compute_iba

OUT = Path('/home/v-seungplee/llm-addiction/sae_v3_analysis/results/headline_robustness.json')

# Headline cells: SM I_BA at L22 for both models, ±G and all_variable
# (focus on Gemma +126% and LLaMA +36% modulation effects)
CELLS = [
    ('gemma', 'sm', 'i_ba', 22),
    ('llama', 'sm', 'i_ba', 22),
]
CONDITIONS = ['plus_G', 'minus_G', 'all_variable']
N_PERM = 200


def condition_mask(meta, condition):
    bt = meta['bet_types']
    pc = meta.get('prompt_conditions')
    if pc is None: return None
    if hasattr(pc[0], 'decode'):
        pc = np.array([s.decode() for s in pc])
    if condition == 'all_variable':
        return bt == 'variable'
    if condition == 'plus_G':
        has_G = np.array(['G' in s for s in pc])
        return (bt == 'variable') & has_G
    if condition == 'minus_G':
        has_G = np.array(['G' in s for s in pc])
        return (bt == 'variable') & ~has_G
    return None


def fit_cv_folds(X_dense, target, balances, rounds, top_k=TOP_K, alpha=RIDGE_ALPHA):
    """Return list of fold-level R² values."""
    n_feat = X_dense.shape[1]
    k = min(top_k, n_feat)
    kf = KFold(5, shuffle=True, random_state=42)
    fold_r2s = []
    for tr, te in kf.split(X_dense):
        res_tr, res_te = nl_deconfound_split(
            target[tr], balances[tr], rounds[tr],
            target[te], balances[te], rounds[te],
        )
        corrs = np.array([
            abs(spearmanr(X_dense[tr, j], res_tr)[0]) if X_dense[tr, j].std() > 0 else 0
            for j in range(n_feat)
        ])
        idx = np.argsort(corrs)[-k:]
        sc = StandardScaler()
        Xtr = sc.fit_transform(X_dense[tr][:, idx])
        Xte = sc.transform(X_dense[te][:, idx])
        pred = Ridge(alpha=alpha).fit(Xtr, res_tr).predict(Xte)
        fold_r2s.append(r2_score(res_te, pred))
    return fold_r2s


def perm_null_r2(X_dense, target, balances, rounds, gids, n_perm=N_PERM, seed=42):
    """Block-permutation null: shuffle target by game blocks, refit, get R² distribution.
    Returns null R² array + p-value vs observed R²."""
    rng = np.random.default_rng(seed)
    unique_games = np.unique(gids)
    null_r2s = []
    for _ in range(n_perm):
        # Shuffle game labels
        perm_gids = rng.permutation(unique_games)
        gid_map = dict(zip(unique_games, perm_gids))
        # Get shuffled target indices: where game_id_orig was, place game_id_perm's target
        # Simpler approach: randomly permute targets within game blocks
        target_perm = target.copy()
        # Block-level shuffle: assign new game ids by random permutation; targets follow
        new_targets = np.zeros_like(target)
        for orig_gid, new_gid in zip(unique_games, perm_gids):
            old_mask = gids == orig_gid
            new_mask = gids == new_gid
            # Use new_gid's targets for orig_gid's rows (one direction)
            n_old = old_mask.sum()
            n_new = new_mask.sum()
            if n_old == n_new:
                new_targets[old_mask] = target[new_mask]
            else:
                # Different game lengths — sample with replacement
                new_targets[old_mask] = rng.choice(target[new_mask], size=n_old, replace=True)
        try:
            null_folds = fit_cv_folds(X_dense, new_targets, balances, rounds)
            null_r2s.append(np.mean(null_folds))
        except Exception:
            pass
    return null_r2s


def main():
    results = {}
    for model, task, indicator, layer in CELLS:
        print(f'\n=== {model} {task} {indicator} L{layer} ===', flush=True)
        sp, meta = load_sae_and_meta(model, task, layer)
        if sp is None:
            print('  SKIP: no SAE'); continue
        target, balances = compute_iba(meta, model, task)
        if target is None:
            print('  SKIP: no behav'); continue
        rounds = meta['round_nums'].astype(float)
        gids = meta['game_ids']
        cell_key = f'{model}_{task}_{indicator}_L{layer}'
        results[cell_key] = {'subsets': {}}

        for cond in CONDITIONS:
            print(f'  cond={cond} ...', flush=True)
            mask = condition_mask(meta, cond)
            if mask is None:
                print(f'    SKIP no mask'); continue
            valid = (mask & ~np.isnan(target) & ~np.isnan(balances)
                     & (balances > 0) & (target > 0))
            n = int(valid.sum())
            if n < 100:
                print(f'    SKIP n={n}'); continue
            X_sparse = sp[valid]
            t = target[valid]
            bal = balances[valid]
            rn = rounds[valid]
            g = gids[valid] if hasattr(gids, '__getitem__') else gids
            nnz = np.diff(X_sparse.tocsc().indptr)
            active = np.where(nnz > 10)[0]
            X = X_sparse[:, active].toarray()

            # 1) Fold-level R²
            t0 = time.time()
            fold_r2s = fit_cv_folds(X, t, bal, rn)
            mean_r2 = float(np.mean(fold_r2s))
            std_r2 = float(np.std(fold_r2s, ddof=1))
            t_fold = time.time() - t0
            print(f'    fold R²: {mean_r2:+.4f} ± {std_r2:.4f} (folds: {[f"{r:+.3f}" for r in fold_r2s]}) [{t_fold:.0f}s]', flush=True)

            # 2) Permutation null (only on all_variable to save time)
            null_r2s = []
            perm_p = None
            if cond == 'all_variable':
                t0 = time.time()
                # Smaller n_perm for compute budget
                null_r2s = perm_null_r2(X, t, bal, rn, g, n_perm=50)
                if null_r2s:
                    null_arr = np.array(null_r2s)
                    perm_p = float((null_arr >= mean_r2).sum() / len(null_arr))
                    print(f'    perm null (n=50): mean={null_arr.mean():+.4f}, '
                          f'95%ile={np.percentile(null_arr, 95):.4f}, p={perm_p:.4f} [{time.time()-t0:.0f}s]', flush=True)

            results[cell_key]['subsets'][cond] = {
                'n': n,
                'mean_r2': mean_r2,
                'std_r2': std_r2,
                'fold_r2s': fold_r2s,
                'null_r2s_sample': null_r2s[:10] if null_r2s else None,
                'null_n': len(null_r2s),
                'null_mean': float(np.mean(null_r2s)) if null_r2s else None,
                'null_95': float(np.percentile(null_r2s, 95)) if null_r2s else None,
                'perm_p': perm_p,
            }

        with open(OUT, 'w') as f:
            json.dump(results, f, indent=2)

    print(f'\nSaved {OUT}')


if __name__ == '__main__':
    main()
