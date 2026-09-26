"""Extract §4.1 Ridge weights for downstream M3' steering.

§4.1 pipeline (from run_groupkfold_recompute.py):
  - SAE features → Top-K=200 by |Spearman ρ| with deconfounded target
  - StandardScaler + Ridge(α=100)
  - 5-fold GroupKFold by game_id

This script re-runs the §4.1 pipeline and ALSO saves:
  - Ridge coefficient vector w (200,)
  - Selected feature indices into 131072-dim SAE space (200,)
  - StandardScaler params (mean, scale) for projection back

Output: results/v19_multi_patching/M3prime_indicator_steering/direction_metadata/{cell}.json

Usage:
    python extract_section4_ridge_weights.py --model gemma --task sm --indicator i_ba
    python extract_section4_ridge_weights.py --model gemma --task sm --indicator i_lc
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

sys.path.insert(0, '/home/v-seungplee/llm-addiction/sae_v3_analysis/src')
from run_perm_null_ilc import (
    load_sae_and_meta, nl_deconfound_split, TOP_K, RIDGE_ALPHA,
)
from run_groupkfold_recompute import compute_loss_chasing_continuous
from run_comprehensive_robustness import compute_iba

LAYER = 22
OUT_DIR = Path('/home/v-seungplee/llm-addiction/sae_v3_analysis/results/v19_multi_patching/'
               'M3prime_indicator_steering/direction_metadata')


def fit_ridge_full_data(X_dense, target, balances, rounds, groups, n_splits=5):
    """Refit Ridge on full data (no fold) but using fold-level deconfound +
    Top-K=200 selection averaged across folds for stability.

    Returns the ridge weights, selected feature indices, and scaler params.
    Also returns mean R² across folds for reproducibility check.
    """
    n_feat = X_dense.shape[1]
    k = min(TOP_K, n_feat)

    gkf = GroupKFold(n_splits=n_splits)
    fold_r2 = []
    selected_indices_per_fold = []

    # Fold-level: select Top-K by |Spearman ρ| with deconfounded target on train fold
    for tr, te in gkf.split(X_dense, groups=groups):
        res_tr, res_te = nl_deconfound_split(
            target[tr], balances[tr], rounds[tr],
            target[te], balances[te], rounds[te],
        )
        corrs = np.array([
            abs(spearmanr(X_dense[tr, j], res_tr)[0])
            if X_dense[tr, j].std() > 0 else 0
            for j in range(n_feat)
        ])
        idx = np.argsort(corrs)[-k:]
        selected_indices_per_fold.append(idx)

        sc = StandardScaler()
        Xtr = sc.fit_transform(X_dense[tr][:, idx])
        Xte = sc.transform(X_dense[te][:, idx])
        ridge_fold = Ridge(alpha=RIDGE_ALPHA).fit(Xtr, res_tr)
        from sklearn.metrics import r2_score
        fold_r2.append(r2_score(res_te, ridge_fold.predict(Xte)))

    # Final fit: use union of top-K across folds OR most-frequent indices
    # Strategy: use indices from the fold with median R²
    median_fold = int(np.argsort(fold_r2)[len(fold_r2) // 2])
    final_indices = selected_indices_per_fold[median_fold]

    # Fit final Ridge on FULL data with these indices
    res_full, _ = nl_deconfound_split(
        target, balances, rounds, target, balances, rounds  # full deconfound
    )
    sc = StandardScaler()
    X_final = sc.fit_transform(X_dense[:, final_indices])
    ridge_final = Ridge(alpha=RIDGE_ALPHA).fit(X_final, res_full)

    return {
        'ridge_coef': ridge_final.coef_.tolist(),                   # (200,)
        'ridge_intercept': float(ridge_final.intercept_),
        'feature_indices': final_indices.tolist(),                  # into 131k SAE space
        'scaler_mean': sc.mean_.tolist(),                           # (200,)
        'scaler_scale': sc.scale_.tolist(),                         # (200,)
        'fold_r2': fold_r2,
        'fold_r2_mean': float(np.mean(fold_r2)),
        'fold_r2_std': float(np.std(fold_r2, ddof=1)) if len(fold_r2) > 1 else 0.0,
        'n_samples': int(X_dense.shape[0]),
        'n_features_total': int(X_dense.shape[1]),
        'top_k': k,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['gemma', 'llama'], required=True)
    ap.add_argument('--task', choices=['sm', 'ic', 'mw'], required=True)
    ap.add_argument('--indicator', choices=['i_lc', 'i_ba', 'i_ec'], required=True)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cell = f'{args.model}_{args.task}_{args.indicator}_L{LAYER}'
    out_path = OUT_DIR / f'{cell}.json'

    print(f'[load] {args.model} {args.task} L{LAYER}')
    sp, meta = load_sae_and_meta(args.model, args.task, LAYER)
    if sp is None:
        print(f'  SAE missing for {args.model}/{args.task}/L{LAYER}', file=sys.stderr)
        sys.exit(1)

    print(f'[target] computing {args.indicator}')
    if args.indicator == 'i_lc':
        target, balances = compute_loss_chasing_continuous(meta, args.model, args.task)
    elif args.indicator == 'i_ba':
        result = compute_iba(meta, args.model, args.task)
        if result is None:
            print(f'  compute_iba returned None', file=sys.stderr); sys.exit(1)
        target, balances = result
    elif args.indicator == 'i_ec':
        result = compute_iba(meta, args.model, args.task)
        if result is None:
            print(f'  compute_iba returned None', file=sys.stderr); sys.exit(1)
        bet_ratios, balances = result
        target = np.where(np.isnan(bet_ratios), np.nan,
                          (bet_ratios >= 0.5).astype(float))

    bt = meta['bet_types']
    valid = (bt == 'variable') & ~np.isnan(target) & ~np.isnan(balances) & (balances > 0)
    if args.indicator == 'i_ba':
        valid = valid & (target > 0)

    n = int(valid.sum())
    if n < 100:
        print(f'  n<100 ({n}); not enough samples', file=sys.stderr); sys.exit(1)

    X_sparse = sp[valid]
    t = target[valid]
    bal = balances[valid]
    rn = meta['round_nums'][valid].astype(float)
    gids = np.asarray(meta['game_ids'])[valid]

    nnz = np.diff(X_sparse.tocsc().indptr)
    active = np.where(nnz > 10)[0]
    X = X_sparse[:, active].toarray()
    if X.shape[1] == 0:
        print(f'  no active features', file=sys.stderr); sys.exit(1)

    print(f'[fit] n={n}, n_active={X.shape[1]} → Ridge fit')
    result = fit_ridge_full_data(X, t, bal, rn, gids)
    result['model'] = args.model
    result['task'] = args.task
    result['indicator'] = args.indicator
    result['layer'] = LAYER
    result['active_feature_subset'] = active.tolist()  # to map back to 131072 SAE space

    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'[save] {out_path}')
    print(f'  fold R²: mean={result["fold_r2_mean"]:.4f} ± {result["fold_r2_std"]:.4f}')
    print(f'  ridge w: shape=({len(result["ridge_coef"])},), '
          f'norm={np.linalg.norm(result["ridge_coef"]):.4f}')


if __name__ == '__main__':
    main()
