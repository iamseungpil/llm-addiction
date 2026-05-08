"""§4.3 multi-indicator ±G modulation (I_LC + I_EC, I_BA already in audit).

For each (model, task, indicator) where label variance allows:
  Re-fit SAE→indicator Ridge readout SEPARATELY within each prompt condition
  (+G, -G, +M, -M, all-variable, fixed), report 5-fold CV R^2 with strict
  within-fold RF deconfound (matching run_perm_null_ilc.py methodology).
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
from sklearn.ensemble import RandomForestRegressor
sys.path.insert(0, '/home/v-seungplee/llm-addiction/sae_v3_analysis/src')
from run_perm_null_ilc import (
    load_sae_and_meta, compute_loss_chasing, nl_deconfound_split,
    TOP_K, RIDGE_ALPHA,
)
from run_comprehensive_robustness import compute_iba

OUT = Path('/home/v-seungplee/llm-addiction/sae_v3_analysis/results/condition_modulation_multi.json')

# (model, task, indicator, layer): which cells to test
# Layers from paper_neural_audit / strict CV peaks:
#  I_LC peak: Gemma L18, LLaMA L22
#  I_BA / I_EC peak: Gemma L24, LLaMA L16
CELLS = [
    # I_LC across all 6 (model, task)
    ('gemma','sm','i_lc',18), ('gemma','ic','i_lc',18), ('gemma','mw','i_lc',18),
    ('llama','sm','i_lc',22), ('llama','ic','i_lc',22), ('llama','mw','i_lc',22),
    # I_EC SM-only
    ('gemma','sm','i_ec',24), ('llama','sm','i_ec',16),
    # I_BA also re-runs for parity (uses paper_neural_audit values normally,
    # but we want strict-CV reproduction here)
    ('gemma','sm','i_ba',24), ('gemma','mw','i_ba',24),
    ('llama','sm','i_ba',16), ('llama','mw','i_ba',16),
]

CONDITIONS = ['plus_G', 'minus_G', 'plus_M', 'minus_M', 'all_variable', 'fixed_all']


def compute_iec(meta, model, task):
    """I_EC = 1 if bet_ratio >= 0.5 else 0; uses same loader as I_BA."""
    iba_result = compute_iba(meta, model, task)
    if iba_result is None:
        return None, None
    bet_ratios, balances = iba_result
    iec = np.where(np.isnan(bet_ratios), np.nan,
                   (bet_ratios >= 0.5).astype(float))
    return iec, balances


def get_label(meta, model, task, indicator):
    if indicator == 'i_lc':
        return compute_loss_chasing(meta, model, task)
    if indicator == 'i_ba':
        return compute_iba(meta, model, task)
    if indicator == 'i_ec':
        return compute_iec(meta, model, task)
    return None, None


def condition_mask(meta, condition):
    """Boolean mask for a prompt-condition × bet-type subset."""
    bt = meta['bet_types']
    pc = meta.get('prompt_conditions')
    if pc is None:
        return None
    # Decode bytes to str if needed
    if hasattr(pc[0], 'decode'):
        pc = np.array([s.decode() for s in pc])
    if condition == 'all_variable':
        return bt == 'variable'
    if condition == 'fixed_all':
        return bt == 'fixed'
    # ±G / ±M: G label appears in prompt_conditions like 'G', 'GM', 'GMP', etc.
    if condition == 'plus_G':
        has_G = np.array(['G' in s for s in pc])
        return (bt == 'variable') & has_G
    if condition == 'minus_G':
        has_G = np.array(['G' in s for s in pc])
        return (bt == 'variable') & ~has_G
    if condition == 'plus_M':
        has_M = np.array(['M' in s for s in pc])
        return (bt == 'variable') & has_M
    if condition == 'minus_M':
        has_M = np.array(['M' in s for s in pc])
        return (bt == 'variable') & ~has_M
    return None


def fit_strict_cv(X_dense, target, balances, rounds, top_k=TOP_K, alpha=RIDGE_ALPHA):
    """5-fold CV with within-fold RF deconfound + Spearman top-K."""
    n_feat = X_dense.shape[1]
    k = min(top_k, n_feat)
    kf = KFold(5, shuffle=True, random_state=42)
    r2s = []
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
        r2s.append(r2_score(res_te, pred))
    return float(np.mean(r2s))


def main():
    results = {}
    t_total = time.time()
    for model, task, indicator, layer in CELLS:
        key = f'{model}_{task}_{indicator}_L{layer}'
        print(f'\n=== {key} ===', flush=True)
        sp, meta = load_sae_and_meta(model, task, layer)
        if sp is None:
            print('  SKIP: no SAE'); continue
        target, balances = get_label(meta, model, task, indicator)
        if target is None:
            print('  SKIP: no behav'); continue

        rounds = meta['round_nums'].astype(float)
        results[key] = {
            'model': model, 'task': task, 'indicator': indicator, 'layer': layer,
            'subsets': {},
        }
        for cond in CONDITIONS:
            mask = condition_mask(meta, cond)
            if mask is None:
                continue
            valid = (mask & ~np.isnan(target) & ~np.isnan(balances)
                     & (balances > 0))
            if indicator == 'i_ba':
                valid = valid & (target > 0)
            n = int(valid.sum())
            if n < 100:
                results[key]['subsets'][cond] = {'r2': None, 'n': n, 'reason': 'n<100'}
                print(f'  {cond}: SKIP n={n}'); continue
            X_sparse = sp[valid]
            t = target[valid]
            bal = balances[valid]
            rn = rounds[valid]
            # Active feature filter
            nnz = np.diff(X_sparse.tocsc().indptr)
            active = np.where(nnz > 10)[0]
            X = X_sparse[:, active].toarray()
            t0 = time.time()
            try:
                r2 = fit_strict_cv(X, t, bal, rn)
            except Exception as e:
                print(f'  {cond}: ERROR {e}')
                results[key]['subsets'][cond] = {'r2': None, 'n': n, 'reason': str(e)[:50]}
                continue
            elapsed = time.time() - t0
            results[key]['subsets'][cond] = {'r2': r2, 'n': n, 'mean_target': float(t.mean())}
            print(f'  {cond}: n={n}, R²={r2:+.4f} ({elapsed:.0f}s)', flush=True)

        # Save incrementally
        with open(OUT, 'w') as f:
            json.dump(results, f, indent=2)
    print(f'\nTotal time: {time.time()-t_total:.0f}s -> {OUT}', flush=True)


if __name__ == '__main__':
    main()
