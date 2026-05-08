"""Single-feature univariate analysis: per cell, identify the top-1 SAE feature
by absolute Spearman correlation with the indicator, and report its univariate
R² + which task-overlap class it belongs to.

This complements the Top-200 Ridge analysis (Table 1) with a cleaner mech-interp
story: ``the strongest single feature explains X% of variance, and the same
feature appears in N out of 3 tasks within the same model''.
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
    load_sae_and_meta, nl_deconfound_split, BEHAVIORAL_ROOT, RIDGE_ALPHA,
)
from run_comprehensive_robustness import compute_iba

OUT = Path('/home/v-seungplee/llm-addiction/sae_v3_analysis/results/single_feature_analysis.json')
LAYER = 22

# Cells we care about: 6 (model, task) cells, all-variable subset
CELLS = [
    ('gemma', 'sm'), ('gemma', 'ic'), ('gemma', 'mw'),
    ('llama', 'sm'), ('llama', 'ic'), ('llama', 'mw'),
]


def compute_loss_chasing_continuous(meta, model, paradigm):
    """§2/§3 continuous I_LC formula (post-loss only)."""
    if paradigm == 'sm':
        if model == 'gemma':
            gpath = BEHAVIORAL_ROOT / 'slot_machine/gemma_v4_role/final_gemma_20260227_002507.json'
        else:
            gpath = BEHAVIORAL_ROOT / 'slot_machine/llama_v4_role/final_llama_20260315_062428.json'
        raw = json.load(open(gpath))
        games_data = raw.get('results', raw.get('games', []))
        if isinstance(games_data, dict): games_data = list(games_data.values())
    elif paradigm == 'mw':
        mw_dir = BEHAVIORAL_ROOT / 'mystery_wheel' / ('gemma_v2_role' if model == 'gemma' else 'llama_v2_role')
        games_data = []
        for f in sorted(mw_dir.glob(f'{model}_mysterywheel_*.json')):
            d = json.load(open(f))
            results = d.get('results', d.get('games', []))
            if isinstance(results, dict): games_data.extend(results.values())
            else: games_data.extend(results)
    elif paradigm == 'ic':
        ic_dir = BEHAVIORAL_ROOT / 'investment_choice' / ('v2_role_gemma' if model == 'gemma' else 'v2_role_llama')
        games_data = []
        for f in sorted(ic_dir.glob('*.json')):
            d = json.load(open(f))
            results = d.get('results', d.get('games', []))
            if isinstance(results, dict): games_data.extend(results.values())
            else: games_data.extend(results)
    else:
        return None, None

    game_map = {i + 1: g for i, g in enumerate(games_data)}
    n = len(meta['game_ids'])
    lc = np.full(n, np.nan)
    balances_out = (meta['balances'].astype(float).copy()
                    if meta.get('balances') is not None else np.full(n, np.nan))

    for i in range(n):
        gid = meta['game_ids'][i]
        rn = int(meta['round_nums'][i]) - 1
        g = game_map.get(gid) or game_map.get(str(gid))
        if g is None and isinstance(gid, (np.integer, int)):
            g = game_map.get(int(gid))
        if g is None: continue
        raw_decs = g.get('decisions', g.get('history', []))
        decs = [d for d in raw_decs if d.get('action') != 'skip' and not d.get('skipped', False)]
        hist = g.get('history', decs)
        if rn >= len(decs) or rn < 1: continue

        dec = decs[rn]
        prev_dec = decs[rn - 1]
        bet_val = dec.get('parsed_bet') or dec.get('bet') or dec.get('bet_amount')
        bal_val = dec.get('balance_before') or dec.get('balance')
        prev_bet = prev_dec.get('parsed_bet') or prev_dec.get('bet') or prev_dec.get('bet_amount')
        prev_bal = prev_dec.get('balance_before') or prev_dec.get('balance')
        if any(v is None for v in [bet_val, prev_bet, prev_bal]): continue
        try:
            bet = float(bet_val)
            bal = float(bal_val) if bal_val is not None else float(balances_out[i])
            p_bet, p_bal = float(prev_bet), float(prev_bal)
        except (ValueError, TypeError):
            continue
        if bet <= 0 or bal <= 0 or p_bal <= 0: continue
        balances_out[i] = bal
        br = min(bet / bal, 1.0)
        p_br = min(p_bet / p_bal, 1.0)
        prev_loss = False
        if rn - 1 < len(hist):
            prev_loss = not hist[rn - 1].get('win', str(hist[rn - 1].get('result', '')) == 'W')
        if prev_loss and p_br > 0:
            lc[i] = max(0.0, (br - p_br) / p_br)
    return lc, balances_out


def fit_single_feature_cv(X_col, target, balances, rounds):
    """5-fold CV with within-fold deconfound, single feature only."""
    kf = KFold(5, shuffle=True, random_state=42)
    r2s = []
    for tr, te in kf.split(X_col):
        res_tr, res_te = nl_deconfound_split(
            target[tr], balances[tr], rounds[tr],
            target[te], balances[te], rounds[te],
        )
        if X_col[tr].std() == 0:
            r2s.append(0.0); continue
        sc = StandardScaler()
        Xtr = sc.fit_transform(X_col[tr].reshape(-1, 1))
        Xte = sc.transform(X_col[te].reshape(-1, 1))
        pred = Ridge(alpha=RIDGE_ALPHA).fit(Xtr, res_tr).predict(Xte)
        r2s.append(r2_score(res_te, pred))
    return float(np.mean(r2s)), float(np.std(r2s, ddof=1))


def find_top_features(model, task, indicator='i_ba'):
    """Return top features by all-variable Spearman, with single-feature R²."""
    sp, meta = load_sae_and_meta(model, task, LAYER)
    if sp is None: return None
    if indicator == 'i_ba':
        target, balances = compute_iba(meta, model, task)
    elif indicator == 'i_lc':
        target, balances = compute_loss_chasing_continuous(meta, model, task)
    else:
        return None
    if target is None: return None

    bt = meta['bet_types']
    valid = (bt == 'variable') & ~np.isnan(target) & ~np.isnan(balances) & (balances > 0)
    if indicator == 'i_ba':
        valid = valid & (target > 0)
    n = int(valid.sum())
    if n < 100: return {'n': n, 'reason': 'n<100'}

    X_sparse = sp[valid]
    t = target[valid]
    bal = balances[valid]
    rn = meta['round_nums'][valid].astype(float)

    nnz = np.diff(X_sparse.tocsc().indptr)
    active = np.where(nnz > 10)[0]
    X = X_sparse[:, active].toarray()

    # Spearman correlation per feature
    corrs = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        if X[:, j].std() > 0:
            r = spearmanr(X[:, j], t)[0]
            corrs[j] = abs(r) if not np.isnan(r) else 0
    top_idx_sorted = np.argsort(corrs)[-10:][::-1]  # Top-10 indices in active

    # Single-feature R² for top-3
    top_results = []
    for rank, idx in enumerate(top_idx_sorted[:3]):
        feat_id = int(active[idx])
        spearman_r = float(corrs[idx])
        r2_mean, r2_std = fit_single_feature_cv(X[:, idx], t, bal, rn)
        top_results.append({
            'rank': rank + 1,
            'feature_id': feat_id,
            'spearman_r_abs': spearman_r,
            'r2_cv_mean': r2_mean,
            'r2_cv_std': r2_std,
        })
    # Top-200 set for overlap analysis
    top200_idx = active[np.argsort(corrs)[-200:]]
    return {
        'n': n,
        'n_active_features': len(active),
        'top3': top_results,
        'top200_set': sorted(map(int, top200_idx)),
    }


def main():
    results = {}
    for indicator in ['i_ba', 'i_lc']:
        for model, task in CELLS:
            key = f'{model}_{task}_{indicator}_L{LAYER}'
            print(f'\n=== {key} ===', flush=True)
            r = find_top_features(model, task, indicator)
            if r is None or r.get('reason'):
                print(f'  SKIP: {r}'); continue
            results[key] = r
            print(f'  n={r["n"]}, active features={r["n_active_features"]}')
            for top in r['top3']:
                print(f'  rank{top["rank"]}: feat #{top["feature_id"]}, '
                      f'|Spearman|={top["spearman_r_abs"]:.3f}, '
                      f'CV R² = {top["r2_cv_mean"]:+.4f} ± {top["r2_cv_std"]:.4f}')

    # Cross-task feature overlap for top-1 feature (per indicator, within model)
    print('\n=== Top-1 feature cross-task check ===')
    for indicator in ['i_ba', 'i_lc']:
        for model in ['gemma', 'llama']:
            top1s = {}
            for task in ['sm', 'ic', 'mw']:
                key = f'{model}_{task}_{indicator}_L{LAYER}'
                if key in results and results[key].get('top3'):
                    top1s[task] = results[key]['top3'][0]['feature_id']
            print(f'{model} {indicator} Top-1 feature IDs: {top1s}')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved {OUT}')


if __name__ == '__main__':
    main()
