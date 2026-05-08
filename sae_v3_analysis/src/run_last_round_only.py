"""Last-round-only analysis: for each game, take the LAST decision round's
hidden state and indicator label, then run the same SAE→indicator regression.

Compare to full-round analysis (Table 1):
  - n drops from ~12K to ~3.2K games (Gemma SM)
  - Each game = 1 sample (last decision)
  - Tests whether the FINAL bet's hidden state predicts the final indicator
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
    load_sae_and_meta, nl_deconfound_split, BEHAVIORAL_ROOT, TOP_K, RIDGE_ALPHA,
)
from run_comprehensive_robustness import compute_iba

OUT = Path('/home/v-seungplee/llm-addiction/sae_v3_analysis/results/last_round_only.json')
LAYER = 22

CELLS = [
    ('gemma', 'sm'), ('gemma', 'ic'), ('gemma', 'mw'),
    ('llama', 'sm'), ('llama', 'ic'), ('llama', 'mw'),
]


def compute_loss_chasing_continuous(meta, model, paradigm):
    """§2/§3 continuous I_LC formula."""
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
            bet = float(bet_val); bal = float(bal_val) if bal_val is not None else float(balances_out[i])
            p_bet, p_bal = float(prev_bet), float(prev_bal)
        except (ValueError, TypeError):
            continue
        if bet <= 0 or bal <= 0 or p_bal <= 0: continue
        balances_out[i] = bal
        br = min(bet / bal, 1.0); p_br = min(p_bet / p_bal, 1.0)
        prev_loss = False
        if rn - 1 < len(hist):
            prev_loss = not hist[rn - 1].get('win', str(hist[rn - 1].get('result', '')) == 'W')
        if prev_loss and p_br > 0:
            lc[i] = max(0.0, (br - p_br) / p_br)
    return lc, balances_out


def fit_strict_cv(X_dense, target, balances, rounds, n_splits=5):
    """5-fold CV with within-fold RF deconfound + Top-200 Spearman."""
    n_feat = X_dense.shape[1]
    k = min(TOP_K, n_feat)
    n_splits = min(n_splits, target.shape[0] // 30)  # smaller folds for tiny n
    if n_splits < 2:
        return None, None
    kf = KFold(n_splits, shuffle=True, random_state=42)
    r2s = []
    for tr, te in kf.split(X_dense):
        try:
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
            pred = Ridge(alpha=RIDGE_ALPHA).fit(Xtr, res_tr).predict(Xte)
            r2s.append(r2_score(res_te, pred))
        except Exception as e:
            print(f'    fold error: {e}')
    if not r2s: return None, None
    return float(np.mean(r2s)), float(np.std(r2s, ddof=1))


def find_last_round_indices(meta):
    """For each game_id, find the row index with the maximum round_num."""
    gids = np.asarray(meta['game_ids'])
    rns = np.asarray(meta['round_nums'])
    last = {}
    for i in range(len(gids)):
        gid = int(gids[i]) if not isinstance(gids[i], (int, np.integer)) else gids[i]
        rn = int(rns[i])
        if gid not in last or rn > last[gid][1]:
            last[gid] = (i, rn)
    return np.array([v[0] for v in last.values()])


def main():
    results = {}
    for indicator in ['i_ba', 'i_lc', 'i_ec']:
        for model, task in CELLS:
            key = f'{model}_{task}_{indicator}_L{LAYER}'
            print(f'\n=== {key} ===', flush=True)
            sp, meta = load_sae_and_meta(model, task, LAYER)
            if sp is None: continue

            # Compute target
            if indicator == 'i_ba':
                target, balances = compute_iba(meta, model, task)
            elif indicator == 'i_lc':
                target, balances = compute_loss_chasing_continuous(meta, model, task)
            elif indicator == 'i_ec':
                iba_result = compute_iba(meta, model, task)
                if iba_result is None: continue
                bet_ratios, balances = iba_result
                target = np.where(np.isnan(bet_ratios), np.nan, (bet_ratios >= 0.5).astype(float))

            if target is None: continue

            # Last round per game
            last_idx = find_last_round_indices(meta)
            print(f'  Total rounds: {len(target)}, unique games (last rounds): {len(last_idx)}')

            # All-variable bet filter
            bt = meta['bet_types']
            mask_all = np.zeros(len(target), dtype=bool)
            mask_all[last_idx] = True
            valid = mask_all & (bt == 'variable') & ~np.isnan(target) & ~np.isnan(balances) & (balances > 0)
            if indicator == 'i_ba':
                valid = valid & (target > 0)
            n = int(valid.sum())
            print(f'  Last-round + variable + valid: n={n}')
            if n < 100:
                print(f'  SKIP: n<100')
                results[key] = {'n': n, 'reason': 'n<100'}
                continue

            X_sparse = sp[valid]
            t = target[valid]
            bal = balances[valid]
            rn = meta['round_nums'][valid].astype(float)
            nnz = np.diff(X_sparse.tocsc().indptr)
            active = np.where(nnz > 5)[0]  # looser since smaller n
            X = X_sparse[:, active].toarray()
            print(f'  Active features: {len(active)}')

            try:
                r2_mean, r2_std = fit_strict_cv(X, t, bal, rn)
                print(f'  R² = {r2_mean:+.4f} ± {r2_std:.4f}', flush=True)
                results[key] = {'n': n, 'r2_mean': r2_mean, 'r2_std': r2_std,
                                'mean_target': float(t.mean())}
            except Exception as e:
                print(f'  ERROR: {e}')
                results[key] = {'n': n, 'error': str(e)[:200]}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved {OUT}')


if __name__ == '__main__':
    main()
