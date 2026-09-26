"""Compute permutation null p-values for all reportable Table 1 cells.

Cells: 6 (model, task) × 3 indicators = 18 max, but I_BA/I_EC are n/a for IC,
and we already have perm null for headline I_BA cells (Gemma SM, LLaMA SM).

Run with n_perm=20 for budget; gives p-resolution of 0.05 (one perm above
observed yields p=0.05). For our R² >= 0.04 cells, expect p < 0.05.
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

OUT = Path('/home/v-seungplee/llm-addiction/sae_v3_analysis/results/table1_perm_null.json')
LAYER = 22
N_PERM = 20

# Cells with reportable R² (from Table 1)
CELLS = [
    ('gemma', 'sm', 'i_lc'), ('gemma', 'sm', 'i_ba'), ('gemma', 'sm', 'i_ec'),
    ('gemma', 'mw', 'i_lc'), ('gemma', 'mw', 'i_ba'),
    ('llama', 'sm', 'i_lc'), ('llama', 'sm', 'i_ba'), ('llama', 'sm', 'i_ec'),
    ('llama', 'mw', 'i_lc'), ('llama', 'mw', 'i_ba'),
]


def compute_loss_chasing_continuous(meta, model, paradigm):
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
            r = d.get('results', d.get('games', []))
            if isinstance(r, dict): games_data.extend(r.values())
            else: games_data.extend(r)
    else:
        return None, None

    game_map = {i + 1: g for i, g in enumerate(games_data)}
    n = len(meta['game_ids'])
    lc = np.full(n, np.nan)
    balances_out = (meta['balances'].astype(float).copy()
                    if meta.get('balances') is not None else np.full(n, np.nan))
    for i in range(n):
        gid = meta['game_ids'][i]; rn = int(meta['round_nums'][i]) - 1
        g = game_map.get(gid) or game_map.get(str(gid))
        if g is None and isinstance(gid, (np.integer, int)): g = game_map.get(int(gid))
        if g is None: continue
        raw_decs = g.get('decisions', g.get('history', []))
        decs = [d for d in raw_decs if d.get('action') != 'skip' and not d.get('skipped', False)]
        hist = g.get('history', decs)
        if rn >= len(decs) or rn < 1: continue
        dec = decs[rn]; prev_dec = decs[rn - 1]
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


def fit_cv(X_dense, target, balances, rounds, n_splits=5):
    n_feat = X_dense.shape[1]; k = min(TOP_K, n_feat)
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
        except Exception:
            pass
    return float(np.mean(r2s)) if r2s else None


def main():
    results = {}
    for model, task, indicator in CELLS:
        key = f'{model}_{task}_{indicator}_L{LAYER}'
        print(f'\n=== {key} ===', flush=True)
        sp, meta = load_sae_and_meta(model, task, LAYER)
        if sp is None: continue

        if indicator == 'i_lc':
            target, balances = compute_loss_chasing_continuous(meta, model, task)
        elif indicator == 'i_ba':
            target, balances = compute_iba(meta, model, task)
        elif indicator == 'i_ec':
            iba_result = compute_iba(meta, model, task)
            if iba_result is None: continue
            bet_ratios, balances = iba_result
            target = np.where(np.isnan(bet_ratios), np.nan, (bet_ratios >= 0.5).astype(float))
        if target is None: continue

        bt = meta['bet_types']
        valid = (bt == 'variable') & ~np.isnan(target) & ~np.isnan(balances) & (balances > 0)
        if indicator == 'i_ba': valid = valid & (target > 0)
        n = int(valid.sum())
        if n < 100:
            print(f'  SKIP n={n}'); continue

        X_sparse = sp[valid]
        t = target[valid]; bal = balances[valid]
        rn = meta['round_nums'][valid].astype(float)
        gids = meta['game_ids'][valid] if hasattr(meta['game_ids'], '__getitem__') else meta['game_ids']
        nnz = np.diff(X_sparse.tocsc().indptr)
        active = np.where(nnz > 10)[0]
        X = X_sparse[:, active].toarray()

        # Observed R²
        t0 = time.time()
        observed = fit_cv(X, t, bal, rn)
        print(f'  observed R² = {observed:+.4f} ({time.time()-t0:.0f}s)', flush=True)

        # Game-block permutation null
        rng = np.random.default_rng(42)
        unique_games = np.unique(gids)
        null_r2s = []
        for p_iter in range(N_PERM):
            perm_gids = rng.permutation(unique_games)
            new_targets = np.empty_like(t)
            for orig_g, new_g in zip(unique_games, perm_gids):
                old_mask = gids == orig_g
                new_mask = gids == new_g
                n_old = old_mask.sum()
                if n_old == new_mask.sum():
                    new_targets[old_mask] = t[new_mask]
                else:
                    new_targets[old_mask] = rng.choice(t[new_mask], size=n_old, replace=True)
            try:
                r = fit_cv(X, new_targets, bal, rn)
                if r is not None: null_r2s.append(r)
            except Exception:
                pass
        if null_r2s:
            null_arr = np.array(null_r2s)
            perm_p = float((null_arr >= observed).sum() / len(null_arr))
            print(f'  perm null (n={len(null_r2s)}): mean={null_arr.mean():+.4f}, '
                  f'95%ile={np.percentile(null_arr, 95):.4f}, p={perm_p:.4f}', flush=True)
        else:
            perm_p = None

        results[key] = {'n': n, 'observed_r2': observed, 'perm_p': perm_p,
                        'n_perm': len(null_r2s) if null_r2s else 0,
                        'null_mean': float(np.mean(null_r2s)) if null_r2s else None,
                        'null_95': float(np.percentile(null_r2s, 95)) if null_r2s else None}
        with open(OUT, 'w') as f:
            json.dump(results, f, indent=2)

    print(f'\nSaved {OUT}')


if __name__ == '__main__':
    main()
