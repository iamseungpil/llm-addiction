"""§4.3 multi-indicator ±G/±M modulation, continuous I_LC matching §2/§3.

For each (model, task, indicator) cell at the per-indicator peak layer:
  Re-fit SAE→indicator Ridge readout SEPARATELY within each prompt condition
  (+G, -G, +M, -M, all-variable, fixed). Report 5-fold CV R^2 with strict
  within-fold RF deconfound.

I_LC labels use the §2 / §3 continuous formula:
    I_LC[t] = max(0, (br_t - br_{t-1}) / br_{t-1})  if round t-1 was a loss
            = 0                                       otherwise
This matches regenerate_fig2_fig4.py compute_sm_metrics_from_games (the
canonical §3 behavioral analysis). The previous binary version (run_perm_null_ilc
compute_loss_chasing) does not match §3 and is replaced here.
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
    load_sae_and_meta, nl_deconfound_split, BEHAVIORAL_ROOT,
    TOP_K, RIDGE_ALPHA,
)
from run_comprehensive_robustness import compute_iba

OUT = Path('/home/v-seungplee/llm-addiction/sae_v3_analysis/results/'
           'condition_modulation_continuous_ilc_L22.json')

CELLS = [
    # I_LC across all 6 (model, task)
    ('gemma','sm','i_lc',22), ('gemma','ic','i_lc',22), ('gemma','mw','i_lc',22),
    ('llama','sm','i_lc',22), ('llama','ic','i_lc',22), ('llama','mw','i_lc',22),
    # I_EC SM-only
    ('gemma','sm','i_ec',22), ('llama','sm','i_ec',22),
    # I_BA also re-runs for parity
    ('gemma','sm','i_ba',22), ('gemma','mw','i_ba',22),
    ('llama','sm','i_ba',22), ('llama','mw','i_ba',22),
]

CONDITIONS = ['plus_G', 'minus_G', 'plus_M', 'minus_M', 'all_variable', 'fixed_all']


def compute_loss_chasing_continuous(meta, model, paradigm):
    """§2/§3 continuous I_LC formula.

    For each round t in the SAE meta dataset:
      label = max(0, (br_t - br_{t-1}) / br_{t-1})  if t-1 was a loss
            = 0                                       otherwise
    """
    if paradigm == 'sm':
        if model == 'gemma':
            gpath = BEHAVIORAL_ROOT / 'slot_machine/gemma_v4_role/final_gemma_20260227_002507.json'
        else:
            gpath = BEHAVIORAL_ROOT / 'slot_machine/llama_v4_role/final_llama_20260315_062428.json'
        with open(gpath) as f:
            raw = json.load(f)
        games_data = raw.get('results', raw.get('games', []))
        if isinstance(games_data, dict):
            games_data = list(games_data.values())
    elif paradigm == 'mw':
        mw_dir = (BEHAVIORAL_ROOT / 'mystery_wheel'
                  / ('gemma_v2_role' if model == 'gemma' else 'llama_v2_role'))
        games_data = []
        for f in sorted(mw_dir.glob(f'{model}_mysterywheel_*.json')):
            d = json.load(open(f))
            results = d.get('results', d.get('games', []))
            if isinstance(results, dict):
                games_data.extend(results.values())
            else:
                games_data.extend(results)
    elif paradigm == 'ic':
        ic_dir = (BEHAVIORAL_ROOT / 'investment_choice'
                  / ('v2_role_gemma' if model == 'gemma' else 'v2_role_llama'))
        games_data = []
        for f in sorted(ic_dir.glob('*.json')):
            d = json.load(open(f))
            results = d.get('results', d.get('games', []))
            if isinstance(results, dict):
                games_data.extend(results.values())
            else:
                games_data.extend(results)
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
        if g is None:
            continue
        raw_decs = g.get('decisions', g.get('history', []))
        decs = [d for d in raw_decs if d.get('action') != 'skip'
                and not d.get('skipped', False)]
        hist = g.get('history', decs)
        if rn >= len(decs) or rn < 1:
            if rn < len(decs):
                dec = decs[rn]
                bet_val = dec.get('parsed_bet') or dec.get('bet') or dec.get('bet_amount')
                bal_val = dec.get('balance_before') or dec.get('balance')
                if bal_val is not None:
                    balances_out[i] = float(bal_val)
                if bet_val is not None and float(bet_val) > 0:
                    lc[i] = 0.0  # not loss-following -> 0 contribution
            continue

        dec = decs[rn]
        prev_dec = decs[rn - 1]
        bet_val = dec.get('parsed_bet') or dec.get('bet') or dec.get('bet_amount')
        bal_val = dec.get('balance_before') or dec.get('balance')
        prev_bet = prev_dec.get('parsed_bet') or prev_dec.get('bet') or prev_dec.get('bet_amount')
        prev_bal = prev_dec.get('balance_before') or prev_dec.get('balance')
        if any(v is None for v in [bet_val, prev_bet, prev_bal]):
            continue
        try:
            bet = float(bet_val)
            bal = float(bal_val) if bal_val is not None else float(balances_out[i])
            p_bet, p_bal = float(prev_bet), float(prev_bal)
        except (ValueError, TypeError):
            continue
        if bet <= 0 or bal <= 0 or p_bal <= 0:
            continue
        balances_out[i] = bal
        br = min(bet / bal, 1.0)
        p_br = min(p_bet / p_bal, 1.0)

        prev_loss = False
        if rn - 1 < len(hist):
            h = hist[rn - 1]
            prev_loss = not h.get('win', str(h.get('result', '')) == 'W')
        # Strict §3 mirror: only loss-following rounds get a label.
        # Non-loss-following rounds remain NaN (excluded from regression).
        if prev_loss and p_br > 0:
            lc[i] = max(0.0, (br - p_br) / p_br)
    return lc, balances_out


def compute_iec(meta, model, task):
    iba_result = compute_iba(meta, model, task)
    if iba_result is None:
        return None, None
    bet_ratios, balances = iba_result
    iec = np.where(np.isnan(bet_ratios), np.nan,
                   (bet_ratios >= 0.5).astype(float))
    return iec, balances


def get_label(meta, model, task, indicator):
    if indicator == 'i_lc':
        return compute_loss_chasing_continuous(meta, model, task)
    if indicator == 'i_ba':
        return compute_iba(meta, model, task)
    if indicator == 'i_ec':
        return compute_iec(meta, model, task)
    return None, None


def condition_mask(meta, condition):
    bt = meta['bet_types']
    pc = meta.get('prompt_conditions')
    if pc is None:
        return None
    if hasattr(pc[0], 'decode'):
        pc = np.array([s.decode() for s in pc])
    if condition == 'all_variable':
        return bt == 'variable'
    if condition == 'fixed_all':
        return bt == 'fixed'
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

        with open(OUT, 'w') as f:
            json.dump(results, f, indent=2)
    print(f'\nTotal time: {time.time()-t_total:.0f}s -> {OUT}', flush=True)


if __name__ == '__main__':
    main()
