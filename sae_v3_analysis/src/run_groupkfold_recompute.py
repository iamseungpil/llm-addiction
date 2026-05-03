"""GroupKFold (by game_id) recomputation for §4.1 + §4.3 — fixes within-game leakage.

§4.1: Table 1 cells — 6 (model, task) × 3 indicators at the layer set by the
      module-level constant LAYER (default 22 = body-cited).
§4.3: Condition modulation — same cells × {plus_G, minus_G, plus_M, minus_M,
      all_variable, fixed_all} at LAYER.

Output (filenames automatically follow LAYER, default 22):
  results/table1_groupkfold_L{LAYER}.json
  results/condition_modulation_groupkfold_L{LAYER}.json

For appendix C.2 layer sweep across {L8, L12, L25, L30}, use the wrapper:
  python run_groupkfold_layer_sweep.py --layer 8

Pipeline (from run_perm_null_ilc.py, paper-canonical):
  - Top-K=200 SAE features by |Spearman ρ| with deconfounded target
  - Within-fold RandomForest deconfound on [bal, rn, bal², log1p(bal), bal·rn]
  - StandardScaler + Ridge(α=100)
  - 5-fold GroupKFold by game_id (no shuffle; deterministic group→fold map)
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

sys.path.insert(0, '/home/v-seungplee/llm-addiction/sae_v3_analysis/src')
from run_perm_null_ilc import (
    load_sae_and_meta, nl_deconfound_split, BEHAVIORAL_ROOT, TOP_K, RIDGE_ALPHA,
)
from run_comprehensive_robustness import compute_iba

LAYER = 22
RESULTS_DIR = Path('/home/v-seungplee/llm-addiction/sae_v3_analysis/results')

CELLS = [
    ('gemma', 'sm'), ('gemma', 'ic'), ('gemma', 'mw'),
    ('llama', 'sm'), ('llama', 'ic'), ('llama', 'mw'),
]


def compute_loss_chasing_continuous(meta, model, paradigm):
    """§2/§3 continuous I_LC formula — copied from run_table1_perm_null.py for parity."""
    if paradigm == 'sm':
        if model == 'gemma':
            gpath = BEHAVIORAL_ROOT / 'slot_machine/gemma_v4_role/final_gemma_20260227_002507.json'
        else:
            gpath = BEHAVIORAL_ROOT / 'slot_machine/llama_v4_role/final_llama_20260315_062428.json'
        raw = json.load(open(gpath))
        games_data = raw.get('results', raw.get('games', []))
        if isinstance(games_data, dict):
            games_data = list(games_data.values())
    elif paradigm == 'mw':
        mw_dir = BEHAVIORAL_ROOT / 'mystery_wheel' / (
            'gemma_v2_role' if model == 'gemma' else 'llama_v2_role')
        games_data = []
        for f in sorted(mw_dir.glob(f'{model}_mysterywheel_*.json')):
            d = json.load(open(f))
            r = d.get('results', d.get('games', []))
            if isinstance(r, dict):
                games_data.extend(r.values())
            else:
                games_data.extend(r)
    elif paradigm == 'ic':
        ic_dir = BEHAVIORAL_ROOT / 'investment_choice' / (
            'v2_role_gemma' if model == 'gemma' else 'v2_role_llama')
        games_data = []
        for f in sorted(ic_dir.glob('*.json')):
            d = json.load(open(f))
            r = d.get('results', d.get('games', []))
            if isinstance(r, dict):
                games_data.extend(r.values())
            else:
                games_data.extend(r)
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
        decs = [d for d in raw_decs if d.get('action') != 'skip' and not d.get('skipped', False)]
        hist = g.get('history', decs)
        if rn >= len(decs) or rn < 1:
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
            prev_loss = not hist[rn - 1].get('win', str(hist[rn - 1].get('result', '')) == 'W')
        if prev_loss and p_br > 0:
            lc[i] = max(0.0, (br - p_br) / p_br)
    return lc, balances_out


def get_meta_field(meta, field, default=None):
    """Best-effort field accessor for prompt_combo / prompt_condition."""
    for k in (field, f'{field}s', 'prompt_combo', 'prompt_condition',
              'prompt_combos', 'prompt_conditions'):
        if k in meta and meta[k] is not None:
            return meta[k]
    return default


def fit_groupkfold(X_dense, target, balances, rounds, groups, n_splits=5):
    """5-fold GroupKFold (by game_id) with within-fold RF deconfound + Top-200 Spearman.

    Returns: (mean_r2, std_r2)
    """
    n_feat = X_dense.shape[1]
    k = min(TOP_K, n_feat)
    n_unique_groups = len(np.unique(groups))
    if n_unique_groups < n_splits:
        n_splits = max(2, n_unique_groups // 2)
    if n_unique_groups < 4:
        return None, None
    gkf = GroupKFold(n_splits=n_splits)
    r2s = []
    for tr, te in gkf.split(X_dense, groups=groups):
        try:
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
            sc = StandardScaler()
            Xtr = sc.fit_transform(X_dense[tr][:, idx])
            Xte = sc.transform(X_dense[te][:, idx])
            pred = Ridge(alpha=RIDGE_ALPHA).fit(Xtr, res_tr).predict(Xte)
            r2s.append(r2_score(res_te, pred))
        except Exception as e:
            print(f'    fold error: {type(e).__name__}: {e}')
    if not r2s:
        return None, None
    return float(np.mean(r2s)), float(np.std(r2s, ddof=1) if len(r2s) > 1 else 0.0)


def fit_one_subset(meta, sp, model, task, indicator, valid_filter=None):
    """Build (X, target, bal, rn, groups) for one cell, optionally filtered by prompt subset.

    valid_filter: None (all-variable) or one of 'plus_G', 'minus_G', 'plus_M', 'minus_M', 'fixed_all'.
    Returns dict with r2_mean, r2_std, n.
    """
    if indicator == 'i_lc':
        target, balances = compute_loss_chasing_continuous(meta, model, task)
    elif indicator == 'i_ba':
        result = compute_iba(meta, model, task)
        if result is None:
            return {'reason': 'compute_iba returned None'}
        target, balances = result
    elif indicator == 'i_ec':
        result = compute_iba(meta, model, task)
        if result is None:
            return {'reason': 'compute_iba returned None'}
        bet_ratios, balances = result
        target = np.where(np.isnan(bet_ratios), np.nan,
                          (bet_ratios >= 0.5).astype(float))

    if target is None:
        return {'reason': 'target is None'}

    bt = meta['bet_types']
    valid = (bt == 'variable') & ~np.isnan(target) & ~np.isnan(balances) & (balances > 0)
    if indicator == 'i_ba':
        valid = valid & (target > 0)

    # Optional prompt-condition filter
    pc = get_meta_field(meta, 'prompt_combo')
    if valid_filter is not None and pc is not None:
        if valid_filter == 'plus_G':
            valid = valid & np.array(['G' in str(p) for p in pc])
        elif valid_filter == 'minus_G':
            valid = valid & np.array(['G' not in str(p) for p in pc])
        elif valid_filter == 'plus_M':
            valid = valid & np.array(['M' in str(p) for p in pc])
        elif valid_filter == 'minus_M':
            valid = valid & np.array(['M' not in str(p) for p in pc])
        elif valid_filter == 'fixed_all':
            valid = (~np.isnan(target)) & ~np.isnan(balances) & (balances > 0) & (bt == 'fixed')
            if indicator == 'i_ba':
                valid = valid & (target > 0)

    n = int(valid.sum())
    if n < 100:
        return {'reason': f'n<100 ({n})', 'n': n}

    X_sparse = sp[valid]
    t = target[valid]
    bal = balances[valid]
    rn = meta['round_nums'][valid].astype(float)
    gids = np.asarray(meta['game_ids'])[valid]

    nnz = np.diff(X_sparse.tocsc().indptr)
    active = np.where(nnz > 10)[0]
    X = X_sparse[:, active].toarray()
    if X.shape[1] == 0:
        return {'reason': 'no active features', 'n': n}

    n_groups = len(np.unique(gids))
    r2_mean, r2_std = fit_groupkfold(X, t, bal, rn, gids)
    return {
        'n': n,
        'n_groups': int(n_groups),
        'r2_mean': r2_mean,
        'r2_std': r2_std,
        'mean_target': float(np.nanmean(t)) if r2_mean is not None else None,
    }


def main():
    # ---- §4.1 Table 1 (all_variable) ----
    print('=== §4.1 Table 1: GroupKFold by game_id ===')
    table1 = {}
    t0 = time.time()
    for model, task in CELLS:
        for indicator in ['i_lc', 'i_ba', 'i_ec']:
            key = f'{model}_{task}_{indicator}_L{LAYER}'
            print(f'\n[{time.time()-t0:6.0f}s] === {key} ===', flush=True)
            sp, meta = load_sae_and_meta(model, task, LAYER)
            if sp is None:
                print('  SAE missing')
                continue
            res = fit_one_subset(meta, sp, model, task, indicator)
            table1[key] = res
            if res.get('r2_mean') is not None:
                print(f'  n={res["n"]} groups={res["n_groups"]} '
                      f'R²={res["r2_mean"]:+.4f} ± {res["r2_std"]:.4f}', flush=True)
            else:
                print(f'  SKIP: {res.get("reason")}', flush=True)
            with open(RESULTS_DIR / f'table1_groupkfold_L{LAYER}.json', 'w') as f:
                json.dump(table1, f, indent=2)

    # ---- §4.3 condition modulation ----
    print('\n\n=== §4.3 condition modulation: GroupKFold by game_id ===')
    cm = {}
    for model, task in CELLS:
        for indicator in ['i_lc', 'i_ba', 'i_ec']:
            key = f'{model}_{task}_{indicator}_L{LAYER}'
            print(f'\n[{time.time()-t0:6.0f}s] === {key} ===', flush=True)
            sp, meta = load_sae_and_meta(model, task, LAYER)
            if sp is None:
                continue
            cm[key] = {'model': model, 'task': task, 'indicator': indicator,
                       'layer': LAYER, 'subsets': {}}
            for subset in ['all_variable', 'plus_G', 'minus_G', 'plus_M', 'minus_M', 'fixed_all']:
                vf = None if subset == 'all_variable' else subset
                res = fit_one_subset(meta, sp, model, task, indicator, valid_filter=vf)
                cm[key]['subsets'][subset] = res
                rm = res.get('r2_mean')
                if rm is not None:
                    print(f'  {subset:12s}  n={res["n"]:>5}  groups={res["n_groups"]:>5}  '
                          f'R²={rm:+.4f}', flush=True)
                else:
                    print(f'  {subset:12s}  SKIP: {res.get("reason")}', flush=True)
                with open(RESULTS_DIR / f'condition_modulation_groupkfold_L{LAYER}.json', 'w') as f:
                    json.dump(cm, f, indent=2)

    print(f'\nDone in {time.time()-t0:.0f}s')
    print(f'Saved: {RESULTS_DIR / f"table1_groupkfold_L{LAYER}.json"}')
    print(f'Saved: {RESULTS_DIR / f"condition_modulation_groupkfold_L{LAYER}.json"}')


if __name__ == '__main__':
    main()
