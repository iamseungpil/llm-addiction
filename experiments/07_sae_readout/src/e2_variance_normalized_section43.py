"""E2: balance-stratified §4.3 modulation re-analysis.

Reviewer concern: "Δ R² between +G and -G subsets may just reflect balance
distribution differences, not condition modulation per se." This script
re-runs §4.3 modulation while restricting each subset to a fixed balance
window, so the contrast is held within a comparable balance regime.

For each (model, task, indicator) cell at L22:
  1. Compute balance distribution percentiles across 'all_variable' subset
  2. Define overlapping balance windows (Q1=p10-p40, Q2=p30-p70, Q3=p60-p90)
  3. For each subset × window, refit the §4.3 strict 5-fold CV pipeline
     (within-fold RF deconfound + Top-K=200 + Ridge α=100)
  4. Report R² per (subset, window) cell + Δ R² (+G vs -G) per window

If Δ R² flips sign across balance windows: modulation is balance-driven.
If Δ R² stays consistently positive: condition modulation is real.

Output: results/v19_multi_patching/E2_variance_normalized/{model}_{task}.json
        results/v19_multi_patching/E2_variance_normalized/_summary.md
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, '/home/v-seungplee/llm-addiction/sae_v3_analysis/src')
from run_perm_null_ilc import load_sae_and_meta, TOP_K, RIDGE_ALPHA  # noqa: E402
from run_condition_modulation_continuous_ilc import (  # noqa: E402
    get_label, condition_mask, fit_strict_cv,
)

OUT_DIR = Path('/home/v-seungplee/llm-addiction/sae_v3_analysis/results/'
               'v19_multi_patching/E2_variance_normalized')

CELLS = [
    ('gemma', 'sm', 'i_ba', 22),
    ('gemma', 'sm', 'i_lc', 22),
    ('llama', 'sm', 'i_ba', 22),
    ('llama', 'sm', 'i_lc', 22),
]

CONTRAST_PAIRS = [('plus_G', 'minus_G'), ('plus_M', 'minus_M')]

# Overlapping balance windows by percentile (within all_variable)
WINDOWS = {
    'Q_low':  (10.0, 40.0),
    'Q_mid':  (30.0, 70.0),
    'Q_high': (60.0, 90.0),
}


def fit_within_window(sp, meta, target, balances, rounds, mask, win_lo, win_hi):
    valid = mask & ~np.isnan(target) & ~np.isnan(balances) & (balances > 0)
    valid = valid & (balances >= win_lo) & (balances <= win_hi)
    n = int(valid.sum())
    if n < 100:
        return None, n
    X_sparse = sp[valid]
    t = target[valid]
    bal = balances[valid]
    rn = rounds[valid]
    nnz = np.diff(X_sparse.tocsc().indptr)
    active = np.where(nnz > 10)[0]
    if active.size == 0:
        return None, n
    X = X_sparse[:, active].toarray()
    try:
        r2 = fit_strict_cv(X, t, bal, rn, top_k=TOP_K, alpha=RIDGE_ALPHA)
        return r2, n
    except Exception as e:
        return f'ERROR: {type(e).__name__}: {e}', n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', help='optional filter like gemma_sm_i_ba_L22')
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    overall_summary = {}
    for model, task, indicator, layer in CELLS:
        key = f'{model}_{task}_{indicator}_L{layer}'
        if args.cell and args.cell != key:
            continue
        print(f'\n=== {key} ===', flush=True)

        sp, meta = load_sae_and_meta(model, task, layer)
        if sp is None:
            print('  SKIP: no SAE'); continue
        target, balances = get_label(meta, model, task, indicator)
        if target is None:
            print('  SKIP: no labels'); continue

        rounds = meta['round_nums'].astype(float)

        all_var_mask = condition_mask(meta, 'all_variable')
        valid_av = all_var_mask & ~np.isnan(target) & ~np.isnan(balances) & (balances > 0)
        if indicator == 'i_ba':
            valid_av = valid_av & (target > 0)
        if int(valid_av.sum()) < 100:
            print(f'  SKIP: all_variable n<100'); continue

        bal_var = balances[valid_av]
        ptiles = {name: (np.percentile(bal_var, lo), np.percentile(bal_var, hi))
                  for name, (lo, hi) in WINDOWS.items()}
        print(f'  balance percentile windows:', flush=True)
        for name, (lo, hi) in ptiles.items():
            print(f'    {name}: [${lo:.0f}, ${hi:.0f}]', flush=True)

        cell_results = {
            'model': model, 'task': task, 'indicator': indicator, 'layer': layer,
            'balance_windows': {n: {'lo': lo, 'hi': hi} for n, (lo, hi) in ptiles.items()},
            'subsets': {},
            'contrasts': {},
        }

        for win_name, (win_lo, win_hi) in ptiles.items():
            print(f'  [window={win_name}]', flush=True)
            cell_results['subsets'][win_name] = {}
            for subset in ['plus_G', 'minus_G', 'plus_M', 'minus_M']:
                mask = condition_mask(meta, subset)
                if indicator == 'i_ba':
                    mask = mask & (target > 0)
                t0 = time.time()
                r2, n = fit_within_window(sp, meta, target, balances, rounds,
                                          mask, win_lo, win_hi)
                cell_results['subsets'][win_name][subset] = {
                    'r2': r2 if not isinstance(r2, str) else None,
                    'note': r2 if isinstance(r2, str) else None,
                    'n': n,
                }
                if isinstance(r2, str) or r2 is None:
                    print(f'    {subset}: n={n} -> {r2}', flush=True)
                else:
                    print(f'    {subset}: n={n}, R²={r2:+.4f} ({time.time()-t0:.0f}s)',
                          flush=True)

            cell_results['contrasts'][win_name] = {}
            for plus_key, minus_key in CONTRAST_PAIRS:
                plus = cell_results['subsets'][win_name].get(plus_key, {}).get('r2')
                minus = cell_results['subsets'][win_name].get(minus_key, {}).get('r2')
                delta = (plus - minus) if (plus is not None and minus is not None) else None
                cell_results['contrasts'][win_name][f'{plus_key}_minus_{minus_key}'] = delta

        out_path = OUT_DIR / f'{key}.json'
        with open(out_path, 'w') as f:
            json.dump(cell_results, f, indent=2)
        print(f'  [save] {out_path}', flush=True)
        overall_summary[key] = cell_results

    summary_md = OUT_DIR / '_summary.md'
    lines = ['# E2 — balance-stratified §4.3 modulation', '']
    lines.append('Reviewer concern: Δ R² between condition subsets may be a balance confound.')
    lines.append('Test: refit §4.3 pipeline within fixed balance windows. If condition')
    lines.append('modulation is real, Δ R² should remain positive across all windows.')
    lines.append('')
    for key, cell in overall_summary.items():
        lines.extend(['', f'## {key}', '',
                      '| balance window | +G R² | −G R² | Δ(+G,−G) | +M R² | −M R² | Δ(+M,−M) |',
                      '|---|---|---|---|---|---|---|'])
        for win_name in WINDOWS:
            ss = cell['subsets'].get(win_name, {})
            cs = cell['contrasts'].get(win_name, {})

            def fmt(v):
                if v is None:
                    return '—'
                return f'{v:+.4f}'

            lines.append(
                f'| {win_name} '
                f'| {fmt(ss.get("plus_G", {}).get("r2"))} '
                f'| {fmt(ss.get("minus_G", {}).get("r2"))} '
                f'| {fmt(cs.get("plus_G_minus_minus_G"))} '
                f'| {fmt(ss.get("plus_M", {}).get("r2"))} '
                f'| {fmt(ss.get("minus_M", {}).get("r2"))} '
                f'| {fmt(cs.get("plus_M_minus_minus_M"))} |')

    with open(summary_md, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\n[save] {summary_md}', flush=True)


if __name__ == '__main__':
    main()
