"""E3: balance-stratified §4.2 BK contrast audit.

Reviewer concern: the LOTO-PCA shared-axis AUC=0.74/0.80 (Gemma IC/SM)
might just reflect balance differences — bankruptcy rounds happen when
balance is low; voluntary-stop rounds when balance is high. The "shared
geometry" might encode "low balance" rather than "risky outcome."

Test: rerun the §4.2 BK audit with two balance-controlled variants:
  V1. Balance-matched subsample: per task, draw a BK / voluntary-stop
      subsample with overlapping balance distributions (stratified by
      balance percentile). Re-compute BK direction + LOTO PCA AUC.
  V2. Balance-residual hidden state: regress balance out of each h-dim
      (linear residualization), then re-compute BK direction + LOTO.

If shared-axis AUC remains substantially above the random-direction
baseline under both V1 and V2, the shared geometry is not a balance
artefact.

Output:
  results/v19_multi_patching/E3_balance_stratified/{model}_L{layer}.json
  results/v19_multi_patching/E3_balance_stratified/_summary.md
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '/home/v-seungplee/llm-addiction/sae_v3_analysis/src')

DATA = Path('/home/v-seungplee/data/llm-addiction/sae_features_v3')
OUT_DIR = Path('/home/v-seungplee/llm-addiction/sae_v3_analysis/results/'
               'v19_multi_patching/E3_balance_stratified')
TASK_DIRS = {'sm': 'slot_machine', 'ic': 'investment_choice', 'mw': 'mystery_wheel'}


def load_task(model: str, task: str, layer: int):
    f = DATA / TASK_DIRS[task] / model / 'hidden_states_dp.npz'
    d = np.load(f, allow_pickle=False)
    layers = list(d['layers'])
    if layer not in layers:
        return None
    li = layers.index(layer)
    H = d['hidden_states'][:, li, :].astype(np.float32)
    out = d['game_outcomes']
    bal = d['balances'].astype(np.float32)
    valid = ((out == 'bankruptcy') | (out == 'voluntary_stop')) & ~np.isnan(bal)
    return {
        'H': H[valid],
        'bk': (out[valid] == 'bankruptcy').astype(int),
        'bal': bal[valid],
    }


def bk_direction(H, bk):
    v = H[bk == 1].mean(0) - H[bk == 0].mean(0)
    return v / max(np.linalg.norm(v), 1e-12)


def balance_matched_subsample(d: dict, n_bins: int = 6, seed: int = 42) -> dict:
    """Stratified subsample so BK and voluntary-stop have overlapping balance distributions.

    Strategy: bin balance into n_bins equal-population quantiles using full
    dataset; within each bin keep min(n_bk, n_st) of each class.
    """
    bal = d['bal']
    bk = d['bk']
    if bk.sum() == 0 or (bk == 0).sum() == 0:
        return d
    # Quantile bins from full balance distribution
    qs = np.quantile(bal, np.linspace(0, 1, n_bins + 1))
    qs[0] -= 1.0; qs[-1] += 1.0  # avoid edge issues
    bins = np.digitize(bal, qs[1:-1])  # 0..n_bins-1

    rng = np.random.RandomState(seed)
    keep = []
    for b in range(n_bins):
        idx_bk = np.where((bins == b) & (bk == 1))[0]
        idx_st = np.where((bins == b) & (bk == 0))[0]
        n_take = min(len(idx_bk), len(idx_st))
        if n_take == 0:
            continue
        keep.append(rng.choice(idx_bk, n_take, replace=False))
        keep.append(rng.choice(idx_st, n_take, replace=False))
    if not keep:
        return None
    keep = np.concatenate(keep)
    return {'H': d['H'][keep], 'bk': d['bk'][keep], 'bal': d['bal'][keep]}


def residualize_balance(H: np.ndarray, bal: np.ndarray) -> np.ndarray:
    """Per-dim linear residual: H_d - (a_d * bal + b_d)."""
    bal_c = bal - bal.mean()
    var = (bal_c ** 2).sum()
    if var <= 0:
        return H
    coef = (H.T @ bal_c) / var  # (D,)
    pred = bal_c[:, None] * coef[None, :]  # (n, D)
    return H - pred


def loto_audit(per_task: dict[str, dict], n_random: int = 30, seed: int = 42) -> dict:
    """Run cosine + LOTO PCA + 1D transfer audit."""
    tasks = list(per_task.keys())
    for t in tasks:
        per_task[t]['dir'] = bk_direction(per_task[t]['H'], per_task[t]['bk'])

    cos = {}
    for i, ta in enumerate(tasks):
        for tb in tasks[i+1:]:
            cos[f'{ta}_{tb}'] = float(
                per_task[ta]['dir'] @ per_task[tb]['dir'] /
                max(np.linalg.norm(per_task[ta]['dir']) *
                    np.linalg.norm(per_task[tb]['dir']), 1e-12))

    transfer = {}
    for src in tasks:
        for tgt in tasks:
            if src == tgt:
                continue
            scores = per_task[tgt]['H'] @ per_task[src]['dir']
            if per_task[tgt]['bk'].sum() == 0 or (per_task[tgt]['bk'] == 0).sum() == 0:
                transfer[f'{src}_to_{tgt}'] = None
                continue
            transfer[f'{src}_to_{tgt}'] = float(roc_auc_score(per_task[tgt]['bk'], scores))

    loto = {}
    rng = np.random.default_rng(seed)
    for held in tasks:
        others = [t for t in tasks if t != held]
        contrasts = np.stack([per_task[t]['dir'] for t in others], axis=0)
        Vt = np.linalg.svd(contrasts, full_matrices=False)[2]
        shared = Vt[0]
        scores = per_task[held]['H'] @ shared
        bk_held = per_task[held]['bk']
        if bk_held.sum() == 0 or (bk_held == 0).sum() == 0:
            loto[held] = None; continue
        auc_shared = float(roc_auc_score(bk_held, scores))
        if auc_shared < 0.5:
            auc_shared = 1 - auc_shared
        rand_aucs = []
        for _ in range(n_random):
            r = rng.normal(0, 1, shared.shape[0])
            r /= np.linalg.norm(r)
            s = per_task[held]['H'] @ r
            a = float(roc_auc_score(bk_held, s))
            if a < 0.5: a = 1 - a
            rand_aucs.append(a)
        loto[held] = {
            'auc_shared': auc_shared,
            'rand_mean': float(np.mean(rand_aucs)),
            'rand_p95': float(np.percentile(rand_aucs, 95)),
            'gain_vs_rand': auc_shared - float(np.mean(rand_aucs)),
        }

    return {'cosines': cos, 'transfer_auc': transfer, 'loto_pca': loto}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['gemma', 'llama'], default=None,
                    help='if omitted: run gemma + llama')
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = [('gemma', 22), ('llama', 22)]
    if args.model:
        targets = [t for t in targets if t[0] == args.model]

    summary_lines = ['# E3 — balance-stratified §4.2 BK audit', '']
    summary_lines.append(
        'Reviewer concern: the §4.2 LOTO-PCA shared-axis AUC may track')
    summary_lines.append(
        'balance rather than risk-ending semantics. We rerun §4.2 with')
    summary_lines.append(
        'balance-matched subsamples (V1) and balance-residualised hidden')
    summary_lines.append(
        'states (V2). If shared-axis AUC remains >> random baseline,')
    summary_lines.append('the geometry is not a balance artefact.')
    summary_lines.append('')

    for model, layer in targets:
        per_task = {}
        per_task_v2 = {}
        for task in ['sm', 'ic', 'mw']:
            d = load_task(model, task, layer)
            if d is None:
                print(f'[{model}/{task}/L{layer}] missing layer; skip cell')
                per_task = None
                break
            per_task[task] = d
            d2 = {'H': residualize_balance(d['H'], d['bal']),
                  'bk': d['bk'], 'bal': d['bal']}
            per_task_v2[task] = d2

        if per_task is None:
            continue

        result = {'model': model, 'layer': layer, 'variants': {}}

        # Baseline (no balance control) — for sanity
        result['variants']['baseline'] = loto_audit({t: per_task[t] for t in per_task})

        # V1: balance-matched subsample
        v1 = {}
        for t, d in per_task.items():
            sub = balance_matched_subsample(d)
            if sub is None or sub['bk'].sum() < 5 or (sub['bk'] == 0).sum() < 5:
                v1 = None; break
            v1[t] = sub
        if v1 is not None:
            result['variants']['v1_balance_matched'] = loto_audit(v1)
            result['variants']['v1_balance_matched']['n_per_task'] = {
                t: {'n_bk': int(v1[t]['bk'].sum()),
                    'n_st': int((v1[t]['bk'] == 0).sum())}
                for t in v1}

        # V2: balance-residualised hidden states
        result['variants']['v2_balance_residualised'] = loto_audit(per_task_v2)

        # n per task
        result['n_per_task_baseline'] = {
            t: {'n_bk': int(per_task[t]['bk'].sum()),
                'n_st': int((per_task[t]['bk'] == 0).sum())}
            for t in per_task}

        out_path = OUT_DIR / f'{model}_L{layer}.json'
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'[save] {out_path}')

        summary_lines.append(f'## {model.upper()} L{layer}')
        summary_lines.append('')
        summary_lines.append('Shared-axis AUC by held-out task and variant:')
        summary_lines.append('')
        summary_lines.append(
            '| held-out | baseline | V1 (bal-matched) | V2 (bal-residual) | rand baseline |')
        summary_lines.append('|---|---|---|---|---|')
        for held in ['sm', 'ic', 'mw']:
            row = [held]
            for var in ['baseline', 'v1_balance_matched', 'v2_balance_residualised']:
                v = result['variants'].get(var, {})
                lp = (v.get('loto_pca') or {}).get(held)
                if lp is None:
                    row.append('—')
                else:
                    row.append(f"{lp['auc_shared']:.3f}")
            v = result['variants'].get('baseline', {})
            lp = (v.get('loto_pca') or {}).get(held)
            row.append(f"{lp['rand_mean']:.3f}" if lp else '—')
            summary_lines.append('| ' + ' | '.join(row) + ' |')
        summary_lines.append('')

    with open(OUT_DIR / '_summary.md', 'w') as f:
        f.write('\n'.join(summary_lines) + '\n')
    print(f'[save] {OUT_DIR / "_summary.md"}')


if __name__ == '__main__':
    main()
