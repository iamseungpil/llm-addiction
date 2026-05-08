"""§4.2 BK contrast audit at consistent peak layer (Gemma L23, LLaMA L31).

Computes for each model at the body §4.1's chosen layer:
  - BK direction per task (mean(hs|bankruptcy) - mean(hs|voluntary_stop))
  - Cosine alignment between per-task BK directions
  - Sparse-feature transfer R² (off-diagonal: train on task A's BK direction
    via SAE features, test on task B's BK label)
  - LOTO PCA: rank-1 shared direction from 2 tasks' BK contrasts; AUC of
    held-out task's BK separation along that shared direction
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, '/home/v-seungplee/llm-addiction/sae_v3_analysis/src')

DATA = Path('/home/v-seungplee/data/llm-addiction/sae_features_v3')
OUT = Path('/home/v-seungplee/llm-addiction/sae_v3_analysis/results/rq2_audit_consistent_layer.json')
TASK_DIRS = {'sm': 'slot_machine', 'ic': 'investment_choice', 'mw': 'mystery_wheel'}


def load_hidden_bk(model: str, task: str, layer: int):
    """Return (H, bk_labels) for task: rows are bk-or-voluntary rounds."""
    f = DATA / TASK_DIRS[task] / model / 'hidden_states_dp.npz'
    d = np.load(f, allow_pickle=False)
    layers = list(d['layers'])
    if layer not in layers:
        return None, None
    li = layers.index(layer)
    H = d['hidden_states'][:, li, :]
    out = d['game_outcomes']
    valid = (out == 'bankruptcy') | (out == 'voluntary_stop')
    bk = (out[valid] == 'bankruptcy').astype(int)
    return H[valid], bk


def bk_dir(H, bk):
    v = H[bk == 1].mean(0) - H[bk == 0].mean(0)
    norm = np.linalg.norm(v)
    return v / max(norm, 1e-12)


def cosine(u, v):
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))


def compute_audits(model: str, layer: int):
    """Per-task BK directions + cosines + LOTO PCA + 1D linear-probe transfer."""
    tasks = ['sm', 'ic', 'mw']
    data = {}
    for t in tasks:
        res = load_hidden_bk(model, t, layer)
        if res[0] is None:
            return {'error': f'layer {layer} not in {model}/{t}'}
        H, bk = res
        data[t] = {'H': H, 'bk': bk, 'dir': bk_dir(H, bk),
                   'n_bk': int(bk.sum()), 'n_st': int((bk == 0).sum())}

    # --- (i) Cosine alignment per pair ---
    cosines = {}
    for i, ta in enumerate(tasks):
        for tb in tasks[i+1:]:
            cosines[f'{ta}_{tb}'] = cosine(data[ta]['dir'], data[tb]['dir'])

    # --- (ii) Cross-task BK direction transfer (1D linear probe) ---
    # Apply task A's BK direction to task B's hidden states; AUC for B's BK label
    transfer_auc = {}
    for src in tasks:
        for tgt in tasks:
            if src == tgt: continue
            scores = data[tgt]['H'] @ data[src]['dir']
            try:
                auc = roc_auc_score(data[tgt]['bk'], scores)
            except Exception:
                auc = None
            transfer_auc[f'{src}_to_{tgt}'] = auc

    # --- (iii) LOTO PCA: rank-1 shared direction from 2 tasks ---
    loto = {}
    for held in tasks:
        others = [t for t in tasks if t != held]
        contrasts = np.stack([data[t]['dir'] for t in others], axis=0)  # (2, D)
        U, S, Vt = np.linalg.svd(contrasts, full_matrices=False)
        shared = Vt[0]  # rank-1 shared direction
        scores = data[held]['H'] @ shared
        try:
            auc_shared = roc_auc_score(data[held]['bk'], scores)
            if auc_shared < 0.5:
                auc_shared = 1 - auc_shared  # orientation
        except Exception:
            auc_shared = None
        # Random-direction baseline (mean of 30 random unit vectors)
        rng = np.random.default_rng(42)
        random_aucs = []
        for _ in range(30):
            r = rng.normal(0, 1, shared.shape[0])
            r /= np.linalg.norm(r)
            s = data[held]['H'] @ r
            try:
                a = roc_auc_score(data[held]['bk'], s)
                if a < 0.5: a = 1 - a
                random_aucs.append(a)
            except Exception:
                pass
        rand_mean = float(np.mean(random_aucs)) if random_aucs else None
        # Label-shuffle baseline
        shuf_aucs = []
        for _ in range(30):
            shuf_bk = rng.permutation(data[held]['bk'])
            try:
                a = roc_auc_score(shuf_bk, scores)
                if a < 0.5: a = 1 - a
                shuf_aucs.append(a)
            except Exception:
                pass
        shuf_mean = float(np.mean(shuf_aucs)) if shuf_aucs else None
        loto[held] = {
            'auc_shared': auc_shared,
            'rand_mean': rand_mean,
            'shuf_mean': shuf_mean,
            'vs_rand': auc_shared - rand_mean if (auc_shared and rand_mean) else None,
            'vs_shuf': auc_shared - shuf_mean if (auc_shared and shuf_mean) else None,
        }

    return {
        'model': model, 'layer': layer,
        'n_per_task': {t: {'n_bk': data[t]['n_bk'], 'n_st': data[t]['n_st']} for t in tasks},
        'cosines': cosines,
        'cross_task_transfer_auc': transfer_auc,
        'loto_pca': loto,
    }


def main():
    results = {}
    for model, layer in [('gemma', 23), ('llama', 31)]:
        print(f'\n=== {model} L{layer} ===')
        r = compute_audits(model, layer)
        results[f'{model}_L{layer}'] = r
        if 'error' in r:
            print(f'  ERROR: {r["error"]}')
            continue
        print(f'  Cosines: {r["cosines"]}')
        print(f'  Transfer AUC: {r["cross_task_transfer_auc"]}')
        print(f'  LOTO PCA: {r["loto_pca"]}')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\nSaved {OUT}')


if __name__ == '__main__':
    main()
