"""§4.2 cross-task BK audit, ALL layers — isolated all-layer extension.

The paper's §4.2 (e3_balance_stratified_section42 + run_spine_layer.run_crosstask)
only runs at the 5 dp layers {8,12,22,25,30}, because e3.load_task reads
hidden_states_dp.npz which carries those layers only. This module fills the gap:
the SAME §4.2 estimator (e3.bk_direction / e3.loto_audit / e3.balance_matched_subsample
/ e3.residualize_balance, imported UNMODIFIED) run at EVERY layer.

It does this WITHOUT editing any paper script. The only new thing is a loader
that produces e3.load_task's exact output {H, bk, bal} from all-layer sources:

  Route A (paper-faithful, bit-exact):  checkpoint/phase_a_hidden_states.npz
    carries raw all-layer hidden states (N, n_layers, 3584), row-aligned to the
    sibling sae_features_L*.npz. The dp decision-point set == rows where
    sae['is_last_round']==True. Taking phase_a[:, layer, :] on those rows, with
    (outcome, balance) labels from the row-aligned sae file, reproduces the dp
    file BIT-EXACTLY at every dp layer (verified this session: max abs diff 0.0,
    np.array_equal True; outcomes 0 mismatch; balances 0.0 diff). Route A exists
    only where phase_a exists: gemma x3 + llama-IC.

  Route B (all-cell, concordant): densify sae_features_L{layer}.npz to (N_dp,
    n_active_cols) and build {H, bk, bal} on the is_last_round rows. Different
    representation (SAE acts, not dense hidden) -> CONCORDANT only, never the
    §4.2 headline. Covers all 6 cells (incl. llama SM/MW which lack phase_a).

The valid filter, float32 cast, seeds and audit math are e3's, untouched, so at
the 5 dp layers Route A's numbers == run_crosstask's numbers.

Output (DISTINCT namespace — never collides with the running indicators/condition
sweep or the existing 5-layer 'crosstask' profiles):
  read_profile_{model}_crosstask42_L{layer}.json   (section name 'crosstask42')
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e3_balance_stratified_section42 as e3  # reused UNMODIFIED

TASK_DIRS = e3.TASK_DIRS  # {'sm':'slot_machine', ...}
TASK_DIRS_INV = {v: k for k, v in TASK_DIRS.items()}  # 'slot_machine' -> 'sm'
ROUTE_A_CELLS = {('gemma', 'sm'), ('gemma', 'ic'), ('gemma', 'mw'), ('llama', 'ic')}


def _cell_dir(model: str, task: str) -> Path:
    """e3.DATA is rebound to the snapshot root by _redirect; reuse it."""
    return Path(e3.DATA) / TASK_DIRS[task] / model


def _labels_from_sae(cell: Path, layer: int, any_layer: bool = False):
    """Row-aligned sae file for labels (game_outcomes/balances/is_last_round).

    phase_a carries NO labels of its own; they come from the SAME-cell sae file,
    whose label arrays are identical across layers. Route A only needs labels, so
    any_layer=True lets it use any available sae file in the cell (e.g. when only
    L22 was synced). Route B needs the requested layer's own SAE features, so it
    requires the exact file (any_layer=False).
    """
    f = cell / f'sae_features_L{layer}.npz'
    if f.exists():
        return np.load(f, allow_pickle=True)
    if any_layer:
        for g in sorted(cell.glob('sae_features_L*.npz')):
            return np.load(g, allow_pickle=True)
    return None


_DP_SLAB_CACHE: dict = {}  # cell -> (slab[dp_rows, n_layers, 3584], out, bal)


def _dp_slab(cell: Path):
    """dp-row x all-layer hidden-state slab + dp labels, computed ONCE per cell.

    A per-layer mmap column slice p['hidden_states'][:, layer, :] strides the
    whole ~12.8GB file (~55s/layer). Instead read the 3200 dp rows once (each row
    is contiguous on disk -> fast fancy-row index) into a ~1.9GB slab, so every
    per-layer call is an instant in-RAM slice. Returns None if phase_a/sae absent.
    """
    key = str(cell)
    if key in _DP_SLAB_CACHE:
        return _DP_SLAB_CACHE[key]
    pa = cell / 'checkpoint' / 'phase_a_hidden_states.npz'
    s = _labels_from_sae(cell, -1, any_layer=True)  # labels are layer-invariant
    if not pa.exists() or s is None:
        _DP_SLAB_CACHE[key] = None
        return None
    p = np.load(pa, allow_pickle=False, mmap_mode='r')
    ilr = np.where(s['is_last_round'].astype(bool))[0]
    slab = np.ascontiguousarray(p['hidden_states'][ilr])  # (n_dp, n_layers, 3584)
    out = s['game_outcomes'][ilr]
    bal = s['balances'][ilr].astype(np.float32)           # e3 casts to float32
    _DP_SLAB_CACHE[key] = (slab, out, bal)
    return _DP_SLAB_CACHE[key]


def load_task_routeA(model: str, task: str, layer: int):
    """Paper-faithful all-layer load: phase_a hidden states + dp labels.

    Reproduces e3.load_task's {H, bk, bal} bit-exactly at the dp layers, and
    extends the SAME representation to every other layer (phase_a carries all
    model layers). Returns None if phase_a or the layer is unavailable.
    """
    cell = _cell_dir(model, task)
    slab = _dp_slab(cell)
    if slab is None:
        return None
    slab_H, out, bal = slab
    n_layers = slab_H.shape[1]      # gemma 42 vs llama 32 — read, don't hardcode
    if not (0 <= layer < n_layers):
        return None
    H = slab_H[:, layer, :].astype(np.float32)
    valid = ((out == 'bankruptcy') | (out == 'voluntary_stop')) & ~np.isnan(bal)
    return {'H': H[valid], 'bk': (out[valid] == 'bankruptcy').astype(int), 'bal': bal[valid]}


def routeB_active_cols(model: str, task: str, layer: int):
    """Active SAE column ids on this cell's dp rows (for the shared-space union).

    Returns None if the cell's sae file is unavailable. Used by run_crosstask42 to
    build ONE shared column space across all three tasks before densifying, so the
    per-task H matrices share a feature dim and e3.loto_audit's cross-task matmuls
    (H @ src_dir, H @ shared) are conformable.
    """
    cell = _cell_dir(model, task)
    s = _labels_from_sae(cell, layer)
    if s is None:
        return None
    ilr = s['is_last_round'].astype(bool)
    dp_rows = np.where(ilr)[0]
    row_map = -np.ones(s['game_ids'].shape[0], dtype=np.int64)
    row_map[dp_rows] = np.arange(dp_rows.shape[0])
    sel = row_map[s['row_indices']] >= 0
    return np.unique(s['col_indices'][sel])


def load_task_routeB(model: str, task: str, layer: int, shared_cols=None):
    """All-cell concordant load: densify SAE feats on the is_last_round rows.

    Different representation than route A (SAE acts vs dense hidden) — reported as
    concordance, never the §4.2 headline. Covers cells without phase_a (llama SM/MW).

    `shared_cols` (sorted unique col ids) pins the SAE column space SHARED across
    all three tasks. Without it each task would compact its own active columns via
    np.unique, yielding incompatible feature dims (sm=504, ic=468, mw=547) and
    breaking e3.loto_audit's cross-task matmuls. When None (single-cell use) the
    loader falls back to this cell's own active columns.
    """
    cell = _cell_dir(model, task)
    s = _labels_from_sae(cell, layer)
    if s is None:
        return None
    ilr = s['is_last_round'].astype(bool)
    # Densify ONLY the dp rows x SHARED columns — never allocate the full
    # (N, 131072) ~11GB matrix. Map dp rows to compact indices, keep only the
    # triplets on those rows, then place them into the shared column basis.
    dp_rows = np.where(ilr)[0]
    row_map = -np.ones(s['game_ids'].shape[0], dtype=np.int64)
    row_map[dp_rows] = np.arange(dp_rows.shape[0])
    ri = s['row_indices']; ci = s['col_indices']; vv = s['values']
    sel = row_map[ri] >= 0
    ri_d = row_map[ri[sel]]
    ci_raw = ci[sel]
    if shared_cols is None:
        shared_cols = np.unique(ci_raw)  # fall back to this cell's own columns
    # Map each triplet's column id into the shared basis. Triplets whose column is
    # not in the shared space (cannot happen when shared_cols is the union) are
    # dropped via searchsorted-validity guard.
    pos = np.searchsorted(shared_cols, ci_raw)
    pos = np.clip(pos, 0, shared_cols.shape[0] - 1)
    in_space = shared_cols[pos] == ci_raw
    ci_d = pos[in_space]
    ri_d = ri_d[in_space]
    vv_d = vv[sel][in_space]
    H = np.zeros((dp_rows.shape[0], shared_cols.shape[0]), dtype=np.float32)
    H[ri_d, ci_d] = vv_d
    out = s['game_outcomes'][ilr]
    bal = s['balances'][ilr].astype(np.float32)
    valid = ((out == 'bankruptcy') | (out == 'voluntary_stop')) & ~np.isnan(bal)
    return {'H': H[valid], 'bk': (out[valid] == 'bankruptcy').astype(int), 'bal': bal[valid]}


LOADERS = {'A': load_task_routeA, 'B': load_task_routeB}


def run_crosstask42(model: str, layer: int, route: str, task: str = 'sm',
                    data_root=None):
    """§4.2 cross-task audit at one layer via the chosen route.

    Cross-task is per-(model, layer): build all three tasks, hold each out.
    Identical variant structure to e3.main (baseline / v1_balance_matched /
    v2_balance_residualised), computed by e3.loto_audit UNMODIFIED.

    `task` selects the held-out cell surfaced at the top level so the result is
    ALSO shaped like run_spine_layer.run_crosstask (held_out / loto_pca /
    transfer_auc / cosines) — spine_stats reads result['loto_pca']['auc_shared'].
    `data_root` (optional) rebinds e3.DATA for this call (else uses current e3.DATA).
    """
    if data_root is not None:
        e3.DATA = Path(data_root)
    load = LOADERS[route]

    # Route B: build ONE shared SAE column space (union of each task's active dp
    # columns) so all three task H matrices share a feature dim. Without this the
    # per-task np.unique compaction gives incompatible dims (sm=504, ic=468,
    # mw=547) and e3.loto_audit's cross-task matmuls fail (size 468 vs 504).
    shared_cols = None
    if route == 'B':
        cols = []
        for t in ['sm', 'ic', 'mw']:
            c = routeB_active_cols(model, t, layer)
            if c is None:
                return {'reason': f'route B unavailable for {model}/{t}/L{layer}'}
            cols.append(c)
        shared_cols = np.unique(np.concatenate(cols))

    per_task = {}
    for t in ['sm', 'ic', 'mw']:
        d = load(model, t, layer, shared_cols) if route == 'B' else load(model, t, layer)
        if d is None:
            return {'reason': f'route {route} unavailable for {model}/{t}/L{layer}'}
        per_task[t] = d

    out = {'model': model, 'layer': layer, 'route': route, 'variants': {}}
    baseline = e3.loto_audit({t: per_task[t] for t in per_task})
    out['variants']['baseline'] = baseline

    v1 = {}
    for t, d in per_task.items():
        sub = e3.balance_matched_subsample(d)
        if sub is None or sub['bk'].sum() < 5 or (sub['bk'] == 0).sum() < 5:
            v1 = None
            break
        v1[t] = sub
    if v1 is not None:
        out['variants']['v1_balance_matched'] = e3.loto_audit(v1)

    per_task_v2 = {t: {'H': e3.residualize_balance(d['H'], d['bal']),
                       'bk': d['bk'], 'bal': d['bal']}
                   for t, d in per_task.items()}
    out['variants']['v2_balance_residualised'] = e3.loto_audit(per_task_v2)

    out['n_per_task'] = {t: {'n_bk': int(per_task[t]['bk'].sum()),
                             'n_st': int((per_task[t]['bk'] == 0).sum())}
                         for t in per_task}

    # run_crosstask-shaped top-level view of the held-out task (baseline variant),
    # so spine_stats and any run_crosstask consumer can read it unchanged.
    held = task if task in per_task else 'sm'
    out['held_out'] = held
    out['loto_pca'] = baseline['loto_pca'].get(held)
    out['transfer_auc'] = baseline['transfer_auc']
    out['cosines'] = baseline['cosines']
    return out


# Section dispatch registered ONLY here (NOT in run_spine_layer.SECTIONS). The
# sweep driver imports this to add 'crosstask42' without touching paper scripts.
def section_handler(model: str, task: str, layer: int):
    """run_spine_sweep-compatible (model, task, layer) -> result.

    Cross-task is per-(model, layer): the audit runs once; `task` only selects the
    held-out cell surfaced at the top level. Route is chosen from CROSSTASK42_ROUTE
    env (default A, falling back to B where phase_a is absent for that cell).
    """
    para = TASK_DIRS_INV.get(task, task)  # long name -> short code (already short -> unchanged)
    route = os.environ.get('CROSSTASK42_ROUTE', 'A')
    if route == 'A' and not all((model, t) in ROUTE_A_CELLS for t in ['sm', 'ic', 'mw']):
        route = 'B'  # some task in this model lacks phase_a -> all-cell route
    return run_crosstask42(model, layer, route, task=para)


def main():
    ap = argparse.ArgumentParser(description='§4.2 cross-task BK audit, all layers (isolated).')
    ap.add_argument('--model', required=True, choices=['gemma', 'llama'])
    ap.add_argument('--layer', required=True, type=int)
    ap.add_argument('--route', default='A', choices=['A', 'B'])
    ap.add_argument('--data-root', default=os.environ.get('LLM_ADDICTION_DATA_ROOT'),
                    help='sae_features_v3 snapshot (rebinds e3.DATA).')
    ap.add_argument('--out-dir', default=os.environ.get(
        'CROSSTASK42_OUT_DIR',
        str(Path(__file__).resolve().parents[2] / 'multilayer_causal/results/spine')))
    args = ap.parse_args()

    if args.data_root:
        e3.DATA = Path(args.data_root)

    result = run_crosstask42(args.model, args.layer, args.route)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'read_profile_{args.model}_crosstask42_L{args.layer}.json'
    with open(out_path, 'w') as f:
        json.dump({'model': args.model, 'section': 'crosstask42',
                   'layer': args.layer, 'route': args.route, 'result': result}, f, indent=2)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
