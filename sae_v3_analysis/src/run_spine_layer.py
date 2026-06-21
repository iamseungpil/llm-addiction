"""All-layer §4 read spine — one (model, task, section, layer) read profile.

Thin, isolated wrapper around the paper functions (§4.1/§4.2/§4.3). It does NOT
modify any paper script: the only behaviour it changes is (a) widening the
layer guard (no VALID_LAYERS set — a layer is valid iff its data file exists)
and (b) redirecting DATA_ROOT at the requested snapshot via --data-root / env.

Sections (all reuse paper functions unchanged at the requested layer):
  indicators  §4.1  load_sae_and_meta -> fit_one_subset (SAE-feature Ridge,
              GroupKFold by game_id, RF deconfound). 3 indicators (i_lc/i_ba/i_ec).
  crosstask   §4.2  e3_balance_stratified_section42.load_task + loto_audit
              (diff-of-means BK axis, SVD LOTO, random-direction baseline).
  condition   §4.3  fit_one_subset over the ±G/±M subsets (GroupKFold, leakage-fixed)
              + post-hoc ΔR2(+G)=r2(plus_G)-r2(minus_G), ΔR2(+M) likewise.

Reproducibility invariant: at layer 22 the `indicators` output equals what
run_groupkfold_layer_sweep.py / fit_one_subset produces for the same cell —
it is the same function, just unguarded.

Usage:
  python run_spine_layer.py --model gemma --task slot_machine \
      --section indicators --layer 22 --data-root /path/to/sae_features_v3
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT_DIR = Path(os.environ.get(
    'SPINE_OUT_DIR',
    Path(__file__).resolve().parents[2] / 'multilayer_causal/results/spine'))
# CLI task name -> short paradigm code used by the paper loaders.
TASK_SHORT = {'slot_machine': 'sm', 'investment_choice': 'ic', 'mystery_wheel': 'mw'}


def _redirect_data_root(data_root: Path):
    """Point the hardcoded paper DATA_ROOT at the snapshot. No paper file edited:
    we rebind the module-level Path constants the loaders read."""
    # Several paper modules run RESULTS_DIR.mkdir() at IMPORT time on a hardcoded
    # /home path that is unwritable on a node (PermissionError kills the import).
    # The spine sweep never writes to RESULTS_DIR, so swallow only that failure
    # during the paper imports, then restore normal mkdir for our own outputs.
    import pathlib
    _orig_mkdir = pathlib.Path.mkdir

    def _safe_mkdir(self, *a, **k):
        try:
            return _orig_mkdir(self, *a, **k)
        except (PermissionError, OSError):
            return None
    pathlib.Path.mkdir = _safe_mkdir
    try:
        import run_perm_null_ilc as rpn
        import run_groupkfold_recompute as gkf
        import run_comprehensive_robustness as rcr
    finally:
        pathlib.Path.mkdir = _orig_mkdir
    _do_redirect(data_root, rpn, gkf, rcr)


def _do_redirect(data_root: Path, rpn, gkf, rcr):
    rpn.DATA_ROOT = data_root
    behav = data_root.parent / 'behavioral'
    rpn.BEHAVIORAL_ROOT = behav
    gkf.BEHAVIORAL_ROOT = behav  # bound at import; rebind for compute_loss_chasing_continuous
    # compute_iba (I_BA + I_EC source) reads rcr's OWN module-level constants, not
    # gkf's import alias — without these two rebinds it keeps reading the hardcoded
    # path and silently returns 'compute_iba returned None' via --data-root/env.
    rcr.DATA_ROOT = data_root
    rcr.BEHAVIORAL_ROOT = behav
    try:
        import e3_balance_stratified_section42 as e3
        e3.DATA = data_root
    except Exception:
        pass


def run_indicators(model, task, layer):
    """§4.1 — fit_one_subset per indicator at this layer (L22-reproducing path)."""
    from run_groupkfold_recompute import fit_one_subset
    from run_perm_null_ilc import load_sae_and_meta
    para = TASK_SHORT[task]
    sp, meta = load_sae_and_meta(model, para, layer)
    if sp is None:
        return {'reason': 'SAE missing'}
    out = {}
    for indicator in ['i_lc', 'i_ba', 'i_ec']:
        out[indicator] = fit_one_subset(meta, sp, model, para, indicator)
    return out


def run_condition(model, task, layer):
    """§4.3 — per-subset GroupKFold refit + post-hoc ΔR2 sharpening contrast."""
    from run_groupkfold_recompute import fit_one_subset
    from run_perm_null_ilc import load_sae_and_meta
    para = TASK_SHORT[task]
    sp, meta = load_sae_and_meta(model, para, layer)
    if sp is None:
        return {'reason': 'SAE missing'}
    out = {}
    for indicator in ['i_lc', 'i_ba', 'i_ec']:
        subsets = {}
        for subset in ['all_variable', 'plus_G', 'minus_G', 'plus_M', 'minus_M', 'fixed_all']:
            vf = None if subset == 'all_variable' else subset
            subsets[subset] = fit_one_subset(meta, sp, model, para, indicator, valid_filter=vf)

        r2 = {s: subsets[s].get('r2_mean') for s in subsets}
        dg = (r2['plus_G'] - r2['minus_G']) if None not in (r2['plus_G'], r2['minus_G']) else None
        dm = (r2['plus_M'] - r2['minus_M']) if None not in (r2['plus_M'], r2['minus_M']) else None
        out[indicator] = {'subsets': subsets, 'delta_r2_G': dg, 'delta_r2_M': dm}
    return out


def run_crosstask(model, task, layer):
    """§4.2 — diff-of-means BK axis + LOTO shared axis vs random baseline.

    Cross-task is per-(model, layer): builds all three tasks and holds each out.
    `task` selects which held-out cell to report (the full loto_audit is run once).
    """
    import e3_balance_stratified_section42 as e3
    per_task = {}
    for t in ['sm', 'ic', 'mw']:
        try:
            d = e3.load_task(model, t, layer)  # raises if hidden_states_dp.npz absent
        except FileNotFoundError:
            d = None
        if d is None:
            return {'reason': f'hidden_states_dp missing (file or layer {layer}) for {t}'}
        per_task[t] = d
    audit = e3.loto_audit(per_task)
    para = TASK_SHORT[task]
    return {
        'held_out': para,
        'loto_pca': audit['loto_pca'].get(para),
        'transfer_auc': audit['transfer_auc'],
        'cosines': audit['cosines'],
    }


SECTIONS = {
    'indicators': run_indicators,
    'crosstask': run_crosstask,
    'condition': run_condition,
}


def main():
    ap = argparse.ArgumentParser(description='All-layer §4 read spine (one cell).')
    ap.add_argument('--model', required=True, choices=['gemma', 'llama'])
    ap.add_argument('--task', required=True, choices=list(TASK_SHORT))
    ap.add_argument('--section', required=True, choices=list(SECTIONS))
    ap.add_argument('--layer', required=True, type=int)
    ap.add_argument('--data-root', default=os.environ.get('LLM_ADDICTION_DATA_ROOT'),
                    help='Path to sae_features_v3 snapshot (overrides hardcoded paper DATA_ROOT).')
    args = ap.parse_args()

    if args.data_root:
        _redirect_data_root(Path(args.data_root))

    result = SECTIONS[args.section](args.model, args.task, args.layer)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f'read_profile_{args.model}_{args.task}_{args.section}_L{args.layer}.json'
    payload = {'model': args.model, 'task': args.task, 'section': args.section,
               'layer': args.layer, 'result': result}
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
