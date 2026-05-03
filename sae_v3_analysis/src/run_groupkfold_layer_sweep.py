"""GroupKFold layer sweep — extends run_groupkfold_recompute.py to non-L22 layers.

Runs only §4.1 (Table 1 cells, all_variable subset) at the requested layer.
§4.3 condition modulation stays at L22 (separate run).

Usage:
  python run_groupkfold_layer_sweep.py --layer 8
  python run_groupkfold_layer_sweep.py --layer 12
  python run_groupkfold_layer_sweep.py --layer 25
  python run_groupkfold_layer_sweep.py --layer 30
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, '/home/v-seungplee/llm-addiction/sae_v3_analysis/src')
from run_groupkfold_recompute import (
    CELLS, RESULTS_DIR, fit_one_subset,
)
from run_perm_null_ilc import load_sae_and_meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layer', type=int, required=True)
    args = ap.parse_args()
    layer = args.layer

    out_path = RESULTS_DIR / f'table1_groupkfold_L{layer}.json'
    print(f'=== §4.1 Table 1 (all_variable) GroupKFold sweep at L{layer} ===')
    print(f'output: {out_path}')

    table1 = {}
    t0 = time.time()
    for model, task in CELLS:
        for indicator in ['i_lc', 'i_ba', 'i_ec']:
            key = f'{model}_{task}_{indicator}_L{layer}'
            print(f'\n[{time.time()-t0:6.0f}s] === {key} ===', flush=True)
            try:
                sp, meta = load_sae_and_meta(model, task, layer)
            except Exception as e:
                print(f'  load error: {type(e).__name__}: {e}', flush=True)
                table1[key] = {'reason': f'load error: {e}'}
                continue
            if sp is None:
                print(f'  SAE missing at L{layer}', flush=True)
                table1[key] = {'reason': 'SAE missing'}
                continue
            res = fit_one_subset(meta, sp, model, task, indicator)
            table1[key] = res
            if res.get('r2_mean') is not None:
                print(f'  n={res["n"]} groups={res["n_groups"]} '
                      f'R²={res["r2_mean"]:+.4f} ± {res["r2_std"]:.4f}', flush=True)
            else:
                print(f'  SKIP: {res.get("reason")}', flush=True)
            with open(out_path, 'w') as f:
                json.dump(table1, f, indent=2)

    print(f'\nDone L{layer} in {time.time()-t0:.0f}s')
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
