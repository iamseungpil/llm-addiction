"""GroupKFold band sweep — §4.1 Table 1 cells at the causal WRITE-BAND layers.

The causal battery locates the write band at L16-21 (Gemma) / L14-19 (LLaMA).
This runs the exact Table-1 read pipeline (same top-200 SAE features, same
Ridge, same GroupKFold-by-game folds) at each band layer, so the appendix can
report whether the write band also reads. One model per invocation; each
model is swept only over its own band.

Usage:
  python run_groupkfold_band_sweep.py --model gemma   # L16..L21
  python run_groupkfold_band_sweep.py --model llama   # L14..L19
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, '/home/v-seungplee/llm-addiction/sae_v3_analysis/src')
from run_groupkfold_recompute import (
    CELLS, RESULTS_DIR, fit_one_subset,
)
from run_perm_null_ilc import load_sae_and_meta


BAND = {
    'gemma': list(range(16, 22)),   # L16-21, §4.1 Gemma write band
    'llama': list(range(14, 20)),   # L14-19, §4.1 LLaMA write band
}


def main():
    ap = argparse.ArgumentParser(
        description='§4.1 Table 1 (all_variable) GroupKFold sweep over the '
                    'causal write-band layers of one model.'
    )
    ap.add_argument('--model', choices=sorted(BAND), required=True)
    args = ap.parse_args()
    model = args.model
    layers = BAND[model]
    tasks = [t for m, t in CELLS if m == model]

    out_path = RESULTS_DIR / f'table1_groupkfold_band_{model}.json'
    print(f'=== §4.1 Table 1 band sweep: {model} L{layers[0]}-L{layers[-1]} ===')
    print(f'output: {out_path}')

    out = {}
    t0 = time.time()
    for layer in layers:
        for task in tasks:
            for indicator in ['i_lc', 'i_ba', 'i_ec']:
                key = f'{model}_{task}_{indicator}_L{layer}'
                print(f'\n[{time.time()-t0:6.0f}s] === {key} ===', flush=True)
                try:
                    sp, meta = load_sae_and_meta(model, task, layer)
                except Exception as e:
                    print(f'  load error: {type(e).__name__}: {e}', flush=True)
                    out[key] = {'reason': f'load error: {e}'}
                    continue
                if sp is None:
                    print(f'  SAE features missing at L{layer}', flush=True)
                    out[key] = {'reason': 'SAE missing'}
                    continue
                res = fit_one_subset(meta, sp, model, task, indicator)
                out[key] = res
                if res.get('r2_mean') is not None:
                    print(f'  n={res["n"]} groups={res["n_groups"]} '
                          f'R²={res["r2_mean"]:+.4f} ± {res["r2_std"]:.4f}',
                          flush=True)
                else:
                    print(f'  SKIP: {res.get("reason")}', flush=True)
                with open(out_path, 'w') as f:
                    json.dump(out, f, indent=2)

    print(f'\nDone {model} band in {time.time()-t0:.0f}s')
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
