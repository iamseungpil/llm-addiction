"""Generate the appendix write-band readout table from
table1_groupkfold_band_{model}.json + the L22 reference.

Rows: the body-reportable Table-1 cells. Columns: the model's causal
write-band layers plus the body-cited L22 for comparison. Two blocks,
one per model, because the write bands differ (Gemma L16-21, LLaMA L14-19).

Output: LaTeX table to stdout + a min/max summary per block.
"""
from __future__ import annotations
import json
from pathlib import Path

RESULTS = Path('/home/v-seungplee/llm-addiction/sae_v3_analysis/results')
BAND = {'gemma': list(range(16, 22)), 'llama': list(range(14, 20))}
TASKS = ['sm', 'ic', 'mw']
INDICATORS = ['i_lc', 'i_ba', 'i_ec']
INDICATOR_TEX = {'i_lc': r'$I_{\text{LC}}$', 'i_ba': r'$I_{\text{BA}}$',
                 'i_ec': r'$I_{\text{EC}}$'}
MODEL_TEX = {'gemma': 'Gemma', 'llama': 'LLaMA'}
TASK_TEX = {'sm': 'SM', 'ic': 'IC', 'mw': 'MW'}

# Same reportable set as the L8-L30 sweep table
BODY_REPORTABLE = {
    ('gemma', 'sm', 'i_lc'), ('gemma', 'sm', 'i_ba'), ('gemma', 'sm', 'i_ec'),
    ('gemma', 'ic', 'i_ba'),
    ('gemma', 'mw', 'i_lc'), ('gemma', 'mw', 'i_ba'),
    ('llama', 'sm', 'i_lc'), ('llama', 'sm', 'i_ba'), ('llama', 'sm', 'i_ec'),
    ('llama', 'ic', 'i_ba'), ('llama', 'ic', 'i_ec'),
    ('llama', 'mw', 'i_lc'), ('llama', 'mw', 'i_ba'),
}


def fmt(r):
    if r is None or r < 0.01:
        return '---'
    return f'{r:+.3f}'.replace('+', '\\hphantom{$-$}')


def main():
    l22 = json.load(open(RESULTS / 'table1_groupkfold_L22.json'))
    for model in ('gemma', 'llama'):
        band = json.load(open(RESULTS / f'table1_groupkfold_band_{model}.json'))
        layers = BAND[model]
        print(f'% ===== {MODEL_TEX[model]} block: band '
              f'L{layers[0]}-L{layers[-1]} + L22 reference =====')
        header = ' & '.join([f'L{L}' for L in layers])
        print(f'{MODEL_TEX[model]} & Task & Indicator & {header} '
              f'& L22 \\\\')
        print('\\midrule')
        band_vals = []
        for task in TASKS:
            first = True
            for ind in INDICATORS:
                if (model, task, ind) not in BODY_REPORTABLE:
                    continue
                cells = []
                for L in layers:
                    r = band.get(f'{model}_{task}_{ind}_L{L}', {})
                    v = r.get('r2_mean')
                    cells.append(fmt(v))
                    if v is not None and v >= 0.01:
                        band_vals.append((v, f'{task}/{ind}/L{L}'))
                ref = l22.get(f'{model}_{task}_{ind}_L22', {}).get('r2_mean')
                lead = TASK_TEX[task] if first else ''
                first = False
                print(f' & {lead} & {INDICATOR_TEX[ind]} & '
                      + ' & '.join(cells) + f' & {fmt(ref)} \\\\')
        if band_vals:
            lo = min(band_vals)
            hi = max(band_vals)
            print(f'% {model}: band min {lo[0]:.3f} ({lo[1]}), '
                  f'max {hi[0]:.3f} ({hi[1]})')
        print()


if __name__ == '__main__':
    main()
