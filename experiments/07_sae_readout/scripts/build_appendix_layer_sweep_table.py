"""Generate the appendix layer-sweep table from table1_groupkfold_L{layer}.json files.

Output:
  - LaTeX table to stdout
  - Worst-case-per-layer summary
"""
from __future__ import annotations
import json
from pathlib import Path

RESULTS = Path('/home/v-seungplee/llm-addiction/sae_v3_analysis/results')
LAYERS = [8, 12, 22, 25, 30]
CELLS = [
    ('gemma', 'sm'), ('gemma', 'ic'), ('gemma', 'mw'),
    ('llama', 'sm'), ('llama', 'ic'), ('llama', 'mw'),
]
INDICATORS = ['i_lc', 'i_ba', 'i_ec']
INDICATOR_TEX = {'i_lc': r'$I_{\text{LC}}$', 'i_ba': r'$I_{\text{BA}}$', 'i_ec': r'$I_{\text{EC}}$'}
MODEL_TEX = {'gemma': 'Gemma', 'llama': 'LLaMA'}
TASK_TEX = {'sm': 'SM', 'ic': 'IC', 'mw': 'MW'}

# Body Table 1 reports these cells; others are n.s./n/c
BODY_REPORTABLE = {
    ('gemma','sm','i_lc'), ('gemma','sm','i_ba'), ('gemma','sm','i_ec'),
    ('gemma','ic','i_ba'),
    ('gemma','mw','i_lc'), ('gemma','mw','i_ba'),
    ('llama','sm','i_lc'), ('llama','sm','i_ba'), ('llama','sm','i_ec'),
    ('llama','ic','i_ba'), ('llama','ic','i_ec'),
    ('llama','mw','i_lc'), ('llama','mw','i_ba'),
}


def load(layer):
    p = RESULTS / f'table1_groupkfold_L{layer}.json'
    if not p.exists():
        return None
    return json.load(open(p))


def fmt(r):
    if r is None:
        return '---'
    return f'{r:+.3f}'.replace('+', '\\hphantom{$-$}')


def main():
    data = {L: load(L) for L in LAYERS}
    missing = [L for L in LAYERS if data[L] is None]
    if missing:
        print(f'% MISSING layers: {missing}', flush=True)

    # Compute worst-case across body-reportable cells per layer
    print('% Worst-case R² (over body-reportable cells) per layer:')
    for L in LAYERS:
        if data[L] is None:
            continue
        vals = []
        for m, t in CELLS:
            for ind in INDICATORS:
                if (m, t, ind) not in BODY_REPORTABLE:
                    continue
                k = f'{m}_{t}_{ind}_L{L}'
                rec = data[L].get(k, {})
                r = rec.get('r2_mean')
                if r is not None:
                    vals.append((r, m, t, ind))
        vals.sort()
        worst = vals[0] if vals else None
        if worst:
            print(f'%   L{L:>2}: min={worst[0]:+.4f} ({worst[1]}/{worst[2]}/{worst[3]}), n_reportable={len(vals)}')

    # LaTeX table — rows: cells, cols: layers
    print()
    print(r'\begin{table}[ht!]')
    print(r'\centering')
    print(r'\footnotesize')
    print(r'\caption{GroupKFold layer-sweep replication of §4.1 Table~\ref{tab:neurips-sae-results}. '
          r'$R^2$ at each layer $\in\{$L8,L12,L22,L25,L30$\}$, same pipeline (Top-200 SAE features, Ridge $\alpha=100$, '
          r'5-fold GroupKFold by \texttt{game\_id}, within-fold RF deconfound). Cells with $R^2<0.01$ are reported as '
          r'\texttt{---} (below null floor). \cellcolor{green!12}Green = body-cited L22 cell.}')
    print(r'\label{tab:appendix-groupkfold-sweep}')
    cols = 'llc' + 'c' * len(LAYERS)
    print(rf'\begin{{tabular}}{{{cols}}}')
    print(r'\toprule')
    header = ['Model', 'Task', 'Indicator'] + [f'L{L}' for L in LAYERS]
    print(' & '.join(header) + r' \\')
    print(r'\midrule')
    for m, t in CELLS:
        for i, ind in enumerate(INDICATORS):
            if (m, t, ind) not in BODY_REPORTABLE:
                continue
            row = [
                MODEL_TEX[m] if i == 0 else '',
                TASK_TEX[t] if i == 0 else '',
                INDICATOR_TEX[ind],
            ]
            for L in LAYERS:
                if data[L] is None:
                    row.append('---')
                    continue
                k = f'{m}_{t}_{ind}_L{L}'
                rec = data[L].get(k, {})
                r = rec.get('r2_mean')
                cell = '---' if r is None or r < 0.01 else f'{r:.3f}'
                if L == 22:
                    cell = r'\cellcolor{green!12}' + cell
                row.append(cell)
            print(' & '.join(row) + r' \\')
        print(r'\addlinespace[2pt]')
    print(r'\bottomrule')
    print(r'\end{tabular}')
    print(r'\end{table}')


if __name__ == '__main__':
    main()
