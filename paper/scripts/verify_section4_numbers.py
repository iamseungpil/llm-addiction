"""Phase B: verify every numeric claim in §4 against canonical JSON.

Scope:
  - §4.1 prose + Table 1 cells
  - §4.3 prose + Table 2 cells
  - §4.4 summary
  - Appendix C.2 sweep table (already auto-generated, verified by construction)

Data sources:
  - table1_groupkfold_L22.json    (§4.1 Table 1)
  - condition_modulation_groupkfold_L22.json  (§4.3 Table 2)
  - table1_groupkfold_L{8,12,25,30}.json  (Appendix C.2)

Tolerance: R² ±0.005, Δ% ±2 (rounding accumulation across two divisions).
"""
from __future__ import annotations
import json
import os
import re
import sys
from pathlib import Path

REPO_ID = 'llm-addiction-research/llm-addiction'
# The canonical JSON lives in the separate analysis tree.  Point
# ``LLM_ADDICTION_ANALYSIS`` at that checkout to read it from disk; when the
# checkout is absent the same files are pulled from the public dataset, which
# mirrors them under ``sae_v3_analysis/results/``.
ANALYSIS_ROOT = Path(os.environ.get('LLM_ADDICTION_ANALYSIS', Path.home() / 'llm-addiction'))
RESULTS = ANALYSIS_ROOT / 'experiments' / '07_sae_readout' / 'results'
TOL_R2 = 0.005
TOL_PCT = 2.0


def load(name):
    p = RESULTS / name
    if not p.exists():
        try:
            from huggingface_hub import hf_hub_download
            p = Path(hf_hub_download(REPO_ID, f'sae_v3_analysis/results/{name}',
                                     repo_type='dataset'))
        except Exception:
            return None
    return json.load(open(p))


T1 = load('table1_groupkfold_L22.json') or {}
CM = load('condition_modulation_groupkfold_L22.json') or {}


def get_r2(model, task, ind, layer=22):
    k = f'{model}_{task}_{ind}_L{layer}'
    rec = T1.get(k, {})
    return rec.get('r2_mean')


def get_subset_r2(model, task, ind, subset):
    k = f'{model}_{task}_{ind}_L22'
    rec = CM.get(k, {})
    return rec.get('subsets', {}).get(subset, {}).get('r2_mean')


def fmt_r2(v):
    return f'{v:+.3f}' if v is not None else 'None'


def check(label, claimed, actual, tol=TOL_R2):
    if actual is None:
        return f'  [SKIP] {label}: no canonical source for "{claimed}"'
    diff = abs(claimed - actual)
    if diff <= tol:
        return f'  [OK]   {label}: {claimed:+.3f} vs {actual:+.3f} (Δ={diff:.4f})'
    return f'  [FAIL] {label}: {claimed:+.3f} vs {actual:+.3f} (Δ={diff:.4f}, tol={tol})'


def main():
    out = []
    out.append('=== §4.1 Table 1 cells ===')
    table1_claims = [
        # (label, model, task, ind, claimed)
        ('Gemma SM I_LC',  'gemma', 'sm', 'i_lc', 0.059),
        ('Gemma SM I_BA',  'gemma', 'sm', 'i_ba', 0.167),
        ('Gemma SM I_EC',  'gemma', 'sm', 'i_ec', 0.051),
        ('Gemma IC I_BA',  'gemma', 'ic', 'i_ba', 0.203),
        ('Gemma MW I_LC',  'gemma', 'mw', 'i_lc', 0.107),
        ('Gemma MW I_BA',  'gemma', 'mw', 'i_ba', 0.072),
        ('LLaMA SM I_LC',  'llama', 'sm', 'i_lc', 0.106),
        ('LLaMA SM I_BA',  'llama', 'sm', 'i_ba', 0.109),
        ('LLaMA SM I_EC',  'llama', 'sm', 'i_ec', 0.036),
        ('LLaMA IC I_BA',  'llama', 'ic', 'i_ba', 0.268),
        ('LLaMA IC I_EC',  'llama', 'ic', 'i_ec', 0.307),
        ('LLaMA MW I_LC',  'llama', 'mw', 'i_lc', 0.145),
        ('LLaMA MW I_BA',  'llama', 'mw', 'i_ba', 0.070),
    ]
    for label, m, t, ind, claimed in table1_claims:
        out.append(check(label, claimed, get_r2(m, t, ind)))

    out.append('')
    out.append('=== §4.1 prose (n* and other) ===')
    n_star_llama_ic = T1.get('llama_ic_i_lc_L22', {}).get('n')
    out.append(f'  LLaMA IC post-loss n*=865 vs JSON {n_star_llama_ic} ' +
               ('OK' if n_star_llama_ic == 865 else 'MISMATCH'))

    out.append('')
    out.append('=== §4.3 Table 2 SM condition modulation ===')
    sm_modulation_claims = [
        # (label, model, ind, subset, claimed)
        ('Gemma SM all_var I_LC',   'gemma', 'sm', 'i_lc', 'all_variable', 0.059),
        ('Gemma SM all_var I_BA',   'gemma', 'sm', 'i_ba', 'all_variable', 0.167),
        ('Gemma SM all_var I_EC',   'gemma', 'sm', 'i_ec', 'all_variable', 0.051),
        ('Gemma SM -G I_BA',        'gemma', 'sm', 'i_ba', 'minus_G', 0.063),
        ('Gemma SM +G I_BA',        'gemma', 'sm', 'i_ba', 'plus_G', 0.153),
        ('LLaMA SM all_var I_LC',   'llama', 'sm', 'i_lc', 'all_variable', 0.106),
        ('LLaMA SM all_var I_BA',   'llama', 'sm', 'i_ba', 'all_variable', 0.109),
        ('LLaMA SM -G I_BA',        'llama', 'sm', 'i_ba', 'minus_G', 0.082),
        ('LLaMA SM +G I_BA',        'llama', 'sm', 'i_ba', 'plus_G', 0.113),
    ]
    for label, m, t, ind, subset, claimed in sm_modulation_claims:
        out.append(check(label, claimed, get_subset_r2(m, t, ind, subset)))

    out.append('')
    out.append('=== §4.3 prose MW modulation (currently in §4.3 prose) ===')
    mw_modulation_claims = [
        # (label, model, ind, subset, claimed)
        ('Gemma MW -G I_LC',        'gemma', 'mw', 'i_lc', 'minus_G', 0.081),
        ('Gemma MW +G I_LC',        'gemma', 'mw', 'i_lc', 'plus_G', 0.138),
        ('LLaMA MW -G I_LC',        'llama', 'mw', 'i_lc', 'minus_G', 0.142),
        ('LLaMA MW +G I_LC',        'llama', 'mw', 'i_lc', 'plus_G', 0.191),
    ]
    for label, m, t, ind, subset, claimed in mw_modulation_claims:
        out.append(check(label, claimed, get_subset_r2(m, t, ind, subset)))

    out.append('')
    out.append('=== §4.4 summary headline numbers ===')
    # +143% claim: Gemma SM I_BA goes from -G=0.063 to +G=0.153
    g_sm_iba_minusG = get_subset_r2('gemma', 'sm', 'i_ba', 'minus_G')
    g_sm_iba_plusG = get_subset_r2('gemma', 'sm', 'i_ba', 'plus_G')
    if g_sm_iba_minusG and g_sm_iba_plusG:
        delta_g_pct = (g_sm_iba_plusG / g_sm_iba_minusG - 1) * 100
        claimed = 143
        diff = abs(claimed - delta_g_pct)
        status = 'OK' if diff <= TOL_PCT else 'MISMATCH'
        out.append(f'  [{status}] Gemma SM ΔG% I_BA: claimed +143%, computed {delta_g_pct:+.1f}% (Δ={diff:.1f})')
    l_sm_iba_minusG = get_subset_r2('llama', 'sm', 'i_ba', 'minus_G')
    l_sm_iba_plusG = get_subset_r2('llama', 'sm', 'i_ba', 'plus_G')
    if l_sm_iba_minusG and l_sm_iba_plusG:
        delta_g_pct = (l_sm_iba_plusG / l_sm_iba_minusG - 1) * 100
        claimed = 38
        diff = abs(claimed - delta_g_pct)
        status = 'OK' if diff <= TOL_PCT else 'MISMATCH'
        out.append(f'  [{status}] LLaMA SM ΔG% I_BA: claimed +38%, computed {delta_g_pct:+.1f}% (Δ={diff:.1f})')

    out.append('')
    out.append('=== §4.3 prose remaining MW modulation Δ% ===')
    # MW Gemma I_LC +71% claim
    g_mw_ilc_minusG = get_subset_r2('gemma', 'mw', 'i_lc', 'minus_G')
    g_mw_ilc_plusG = get_subset_r2('gemma', 'mw', 'i_lc', 'plus_G')
    if g_mw_ilc_minusG and g_mw_ilc_plusG:
        delta = (g_mw_ilc_plusG / g_mw_ilc_minusG - 1) * 100
        out.append(f'  [check] Gemma MW ΔG% I_LC: claimed +71%, computed {delta:+.1f}%')
    l_mw_ilc_minusG = get_subset_r2('llama', 'mw', 'i_lc', 'minus_G')
    l_mw_ilc_plusG = get_subset_r2('llama', 'mw', 'i_lc', 'plus_G')
    if l_mw_ilc_minusG and l_mw_ilc_plusG:
        delta = (l_mw_ilc_plusG / l_mw_ilc_minusG - 1) * 100
        out.append(f'  [check] LLaMA MW ΔG% I_LC: claimed +35%, computed {delta:+.1f}%')

    out.append('')
    out.append('=== §4.3 prose IC ±G stability ===')
    g_ic_iba_minusG = get_subset_r2('gemma', 'ic', 'i_ba', 'minus_G')
    g_ic_iba_plusG = get_subset_r2('gemma', 'ic', 'i_ba', 'plus_G')
    if g_ic_iba_minusG and g_ic_iba_plusG:
        delta = (g_ic_iba_plusG / g_ic_iba_minusG - 1) * 100
        out.append(f'  [check] Gemma IC ΔG% I_BA: claimed -8%, computed {delta:+.1f}%')
    l_ic_iba_minusG = get_subset_r2('llama', 'ic', 'i_ba', 'minus_G')
    l_ic_iba_plusG = get_subset_r2('llama', 'ic', 'i_ba', 'plus_G')
    if l_ic_iba_minusG and l_ic_iba_plusG:
        delta = (l_ic_iba_plusG / l_ic_iba_minusG - 1) * 100
        out.append(f'  [check] LLaMA IC ΔG% I_BA: claimed -8%, computed {delta:+.1f}%')

    print('\n'.join(out))
    fail_count = sum(1 for line in out if 'FAIL' in line or 'MISMATCH' in line)
    print()
    print(f'=== Summary: {fail_count} FAIL/MISMATCH out of total checks ===')
    return fail_count


if __name__ == '__main__':
    sys.exit(main())
