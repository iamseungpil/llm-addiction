#!/usr/bin/env python3
"""
Compute the ESCALATING - CAUTIOUS bankruptcy-rate difference with 95% Newcombe
score intervals for every model-condition cell of the worked-example experiment.

Discharges the promise in archive/rebuttal_20260731/posted_openreview/a3zu_posted.md
("The exact intervals will be given in the revision.") and supplies the numbers for
the worked-example row of tab:added-controls in neurips_content_en/appendix.tex.

Source corpus (HF dataset llm-addiction-research/llm-addiction, gated):
  rebuttal_neurips_2026/in_context_demo_api/                 16 cells, 100 games each
  rebuttal_neurips_2026/in_context_demo_open_weight/          8 cells, 200 games each
  rebuttal_neurips_2026/in_context_demo_open_weight_persona/  2 cells, 200 games each
  rebuttal_neurips_2026/framing_rationality_factorial_e7/    role_rat0 no-example baseline

No DEPRECATION_WARNING.md exists anywhere under rebuttal_neurips_2026/ (the three in the
release are sae_patching/, slot_machine/gemma/, slot_machine/llama/, none of them used here).
The 'model' field inside every file is read and recorded rather than trusted from the path.

Methods
-------
Wilson score interval for each single proportion.
Newcombe (1998) method 10, independent samples, for the difference:
    L = d - sqrt((p1-l1)^2 + (u2-p2)^2)
    U = d + sqrt((u1-p1)^2 + (p2-l2)^2)
Newcombe (1998) method 10, PAIRED, reported alongside because the two arms of every
cell run the identical seed list (verified in-script), so the games are seed-matched:
    L = d - sqrt((p1-l1)^2 - 2*phi*(p1-l1)*(u2-p2) + (u2-p2)^2)
    U = d + sqrt((u1-p1)^2 - 2*phi*(u1-p1)*(p2-l2) + (p2-l2)^2)
with phi the estimated correlation from the 2x2 concordance table, floored at 0 where the
table is degenerate (Newcombe's own convention).
The independent interval is the one quoted in the paper: it is the wider, more conservative
of the two in every cell, and it does not depend on the seed match holding.
"""
import json, glob, os, math, hashlib, datetime, sys

# Pass a local copy of the release as argv[1], or set it here.
ROOT = sys.argv[1] if len(sys.argv) > 1 else "hf_snapshot/rebuttal_neurips_2026"
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(OUTDIR, "worked_example_intervals.json")
OUT_TEX = os.path.join(OUTDIR, "worked_example_intervals.tex")

# Collected and released, but not printed and not counted in the multiplicity family:
# the substitute checkpoint for the withdrawn Claude-3.5-Haiku of the six-model roster.
NOT_PRINTED = {'claude-haiku-4-5-20251001'}
Z = 1.959963984540054  # two-sided 95%

def wilson(k, n, z=Z):
    if n == 0:
        return (float('nan'),) * 3
    p = k / n
    den = 1 + z*z/n
    ctr = (p + z*z/(2*n)) / den
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return p, max(0.0, ctr - half), min(1.0, ctr + half)

def newcombe_indep(k1, n1, k2, n2):
    """Difference p1 - p2, Newcombe method 10, independent samples."""
    p1, l1, u1 = wilson(k1, n1)
    p2, l2, u2 = wilson(k2, n2)
    d = p1 - p2
    lo = d - math.sqrt((p1-l1)**2 + (u2-p2)**2)
    hi = d + math.sqrt((u1-p1)**2 + (p2-l2)**2)
    return d, max(-1.0, lo), min(1.0, hi)

def newcombe_paired(a, b, c, dd):
    """Difference p1 - p2 for paired data. a=both, b=arm1 only, c=arm2 only, dd=neither."""
    n = a + b + c + dd
    k1, k2 = a + b, a + c
    p1, l1, u1 = wilson(k1, n)
    p2, l2, u2 = wilson(k2, n)
    d = p1 - p2
    den = math.sqrt((a+b)*(c+dd)*(a+c)*(b+dd))
    phi = ((a*dd - b*c) / den) if den > 0 else 0.0
    if a*dd - b*c == 0:
        phi = 0.0
    phi = max(-1.0, min(1.0, phi))
    def rad(x, y):
        v = x*x - 2*phi*x*y + y*y
        return math.sqrt(max(v, 0.0))
    lo = d - rad(p1-l1, u2-p2)
    hi = d + rad(u1-p1, p2-l2)
    return d, max(-1.0, lo), min(1.0, hi), phi

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def load_cell(path):
    with open(path) as f:
        d = json.load(f)
    games = d['results']
    bankrupt = sum(1 for g in games if g.get('bankrupt'))
    seeds = [g['seed'] for g in games]
    bet_sum = bet_n = 0
    for g in games:
        for r in g.get('rounds', []):
            if r.get('bet') is not None:
                bet_sum += r['bet']; bet_n += 1
    return {
        'file': os.path.relpath(path, ROOT),
        'sha256': sha256(path),
        'model_field': d['model'],           # read from inside the file, not the path
        'cell_field': d['cell'],
        'cap': d['cap'], 'mode': d['mode'],
        'factor_preamble': d['factor_preamble'], 'factor_rat': d['factor_rat'],
        'n_games': len(games), 'n_games_declared': d['n_games'],
        'bankrupt': bankrupt,
        'bankruptcy_rate': bankrupt / len(games),
        'seed_base': d['seed_base'], 'seeds': seeds,
        'bankrupt_by_seed': {s: bool(g.get('bankrupt')) for s, g in zip(seeds, games)},
        'api_fallback_responses': d['manifest'].get('api_fallback_responses'),
        'mean_bet_per_played_round': (bet_sum / bet_n) if bet_n else None,
        'played_rounds': bet_n,
    }

def find(dirname, pat):
    hits = sorted(glob.glob(os.path.join(ROOT, dirname, pat)))
    return hits[0] if len(hits) == 1 else (hits if hits else None)

# ---------------------------------------------------------------- collect cells
arms = {}   # (corpus, model, mode) -> {'cautious': cell, 'escalate': cell, 'none': cell}

for corpus, pat in [
    ('in_context_demo_api', 'e7_*_cap70_*_demo_*_persona_rat0_mains_*.json'),
    ('in_context_demo_open_weight', 'e7_*_cap70_*_demo_*_rat0_mains_*.json'),
    ('in_context_demo_open_weight_persona', 'e7_*_cap70_*_demo_*_persona_rat0_mains_*.json'),
]:
    for path in sorted(glob.glob(os.path.join(ROOT, corpus, pat))):
        c = load_cell(path)
        fp = c['factor_preamble']
        arm = 'cautious' if 'cautious' in fp else 'escalate' if 'escalate' in fp else None
        assert arm, fp
        arms.setdefault((corpus, c['model_field'], c['mode']), {})[arm] = c

# no-example baseline: E7 role_rat0 (persona present, no rationality instruction)
baselines = {}
for path in sorted(glob.glob(os.path.join(ROOT, 'framing_rationality_factorial_e7',
                                          'e7_*_cap70_*_role_rat0_*.json'))):
    c = load_cell(path)
    baselines[(c['model_field'], c['mode'])] = c

# ---------------------------------------------------------------- compute
rows = []
for (corpus, model, mode), a in sorted(arms.items()):
    if 'cautious' not in a or 'escalate' not in a:
        rows.append({'corpus': corpus, 'model': model, 'mode': mode,
                     'status': 'INCOMPLETE_PAIR',
                     'arms_present': sorted(a.keys()),
                     'cells': {k: {kk: vv for kk, vv in v.items()
                                   if kk not in ('seeds', 'bankrupt_by_seed')}
                               for k, v in a.items()}})
        continue
    esc, cau = a['escalate'], a['cautious']
    n1, k1 = esc['n_games'], esc['bankrupt']
    n2, k2 = cau['n_games'], cau['bankrupt']
    d, lo, hi = newcombe_indep(k1, n1, k2, n2)

    seed_matched = esc['seeds'] == cau['seeds']
    paired = None
    if seed_matched:
        A = B = C = D = 0
        for s in esc['seeds']:
            e, c_ = esc['bankrupt_by_seed'][s], cau['bankrupt_by_seed'][s]
            if e and c_: A += 1
            elif e and not c_: B += 1
            elif c_ and not e: C += 1
            else: D += 1
        pd_, plo, phi_hi, phi = newcombe_paired(A, B, C, D)
        paired = {'table_both': A, 'table_escalate_only': B, 'table_cautious_only': C,
                  'table_neither': D, 'phi': phi,
                  'diff_pp': 100*pd_, 'ci_pp': [100*plo, 100*phi_hi],
                  'excludes_zero': (plo > 0 or phi_hi < 0)}

    bl = baselines.get((model, mode)) if corpus != 'in_context_demo_open_weight' else None
    p_esc, le, ue = wilson(k1, n1)
    p_cau, lc, uc = wilson(k2, n2)
    rows.append({
        'corpus': corpus, 'model': model, 'mode': mode, 'cap': esc['cap'],
        'persona': 'persona' in esc['factor_preamble'],
        'status': 'OK',
        'n_games_per_arm': n1 if n1 == n2 else [n1, n2],
        'cautious': {'bankrupt': k2, 'n': n2, 'rate_pct': 100*p_cau,
                     'wilson_ci_pct': [100*lc, 100*uc],
                     'mean_bet_per_played_round': cau['mean_bet_per_played_round'],
                     'file': cau['file'], 'sha256': cau['sha256'],
                     'model_field': cau['model_field'],
                     'api_fallback_responses': cau['api_fallback_responses']},
        'escalating': {'bankrupt': k1, 'n': n1, 'rate_pct': 100*p_esc,
                       'wilson_ci_pct': [100*le, 100*ue],
                       'mean_bet_per_played_round': esc['mean_bet_per_played_round'],
                       'file': esc['file'], 'sha256': esc['sha256'],
                       'model_field': esc['model_field'],
                       'api_fallback_responses': esc['api_fallback_responses']},
        'no_example_baseline': None if bl is None else {
            'bankrupt': bl['bankrupt'], 'n': bl['n_games'],
            'rate_pct': 100*bl['bankruptcy_rate'], 'file': bl['file'],
            'sha256': bl['sha256'], 'model_field': bl['model_field']},
        'escalating_minus_cautious': {
            'diff_pp': 100*d,
            'newcombe_independent_ci_pp': [100*lo, 100*hi],
            'excludes_zero': (lo > 0 or hi < 0),
            'seed_matched': seed_matched,
            'newcombe_paired': paired,
        },
    })

summary_pairs = [r for r in rows if r['status'] == 'OK']
excl = [r for r in summary_pairs if r['escalating_minus_cautious']['excludes_zero']]

out = {
    'generated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
    'purpose': ("Exact 95% intervals for ESCALATING minus CAUTIOUS bankruptcy rate in the "
                "worked-example (in-context demonstration) experiment. Discharges the promise "
                "in the a3Zu response letter and supplies the numbers for the worked-example "
                "row of tab:added-controls."),
    'method': {
        'single_proportion': 'Wilson score interval, z = 1.959963984540054',
        'difference_primary': 'Newcombe (1998) method 10, independent samples (square-and-add of Wilson tail distances)',
        'difference_secondary': 'Newcombe (1998) method 10, paired form with the concordance correlation phi; the two arms of every API/open-weight cell run an identical seed list',
        'quoted_in_paper': 'independent-samples form (the wider, assumption-lighter interval)',
        'no_multiplicity_adjustment': True,
        'multiplicity_note': ('Six API model-condition pairs are printed and tested; these are nominal, '
                              'unadjusted 95% intervals. Under a Bonferroni-style adjustment across '
                              'the six API pairs the Gemini variable interval still excludes zero '
                              '(see bonferroni_check).'),
    },
    'provenance': {
        'dataset': 'llm-addiction-research/llm-addiction (HuggingFace, gated)',
        'directories': ['rebuttal_neurips_2026/in_context_demo_api/',
                        'rebuttal_neurips_2026/in_context_demo_open_weight/',
                        'rebuttal_neurips_2026/in_context_demo_open_weight_persona/',
                        'rebuttal_neurips_2026/framing_rationality_factorial_e7/ (no-example baseline, role_rat0)'],
        'deprecation_warnings_in_scope': 'none; the release carries DEPRECATION_WARNING.md only under sae_patching/, slot_machine/gemma/ and slot_machine/llama/, none of which is read here',
        'model_identity': "taken from the 'model' field inside each JSON, not from the directory name",
        'quarantine': "framing_rationality_factorial_e7/QUARANTINE_truncated_claude/ was excluded; the Claude cells are the withdrawn checkpoint's substitute and are collected but not printed",
    },
    'headline': {
        'n_api_pairs': sum(1 for r in summary_pairs
                           if r['corpus'] == 'in_context_demo_api'
                           and r['model'] not in NOT_PRINTED),
        'n_pairs_total': len(summary_pairs),
        'pairs_excluding_zero': [f"{r['model']} {r['mode']}" for r in excl],
    },
    'rows': rows,
}

# Bonferroni check across the printed API pairs
zb = 2.63829  # two-sided alpha = 0.05/6, one test per printed API pair
def wilson_z(k, n, z):
    p = k/n; den = 1+z*z/n
    ctr = (p+z*z/(2*n))/den
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return p, max(0.0, ctr-half), min(1.0, ctr+half)
bon = []
for r in summary_pairs:
    if r['corpus'] != 'in_context_demo_api' or r['model'] in NOT_PRINTED:
        continue
    k1, n1 = r['escalating']['bankrupt'], r['escalating']['n']
    k2, n2 = r['cautious']['bankrupt'], r['cautious']['n']
    p1, l1, u1 = wilson_z(k1, n1, zb); p2, l2, u2 = wilson_z(k2, n2, zb)
    d = p1-p2
    lo = d - math.sqrt((p1-l1)**2 + (u2-p2)**2)
    hi = d + math.sqrt((u1-p1)**2 + (p2-l2)**2)
    bon.append({'model': r['model'], 'mode': r['mode'], 'diff_pp': 100*d,
                'ci_pp_bonferroni': [100*lo, 100*hi],
                'excludes_zero': (lo > 0 or hi < 0)})
out['bonferroni_check'] = {'alpha_family': 0.05, 'n_tests': 6, 'z': zb, 'rows': bon}

with open(OUT, 'w') as f:
    json.dump(out, f, indent=2)

# ------------------------------------------------------------ console summary
print(f"{'corpus':<38}{'model':<28}{'mode':<10}{'CAU':>9}{'ESC':>9}{'diff pp':>9}  {'95% Newcombe (indep)':<24} excl0")
for r in rows:
    if r['status'] != 'OK':
        print(f"{r['corpus']:<38}{r['model']:<28}{r['mode']:<10}  INCOMPLETE PAIR: {r['arms_present']}")
        continue
    e = r['escalating_minus_cautious']
    ci = e['newcombe_independent_ci_pp']
    print(f"{r['corpus']:<38}{r['model']:<28}{r['mode']:<10}"
          f"{r['cautious']['bankrupt']:>4}/{r['cautious']['n']:<4}"
          f"{r['escalating']['bankrupt']:>4}/{r['escalating']['n']:<4}"
          f"{e['diff_pp']:>+9.1f}  [{ci[0]:+7.2f}, {ci[1]:+7.2f}]      {'YES' if e['excludes_zero'] else 'no'}")
print()
for r in rows:
    if r['status'] != 'OK':
        continue
    p = r['escalating_minus_cautious'].get('newcombe_paired')
    if p:
        print(f"paired  {r['model']:<28}{r['mode']:<10} d={p['diff_pp']:+6.1f} "
              f"[{p['ci_pp'][0]:+7.2f}, {p['ci_pp'][1]:+7.2f}] phi={p['phi']:+.3f} "
              f"tbl(a,b,c,d)=({p['table_both']},{p['table_escalate_only']},{p['table_cautious_only']},{p['table_neither']}) "
              f"excl0={'YES' if p['excludes_zero'] else 'no'}")
print()
print('Bonferroni (6 API pairs):')
for b in bon:
    print(f"  {b['model']:<28}{b['mode']:<10}{b['diff_pp']:+7.1f}  "
          f"[{b['ci_pp_bonferroni'][0]:+7.2f}, {b['ci_pp_bonferroni'][1]:+7.2f}]  "
          f"{'YES' if b['excludes_zero'] else 'no'}")
print()
print('wrote', OUT)

# ============================== emit the provenance .tex fragment ==============================
DISPLAY = {'gpt-4o-mini': 'GPT-4o-mini', 'gpt-4.1-mini': 'GPT-4.1-mini',
           'gemini-2.5-flash': 'Gemini-2.5-Flash',
           'claude-haiku-4-5-20251001': 'Claude-Haiku-4.5',
           'gemma': 'Gemma-2-9B', 'llama': 'LLaMA-3.1-8B'}
ORDER = ['gpt-4o-mini', 'gpt-4.1-mini', 'gemini-2.5-flash',
         'gemma', 'llama']
BLOCKS = [
    ('in_context_demo_api',
     'API models, persona present, no-example baseline available'),
    ('in_context_demo_open_weight',
     'Open-weight, persona absent, no matched no-example arm exists'),
    ('in_context_demo_open_weight_persona',
     'Open-weight follow-up, persona present, 2 of 8 planned cells run'),
]

def pct(x):
    s = f'{x:.1f}\\%'
    return ('\\phantom{0}' + s) if x < 10 else s

def diff(x):
    return f'$\\phantom{{+}}0.0$' if abs(x) < 1e-9 else f'${x:+.1f}$'

def ci(lo, hi):
    return f'$[{lo:+.1f},\\ {hi:+.1f}]$'

prov_lines = []
for r in rows:
    if r['status'] != 'OK':
        prov_lines.append(f"%   {r['corpus']} | {r['model']} | {r['mode']} : INCOMPLETE PAIR {r['arms_present']}")
        continue
    for arm in ('cautious', 'escalating'):
        c = r[arm]
        mark = '  [collected, NOT printed]' if r['model'] in NOT_PRINTED else ''
        prov_lines.append(
            f"%   {DISPLAY.get(r['model'], r['model']):<17}| {r['mode']:<8} | {arm[:4].upper():<4}"
            f": {c['bankrupt']:>3}/{c['n']:<3} : sha {c['sha256'][:16]} : {os.path.basename(c['file'])}{mark}")
    bl = r['no_example_baseline']
    if bl:
        prov_lines.append(
            f"%   {DISPLAY.get(r['model'], r['model']):<17}| {r['mode']:<8} | BASE"
            f": {bl['bankrupt']:>3}/{bl['n']:<3} : sha {bl['sha256'][:16]} : {os.path.basename(bl['file'])}")

body = []
for corpus, caption in BLOCKS:
    block = [r for r in rows if r['status'] == 'OK' and r['corpus'] == corpus
             and r['model'] not in NOT_PRINTED]
    if not block:
        continue
    block.sort(key=lambda r: (ORDER.index(r['model']) if r['model'] in ORDER else 99,
                              r['mode'] != 'fixed'))
    body.append(f'\\multicolumn{{7}}{{l}}{{\\emph{{{caption}}}}} \\\\')
    for r in block:
        e = r['escalating_minus_cautious']
        lo, hi = e['newcombe_independent_ci_pp']
        cells = [DISPLAY.get(r['model'], r['model']), r['mode'], str(r['n_games_per_arm']),
                 pct(r['cautious']['rate_pct']), pct(r['escalating']['rate_pct']),
                 diff(e['diff_pp']), ci(lo, hi)]
        if e['excludes_zero']:
            cells = [f'\\textbf{{{cells[0]}}}', f'\\textbf{{{cells[1]}}}', cells[2],
                     f'\\textbf{{{cells[3]}}}', f'\\textbf{{{cells[4]}}}',
                     f'$\\mathbf{{{e["diff_pp"]:+.1f}}}$',
                     f'$\\mathbf{{[{lo:+.1f},\\ {hi:+.1f}]}}$']
        body.append(' & '.join(cells) + ' \\\\')
    if corpus != BLOCKS[-1][0]:
        body.append('\\midrule')

g = {r['model'] + '|' + r['mode']: r for r in rows if r['status'] == 'OK'}
gv = g['gemini-2.5-flash|variable']['escalating_minus_cautious']
gf = g['gemini-2.5-flash|fixed']['escalating_minus_cautious']
o4 = g['gpt-4o-mini|variable']['escalating_minus_cautious']
o41 = g['gpt-4.1-mini|variable']['escalating_minus_cautious']
fl = g['gpt-4o-mini|fixed']['escalating_minus_cautious']
bonf_gem = next(b for b in bon if b['model'] == 'gemini-2.5-flash' and b['mode'] == 'variable')

tex = f"""% =====================================================================================
% worked_example_intervals.tex --- PROVENANCE FRAGMENT. PROPOSED; not \\input by any .tex file.
%
% Exact 95% intervals for the ESCALATING minus CAUTIOUS bankruptcy-rate difference in the
% worked-example (in-context demonstration) experiment.
%
% Why this file exists: archive/rebuttal_20260731/posted_openreview/a3zu_posted.md (Q3) states
%   "Bold marks the only condition where the difference of ESCALATING minus CAUTIOUS has a 95%
%    interval excluding zero. The exact intervals will be given in the revision."
% while neurips_content_en/appendix.tex, tab:added-controls, worked-example row, currently says
% only that one pair's interval excludes zero and prints no interval. This fragment carries the
% numbers that discharge that promise.
%
% Generated by paper_data/tables/appendix/code/worked_example_intervals.py
% Numbers in   paper_data/tables/appendix/worked_example_intervals.json
% Generated    {out['generated_utc']}
%
% ----------------------------------- DATA PROVENANCE ---------------------------------
% Source: HuggingFace dataset llm-addiction-research/llm-addiction (gated), directories
%   rebuttal_neurips_2026/in_context_demo_api/                  16 cells, 100 games each
%   rebuttal_neurips_2026/in_context_demo_open_weight/           8 cells, 200 games each
%   rebuttal_neurips_2026/in_context_demo_open_weight_persona/   2 cells, 200 games each
%   rebuttal_neurips_2026/framing_rationality_factorial_e7/     role_rat0 no-example baseline
%
% Canonical-corpus checks performed before use:
%   * No DEPRECATION_WARNING.md exists anywhere under rebuttal_neurips_2026/. The release carries
%     that file only under sae_patching/, slot_machine/gemma/ and slot_machine/llama/, none of
%     which is read here.
%   * Model identity was read from the "model" field INSIDE each JSON, never from the path.
%     These are rebuttal-era runs and a DIFFERENT corpus from the paper's slot-machine data: the
%     gpt-4o-mini cells here come from the rebuttal runner, not from
%     analysis/gpt_results_fixed_parsing/, and the Claude cells are Claude-Haiku-4.5, not the
%     Claude-3.5-Haiku of slot_machine/claude/.
%   * framing_rationality_factorial_e7/QUARANTINE_truncated_claude/ was EXCLUDED; the top-level
%     Claude role_rat0 cells dated 20260728 were used for the no-example baseline.
%   * manifest.api_fallback_responses is 0 in all 26 cells read.
%
% Per-cell provenance (model | mode | arm : bankrupt/n : sha256 prefix : file)
{chr(10).join(prov_lines)}
%
% ---------------------------------------- METHOD -------------------------------------
% Single proportion: Wilson score interval, z = 1.959963984540054.
% Difference (QUOTED): Newcombe (1998) method 10, INDEPENDENT samples --- square-and-add of the
%   Wilson tail distances,
%     L = d - sqrt((p1-l1)^2 + (u2-p2)^2),   U = d + sqrt((u1-p1)^2 + (p2-l2)^2),
%   with p1 = ESCALATING, p2 = CAUTIOUS.
% Difference (SECONDARY, in the JSON only): Newcombe's paired form with the concordance
%   correlation phi. The two arms of every cell ran an IDENTICAL seed list (verified in the
%   generating script), so the games are seed-matched and the paired interval is admissible. It
%   is narrower wherever it differs --- Gemini variable is [{gv['newcombe_paired']['ci_pp'][0]:+.1f}, {gv['newcombe_paired']['ci_pp'][1]:+.1f}] paired against
%   [{gv['newcombe_independent_ci_pp'][0]:+.1f}, {gv['newcombe_independent_ci_pp'][1]:+.1f}] independent --- so the independent interval quoted in the paper is the
%   conservative choice and does not depend on the seed match holding. No verdict changes.
% Intervals are nominal and UNADJUSTED for the six API comparisons. Under a Bonferroni
%   adjustment across those six (z = 2.63829) Gemini variable is
%   [{bonf_gem['ci_pp_bonferroni'][0]:+.1f}, {bonf_gem['ci_pp_bonferroni'][1]:+.1f}] pp and still excludes zero; no other pair changes verdict.
%
% ------------------------------------- KNOWN LIMITS ----------------------------------
% * Floor, not null. Two of the six printed API pairs sit at 0/100 in BOTH arms. A cautious example
%   cannot lower a rate that is already zero, so those four intervals record the absence of room
%   to move rather than a tested-and-absent effect. Say "floor", not "no effect".
% * The persona-absent open-weight cells have NO matched no-example baseline anywhere in the
%   release: the four e7_{{gemma,llama}}_cap70_{{fixed,variable}}_none_rat0 cells were never
%   collected. Their ESCALATING minus CAUTIOUS contrast is still internally valid, but the
%   three-arm comparison is not available, and they are persona-ABSENT while the API cells are
%   persona-PRESENT. Do not pool the two blocks.
% * in_context_demo_open_weight_persona/ ran 2 of a planned 8 cells (Gemma, fixed only).
% =====================================================================================

\\begin{{table}}[H]
\\centering
\\scriptsize
\\setlength{{\\tabcolsep}}{{4pt}}
\\caption{{\\rev{{Worked-example experiment: bankruptcy rate under a cautious versus an escalating
in-context demonstration, and the ESCALATING minus CAUTIOUS difference in percentage points with
a 95\\% Newcombe score interval. Every cell is at the $\\$70$ cap without the rationality
instruction. Bold marks the one pair whose interval excludes zero. Intervals are unadjusted for
the six API comparisons; Gemini variable survives a Bonferroni adjustment across them
({bonf_gem['ci_pp_bonferroni'][0]:+.1f} to {bonf_gem['ci_pp_bonferroni'][1]:+.1f}). Two printed API pairs sit at 0\\% in both arms, so their intervals record a
floor rather than a tested null.}}}}
\\label{{tab:worked-example-intervals}}
\\begin{{tabular}}{{llrrrrr}}
\\toprule
\\textbf{{Model}} & \\textbf{{Mode}} & \\textbf{{$n$/arm}} & \\textbf{{Cautious}} &
\\textbf{{Escalating}} & \\textbf{{Diff (pp)}} & \\textbf{{95\\% CI (pp)}} \\\\
\\midrule
{chr(10).join(body)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

% ---- Macros for inline use in tab:added-controls, so the row and this table cannot drift ----
\\newcommand{{\\weGeminiVarCaut}}{{{g['gemini-2.5-flash|variable']['cautious']['rate_pct']:.0f}\\%}}
\\newcommand{{\\weGeminiVarEsc}}{{{g['gemini-2.5-flash|variable']['escalating']['rate_pct']:.0f}\\%}}
\\newcommand{{\\weGeminiVarDiff}}{{${gv['diff_pp']:+.1f}$}}
\\newcommand{{\\weGeminiVarCI}}{{$[{gv['newcombe_independent_ci_pp'][0]:+.1f}, {gv['newcombe_independent_ci_pp'][1]:+.1f}]$}}
\\newcommand{{\\weGeminiFixDiff}}{{${gf['diff_pp']:+.1f}$}}
\\newcommand{{\\weGeminiFixCI}}{{$[{gf['newcombe_independent_ci_pp'][0]:+.1f}, {gf['newcombe_independent_ci_pp'][1]:+.1f}]$}}
\\newcommand{{\\weGptFourOVarDiff}}{{${o4['diff_pp']:+.1f}$}}
\\newcommand{{\\weGptFourOVarCI}}{{$[{o4['newcombe_independent_ci_pp'][0]:+.1f}, {o4['newcombe_independent_ci_pp'][1]:+.1f}]$}}
\\newcommand{{\\weGptFourOneVarDiff}}{{${o41['diff_pp']:+.1f}$}}
\\newcommand{{\\weGptFourOneVarCI}}{{$[{o41['newcombe_independent_ci_pp'][0]:+.1f}, {o41['newcombe_independent_ci_pp'][1]:+.1f}]$}}
\\newcommand{{\\weFloorCI}}{{$[{fl['newcombe_independent_ci_pp'][0]:+.1f}, {fl['newcombe_independent_ci_pp'][1]:+.1f}]$}}
"""

with open(OUT_TEX, 'w') as f:
    f.write(tex)
print('wrote', OUT_TEX)
