"""Aggregate M3' indicator-direction steering trials into dose-response stats.

Reads trials.jsonl from results/v19_multi_patching/M3prime_indicator_steering/
{model}_{task}_{condition}_n{N}/ for all conditions of a given (model, task)
and computes:

  - Per-condition: n, mean bet_ratio, mean amount, stop rate, mean balance,
    bootstrap 95% CI on bet_ratio mean
  - Dose-response: Pearson r(α_sigma, mean_bet_ratio) across the dose ladder
    {alpha-2, alpha-1, alpha+0, alpha+1, alpha+2, alpha+3}; bootstrap 95% CI
    on r and Spearman ρ as a robustness alternative
  - Cohen h: stop-rate change between α=-2 and α=+3 (extreme contrast) and
    between α=0 and α=+2 (canonical contrast)
  - Specificity contrasts (when present): random / L8 / ILC vs alpha+2 effect
    sizes (Cohen h on stop, Welch t on bet_ratio)
  - Effect of direction sign: sign(Pearson r) consistent with the predictor
    sign of the indicator

Output:
  results/v19_multi_patching/M3prime_indicator_steering/aggregated/{model}_{task}.json
  results/v19_multi_patching/M3prime_indicator_steering/aggregated/{model}_{task}_summary.md

Usage:
    python aggregate_m3prime_dose_response.py --model gemma --task sm
"""
from __future__ import annotations
import argparse, json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path('/home/v-seungplee/llm-addiction/sae_v3_analysis')
M3P_ROOT = ROOT / 'results/v19_multi_patching/M3prime_indicator_steering'
OUT_DIR = M3P_ROOT / 'aggregated'

DOSE_LADDER = ['alpha-2', 'alpha-1', 'alpha+0', 'alpha+1', 'alpha+2', 'alpha+3']
ALPHA_LADDER = {
    'alpha-2': -2.0, 'alpha-1': -1.0, 'alpha+0': 0.0,
    'alpha+1': +1.0, 'alpha+2': +2.0, 'alpha+3': +3.0,
}
SPECIFICITY = ['random', 'L8', 'ILC']


def load_trials_for_condition(model: str, task: str, condition: str) -> list[dict]:
    """Look up trials.jsonl for a condition; resolve any n suffix."""
    candidates = sorted(M3P_ROOT.glob(f'{model}_{task}_{condition}_n*'))
    rows = []
    for cand in candidates:
        jp = cand / 'trials.jsonl'
        if not jp.exists():
            continue
        for line in open(jp):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def bootstrap_mean_ci(x: np.ndarray, n_iter: int = 5000, seed: int = 42) -> tuple[float, float]:
    """Percentile bootstrap 95% CI on the sample mean."""
    if x.size == 0:
        return float('nan'), float('nan')
    rng = np.random.RandomState(seed)
    means = []
    for _ in range(n_iter):
        idx = rng.randint(0, x.size, size=x.size)
        means.append(float(x[idx].mean()))
    lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    return lo, hi


def bootstrap_pearson_ci(alpha_per_trial: np.ndarray, y_per_trial: np.ndarray,
                         n_iter: int = 5000, seed: int = 42) -> tuple[float, float, float]:
    """Trial-level resampling: returns point r, lo, hi."""
    if alpha_per_trial.size < 3:
        return float('nan'), float('nan'), float('nan')
    if alpha_per_trial.std() == 0 or y_per_trial.std() == 0:
        return 0.0, float('nan'), float('nan')
    r_point = float(np.corrcoef(alpha_per_trial, y_per_trial)[0, 1])
    rng = np.random.RandomState(seed)
    rs = []
    n = alpha_per_trial.size
    for _ in range(n_iter):
        idx = rng.randint(0, n, size=n)
        a, b = alpha_per_trial[idx], y_per_trial[idx]
        if a.std() == 0 or b.std() == 0:
            rs.append(0.0)
            continue
        rs.append(float(np.corrcoef(a, b)[0, 1]))
    return r_point, float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def cohen_h(p1: float, p2: float) -> float:
    """Effect size for two proportions (Cohen 1988)."""
    p1 = min(max(p1, 1e-9), 1 - 1e-9)
    p2 = min(max(p2, 1e-9), 1 - 1e-9)
    phi1 = 2 * math.asin(math.sqrt(p1))
    phi2 = 2 * math.asin(math.sqrt(p2))
    return phi1 - phi2


def welch_t(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Welch's t and approximate dof; returns (t, df)."""
    if a.size < 2 or b.size < 2:
        return float('nan'), float('nan')
    va, vb = a.var(ddof=1), b.var(ddof=1)
    na, nb = a.size, b.size
    se = math.sqrt(va / na + vb / nb) if (va + vb) > 0 else float('nan')
    t = (a.mean() - b.mean()) / se if se else float('nan')
    df_num = (va / na + vb / nb) ** 2
    df_den = ((va / na) ** 2) / (na - 1) + ((vb / nb) ** 2) / (nb - 1)
    df = df_num / df_den if df_den > 0 else float('nan')
    return t, df


def per_condition_stats(rows: list[dict]) -> dict:
    """Compute summary stats for one condition's trials."""
    if not rows:
        return {'n': 0}
    bet_ratios = np.array([r['outcome']['bet_ratio'] for r in rows], dtype=float)
    amounts = np.array([r['outcome']['amount'] for r in rows], dtype=float)
    stops = np.array([1.0 if r['outcome']['action'] == 'stop' else 0.0 for r in rows])
    balances = np.array([r['source_state']['balance_in_prompt'] for r in rows], dtype=float)
    lo_br, hi_br = bootstrap_mean_ci(bet_ratios)
    return {
        'n': int(len(rows)),
        'mean_bet_ratio': float(bet_ratios.mean()),
        'sd_bet_ratio': float(bet_ratios.std(ddof=1)) if bet_ratios.size > 1 else 0.0,
        'ci95_bet_ratio': [lo_br, hi_br],
        'mean_amount': float(amounts.mean()),
        'stop_rate': float(stops.mean()),
        'mean_balance_in_prompt': float(balances.mean()),
        '_bet_ratios': bet_ratios.tolist(),
        '_stops': stops.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['gemma', 'llama'], required=True)
    ap.add_argument('--task', choices=['sm', 'ic'], default='sm')
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f'{args.model}_{args.task}.json'
    md_path = OUT_DIR / f'{args.model}_{args.task}_summary.md'

    # Per-condition aggregation
    per_cond = {}
    for cond in DOSE_LADDER + SPECIFICITY:
        rows = load_trials_for_condition(args.model, args.task, cond)
        per_cond[cond] = per_condition_stats(rows)

    # Dose-response: only conditions with data
    dose_alpha_per_trial = []
    dose_br_per_trial = []
    dose_means = []
    for cond in DOSE_LADDER:
        s = per_cond[cond]
        if s['n'] == 0:
            dose_means.append(None)
            continue
        a = ALPHA_LADDER[cond]
        for br in s['_bet_ratios']:
            dose_alpha_per_trial.append(a)
            dose_br_per_trial.append(br)
        dose_means.append({'alpha': a, 'mean_bet_ratio': s['mean_bet_ratio'],
                           'ci95': s['ci95_bet_ratio'], 'n': s['n']})

    if len(dose_alpha_per_trial) >= 3:
        a_arr = np.array(dose_alpha_per_trial)
        b_arr = np.array(dose_br_per_trial)
        r_point, r_lo, r_hi = bootstrap_pearson_ci(a_arr, b_arr)
        # Spearman (rank-based, robustness)
        from scipy.stats import spearmanr
        rho, p_rho = spearmanr(a_arr, b_arr)
        dose_response = {
            'pearson_r': r_point,
            'pearson_ci95': [r_lo, r_hi],
            'spearman_rho': float(rho),
            'spearman_p': float(p_rho),
            'n_trials_total': int(a_arr.size),
            'dose_means': dose_means,
        }
    else:
        dose_response = {
            'pearson_r': None,
            'pearson_ci95': [None, None],
            'spearman_rho': None,
            'spearman_p': None,
            'n_trials_total': len(dose_alpha_per_trial),
            'dose_means': dose_means,
            'note': 'Insufficient data for dose-response (need at least 3 trials).',
        }

    # Effect-size contrasts on stop rate
    contrasts = {}
    if per_cond['alpha-2']['n'] > 0 and per_cond['alpha+3']['n'] > 0:
        contrasts['cohen_h_stop_extreme'] = cohen_h(
            per_cond['alpha-2']['stop_rate'], per_cond['alpha+3']['stop_rate'])
    if per_cond['alpha+0']['n'] > 0 and per_cond['alpha+2']['n'] > 0:
        contrasts['cohen_h_stop_canonical'] = cohen_h(
            per_cond['alpha+0']['stop_rate'], per_cond['alpha+2']['stop_rate'])

    # Specificity: each null vs alpha+2 should produce a smaller effect
    specificity = {}
    if per_cond['alpha+2']['n'] > 0:
        ref_br = np.array(per_cond['alpha+2']['_bet_ratios'])
        ref_stop = per_cond['alpha+2']['stop_rate']
        for spec in SPECIFICITY:
            s = per_cond[spec]
            if s['n'] == 0:
                continue
            spec_br = np.array(s['_bet_ratios'])
            t, df = welch_t(ref_br, spec_br)
            specificity[spec] = {
                'n': s['n'],
                'mean_bet_ratio': s['mean_bet_ratio'],
                'stop_rate': s['stop_rate'],
                'cohen_h_stop_vs_alpha+2': cohen_h(ref_stop, s['stop_rate']),
                'welch_t_bet_ratio_vs_alpha+2': t,
                'welch_df': df,
            }

    # Strip raw arrays before saving
    per_cond_clean = {}
    for cond, s in per_cond.items():
        per_cond_clean[cond] = {k: v for k, v in s.items() if not k.startswith('_')}

    out = {
        'model': args.model,
        'task': args.task,
        'per_condition': per_cond_clean,
        'dose_response': dose_response,
        'contrasts': contrasts,
        'specificity': specificity,
    }
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'[save] {out_path}')

    # Markdown summary
    lines = [
        f'# M3\' steering — {args.model.upper()} / {args.task.upper()}',
        '',
        '## Per-condition',
        '',
        '| condition | n | mean bet_ratio | 95% CI | stop rate | mean $ |',
        '|---|---|---|---|---|---|',
    ]
    for cond in DOSE_LADDER + SPECIFICITY:
        s = per_cond_clean[cond]
        if s['n'] == 0:
            lines.append(f'| {cond} | 0 | — | — | — | — |')
            continue
        lo, hi = s['ci95_bet_ratio']
        lines.append(
            f"| {cond} | {s['n']} | {s['mean_bet_ratio']:.3f} | "
            f"[{lo:.3f}, {hi:.3f}] | {s['stop_rate']:.3f} | "
            f"{s['mean_amount']:.1f} |")

    lines.extend(['', '## Dose-response (predictor → controller test)'])
    if dose_response['pearson_r'] is not None:
        r = dose_response['pearson_r']
        lo, hi = dose_response['pearson_ci95']
        lines.append(
            f"- Pearson r(α_σ, bet_ratio) = {r:+.3f} "
            f"(95% CI [{lo:+.3f}, {hi:+.3f}], n={dose_response['n_trials_total']})")
        lines.append(
            f"- Spearman ρ = {dose_response['spearman_rho']:+.3f} "
            f"(p = {dose_response['spearman_p']:.3g})")
    else:
        lines.append('- Insufficient trials for dose-response yet.')

    lines.extend(['', '## Effect sizes'])
    if 'cohen_h_stop_extreme' in contrasts:
        lines.append(
            f"- Cohen h (stop rate, α=-2 vs α=+3): "
            f"{contrasts['cohen_h_stop_extreme']:+.3f}")
    if 'cohen_h_stop_canonical' in contrasts:
        lines.append(
            f"- Cohen h (stop rate, α=0 vs α=+2): "
            f"{contrasts['cohen_h_stop_canonical']:+.3f}")

    if specificity:
        lines.extend(['', '## Specificity (vs alpha+2 reference)'])
        for spec, sd in specificity.items():
            lines.append(
                f"- **{spec}**: n={sd['n']}, mean bet_ratio={sd['mean_bet_ratio']:.3f}, "
                f"stop rate={sd['stop_rate']:.3f}, "
                f"Cohen h vs α=+2 (stop) = {sd['cohen_h_stop_vs_alpha+2']:+.3f}, "
                f"Welch t (bet_ratio) = {sd['welch_t_bet_ratio_vs_alpha+2']:.2f} "
                f"(df={sd['welch_df']:.1f})")

    with open(md_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'[save] {md_path}')


if __name__ == '__main__':
    main()
