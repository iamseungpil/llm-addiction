"""E7: mixed-effects re-analysis of §3 condition × bet-type effects.

Reviewer concern: §3's per-model differences and pooled-mean comparisons
("variable betting raises I_BA from 0.10 to 0.29") ignore clustering —
games are nested within models, rounds within games. A reviewer asks
whether the bet_type and prompt-component effects survive when the
clustering is modeled explicitly.

Two complementary checks:

  (1) Game-level (binary outcome = bankruptcy):
      Logistic GEE with cluster=model_id; report bet_type and G/M
      coefficients with cluster-robust SE. Bonus: per-game-id bootstrap
      95% CI on the bankruptcy rate gap (variable − fixed) per model.

  (2) Round-level (continuous outcome = bet_ratio):
      Linear mixed model on bet_ratio with fixed effects bet_type,
      G, M and random intercept per (model, game_id). Restricted to
      Gemma + LLaMA local data where round-level details are full.

Output:
  results/v19_multi_patching/E7_mixed_effects_section3/results.json
  results/v19_multi_patching/E7_mixed_effects_section3/_summary.md
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/v-seungplee/LLM_Addiction_NMT_KOR')

OUT_DIR = Path('/home/v-seungplee/llm-addiction/sae_v3_analysis/results/'
               'v19_multi_patching/E7_mixed_effects_section3')
LOCAL_SM = Path('/home/v-seungplee/data/llm-addiction/behavioral/slot_machine')

PROMPT_FEATS = ['G', 'M', 'H', 'W', 'P']


def explode_local(model_name: str, model_dir: str) -> pd.DataFrame:
    """Return long-format DF with one row per (game, round) for local model."""
    files = sorted((LOCAL_SM / model_dir).glob('final_*.json'))
    rows = []
    for f in files:
        d = json.load(open(f))
        games = d.get('results', d.get('games', []))
        if isinstance(games, dict):
            games = list(games.values())
        for g_idx, g in enumerate(games):
            game_id = g.get('game_id', g_idx)
            bt = g.get('bet_type')
            combo = g.get('prompt_combo', '') or ''
            outcome = str(g.get('outcome', g.get('final_outcome', ''))).lower()
            bankrupt = bool(g.get('bankruptcy')) or outcome in ('bankrupt', 'bankruptcy')
            decs = g.get('decisions', [])
            row = {
                'model': model_name,
                'game_id': f'{model_name}::{game_id}',
                'bet_type': bt,
                'is_variable': 1 if bt == 'variable' else 0,
                'bankrupt': int(bankrupt),
                'n_rounds': len(decs),
            }
            for f_ in PROMPT_FEATS:
                row[f'has_{f_}'] = 1 if f_ in combo else 0
            rows.append(row)
            for r_idx, dec in enumerate(decs):
                bet = dec.get('bet') or dec.get('parsed_bet') or dec.get('bet_amount')
                bal = dec.get('balance_before')
                if bet is None or bal is None or float(bal) <= 0 or float(bet) <= 0:
                    continue
                ratio = min(float(bet) / float(bal), 1.0)
                rrow = dict(row)
                rrow.update({
                    'level': 'round',
                    'round_idx': r_idx,
                    'bet': float(bet),
                    'balance': float(bal),
                    'bet_ratio': ratio,
                })
                rows.append(rrow)
    df = pd.DataFrame(rows)
    df['level'] = df.get('level', 'game').fillna('game')
    return df


def gee_logistic_game_level(df: pd.DataFrame, scope: str = 'variable_only') -> dict:
    """Game-level GEE: bankrupt ~ G + M with cluster=model.

    Restricted to variable bet_type by default — under fixed bet_type, BK
    rate is 0 for Gemma and 0.4% for LLaMA, which produces perfect
    separation in a logistic model. The §3 reviewer concern is about
    prompt-component effects under autonomy, so this scope matches the
    canonical comparison.
    """
    from statsmodels.genmod.generalized_estimating_equations import GEE
    from statsmodels.genmod.cov_struct import Exchangeable
    from statsmodels.genmod.families import Binomial

    games = df[df['level'] == 'game'].copy()
    if scope == 'variable_only':
        games = games[games['is_variable'] == 1]
    if games.empty:
        return {'n_games': 0, 'scope': scope}

    games['model_idx'] = pd.Categorical(games['model']).codes
    formula = 'bankrupt ~ has_G + has_M'
    try:
        model = GEE.from_formula(formula, groups='model_idx', data=games,
                                 cov_struct=Exchangeable(), family=Binomial())
        res = model.fit()
        coefs = {name: float(res.params[name]) for name in res.params.index}
        ses = {name: float(res.bse[name]) for name in res.params.index}
        zvals = {name: float(res.tvalues[name]) for name in res.params.index}
        pvals = {name: float(res.pvalues[name]) for name in res.params.index}
        return {
            'n_games': int(games.shape[0]),
            'n_clusters': int(games['model'].nunique()),
            'scope': scope,
            'coefs': coefs, 'ses': ses, 'z': zvals, 'p': pvals,
            'note': 'logistic GEE bankrupt ~ has_G + has_M, cluster=model_id, '
                    'exchangeable; variable-betting subset only.',
        }
    except Exception as e:
        return {'n_games': int(games.shape[0]), 'scope': scope,
                'error': f'{type(e).__name__}: {e}'}


def bootstrap_bk_gap(df: pd.DataFrame, n_iter: int = 1000, seed: int = 42) -> dict:
    """Per-model bootstrap-by-game CI on bankruptcy rate gap (variable - fixed)."""
    games = df[df['level'] == 'game']
    rng = np.random.RandomState(seed)
    out = {}
    for model_name, sub in games.groupby('model'):
        var_games = sub[sub['is_variable'] == 1]
        fix_games = sub[sub['is_variable'] == 0]
        if var_games.empty or fix_games.empty:
            continue
        var_n, fix_n = var_games.shape[0], fix_games.shape[0]
        gaps = []
        for _ in range(n_iter):
            v = var_games.iloc[rng.randint(0, var_n, var_n)]['bankrupt'].mean()
            f = fix_games.iloc[rng.randint(0, fix_n, fix_n)]['bankrupt'].mean()
            gaps.append(v - f)
        out[model_name] = {
            'point_gap': float(var_games['bankrupt'].mean() - fix_games['bankrupt'].mean()),
            'ci95': [float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))],
            'n_var': int(var_n), 'n_fix': int(fix_n),
        }
    return out


def cluster_bootstrap_round(df: pd.DataFrame, n_iter: int = 1000,
                             seed: int = 42) -> dict:
    """Cluster-bootstrap (over game_id) of OLS bet_ratio ~ has_G + has_M.

    Replacement for LMM when RE covariance is singular: resamples games,
    refits OLS each time, and reports point estimate + 95% percentile CI
    on each coefficient. This is the cluster-robust alternative the
    reviewer is actually asking for.
    """
    from sklearn.linear_model import LinearRegression
    rounds = df[(df['level'] == 'round') & (df['is_variable'] == 1)].copy()
    if rounds.empty:
        return {'n_rounds': 0}
    rounds = rounds.dropna(subset=['bet_ratio', 'has_G', 'has_M'])

    X_cols = ['has_G', 'has_M']
    games = rounds['game_id'].unique()
    by_game = {g: rounds[rounds['game_id'] == g] for g in games}

    full_X = rounds[X_cols].values.astype(float)
    full_y = rounds['bet_ratio'].values.astype(float)
    point = LinearRegression().fit(full_X, full_y)
    point_coefs = dict(zip(X_cols, point.coef_.tolist()))
    point_coefs['Intercept'] = float(point.intercept_)

    rng = np.random.RandomState(seed)
    boots = {c: [] for c in X_cols + ['Intercept']}
    for _ in range(n_iter):
        idx = rng.randint(0, len(games), size=len(games))
        chunks = [by_game[games[j]] for j in idx]
        sub = pd.concat(chunks, ignore_index=True)
        Xs, ys = sub[X_cols].values.astype(float), sub['bet_ratio'].values.astype(float)
        if Xs.shape[0] < 10:
            continue
        try:
            est = LinearRegression().fit(Xs, ys)
            for c, v in zip(X_cols, est.coef_):
                boots[c].append(float(v))
            boots['Intercept'].append(float(est.intercept_))
        except Exception:
            continue

    cis = {c: [float(np.percentile(boots[c], 2.5)),
               float(np.percentile(boots[c], 97.5))]
           if boots[c] else [None, None]
           for c in boots}
    return {
        'n_rounds': int(rounds.shape[0]),
        'n_games': int(rounds['game_id'].nunique()),
        'scope': 'variable_only',
        'point_coefs': point_coefs,
        'ci95': cis,
        'note': 'Cluster bootstrap over game_id, OLS bet_ratio ~ has_G + has_M.',
    }


def lmm_round_level(df: pd.DataFrame, scope: str = 'variable_only') -> dict:
    """Round-level mixed model: bet_ratio ~ has_G + has_M + (1|game_id).

    Restricted to variable bet_type to avoid bet_type collinearity.
    """
    from statsmodels.regression.mixed_linear_model import MixedLM

    rounds = df[df['level'] == 'round'].copy()
    if scope == 'variable_only':
        rounds = rounds[rounds['is_variable'] == 1]
    if rounds.empty:
        return {'n_rounds': 0, 'scope': scope}
    rounds = rounds.dropna(subset=['bet_ratio', 'has_G', 'has_M'])

    formula = 'bet_ratio ~ has_G + has_M'
    try:
        md = MixedLM.from_formula(formula, groups='game_id', data=rounds)
        mdf = md.fit(method='lbfgs', maxiter=400, disp=False)
        coefs = {name: float(mdf.params[name]) for name in mdf.params.index
                 if name in mdf.bse.index}
        ses = {name: float(mdf.bse[name]) for name in mdf.params.index
               if name in mdf.bse.index}
        zvals = {name: float(mdf.tvalues[name]) for name in mdf.params.index
                 if name in mdf.bse.index}
        pvals = {name: float(mdf.pvalues[name]) for name in mdf.params.index
                 if name in mdf.bse.index}
        return {
            'n_rounds': int(rounds.shape[0]),
            'n_games': int(rounds['game_id'].nunique()),
            'scope': scope,
            'coefs': coefs, 'ses': ses, 'z': zvals, 'p': pvals,
            'random_effect_var': float(mdf.cov_re.iloc[0, 0])
                                   if mdf.cov_re is not None else None,
            'note': 'LMM bet_ratio ~ has_G + has_M + (1|game_id); variable-betting subset.',
        }
    except Exception as e:
        return {'n_rounds': int(rounds.shape[0]), 'scope': scope,
                'error': f'{type(e).__name__}: {e}'}


def per_model_pooled_means(df: pd.DataFrame) -> dict:
    """Recover §3 figures from this dataset for sanity check."""
    games = df[df['level'] == 'game']
    rounds = df[df['level'] == 'round']
    out = {}
    for model_name, sub_g in games.groupby('model'):
        m = {'bet_type': {}}
        for bt, gsub in sub_g.groupby('bet_type'):
            br_rounds = rounds[(rounds['model'] == model_name)
                                & (rounds['bet_type'] == bt)]
            m['bet_type'][bt] = {
                'n_games': int(gsub.shape[0]),
                'bk_rate': float(gsub['bankrupt'].mean()),
                'mean_bet_ratio': (float(br_rounds['bet_ratio'].mean())
                                   if not br_rounds.empty else None),
            }
        out[model_name] = m
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print('[load] Gemma local')
    gemma_df = explode_local('Gemma-2-9B', 'gemma_v4_role')
    print(f'  rows: {gemma_df.shape[0]} ({(gemma_df["level"]=="game").sum()} games, '
          f'{(gemma_df["level"]=="round").sum()} rounds)')

    print('[load] LLaMA local')
    llama_df = explode_local('LLaMA-3.1-8B', 'llama_v4_role')
    print(f'  rows: {llama_df.shape[0]} ({(llama_df["level"]=="game").sum()} games, '
          f'{(llama_df["level"]=="round").sum()} rounds)')

    df_all = pd.concat([gemma_df, llama_df], ignore_index=True)

    pooled = per_model_pooled_means(df_all)
    print(f'\n[sanity] pooled means per model:')
    for m, v in pooled.items():
        for bt, stats in v['bet_type'].items():
            print(f'  {m} {bt}: n_games={stats["n_games"]} bk_rate={stats["bk_rate"]:.3f} '
                  f'mean_bet_ratio={stats["mean_bet_ratio"]}')

    print('\n[gee] game-level GEE (variable-betting subset)')
    gee = gee_logistic_game_level(df_all, scope='variable_only')
    if 'coefs' in gee:
        for term in gee['coefs']:
            print(f'  {term}: β={gee["coefs"][term]:+.3f} (SE {gee["ses"][term]:.3f}, '
                  f'z={gee["z"][term]:.2f}, p={gee["p"][term]:.3g})')
    else:
        print(f'  GEE failed: {gee.get("error", "n_games=0")}')

    print('\n[boot] per-model bootstrap-by-game CI on (variable − fixed) BK gap')
    boot = bootstrap_bk_gap(df_all, n_iter=2000)
    for m, v in boot.items():
        lo, hi = v['ci95']
        print(f'  {m}: gap={v["point_gap"]:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}]')

    print('\n[lmm] round-level LMM (bet_ratio, variable-betting subset)')
    lmm = lmm_round_level(df_all, scope='variable_only')
    if 'coefs' in lmm:
        for term in lmm['coefs']:
            print(f'  {term}: β={lmm["coefs"][term]:+.4f} (SE {lmm["ses"][term]:.4f}, '
                  f'z={lmm["z"][term]:.2f}, p={lmm["p"][term]:.3g})')
    else:
        print(f'  LMM failed: {lmm.get("error", "n_rounds=0")}')

    print('\n[boot-round] cluster-bootstrap over game_id (OLS bet_ratio)')
    boot_round = cluster_bootstrap_round(df_all, n_iter=2000)
    if 'point_coefs' in boot_round:
        for term in boot_round['point_coefs']:
            lo, hi = boot_round['ci95'][term]
            print(f"  {term}: β={boot_round['point_coefs'][term]:+.4f}, "
                  f"95% CI [{lo:+.4f}, {hi:+.4f}]")

    out = {
        'pooled_means': pooled,
        'gee_game_level': gee,
        'bootstrap_bk_gap': boot,
        'lmm_round_level': lmm,
        'cluster_bootstrap_round': boot_round,
        'note': 'E7: clustering-aware re-analysis of §3 effects on local Gemma + LLaMA. '
                'API models excluded because round-level details require HF roundtrip; '
                'their game-level effect is reproduced via bootstrap_bk_gap when those '
                'rows are added.',
    }
    out_path = OUT_DIR / 'results.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\n[save] {out_path}')

    # Markdown summary
    lines = ['# E7 — clustering-aware §3 re-analysis', '',
             'Reviewer concern: §3 pools across games and ignores game-level',
             'clustering. We refit the bet-type and ±G/±M effects with',
             'cluster-robust GEE (game-level) and a linear mixed model',
             '(round-level), reporting whether the §3 effects survive.',
             '', '## Sanity: pooled per-model means', '',
             '| model | bet_type | n_games | BK rate | mean bet_ratio |',
             '|---|---|---|---|---|']
    for m, v in pooled.items():
        for bt, stats in v['bet_type'].items():
            br = ('—' if stats['mean_bet_ratio'] is None
                  else f'{stats["mean_bet_ratio"]:.3f}')
            lines.append(f"| {m} | {bt} | {stats['n_games']} | "
                         f"{stats['bk_rate']:.3f} | {br} |")

    lines.extend(['', '## Game-level logistic GEE (cluster=model)', ''])
    if 'coefs' in gee:
        lines.append(f"Scope: {gee.get('scope', '?')}, "
                     f"n_games={gee['n_games']}, n_clusters={gee['n_clusters']}.")
        lines.append('')
        lines.append('| term | β | SE | z | p |')
        lines.append('|---|---|---|---|---|')
        for term in gee['coefs']:
            lines.append(f"| {term} | {gee['coefs'][term]:+.3f} | "
                         f"{gee['ses'][term]:.3f} | {gee['z'][term]:.2f} | "
                         f"{gee['p'][term]:.3g} |")
    else:
        lines.append(f"GEE failed: `{gee.get('error', 'unknown')}`")

    lines.extend(['', '## Per-model bootstrap-by-game CI on (variable − fixed) BK gap', '',
                  '| model | gap | 95% CI | n_var | n_fix |',
                  '|---|---|---|---|---|'])
    for m, v in boot.items():
        lo, hi = v['ci95']
        lines.append(f"| {m} | {v['point_gap']:+.3f} | "
                     f"[{lo:+.3f}, {hi:+.3f}] | {v['n_var']} | {v['n_fix']} |")

    lines.extend(['', '## Round-level LMM on bet_ratio (RE: game_id)', ''])
    if 'coefs' in lmm:
        lines.append(f"Scope: {lmm.get('scope', '?')}, "
                     f"n_rounds={lmm['n_rounds']}, n_games={lmm['n_games']}.")
        lines.append('')
        lines.append('| term | β | SE | z | p |')
        lines.append('|---|---|---|---|---|')
        for term in lmm['coefs']:
            lines.append(f"| {term} | {lmm['coefs'][term]:+.4f} | "
                         f"{lmm['ses'][term]:.4f} | {lmm['z'][term]:.2f} | "
                         f"{lmm['p'][term]:.3g} |")
        rev = lmm.get('random_effect_var')
        if rev is not None:
            lines.append('')
            lines.append(f"Random-effect variance (game_id intercept): {rev:.5f}")
    else:
        lines.append(f"LMM failed: `{lmm.get('error', 'unknown')}`")

    lines.extend(['', '## Round-level cluster-bootstrap over game_id', ''])
    if 'point_coefs' in boot_round:
        lines.append(f"Scope: variable-betting only. n_rounds={boot_round['n_rounds']}, "
                     f"n_games={boot_round['n_games']}.")
        lines.append('')
        lines.append('| term | β | 95% CI |')
        lines.append('|---|---|---|')
        for term in boot_round['point_coefs']:
            lo, hi = boot_round['ci95'][term]
            ci = f"[{lo:+.4f}, {hi:+.4f}]" if lo is not None else '—'
            lines.append(f"| {term} | {boot_round['point_coefs'][term]:+.4f} | {ci} |")

    md_path = OUT_DIR / '_summary.md'
    with open(md_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'[save] {md_path}')


if __name__ == '__main__':
    main()
