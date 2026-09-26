# E7 — clustering-aware §3 re-analysis

Reviewer concern: §3 pools across games and ignores game-level
clustering. We refit the bet-type and ±G/±M effects with
cluster-robust GEE (game-level) and a linear mixed model
(round-level), reporting whether the §3 effects survive.

## Sanity: pooled per-model means

| model | bet_type | n_games | BK rate | mean bet_ratio |
|---|---|---|---|---|
| Gemma-2-9B | fixed | 1600 | 0.000 | 0.102 |
| Gemma-2-9B | variable | 1600 | 0.054 | 0.264 |
| LLaMA-3.1-8B | fixed | 1600 | 0.004 | 0.101 |
| LLaMA-3.1-8B | variable | 1600 | 0.723 | 0.238 |

## Game-level logistic GEE (cluster=model)

Scope: variable_only, n_games=3200, n_clusters=2.

| term | β | SE | z | p |
|---|---|---|---|---|
| Intercept | -0.452 | 1.318 | -0.34 | 0.732 |
| has_G | +0.527 | 0.113 | 4.67 | 3.07e-06 |
| has_M | +0.473 | 0.228 | 2.07 | 0.0383 |

## Per-model bootstrap-by-game CI on (variable − fixed) BK gap

| model | gap | 95% CI | n_var | n_fix |
|---|---|---|---|---|
| Gemma-2-9B | +0.054 | [+0.043, +0.066] | 1600 | 1600 |
| LLaMA-3.1-8B | +0.719 | [+0.696, +0.741] | 1600 | 1600 |

## Round-level LMM on bet_ratio (RE: game_id)

Scope: variable_only, n_rounds=57797, n_games=3195.

| term | β | SE | z | p |
|---|---|---|---|---|
| Intercept | -0.0432 | nan | nan | nan |
| has_G | +0.0579 | nan | nan | nan |
| has_M | -0.0599 | nan | nan | nan |
| game_id Var | +0.0000 | nan | nan | nan |

Random-effect variance (game_id intercept): 0.00000

## Round-level cluster-bootstrap over game_id

Scope: variable-betting only. n_rounds=57797, n_games=3195.

| term | β | 95% CI |
|---|---|---|
| has_G | +0.0498 | [+0.0370, +0.0629] |
| has_M | +0.0555 | [+0.0424, +0.0686] |
| Intercept | +0.1921 | [+0.1827, +0.2023] |
