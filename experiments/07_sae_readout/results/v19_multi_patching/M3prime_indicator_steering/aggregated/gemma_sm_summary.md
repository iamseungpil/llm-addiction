# M3' steering — GEMMA / SM

## Per-condition

| condition | n | mean bet_ratio | 95% CI | stop rate | mean $ |
|---|---|---|---|---|---|
| alpha-2 | 50 | 0.064 | [0.041, 0.088] | 0.580 | 6.5 |
| alpha-1 | 50 | 0.056 | [0.034, 0.081] | 0.620 | 5.9 |
| alpha+0 | 50 | 0.051 | [0.031, 0.073] | 0.640 | 4.9 |
| alpha+1 | 50 | 0.062 | [0.039, 0.087] | 0.620 | 6.0 |
| alpha+2 | 50 | 0.060 | [0.038, 0.083] | 0.600 | 6.0 |
| alpha+3 | 50 | 0.064 | [0.041, 0.089] | 0.600 | 6.7 |
| random | 50 | 0.064 | [0.042, 0.089] | 0.580 | 6.4 |
| L8 | 50 | 0.052 | [0.031, 0.075] | 0.660 | 4.9 |
| ILC | 50 | 0.066 | [0.043, 0.091] | 0.580 | 6.6 |

## Dose-response (predictor → controller test)
- Pearson r(α_σ, bet_ratio) = +0.013 (95% CI [-0.102, +0.127], n=300)
- Spearman ρ = +0.010 (p = 0.869)

## Effect sizes
- Cohen h (stop rate, α=-2 vs α=+3): -0.041
- Cohen h (stop rate, α=0 vs α=+2): +0.082

## Specificity (vs alpha+2 reference)
- **random**: n=50, mean bet_ratio=0.064, stop rate=0.580, Cohen h vs α=+2 (stop) = +0.041, Welch t (bet_ratio) = -0.26 (df=97.8)
- **L8**: n=50, mean bet_ratio=0.052, stop rate=0.660, Cohen h vs α=+2 (stop) = -0.124, Welch t (bet_ratio) = 0.50 (df=98.0)
- **ILC**: n=50, mean bet_ratio=0.066, stop rate=0.580, Cohen h vs α=+2 (stop) = +0.041, Welch t (bet_ratio) = -0.37 (df=97.7)
