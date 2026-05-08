# E2 — balance-stratified §4.3 modulation

Reviewer concern: Δ R² between condition subsets may be a balance confound.
Test: refit §4.3 pipeline within fixed balance windows. If condition
modulation is real, Δ R² should remain positive across all windows.


## gemma_sm_i_ba_L22

| balance window | +G R² | −G R² | Δ(+G,−G) | +M R² | −M R² | Δ(+M,−M) |
|---|---|---|---|---|---|---|
| Q_low | +0.1282 | +0.0458 | +0.0824 | +0.1660 | +0.1068 | +0.0592 |
| Q_mid | +0.1909 | +0.0443 | +0.1466 | +0.1504 | +0.1588 | -0.0084 |
| Q_high | +0.2480 | +0.1104 | +0.1376 | +0.1456 | +0.1521 | -0.0066 |

## gemma_sm_i_lc_L22

| balance window | +G R² | −G R² | Δ(+G,−G) | +M R² | −M R² | Δ(+M,−M) |
|---|---|---|---|---|---|---|
| Q_low | -0.0017 | -0.0106 | +0.0089 | +0.0589 | -0.0079 | +0.0668 |
| Q_mid | +0.0100 | +0.0517 | -0.0417 | +0.0584 | +0.0498 | +0.0086 |
| Q_high | -0.0337 | +0.0093 | -0.0430 | -0.0100 | +0.0036 | -0.0136 |

## llama_sm_i_ba_L22

| balance window | +G R² | −G R² | Δ(+G,−G) | +M R² | −M R² | Δ(+M,−M) |
|---|---|---|---|---|---|---|
| Q_low | +0.0819 | +0.0636 | +0.0183 | +0.0748 | +0.0584 | +0.0164 |
| Q_mid | +0.1597 | +0.1132 | +0.0465 | +0.1201 | +0.0996 | +0.0204 |
| Q_high | +0.2124 | +0.1746 | +0.0378 | +0.1876 | +0.1463 | +0.0413 |

## llama_sm_i_lc_L22

| balance window | +G R² | −G R² | Δ(+G,−G) | +M R² | −M R² | Δ(+M,−M) |
|---|---|---|---|---|---|---|
| Q_low | +0.1623 | +0.1395 | +0.0228 | +0.1472 | +0.1218 | +0.0254 |
| Q_mid | +0.0744 | +0.0926 | -0.0183 | +0.0586 | +0.0861 | -0.0275 |
| Q_high | +0.0767 | +0.0941 | -0.0174 | +0.0806 | +0.0713 | +0.0093 |
