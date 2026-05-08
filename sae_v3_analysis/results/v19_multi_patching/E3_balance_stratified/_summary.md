# E3 — balance-stratified §4.2 BK audit

Reviewer concern: the §4.2 LOTO-PCA shared-axis AUC may track
balance rather than risk-ending semantics. We rerun §4.2 with
balance-matched subsamples (V1) and balance-residualised hidden
states (V2). If shared-axis AUC remains >> random baseline,
the geometry is not a balance artefact.

## GEMMA L22

Shared-axis AUC by held-out task and variant:

| held-out | baseline | V1 (bal-matched) | V2 (bal-residual) | rand baseline |
|---|---|---|---|---|
| sm | 0.688 | 0.568 | 0.543 | 0.661 |
| ic | 0.590 | 0.691 | 0.554 | 0.606 |
| mw | 0.527 | 0.522 | 0.544 | 0.691 |

## LLAMA L22

Shared-axis AUC by held-out task and variant:

| held-out | baseline | V1 (bal-matched) | V2 (bal-residual) | rand baseline |
|---|---|---|---|---|
| sm | 0.716 | 0.707 | 0.657 | 0.597 |
| ic | 0.690 | 0.572 | 0.655 | 0.650 |
| mw | 0.763 | 0.712 | 0.670 | 0.597 |

