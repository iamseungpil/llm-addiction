# V24: RQ2 Gemma Aligned-Basis Sweep Summary

**Date**: 2026-04-10  
**Artifacts**: `results/robustness/rq2_aligned_hidden_transfer_gemma_centroid_pca_*_e8g1_*.json`  
**Question**: does RQ2 become stronger if the common substrate is modeled as a
low-rank hidden-state geometry rather than direct sparse-feature reuse?

## Executive Summary

The sweep supports a more precise version of the RQ2 claim.

- Sparse feature transfer still fails and should remain the negative control.
- A **low-rank aligned hidden basis** does recover reusable cross-task signal in
  Gemma, but only partially.
- The best results come from **rank-1** bases, not higher rank settings.
- The strongest paper-facing interpretation is therefore:
  **behavior converges, sparse circuits diverge, and a weak-to-moderate shared
  hidden geometry remains, but final readouts are still task-specific.**

This strengthens the current paper by replacing a single hidden-state audit with
a small, explicit layer/rank sweep.

## Sweep Grid

- model: `gemma`
- basis method: `centroid_pca`
- layers: `8, 12, 22, 25, 30`
- requested ranks: `1, 2, 3`
- note: with three tasks and leave-target-out construction, effective rank
  saturates quickly, which explains why rank-2 and rank-3 runs often collapse
  to the same result.

## Aggregate Pattern

| layer | rank | positive vs random | positive vs shuffled | mean aligned-random | mean aligned-shuffled |
|---|---:|---:|---:|---:|---:|
| 8 | 1 | 4/6 | 4/6 | 0.0455 | 0.0883 |
| 12 | 1 | 6/6 | 6/6 | 0.0827 | 0.0685 |
| 22 | 1 | 6/6 | 4/6 | 0.1020 | 0.0753 |
| 25 | 1 | 4/6 | 6/6 | 0.0723 | 0.0185 |
| 30 | 1 | 2/6 | 4/6 | 0.0053 | 0.0719 |

Higher-rank settings were never clearly better than `rank=1` and often became
weaker.

## Best Configurations

### Broadest positive pattern: `L12, rank=1`

- Beats both random and shuffled controls on all six pairwise transfers.
- Representative improvements:
  - `IC -> SM`: aligned AUC `0.8505`, `+0.1766` vs random, `+0.1763` vs shuffled
  - `MW -> IC`: aligned AUC `0.6201`, `+0.0395` vs random, `+0.0178` vs shuffled
  - `IC -> MW`: aligned AUC `0.6739`, `+0.0319` vs random, `+0.0113` vs shuffled
- Still below full target readouts (`0.9505` to `0.9786`), so this is not a
  shared end-to-end readout story.

### Highest mean gain over random: `L22, rank=1`

- Mean aligned-random gain across six transfers: `+0.1020`
- Strong directions:
  - `IC -> SM`: `+0.1742`
  - `MW -> IC`: `+0.1195`
- Weak directions remain:
  - `IC -> MW`: `+0.0122` vs random, but `-0.0771` vs shuffled
  - `SM -> MW`: same pattern

## Readout Decomposition

The decomposition reinforces the same story: the shared basis helps, but does
not replace task-specific readouts.

At `L22, rank=1`:

| task | shared norm fraction | shared-only AUC | residual-only AUC | full AUC |
|---|---:|---:|---:|---:|
| IC | 0.0312 | 0.7355 | 0.9336 | 0.9560 |
| MW | 0.0381 | 0.5242 | 0.9493 | 0.9452 |
| SM | 0.1124 | 0.8003 | 0.6391 | 0.9732 |

Interpretation:

- the shared component is real enough to improve over random controls;
- but the full task readout still depends heavily on task-specific residual
  structure;
- this supports **partial shared geometry + task-specific readout**, not a
  universal shared circuit.

## Paper-Facing Decision

Recommended main-text claim:

> Sparse feature transfer fails, but a low-rank aligned hidden-state basis
> recovers partial cross-task signal. The effect is strongest for a rank-1
> basis at mid-to-late layers and consistently remains below full task-specific
> readouts, indicating shared geometry with task-specific readout rather than a
> single reusable circuit.

Recommended main-text anchors:

- use `L12,r1` as the cleanest “broad positive across all directions” example
- use `L22,r1` as the “highest average gain” example
- keep higher-rank failures out of the main narrative and use them only as
  appendix support for the low-rank interpretation

## Next Step

1. Update `3.results.tex` RQ2 subsection with the sweep-based interpretation.
2. Update `4.discussion.tex` and `5.methods.tex` to match the new mainline.
3. Keep raw cross-domain steering as supplementary evidence rather than the core
   RQ2 argument.
