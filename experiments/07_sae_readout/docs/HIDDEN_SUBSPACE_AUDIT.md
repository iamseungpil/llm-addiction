# Hidden-State Shared-Subspace Audit

This note records the paper-safe auxiliary audit used to qualify the RQ2
interpretation.

## Main conclusion

Sparse-feature transfer failure does **not** imply complete absence of
cross-task shared structure. The audited hidden-state analyses instead support a
two-level interpretation:

1. A low-dimensional hidden-state geometry is partially shared across tasks.
2. Task-specific readout directions on top of that geometry can still diverge.

## Canonical artifact

- `results/shared_subspace_hidden_audit_20260410.json`

## Direct findings

### Gemma (IC/SM/MW, L22, direct rerun)

- Cross-domain transfer AUC remains strong in several directions:
  - `IC->SM = 0.7458`
  - `IC->MW = 0.8264`
  - `SM->MW = 0.9197`
  - `MW->SM = 0.8829`
- Per-task hidden-state BK weight vectors are still nearly orthogonal:
  - `cos(IC,SM) = 0.0422`
  - `cos(IC,MW) = -0.0262`
  - `cos(SM,MW) = -0.0262`
- Yet a 3D PCA basis built from those three weight vectors preserves high
  within-task separability:
  - `IC = 0.8647`
  - `SM = 0.9008`
  - `MW = 0.9714`

### LLaMA (IC/SM, direct rerun of V10 symmetric path)

- Cross-domain transfer is asymmetric rather than uniformly strong:
  - `SM->IC` rises from `0.6045` at `L8` to `0.7494` at `L30`
  - `IC->SM` is only above chance at `L8 = 0.5773` and falls below chance in
    deeper layers (`L22 = 0.4643`, `L30 = 0.4545`)
- Even so, the task-specific weight vectors at `L22` remain nearly orthogonal
  (`cos = 0.0383`), while the 2D subspace they span still supports strong
  within-task classification:
  - `IC = 0.9006`
  - `SM = 0.9428`

## Paper use

The safe paper claim is:

> cross-task divergence at the sparse-feature or local-direction level does not
> rule out partial shared hidden-state geometry; instead, the current evidence
> is most consistent with a shared low-dimensional risk subspace plus
> task-specific readouts.
