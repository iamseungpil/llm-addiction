# Paper Canonical Paths

This note pins down the smallest code and artifact set that should be treated
as canonical for the Korean paper.

For paper-element level provenance, use `docs/PAPER_MANIFEST.md` as the first
entry point.

## 1. Paper-critical scripts

| Purpose | Canonical file | Notes |
| --- | --- | --- |
| Build paper-facing neural metrics manifest | `src/build_paper_neural_audit.py` | Single source for paper numbers and provenance |
| Round-level metric utilities | `src/run_comprehensive_robustness.py` | Loads SAE features and computes `I_BA` |
| Loss-chasing label construction | `src/run_perm_null_ilc.py` | Computes `I_LC` labels used by paper audits |
| Strict CV evaluation helpers | `src/run_probe_selectivity_controls.py` | Reused for audited `R^2` computation |
| Paper neural figures | `src/plot_neural_figures.py` | Must read `results/paper_neural_audit.json`; no hard-coded paper numbers |

## 2. Paper-critical artifacts

| Artifact | Path | Status |
| --- | --- | --- |
| Archived nonlinear-deconfounded sweep | `results/v17_nonlinear_deconfound.txt` | Locked source for archived `I_LC` table values |
| Steering representative case | `results/json/v14_exp1_llama_sm_perm20_20260331_153127.json` | Same-domain steering |
| Cross-domain steering summary | `results/json/v12_crossdomain_steering.json` | Exploratory transfer steering |
| Paper metrics manifest | `results/paper_neural_audit.json` | Generated canonical summary |
| Hidden-state shared-subspace audit | `results/shared_subspace_hidden_audit_20260410.json` | Auxiliary RQ2 audit for paper-safe hidden-state geometry claims |

## 3. Archive-only families

The following are useful research history, but should not be used as direct
paper sources without re-auditing:

- `results/reports/v18_comprehensive_study.md`
- `results/reports/v19_status_and_plan.md`
- `results/sae_v13_comprehensive_study*.md`
- `src/run_v12_*`, `src/run_v14_*`, `src/run_v15_*`, `src/run_v16_*` beyond the
  specific JSON artifacts cited above
- older hidden-state and cross-model report generators

## 4. Rule

If a number appears in the paper, it must be traceable to one of:

1. `results/paper_neural_audit.json`
2. the raw JSON/TXT artifact referenced inside that manifest

If a number is not in that chain, it is not paper-safe.
