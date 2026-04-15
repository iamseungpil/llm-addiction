# Results Directory Guide

This directory mixes raw outputs, paper figures, historical report snapshots, and monitoring files. Use the groups below as the stable entry points.

## Paper-safe files first

If you are validating a number or claim that appears in the Korean paper, check
these files before reading historical study markdown:

- `paper_neural_audit.json`
- `shared_subspace_hidden_audit_20260410.json`
- `v17_nonlinear_deconfound.txt`
- `robustness/permutation_null_ilc.json`
- `robustness/probe_selectivity_controls_smoke.json`

If a paper claim is not traceable to one of these artifacts or to the raw files
they cite, it should be treated as non-canonical until re-audited.

For steering specifically, read `reports/current_status_and_rerun_plan_20260414.md`
before treating any older `v12_*`, `v14_*`, or `v16_*` JSON as paper-safe.

## High-value subdirectories

### `json/`

Raw experiment outputs and machine-readable summaries.

- Use this first when you want the original numeric result of a run.
- Naming pattern:
  - `v12_*`: steering and cross-domain steering
  - `v13_*`: RQ3 and integrated analysis summaries
  - `v14_*`: causal follow-up runs
  - `v15_*`, `v16_*`: later steering follow-ups
  - `aligned_steering_*`: exact-replay causal runs using empirical behavioral condition catalogs

### `figures/`

Paper/report-ready figures.

- `v13_fig*.png` are the main integrated-study figures.
- Earlier `v4_`, `v5_`, `v10_`, `v12_` files are versioned historical figures.

### `logs/`

Runtime logs from longer jobs.

- Use these to confirm whether a run is active, stalled, or finished.
- The most relevant current-style steering logs follow names such as `v16_*_run.log`.
- Exact-replay steering code path:
  - `src/exact_behavioral_replay.py`
  - used by `src/run_aligned_factor_steering.py` and `src/run_v16_multilayer_steering.py`
  - replays empirical `prompt_condition × bet_type × bet_constraint` mixtures from `data/behavioral/*`

### `robustness/`

Reviewer-facing robustness and selectivity outputs.

- Includes the most relevant outputs for the current “is the signal real vs nuisance” question.
- This is the first place to look for:
  - permutation-null outputs
  - selectivity controls
  - deconfounded probe metrics

### Paper-safe manifests

- `paper_neural_audit.json`
  - canonical paper-facing neural numbers for the current Korean paper
- `shared_subspace_hidden_audit_20260410.json`
  - auxiliary hidden-state audit used to qualify the RQ2 shared-geometry claim

### `reports/`

Human-readable plans, audits, and decision documents.

- Latest runtime status:
  - `reports/current_status_and_rerun_plan_20260414.md`
- Latest planning anchor:
  - `reports/v23_workspace_and_rq_plan_20260410.md`
- Latest findings anchor:
  - `reports/v17_final_neural_findings.md`
- Earlier plans remain useful as historical context, but should not override newer ones without checking dates.

### `legacy/`

Historical clutter that is preserved for reference but is not part of the
current paper-safe path.

- This is the right place for:
  - stale monitor traces
  - build companion artifacts
  - older presentation-only outputs

## Top-level file types

### Historical integrated studies

- `sae_v*_*.md`
- `sae_v*_*.pdf`
- `sae_v*_*.tex`

These are versioned report snapshots. Treat them as archives unless a later plan explicitly points back to one.

### Build/export scripts

- `build_v12_pdf.py`
- `build_v13_pdf.py`
- `build_v14_pdf.py`
- `generate_v14_report.py`

These turn study markdown into PDF/figure bundles.

### Monitoring files

- `v14_monitor_status.md`
- `v14_monitor_status.json`
- `v14_automation.log`

These summarize automation state. They do not replace the raw `json/` and `logs/` artifacts.

## Recommended lookup order

1. Read `reports/v23_workspace_and_rq_plan_20260410.md` for what is currently considered valid.
2. Read `reports/current_status_and_rerun_plan_20260414.md` if the claim touches steering or reruns.
3. Open the relevant raw output in `json/` or `robustness/`.
4. Check the corresponding runtime log in `logs/` if the result looks incomplete.
5. Only then consult the older `sae_v*.md/.pdf` snapshots for presentation context.
6. Check `legacy/` only if you explicitly need historical auxiliary files.

## Steering provenance rule

If a steering result is intended to support a paper claim, it must satisfy both:

1. The run uses `src/exact_behavioral_replay.py` rather than the older hand-written
   prompt/game sandbox.
2. The result JSON or log records the replayed behavioral condition catalog
   (prompt condition, bet type, and bet constraint where applicable).

Older steering files remain useful as side evidence or archived history, but
they should not be treated as exact behavioral replications.

## Useful anchors

- Current rerun status:
  - `reports/current_status_and_rerun_plan_20260414.md`
- Session progress log: `session_progress_20260331.md`
- Integrated Korean report snapshot: `sae_v13_comprehensive_study_ko.md`
- Integrated English report snapshot: `sae_v13_comprehensive_study.md`
- Paper incorporation notes: `paper_incorporation_plan.md`
- Paper-safe artifact map: `../docs/PAPER_CANONICAL.md`
- Hidden-state shared-subspace note: `../docs/HIDDEN_SUBSPACE_AUDIT.md`
