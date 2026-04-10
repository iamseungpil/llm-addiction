# Workspace Runbook

This note defines how to keep `sae_v3_analysis` reproducible while reducing
noise in the working tree.

## 1. Priority order

When deciding whether to move, ignore, upload, or cite an artifact, use this
order:

1. `paper-critical`
2. `runtime-critical`
3. `legacy-reusable`
4. `generated-noise`

## 2. Paper-critical assets

These are the first files to inspect when validating the Korean paper.

### Manifests and audits

- `docs/PAPER_CANONICAL.md`
- `docs/PAPER_MANIFEST.md`
- `docs/EXPERIMENT_INDEX.md`
- `docs/HIDDEN_SUBSPACE_AUDIT.md`
- `results/paper_neural_audit.json`
- `results/shared_subspace_hidden_audit_20260410.json`

### Neural paper code

- `src/build_paper_neural_audit.py`
- `src/plot_neural_figures.py`
- `src/run_comprehensive_robustness.py`
- `src/run_perm_null_ilc.py`
- `src/run_probe_selectivity_controls.py`

### Canonical raw outputs cited by the paper

- `results/v17_nonlinear_deconfound.txt`
- `results/robustness/permutation_null_ilc.json`
- `results/robustness/probe_selectivity_controls_smoke.json`
- `results/json/v14_exp1_llama_sm_perm20_20260331_153127.json`
- `results/json/v12_crossdomain_steering.json`

### Paper-critical external data roots

- `/home/v-seungplee/data/llm-addiction/behavioral`
- `/home/v-seungplee/data/llm-addiction/sae_features_v3`

These paths should not be renamed or moved during cleanup.

## 3. Runtime-critical assets

These files may not appear in the paper directly, but current scripts still
depend on them or on their existing directory layout.

- `src/`
- `scripts/`
- `results/json/`
- `results/logs/`
- `results/robustness/`
- `results/figures/`

Do not move raw JSON, logs, or runner scripts unless every dependent script is
updated in the same change.

## 4. Legacy-reusable assets

These remain useful as historical context or fallback references, but they are
not current paper sources.

### Historical study snapshots

- top-level `results/sae_v*_*.md`
- top-level `results/sae_v*_*.pdf`
- top-level `results/sae_v*_*.tex`

### Historical helpers and monitor traces

- old `results/build_v*.py` and `results/export_v*.py`
- old `results/generate_*report*.py`
- old `results/v14_*` monitor files
- exploratory runners such as:
  - `src/run_causal_patching.py`
  - `src/run_causal_patching_v2.py`
  - `src/run_mediation_analysis.py`
  - `src/run_ablation_a1_a3.py`
  - `src/run_b1_b2_bruteforce.py`

These should be indexed as legacy before moving or de-emphasizing them.

## 5. Generated-noise

These should be ignored or removed whenever safe.

- `__pycache__/`
- LaTeX build products:
  - `*.aux`
  - `*.log`
  - `*.out`
  - `*.fls`
  - `*.fdb_latexmk`
  - `*.xdv`
  - `*.toc`
- stale lock files and ad hoc monitor stdout dumps

## 6. Cleanup policy

### Safe moves

- historical report companion artifacts such as `*.aux`, `*.log`, `*.out`,
  `*.toc`
- stale monitor logs that are no longer used by active runners
- duplicate presentation-only artifacts

### Not safe to move without coordinated code edits

- `results/json/*.json`
- `results/robustness/*.json`
- `results/logs/*.log`
- any file under `src/` that still has active references in docs or scripts
- any external data root under `/home/v-seungplee/data/llm-addiction`

## 7. HF / git policy

### Git

Commit:

- manifests
- indexes
- paper-safe code
- plan documents
- safe legacy moves

Do not commit:

- transient runtime logs from active experiments
- generated LaTeX build outputs

### Hugging Face

Upload:

- paper manifests
- canonical raw outputs needed to reproduce paper numbers
- key runner scripts for paper-critical analyses
- compact reference notes that explain provenance

Do not upload:

- every exploratory log
- duplicated compiled artifacts
- obvious generated noise

## 8. Current experimental priority

1. finish workspace curation and manifest cleanup
2. run full RQ1 selectivity controls
3. run aligned low-rank RQ2 transfer smoke
4. expand to full RQ2 transfer only if smoke passes
5. run aligned-factor steering only if aligned transfer gives a stable gain
