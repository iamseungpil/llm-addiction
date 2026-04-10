# SAE V3 Analysis

This directory is the main analysis workspace for the SAE-based gambling experiments used by the paper and follow-up robustness work.

## Source of truth

- The main paper-scale neural readout path is a **round-level SAE sparse feature pipeline**.
- The canonical raw inputs live outside the repo under `/home/v-seungplee/data/llm-addiction`.
- This directory mostly contains:
  - analysis code in `src/`
  - run helpers in the repo root and `scripts/`
  - experiment outputs in `results/`
  - planning and status notes in `results/reports/`

## Paper-safe entry points

If the immediate goal is to check or update the Korean paper, start from these
files before opening older reports:

- `docs/PAPER_CANONICAL.md`
- `docs/EXPERIMENT_INDEX.md`
- `docs/HIDDEN_SUBSPACE_AUDIT.md`
- `results/paper_neural_audit.json`
- `results/shared_subspace_hidden_audit_20260410.json`

These files define the shortest reproducible path from paper text to code/data.

For cleanup policy and legacy rules, also read:

- `docs/WORKSPACE_RUNBOOK.md`

## Quick navigation

### Data roots

- Behavioral logs: `/home/v-seungplee/data/llm-addiction/behavioral`
- SAE features: `/home/v-seungplee/data/llm-addiction/sae_features_v3`

### Core code

- Shared config: `src/config.py`
- Shared data loading: `src/data_loader.py`
- Main within-domain steering runner: `src/run_v12_all_steering.py`
- Current multilayer steering follow-up: `src/run_v16_multilayer_steering.py`
- Robustness / reviewer-check pipeline: `src/run_comprehensive_robustness.py`
- Round-level `I_LC` label path: `src/run_perm_null_ilc.py`
- Probe selectivity controls: `src/run_probe_selectivity_controls.py`

### Key outputs

- Raw JSON experiment dumps: `results/json/`
- Logs: `results/logs/`
- Figures: `results/figures/`
- Robustness outputs: `results/robustness/`
- Plans and status docs: `results/reports/`
- Paper-safe neural manifests:
  - `results/paper_neural_audit.json`
  - `results/shared_subspace_hidden_audit_20260410.json`
- Historical integrated studies: `results/sae_v*_*.md`, `results/sae_v*_*.pdf`

## Experiment structure

### RQ1: within-paradigm neural signal

- Classification and robustness:
  - `src/run_comprehensive_robustness.py`
  - `src/run_perm_null.py`
  - `src/run_perm_null_ilc.py`
  - `src/run_probe_selectivity_controls.py`
- Representative outputs:
  - `results/robustness/`
  - `results/v17_nonlinear_deconfound.txt`
  - `results/reports/v17_final_neural_findings.md`

### RQ2: cross-domain transfer and shared structure

- Correlational transfer and shared-subspace analyses:
  - `src/run_v7_cross_domain_features.py`
  - `src/run_hidden_state_analyses.py`
  - `src/cross_domain.py`
  - `src/analyze_v12_results.py`
  - `src/run_llama_v10_symmetric.py`
- Causal cross-domain steering:
  - `src/run_v12_crossdomain_steering.py`
  - `src/run_v12_all_steering.py`
- Representative outputs:
  - `results/json/v12_crossdomain_steering.json`
  - `results/shared_subspace_hidden_audit_20260410.json`
  - `results/figures/v13_fig3_crossdomain_transfer.png`
  - `results/figures/v13_fig4_crossdomain_steering.png`

### RQ3: condition and prompt modulation

- Condition-level analyses:
  - `src/condition_analysis.py`
  - `src/condition_analysis_v2.py`
  - `src/analyze_llama_rq3.py`
  - `src/run_temperature_control.py`
- Representative outputs:
  - `results/json/v13_llama_rq3_analysis.json`
  - `results/temperature_control/`
  - `results/figures/v5_fig4_condition_encoding.png`

### Causal follow-up / steering

- Canonical steering base:
  - `src/run_v12_all_steering.py`
- V14 follow-up wrappers and automation:
  - `src/run_v14_experiments.py`
  - `src/run_v14_parallel.py`
  - `results/monitor_v14_and_report.py`
- Later follow-up:
  - `src/run_v15_steering.py`
  - `src/run_v16_multilayer_steering.py`

## Behavioral and feature layout

Each paradigm is mirrored across behavioral logs and SAE feature dumps.

### Paradigms

- `investment_choice` = `IC`
- `slot_machine` = `SM`
- `mystery_wheel` = `MW`

### Models

- `gemma`
- `llama`

### Behavioral folders

- IC:
  - `/home/v-seungplee/data/llm-addiction/behavioral/investment_choice/v2_role_gemma`
  - `/home/v-seungplee/data/llm-addiction/behavioral/investment_choice/v2_role_llama`
- SM:
  - `/home/v-seungplee/data/llm-addiction/behavioral/slot_machine/gemma_v4_role`
  - `/home/v-seungplee/data/llm-addiction/behavioral/slot_machine/llama_v4_role`
- MW:
  - `/home/v-seungplee/data/llm-addiction/behavioral/mystery_wheel/gemma_v2_role`
  - `/home/v-seungplee/data/llm-addiction/behavioral/mystery_wheel/llama_v2_role`

### SAE feature folders

- IC:
  - `/home/v-seungplee/data/llm-addiction/sae_features_v3/investment_choice/gemma`
  - `/home/v-seungplee/data/llm-addiction/sae_features_v3/investment_choice/llama`
- SM:
  - `/home/v-seungplee/data/llm-addiction/sae_features_v3/slot_machine/gemma`
  - `/home/v-seungplee/data/llm-addiction/sae_features_v3/slot_machine/llama`
- MW:
  - `/home/v-seungplee/data/llm-addiction/sae_features_v3/mystery_wheel/gemma`
  - `/home/v-seungplee/data/llm-addiction/sae_features_v3/mystery_wheel/llama`

Each SAE folder typically contains:

- `sae_features_L*.npz`: sparse round-level SAE activations
- `sae_features_L*.json`: per-layer metadata summaries
- `extraction_summary.json`: extraction completeness summary
- `hidden_states_dp.npz`: decision-point hidden states
- optional `checkpoint/phase_a_hidden_states.npz`: all-round hidden states

## Best entry points

- If you want the current plan: `results/reports/v23_workspace_and_rq_plan_20260410.md`
- If you want the current findings summary: `results/reports/v17_final_neural_findings.md`
- If you want the March-to-April progress log: `results/session_progress_20260331.md`
- If you want a full artifact map: `docs/EXPERIMENT_INDEX.md`
- If you want output folder meanings: `results/README.md`
- If you want the smallest paper-safe file set: `docs/PAPER_CANONICAL.md`
- If you want the hidden-state RQ2 audit: `docs/HIDDEN_SUBSPACE_AUDIT.md`

## Caution

- Many `sae_v*.md/.pdf/.tex` files in `results/` are historical snapshots, not all equally current.
- `src/run_v14_causal_validation.py` exists, but the active steering base is still centered on `src/run_v12_all_steering.py` plus later wrappers.
- Avoid moving raw files unless every dependent script is updated; the current organization intentionally adds navigation without changing runtime paths.
- Historical clutter is being moved only into `results/legacy/`; runtime paths remain unchanged by policy.
