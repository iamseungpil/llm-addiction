# 07 — SAE readout (Tables 1–3)

**Question.** Can the risk indicators (betting aggressiveness `I_BA`, loss chasing `I_LC`, extreme
choice `I_EC`) be read from the decision-time hidden state, do the three tasks share a risk
subspace, and does a goal prompt make the readout sharper?

**Paper result.** Finding 6, Table 1 (`tab:neurips-sae-results`, readout R² at layer 22 with fold
SEs and a 200-draw permutation null); Finding 7, Table 2 (`tab:rq2-sharing`, cross-task sharing);
Finding 8, Table 3 (`tab:condition-modulation`, condition modulation). Appendix: layer sweep, band
readout, selectivity controls and hidden-subspace audits.

## Design (from the code)

- Inputs: the open-weight games of [01](../01_slot_machine/README.md) (`*_v4_role`),
  [02](../02_investment_choice/README.md) (`v2_role`) and [06](../06_mystery_wheel/README.md)
  (`v2_role`), replayed to extract round-level hidden states and SAE features for Gemma-2-9B
  (Gemma Scope) and LLaMA-3.1-8B (Llama Scope).
- Readout (`src/run_groupkfold_recompute.py`, helpers in `src/run_perm_null_ilc.py`): layer 22;
  within each fold, a random forest removes balance and round terms from the target, the top 200
  SAE features by |Spearman ρ| are kept, then StandardScaler + Ridge(α = 100); 5-fold GroupKFold
  by game.
- Permutation null (`src/run_table1_perm_null.py`): 200 game-block permutations of the target.
- Cross-task sharing (`src/cross_domain.py`, `src/run_rq2_aligned_hidden_transfer_sweep.py`) and
  condition modulation (the `plus_G` / `minus_G` / `plus_M` / `minus_M` cells written by
  `src/run_groupkfold_recompute.py`; `src/condition_analysis_v2.py`).
- `src/exact_behavioral_replay.py` rebuilds prompts with the original runners of 01, 02 and 06.

## Quick Start

```bash
# Table 1 and Table 3 cells (reads sae_features_v3/ and behavioral/ at the paths in src/run_perm_null_ilc.py)
python experiments/07_sae_readout/src/run_groupkfold_recompute.py
python experiments/07_sae_readout/src/run_table1_perm_null.py
# Table 2
python experiments/07_sae_readout/src/run_rq2_aligned_hidden_transfer_sweep.py --help
# Runs anywhere, no data or GPU
python3 experiments/07_sae_readout/tests/test_distortion_outcome_schema.py
```

Data roots are absolute paths on the original machine (`/home/v-seungplee/data/llm-addiction/...`
in `src/config.py` and `src/run_perm_null_ilc.py`); point them at your copy of the dataset.
Results are written under `results/` in this folder.

## Data on Hugging Face

`sae_v3_analysis/results/` (the dataset keeps the old folder name) and `sae_features_v3/` (hidden
states and SAE features, tens of GB).

## Figures and tables (paper repository, private)

- `scripts/tables/body_tables.py` — Tables 1–3.
- `scripts/tables/table1_perm_null.py` — the Table 1 permutation column.
- `scripts/tables/appendix_neural_tables.py` — the neural appendix tables.

---

The rest of this page is the workspace guide written while the analysis ran. Absolute paths in it
refer to the original cluster. The steering scripts it names (v12/v14/v16) are superseded by the
Figure 4 battery in [08](../08_steering/README.md); the dataset keeps them under `legacy/` as
do-not-cite.

## SAE V3 analysis workspace (detailed guide)

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
- Table 1 = `results/table1_groupkfold_L22.json`; Table 3 = `results/condition_modulation_groupkfold_L22.json`; Table 2 = `results/rq2_aligned_hidden_transfer_*L22_r1*.json` (strict-CV GroupKFold pipeline)
- `results/shared_subspace_hidden_audit_20260410.json`

These files define the shortest reproducible path from paper text to code/data.

For cleanup policy and legacy rules, also read:

- `docs/WORKSPACE_RUNBOOK.md`
- `docs/PUBLIC_RELEASE_INDEX_20260410.md`

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
  - Table 1 = `results/table1_groupkfold_L22.json`; Table 3 = `results/condition_modulation_groupkfold_L22.json`; Table 2 = `results/rq2_aligned_hidden_transfer_*L22_r1*.json` (strict-CV GroupKFold pipeline)
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
