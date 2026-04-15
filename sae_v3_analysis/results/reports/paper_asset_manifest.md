# Paper Asset Manifest — Verified (2026-04-13)

Every paper claim → exact file that produced it. All file paths verified by direct inspection.

**Master source of truth for all neural numbers**: `sae_v3_analysis/results/paper_neural_audit.json` (5.2 KB). This single file contains the numbers appearing in Table 1, Table 2, and the RQ3 steering paragraph.

Path conventions:
- `code/` → `/home/v-seungplee/llm-addiction/`
- `data/` → `/home/v-seungplee/data/llm-addiction/`
- `paper/` → `/home/v-seungplee/LLM_Addiction_NMT_KOR/`

---

## Section 3.1 — Behavioral Experiments

### Slot Machine (6 models × 64 conditions × 50 reps = 19,200 games per model)
- **Source data**: `data/behavioral/slot_machine/{gemma_v4_role,llama_v4_role}/final_{model}_{timestamp}.json` (25,600 total games across 6 models; local + mirrored on HuggingFace)
- **Runner**: `code/paper_experiments/slot_machine_6models/src/llama_gemma_experiment.py` (LLaMA/Gemma), `run_{gpt5,claude,gemini}_experiment.py` (API models)
- **Original prompt**: includes `ROLE_INSTRUCTION` + 32 combinations of {G, M, H, W, P} components, max 100 rounds, variable or fixed $10 bets
- **Paper**: §3.1 behavioral section

### Investment Choice
- **Source data**: `data/behavioral/investment_choice/{v2_role_gemma,v2_role_llama}/{gemma,llama}_investment_c{10,30,50,70}_*.json` (1,600 games each)
- **Storage quirk**: IC stores `full_prompt` per decision (unlike SM which reconstructs it)
- **Runner**: `code/paper_experiments/investment_choice_experiment/src/` (plus v2_role variant)
- **Paper**: §3.1 ablation

### Mystery Wheel
- **Source data**: `data/behavioral/mystery_wheel/{gemma_v2_role,llama_v2_role}/{gemma,llama}_mysterywheel_c*.json`
- **Storage quirk**: MW also stores `full_prompt` per decision
- **Runner**: `code/exploratory_experiments/alternative_paradigms/src/mystery_wheel/run_experiment.py`
- **Paper**: §3.1 (MW is part of the paradigm triplet used throughout §3.2)

---

## Section 3.2 — Neural Analysis

### Table 1 (`tab:sae-results`) — SAE per-round readout R²

**Master file**: `code/sae_v3_analysis/results/paper_neural_audit.json` → sections `rq1_ilc` and `rq1_direct`.

| Paper cell | audit file section | Value |
|---|---|---|
| Gemma SM I_LC 0.244 | `rq1_ilc.gemma_sm.r2` | 0.2436 |
| Gemma MW I_LC 0.482 | `rq1_ilc.gemma_mw.r2` | 0.4823 |
| Gemma IC I_LC 0.476 | `rq1_ilc.gemma_ic.r2` | 0.4755 |
| LLaMA SM I_LC 0.325 | `rq1_ilc.llama_sm.r2` | 0.3245 |
| LLaMA MW I_LC 0.779 | `rq1_ilc.llama_mw.r2` | 0.7791 |
| LLaMA IC I_LC 0.381 | `rq1_ilc.llama_ic.r2` | 0.3807 |
| Gemma SM I_BA 0.161 | `rq1_direct.gemma_sm_i_ba.r2` | 0.1612 |
| LLaMA SM I_BA 0.121 | `rq1_direct.llama_sm_i_ba.r2` | 0.1211 |
| Gemma MW I_BA 0.056 | `rq1_direct.gemma_mw_i_ba.r2` | 0.0563 |
| LLaMA MW I_BA 0.068 | `rq1_direct.llama_mw_i_ba.r2` | 0.0681 |
| Gemma SM I_EC 0.053 | `rq1_direct.gemma_sm_i_ec.r2` | 0.0534 |
| LLaMA SM I_EC 0.042 | `rq1_direct.llama_sm_i_ec.r2` | 0.0418 |

- **Source features**: `data/sae_features_v3/{slot_machine,investment_choice,mystery_wheel}/{gemma,llama}/sae_features_L{0..41}.npz` + `extraction_summary.json`
- **Feature extraction runner**: `code/sae_v3_analysis/src/extract_all_rounds.py` (per-round); uses stored `full_prompt` for IC/MW and reconstructed prompt for SM
- **Readout runner**: v18 pipeline (RF deconfound → top-200 Spearman → Ridge α=100, 5-fold CV)
- **Supporting report**: `code/sae_v3_analysis/results/reports/v18_comprehensive_study.md`

### Table 2 (`tab:behavior-convergence`) — Behavioral metrics per (model × task)

**Source**: computed directly from raw behavioral JSONs (not cached). Derivation script is inline in `code/sae_v3_analysis/results/reports/rq2_causal_experiment_plan_v3.md` v4 addendum + recomputable from `data/behavioral/*/*/*.json`.

| Paper cell | Derivation |
|---|---|
| Gemma SM I_LC 0.307, I_BA 0.173, I_EC 0.029 | Aggregated over 3,200 SM games |
| Gemma IC I_LC 0.323, I_BA 0.211, I_EC 0.095 | 1,600 IC games |
| Gemma MW I_LC 0.404, I_BA 0.095, I_EC 0.002 | 3,200 MW games |
| LLaMA SM I_LC 0.166, I_BA 0.225, I_EC 0.135 | 3,200 SM games |
| LLaMA IC I_LC 0.296, I_BA 0.343, I_EC 0.364 | 1,600 IC games |
| LLaMA MW I_LC 0.345, I_BA 0.285, I_EC 0.175 | 3,200 MW games |

### Table 3 (`tab:condition-modulation`) — RQ3 condition-wise R²

**Master file**: `code/sae_v3_analysis/results/paper_neural_audit.json` → `rq3_condition_i_ba`.

Every cell in Table 3 comes from `rq3_condition_i_ba.{gemma,llama}_sm_i_ba.subsets.{all_variable,plus_G,minus_G,plus_M,minus_M,fixed_all}.r2`.

### Table — `tab:selectivity-controls` (RQ1 selectivity)

**Source**: `code/sae_v3_analysis/results/robustness/probe_selectivity_controls.json`

| Paper cell | JSON key |
|---|---|
| Gemma SM I_LC real 0.246 / ctrl −0.065 / p=0.048 | `gemma_sm_L24_i_lc.{real_group_r2, control_r2_mean, p_selectivity}` |
| LLaMA SM I_LC real 0.345 / ctrl −0.107 / p=0.048 | `llama_sm_L16_i_lc.{...}` |
| Gemma MW I_BA real 0.058 / ctrl −0.099 / p=0.048 | `gemma_mw_L24_i_ba.{...}` |
| LLaMA MW I_BA real 0.069 / ctrl −0.123 / p=0.048 | `llama_mw_L16_i_ba.{...}` |

**Runner**: `code/sae_v3_analysis/src/run_probe_selectivity_controls.py`

### Other RQ1 robustness checks

- **Full-pipeline permutation null (Gemma SM I_LC)** — paper claim: "R² null mean −0.022 ± 0.003, actual 0.244, p=0.019"
  - Source: `code/sae_v3_analysis/results/robustness/permutation_null_ilc.json` (key `gemma_sm_L24`)
- **Game-level block permutation I_BA** — paper claim: "4 configs all p<0.005"
  - Source: `code/sae_v3_analysis/results/robustness/permutation_null.json` (keys `gemma_sm_L24`, `gemma_mw_L24`, `llama_sm_L16`, `llama_mw_L16`)

### Figure `fig:neural-analysis` — Layer profile + condition modulation

- **Panel (a) layer profile**: input from raw SAE features L0–L41 + v18 pipeline result. Direct CSV: `code/sae_v3_analysis/results/gemma_sm_42layer_sweep.csv` (Gemma) and `gemma_sm_42layer_sweep_sae.csv` (SAE version). LLaMA equivalent is computed inline in the figure generation script.
- **Panel (b) condition modulation**: `paper_neural_audit.json → rq3_condition_i_ba`
- **Figure generator**: `paper/generate_paper_figures.py` (in paper repo root)
- **Output**: `paper/images/neural_analysis_combined.pdf`

### Figure `fig:cross-transfer` — RQ2 sparse transfer failure

- **Source**: `code/sae_v3_analysis/results/iba_cross_task_transfer.json` (top-level keys: `gemma`, `llama`; contains R² values for SM↔MW transfers)
- **Figure output**: `paper/images/cross_paradigm_transfer.pdf`

### Figure `fig:condition-modulation`

- **Source**: same as Table 3 (`paper_neural_audit.json → rq3_condition_i_ba`)
- **Figure output**: `paper/images/condition_modulation_iba.pdf`

---

## Section 3.2 — RQ2 Aligned Hidden Transfer Sweep (v24 report)

**Full sweep results directory**: `code/sae_v3_analysis/results/robustness/rq2_aligned_hidden_transfer_{model}_centroid_pca_L{layer}_r{rank}_*.json`

| Paper cell | File |
|---|---|
| Gemma L12 r1 6/6 beat random+shuffled | `rq2_aligned_hidden_transfer_gemma_centroid_pca_L12_r1_e8g1_L12_r1.json` |
| Gemma L22 r1 +0.102 AUC mean | `rq2_aligned_hidden_transfer_gemma_centroid_pca_L22_r1_e8g1_L22_r1.json` |
| Gemma L22 r1 SM shared 0.800 vs residual 0.639 | `rq2_aligned_hidden_transfer_gemma_centroid_pca_L22_r1_e8g1_L22_r1.json → summary.readout_decomposition.sm` |
| LLaMA L25 r1 +0.092 AUC mean | (running on E8, pending — will be `rq2_aligned_hidden_transfer_llama_centroid_pca_L25_r1_e8_llama_*.json`) |
| LLaMA L22 r1 IC shared 0.791 vs residual 0.480 | (on E8, pending full run) |

**Aggregate report**: `code/sae_v3_analysis/results/reports/v24_rq2_sweep_summary_20260410.md`
**Runner**: `code/sae_v3_analysis/src/run_rq2_aligned_hidden_transfer.py` + `run_rq2_aligned_hidden_transfer_sweep.py`

---

## Section 3.2 — RQ3 Steering (current paper numbers)

**Paper claim**: LLaMA SM L22, ρ=0.919, p=0.003, perm p=0.048, BK 38%→60%

**Master file**: `paper_neural_audit.json → steering.same_domain`
```json
{"model":"llama","task":"sm","layer":22,"rho":0.919,"p":0.003437,"permutation_p":0.0476,"baseline_bk":0.55}
```

**Underlying raw data** (multiple related runs):
- `code/sae_v3_analysis/results/json/v12_n200_20260327_030745.json` — 200 trials LLaMA SM L22; contains `bk_direction.rho=0.9643`, `bk_direction.p=0.000454` (updated run, stronger than paper's 0.919)
- `code/sae_v3_analysis/results/json/v12_random_steering_20260326_172803.json` — 20 random directions for permutation null
- `code/sae_v3_analysis/results/json/bk_steering_20260325_102705.json` — initial BK direction steering
- `code/sae_v3_analysis/results/json/v12_llama_sm_L22_20260328_091923.json` — layer-specific rerun

**Cross-task steering** (paper: sign reversal ρ=−0.964, −0.818)
- `code/sae_v3_analysis/results/json/v12_crossdomain_steering.json` → `within_domain_rho`, `cross_domain_results`
- `paper_neural_audit.json → steering.cross_domain_significant`

**Multi-layer robustness** (V16, local VM):
- `code/sae_v3_analysis/results/json/v16_llama_20260408_175255.json` — L8, L12, L22, L25 completed (ρ=0.929, −0.179, 0.793, 0.750)
- `code/sae_v3_analysis/results/reports/v24_rq2_sweep_summary_20260410.md` — interpretation

**Figures**:
- `paper/images/steering_dose_response.pdf` — panels (a) within-domain, (b) cross-domain
- Generator: `paper/generate_paper_figures.py`

### Steering replay rule

For any new claim-bearing steering rerun, the canonical runtime path is:

- `code/sae_v3_analysis/src/exact_behavioral_replay.py`
- `code/sae_v3_analysis/src/run_aligned_factor_steering.py`
- `code/sae_v3_analysis/src/run_v16_multilayer_steering.py`

This path reuses the empirical behavioral condition catalogs from:

- `data/behavioral/slot_machine/{gemma_v4_role,llama_v4_role}/`
- `data/behavioral/investment_choice/{v2_role_gemma,v2_role_llama}/`
- `data/behavioral/mystery_wheel/{gemma_v2_role,llama_v2_role}/`

and therefore preserves:

- prompt-condition mixtures,
- fixed/variable bet-type mixtures,
- IC/MW constraint settings,
- model-specific ROLE-instruction asymmetry present in the original behavioral runs.

---

## Section 3.2 — RQ2/RQ3 New Causal Experiments (v4, 2026-04-13, in progress)

**Status**: 4 parallel runs on E8 node with corrected prompts. See `code/sae_v3_analysis/results/reports/rq2_causal_experiment_plan_v3.md` §v4 addendum for full protocol.

| Experiment | GPU | Script | Output (pending) |
|---|:---:|---|---|
| Exp C — LLaMA L25 SM per-task steering (gate) | 0 | `run_aligned_factor_steering.py --experiment c --model llama` | `aligned_steering_C_llama_*.json` |
| V14 replication — LLaMA L22 SM | 1 | `run_v16_multilayer_steering.py --model llama --layers 2 --alpha-mode absolute --tag v14repl_fixed2` | `v16_llama_*_v14repl_fixed2.json` |
| Exp B — Gemma L12 shared axis | 2 | `run_aligned_factor_steering.py --experiment b --model gemma` | `aligned_steering_B_gemma_*.json` |
| Exp A — LLaMA L25 shared axis | 3 | `run_aligned_factor_steering.py --experiment a --model llama` | `aligned_steering_A_llama_*.json` |

All archived wrong-prompt attempts live under `results/json/archived_wrong_prompts/`.

---

## Code Runners (quick reference)

| Purpose | Script |
|---|---|
| SM/LLaMA+Gemma behavioral runner | `code/paper_experiments/slot_machine_6models/src/llama_gemma_experiment.py` |
| IC behavioral runner | `code/paper_experiments/investment_choice_experiment/src/run_experiment.py` |
| MW behavioral runner | `code/exploratory_experiments/alternative_paradigms/src/mystery_wheel/run_experiment.py` |
| Exact behavioral replay bridge for steering | `code/sae_v3_analysis/src/exact_behavioral_replay.py` |
| Per-round SAE feature extraction | `code/sae_v3_analysis/src/extract_all_rounds.py` |
| v18 comprehensive readout pipeline (Table 1) | `code/sae_v3_analysis/src/run_v18_*.py` (refer to v18 report for exact script name) |
| Selectivity controls (RQ1 Table) | `code/sae_v3_analysis/src/run_probe_selectivity_controls.py` |
| Permutation null I_BA | `code/sae_v3_analysis/src/run_perm_null.py` |
| Permutation null I_LC | `code/sae_v3_analysis/src/run_perm_null_ilc.py` |
| Cross-paradigm sparse transfer | `code/sae_v3_analysis/src/run_iba_cross_task_probe.py` |
| Hidden-state aligned low-rank transfer | `code/sae_v3_analysis/src/run_rq2_aligned_hidden_transfer.py` |
| Multi-layer sweep wrapper | `code/sae_v3_analysis/src/run_rq2_aligned_hidden_transfer_sweep.py` |
| V14/V12 per-task BK steering | `code/sae_v3_analysis/src/run_v12_all_steering.py` |
| V16 multi-layer steering | `code/sae_v3_analysis/src/run_v16_multilayer_steering.py` |
| Shared-axis causal steering (v4 Exp A/B/C) | `code/sae_v3_analysis/src/run_aligned_factor_steering.py` |

---

## HuggingFace Mirror

Root: `llm-addiction-research/llm-addiction` (dataset repo)

Top-level folders:
- `behavioral/` — raw game JSONs, mirrors `data/behavioral/`
- `sae_features_v3/` — SAE sparse features per layer per task per model, mirrors `data/sae_features_v3/`
- `sae_v3_analysis/` — analysis scripts, reports, and results (to be refreshed with this manifest)
- `paper_experiments/` — original behavioral runner scripts

The HF repo's README.md (root) should point to this manifest and the paper PDF.

---

## Verification checklist (2026-04-13)

- [x] All master numbers resolved to `paper_neural_audit.json`
- [x] v18 report marked as Table 1 interpretation, not source
- [x] Selectivity controls JSON verified (`probe_selectivity_controls.json`)
- [x] Permutation nulls verified (`permutation_null.json`, `permutation_null_ilc.json`)
- [x] V12/V14 steering files located and rho values cross-checked (0.919 in audit; 0.9643 in v12_n200 rerun)
- [x] RQ2 Gemma sweep files verified (15 JSON files under `results/robustness/`)
- [x] Cross-task transfer file located (`iba_cross_task_transfer.json`)
- [x] Archive policy documented (`archived_wrong_prompts/` for v4 wrong-prompt runs)
- [ ] v4 Exp A/B/C results (in progress)
- [ ] V14 replication under fixed prompts (in progress)
- [ ] HF mirror refresh with new manifest
