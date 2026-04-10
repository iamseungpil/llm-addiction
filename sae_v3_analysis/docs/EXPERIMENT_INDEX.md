# Experiment Index

This file maps each experiment family to its code, input data, main outputs, and paper-facing role.

## 1. Canonical data paths

| Item | Path | Notes |
| --- | --- | --- |
| Behavioral root | `/home/v-seungplee/data/llm-addiction/behavioral` | Raw model-game logs |
| SAE root | `/home/v-seungplee/data/llm-addiction/sae_features_v3` | Round-level sparse SAE features |
| Analysis code | `/home/v-seungplee/llm-addiction/sae_v3_analysis/src` | Main scripts |
| Analysis outputs | `/home/v-seungplee/llm-addiction/sae_v3_analysis/results` | Figures, JSON, logs, reports |
| Paper repo | `/home/v-seungplee/LLM_Addiction_NMT_KOR` | Korean paper source |

## 2. Paradigm and model map

| Code | Paradigm | Behavioral folders | SAE folders |
| --- | --- | --- | --- |
| `ic` | Investment Choice | `behavioral/investment_choice/v2_role_gemma`, `behavioral/investment_choice/v2_role_llama` | `sae_features_v3/investment_choice/gemma`, `sae_features_v3/investment_choice/llama` |
| `sm` | Slot Machine | `behavioral/slot_machine/gemma_v4_role`, `behavioral/slot_machine/llama_v4_role` | `sae_features_v3/slot_machine/gemma`, `sae_features_v3/slot_machine/llama` |
| `mw` | Mystery Wheel | `behavioral/mystery_wheel/gemma_v2_role`, `behavioral/mystery_wheel/llama_v2_role` | `sae_features_v3/mystery_wheel/gemma`, `sae_features_v3/mystery_wheel/llama` |

## 3. Shared infrastructure

| Purpose | Main files | What they do |
| --- | --- | --- |
| Path constants | `src/config.py` | Declares repo/data roots and paradigm metadata |
| Data loading | `src/data_loader.py` | Loads round-level SAE features, decision-point features, and hidden states |
| Report extraction | `extract_all_results.py`, `extract_final_rq.py`, `extract_rq_summary.py`, `print_rq_summary.py` | Pulls results into report-oriented summaries |
| Batch helpers | `run_s1.sh`, `run_extract_gemma.sh`, `run_extract_llama_mw.sh`, `run_gemma_steering.sh`, `run_llama_ic_mw.sh` | Convenience entry points for larger runs |

## 4. Experiment families

### A. Feature extraction and hidden-state prep

| Family | Main scripts | Inputs | Outputs | Notes |
| --- | --- | --- | --- | --- |
| All-round extraction | `src/extract_all_rounds.py`, `src/extract_all_hidden_states.py` | Behavioral logs, model checkpoints | `sae_features_L*.npz`, `phase_a_hidden_states.npz` | Creates the round-level source of truth |
| Decision-point extraction | `src/extract_all_layers_dp.py`, `src/extract_early_round_hidden_states.py` | Hidden states / game traces | `hidden_states_dp.npz` and early-round assets | Used by hidden-state and steering analyses |
| LLaMA task-specific extraction | `src/extract_llama_ic.py`, `src/extract_llama_sm.py`, `src/extract_llama_mw.py`, `src/extract_llama_hidden_states.py` | LLaMA raw runs | LLaMA-specific feature dumps | Historical extraction path for cross-model symmetry |

### B. RQ1: bankruptcy signal inside each task

| Family | Main scripts | Inputs | Outputs | Paper role |
| --- | --- | --- | --- | --- |
| Basic BK classification | `src/classify_bk.py`, `src/run_all_analyses.py`, `src/run_improved_v4.py` | SAE features, game labels | figure-ready AUC summaries | Early core evidence |
| Early prediction / trajectory | `src/round_trajectory.py`, `src/run_v7_r1_transfer.py` | Early-round hidden states or SAE features | `results/json/round_trajectory_*.json`, temporal figures | Supports “signal appears early” claims |
| Balance confound audit | `src/run_balance_confound_analysis.py`, `src/plot_balance_confound.py` | SAE features, balances | `results/figures/exp2a_balance_controlled.png`, related tables | Controls for balance leakage |
| Deconfounded robustness | `src/run_comprehensive_robustness.py`, `src/run_perm_null.py`, `src/run_perm_null_ilc.py`, `src/run_ilc_diagnostic.py` | Round-level SAE features, metadata, behavioral logs | `results/robustness/*`, `results/v17_nonlinear_deconfound.txt` | Current canonical reviewer-facing robustness path |
| Probe selectivity controls | `src/run_probe_selectivity_controls.py` | Same as above | `results/robustness/probe_selectivity_*.json` | Stronger nuisance-control variant for RQ1 |

### C. RQ2: cross-domain transfer and shared structure

| Family | Main scripts | Inputs | Outputs | Paper role |
| --- | --- | --- | --- | --- |
| Feature-level cross-domain | `src/run_v7_cross_domain_features.py`, `src/cross_domain.py` | SAE features across tasks | transfer AUC summaries and overlap stats | Main correlational transfer evidence |
| Cross-bettype / shared factor | `src/run_f1_cross_bettype_transfer.py`, `src/run_gap_filling.py`, `src/run_hidden_gaps.py` | Task-paired features and labels | transfer matrices and diagnostics | Explains partial invariance structure |
| LLaMA symmetry analyses | `src/run_llama_symmetric.py`, `src/run_llama_v10_symmetric.py`, `src/run_llama_3paradigm.py` | LLaMA IC/SM/MW assets | `results/json/llama_*` summaries | Cross-model symmetry and extension |
| Cross-domain summary | `src/analyze_v12_results.py` | prior JSON results | `results/json/v12_cross_analysis_summary.json`, figures | Consolidates transfer findings |

### D. RQ3: prompt and condition modulation

| Family | Main scripts | Inputs | Outputs | Paper role |
| --- | --- | --- | --- | --- |
| Condition encoding | `src/condition_analysis.py`, `src/condition_analysis_v2.py`, `src/generate_v8_condition_figures.py` | Condition-labeled features | `v5_fig4_condition_encoding.png`, related condition figures | Main condition-encoding evidence |
| LLaMA RQ3 update | `src/analyze_llama_rq3.py` | LLaMA condition-specific outputs | `results/json/v13_llama_rq3_analysis.json` | Extends RQ3 symmetry to LLaMA |
| Temperature and behavior modulation | `src/run_temperature_control.py`, `src/plot_temperature_robustness.py` | temperature-conditioned runs | `results/temperature_control/` outputs | Extra robustness / appendix-style analysis |
| Distortion / escalation probes | `src/run_distortion_quantification.py`, `src/run_multimodel_distortion_analysis.py`, `src/run_escalation_analysis.py` | behavior traces | `results/distortion*`, `results/escalation/` | Follow-up behavioral characterization |

### E. Causal steering and intervention

| Family | Main scripts | Inputs | Outputs | Notes |
| --- | --- | --- | --- | --- |
| Base within-domain steering | `src/run_v12_all_steering.py`, `src/run_v12_random_steering.py`, `src/run_bk_steering.py` | `hidden_states_dp.npz`, BK direction vectors | `results/json/v12_*.json`, steering figures | Canonical steering base |
| Cross-domain steering | `src/run_v12_crossdomain_steering.py` | source-task directions + target-task games | `results/json/v12_crossdomain_steering.json` | Causal transfer counterpart to RQ2 |
| V14 follow-up experiments | `src/run_v14_experiments.py`, `src/run_v14_parallel.py`, `src/run_v14_validation.sh`, `src/run_v14_gemma_followup.sh`, `results/monitor_v14_and_report.py` | V12 steering path + extra random controls | `results/json/v14_*`, `results/v14_*_log.txt`, monitor files | Follow-up random-control and symmetry work |
| Later steering refinements | `src/run_v15_steering.py`, `src/run_v16_multilayer_steering.py` | V12/V14 outputs and hidden states | `results/json/v15_*`, `results/json/v16_*`, `results/logs/v16_*` | Multilayer and specificity refinement |
| Other intervention probes | `src/run_causal_patching.py`, `src/run_causal_patching_v2.py`, `src/run_mediation_analysis.py`, `src/run_iba_cross_task_probe.py`, `src/run_ablation_a1_a3.py`, `src/run_b1_b2_bruteforce.py` | task-specific hidden states and labels | `results/json/causal_*`, ablation JSONs | Mostly exploratory or appendix-style |

### F. Figure and report generation

| Family | Main scripts | Inputs | Outputs | Notes |
| --- | --- | --- | --- | --- |
| Figure generation | `src/generate_behavioral_figures.py`, `src/generate_v13_figures.py`, `src/plot_neural_figures.py` | summary JSONs and tables | `results/figures/*.png` | Produces presentation assets |
| Historical report builds | `results/build_v12_pdf.py`, `results/build_v13_pdf.py`, `results/build_v14_pdf.py` | versioned markdown | `.tex`, `.pdf`, rendered figures | Snapshot reporting flow |
| Historical report generation | `results/generate_v14_report.py`, `results/generate_report_pdf.py`, `results/generate_report_figures.py` | JSON + study markdown | study PDFs | Mostly archive / status docs |

## 5. Canonical documents by purpose

| Need | File |
| --- | --- |
| Current experiment plan | `results/reports/v23_workspace_and_rq_plan_20260410.md` |
| Current findings summary | `results/reports/v17_final_neural_findings.md` |
| Paper-safe neural manifest | `results/paper_neural_audit.json` |
| Hidden-state RQ2 audit | `results/shared_subspace_hidden_audit_20260410.json` |
| Session history and decisions | `results/session_progress_20260331.md` |
| Historical integrated study | `results/sae_v13_comprehensive_study.md` |
| Historical integrated study (Korean) | `results/sae_v13_comprehensive_study_ko.md` |

## 6. Paper mapping

| Paper area | Main evidence source in `sae_v3_analysis` |
| --- | --- |
| RQ1 neural signal | `results/robustness/`, `results/v17_nonlinear_deconfound.txt`, BK classification figures |
| RQ2 cross-domain transfer | `results/json/v12_crossdomain_steering.json`, `results/shared_subspace_hidden_audit_20260410.json`, `results/figures/v13_fig3_crossdomain_transfer.png`, `results/figures/v13_fig4_crossdomain_steering.png` |
| RQ3 condition effects | `results/json/v13_llama_rq3_analysis.json`, condition figures, prompt-component figures |
| Steering appendix / causal paragraph | `results/json/v12_*`, `results/json/v14_*`, `results/json/v16_*`, corresponding logs |

The paper text itself is maintained in `/home/v-seungplee/LLM_Addiction_NMT_KOR`.

## 7. Recommended workflow

1. Start from `results/reports/v23_workspace_and_rq_plan_20260410.md`.
2. Open the raw artifact in `results/json/` or `results/robustness/`.
3. Check the corresponding runner in `src/`.
4. Verify the input folder in `/home/v-seungplee/data/llm-addiction/`.
5. Only then update the paper repo.

For cleanup and retention policy, also read `docs/WORKSPACE_RUNBOOK.md`.

## 8. Intentional non-changes

- No raw data paths were moved.
- No experiment script names were changed.
- No historical files were deleted.

This index is intended to make the existing structure navigable without breaking current scripts or references.
