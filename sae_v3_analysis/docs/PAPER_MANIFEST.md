# Paper Manifest

This file is the paper-facing manifest for the Korean manuscript.

Each paper element should be traceable through:

1. paper element
2. exact artifact
3. generating script
4. input dataset path or upstream manifest

## Neural Section

| Paper element | Artifact | Script | Upstream input / note |
| --- | --- | --- | --- |
| Table 1 (`I_LC`, `I_BA`, `I_EC`) | `results/paper_neural_audit.json` | `src/build_paper_neural_audit.py` | `results/v17_nonlinear_deconfound.txt`, `results/robustness/permutation_null_ilc.json`, direct rerun assets |
| Figure 3 neural panel | `images/neural_analysis_combined.pdf` in paper repo | `src/plot_neural_figures.py` | reads `results/paper_neural_audit.json` |
| Figure 4 cross-paradigm transfer | `images/cross_paradigm_transfer.pdf` in paper repo | `src/plot_neural_figures.py` | representative sparse-transfer figure |
| RQ2 hidden-state geometry paragraph | `results/shared_subspace_hidden_audit_20260410.json` | summarized in `docs/HIDDEN_SUBSPACE_AUDIT.md` | decision-point `hidden_states_dp.npz` assets |
| RQ3 condition table | `results/paper_neural_audit.json` → `rq3_condition_i_ba` | `src/build_paper_neural_audit.py` | direct rerun subsets for Gemma/LLaMA SM `I_BA` |
| Same-domain steering paragraph | `results/paper_neural_audit.json` → `steering.same_domain` | `src/build_paper_neural_audit.py` | representative same-domain steering JSON |
| Cross-domain steering appendix table | `results/paper_neural_audit.json` → `steering.cross_domain_significant` | `src/build_paper_neural_audit.py` | supportive appendix evidence only |

## Behavioral / Language Section

| Paper element | Artifact | Script | Upstream input / note |
| --- | --- | --- | --- |
| Main behavioral figures/tables | behavioral source manifests in HF dataset | project behavioral aggregation scripts | source manifests record raw/corrected/local provenance |
| Language distortion reanalysis | canonical decision table, per-model CSVs, pooled CSVs, summary figure in HF dataset | language reanalysis scripts used for current manuscript | API-model raw text may be complemented by preserved local v4 archive |

## Rule

If a paper number cannot be mapped through this table, it should not appear in the manuscript until it is re-audited.
