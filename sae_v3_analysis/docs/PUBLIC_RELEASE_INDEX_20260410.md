# Public Release Index (2026-04-10)

This note defines the public-facing layout for the current LLM addiction
workspace so that paper numbers, code, and upstream data can be found without
guesswork.

## 1. Git repositories

| Role | Local path | Remote |
| --- | --- | --- |
| Analysis code + results + manifests | `/home/v-seungplee/llm-addiction/sae_v3_analysis` | `github.com/iamseungpil/llm-addiction` |
| Korean paper source | `/home/v-seungplee/LLM_Addiction_NMT_KOR` | `github.com/iamseungpil/LLM_Addiction_NMT_KOR` |

## 2. Hugging Face dataset layout

Dataset repo: `llm-addiction-research/llm-addiction`

| HF path | Contents | Primary use |
| --- | --- | --- |
| `sae_v3_analysis/docs/` | runbooks, manifests, indices | provenance and navigation |
| `sae_v3_analysis/src/` | analysis and experiment scripts | exact code used to produce reported artifacts |
| `sae_v3_analysis/scripts/` | cluster / launch helpers | rerunning larger sweeps |
| `sae_v3_analysis/results/` | JSON outputs, figures, robustness files, reports | reported results and paper-facing audits |
| `papers/LLM_Addiction_NMT_KOR/` | Korean paper source and compiled PDF | manuscript source of truth |
| `behavioral/` | raw behavioral logs for IC / SM / MW | paper inputs and behavior reanalysis |
| `sae_features_v3/` | sparse SAE features and hidden-state assets | neural analysis inputs |

## 3. First files to open

If the goal is to verify a paper claim, use this order:

1. `sae_v3_analysis/docs/PAPER_MANIFEST.md`
2. `sae_v3_analysis/docs/EXPERIMENT_INDEX.md`
3. `sae_v3_analysis/results/paper_neural_audit.json`
4. `sae_v3_analysis/results/shared_subspace_hidden_audit_20260410.json`
5. `papers/LLM_Addiction_NMT_KOR/sn-article.pdf`

## 4. Current paper-critical inputs

| Input type | HF path |
| --- | --- |
| Behavioral logs | `behavioral/investment_choice/*`, `behavioral/slot_machine/*`, `behavioral/mystery_wheel/*` |
| SAE features | `sae_features_v3/investment_choice/*`, `sae_features_v3/slot_machine/*`, `sae_features_v3/mystery_wheel/*` |
| Paper-facing neural outputs | `sae_v3_analysis/results/robustness/`, `sae_v3_analysis/results/json/`, `sae_v3_analysis/results/reports/` |

## 5. Release intent

This release is organized for traceability rather than minimal storage:

- GitHub remains the version-controlled source for code and paper text.
- Hugging Face contains the reproducibility snapshot: inputs, outputs, paper,
  and navigation docs in a stable directory layout.
- Historical files are preserved unless they are obvious generated noise.
