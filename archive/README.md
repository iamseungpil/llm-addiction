# archive — code the camera-ready paper does not use

Nothing here produced a figure or table of the camera-ready paper. It is kept for provenance: it
records what was tried, and some of it produced folders that are still on the Hugging Face dataset
(`llm-addiction-research/llm-addiction`). Each folder kept its internal structure; `archive/X` was
`X` before the reorganisation (see [`../PATH_MAP.md`](../PATH_MAP.md)). Paths inside these files
still describe the old layout, and most scripts write to absolute paths on the original machines.

Archived runners that `import common` still work: `exploratory_experiments/alternative_paradigms/src/common/__init__.py`
forwards to [`experiments/shared/common/`](../experiments/shared/README.md).

## `paper_experiments/`

| Folder | What it is | HF folder |
|---|---|---|
| `README.md` | The ICLR 2026 index of this folder (out of date) | — |
| `investment_choice_experiment/` | First 10-round investment-choice runner (API models) | `investment_choice/initial/` |
| `llama_sae_analysis/` | Earlier LLaMA SAE feature and activation-patching pipeline; its patching claims were withdrawn | `sae_patching/` |
| `pathway_token_analysis/` | Token-level pathway and word-feature analysis (submitted §5 draft) | `analysis/pathway_token_analysis/` |
| `m1_portfolio_discriminant/` | Rebuttal-period portfolio-task control (not cited) | none on this dataset |
| `m2_persona_decoupling/` | Rebuttal-period first-person vs role framing control (not cited; superseded by 05) | none on this dataset |
| `m5_compliance_residualisation/` | Residualising SAE features against compliance directions (not cited) | none on this dataset |
| `d_distributed_effect/` | Top-K SAE feature removal robustness (not cited) | none on this dataset |
| `track_L_length_confound/` | Game-length / survival confound re-analysis (not cited) | `paper_neurips_2026/track_L_length_confound/` |

## `legacy/`

"Legacy" was a historical folder name, not a verdict; the parts of it that produced paper data
moved to `experiments/` (the API slot-machine runners, the October 2025 LLaMA/Gemma runs, the
investment-choice API runners and the matched-cap restart runners). What is left:

| Folder | What it is | HF folder |
|---|---|---|
| `README.md` | The old status table of `legacy/` | — |
| `analysis/` | Early cross-model analysis scripts; `analyze_all_6_models.py` is marked do-not-cite | — |
| `causal_feature_discovery/` | Notes from the first causal-feature experiment | — |
| `experiment_2_multilayer_patching_L1_31/` | Multi-layer SAE patching across L1–31 (superseded) | — |
| `experiment_corrected_sae_analysis/` | Re-run of the early SAE analysis with corrected parsing | — |
| `experiment_pathway_token_analysis/` | Earlier version of the pathway/token analysis | — |
| `figures/` | Feature-activation distribution figures from the early SAE work | — |
| `investment_choice_bet_constraint_cot/` | CoT variant of the investment-choice API runner, 29 of 32 cells collected (not used). Its `analysis/create_choice_distribution_cot.py` is the ancestor of the appendix CoT-distribution figure and reads the extended-CoT results of 02 | `investment_choice/bet_constraint_cot/` |
| `investment_choice_experiment/` | Older copy of the first investment-choice runner | — |
| `steering_vector_experiment/` | Earlier contrastive steering-vector pipeline (steering claims were removed from §4) | — |

## `exploratory_experiments/`

| Folder | What it is | HF folder |
|---|---|---|
| `README.md` | The old index of this folder | — |
| `alternative_paradigms/` | Other gambling tasks on the shared harness: blackjack, coin flip, card flip, dice rolling, a stock-trading plan, plus their launchers and logs. The investment-choice and mystery-wheel runners and `common/` moved to `experiments/` | `coin_flip/`, `card_flip/` |
| `additional_experiments/` | Post-submission SAE analyses (condition comparison, cross-domain SAE comparison, investment-choice SAE) | — |
| `gemma_sae_experiment/` | Gemma-2-9B version of the early SAE pipeline | — |
| `lr_classification_experiment/` | Logistic-regression bankruptcy prediction from hidden states | — |
| `steering_vector_analysis/` | CAA steering-vector pipeline; its `data/results` link points at the run's output | `analysis/steering_vector_experiment/` |

## `analysis/` and `scripts/`

| Folder | What it is |
|---|---|
| `analysis/` | Monitoring, model-loading tests and early analysis scripts for the open-weight runs |
| `scripts/` | Shell launchers for the open-weight runs (blackjack, investment choice, Gemma/LLaMA full mode), with SLURM directives disabled |
