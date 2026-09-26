# Path map: old layout → new layout

The repository was reorganised into one folder per experiment (`experiments/NN_name/`). Everything
was moved with `git mv`, so `git log --follow <new path>` shows each file's full history. The git
tag **`pre-reorg`** marks the last commit with the old layout (`36fe502`); check it out to run a
script exactly as it was, or to follow an old path written in a plan, note or HF manifest.

One row per moved unit. A trailing `/` means the folder moved with everything in it.

## Paper code → `experiments/`

| Old path | New path |
|---|---|
| `paper_experiments/slot_machine_6models/` | `experiments/01_slot_machine/` |
| `legacy/gpt_experiments/` | `experiments/01_slot_machine/original_api_runners/gpt_experiments/` |
| `legacy/claude_experiment/` | `experiments/01_slot_machine/original_api_runners/claude_experiment/` |
| `legacy/gemini_experiment/` | `experiments/01_slot_machine/original_api_runners/gemini_experiment/` |
| `legacy/gpt5_experiment/` | `experiments/01_slot_machine/original_api_runners/gpt5_experiment/` |
| `legacy/experiment_0_llama_gemma_restart/` | `experiments/01_slot_machine/experiment_0_llama_gemma_restart/` |
| `legacy/investment_choice_bet_constraint/` | `experiments/02_investment_choice/api_bet_constraint/` |
| `exploratory_experiments/alternative_paradigms/src/investment_choice/` | `experiments/02_investment_choice/open_weight/investment_choice/` |
| `paper_experiments/investment_choice_extended_cot/` | `experiments/02_investment_choice/appendix_extended_cot/` |
| `legacy/investment_choice_extended_cot/` | `experiments/02_investment_choice/appendix_extended_cot_legacy_copy/` |
| `paper_experiments/sm_cap_ablation/` | `experiments/03_matched_cap/sm_cap_ablation/` |
| `legacy/gpt_variable_max_bet_experiment/` | `experiments/03_matched_cap/gpt_variable_max_bet_experiment/` |
| `legacy/gpt_fixed_bet_size_experiment/` | `experiments/03_matched_cap/gpt_fixed_bet_size_experiment/` |
| `paper_experiments/track0_w3_replication/` | `experiments/03_matched_cap/track0_w3_replication/` |
| `paper_experiments/e8_constraint_choice/` | `experiments/04_choice_ladder/` |
| `paper_experiments/e7_factorial/` | `experiments/05_framing_worked_example/` |
| `exploratory_experiments/alternative_paradigms/src/mystery_wheel/` | `experiments/06_mystery_wheel/mystery_wheel/` |
| `run_mw_vllm.py`, `run_mw_parallel.py` | `experiments/06_mystery_wheel/run_mw_vllm.py`, `experiments/06_mystery_wheel/run_mw_parallel.py` |
| `sae_v3_analysis/` | `experiments/07_sae_readout/` |
| `multilayer_causal/` | `experiments/08_steering/multilayer_causal/` |
| `docs/` (the `multilayer_causal` design specs: `docs/README.md`, `docs/superpowers/`) | `experiments/08_steering/docs/` |
| `paper_experiments/e2_coding/` | `experiments/09_audits/` |
| `exploratory_experiments/alternative_paradigms/src/common/` | `experiments/shared/common/` |

## Code no paper figure or table uses → `archive/`

The internal structure is kept, so `X` became `archive/X`.

| Old path | New path |
|---|---|
| `paper_experiments/` (the rest: `README.md`, `investment_choice_experiment/`, `m1_portfolio_discriminant/`, `m2_persona_decoupling/`, `m5_compliance_residualisation/`, `d_distributed_effect/`, `track_L_length_confound/`, `llama_sae_analysis/`, `pathway_token_analysis/`) | `archive/paper_experiments/` |
| `legacy/` (the rest: `README.md`, `analysis/`, `causal_feature_discovery/`, `experiment_2_multilayer_patching_L1_31/`, `experiment_corrected_sae_analysis/`, `experiment_pathway_token_analysis/`, `figures/`, `investment_choice_bet_constraint_cot/`, `investment_choice_experiment/`, `steering_vector_experiment/`) | `archive/legacy/` |
| `exploratory_experiments/` (the rest, including the other alternative paradigms) | `archive/exploratory_experiments/` |
| `analysis/` | `archive/analysis/` |
| `scripts/` | `archive/scripts/` |

## Records → `docs/`

| Old path | New path |
|---|---|
| `rebuttal_review/` (holds `CAMERA_READY_MAP.md`) | `docs/rebuttal_review/` |
| `plans/` | `docs/plans/` |
| `drafts/` | `docs/drafts/` |
| `amlt/` | `docs/amlt/` |
| `PLAN_4NODE_EXECUTION_2026_05_07.md`, `PLAN_TRACK0_W3_v5.md`, `PLAN_TRACK_L_LENGTH_CONFOUND_v1.md` | `docs/` (same names) |
| `EXPERIMENT_DESIGN_COMPARISON.md`, `STRUCTURE.md`, `MANIFEST.md`, `SLURM_GUIDE.md`, `ev_transparency_gambling_avoidance.md`, `token_truncation_root_cause_analysis.md`, `token_truncation_root_cause_analysis.pdf` | `docs/` (same names) |
| `investment_choice_bet_constraint_cot/code_review_report.pdf` | `docs/investment_choice_bet_constraint_cot/code_review_report.pdf` |
| `PAPER_CANONICAL_CODE.md` | `docs/PAPER_CANONICAL_CODE.md` (superseded by the README and this file) |

`site/`, `.github/`, `README.md` and `CLAUDE.md` stayed where they were.

## What changed inside files

- **Imports and path roots.** Every `sys.path` entry, `Path(__file__).parents[k]` and shell
  `PYTHONPATH` that pointed at a moved folder now points at the new one. The alternative-paradigm
  harness used to be one import root (`exploratory_experiments/alternative_paradigms/src`); it is
  now three: `experiments/shared` (`common`), `experiments/02_investment_choice/open_weight`
  (`investment_choice`) and `experiments/06_mystery_wheel` (`mystery_wheel`). The two runners add
  `experiments/shared` to `sys.path` themselves.
- **`multilayer_causal` is still a Python package** imported as `multilayer_causal.src...`, so it
  keeps its name inside `experiments/08_steering/`. Run its commands from `experiments/08_steering/`.
- **Archived runners that `import common`** reach it through
  `archive/exploratory_experiments/alternative_paradigms/src/common/__init__.py`, which forwards to
  `experiments/shared/common/`.
- **Absolute paths to a checkout of this repository** on the original machines
  (`/home/v-seungplee/llm-addiction/...`, `/home/jovyan/llm-addiction/...`,
  `/scratch/llm_addiction/...`, `/scratch/llm-addiction/...`, `/scratch/code/llm-addiction/...`)
  were rewritten to the new layout, since those checkouts are clones of this repository. A few
  that decide imports or outputs now derive from `__file__` instead. Data paths
  (`/data/llm_addiction/...`, `/home/v-seungplee/data/...`, `/scratch/x3415a02/data/...`) are
  unchanged. The cluster job files were not re-submitted after the rewrite.
- **Hugging Face paths are unchanged.** The dataset still uses `sae_v3_analysis/...`, `legacy/...`
  and `experiments/sec4_causal/...`; those strings are dataset folders, not repository folders.
- **Records keep their old wording.** Plans, review notes and reports under `docs/` and `archive/`
  describe the repository as it was when they were written; read old paths in them through this
  table. Relative links in them were updated so they still resolve.
