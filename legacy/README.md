# `legacy/` in this repository does NOT mean deprecated

Read this before deciding anything about a file under here.

The name is historical. These directories were called `legacy/` because they sit
outside the unified `paper_experiments/` tree, not because their contents are wrong.
**Several of them still hold paper-canonical code.** The authority is
[`../PAPER_CANONICAL_CODE.md`](../PAPER_CANONICAL_CODE.md), which exists specifically
to give the figure → code → data map; that file decides, not this directory name.

Nothing is ever deleted here. Files that turn out to be wrong are kept and labelled.

## Do not confuse this with `legacy/` on the released dataset

The HF dataset `llm-addiction-research/llm-addiction` also has a top-level `legacy/`,
and **there the word does mean retired and do-not-cite**: `v12_steering_invalidated/`,
`v14_steering/`, `v16_steering/`, `v17_leaky_pipeline/`, `pre_groupkfold_sweep/`. The
dataset card lists them in its historical table. The
label files for that tree are staged in this repository at
[`../sae_v3_analysis/release_labels/legacy/`](../sae_v3_analysis/release_labels/legacy/).

Two directories, one word, opposite meanings. The code repository's `legacy/` is the
permissive one.

## Status of what is here

| Path | Status |
|---|---|
| `gpt_experiments/`, `claude_experiment/`, `gemini_experiment/`, `gpt5_experiment/` | **Paper-canonical.** Older copies of the API slot-machine runners that `paper_experiments/slot_machine_6models/src/run_*_experiment.py` later canonicalised. Prefer the `paper_experiments/` copies for new work; these produced the released exports. |
| `gpt_fixed_bet_size_experiment/`, `gpt_variable_max_bet_experiment/` | **Superseded in place.** The two Figure 3d cap-ablation runners moved to `paper_experiments/sm_cap_ablation/src/` on 2026-05-08 for findability. What remains here is the surrounding analysis and figure code plus the per-cap restart/split variants of the variable runner. |
| `investment_choice_bet_constraint_cot/analysis/create_choice_distribution_cot.py` | **Paper-canonical, and misfiled.** It is the ancestor generator of appendix Figure 13. Copied, with its provenance written out, to the paper repo at `scripts/figures/recovered/`. |
| `investment_choice_experiment/`, `investment_choice_extended_cot/`, `investment_choice_bet_constraint/` | Older investment-choice trees; the canonical runners are under `paper_experiments/`. |
| `analysis/` | Mixed. `analyze_all_6_models.py` is **DEPRECATED, do not cite** — see the banner at the top of that file. The rest is exploratory. |
| `writing/table_figure/` | **Deleted by commit `16acccf` and not restored here.** It held the generators for six appendix figures. The ones the camera-ready depends on were recovered from history into the paper repo at `scripts/figures/recovered/`, with a README naming which printed figure each one produced. Recover any of the other 16 with `git show 16acccf^:legacy/writing/table_figure/<name>.py`. |
| `causal_feature_discovery/`, `experiment_2_multilayer_patching_L1_31/`, `experiment_corrected_sae_analysis/`, `steering_vector_experiment/`, `experiment_0_llama_gemma_restart/`, `experiment_pathway_token_analysis/`, `figures/` | Exploratory or superseded. The paper's neural pipeline is `sae_v3_analysis/`; all steering claims were removed from §4. |

## If you are about to cite something from here

1. Check `../PAPER_CANONICAL_CODE.md` for the figure-to-code map.
2. Check the historical table in the Hugging Face dataset card for the do-not-cite list.
3. Where a README and an executable generator disagree, the generator decides.
