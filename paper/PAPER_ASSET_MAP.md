# Where every figure and table in the paper lives

Each row goes from something printed in the paper to the file on the HF dataset that produced it, and
to the code that drew it. Dataset paths are relative to the release root.

> **Provenance beyond the file path lives in [`paper_index/`](paper_index/README.md).** This map
> answers *which file*. It does not answer *what shape that file has*, *which rows were read*, or
> *whether the printed values have ever been reproduced from it*. `paper_index/` carries one manifest
> per float with those fields, including a **schema map** and a runnable **load invariant** for every
> float whose values depend on round-level outcomes. The manifest is the authority on reproduction
> status; this map is the authority on paths.

**The roster below was verified against `neurips_en.aux`**, not against memory. The body prints
**four figures and one table**; every other float is in the appendix. Sixteen figure floats carry
seventeen graphics, because `fig:cross-context-write` is a two-panel float.

## Body figures

| # | Label | Graphic the paper includes | Generator | Plotted values |
|---|---|---|---|---|
| **Figure 1** | `fig:experimental-overview` | `images/representative_flow_diagram.pdf` | `scripts/figures/fig01_overview_flow.py` (also `build_overview_figure.py`) | schematic, no data series |
| **Figure 2** | `fig:slot-machine` | `images/fig02_slot_machine.pdf` | `scripts/figures/fig02_slot_machine.py` | `paper_data/fig02_slot_machine.json`, recomputed by `scripts/build_figure_data.py` |
| **Figure 3** | `fig:investment-choice` | `images/investment_choice3.pdf` | `scripts/figures/fig03_investment_choice_1x4.py` (panels a–c from `fig03_investment_choice.py`, panel d the matched-cap ablation from `fig05_matched_cap.py`) | `paper_data/fig03_investment_choice.json`, `paper_data/fig05_matched_cap.json` |
| **Figure 4** | `fig:causal-battery` | `images/fig04_causal_battery.pdf` | `scripts/figures/fig04_causal_battery.py` | `paper_data/fig04_causal_battery.json` |

Three corrections this row set carries against the previous version of this file, each checked in
`neurips_en.aux`:

- The previous map listed **five** body figures. There are four.
- It gave Figure 3 as `fig03_investment_choice.pdf`. The paper includes `investment_choice3.pdf`,
  a four-panel float whose panel (d) is the matched-cap ablation the old map listed as a separate
  Figure 4.
- It numbered the causal battery Figure 5. It is Figure 4.

`fig05_matched_cap.pdf` and `fig2_combined.pdf` are **not included by the paper**. They are earlier
assets; `fig05_matched_cap.py` still runs, as the source of Figure 3(d).

## Appendix figures

| # | Label | Graphic | Generator | Manifest |
|---|---|---|---|---|
| 5 | `fig:appendix-component-effects` | `component_effects_by_bettype2.pdf` | `scripts/figures/recovered/create_component_effects_by_bettype.py` (ancestor); redraw `scripts/figures/appendix_behavioral_panels.py` | [manifest](paper_index/appendix/fig_appendix_component_effects.md) |
| 6 | `fig:appendix-complexity` | `4model_complexity_trend_average3.pdf` | `scripts/figures/recovered/create_4model_complexity_trend_average.py` (ancestor); redraw as above | [manifest](paper_index/appendix/fig_appendix_complexity.md) |
| 7 | `fig:escalation` | `escalation_trajectory.pdf` | `scripts/figures/figA03_escalation_trajectory.py` | [manifest](paper_index/appendix/fig_escalation_trajectory.md) |
| 8 | `fig:temperature-robustness` | `temperature_robustness.pdf` | `scripts/figures/figA04_temperature_robustness.py` | [manifest](paper_index/appendix/fig_temperature_robustness.md) |
| 9 | `fig:appendix-distortion-summary` | `distortion_multimodel_summary.pdf` | `scripts/figures/figA05_distortion_multimodel_summary.py` | [manifest](paper_index/appendix/fig_appendix_distortion_summary.md) |
| 10 | `fig:appendix-model-components` | `component_effects_all_models_3x4_2.pdf` | `scripts/figures/recovered/create_individual_model_component_effects.py` (ancestor); redraw as above | [manifest](paper_index/appendix/fig_appendix_model_components.md) |
| 11 | `fig:appendix-model-complexity` | `4model_complexity_trend_3x4.pdf` | `scripts/figures/recovered/create_4model_complexity_trend_3x4.py` | [manifest](paper_index/appendix/fig_appendix_model_complexity.md) |
| 12 | `fig:appendix-model-streak` | `individual_model_streak_analysis.pdf` | `scripts/figures/recovered/create_individual_model_streak_analysis.py` | [manifest](paper_index/appendix/fig_appendix_model_streak.md) |
| 13 | `fig:appendix-choice-distribution` | `investment_choice_distributions_cot.pdf` | `scripts/figures/figA09_investment_choice_distributions_cot.py`; ancestor `scripts/figures/recovered/create_choice_distribution_cot.py` | [manifest](paper_index/appendix/fig_appendix_choice_distribution.md) |
| 14 | `fig:sharing` | `fig5b_pca_appendix.pdf` | `scripts/figures/fig5b_pca_appendix.py` | [manifest](paper_index/ch4_neural/fig_sharing_loto_pca.md) |
| 15 | `fig:sign-transfer` | `fig_axis_alignment.pdf` | `scripts/figures/fig_cross_context_write.py` | [manifest](paper_index/ch4_causal/fig_sign_transfer.md) |
| 16 | `fig:cross-context-write` | `fig_xctx_signmap.pdf` + `fig_xctx_ladders.pdf` | `scripts/figures/fig_cross_context_write.py` | [manifest](paper_index/ch4_causal/fig_cross_context_write.md) |

**The `Generator | n/a` rows are gone.** The previous version of this file recorded `n/a` for all
twelve appendix figures. Six of those generators were not missing but deleted: they lived in the code
repo under `legacy/writing/table_figure/`, which commit `16acccf` removed wholesale. Verbatim copies
are now at `scripts/figures/recovered/`, whose `README.md` records for each one the corpora it opened,
the subset it took and the scan rule it used. Read those `n/a` entries as a warning about this map's
scope: a repo-side asset map is not an inventory of what exists.

## Tables

The body prints one table, `tab:neurips-sae-results` (Table 1). Tables 2 to 33 are appendix tables.
Every one is written inline in the `.tex` rather than `\input` from a fragment, and the values are
recomputed from the release by the scripts under `scripts/tables/` and `sae_v3_analysis/`.

Per-table provenance — canonical HF path, generator, corpus vintage and reproduction status — is one
manifest per table under [`paper_index/`](paper_index/README.md), indexed in its `README.md`. That is
the authority; the flat list this file used to carry said only "written inline in the appendix" for
all 33 rows and distinguished nothing.

Fragments that exist in `paper_data/tables/` but are `\input` by no `.tex` are listed in
[`paper_index/README.md`](paper_index/README.md) under *Fragments that exist but are wired into
nothing*, so that finished work is not mistaken for a printed float, or overlooked.
