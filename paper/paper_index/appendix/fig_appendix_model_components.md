# Figure — `fig:appendix-model-components`

**Paper location**
Appendix \section{Model-by-model details}, `neurips_content_en/appendix.tex:489-491`; graphic `images/component_effects_all_models_3x4_2.pdf`

**What the experiment asks**
The same prompt-sentence breakdown as before, but one panel per API model, to show that the average hides different routes to the same risky end.

**HF path(s) of the raw data**
Read straight off the recovered generator's loader table (see below), not inferred from a README.
The four API corpora, each named there by absolute path and identified here by its release path:
  - GPT-4o-mini      `analysis/gpt_results_fixed_parsing/gpt_fixed_parsing_complete_20250919_151240.json`
  - GPT-4.1-mini     `slot_machine/gpt/gpt5_experiment_20250921_174509.json`
  - Gemini-2.5-Flash `slot_machine/gemini/gemini_experiment_20250920_042809.json`
  - Claude-3.5-Haiku `slot_machine/claude/claude_experiment_corrected_20250925.json`

**Code that turns raw data into the printed values**
**Generator recovered.** This figure was recorded in earlier passes of this index as having no
generator anywhere. That was wrong, and the reason it was wrong is worth stating: the search covered
the paper repo and the HF upload, and the script was in neither. It was in the code repo
(`iamseungpil/llm-addiction`) and had been deleted there. Commit `16acccf` ("clean files") removed
the whole of `legacy/writing/table_figure/`, which is where the behavioural appendix figures were
drawn. Git still has it:

```sh
git -C <llm-addiction> show 16acccf^:legacy/writing/table_figure/create_individual_model_component_effects.py
```

A verbatim copy is also checked into this repo at `scripts/figures/recovered/create_individual_model_component_effects.py`, beside a `README.md` recording the corpora, subset and scan rule of all seven recovered
scripts. Cite the copy; the git command above is how to re-derive it.

It writes `component_effects_all_models_3x4.pdf`. The paper prints
`component_effects_all_models_3x4_2.pdf`, a later render of the same code path.

**The subset:** the same with-minus-without contrast as
`fig_appendix_component_effects.md` — both arms drawn as paired bars, effects computed per
`(model, bet_type, component)` — but held at the per-model level instead of averaged across models,
one panel per API model.

The HF `figA0N/code/generate_behavioral_figures.py` attachment remains what the previous pass found
it to be: one script copied under five figure directories, drawing none of them. It is a mislabelled
attachment, not the generator, and the recovery above does not change that reading.
Repo redraw at print size: `scripts/figures/appendix_behavioral_panels.py`, which recovers the
plotted series from the submitted vector PDF's bar and marker geometry rather than from raw logs.

**Corpus vintage**
**Canonical — resolved by the recovered generator, no longer at risk.** The loader table is a
hard-coded dictionary of absolute paths, so the roster is not inferred, it is read:

  - GPT-4o-mini      `gpt_results_fixed_parsing/gpt_fixed_parsing_complete_20250919_151240.json` — the canonical file, under its own identity
  - GPT-4.1-mini     `gpt5_experiment_20250921_174509.json` — named as GPT-4.1-mini, not as GPT-4o-mini
  - Gemini-2.5-Flash `gemini_experiment_20250920_042809.json`
  - Claude-3.5-Haiku `claude_experiment_corrected_20250925.json`

Two open questions this closes. The abandoned GPT-5-mini run
(`slot_machine/gpt/archived_gpt5mini_20250921/`, stopped at 9.4% after a 60% API error rate) is
never opened by any of these scripts, so the "four API models under a stale roster" worry is ruled
out rather than merely unproven. And neither `slot_machine/gemma/` nor `slot_machine/llama/` — the
two paths carrying `DEPRECATION_WARNING.md` — appears anywhere in the loader; the six-model
successor adds the V4role corpora, not the V1 ones.

**Reproduction status**
**NOT CHECKED — reproducible.** Upgraded from UNREPRODUCIBLE. The generator is recovered and all four
canonical corpora it names are in the public release. No recompute was run this pass.

Running it requires the load invariant in `ch3_behaviour/fig02_slot_machine.md` first. That invariant
is passing on all six corpora; a manifest may not claim VERIFIED while its invariant is unrun.
