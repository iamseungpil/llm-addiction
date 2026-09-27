# Figure — `fig:appendix-model-complexity`

**Paper location**
Appendix \section{Model-by-model details}, `neurips_content_en/appendix.tex:498-500`; graphic `images/4model_complexity_trend_3x4.pdf`

**What the experiment asks**
Per API model, does stacking more instruction sentences into the prompt track higher risk, and how tight is that relationship?

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
git -C <llm-addiction> show 16acccf^:legacy/writing/table_figure/create_4model_complexity_trend_3x4.py
```

A verbatim copy is also checked into this repo at `scripts/figures/recovered/create_4model_complexity_trend_3x4.py`, beside a `README.md` recording the corpora, subset and scan rule of all seven recovered
scripts. Cite the copy; the git command above is how to re-derive it.

It writes `4model_complexity_trend_3x4.pdf` — the exact filename the paper includes, with no
revision suffix. This is a direct generator for the printed asset.

**The subset:**
  - **Arm:** both, pooled. No `bet_type` filter.
  - **Complexity:** as in `fig_appendix_complexity.md` — `0` for `BASE`, else the count of module
    letters in `prompt_combo`, tested against `['G','M','P','R','W']`.
  - **Panels:** three metrics by four models; unlike the averaged sibling there is no cross-model
    mean, each panel is one model's own games.
  - **The printed r:** `scipy.stats.pearsonr` per panel over that model's complexity-level means, and
    the value is stamped on the panel as `r = {value:.3f}`.

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
**NOT CHECKED — reproducible.** Upgraded from UNREPRODUCIBLE. The generator is recovered, it writes
the printed filename, and all four canonical corpora it names are in the public release, so nothing
blocks a recompute. None was run this pass, and the honest verdict for a recompute that was not run
is NOT CHECKED.

Running it requires the load invariant in `ch3_behaviour/fig02_slot_machine.md` first. That invariant
is passing on all six corpora; a manifest may not claim VERIFIED while its invariant is unrun.
