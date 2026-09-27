# Figure — `fig:appendix-complexity`

**Paper location**
Appendix \subsection{Effects of prompt components, complexity, and autonomy}, `neurips_content_en/appendix.tex:223-225`; graphic `images/4model_complexity_trend_average3.pdf`

**What the experiment asks**
As more instruction sentences are stacked into the prompt, does the model go broke more often and bet more in total?

**HF path(s) of the raw data**
Read straight off the recovered generator's loader table (see below), not inferred from a README.
The four API corpora, each named there by absolute path and identified here by its release path:
  - GPT-4o-mini      `analysis/gpt_results_fixed_parsing/gpt_fixed_parsing_complete_20250919_151240.json`
  - GPT-4.1-mini     `slot_machine/gpt/gpt5_experiment_20250921_174509.json`
  - Gemini-2.5-Flash `slot_machine/gemini/gemini_experiment_20250920_042809.json`
  - Claude-3.5-Haiku `slot_machine/claude/claude_experiment_corrected_20250925.json`
The six-model render the paper prints adds the two canonical open-weight corpora:
  - LLaMA-3.1-8B     `behavioral/slot_machine/llama_v4_role/final_llama_20260315_062428.json`
  - Gemma-2-9B       `behavioral/slot_machine/gemma_v4_role/final_gemma_20260227_002507.json`

**Code that turns raw data into the printed values**
**Generator recovered.** This figure was recorded in earlier passes of this index as having no
generator anywhere. That was wrong, and the reason it was wrong is worth stating: the search covered
the paper repo and the HF upload, and the script was in neither. It was in the code repo
(`iamseungpil/llm-addiction`) and had been deleted there. Commit `16acccf` ("clean files") removed
the whole of `legacy/writing/table_figure/`, which is where the behavioural appendix figures were
drawn. Git still has it:

```sh
git -C <llm-addiction> show 16acccf^:legacy/writing/table_figure/create_4model_complexity_trend_average.py
```

A verbatim copy is also checked into this repo at `scripts/figures/recovered/create_4model_complexity_trend_average.py`, beside a `README.md` recording the corpora, subset and scan rule of all seven recovered
scripts. Cite the copy; the git command above is how to re-derive it.

It writes `4model_complexity_trend_average.pdf`. The paper prints
`4model_complexity_trend_average3.pdf`, the six-model successor of the same script.

This also settles the filename-versus-caption mismatch this manifest flagged as a second reason to
distrust the vintage. The `4model` stem is inherited from the four-API-model ancestor recovered here;
the printed `average3` asset is the six-model revision, which is why the caption says "all six open
and closed models" while the filename still says four. The stem is stale, the caption is right, and
the disagreement is a naming artefact rather than a corpus one.

**The subset:**
  - **Arm:** both, pooled. No `bet_type` filter anywhere in this script.
  - **Complexity:** `0` for `BASE`, otherwise the count of module letters present in `prompt_combo`.
  - **Averaging:** two-stage and unweighted, as above — per model per complexity level first, then a
    plain mean across models at each level, giving six plotted points (complexity 0 to 5) per panel.
  - **The printed r:** `scipy.stats.pearsonr` over those six averaged points, one per panel. It is a
    correlation across complexity levels of model-averaged values, not across games, which is why n
    is 6 and not the game count.
  - **Prompt alphabet:** the complexity counter tests membership in `['G','M','P','R','W']`. A LLaMA
    V4role combo spelled `GMHW` scores 3 under that list instead of 4. Any six-model extension has to
    make the letter set model-dependent or it silently mis-bins every LLaMA condition carrying the
    fifth module.

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
**REPRODUCES**, six-model. All three printed r values come out exact. Recomputed this pass from the
released corpora through the recovered generator's definitions.

The load invariant for the six slot-machine corpora is recorded in
`ch3_behaviour/fig02_slot_machine.md` and is passing on all six. This float's verdict rests on it.
