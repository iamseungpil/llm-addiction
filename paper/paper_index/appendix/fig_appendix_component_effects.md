# Figure — `fig:appendix-component-effects`

**Paper location**
Appendix \subsection{Effects of prompt components, complexity, and autonomy}, `neurips_content_en/appendix.tex:214-216`; graphic `images/component_effects_by_bettype2.pdf`

**What the experiment asks**
Which of the five optional prompt sentences actually pushed models toward riskier play, split by whether the model could choose its bet size?

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

The figA01 README's claim of `slot_machine/{gemma,llama,...}/final_*.json` — the deprecated V1 tree —
is superseded. It described an earlier variant of the artwork and is contradicted by the executable
generator, which opens none of those paths.

**Code that turns raw data into the printed values**
**Generator recovered.** This figure was recorded in earlier passes of this index as having no
generator anywhere. That was wrong, and the reason it was wrong is worth stating: the search covered
the paper repo and the HF upload, and the script was in neither. It was in the code repo
(`iamseungpil/llm-addiction`) and had been deleted there. Commit `16acccf` ("clean files") removed
the whole of `legacy/writing/table_figure/`, which is where the behavioural appendix figures were
drawn. Git still has it:

```sh
git -C <llm-addiction> show 16acccf^:legacy/writing/table_figure/create_component_effects_by_bettype.py
```

A verbatim copy is also checked into this repo at `scripts/figures/recovered/create_component_effects_by_bettype.py`, beside a `README.md` recording the corpora, subset and scan rule of all seven recovered
scripts. Cite the copy; the git command above is how to re-derive it.

It writes `component_effects_by_bettype.pdf`. The paper prints
`component_effects_by_bettype2.pdf` — the six-model successor of the same script. The trailing digit
is this project's revision marker on appendix artwork and appears on three of the five recovered
figures; it is a later render of the same code path, not a different figure.

**The subset**, which is what no earlier pass recorded:
  - **Arm:** both. `bet_type` is a grouping variable here, not a filter: fixed and variable are drawn
    as paired bars, and the printed contrast between them is the point of the figure.
  - **Effect definition:** for each `(model, bet_type, component)`, the with-minus-without contrast
    on games, `with_comp` = games whose `prompt_combo` contains the component letter, `without_comp`
    = all the rest. Bankruptcy in percentage points, total bet and rounds in raw units.
  - **Averaging:** two-stage and unweighted. Effects are computed per model, then averaged across
    models by `groupby(['bet_type','component']).mean()`. It is a mean of per-model effects, not a
    pooled-games effect, and the two do not agree when model game counts differ.
  - **Prompt alphabet:** the script carries the H/R mapping explicitly —
    `search_component = 'R' if component == 'H' else component` — because all four API corpora spell
    the fifth module `R`. Extending the loader to six models makes that mapping model-dependent: the
    LLaMA V4role corpus spells it `H`, so a flat H-to-R rewrite drops LLaMA's hidden-patterns
    conditions from both sides of the contrast. See the *Schema map* point 4 in the slot-machine
    manifests.

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

This supersedes the "strongest remaining vintage risk" flag this manifest carried. That flag rested
on the figA01 README naming the deprecated V1 tree, and on there being no generator to overrule it.
There is now a generator, and this index's own rule applies: where a README and an executable
generator disagree, the generator decides.

**Reproduction status**
**REPRODUCES**, six-model, exact on the printed values. Recomputed this pass from the released
corpora through the recovered generator's definitions; the printed values come out of the release
with no residual.

The load invariant for the six slot-machine corpora is recorded in
`ch3_behaviour/fig02_slot_machine.md` and is passing on all six. This float's verdict rests on it:
were the invariant unrun, the correct verdict here would be NOT CHECKED regardless of how well the
values agreed.
