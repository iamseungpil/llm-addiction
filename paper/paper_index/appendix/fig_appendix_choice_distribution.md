# Figure — `fig:appendix-choice-distribution`

**Paper location**
Appendix \subsection{Model-by-model details}, `neurips_content_en/appendix.tex:516-518`;
graphic `images/investment_choice_distributions_cot.pdf`

**What the experiment asks**
In the investment task run with chain-of-thought reasoning, how does the spread of chosen options
shift from the safe end to the high-variance end as the bet cap rises?

**HF path(s) of the raw data**
`investment_choice/extended_cot/` (922 files in the release) and
`investment_choice/bet_constraint_cot/` (640 files), both named by the figA09 README and both
read by the generator.

**Code that turns raw data into the printed values**
Two copies of the same generator exist, and this pass located the second.

Published: HF `paper_neurips_2026/figures/appendix/figA09_ic_distributions_cot/code/create_choice_distribution_cot.py`
— a real generator for this float, not a placeholder: it writes
`investment_choice_distributions_cot.pdf`, the exact filename the paper includes.

Recovered in the code repo (`iamseungpil/llm-addiction`), present in the working tree and needing no
git archaeology, at
`legacy/investment_choice_bet_constraint_cot/analysis/create_choice_distribution_cot.py`, beside its
own `investment_choice_distributions_cot.pdf` and `.png` outputs. Unlike the five behavioural
appendix figures, this one was not deleted by commit `16acccf`. A verbatim copy is checked into
this repo at `scripts/figures/recovered/create_choice_distribution_cot.py`. That a paper-canonical
generator sits under `legacy/` is itself worth recording: `legacy/` is this project's do-not-cite
marker everywhere else.

**The subset**, read off the generator:
  - **Corpus, despite the directory:** the script lives under
    `legacy/investment_choice_bet_constraint_cot/` but its `RESULTS_DIR` points at the **extended
    CoT** results, not the bet-constraint ones. The folder name is not the corpus.
  - **Deduplication:** results files are sorted in reverse and the **first** file seen per
    `(model, bet_type, bet_constraint)` is kept, the rest dropped. Loading the directory without that
    de-duplication double-counts every re-run combination.
  - **Aggregation key:** `(model, bet_type)`, with `bet_constraint` collapsed — the caps are pooled,
    not held as a variable, inside each panel.
  - **Arm:** both, as the figure's two rows — fixed on top, variable below.
  - **Conditions:** exactly `['BASE', 'G', 'M', 'GM']`, four stacked bars per panel.
  - **The plotted quantity:** `decision['choice']` counted over options 1 to 4 and divided by the
    number of decisions in that cell, times 100 — a share of decisions, not of games, which is why
    every stack sums to 100.
  - **Roster:** `gpt4o_mini`, `gpt41_mini`, `claude_haiku`, `gemini_flash`. Note the display name the
    script assigns the fourth is **Gemini-2.0-Flash**, not the Gemini-2.5-Flash the slot-machine
    figures use. That is a roster fact about the CoT investment corpora, not a typo to correct here.

Repo redraw at print size: `scripts/figures/figA09_investment_choice_distributions_cot.py` with
sidecar `scripts/figures/data/figA09_investment_choice_distributions_cot.json`.

**Corpus vintage**
Canonical as far as the deprecation markers reach: the CoT investment-choice corpora are their own
tree, and the three `DEPRECATION_WARNING.md` files in the release
(`sae_patching/`, `slot_machine/gemma/`, `slot_machine/llama/`) do not touch it. The
`legacy/` do-not-cite paths recorded in `NEURIPS_CANONICAL_INDEX.md` §5 are neural-pipeline
directories and are likewise unrelated.

This manifest previously carried the "four API models / GPT-5-mini substitution" vintage warning.
That paragraph was copied from the three sibling model-by-model figures and does not apply here:
this float is investment choice under chain-of-thought, not a four-API-model slot-machine panel.

**Reproduction status**
**NOT CHECKED — reproducible, generator located in two places.** Both input corpora are in the
release: `investment_choice/extended_cot/` (922 files) and `investment_choice/bet_constraint_cot/`
(640 files), counts re-confirmed this pass against the full listing. Nothing blocks a recompute; none
was run for this index, and the honest verdict for a recompute that was not run is NOT CHECKED.

The repo sidecar records the 32 stacked distributions recovered from the submitted PDF's rectangle
geometry, every stack summing to 100.000%, so the redraw carries the submitted numbers even though
they have not been re-derived from the CoT logs.

This float reads investment-choice CoT corpora, not slot machine, so the slot-machine load invariant
does not apply to it. It has no equivalent single-constant invariant: the CoT task has no fixed win
rate. Its load check is the de-duplication rule above — a correct load keeps one file per
`(model, bet_type, bet_constraint)` and yields four conditions in every one of the eight panels; a
load that returns duplicated combinations or fewer than four conditions has read the directory wrong.

Composition note carried over: the stacks are labelled "Option 1-4" while Figure 3(b) labels the
same four colours "Safe exit / Low var. / Mid var. / High var.".
