# Table — `tab:worked-example-intervals`

**Paper location**
Appendix \subsection{Added controls}, worked-example block, `neurips_content_en/appendix.tex:1226`
(float opens at `:1215`; the label sits at `:1226`).

**What the experiment asks**
If you first show the model a short transcript of somebody betting recklessly rather than
carefully, does it go broke more often afterwards?

**HF path(s) of the raw data**
`rebuttal_neurips_2026/in_context_demo_api/` — 16 cells, 100 games each; the six pairs printed
in the paper are GPT-4o-mini, GPT-4.1-mini and Gemini-2.5-Flash, each fixed and variable, persona
present, `rat0`. The two Claude-Haiku-4.5 pairs are read, marked `[collected, NOT printed]` in the
fragment's provenance block, and excluded from the table and from the multiplicity family.
Also read by the generator but **not printed**: `rebuttal_neurips_2026/in_context_demo_open_weight/`
(8 cells, 200 games), `rebuttal_neurips_2026/in_context_demo_open_weight_persona/` (2 of 8 planned
cells), and `rebuttal_neurips_2026/framing_rationality_factorial_e7/` for the `role_rat0`
no-example baseline.

**Code that turns raw data into the printed values**
`paper_data/tables/appendix/code/worked_example_intervals.py`
-> `paper_data/tables/appendix/worked_example_intervals.tex` and `worked_example_intervals.json`
(repo-only; not in the HF camera-ready upload). Intervals are Newcombe's hybrid-score method on
two Wilson score intervals.

**Corpus vintage**
Rebuttal-era, canonical. No `DEPRECATION_WARNING.md` exists anywhere under
`rebuttal_neurips_2026/` — the release carries that file at exactly three paths
(`sae_patching/`, `slot_machine/gemma/`, `slot_machine/llama/`), verified 2026-08-27 against the
full 9,449-file HF listing, and none of them is read here. Model identity was taken from the
`model` field inside each JSON, not from the path. The generator explicitly excludes
`framing_rationality_factorial_e7/QUARANTINE_truncated_claude/`, so the Claude cells are the
re-collected Claude-Haiku-4.5 runs rather than the truncated earlier ones.

One scoping difference the manifest records because the printed table does not: the fragment
carries 13 rows and the paper prints 8. The five omitted rows are the open-weight cells, which
have no matched no-example arm, and the caption's phrase "all eight API model-condition pairs"
is accurate for what is printed. The omission is a scoping choice, not a data defect.


**Printed scope (camera-ready, 2026-09-08)**
The paper prints six pairs, not eight: the Claude-3.5-Haiku checkpoint had been withdrawn from
the API and the substitute Claude-Haiku-4.5 pairs (both 0.0 -> 0.0) are released but not printed.
The Bonferroni-adjusted interval on the one significant pair is recomputed across six comparisons,
[+13.1, +46.3] at z = 2.6383, replacing [+12.4, +46.8] across eight. Two printed pairs sit at 0%
in both arms, not four.


**Regenerated 2026-09-08** from the three `in_context_demo_*` corpora. Before the model change the
generator reproduced its committed output byte for byte apart from the timestamp; after it, all six
printed rows and the Bonferroni interval on Gemini variable, [+13.1, +46.3] at z = 2.63829 for
six comparisons, match `neurips_content_en/appendix.tex` exactly.

**Reproduction status**
VERIFIED, 2026-08-27. All eight printed rows — cautious rate, escalating rate, difference and
95% interval, 40 numeric cells plus the eight $n$/arm — match
`paper_data/tables/appendix/worked_example_intervals.tex` exactly; **max deviation 0.0 percentage
points**. The caption's Bonferroni interval for the Gemini variable pair, $[+12.4, +46.8]$, also
matches the fragment's caption.
