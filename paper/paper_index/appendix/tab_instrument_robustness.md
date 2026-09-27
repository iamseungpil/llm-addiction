# Table — `tab:instrument-robustness`

**Paper location**
Appendix \subsection{Supplementary evidence for gambling-related language markers}, `neurips_content_en/appendix.tex:393`

**What the experiment asks**
If you swap the word-list for two other ones, does the goal prompt still look like it makes the models talk more like gamblers?

**HF path(s) of the raw data**
`rebuttal_neurips_2026/nested_baseline_and_audits_e2/multi_instrument_results.json` (plus `multi_instrument_results_FULL.json`, `multi_instrument_ABLATED.json`, `multi_instrument_full.log`).

**Code that turns raw data into the printed values**
HF `rebuttal_neurips_2026/nested_baseline_and_audits_e2/src/multi_instrument_robustness.py`. No regeneration fragment in `paper_data/tables/`.

**Corpus vintage**
Rebuttal-era corpus (`rebuttal_neurips_2026/`). No `DEPRECATION_WARNING.md` exists anywhere under `rebuttal_neurips_2026/` — the release carries that file only under `sae_patching/`, `slot_machine/gemma/` and `slot_machine/llama/`.

Vintage caveat from `VERIFIED_FACTS.md` §L: 'window scoping restored (supersedes the earlier instrument-battery numbers)'. An earlier instrument-battery result was withdrawn; anyone re-deriving this table must use the post-§L numbers, and `multi_instrument_results.json` must not be confused with the SUPERSEDED files sitting beside it in the same directory.

**Reproduction status**
NOT CHECKED — the released JSON exists but was not compared cell-by-cell against the printed table here.
