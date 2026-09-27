# Table — `tab:companion-metrics`

**Paper location**
Appendix \subsection{Participation and cumulative exposure}
(`\label{appendix:participation-and-exposure}`), `neurips_content_en/appendix.tex:1343`.

**What the experiment asks**
Next to every headline bankruptcy rate, how often did the model actually play, how much did it
stake when it did, and did it wager again after its first loss?

**HF path(s) of the raw data**
The primary slot-machine corpora, canonical six-model roster — the same files behind
Figure 2(a), each identified by the `model` field inside the file rather than by its folder:
  - GPT-4o-mini      `analysis/gpt_results_fixed_parsing/gpt_fixed_parsing_complete_20250919_151240.json`  (`model` = `gpt-4o-mini-corrected`)
  - GPT-4.1-mini     `slot_machine/gpt/gpt5_experiment_20250921_174509.json`  (`model` = `gpt-4.1-mini`)
  - Gemini-2.5-Flash `slot_machine/gemini/gemini_experiment_20250920_042809.json`
  - Claude-3.5-Haiku `slot_machine/claude/claude_experiment_corrected_20250925.json`
  - LLaMA-3.1-8B     `behavioral/slot_machine/llama_v4_role/final_llama_20260315_062428.json`
  - Gemma-2-9B       `behavioral/slot_machine/gemma_v4_role/final_gemma_20260227_002507.json`

**Code that turns raw data into the printed values**
`paper_data/tables/appendix/code/gbsa_companion_metrics.py`
-> `paper_data/tables/appendix/gbsa_companion_metrics.tex` and `gbsa_companion_metrics.json`
(repo-only; not in the HF camera-ready upload). Bankruptcy, participation and re-betting carry
95% Wilson intervals; the mean wager carries a 95% bootstrap interval resampling whole games.

**Corpus vintage**
Canonical. This is the float where the folder-name trap matters most and the generator handles it
correctly: it takes GPT-4o-mini from `analysis/gpt_results_fixed_parsing/` and lists
`slot_machine/gpt/` separately as GPT-4.1-mini, and it takes the open-weight rows from the
V4role corpora. Its own header states that `slot_machine/{gemma,llama}/` carry
`DEPRECATION_WARNING.md` and are not used. Contrast `tab:information-not-bottleneck`, which was
built from exactly those deprecated paths before it was corrected.

The fragment carries its own cross-check against a second artefact: "Cross-check against
`paper_data/figure_data.json`: bankruptcy k and participation k agree in all 12 cells: True", so
this table and Figure 2(a) are pinned to the same counts.

**Reproduction status**
VERIFIED, 2026-08-27. All 12 rows x 4 measures, 60 numeric cells counting interval bounds and the
bracketed at-risk denominators, match `paper_data/tables/appendix/gbsa_companion_metrics.tex`
exactly; **max deviation 0.0**. The narrative figures in the paragraph above the table also
reproduce from the fragment: participation 62.2% to 99.4% across the six models in the fixed arm,
mean realised wager \$10.8 to \$44.8, and Gemini's 68.8% versus 91.6% re-bet asymmetry.

One wording difference, not a numeric one: the fragment labels the arms "forced" and "choosing",
the paper prints "Fixed" and "Variable". The underlying cells are identical.
