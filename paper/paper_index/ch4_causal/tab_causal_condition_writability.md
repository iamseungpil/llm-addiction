# Table — `tab:causal-condition-writability`

**Paper location**
Appendix \subsection{Cross-task transfer and condition writability}, `neurips_content_en/appendix.tex:939`

**What the experiment asks**
Is the 'bet bigger' direction easier to push when the model has been given a goal than when it has been told to maximise reward?

**HF path(s) of the raw data**
`experiments/sec4_causal/checkpoints/sec4_w14/*.jsonl`; analysis `experiments/sec4_causal/analysis/sec4_w14_analysis.json` and `sec4_w14_analysis_robustness_20260710.json`.

**Code that turns raw data into the printed values**
`scripts/tables/appendix_neural_tables.py` -> `paper_data/tables/appendix/A10_condition_writability.tex`, which calls the canonical analyser `multilayer_causal/src/sec4_stats.analyze_w14` with a seeded 1000x bootstrap so the CIs are reproducible.

**Corpus vintage**
Canonical W14 twin-graft ladders. No `DEPRECATION_WARNING.md` applies.

Two caveats the fragment records: (i) Gemma's headline +0.0237 [+0.0179,+0.0297] survives four of four robustness variants but shrinks to +0.0082 when extremes are removed; (ii) LLaMA's -0.0156 loses its zero-excluding interval under two of the four variants, and its parse rate at dose +3 falls to 0.72-0.79, so the LLaMA verdict rests on a thinner denominator than Gemma's.

**Reproduction status**
VERIFIED, 2026-08-25. 21 numeric values; max deviation 0.000, including the paper's +0.0469 / +0.0358 / +0.0218 slopes and the twin contrast +0.0237.
