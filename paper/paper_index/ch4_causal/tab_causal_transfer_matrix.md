# Table — `tab:causal-transfer-matrix`

**Paper location**
Appendix \subsection{Cross-task transfer and condition writability}, `neurips_content_en/appendix.tex:904`

**What the experiment asks**
A direction built inside one game is pushed while the model plays a different game: does the behaviour move the way it was predicted to, in each of twelve pre-registered combinations?

**HF path(s) of the raw data**
`experiments/sec4_causal/checkpoints/sec4_w7/*.jsonl`; adjudication record `experiments/sec4_causal/analysis/sec4_w7_adjudication.json`.

**Code that turns raw data into the printed values**
`scripts/tables/appendix_neural_tables.py` -> `paper_data/tables/appendix/A09_steering_matrix.tex`.

**Corpus vintage**
Canonical W7 rollouts. **DEFECT 3, recorded by the regeneration script:** before that regeneration the published cells existed *only as literals* inside `scripts/gen_fig_cross_context_write.py` — there was no code path from the rollouts to the printed numbers, and the project's own `INDEX.md` disagreed with the paper on two cells. All twelve cells are now recomputed from the raw `.jsonl`.
No `DEPRECATION_WARNING.md` applies. The MW column is greyed in the paper because its baseline spin rate already sits at the task ceiling, so nulls there are uninformative.

**Reproduction status**
VERIFIED, 2026-08-25. 12 numeric values (the observed z per cell); max deviation 0.000. The primary tally 7/10 confident cells and 11/12 sign agreement both reproduce and both match the paper text.
One reporting choice is worth naming: z is taken against a null of only three random directions, using the population SD. Under the sample SD, three cells that read |z|>=2 fall below it (smiba->ic -1.78, mwrc->ic +2.43 stays, icrc->sm -2.39 stays); the fragment prints both.
