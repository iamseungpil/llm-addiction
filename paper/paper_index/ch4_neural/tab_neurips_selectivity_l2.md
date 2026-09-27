# Table — `tab:neurips-selectivity-l2`

**Paper location**
Appendix \subsection{Selectivity: how the readout fares against noise-only controls}, `neurips_content_en/appendix.tex:604`

**What the experiment asks**
If the predictor is tested only on prompt wordings it has never seen, does it still work — or was it just memorising the prompt?

**HF path(s) of the raw data**
`sae_v3_analysis/results/robustness/rq1_l2_selectivity.json`.

**Code that turns raw data into the printed values**
`scripts/tables/appendix_neural_tables.py` -> `paper_data/tables/appendix/A04_selectivity_l2.tex` (= HF `paper_neurips_2026/camera_ready/paper_data/tables/appendix/A04_selectivity_l2.tex`).

**Corpus vintage**
Canonical robustness file from the strict-CV pipeline. Note the layers are L24 (Gemma) / L16 (LLaMA), i.e. the peak-layer slices, not the body's L22 — that is the released file's own choice and is stated in the table's own rows. No `DEPRECATION_WARNING.md` applies.

**Reproduction status**
VERIFIED, 2026-08-25. 24 numeric values compared; max deviation 0.000 (the six apparent sign differences are LaTeX typesetting only: the paper writes `$-$0.019` where the fragment writes `-0.019`).
