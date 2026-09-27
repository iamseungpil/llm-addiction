# Table — `tab:appendix-selectivity-controls`

**Paper location**
Appendix \subsection{Selectivity: how the readout fares against noise-only controls}, `neurips_content_en/appendix.tex:625`

**What the experiment asks**
Does the internal readout beat twenty deliberately meaningless controls built from shuffled data?

**HF path(s) of the raw data**
`sae_v3_analysis/results/robustness/probe_selectivity_controls.json`.

**Code that turns raw data into the printed values**
`scripts/tables/appendix_neural_tables.py` -> `paper_data/tables/appendix/A05_selectivity_controls.tex`.

**Corpus vintage**
Canonical robustness file, same pipeline family as `tab:neurips-selectivity-l2` (L24/L16 slices). No `DEPRECATION_WARNING.md` applies.

**Reproduction status**
VERIFIED, 2026-08-25. 28 numeric values compared; max deviation 0.000 (four apparent differences are the `$-$` typesetting artefact only). The p column sits at its permutation floor 0.0476 = 1/21 in all four rows.
