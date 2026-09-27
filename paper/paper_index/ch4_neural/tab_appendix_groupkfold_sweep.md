# Table — `tab:appendix-groupkfold-sweep`

**Paper location**
Appendix, `neurips_content_en/_appendix_layer_sweep_table.tex:6`, \input at `appendix.tex:590`

**What the experiment asks**
Would the same internal-readout result have looked different if a different layer of the network had been read instead of the one the paper reports?

**HF path(s) of the raw data**
`sae_v3_analysis/results/table1_groupkfold_L{8,12,22,25,30}.json`.

**Code that turns raw data into the printed values**
Upstream generator: HF `sae_v3_analysis/scripts/build_appendix_layer_sweep_table.py` (wired in, not rewritten — its stdout was verified byte-identical to the committed .tex).
Regeneration harness: `scripts/tables/appendix_neural_tables.py` -> `paper_data/tables/appendix/A02_layer_sweep.tex`.

**Corpus vintage**
Canonical strict GroupKFold at all five layers. This table is the *replacement* for the deprecated `legacy/pre_groupkfold_sweep/` random-KFold sweep; that legacy folder is not read. No `DEPRECATION_WARNING.md` applies.

**Reproduction status**
VERIFIED, 2026-08-25. 63 numeric values in the printed tabular body vs the regenerated fragment; max deviation 0.000.
