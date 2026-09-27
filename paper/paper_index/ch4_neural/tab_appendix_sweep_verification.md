# Table — `tab:appendix-sweep-verification`

**Paper location**
Appendix \subsection{Reproducibility note for the readout pipeline}, `neurips_content_en/appendix.tex:538`

**What the experiment asks**
An earlier version of the analysis had a bug that let the answer leak into the predictor; this table shows which of the reported numbers survive the fix and which were inflated by the bug.

**HF path(s) of the raw data**
Old side: `legacy/v17_leaky_pipeline/paper_neural_audit.json`. New side: `sae_v3_analysis/results/table1_groupkfold_L22.json`.

**Code that turns raw data into the printed values**
NOT CHECKED — no regeneration script for this float exists in the repo or in HF `paper_neurips_2026/camera_ready/scripts/`.

**Corpus vintage**
Deliberately MIXED, and that is the point of the float: one column is the DEPRECATED leaky pipeline. `legacy/v17_leaky_pipeline/` is marked do-not-cite in `NEURIPS_CANONICAL_INDEX.md` §5 (RandomForest fitted before the CV split). It carries no `DEPRECATION_WARNING.md` file of its own — the marker is the `legacy/` path plus the canonical index — so a reader who only greps for `DEPRECATION_WARNING.md` will not see the warning. Citing the deprecated column *as the comparison* is legitimate; citing it as a result is not.

**Reproduction status**
NOT CHECKED — no generator, and the leaky-pipeline side cannot be re-derived without re-running the withdrawn code.
