# Table — `tab:appendix-sweep-peak-mismatch`

**Paper location**
Appendix \subsection{Reproducibility note for the readout pipeline}, `neurips_content_en/appendix.tex:562`

**What the experiment asks**
Under the old buggy analysis a different layer looked best; this table records that disagreement so nobody re-derives the abandoned layer choice.

**HF path(s) of the raw data**
`legacy/pre_groupkfold_sweep/` (42-layer random-KFold sweep) against the body-cited layers in `sae_v3_analysis/results/table1_groupkfold_L22.json`.

**Code that turns raw data into the printed values**
NOT CHECKED — no regeneration script exists.

**Corpus vintage**
DEPRECATED BY DESIGN. `legacy/pre_groupkfold_sweep/` is the random-KFold leaky sweep, up to 3x divergent from the body, and is the origin of the abandoned L24/L16 peak-layer route. The paper's own caption says the table is 'kept for transparency about the earlier pipeline, and not relied on anywhere in the body'. As with `legacy/v17_leaky_pipeline/`, the do-not-cite marker is the `legacy/` path and `NEURIPS_CANONICAL_INDEX.md` §5, not a `DEPRECATION_WARNING.md` file.

**Reproduction status**
NOT CHECKED — by construction one side is withdrawn output; there is nothing canonical to reproduce it against.
