# Figure — `fig:sign-transfer`

**Paper location**
Appendix \subsection{Cross-task transfer and condition writability}, `neurips_content_en/appendix.tex:926`; graphic `images/fig_axis_alignment.pdf`

**What the experiment asks**
Are the three games' 'bet bigger' directions pointing the same way inside the model, or are they nearly unrelated?

**HF path(s) of the raw data**
`experiments/sec4_causal/assets/gemma_*_i_ba_behavioural.npz` (and the readout/confound axes beside them), pulled from HF at run time.

**Code that turns raw data into the printed values**
`scripts/gen_fig_cross_context_write.py` (= HF `paper_neurips_2026/camera_ready/scripts/gen_fig_cross_context_write.py`), which emits `fig_axis_alignment.pdf` alongside the two cross-context panels. Figure on HF: `paper_neurips_2026/camera_ready/figures/fig_axis_alignment.pdf`.

**Corpus vintage**
Canonical `experiments/sec4_causal/assets/` axis files on HF `llm-addiction-research/llm-addiction`. No `DEPRECATION_WARNING.md` applies. The script still needs HF auth (the dataset is gated), but `OUT` is now derived from the repository root (`images/`) and the local ladder directory from `$LLM_ADDICTION_ANALYSIS`, so a fresh clone needs no path edits.

**Reproduction status**
NOT CHECKED — the figure has a generator but no numeric sidecar, and no independent recompute of the plotted cosines was run here.
