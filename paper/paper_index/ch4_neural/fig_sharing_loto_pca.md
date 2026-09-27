# Figure — `fig:sharing`

**Paper location**
Appendix \subsection{Cross-task sharing at the hidden-state and sparse-feature levels}, `neurips_content_en/appendix.tex:719`; graphic `images/fig5b_pca_appendix.pdf`

**What the experiment asks**
A picture of whether the 'about to go broke' pattern learned from two games can spot the same thing in a third game it has never seen.

**HF path(s) of the raw data**
`sae_features_v3/{slot_machine,investment_choice,mystery_wheel}/gemma/hidden_states_dp.npz`, layer 22, bankruptcy vs voluntary-stop endpoints.

**Code that turns raw data into the printed values**
`scripts/gen_fig5b_pca.py` (= HF `paper_neurips_2026/camera_ready/scripts/gen_fig5b_pca.py`). Related: `scripts/gen_fig5b_sharing.py`, `scripts/gen_fig5_panels.py`. Figure on HF: `paper_neurips_2026/camera_ready/figures/fig5b_pca_appendix.pdf`.

**Corpus vintage**
Canonical `sae_features_v3/` hidden states at L22. The script's `DATA` constant is now an *optional* on-disk mirror under `$LLM_ADDICTION_DATA`; when that directory is absent the same six npz files are pulled from HF `llm-addiction-research/llm-addiction` at `sae_features_v3/{slot_machine,investment_choice,mystery_wheel}/gemma/hidden_states_dp.npz`, so a fresh clone re-runs it without repointing anything. No `DEPRECATION_WARNING.md` applies.

Note the panel AUCs printed in the figure titles use a single SVD-based shared axis on the full held-out data, while `tab:rq2-sharing` uses a matched 5-fold CV centroid-PCA estimate at the same layer. The caption says so; the two therefore differ by a few points by construction.

**Reproduction status**
NOT CHECKED (figure as drawn). The matched table estimates are VERIFIED — see `ch4_neural/tab_rq2_sharing.md`.
