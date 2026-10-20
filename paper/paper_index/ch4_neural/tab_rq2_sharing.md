# Table (one float carrying two labels) — `tab:rq2-sharing` / `tab:sharing-transfer`

**Paper location**
Body §4 \subsection{Are internal risk signals shared across tasks?} (`\label{sec:rq2}`),
`neurips_content_en/4.neural.tex:51-77`. Both labels sit on the same `\begin{table}[!ht]`. This
float now lives in the body, not the appendix — an earlier draft carried it as an appendix table,
and `NEURIPS_CANONICAL_INDEX.md` (and this file, before this pass) still called it "body Table 3"
on that basis. The compiled `neurips_en.aux` is the ground truth for the printed number:
`\newlabel{tab:rq2-sharing}{{2}{8}{...}{table.23}{}}` — this is **Table 2**, on page 8, the 23rd
float in document order (behind it are `fig:causal-battery`'s companion tables and the rest of the
appendix floats, all inputted after it). `tab:condition-modulation` prints as Table 3 immediately
after it, matching the paper's own in-body table numbering (Table 1 SAE readout, Table 2 sharing
audit, Table 3 condition modulation).

**What the experiment asks**
Four ways of asking the same question: does anything learned about risk in one game carry over to another game?

**HF path(s) of the raw data**
(i) cosine alignment: `sae_v3_analysis/results/shared_subspace_hidden_audit_20260410.json` (`weight_cosines`)
(ii) sparse-feature transfer: `sae_v3_analysis/results/iba_cross_task_transfer.json`
(iii)/(iv) LOTO PCA AUC: `sae_v3_analysis/results/robustness/rq2_aligned_hidden_transfer_gemma_centroid_pca_L22_r{1,2}.json`, mirrored at `paper_neurips_2026/tables/body/table2_rq2_audits/data/rq2_aligned_hidden_transfer_gemma_centroid_pca_L22_r{1,2}_e8g1_L22_r{1,2}.json`
Underlying states: `sae_features_v3/{task}/gemma/hidden_states_dp.npz` at L22.

**Code that turns raw data into the printed values**
`scripts/tables/appendix_neural_tables.py` -> `paper_data/tables/appendix/A06_rq2_sharing.tex` (full float).
`scripts/tables/body_tables.py` -> `paper_data/tables/body/table3_sharing_transfer.tex` (rows (i)/(ii) only).
Upstream: `tables/body/table2_rq2_audits/code/cross_domain.py`.

**Corpus vintage**
Canonical, but with a live provenance trap recorded in `NEURIPS_CANONICAL_INDEX.md`: the HF manifest once named `rq2_audit_consistent_layer.json` as the representative file for this table. That file is an **error stub** (`layer 23 not in...`), not data. The real values are in `rq2_aligned_hidden_transfer_*_L22_r{1,2}*.json`.

Row (i) is the cosine between per-task **readout** directions, not between BK centroid differences. Its source is the released audit file (`weight_cosines`), written by `experiments/07_sae_readout/src/run_hidden_state_analyses.py`, function `shared_subspace()`: a StandardScaler, then a full-dimension (3,584) logistic regression (C=1, class_weight=balanced), then the cosine of the three weight vectors. A re-run on `hidden_states_dp.npz` at L22 (2026-10-20) gives ic-sm +0.0407, ic-mw -0.0258, sm-mw -0.0276, which matches the printed 0.04 / 0.03 / 0.04 to two decimals. The audit file's third PCA eigenvalue (about 1e-31) is expected: PCA of three centred vectors has rank two.
Two other objects are listed in the header of `A06_rq2_sharing.tex` for comparison; they are different quantities and give different values —
  reference A (raw BK centroid-difference directions): ic-mw +0.3675, ic-sm -0.0541, mw-sm +0.4006
  reference B (per-task PCA(64)+logistic readout directions, the object rows (iii)/(iv) are built from): ic-mw +0.0565, ic-sm +0.0305, mw-sm +0.1188
How to read row (i) (L22, one seed, 2026-10-20): the logistic weights are not stable within a task (split-half cosine 0.02-0.11 over IC, SM, MW), so near-zero cross-task cosines sit at the noise floor; the centroid directions are stable (split-half 0.96-0.99) but their cross-task cosines stay inside a label-permutation null (95th percentile of |cos| 0.35-0.51). Row (i) therefore describes the readouts and does not decide whether the tasks share a direction; rows (ii)-(iv) carry that claim. The paper text names row (i) "BK readout directions" accordingly.

Row (ii): all six off-diagonal transfers are recomputed at Gemma L22 by `scripts/tables/appendix_neural_tables.py` and recorded in the header of `paper_data/tables/appendix/A06_rq2_sharing.tex` (mw->ic -0.29, sm->ic -0.70, ic->mw -30.9, sm->mw -3.78, ic->sm -1.30, mw->sm -0.18). Every entry is negative, which is what the printed `<0` in all three columns states. The older `iba_cross_task_transfer.json` holds only SM<->MW at L18/L24 and is not the source of this row; `paper_data/tables/body/table3_sharing_transfer.tex`, which reads that older file, therefore shows `n/c` for IC and should not be used for this row. No `DEPRECATION_WARNING.md` applies to any path used.

**Reproduction status**
VERIFIED against the released files, 2026-08-25: 30 numeric values in the printed tabular body vs `A06_rq2_sharing.tex`; max deviation 0.000.
Row (i): the printed value is the released audit JSON's, and the method that wrote it is `run_hidden_state_analyses.py::shared_subspace()`; a re-run reproduces it to two decimals (see the vintage note). No part of the table is outstanding.
