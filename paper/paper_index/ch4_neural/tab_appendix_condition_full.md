# Table — `tab:condition-modulation` (formerly also labelled `tab:appendix-condition-full`)

**Paper location**
Body §4 \subsection{Are internal risk signals shared across tasks?} continuation, the float sits at
`neurips_content_en/4.neural.tex:94-99` (`\label{tab:condition-modulation}` on line 99), not in
`appendix.tex`. An earlier pass of this manifest pointed at `appendix.tex:797-798`, which is a
different table (`tab:appendix-condition-multi`, the MW/IC companion, see
`ch4_neural/tab_appendix_condition_multi.md`) — that line-number reference has been corrected here.
The compiled `neurips_en.aux` gives this float's printed number directly:
`\newlabel{tab:condition-modulation}{{3}{9}{...}{table.26}{}}` — **Table 3**, immediately after the
sharing-audit Table 2 (`tab:rq2-sharing`). `NEURIPS_CANONICAL_INDEX.md` and
`scripts/tables/body_tables.py` call it "body Table 4" from an earlier draft's table count; that
number is stale against the compiled paper and is not used here. The float also carried
`tab:appendix-condition-full`, an unreferenced second label that named a body table "appendix"; it
has been removed.

**What the experiment asks**
On the slot machine, does the readout of risky betting get stronger when the model is given a goal, and what happens in the forced-bet arm where the model has no choice to read?

**HF path(s) of the raw data**
Six variable-arm rows: `sae_v3_analysis/results/condition_modulation_groupkfold_L22.json`.
Printed fixed-bet row: `sae_v3_analysis/results/condition_modulation_continuous_ilc_L22.json`.

**Code that turns raw data into the printed values**
`scripts/tables/appendix_neural_tables.py` -> `paper_data/tables/appendix/A08_condition_full.tex`; `scripts/tables/body_tables.py` -> `paper_data/tables/body/table4_condition_modulation.tex`.

**Corpus vintage**
MIXED — flagged as **DEFECT 1** by the regeneration script, and the paper carries no note of it.
The rows 'All variable' / +-G / +-M come from `condition_modulation_groupkfold_L22.json` (GroupKFold by game id). The printed **Fixed** row does not: its values match `condition_modulation_continuous_ilc_L22.json` (continuous-I_LC, plain CV). Two different pipelines are printed in one column block.
Both pipelines' fixed-bet values are emitted and labelled in the fragment, e.g. Gemma SM I_EC: GroupKFold -0.0385 vs continuous-I_LC -1.5105; Gemma SM I_LC: -0.0146 vs -0.0141 (and n differs, 3,177 vs 4,625).
This is a pipeline-vintage error inside one float, not a corpus error: no `DEPRECATION_WARNING.md` applies to either source file.

**Reproduction status**
VERIFIED for the six variable-arm rows, 2026-08-25: the first 30 numeric values match the regenerated fragment exactly, max deviation 0.000 (including the headline 0.063 -> 0.153, Delta_G +143%).
The Fixed row is WRONG-BY-PROVENANCE: it is correct arithmetic on a *different* pipeline than the rows it is printed beside. Not a fabrication; a mislabelled vintage.

**Dispersion added for the camera-ready (2026-09-01)**
Every $R^2$ cell in the six variable-arm rows now prints `value $\pm$ SE`, SE = `r2_std`/sqrt(5) over
the 5 GroupKFold-by-game folds — the same definition Table 1 uses. `r2_std` is a released field of
`condition_modulation_groupkfold_L22.json`, so no re-fit was run; the 30 point estimates were first
reproduced from that file to 3 dp with max deviation 0.000. Sidecar: `paper_data/table3_se_condition_L22.json`
(HF revision `6f3bd8262279ac1924468e7162613e33823acb69`).
The `All variable` row is Table 1's slot-machine row re-printed (its `all_variable` block is identical to
`table1_groupkfold_L22.json`), so its SE is carried over from `paper_data/table1_se_N200.json` rather than
recomputed — the released fold sds and the permutation run's agree to <=0.0007 but would print differently
on LLaMA SM $I_\text{LC}$ and $I_\text{BA}$ (0.004 vs 0.003), and one quantity must not carry two
different intervals in one section. Both values are recorded per cell in the sidecar.
The `Delta_G%`/`Delta_M%` rows stay bare (ratios of two noisy $R^2$; denominators reach 0.008 +- 0.007).
The printed `Fixed` row stays bare: its source, the continuous-$I_\text{LC}$ release, stores only `r2`,
`n` and `mean_target` per subset, so no fold sd exists for it. The dagger footnote's game-grouped
counterparts do have one and now carry it.
`scripts/tables/body_tables.py:table4()` was updated to emit the same `+-` and the two `n` columns, so
the fragment again matches the tabular cell for cell; the only residual diff is the `Fixed` row, which the
emitter deliberately does not produce.
