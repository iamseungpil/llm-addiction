# Figure — `fig:causal-removal`

**Paper location**
Appendix \section{Causal battery: sufficiency, necessity, transfer, and condition writability}
(`\label{appendix:causal-battery}`), `neurips_content_en/appendix.tex:829-834`; graphic
`images/fig04b_causal_removal.pdf`. Companion float to `fig:causal-battery` — same section, same
generator, second half of one battery split across two figures.

**What the experiment asks**
Three sub-questions in one panel set: does projecting the behavioural axis back out lower betting
(necessity, panel a)? Do axes mined on one game move behaviour on the other two, in the direction
pre-registered in advance (transfer, panel b)? And is the behavioural axis easier to push under one
prompt frame than another (condition writability, panel c)?

**HF path(s) of the raw data**
Panel (a), removal (necessity): `experiments/sec4_causal/checkpoints/sec4_w13/*.jsonl` — the same
arm behind `tab:causal-battery-suffnec`.
Panel (b), cross-task transfer: `experiments/sec4_causal/checkpoints/sec4_w7/*.jsonl`; adjudication
record `experiments/sec4_causal/analysis/sec4_w7_adjudication.json` — the same arm behind
`tab:causal-transfer-matrix`.
Panel (c), condition writability: `experiments/sec4_causal/checkpoints/sec4_w14/*.jsonl`; analysis
`experiments/sec4_causal/analysis/sec4_w14_analysis.json` and
`sec4_w14_analysis_robustness_20260710.json` — the same arm behind `tab:causal-condition-writability`.

**Code that turns raw data into the printed values**
`scripts/figures/fig04_causal_battery.py` (= HF
`paper_neurips_2026/camera_ready/scripts/figures/fig04_causal_battery.py`). One script draws both
figures of the battery in a single pass — it writes `images/fig04_causal_battery.pdf` (=
`fig:causal-battery`) and `images/fig04b_causal_removal.pdf` (this figure) from the same loaded
data; see the `for stem in ("fig04_causal_battery", "fig04b_causal_removal")` loop near the end of
the script. Sidecar `paper_data/fig04_causal_battery.json` carries both floats' panel data under
the keys `panel_c_removal`, `panel_d_transfer` and `panel_e_writability`.

**Corpus vintage**
Canonical §4 causal waves — the same W7/W13/W14 corpora documented in
`ch4_causal/tab_causal_transfer_matrix.md` and `ch4_causal/tab_causal_condition_writability.md`. No
`DEPRECATION_WARNING.md` applies to any of the three source trees.

**Reproduction status**
NOT CHECKED (figure as drawn), consistent with its sibling `fig:causal-battery`. The underlying
numbers it plots are VERIFIED elsewhere: panel (a) in `ch4_causal/tab_causal_battery_suffnec.md`,
panel (b) in `ch4_causal/tab_causal_transfer_matrix.md`, panel (c) in
`ch4_causal/tab_causal_condition_writability.md`.
