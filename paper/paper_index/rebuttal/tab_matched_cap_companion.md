# Table — `tab:matched-cap-companion`

**Paper location**
Appendix, immediately after \subsection{Matched-cap intervals for every model and cap},
`neurips_content_en/appendix.tex:1204-1221`. Companion float to `tab:matched-cap-intervals` — same
prose paragraph introduces both, and this table exists specifically to carry the three companion
measures (participation, realised wager, re-betting after a first loss) that explain the earlier
table's fixed-arm rates.

**What the experiment asks**
Beside every one of the sixteen matched-cap bankruptcy cells, how often did the forced arm actually
play, how much did each arm stake when it did, and did either arm keep betting after its first
loss?

**HF path(s) of the raw data**
`rebuttal_neurips_2026/matched_cap_mc32/` — the identical corpus behind `tab:matched-cap-intervals`
(4 models x 4 caps x {fixed, variable} x {BASE, GMPRW}, 50 games per cell). This table reads
participation, realised-wager and re-bet fields off the same per-game records rather than a
separate collection. The fixed-arm re-run after the D5 defect is in
`rebuttal_neurips_2026/track0_fixed_arm_rerun/`, same as its sibling table.

**Code that turns raw data into the printed values**
No regeneration fragment exists for this table specifically. `scripts/figures/fig05_matched_cap.py`
draws the matched-cap bankruptcy panel (`draw_panel_b`) from the same `mc32` corpus but does not
compute participation, realised wager or re-betting — those three columns have no generator found
under `scripts/` on this machine, the same gap `ch4_causal/fig_causal_battery.md` and
`rebuttal/tab_matched_cap_intervals.md` record for their own floats. Numbers of record for the
underlying corpus: `archive/rebuttal_20260731/VERIFIED_FACTS.md` §A1 (Wilson intervals per cell),
§Y/§Y.1 (guard tally, ladder completion).

**Corpus vintage**
Rebuttal-era corpus, identical vintage to `tab:matched-cap-intervals` — see that manifest for the
two vintage traps recorded in `VERIFIED_FACTS.md` (post-fix fixed cells at caps 30/50/70; this run
is explicitly not a reproduction of the paper's own cap ablation). No `DEPRECATION_WARNING.md`
exists under `rebuttal_neurips_2026/`.

**Printed scope (camera-ready, 2026-09-08)**
The paper prints the three models of the main experiments (GPT-4o-mini, GPT-4.1-mini, Gemini-2.5-Flash), as the rebuttal reply did. The Claude-Haiku-4.5 pairs collected in `rebuttal_neurips_2026/matched_cap_mc32/` (claude-haiku-4-5-20251001, 2026-07-28) stand in for the withdrawn Claude-3.5-Haiku checkpoint and remain in the release and in `paper_data/fig05_matched_cap.json` → `panel_b`, but are not printed. Body Figure 3(d) is the three-model pooled mean (`panel_d_body`); the appendix forest plot is the twelve GMPRW pairs (`panel_d_forest_appendix`).

**Reproduction status**
NOT CHECKED — no generator to check against, same status as its sibling `tab:matched-cap-intervals`.
The prose sentence built from this table (participation falling to 36% at the $70 cap on
GPT-4o-mini and 64% on GPT-4.1-mini; realised wager below the forced stake in every printed cell)
has not been independently recomputed from the raw `mc32` records.
