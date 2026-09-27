# Table — `tab:matched-cap-intervals`

**Paper location**
Appendix \subsection{Matched-cap intervals for every model and cap}, `neurips_content_en/appendix.tex:1001`

**What the experiment asks**
When both arms are capped at the same maximum bet, does the arm that gets to choose still go broke more often — at every cap, in every model?

**HF path(s) of the raw data**
`rebuttal_neurips_2026/matched_cap_mc32/` (4 models x 4 caps x {fixed, variable} x {BASE, GMPRW}, 50 games per cell; 64 cells). The fixed-arm re-run after the D5 defect is in `rebuttal_neurips_2026/track0_fixed_arm_rerun/`.

**Code that turns raw data into the printed values**
No regeneration fragment. The closest live code is `scripts/figures/fig05_matched_cap.py`, whose panel (b) reads the same `mc32` corpus and emits `paper_data/fig05_matched_cap.json`.
Numbers of record: `archive/rebuttal_20260731/VERIFIED_FACTS.md` §A1 (Wilson intervals per cell), §Y (guard tally recounted), §Y.1 (ladder completed 64/64, 2026-07-29), §P.1 (dissociation reproduces in three more models once the condition set matches the paper's), §R.1 (the paper's own matched-cap numbers recomputed from the stored games).

**Corpus vintage**
Rebuttal-era corpus, distinct from the paper's slot-machine corpus. No `DEPRECATION_WARNING.md` exists under `rebuttal_neurips_2026/`.

Two vintage traps recorded in `VERIFIED_FACTS.md`: (i) §A1 notes that the fixed cells at caps 30/50/70 are **post-fix** — a pre-fix copy of those cells exists and must not be mixed in; (ii) §A6 states flatly that this run 'is NOT a reproduction of the paper's cap ablation', so it must not be cited as one.
A third, in `tab:added-controls`: the Claude cell here runs **Claude-Haiku-4.5**, not the Claude-3.5-Haiku of the six-model roster, because the older model had been withdrawn from the API.

**Printed scope (camera-ready, 2026-09-08)**
The paper prints the three models of the main experiments (GPT-4o-mini, GPT-4.1-mini, Gemini-2.5-Flash), as the rebuttal reply did. The Claude-Haiku-4.5 pairs collected in `rebuttal_neurips_2026/matched_cap_mc32/` (claude-haiku-4-5-20251001, 2026-07-28) stand in for the withdrawn Claude-3.5-Haiku checkpoint and remain in the release and in `paper_data/fig05_matched_cap.json` → `panel_b`, but are not printed. The release's third exclusion rule — cells scored by the earlier parser, which misparses 2.4% of
Claude's decisions (`matched_cap_mc32/LEGACY_PARSER_claude/`) — is Claude-specific and is no
longer stated in the paper, since no Claude cell is printed. Body Figure 3(d) is the three-model pooled mean (`panel_d_body`); the appendix forest plot is the twelve GMPRW pairs (`panel_d_forest_appendix`).

**Reproduction status**
NOT CHECKED cell-by-cell here. The float carries an open public promise: `archive/rebuttal_20260731/posted_openreview/README.md` item 6 records that the posted letter told reviewer KuK5 'the exact intervals will be given in the revision', and the letter printed bold marks only. The intervals themselves exist in `VERIFIED_FACTS.md` §A1.
