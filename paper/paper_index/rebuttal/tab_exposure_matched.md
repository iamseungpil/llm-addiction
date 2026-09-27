# Table — `tab:exposure-matched`

**Paper location**
Appendix \subsection{Participation and cumulative exposure}, `neurips_content_en/appendix.tex:1309`

**What the experiment asks**
If you stop the clock once both arms have staked the same total number of dollars, does the arm that chose its own bets still go broke more often?

**HF path(s) of the raw data**
`rebuttal_neurips_2026/nested_baseline_and_audits_e2/exposure_matched.json` (six swept thresholds; the table prints four). Underlying games: `rebuttal_neurips_2026/matched_cap_mc32/`.

**Code that turns raw data into the printed values**
HF `rebuttal_neurips_2026/nested_baseline_and_audits_e2/src/exposure_matched.py`.

**Corpus vintage**
Rebuttal-era; no `DEPRECATION_WARNING.md` under `rebuttal_neurips_2026/`. The `_SUPERSEDED_oldscript` files in the same directory are nested-baseline artefacts and are unrelated to this table.
The appendix states two limits that are vintage-relevant: the threshold sweep is post-hoc and was not pre-registered, and the no-discretion arm is better described as a forced-maximum arm than a clean control.

**Reproduction status**
VERIFIED, 2026-08-25. All 32 printed rates plus the four pairs of wagering-game denominators in the caption (29/50, 50/50, 32/50, 18/50) reproduce from `exposure_matched.json`; **max deviation 0.0 percentage points**. The caption's claim that the two omitted thresholds ($150, $500) agree in direction also holds in the released file.
