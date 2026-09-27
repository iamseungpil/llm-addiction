# Table — `tab:choice-ladder`

**Paper location**
Appendix \subsection{The choice ladder with behaviour beside bankruptcy}, `neurips_content_en/appendix.tex:1055`

**What the experiment asks**
Freedom is added one rung at a time — forced stake, one choice at the start, a fresh choice every round, then no cap at all. At which rung does ruin jump?

**HF path(s) of the raw data**
`rebuttal_neurips_2026/policy_choice_ladder_e8/` (100 games per arm, 200 for the one-time-choice arm).

**Code that turns raw data into the printed values**
No regeneration fragment. Numbers of record: `VERIFIED_FACTS.md` §Y (guard tally, recounted 2026-07-28) and §Y.1 (ladder completed 64/64, 2026-07-29).

**Corpus vintage**
Rebuttal-era corpus; no `DEPRECATION_WARNING.md` under `rebuttal_neurips_2026/`.

Denominator rule the appendix states and any recompute must honour: the one-time-choice arm drops two Gemma games whose initial stake could not be parsed; every other rung uses its included-game denominator. The forced-stake figure quoted (2%) is the matched $70 cap cell, while across the four forced caps that arm runs 0%-15% — quoting the single cell without the range would overstate the contrast.

**Reproduction status**
NOT CHECKED — no fragment; the printed 2 / 5 / 85 / 80 percentages were not re-derived from `policy_choice_ladder_e8/` here.
