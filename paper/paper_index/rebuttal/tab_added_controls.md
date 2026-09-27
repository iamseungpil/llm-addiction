# Table — `tab:added-controls`

**Paper location**
Appendix \section{Additional controls for alternative explanations}, `neurips_content_en/appendix.tex:1149`

**What the experiment asks**
Five alternative explanations for the headline result — 'it's just one model', 'it's role-play', 'it's the example we showed it', 'it's the bigger cap', 'the game log alone would tell you' — each with the experiment that tests it and what came back.

**HF path(s) of the raw data**
One row per experiment, all under `rebuttal_neurips_2026/`:
  - matched-cap: `matched_cap_mc32/` (+ `track0_fixed_arm_rerun/`)
  - role-play / rationality factorial: `framing_rationality_factorial_e7/` (44 of 48 cells; `QUARANTINE_truncated_claude/` excluded)
  - worked example: `in_context_demo_api/`, `in_context_demo_open_weight/`, `in_context_demo_open_weight_persona/`
  - choice ladder: `policy_choice_ladder_e8/`
  - nested baseline: `nested_baseline_and_audits_e2/nested_baseline_{raw,sae}_{game,state}.json`

**Code that turns raw data into the printed values**
No single regeneration fragment — this is a narrative summary of five experiments. Per-row support:
  - worked-example intervals: `paper_data/tables/appendix/code/worked_example_intervals.py` -> `worked_example_intervals.{tex,json}` (PROPOSED, not \input by any .tex)
  - rationality instruction per model: `paper_data/tables/appendix/code/gbsa_q3_e7_per_model.py` -> `gbsa_q3_e7_per_model.{tex,json}` (PROPOSED)
  - nested baseline: HF `rebuttal_neurips_2026/nested_baseline_and_audits_e2/src/nested_baseline.py`

**Corpus vintage**
Rebuttal-era throughout; no `DEPRECATION_WARNING.md` under `rebuttal_neurips_2026/`, and `worked_example_intervals.py` records that it checked for one before reading.

Three vintage facts this table itself discloses or must: no row carries a Claude cell, because the roster's Claude-3.5-Haiku checkpoint had been withdrawn and the substitute **Claude-Haiku-4.5** runs are released but not printed; the factorial is 36 of 40 printed cells, the four missing being the open-weight no-role no-rationality arm; and `VERIFIED_FACTS.md` §T.6 records **two defects in the superseded nested-baseline script and two withdrawn artefacts** — the files `nested_baseline_{minimal,rich}_SUPERSEDED_oldscript.json` sit in the same HF directory as the live ones and must not be read.


**Printed scope (camera-ready, 2026-09-08)**
The matched-cap, factorial and worked-example rows print only the models of the main study. The
Claude-3.5-Haiku checkpoint had been withdrawn from the API before these runs, so those rows carry
no Claude cell and the substitute Claude-Haiku-4.5 runs stay in the release unprinted. Recomputed
row figures: matched cap 24 paired contrasts over 2,400 games (14 higher, 8 tied, 2 lower);
factorial 36 of 40 cells with means 22.0 / 8.2 / 1.3 / 0.6 and API-only 6.7 / 0.7 / 1.3 / 0.0;
worked example 12 cells over three models, one of six pairs excluding zero.

**Reproduction status**
NOT CHECKED as a whole. Two rows have partial support elsewhere: the worked-example row's promised exact intervals are computed in `worked_example_intervals.json` (2026-08-25), and the nested-baseline row's 0.1400 / 0.1674 / 0.2046 and 0.1452 / 0.1610 / 0.1903 are the figures `VERIFIED_FACTS.md` §T.1-T.3 record as passing the reproduction gate.
The posted letter (`posted_openreview/README.md`) shows the nested-baseline numbers were published as 0.145 / 0.161 / 0.198, i.e. the state-grouped row; the paper prints both rows, which resolves that.
