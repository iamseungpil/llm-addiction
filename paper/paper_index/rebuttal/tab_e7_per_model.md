# Table — `tab:e7-per-model`

**Paper location**
Appendix \subsection{The framing-by-rationality factorial, model by model}
(`\label{appendix:e7-per-model}`), `neurips_content_en/appendix.tex:1277`.

**What the experiment asks**
The four headline averages of the framing-by-rationality experiment hide six models; which of
them actually go broke more when they can pick their own bet, and does telling them to be
rational change that?

**HF path(s) of the raw data**
`rebuttal_neurips_2026/framing_rationality_factorial_e7/` (top level), cap \$70, 100 games per
cell. The paper prints 36 of 40 nominal cells over five models; the four absent cells are Gemma
and LLaMA at no-framing x no-rationality, both betting modes. The eight released
`claude-haiku-4-5-20251001` cells are read and listed in the fragment's provenance block, marked
`[collected, NOT printed]`, and excluded from the table and both mean rows.
`rebuttal_neurips_2026/framing_rationality_factorial_e7/QUARANTINE_truncated_claude/` is
**excluded**. No Claude row is printed at all: the six-model roster's Claude-3.5-Haiku checkpoint
had been withdrawn from the API, and the substitute Claude-Haiku-4.5 runs stay in the release
unprinted, as in `tab_matched_cap_intervals`.

**Code that turns raw data into the printed values**
`paper_data/tables/appendix/code/gbsa_q3_e7_per_model.py`
-> `paper_data/tables/appendix/gbsa_q3_e7_per_model.tex` and `gbsa_q3_e7_per_model.json`
(repo-only; not in the HF camera-ready upload). Bankruptcy is `game['bankrupt']`; per-arm
intervals are 95% Wilson; the gap is variable minus fixed with a 95% Newcombe hybrid-score
interval. The fragment's `.tex` header carries a per-cell provenance block naming the source JSON
filename for every one of the 44 cells.

**Corpus vintage**
Rebuttal-era, canonical. No `DEPRECATION_WARNING.md` under `rebuttal_neurips_2026/` — checked
2026-08-27 against the full 9,449-file HF listing, which carries that filename at exactly three
paths, none of them read here. The quarantine directory is the vintage trap for this float and
the generator excludes it by name.

The caption discloses a known instrument limit rather than a corpus one: "a parser defect flips
18 of 7,639 decisions in this corpus (0.249%)". That figure is corroborated —
`archive/rebuttal_20260731/VERIFIED_FACTS.md:203` records E7 at 7,639 decisions and a 0.249% flip
rate, and `:655` breaks it into 16 bet-to-stop and 2 stop-to-bet. The fragment's own JSON adds
that a flip rate that low cannot move any bankruptcy count in the table by more than one game.


**Printed scope (camera-ready, 2026-09-08)**
The paper prints five models. The Claude row is dropped: the six-model roster's Claude-3.5-Haiku
checkpoint had been withdrawn from the API, and the substitute Claude-Haiku-4.5 runs — collected
and released under `framing_rationality_factorial_e7/` — are not printed, as in
`tab_matched_cap_intervals`. The printed table therefore covers 36 of 40 nominal cells, not 44 of
48, and the two mean rows are recomputed over the printed models: all available +22.0 / +8.2 /
+1.3 / +0.6, API models +6.7 / +0.7 / +1.3 / +0.0. The parser-defect figures in the caption
(18 flips, 7,639 decisions, 7,223 adjudicable, 412 unadjudicable, 4 truncated) are the audit of
the released corpus as a whole, which still contains the unprinted Claude cells; the caption now
says so. No per-model breakdown of that audit exists in `VERIFIED_FACTS.md`.


**Regenerated 2026-09-08** from `rebuttal_neurips_2026/framing_rationality_factorial_e7/` with
`E7_DIR` pointed at a full snapshot. Before the model change the generator reproduced its committed
output byte for byte apart from the timestamp, so the pipeline is sound; after it, all 20 printed
entries and both mean rows (+22.0 / +8.2 / +1.3 / +0.6 and +6.7 / +0.7 / +1.3 / +0.0) match
`neurips_content_en/appendix.tex` exactly, max deviation 0.0 pp.

**Reproduction status**
VERIFIED, 2026-08-27. All 24 cells (each carrying a fixed rate, a variable rate, a gap and a
two-ended interval) plus both mean-gap rows match
`paper_data/tables/appendix/gbsa_q3_e7_per_model.tex` exactly; **max deviation 0.0 percentage
points**. The four dashes are in the same four positions in both.

The paper's caption adds three statements the fragment's caption does not, all of them checked
here: "44 of 48 nominal cells collected" (matches the fragment header), the quarantine exclusion
(matches the generator), and the parser-defect rate (matches `VERIFIED_FACTS.md`).

**Parser audit, corrected (2026-09-26).** The 0.249% figure above audits the wrong cells: 8 of its 32
cells are the quarantined Claude files (`QUARANTINE_truncated_claude/`, 820 decisions, all 412
unadjudicable decisions and 10 of the 18 flips), and none of the 12 open-weight cells the table prints
were audited. Re-running `reparse_audit.py` over exactly the 36 printed cells gives 12,716 decisions,
15 truncated, 7 unadjudicable, 8 flips in 12,694 adjudicable (0.063%; API cells 8 of 6,815, open-weight
0 of 5,879), at most 2 flips in any cell. The table caption now prints these numbers.
