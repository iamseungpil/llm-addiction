# Rebuttal Table Registry — Submission 24231

Table numbers are assigned centrally here. Do not invent a number and do not renumber.

**One printer per table.** Each table is typeset exactly once, in the file listed under *Owner*.
Every other section refers to it by number and never reprints it, in whole or in part.

**House style.** Each table is introduced by a bold caption line of the form

```latex
\textbf{Rebuttal Table 3.} Participation and exposure at cap \$70, n = 200 per cell.
```

one line, saying what the table shows. Body uses `booktabs` (`\toprule`, `\midrule`,
`\bottomrule`; no vertical rules). Cross-references are written in prose as
"Rebuttal Table 3", not `\ref`.

---

## Assignment

| No. | Contents | Owner (prints it) | Cross-references it |
|---|---|---|---|
| **T1** | Matched-cap bankruptcy grid: 6 models × 4 caps × 2 modes, n = 200 per cell, 95% Wilson intervals | `kuk5.tex` | `gbsa.tex` (Q1, W4), `appendix.tex` (B1, B9) |
| **T2** | Range expansion refuted: executed wager vs. ruin at cap $70 | `kuk5.tex` | `gbsa.tex` (W4) |
| **T3** | Participation and exposure at cap $70, all six models, both arms | `kuk5.tex` | `gbsa.tex` (W4), `appendix.tex` (B2) |
| **T4** | Parser re-parse audit across three corpora (rerun, original E1, E7) | `appendix.tex` | `kuk5.tex`, `a3zu.tex` (Q1), `gbsa.tex` |
| **T5** | Framing × rationality factorial, all completed cells (32 of 44) | `gbsa.tex` | `a3zu.tex` (Q3), `appendix.tex` (B10, B12) |
| **T6** | Factorial main effects on participation, variable arm (pp) | `gbsa.tex` | `a3zu.tex` (Q3) |
| **T7** | Language-instrument battery, goal contrast across instrument variants | `a3zu.tex` | `gbsa.tex` (Q2) |
| **T8** | Prior-work lineage: constructs, coding schemes, lexicons | `a3zu.tex` | `gbsa.tex` (Q2) |
| **T9** | What arrives in the second response: item, reviewer point, decision rule fixed in advance | `appendix.tex` | all three reviewer sections |

## Ownership by file

- `kuk5.tex` — **T1, T2, T3**. All three are matched-cap grid results, and KuK5's Q1 is the
  question that grid answers. gbSA's Q1 asks the same question and refers to these tables.
- `a3zu.tex` — **T7, T8**. Both concern the text measure, which is a3Zu's weakness.
- `gbsa.tex` — **T5, T6**. Both are the framing × rationality factorial, which answers gbSA's Q3
  and bears on gbSA's W2.
- `appendix.tex` — **T4, T9**. T4 is a disclosed defect audit. T9 is the forward register.

## Notes

- `main.tex` prints no numbered Rebuttal Table. Its reviewer-score table and its
  four-corrections table are unnumbered and stay unnumbered.
- The reviewer-score table records **rating and confidence only**. Quality, clarity,
  significance and originality sub-scores were not collected on this venue's form. Do not add
  those rows anywhere.
- Every number in every table must appear verbatim in `VERIFIED_FACTS.md`. Nothing is derived,
  rounded differently, or recombined.
- T1 fixed cells at caps $30, $50 and $70 are post-fix values. Say so in the caption line or
  immediately under it.
- T3's Claude-Haiku fixed-arm figures are essentially a parsing artefact. Annotate that wherever
  T3 is printed or cross-referenced, and point to T4.

## Macros available from `main.tex`

- `\plan{...}` — the planned revision action, at the end of an Author Response block.
- `\redrule` — the red rule that separates a reviewer's quoted words from the Author Response.
