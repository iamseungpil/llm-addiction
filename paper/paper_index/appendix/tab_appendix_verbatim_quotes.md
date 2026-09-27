# Table — `tab:appendix-verbatim-quotes`

**Paper location**
Appendix \subsection{Supplementary evidence for gambling-related language markers}, `neurips_content_en/appendix.tex:448`

**What the experiment asks**
Twelve real sentences the models wrote while gambling, each labelled with which model wrote it, in which game, and on which round.

**HF path(s) of the raw data**
The full response corpus: 190,300 slot-machine responses and 51,467 investment-choice responses, i.e. the canonical six-model slot-machine roster (see `tab_appendix_slot_comprehensive.md`) plus the IC corpora.

**Code that turns raw data into the printed values**
There is no sampling script for this table anywhere, and the regeneration deliberately does not invent one. Instead `scripts/tables/appendix_behavioural_tables.py` emits a **provenance audit**: `paper_data/tables/appendix/verbatim_quotes_provenance.tex` takes each printed row's six coordinates and asks whether the quoted sentence is in the corpus, whether it is there exactly once, and whether it sits where the row says.

**Corpus vintage**
Canonical corpora searched. No `DEPRECATION_WARNING.md` applies to the searched paths.

**Reproduction status**
PARTIALLY VERIFIED, 2026-08-25 — 8 of 12 rows clean, 4 rows defective:
  - row 4 (GPT-4o-mini, IC, **var**, G, r4, rep 38): the sentence exists but at GPT-4o-mini / IC / **fixed** / G / r4 / rep 38. The printed bet-type coordinate is wrong.
  - row 9 (Claude, SM, var, GMP, r21, rep 14): at the stated coordinates, but the sentence recurs in 68 other responses (69 matches). Not distinctive.
  - row 10 (Claude, SM, var, GP, r34, rep 41): same problem, 16 matches.
  - row 11 (Claude, SM, var, M, r12, rep 42): the stated response exists but the quoted sentence is not verbatim in it (93% of its content words present) — a paraphrase or a stitched excerpt.
None of this is a data-vintage error; it is a citation-accuracy error inside the float.
