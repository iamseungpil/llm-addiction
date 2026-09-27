# Table — `tab:distortion-lexicon`

**Paper location**
Appendix \subsection{Supplementary evidence for gambling-related language markers}, `neurips_content_en/appendix.tex:321`

**What the experiment asks**
The frozen word-list used to detect gambling-style reasoning in what the models wrote: four categories and twenty-three regular expressions, all fixed before any counting was done.

**HF path(s) of the raw data**
No measured quantity — this is the instrument itself. Its frozen source is HF `rebuttal_neurips_2026/nested_baseline_and_audits_e2/src/convergent_codebook.FROZEN.py` (with the live variant beside it as `convergent_codebook.py`).

**Code that turns raw data into the printed values**
None — written inline in `appendix.tex`. The instrument is applied by HF `rebuttal_neurips_2026/nested_baseline_and_audits_e2/src/multi_instrument_robustness.py`.

**Corpus vintage**
Not a corpus float. The vintage question here is *instrument* vintage, not data vintage: a FROZEN and a non-frozen copy of the codebook both ship, and only the FROZEN one is the pre-registered instrument. `archive/rebuttal_20260731/VERIFIED_FACTS.md` §Z.3 prints the published lexicon in full and notes two things about it. The posted rebuttal (`posted_openreview/README.md`, promise 2) commits to 4 constructs and 30 expressions with Goodie & Fortune (2013) and GRCS backing; the paper's table prints 23 expressions. That count difference is a live open promise, not a defect in this table.

**Reproduction status**
NOT CHECKED — no numbers to reproduce. The 23-vs-30 expression count against the posted promise is flagged for the authors, not resolved here.
