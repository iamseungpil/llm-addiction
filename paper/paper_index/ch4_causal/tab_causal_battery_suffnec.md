# Table (Table 23) — `tab:causal-battery-suffnec`

**Paper location**
Appendix \subsection{Sufficiency and necessity}, `neurips_content_en/appendix.tex:853`; cited from §4.
Genuinely an appendix float — the body only cross-references it. The compiled `neurips_en.aux` gives
its printed number: `\newlabel{tab:causal-battery-suffnec}{{23}{33}{...}{table.118}{}}` — **Table 23**,
page 33. An earlier pass of this manifest (and the generator's own output path,
`paper_data/tables/body/table2_causal_suffnec.tex`, named from an older draft where this table sat
in the body) called it "body Table 2"; that number belongs to a draft state this paper no longer has
— the current Table 2 is the sharing audit, `tab:rq2-sharing` (see `ch4_neural/tab_rq2_sharing.md`).

**What the experiment asks**
Two demands on a causal claim: adding the direction must raise betting, and taking it away must lower it. This table reports whether each of three candidate directions passes both.

**HF path(s) of the raw data**
Sufficiency: `experiments/sec4_causal/checkpoints/sec4_p0/sec4_{behavioural,readout,confound}_a*.jsonl` and `sec4_w2/sec4_w2_null{1..20}_a{m3,p3}.jsonl` (Gemma); `sec4_w10*` (LLaMA).
Necessity: `experiments/sec4_causal/checkpoints/sec4_w13/sec4_w13_*_{base,behavioural,readout,confound}.jsonl` and `*_summary.json`.
Cross-check: `experiments/sec4_causal/analysis/sec4_w2_analysis.json`.

**Code that turns raw data into the printed values**
`scripts/tables/body_tables.py` -> `paper_data/tables/body/table2_causal_suffnec.tex`. Wave logs: `multilayer_causal/experiments/sec4_causal/INDEX.md` W2/W10/W13.

**Corpus vintage**
Canonical wave checkpoints. In the submitted paper the numbers were hand-transcribed from wave summary JSONs; the fragment recomputes them from the raw `.jsonl` rollouts. No `DEPRECATION_WARNING.md` applies.

One statistical defect the regeneration surfaces: the p column typed into the .tex equals min(1, 2 x p) where p is already an exact two-sided sign test — a second doubling applied on top of a two-sided test, in all six cells.

**Reproduction status**
VERIFIED, 2026-08-25. 20 of the 21 numeric tokens in the printed tabular body match the regenerated fragment exactly (max deviation 0.000); the 21st is a footnote marker the fragment does not emit. `NEURIPS_CANONICAL_INDEX.md` separately records a 3-agent audit on 2026-07-10 that checked these numbers against the wave source files.
