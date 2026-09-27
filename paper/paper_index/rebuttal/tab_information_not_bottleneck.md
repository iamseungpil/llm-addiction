# Table — `tab:information-not-bottleneck`

**Paper location**
Appendix \subsection{Stating the odds in the prompt does not reduce ruin},
`neurips_content_en/appendix.tex:1383`

**What the experiment asks**
If the prompt hands the model the payout and the win rate — everything it needs to work out that
the game loses money — does it go broke less often?

**HF path(s) of the raw data**
The canonical four of the six slot-machine corpora, per `NEURIPS_CANONICAL_INDEX.md` §3, each
identified by the `model` field inside the file rather than by its folder:
  - GPT-4o-mini      `analysis/gpt_results_fixed_parsing/gpt_fixed_parsing_complete_20250919_151240.json`  (`model` = `gpt-4o-mini-corrected`)
  - Claude-3.5-Haiku `slot_machine/claude/claude_experiment_corrected_20250925.json`
  - Gemma-2-9B       `behavioral/slot_machine/gemma_v4_role/final_gemma_20260227_002507.json`
  - LLaMA-3.1-8B     `behavioral/slot_machine/llama_v4_role/final_llama_20260315_062428.json`
3,200 games per model; the table reads the 1,600 variable-arm games of each.

**Code that turns raw data into the printed values**
`scripts/tables/rebuttal_info_bottleneck.py`
-> `paper_data/tables/appendix/info_not_bottleneck.tex` and `rebuttal_info_bottleneck.json`
(repo-only; not in the HF camera-ready upload). The JSON carries a `printed_value_audit` block
that tests the printed digits against every candidate corpus.

**Corpus vintage**
**Canonical — corrected.** This is the float this index was built to catch, and it has since been
fixed in the paper. The state now printed reads from the four canonical corpora above, and the
corrected cells are wrapped in `\rev{}`.

What was wrong, kept here because the failure mode is the reason the corpus-vintage field exists:
three of four model rows and the entire pooled panel had been computed from corpora the dataset
itself marks do-not-cite. The GPT-4o-mini row was actually **GPT-4.1-mini**, taken from
`slot_machine/gpt/` — a folder named for a model it does not contain. The Gemma and LLaMA rows
came from `slot_machine/gemma/` and `slot_machine/llama/`, the two V1 October-2025 trees whose
`DEPRECATION_WARNING.md` files read "CORRUPTED, must not be used" and "MILD CORRUPTION". Every
printed digit reproduced exactly on *some* released corpus; the corpora were the wrong ones.

The size of that error was not cosmetic. The superseded cells read GPT-4o-mini 18.8 / 2.2 (+16.6),
Gemma 49.2 / 22.3 (+26.9), LLaMA 7.8 / 6.4 (+1.3) against the canonical 25.2 / 20.0 (+5.2),
7.5 / 4.8 (+2.8), 79.0 / 70.1 (+8.9) — a 41.7-point gap on the Gemma "both W and P" cell alone —
and the pooled module-count differences reversed sign, from +9.9 / +7.2 to −5.6 / −8.3.

The claim's direction survived the correction for all four models: ruin in the computable cells is
still at least as high everywhere. The pooled panel did not survive it, and the surrounding prose
now states the reversal outright — that once module count is held fixed the two odds modules go
with *less* ruin, at −5.6, −8.3 and −4.8 points.

**Reproduction status**
VERIFIED, 2026-08-27. All four model rows and all three pooled rows — 28 numeric cells counting
interval bounds, plus the seven $n$ values — match
`paper_data/tables/appendix/info_not_bottleneck.tex` exactly; **max deviation 0.0 percentage
points**. The three module-count figures quoted in the surrounding prose (−5.6, −8.3, −4.8) and
the four per-model differences (+5.2, +15.7, +2.8, +8.9) match the same fragment.

The fragment is no longer unwired: what was a PROPOSED correction sitting in `paper_data/` is now
the state of the printed table.
