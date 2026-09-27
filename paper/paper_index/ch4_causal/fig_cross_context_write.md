# Figure — `fig:cross-context-write`

**Paper location**
Appendix \subsection{Cross-task transfer and condition writability}, `neurips_content_en/appendix.tex:921-926`; graphic `images/fig_xctx_ladders_solo.pdf` (half-width, `0.49\textwidth`).

**Now a single graphic, not two.** Earlier drafts carried a two-panel float — the sign-transfer
heatmap (`images/fig_xctx_signmap.pdf`) beside the condition-writability ladders
(`images/fig_xctx_ladders.pdf`). The heatmap panel was cut from this figure; its twelve cells are
now reported only as the table `tab:causal-transfer-matrix`. What remains is the ladders panel
alone, redrawn as `fig_xctx_ladders_solo.pdf` (no panel letter, LLaMA comparison moved to an inset).
Neither `fig_xctx_signmap.pdf` nor the non-solo `fig_xctx_ladders.pdf` is `\includegraphics`'d
anywhere in `neurips_content_en/` any more.

**What the experiment asks**
Under a goal-grafted, reward-grafted, or no-goal Gemma prompt (with a LLaMA comparison inset), how
much does betting rise as the causal push on the behavioural axis gets stronger?

**HF path(s) of the raw data**
`experiments/sec4_causal/checkpoints/sec4_w14/*.jsonl`.

**Code that turns raw data into the printed values**
`scripts/figures/fig_cross_context_write.py` — the current generator, which is the one that emits
the `_solo` filename this float now includes (`fig_ladders(panel_letter=None,
outname="fig_xctx_ladders_solo.pdf")`). It also still writes `fig_xctx_signmap.pdf` and
`fig_axis_alignment.pdf` as a side effect of one shared script, but only the ladders-solo file is
wired into a `\includegraphics`. The older `scripts/gen_fig_cross_context_write.py` is its
predecessor: it writes the two-panel `fig_xctx_signmap.pdf` + `fig_xctx_ladders.pdf` pair that this
float no longer uses (see *Orphan outputs* below) — do not cite it for this float.

**Orphan outputs, not a data problem.** `scripts/gen_fig_cross_context_write.py` and, as a
byproduct, `scripts/figures/fig_cross_context_write.py` itself still produce `fig_xctx_signmap.pdf`
and (the older script only) the non-solo `fig_xctx_ladders.pdf`. Both files exist under `images/`
but nothing in `neurips_content_en/` includes them any more — the heatmap panel they drew is now the
table `tab:causal-transfer-matrix` instead. Left in place rather than deleted, per this pass's brief.

**Corpus vintage**
Canonical W14 (the ladders panel's own data). No `DEPRECATION_WARNING.md` applies. The heatmap data
this float used to draw on (W7) is unaffected by the panel's removal — see `tab_causal_transfer_matrix.md`
for its vintage note (DEFECT 3, now recomputed from the raw `.jsonl`).

**Reproduction status**
NOT CHECKED (ladders panel, figure as drawn). The heatmap numbers this float previously carried are
still VERIFIED, but as the table `tab:causal-transfer-matrix`, not as part of this figure.
