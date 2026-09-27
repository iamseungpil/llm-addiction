# Figure — `fig:escalation`

**Paper location**
Appendix \subsection{Further robustness and trajectory tests},
`neurips_content_en/appendix.tex:276-278`; graphic `images/escalation_trajectory.pdf`

**What the experiment asks**
Within a single game, does a model that can choose its bet keep raising the share of its remaining
money it risks as the game goes on?

**HF path(s) of the raw data**
Derived values: `sae_v3_analysis/results/escalation/escalation_results.json`.
Raw corpus, read by the generator: `behavioral/slot_machine/llama_v4_role/final_llama_20260315_062428.json`
— the **canonical** V4role LLaMA slot-machine run.

**Code that turns raw data into the printed values**
HF `paper_neurips_2026/figures/appendix/figA03_escalation/code/run_escalation_analysis.py`
(analysis; writes `escalation_results.json`).
Repo redraw at print size: `scripts/figures/figA03_escalation_trajectory.py` with sidecar
`scripts/figures/data/figA03_escalation_trajectory.json`.

**Corpus vintage**
**Canonical.** `run_escalation_analysis.py` opens exactly one file, and it is the V4role LLaMA
corpus, not the V1 `slot_machine/llama/` copy that carries `DEPRECATION_WARNING.md`.

The figA03 README's "Raw: `slot_machine/{gemma,llama,...}/final_*.json`" line is boilerplate and
points at the deprecated directory; the executable code does not. **Where a README and a generator
disagree, the generator decides** — the same rule this index applies to folder names versus the
`model` field inside a file.

**Reproduction status**
VERIFIED, 2026-08-27. Both printed means come straight out of `escalation_results.json`:
`h1_var_mean_rho` = 0.4433381 against the printed 0.443, and `h2_fix_mean_rho` = 0.1136209
against the printed 0.114. **Max deviation 0.0004**, i.e. the printed three-decimal roundings are
exact. The caption's permutation result is `h2_perm_p` = 0.0 over 10,000 permutations, which
supports $p < 0.001$.

**Correction to an earlier reading recorded here.** This manifest previously called the figure
UNREPRODUCIBLE on the ground that the printed means reproduced from neither candidate corpus.
That was wrong, and the reason is a caption defect worth carrying:

> the caption calls 0.443 the "mean $\bar r$" of the "bet-to-balance ratio $r_t$". It is not a
> ratio. `run_escalation_analysis.py` computes, per game, the **Spearman correlation** between the
> normalised round index and the bet/balance ratio (`spearmanr(rounds, ratios)`, games with at
> least three wagering rounds), and averages that correlation over games. 0.443 is
> $\bar\rho$, not $\bar r$.

Any recompute of a mean *ratio* therefore misses the printed number by construction, on canonical
and deprecated corpora alike. The artwork's own legend prints $\bar\rho$ while the caption writes
$r_t$ and $\bar r$, so the figure and its caption disagree with each other. The arm sizes are
$n = 1{,}510$ variable and $n = 1{,}405$ fixed games, which are the games clearing the
three-round minimum, not the participation counts.
