# Figure — `fig:temperature-robustness`

**Paper location**
Appendix \subsection{Further robustness and trajectory tests},
`neurips_content_en/appendix.tex:285-287`; graphic `images/temperature_robustness.pdf`

**What the experiment asks**
Does the fixed-versus-free betting gap survive when the model's sampling randomness is turned up
and down?

**HF path(s) of the raw data**
`sae_v3_analysis/results/temperature_control/temperature_control_full_20260406_065510.json`
— `model` = `meta-llama/Llama-3.1-8B-Instruct`, `config.temperatures` = 4 values,
`config.prompts` = 4 groups, `config.bet_types` = fixed and variable, `config.n_reps` = 50, and
1,600 entries in `raw_results` (4 x 4 x 2 x 50). A smoke run
(`temperature_control_smoke_20260405_000859.json`) and the experiment driver
(`sae_v3_analysis/src/run_temperature_control.py`) sit beside it.

The generator published on HF does **not** read that JSON. It parses
`sae_v3_analysis/results/temperature_control/full_run_restart.log`, which is not in the release.

**Code that turns raw data into the printed values**
HF `paper_neurips_2026/figures/appendix/figA04_temperature_robustness/code/plot_temperature_robustness.py`
— published, but it is not the generator of the printed artwork (see below).
Repo redraw at print size: `scripts/figures/figA04_temperature_robustness.py` with sidecar
`scripts/figures/data/figA04_temperature_robustness.json`, which
`scripts/figures/README_appendix_A03_A04_A05_A09.md` records as copied verbatim from the HF run
above.

**Corpus vintage**
**Canonical, and now identified.** The released 16-cell sweep is a single LLaMA-3.1-8B-Instruct
run at 50 repetitions per cell; it is not one of the six primary slot-machine corpora, so the
`slot_machine/{gemma,llama}/` deprecation does not reach it, and no `DEPRECATION_WARNING.md` sits
on the `sae_v3_analysis/` tree. The `GMHW` spelling of the fifth module in the caption matches
`config.prompts`.

This float previously sat in this index as "raw temperature sweep not locatable in the release".
That was wrong; the file above is the sweep, and it corroborates the caption's "16 experimental
conditions (4 temperatures x 4 prompt groups: BASE, G, H, GMHW)" directly from `config`.

**Reproduction status**
UNREPRODUCIBLE as artwork; the caption's design claim is VERIFIED against `config`, 2026-08-27.

Two separate gaps, and it is worth keeping them apart:

1. The HF generator's input, `full_run_restart.log`, is not published, so that script cannot be
   run against the release.
2. The published generator draws a **different figure** from the one the paper prints. It builds
   grouped bars — four temperature pairs per prompt group, a dashed `Fixed ~ 20% (all temps)`
   baseline, and a `Fixed (t=..)/Variable (t=..)` patch legend. The printed PDF is a **line plot**
   over temperature whose own legend reads "Fixed — mean over prompts", "Variable — mean over
   prompts" and "Shaded: min–max over BASE/G/H/GMHW (both series)". Verified by extracting the
   text layer of `images/temperature_robustness.pdf` on 2026-08-27. Whatever drew the printed
   figure is not in the release.

The figure prints no numbers in its caption, so there is nothing to check cell-by-cell; what can
be checked is the design claim, and it holds.
