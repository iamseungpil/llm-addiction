# Appendix figures A3 / A4 / A5 / A9 — print-size regeneration

Four appendix figures were authored on canvases much wider than the width LaTeX
actually prints them at, so `\includegraphics` shrank them and their labels fell
well below legibility. These generators re-author each one at **scale 1.0**: the
PDF's own page width equals the width it is printed at, so a 7 pt label in the
generator is a 7 pt label on paper.

## The printed widths are *not* `\textwidth`

`shared/paper_core.tex` defines

```
\newcommand{\paperfigfull}{0.78\textwidth}
\newcommand{\paperfigmid}{0.65\textwidth}
```

and `\textwidth` is exactly **396 pt = 5.5 in**. Three of these four figures are
included with those macros, not with `\textwidth`. The placement scales below were
read out of the compiled `neurips_en.pdf` content stream (the `q sx 0 0 sy tx ty cm`
that wraps each `/Fm.. Do`), so they are measured, not assumed.

| figure | include width | printed width | old canvas | old scale | new canvas | new scale |
|---|---|---|---|---|---|---|
| `escalation_trajectory.pdf` | `\paperfigmid` = 0.65\textwidth | 257.40 pt | 483.73 x 225.33 pt | 0.532 | 3.575 x 2.30 in (257.4 x 165.6 pt) | 1.000 |
| `temperature_robustness.pdf` | `\paperfigfull` = 0.78\textwidth | 308.88 pt | 502.50 x 241.96 pt | 0.615 | 4.29 x 2.70 in (308.88 x 194.4 pt) | 1.000 |
| `distortion_multimodel_summary.pdf` | `\paperfigfull` = 0.78\textwidth | 308.87 pt | 794.34 x 353.90 pt | 0.389 | 4.29 x 3.05 in (308.88 x 219.6 pt) | 1.000 |
| `investment_choice_distributions_cot.pdf` | `0.95\textwidth` | 376.20 pt | 917.79 x 406.86 pt | 0.410 | 5.225 x 2.95 in (376.2 x 212.4 pt) | 1.000 |

Smallest printed font after regeneration: 7.6 pt (A3), 7.0 pt (A4, A5, A9).

## Generators

```
python3 scripts/figures/figA03_escalation_trajectory.py
python3 scripts/figures/figA04_temperature_robustness.py
python3 scripts/figures/figA05_distortion_multimodel_summary.py
python3 scripts/figures/figA09_investment_choice_distributions_cot.py
```

Each writes `images/<name>.pdf` (and a `.png` preview) under the existing filename,
so no `.tex` changes. `savefig` is called **without** `bbox_inches="tight"` — a tight
bbox would silently change the page size and break scale 1.0.

## Vendored style constants

The HF generator for A4 imports `paper_figure_style`, which exists in neither this
repository nor the HF release. `paper_style_appendix.py` vendors only what these
figures need, all of it read back out of the submitted PDFs' content streams:

* `COLORS["fixed"] = "#59A14F"`, `COLORS["variable"] = "#E15759"`,
  `COLORS["variable_light"] = "#E8A0A0"`, `COLORS["neutral"] = "#C7C7C7"`
* tick / spine / body-text grey `#444444`, grid grey `#DDDDDD` at 0.6 pt,
  spine width 0.8 pt with top and right hidden
* stacked-choice ramp `CHOICE_COLORS` (Option 1..4) and the `RdBu_r` heatmap cmap
* `use_paper_style` / `style_axes` / `panel_title` / `save_pdf_png` helpers

Canvas sizing is deliberately **not** vendored — that is the bug being fixed.

## Data sidecars (`data/`)

None of the four camera-ready figures has a released generator; the scripts under
`paper_neurips_2026/figures/appendix/*/code/` on HF draw earlier, differently styled
versions and depend on corpora that are not in this repository. The plotted values
were therefore recovered so that the regenerated figures carry exactly the submitted
numbers:

* `figA03_escalation_trajectory.json` — 10 bin means and SEMs per series, decoded
  from the submitted PDF's line and error-bar coordinates.
* `figA04_temperature_robustness.json` — copied verbatim from the HF run
  `sae_v3_analysis/results/temperature_control/temperature_control_full_20260406_065510.json`.
  Means and min/max recomputed from it reproduce the submitted PDF's geometry exactly.
* `figA05_distortion_multimodel_summary.json` — heatmap and bar values from the
  submitted PDF; the GPT-4o-mini row was independently re-derived by running the HF
  generator's own loader over `analysis/gpt_results_fixed_parsing/gpt_fixed_parsing_complete_20250919_151240.json`
  (+5.5 / NaN / -11.9 — an exact match).
* `figA09_investment_choice_distributions_cot.json` — 32 stacked distributions from
  the submitted PDF's rectangle geometry; every stack recovers to 100.000%.

## Content defects fixed (visual composition otherwise untouched)

* **A4** — the pink shaded band was defined nowhere. It is the min–max of the
  bankruptcy rate across the four prompt conditions (BASE, G, H, GMHW). It now has a
  legend entry saying so. It was also drawn on *both* series all along, but the Fixed
  band collapses onto the Fixed line (Fixed bankruptcy is 20% in every prompt
  condition, 18–20% at t = 1.0); both bands now carry a hairline boundary and an
  in-axes note states why the Fixed one is invisible.
* **A5** — the GPT-4o-mini x Loss chasing cell was pure white with no number. It is a
  masked NaN, not a zero: Loss chasing is scored on the post-loss decision window
  only, and the GPT-4o-mini export records no per-round outcome, so none of its 14,466
  decisions can be labelled post-loss (n = 0, `pct([]) -> nan`). The cell is now
  hatched, labelled `n/a`, and explained in a figure footnote.
