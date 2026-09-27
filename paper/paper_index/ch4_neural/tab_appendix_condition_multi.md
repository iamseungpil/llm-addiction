# Table — `tab:appendix-condition-multi`

**Paper location**
Appendix \subsection{Autonomy modulation in the investment-choice and mystery-wheel cells}, `neurips_content_en/appendix.tex:761`

**What the experiment asks**
Does giving a model its own profit target make its risky behaviour easier to read off its internals — and does that hold in the other two games, not just the slot machine?

**HF path(s) of the raw data**
`sae_v3_analysis/results/condition_modulation_groupkfold_L22.json`; features `sae_features_v3/{investment_choice,mystery_wheel}/{gemma,llama}/` at L22.

**Code that turns raw data into the printed values**
`scripts/tables/appendix_neural_tables.py` -> `paper_data/tables/appendix/A07_condition_multi.tex`.

**Corpus vintage**
Canonical: every cell at the body's representative layer L22 under GroupKFold-by-game. No `DEPRECATION_WARNING.md` applies.

The `n/a` cells are not missing data: the regeneration reproduces the paper's own double-dagger rule — a subset R^2 below -1.0 (unstable Ridge on a small subset) is printed as n/a. The two affected cells are LLaMA IC +M, whose fitted R^2 are -16.38 and -44.15 with fold SDs of 37 and 100.

**Reproduction status**
VERIFIED, 2026-08-25. 34 numeric values; max deviation 0.000.
