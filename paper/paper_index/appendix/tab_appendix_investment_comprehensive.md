# Table — `tab:appendix-investment-comprehensive`

**Paper location**
Appendix \subsection{Comprehensive numbers across models}, `neurips_content_en/appendix.tex:179`

**What the experiment asks**
The full investment-game scoreboard: for each model, how much a profit-target prompt changed its bankruptcy rate, its taste for the riskiest option, and how often it moved its own goalposts.

**HF path(s) of the raw data**
API 4 models: `investment_choice/bet_constraint/results/` only (`bet_constraint_cot/` is not read).
LLaMA-3.1-8B: `behavioral/investment_choice/v2_role_llama/llama_investment_c{10,30,50,70}_*.json`.
Gemma-2-9B: `behavioral/investment_choice/v2_role_gemma/gemma_investment_c{10,30,50,70}_*.json`.

**Code that turns raw data into the printed values**
`scripts/tables/appendix_behavioural_tables.py` -> `paper_data/tables/appendix/investment_comprehensive.tex`. Indicator definitions imported unchanged from `generate_paper_figures.py`.

**Corpus vintage**
Canonical V2role open-weight + API `bet_constraint`. No `DEPRECATION_WARNING.md` applies.
The same 33-files-for-32-cells duplicate-export hazard described in `ch3_behaviour/fig03_investment_choice.md` applies to this corpus; the resolution is `DUPLICATE_POLICY = later`.

**Reproduction status**
VERIFIED, 2026-08-25. All 40 printed values reproduce; max deviation 0.000. High-risk share intervals are game-clustered bootstraps because the denominator is decisions, not games.
