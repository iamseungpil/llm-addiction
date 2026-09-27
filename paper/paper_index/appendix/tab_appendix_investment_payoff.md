# Table — `tab:appendix-investment-payoff`

**Paper location**
Appendix \subsection{Prompt structure of the investment-choice experiment}, `neurips_content_en/appendix.tex:90`

**What the experiment asks**
The exact odds of the four investment options, so a reader can check that three of them lose the same amount on average and differ only in how wildly they swing.

**HF path(s) of the raw data**
Derived from the payoff text printed in the round prompts themselves: 7,799 open-weight prompts under `behavioral/investment_choice/v2_role_{gemma,llama}/`, and 22,398 API prompts under `investment_choice/bet_constraint/results/`.

**Code that turns raw data into the printed values**
`scripts/tables/appendix_behavioural_tables.py` -> `paper_data/tables/appendix/investment_payoff.tex`. Every moment is derived from the recovered payoff spec, not typed in.

**Corpus vintage**
Canonical, with a real design split the paper's single table does not show. Two distinct payoff specs exist in the release:
  - corrected-design (open-weight) corpus: option 3 pays 3.60x at p=0.25 -> E = -0.10
  - legacy closed-model batch: option 3 pays 3.20x at p=0.25 -> E = **-0.20**
The paper prints the corrected-design spec. The API models were therefore played against a slightly worse mid-variance option than the open-weight models. Both specs are recorded in the fragment header. No `DEPRECATION_WARNING.md` applies to either path.

**Reproduction status**
VERIFIED, 2026-08-25. 24 numeric values; max deviation 0.000 against the corrected-design spec recovered from the open-weight prompts.
