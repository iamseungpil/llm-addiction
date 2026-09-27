# Figure — `fig:investment-choice`

**Paper location**
§3 \subsection{Follow-up checks: scope of goal and bet-size effects}, `neurips_content_en/3.behavior.tex:33-35`; graphic `images/investment_choice3.pdf`

**What the experiment asks**
In the four-option investment game, does telling a model to chase a profit target push it toward the riskiest option and toward ruin — and when the maximum bet is held equal in both arms, does freedom to choose the bet still hurt?

**HF path(s) of the raw data**
Panels (a)-(c), investment choice, 9,600 games:
  - API 4 models: `investment_choice/bet_constraint/results/` only (33 files for 32 cells, see vintage). Mid option pays 3.2x, 10 rounds. `investment_choice/bet_constraint_cot/` (3.6x, 29 of 32 cells) is NOT read by the generator.
  - LLaMA-3.1-8B: `behavioral/investment_choice/v2_role_llama/llama_investment_c{10,30,50,70}_*.json`
  - Gemma-2-9B:   `behavioral/investment_choice/v2_role_gemma/gemma_investment_c{10,30,50,70}_*.json`
Panel (d), the matched-cap control on three API models (GPT-4o-mini, GPT-4.1-mini, Gemini-2.5-Flash; five-module prompt with the role sentence, 50 games per model and cell):
  - `rebuttal_neurips_2026/matched_cap_mc32/final_*.json` (64 top-level files; subfolders QUARANTINE*, LEGACY_PARSER*, TRUNCATED* are not read)
Finding 4's first run (GPT-4o-mini, 32 prompt conditions, 1,600 games per arm and cap; body text only, not plotted):
  - variable arm: the union of `analysis/fixed_variable_comparison/gpt_variable_max_bet_results/restart_complete_10_30_20251019_171400.json`, `..._50_70_20251019_162438.json`, `intermediate_20251016_073043.json` and `intermediate_20251016_075750.json`. The restart files skipped the cells the first run had finished (`experiments_skipped` 600 / 650), so the two restart files alone hold only ~1,290 games per cap; the union holds 1,600 with no duplicate (cap, condition, repetition).
  - fixed arm    `analysis/fixed_variable_comparison/gpt_fixed_bet_size_results/complete_20251016_010653.json`

**Code that turns raw data into the printed values**
Panels (a)-(c): `scripts/figures/fig03_investment_choice.py` -> `paper_data/fig03_investment_choice.json`.
Panel (d): `scripts/figures/fig05_matched_cap.py` -> `paper_data/fig05_matched_cap.json` (`panel_d_body`).
Assembly of the 1x4 layout the paper actually includes: `scripts/figures/fig03_investment_choice_1x4.py`.
**Release gap:** `fig03_investment_choice_1x4.py` is repo-only; HF `paper_neurips_2026/camera_ready/` ships `fig03_investment_choice.py` and `fig05_matched_cap.py` (and a 3-panel `fig03_investment_choice.pdf` plus a standalone `fig05_matched_cap.pdf`), not the combined figure the paper prints.

**Corpus vintage**
Canonical. Two vintage hazards are documented in the generator and must not be lost:
1. The old loader read the IC corpus out of `/tmp/llmadd_hf/...`, a scratch directory, so the figure was reproducible only on a machine that still held that copy. The current script pulls from HF.
2. `investment_choice/bet_constraint/results` holds 33 files for 32 cells: `gemini_flash_70_variable` was exported twice (20251122_161926 and 20251122_162210) and the two runs disagree (719 vs 784 decisions). `DUPLICATE_POLICY = later` is now explicit and the discarded file plus the size of the swing are recorded in `paper_data/fig03_investment_choice.json` under `duplicate_cell`. The old loader resolved this silently by dict-key overwrite.

No `DEPRECATION_WARNING.md` applies to any path read here.

**Reproduction status**
NOT CHECKED (figure as drawn). The underlying per-cell numbers are the ones verified in `tab:appendix-investment-comprehensive` (max deviation 0.000 on 40 printed values, 2026-08-25).


**Finding 4 prose, GPT-4o-mini matched-cap run (1,600 games per arm and cap).** Recomputed on 2026-09-26
from the four variable files listed above (union, no duplicates) and the fixed file: variable
bankruptcy 0.6 / 14.9 / 15.9 / 18.3% at caps 10/30/50/70 (239, 254, 292 of 1,600 at $30/$50/$70);
mean rounds among played games 19.8 / 17.9 / 17.4 at $30/$50/$70; mean wager $15.1 / $17.6 / $20.2.
Fixed arm at $30/$50/$70: bankruptcy 0.0 / 4.7 / 0.4%, mean rounds over all games 1.0 / 0.7 / 0.3.
Participation (games with at least one executed wager): fixed 55 / 46 / 24%, variable 98-99%.
An earlier recompute read only the two restart files (1,309 / 1,291 / 1,265 / 1,285 games;
14.3 / 16.4 / 17.3%) and is superseded.
A game counts as played when `game_history` is non-empty.
