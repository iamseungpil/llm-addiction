# 02 — Investment choice

**Question.** Does asking a model to set its own goal change how much risk it takes, and does the
model keep raising that goal during play?

**Paper result.** Findings 3–4, Figure 3a–c (bankruptcy, risky-option share, moving-target rate by
goal condition); appendix table `tab:appendix-investment-comprehensive`. An appendix figure of
option distributions under chain-of-thought (`investment_choice_distributions_cot.pdf`) comes from
the separate 100-round API run in `appendix_extended_cot/`.

## Design (from the code)

- A $100 bankroll; each round the model picks one of four options: stop and keep the stake, a 50%
  chance of 1.8×, a 25% chance of a middle multiplier, or a 10% chance of 9×. Every gamble has
  expected value 0.9× the stake.
- **API runner** (`api_bet_constraint/src/investment_game.py`): middle option 3.2×, at most
  **10 rounds**; bet caps $10/$30/$50/$70; fixed bet = the cap, variable bet = $1 up to
  min(cap, balance); prompts `BASE`, `G`, `M`, `GM`; 25 games per cell; GPT-4o-mini, GPT-4.1-mini,
  Claude-3.5-Haiku, Gemini-2.5-Flash (4 × 4 caps × 2 modes × 4 prompts × 25 = 3,200 games).
- **Open-weight runner** (`open_weight/investment_choice/game_logic.py`): middle option 3.6×, at
  most **100 rounds**, role sentence prepended; caps $10/$30/$50/$70; prompts `BASE`, `G`, `M`,
  `GM`; 50 games per cell (1,600 per model); LLaMA-3.1-8B and Gemma-2-9B.
- Figure 3a–c pools the 10-round API games with the 100-round open-weight games; the paper states
  this. The moving-target rate is read from free text in the API games and from the recorded goal
  in the open-weight games.
- `G` asks the model to set a target amount itself; `M` asks it to maximise reward.

## Quick Start

```bash
# API models (keys: OPENAI_API_KEY or GPT_API_KEY, CLAUDE_API_KEY or ANTHROPIC_API_KEY, GEMINI_API_KEY)
python experiments/02_investment_choice/api_bet_constraint/src/run_all_experiments.py \
    --model gpt4o --bet_constraint 10 --bet_type fixed --trials 25
python experiments/02_investment_choice/api_bet_constraint/src/run_all_experiments.py \
    --model all --bet_constraint all --bet_type both

# Open-weight models (one GPU); the paper's LLaMA run looped this over caps 10/30/50/70,
# see experiments/07_sae_readout/scripts/run_llama_ic_v2role.sh
python experiments/02_investment_choice/open_weight/investment_choice/run_experiment.py \
    --model llama --gpu 0 --constraint 30 --output-dir out/ic_llama
```

The API runner writes to `/data/llm_addiction/investment_choice_bet_constraint/` (set in
`api_bet_constraint/src/base_experiment.py`); the open-weight runner writes to `--output-dir`.

## Folder contents

| Path | What it is | HF folder |
|---|---|---|
| `api_bet_constraint/` | **The paper's API runner** (3.2× middle option, 10 rounds, caps) | `investment_choice/bet_constraint/` |
| `open_weight/investment_choice/` | **The paper's open-weight runner**; imports `common` from `experiments/shared/` | `behavioral/investment_choice/v2_role_{llama,gemma}/` |
| `appendix_extended_cot/` | Appendix only: 100-round API run with step-by-step reasoning and goal tracking (3.6× middle option) | `investment_choice/extended_cot/` |
| `appendix_extended_cot_legacy_copy/` | An earlier copy of the same runner; differs only in docstrings and log labels | `investment_choice/extended_cot/` |

Not used by the paper and kept in `archive/`: the first 10-round version
(`archive/paper_experiments/investment_choice_experiment/`, HF `investment_choice/initial/`) and the
CoT variant of the bet-constraint runner (`archive/legacy/investment_choice_bet_constraint_cot/`, HF
`investment_choice/bet_constraint_cot/`, 29 of 32 cells). That archived folder also holds
`analysis/create_choice_distribution_cot.py`, the ancestor of the appendix CoT-distribution figure;
it reads the extended-CoT results.

## Figures and tables (paper repository, private)

- `scripts/figures/fig03_investment_choice_1x4.py` — Figure 3 (a–c from these corpora, via
  `scripts/figures/fig03_investment_choice.py`; panel d is the matched cap, see
  [03](../03_matched_cap/README.md)).
- `scripts/tables/appendix_behavioural_tables.py` — `tab:appendix-investment-comprehensive`.
- `scripts/figures/figA09_investment_choice_distributions_cot.py` and
  `scripts/figures/recovered/create_choice_distribution_cot.py` — the appendix CoT figure (the
  camera-ready keeps the submitted artwork).
