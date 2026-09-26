# 01 — Slot machine, six models

**Question.** Does letting a model choose its own bet size raise bankruptcy, and do its bets
escalate after runs of wins or losses?

**Paper result.** Findings 1–2, Figure 2 (a: bankruptcy, b: irrationality indicators, c–d: bet
increase after win and loss streaks); appendix table `tab:appendix-slot-comprehensive`.

## Design (from the code)

- A $100 bankroll; each round the model bets or stops. Win probability 0.30, payout 3.0× the bet
  (expected value −10%), at most 100 rounds (`src/llama_gemma_experiment.py`, `SlotMachineGame`).
- Two betting modes: **fixed** ($10 every round) and **variable** (the model names any bet from $5
  up to its balance).
- Five prompt modules crossed in all 32 combinations: goal-setting `G`, reward maximisation `M`,
  hidden patterns (`H` in the LLaMA files, `R` in the other five), win-reward information `W`,
  probability information `P`.
- 64 conditions × 50 games = 3,200 games per model, 19,200 in total across GPT-4o-mini,
  GPT-4.1-mini, Gemini-2.5-Flash, Claude-3.5-Haiku, LLaMA-3.1-8B and Gemma-2-9B.
- The open-weight runner prepends a role sentence ("participant in a behavioral economics
  simulation"); the API runners use a system prompt and no role sentence.

## Quick Start

Run from the repository root. API keys come from the environment (see the table in the
[top-level README](../../README.md#setup)).

```bash
# Open-weight models (one GPU each)
python experiments/01_slot_machine/src/llama_gemma_experiment.py --model llama --gpu 0
python experiments/01_slot_machine/src/llama_gemma_experiment.py --model gemma --gpu 0

# API models (the default model of each runner is shown; override with GPT5_MODEL, CLAUDE_MODEL, GEMINI_MODEL)
python experiments/01_slot_machine/src/run_gpt5_experiment.py --quick-test      # gpt-4.1-mini
python experiments/01_slot_machine/src/run_claude_experiment.py --quick-test    # claude-3-5-haiku-latest
python experiments/01_slot_machine/src/run_gemini_experiment.py --quick-test    # gemini-2.5-flash
```

Drop `--quick-test` for the full 3,200 games. Outputs go to the absolute folders set in each
runner's `__init__` (`/data/llm_addiction/...` or `/home/jovyan/...`); change `results_dir` before
running elsewhere.

## Folder contents

| Path | What it is |
|---|---|
| `src/llama_gemma_experiment.py` | LLaMA / Gemma runner with the role sentence; produced the `*_v4_role` corpora (Figure 2a–b) |
| `src/run_gpt5_experiment.py`, `src/run_claude_experiment.py`, `src/run_gemini_experiment.py` | Later copies of the three API runners in `original_api_runners/`. They play the same game with the same parser; they label the hidden-pattern module `H` where the originals (and the released files) write `R`, and the GPT copy has clearer names and messages |
| `original_api_runners/` | The runners that actually produced the released API exports: `gpt_experiments/src/gpt_fixed_parsing_experiment.py` (GPT-4o-mini), `gpt5_experiment/run_gpt5_experiment.py` (GPT-4.1-mini), `claude_experiment/run_claude_experiment.py`, `gemini_experiment/run_gemini_experiment.py`, each with the analysis scripts of the time |
| `experiment_0_llama_gemma_restart/` | The October 2025 LLaMA / Gemma runs without the role sentence (`launch.sh` → `experiment_0_restart.py`); Figure 2(c,d) streak panels read these |
| `data/results` | Symlink to the data folder on the original machine (dangling elsewhere) |

`gpt_fixed_parsing_experiment.py` imports `improved_gpt_parsing` from `/home/ubuntu/llm_addiction`,
which is not in the repository; a copy of that parser is at
`experiments/03_matched_cap/sm_cap_ablation/src/improved_gpt_parsing.py`.

## Data on Hugging Face

| Model | Folder |
|---|---|
| LLaMA-3.1-8B, Gemma-2-9B (role sentence; Figure 2a–b) | `behavioral/slot_machine/{llama,gemma}_v4_role/` |
| LLaMA-3.1-8B, Gemma-2-9B (October 2025; Figure 2c–d) | `slot_machine/{llama,gemma}/` |
| GPT-4.1-mini | `slot_machine/gpt/` (the folder name says gpt; the files are GPT-4.1-mini) |
| Claude-3.5-Haiku, Gemini-2.5-Flash | `slot_machine/{claude,gemini}/` |
| GPT-4o-mini | `analysis/gpt_results_fixed_parsing/` |

## Figures and tables (paper repository, private)

- `scripts/figures/fig02_slot_machine.py` — Figure 2a–b.
- `scripts/figures/fig02_cd_streaks.py` — Figure 2c–d, recomputed from the six corpora above.
- `scripts/tables/appendix_behavioural_tables.py` — `tab:appendix-slot-comprehensive`.
- `paper_data/tables/appendix/code/gbsa_companion_metrics.py` — `tab:companion-metrics`
  (participation, realised wager and first-loss re-betting for the primary cells).
