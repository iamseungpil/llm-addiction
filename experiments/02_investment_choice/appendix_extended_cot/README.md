# investment_choice_extended_cot — investment choice for the API models

Part of the investment-choice experiment behind **Findings 3–4** (Figure 3a–c). For the
whole-repository map, see the [top-level README](../../README.md).

## Question

Does asking a model to set its own goal change how much risk it takes? Each round the model picks a
safe exit or one of three losing gambles, under a bet constraint ($10/$30/$50/$70 or unlimited) and
a fixed or variable bet, with prompts crossing goal-setting (`G`) and reward maximisation (`M`).
This version runs up to 100 rounds and asks for step-by-step reasoning with goal tracking.

The open-weight models (LLaMA, Gemma) were run with a different runner,
`exploratory_experiments/alternative_paradigms/src/investment_choice/run_experiment.py`.

## Entry scripts

`src/run_experiment.py` runs one model, one constraint and one bet type. `src/run_all_experiments.py`
loops over them (`--model all`, `--constraint all`, `--bet_type both`).

```bash
python paper_experiments/investment_choice_extended_cot/src/run_experiment.py \
    --model gpt4o --constraint 30 --bet_type variable --trials 50
```

`--model` is `gpt4o`, `gpt41`, `claude` or `gemini`; `--resume <checkpoint.json>` continues a run.

API keys come from the environment: `OPENAI_API_KEY` or `GPT_API_KEY`, `ANTHROPIC_API_KEY`,
`GEMINI_API_KEY`. `run_all_experiments.py` also reads a `.env` file in this folder.

The shell scripts in this folder (`launch_all_conditions.sh`, `run_remaining_sequential.sh`,
`auto_monitor.sh`, `monitor_and_continue.sh`, `start_next_wave.sh`) are the launch and monitoring
wrappers used on the original machine.

## Outputs

Results, logs and checkpoints are written under `/data/llm_addiction/investment_choice_extended_cot/`
(set in `src/base_experiment.py`). `analysis/` holds figures and LaTeX tables made from an earlier
pass over these results.

## Data on Hugging Face

`investment_choice/extended_cot/`. The investment-choice files that the top-level README lists for
Figure 3 are `investment_choice/bet_constraint/`, `investment_choice/bet_constraint_cot/` and
`behavioral/investment_choice/v2_role_{llama,gemma}/`; the dataset's `MANIFEST.md` gives the file
behind each panel.
