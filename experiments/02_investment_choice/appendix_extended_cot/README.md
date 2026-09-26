# Appendix: extended-CoT investment choice (API models)

**Appendix only.** This 100-round run with step-by-step reasoning feeds the appendix
CoT-distribution figure (`investment_choice_distributions_cot.pdf`). Figure 3a–c uses the 10-round
API runner in [`../api_bet_constraint/`](../api_bet_constraint/) and the open-weight runner; see the
[experiment page](../README.md) and the [top-level README](../../../README.md).

## Question

Does asking a model to set its own goal change how much risk it takes? Each round the model picks a
safe exit or one of three losing gambles, under a bet constraint ($10/$30/$50/$70 or unlimited) and
a fixed or variable bet, with prompts crossing goal-setting (`G`) and reward maximisation (`M`).
This version runs up to 100 rounds and asks for step-by-step reasoning with goal tracking.

The open-weight models (LLaMA, Gemma) were run with a different runner,
`experiments/02_investment_choice/open_weight/investment_choice/run_experiment.py`.

## Entry scripts

`src/run_experiment.py` runs one model, one constraint and one bet type. `src/run_all_experiments.py`
loops over them (`--model all`, `--constraint all`, `--bet_type both`).

```bash
python experiments/02_investment_choice/appendix_extended_cot/src/run_experiment.py \
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

`investment_choice/extended_cot/`. Figure 3a–c reads `investment_choice/bet_constraint/` and
`behavioral/investment_choice/v2_role_{llama,gemma}/` instead; `investment_choice/bet_constraint_cot/`
is not used by the paper.
