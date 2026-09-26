# 06 — Mystery wheel

**Question.** Do the risk indicators read from the hidden state carry over to a third gambling task
whose odds the model is never told?

**Paper result.** The third task of §4: the mystery-wheel columns of Table 1 (Finding 6) and the
cross-task sharing of Table 2 (Finding 7); appendix "Rules of the mystery-wheel task". The
behaviour itself is not a body finding; the games are the input to
[07](../07_sae_readout/README.md) and [08](../08_steering/README.md).

## Design (from the code)

- A $100 bankroll; each round the model spins or stops, at most 100 rounds
  (`mystery_wheel/game_logic.py`).
- The wheel pays 2.0× (red), 3.0× (blue) or 8.0× (gold) with hidden probabilities 25%, 8% and
  2%; otherwise the stake is lost. Expected value 0.9× per spin, as in the other tasks. The prompt
  shows zone names and payouts, never the probabilities.
- Fixed vs variable betting under a bet cap (the paper's runs use $30); variable bets start at $5.
- The same five prompt modules as the slot machine (`G`, `M`, `P`, `W`, `R`) in all 32
  combinations × 2 betting modes × 50 games = 3,200 games per model; role sentence prepended.
- Gemma-2-9B ran through `mystery_wheel/run_experiment.py`; LLaMA-3.1-8B through the vLLM launcher
  `run_mw_vllm.py` (same experiment class, same prompts and parser, only generation goes through a
  local vLLM server), with `run_mw_parallel.py` as a multi-process alternative.

## Quick Start

```bash
# Gemma or LLaMA on one GPU (transformers); --quick runs 4 prompts x 20 games
python experiments/06_mystery_wheel/mystery_wheel/run_experiment.py \
    --model gemma --gpu 0 --constraint 30 --output-dir out/mw_gemma

# LLaMA through vLLM: start the server first, then
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct --dtype bfloat16 --port 8000
python experiments/06_mystery_wheel/run_mw_vllm.py

# LLaMA in parallel shards on one GPU (MW_OUT_DIR sets the output folder)
MW_OUT_DIR=out/mw_llama python experiments/06_mystery_wheel/run_mw_parallel.py
MW_OUT_DIR=out/mw_llama python experiments/06_mystery_wheel/run_mw_parallel.py --merge
```

The runners import `common` (model loader, logging, JSON helpers) from
[`experiments/shared/common/`](../shared/README.md); each adds that folder to `sys.path` itself.
`run_mw_vllm.py` writes to the absolute `OUTPUT_DIR` set at its top.

## Data on Hugging Face

`behavioral/mystery_wheel/{llama,gemma}_v2_role/` (games); `sae_features_v3/mystery_wheel/`
(hidden states and SAE features extracted by 07).

## Figures and tables (paper repository, private)

The mystery-wheel columns of Tables 1–2 are built by `scripts/tables/body_tables.py` from the
readout results of [07](../07_sae_readout/README.md).
