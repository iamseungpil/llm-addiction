# shared — code used by more than one experiment

`common/` is the harness of the open-weight task runners: `ModelLoader` (loads LLaMA-3.1-8B,
Gemma-2-9B or Qwen in bf16), `PromptBuilder`, logging, JSON helpers, `set_random_seed` and
`clear_gpu_memory`. It is imported as `from common import ...` by

- [`02_investment_choice/open_weight/investment_choice/`](../02_investment_choice/README.md), and
- [`06_mystery_wheel/`](../06_mystery_wheel/README.md) (`mystery_wheel/run_experiment.py`,
  `run_mw_vllm.py`, `run_mw_parallel.py`).

Both runners put `experiments/shared` on `sys.path` themselves. To import it elsewhere:

```bash
PYTHONPATH=experiments/shared:experiments/02_investment_choice/open_weight:experiments/06_mystery_wheel python ...
```

`common/phase1_feature_extraction.py` and `common/phase2_correlation_analysis.py` are an earlier
SAE pipeline for these tasks; the paper's readout is in
[`07_sae_readout/`](../07_sae_readout/README.md).

This folder used to be `exploratory_experiments/alternative_paradigms/src/common/`. The archived
paradigms that still `import common` (blackjack, coin flip, ...) reach it through a small forwarding
module at `archive/exploratory_experiments/alternative_paradigms/src/common/__init__.py`.
