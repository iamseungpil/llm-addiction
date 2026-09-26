# scripts — shell launchers for the open-weight runs (not cited)

Shell wrappers used to launch LLaMA and Gemma runs on the original GPU machines. They record which
runner was called with which arguments; the paper does not cite them directly. The SLURM directives
at the top of several files are commented out (`[SLURM-DISABLED]`).

| Script | Launches |
|---|---|
| `run_gemma_v4_role_gpu0.sh`, `run_gemma_v4_role_gpu1.sh` | Gemma investment choice and mystery wheel (GPU 0); Gemma slot machine and coin flip (GPU 1) |
| `run_gemma_fullmode.sh`, `run_llama_fullmode.sh`, `run_gemma_c30.sh`, `run_investment_c70.sh` | Open-weight investment choice (`exploratory_experiments/alternative_paradigms/src/investment_choice/run_experiment.py`) |
| `run_gemma_50trials.sh`, `run_c10_then_c50.sh`, `run_c50_after_c10.sh`, `watch_and_launch_50trials.sh` | Sequenced investment-choice runs by bet constraint |
| `run_all_blackjack_experiments.sh`, `run_gemma_blackjack_batch.sh`, `run_missing_llama_blackjack.sh` | Blackjack runs (exploratory, not in the paper) |

Paths inside the scripts point at the original machines. See the [top-level README](../../README.md)
for the runner behind each experiment and where its data is released.
