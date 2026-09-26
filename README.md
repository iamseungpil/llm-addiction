# Can Large Language Models Develop Gambling Addiction?

Seungpil Lee, Donghyun Shin, Yoonjung Lee, Sundong Kim · Gwangju Institute of Science and Technology (GIST) · NeurIPS 2026

**[Paper](https://arxiv.org/abs/2509.22818)** · **[Project page](https://llm-addiction.pages.dev)** · **[Data](https://huggingface.co/datasets/llm-addiction-research/llm-addiction)** (gated, automatic approval)

<p align="center">
  <img src="site/assets/img/og.jpg" alt="Project page share card: a pixel robot at a slot machine" width="720">
  <br><sub>Illustration from the project page. It shows no data.</sub>
</p>

This repository holds the experiment and analysis code for the paper. Six LLMs (GPT-4o-mini,
GPT-4.1-mini, Gemini-2.5-Flash, Claude-3.5-Haiku, LLaMA-3.1-8B, Gemma-2-9B) play
negative-expected-value gambling tasks, a slot machine and an investment choice, under prompts that
either let them choose the bet size or ask them to set their own goal. We score each game with
indicators taken from clinical criteria for problem gambling (betting aggressiveness, loss chasing,
extreme bets) and with bankruptcy. On the two open-weight models we then read those indicators from
the decision-time hidden state with sparse-autoencoder (SAE) features, using a third task (a
mystery wheel) as well, and finally steer the model by adding or removing a direction in that
hidden state.

**Every paper result has one folder under [`experiments/`](experiments/)**, with its own README
and Quick Start. Code the camera-ready does not use is in [`archive/`](archive/README.md); plans and
review notes are in [`docs/`](docs/README.md). The repository was reorganised on 2026-09-26:
[`PATH_MAP.md`](PATH_MAP.md) maps every old path to its new place, and the git tag `pre-reorg`
holds the old layout.

**Code is here; data and figures are elsewhere.** Every game log, SAE feature file and steering run
is on the Hugging Face dataset. Result files are gitignored (`*.json`, `*.csv`, `*.npz`), so a
fresh clone has the code without the results.

## Key findings

- **Choosing the bet raises bankruptcy in every model.** Across all 32 prompt conditions, per-model
  bankruptcy is 0–3.1% with a fixed $10 bet and 5–72% when the model picks its bet (LLaMA-3.1-8B:
  0.4% → 72.3%). Bets also rise more after wins and after losses (Findings 1–2).
- **A self-set goal nearly doubles bankruptcy, and the goal keeps moving.** In investment choice,
  goal-setting prompts raise bankruptcy from about 19% to about 36%. Games in which the model raises
  its own target mid-game go from about 11–17% to about 48–50%. The direction is the same in all six
  models (Findings 3–4).
- **The bet effect comes from revising the bet each round, not from a larger maximum.** With the
  maximum bet matched between arms, the choosing arm still goes broke more often. On a LLaMA ladder,
  bankruptcy is 2% with a forced stake, 5% when the stake is chosen once, 85% when it can be revised
  every round, and 80% with a wider cap. Role framing and an explicit expected-value instruction
  change the size of the gap (Findings 4–5).
- **The risk indicators can be read from the decision-time hidden state.** Which indicator reads best
  depends on the task. On Gemma the tasks share a small risk subspace linked to the balance, and a
  goal prompt makes betting aggressiveness easier to read (Findings 6–8).
- **A direction built from the model's own betting moves the wager. The readout direction does
  not.** Removing the behaviour-built direction lowers betting on both models; the SAE readout
  direction stays within the band of random directions (Finding 9).

## Repository layout

```
experiments/
  01_slot_machine/            F1–F2, Fig. 2        six-model slot machine (+ original API runners, Oct-2025 open-weight runs)
  02_investment_choice/       F3–F4, Fig. 3a–c     API runner, open-weight runner, appendix extended-CoT run
  03_matched_cap/             F4, Fig. 3d          GPT-4o-mini cap ablation and three-API-model replication
  04_choice_ladder/           F5                   forced / chosen once / revised each round / wider cap
  05_framing_worked_example/  F5                   role framing × rationality instruction; worked examples
  06_mystery_wheel/           F6–F7 (third task)   hidden-odds wheel for the neural analyses
  07_sae_readout/             F6–F8, Tables 1–3    SAE readout, cross-task sharing, condition modulation
  08_steering/                F9, Fig. 4           steering and removal battery (multilayer_causal package)
  09_audits/                  appendix             game-log baseline, exposure-matched ruin, codebook audits
  shared/                     —                    `common` harness used by 02 (open-weight) and 06
archive/                      code no paper figure or table uses, old structure kept
docs/                         plans, review notes (rebuttal_review/CAMERA_READY_MAP.md), cluster job files
site/                         project page (https://llm-addiction.pages.dev)
PATH_MAP.md                   old path -> new path
requirements.txt
```

## Setup

```bash
git clone https://github.com/iamseungpil/llm-addiction.git
cd llm-addiction
pip install -r requirements.txt     # Python 3.10 or 3.11
```

- **Python 3.10 or 3.11.** The cloud jobs ran on images with Python 3.10, PyTorch 2.7–2.8 and CUDA
  12.6; the HPC runs used Python 3.11. `vllm` is needed only for
  `experiments/06_mystery_wheel/run_mw_vllm.py`.
- **Open-weight models** (`meta-llama/Llama-3.1-8B-Instruct`, `google/gemma-2-9b-it`) are gated on
  Hugging Face. Accept their licences and log in, or set `HF_TOKEN`.
- **Run commands from the repository root**, except `experiments/08_steering/`, whose package is
  imported by name: run it from that folder.
- **API keys are read from environment variables.** Never commit them; `.env` is gitignored.

| Runner | Environment variables it reads |
|---|---|
| `experiments/01_slot_machine/src/run_gpt5_experiment.py` | `GPT5_API_KEY` or `OPENAI_API_KEY`; model from `GPT5_MODEL` (default `gpt-4.1-mini`) |
| `experiments/01_slot_machine/src/run_claude_experiment.py` | `CLAUDE_API_KEY`; model from `CLAUDE_MODEL` |
| `experiments/01_slot_machine/src/run_gemini_experiment.py` | `GEMINI_API_KEY`; model from `GEMINI_MODEL` |
| `experiments/02_investment_choice/api_bet_constraint/src/models/*_runner.py` | `OPENAI_API_KEY` or `GPT_API_KEY`; `CLAUDE_API_KEY` or `ANTHROPIC_API_KEY`; `GEMINI_API_KEY` |
| `experiments/02_investment_choice/appendix_extended_cot/src/models/*_runner.py` | `OPENAI_API_KEY` or `GPT_API_KEY`; `ANTHROPIC_API_KEY`; `GEMINI_API_KEY` |
| `experiments/03_matched_cap/sm_cap_ablation/src/*.py` | `GPT_API_KEY` or `OPENAI_API_KEY` |
| `experiments/03_matched_cap/track0_w3_replication/src/run_track0_api.py` (also used by `05`) | `OPENAI_API_KEY`; `ANTHROPIC_API_KEY` or `CLAUDE_API_KEY`; `GOOGLE_API_KEY` or `GEMINI_API_KEY` |
| `experiments/03_matched_cap/track0_w3_replication/src/run_mc_ladder.sh` | Sources `$HOME/.env`, then stops unless `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` and `GOOGLE_API_KEY` are all set |
| `experiments/08_steering/multilayer_causal/` | `HF_TOKEN` (model download and checkpoint sync; see below) |

The three slot-machine API runners and `appendix_extended_cot/src/run_all_experiments.py` also
read a `.env` file next to the script (or, for the last one, in the folder above `src/`).
Variables already set in the shell take precedence.

Things to know before running anything:

- **Most scripts write to absolute paths on the machines the experiments ran on**
  (`/data/llm_addiction/...`, `/home/v-seungplee/data/...`, `/home/jovyan/...`). Treat a script
  as a record of what was run, and point its inputs and outputs at your own copy of the dataset
  (most runners take `--output_dir` / `--output-dir`; the rest set `results_dir` in `__init__`).
- **`experiments/08_steering/` syncs checkpoints to the dataset.** When `HF_TOKEN` is set, each arm
  first tries to resume from the released checkpoint and then uploads its progress to
  `llm-addiction-research/llm-addiction`. With a read-only token the uploads fail and are logged;
  `--smoke` turns the sync off.
- **API results will not reproduce bit for bit.** The Claude-3.5-Haiku checkpoint has been withdrawn
  from its API. The stored responses on the dataset are the record.

One check runs anywhere, with no dataset and no GPU:

```bash
python3 experiments/07_sae_readout/tests/test_distortion_outcome_schema.py
```

## Getting the data

The dataset is gated with automatic approval. Accept the conditions on the
[dataset page](https://huggingface.co/datasets/llm-addiction-research/llm-addiction), log in with
`huggingface-cli login`, then download only the folders you need. `data/` is gitignored:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "llm-addiction-research/llm-addiction",
    repo_type="dataset",
    local_dir="data",
    allow_patterns=["behavioral/*", "paper_neurips_2026/camera_ready/paper_index/*"],
)
```

The full dataset is large: `sae_features_v3/` alone holds hidden-state files of tens of GB. The
dataset's `paper_neurips_2026/camera_ready/paper_index/` holds one manifest per figure and table,
naming its files. Dataset folder names predate this reorganisation (for example
`sae_v3_analysis/results/` holds the outputs of `experiments/07_sae_readout/`).

## Experiments

One row per experiment. "Paper" gives the camera-ready Finding and the figure or table, including
appendix table labels (`tab:…`). Code paths are relative to the repository root; data paths are
relative to the root of the Hugging Face dataset.

| # | Question | Paper | Code folder | Entry scripts | Data on HF |
|---|---|---|---|---|---|
| [01](experiments/01_slot_machine/) | Does choosing the bet size raise bankruptcy? | F1–F2, Fig. 2; `tab:appendix-slot-comprehensive` | `experiments/01_slot_machine/` | `src/llama_gemma_experiment.py` (LLaMA, Gemma); API exports from `original_api_runners/{gpt_experiments,gpt5_experiment,claude_experiment,gemini_experiment}/`; Fig. 2c–d open-weight runs from `experiment_0_llama_gemma_restart/` | `behavioral/slot_machine/{llama,gemma}_v4_role/`; `slot_machine/{llama,gemma}/`; `slot_machine/{gpt,claude,gemini}/`; `analysis/gpt_results_fixed_parsing/` (GPT-4o-mini) |
| [02](experiments/02_investment_choice/) | Does a self-set goal change risk preference? | F3–F4, Fig. 3a–c; `tab:appendix-investment-comprehensive` | `experiments/02_investment_choice/` | API: `api_bet_constraint/src/run_all_experiments.py`; open-weight: `open_weight/investment_choice/run_experiment.py`; appendix CoT figure: `appendix_extended_cot/src/run_experiment.py` | `investment_choice/bet_constraint/`; `behavioral/investment_choice/v2_role_{llama,gemma}/`; appendix: `investment_choice/extended_cot/` |
| [03](experiments/03_matched_cap/) | Is the bet effect the freedom to choose or the larger range? | F4, Fig. 3d; Appendix Fig. 18, `tab:matched-cap-intervals`, `tab:matched-cap-companion` | `experiments/03_matched_cap/` | GPT-4o-mini: `sm_cap_ablation/src/gpt_{fixed_bet_size,variable_max_bet}_experiment.py` (+ `gpt_variable_max_bet_experiment/src/gpt_variable_restart_*.py`); three API models: `track0_w3_replication/src/run_track0_api.py`, driver `src/run_mc_ladder.sh` | `analysis/fixed_variable_comparison/`; `rebuttal_neurips_2026/matched_cap_mc32/` |
| [04](experiments/04_choice_ladder/) | Which part of choosing matters: choosing once, revising each round, or a wider cap? | F5; `tab:choice-ladder` | `experiments/04_choice_ladder/` | `src/run_e8.py` | `rebuttal_neurips_2026/policy_choice_ladder_e8/` |
| [05](experiments/05_framing_worked_example/) | Is the gap role-play or instruction following? Can one example move it? | F5; `tab:e7-per-model`, `tab:worked-example-intervals` | `experiments/05_framing_worked_example/` | `src/run_e7.py` | `rebuttal_neurips_2026/framing_rationality_factorial_e7/`; `rebuttal_neurips_2026/in_context_demo_{api,open_weight,open_weight_persona}/` |
| [06](experiments/06_mystery_wheel/) | Third task for the neural analyses: spin or stop with hidden odds | F6 (Table 1 columns), F7; appendix "Rules of the mystery-wheel task" | `experiments/06_mystery_wheel/` | `mystery_wheel/run_experiment.py` (Gemma); `run_mw_vllm.py` / `run_mw_parallel.py` (LLaMA) | `behavioral/mystery_wheel/{llama,gemma}_v2_role/` |
| [07](experiments/07_sae_readout/) | Can the risk indicators be read from the decision-time state, do tasks share it, and does a goal prompt sharpen it? | F6 Table 1, F7 Table 2, F8 Table 3 | `experiments/07_sae_readout/` | `src/run_groupkfold_recompute.py`; `src/run_table1_perm_null.py`; `src/cross_domain.py`; `src/run_rq2_aligned_hidden_transfer_sweep.py`; `src/condition_analysis_v2.py` | `sae_v3_analysis/results/`; `sae_features_v3/` |
| [08](experiments/08_steering/) | Does a direction in the hidden state move the wager? | F9, Fig. 4; `tab:causal-battery-suffnec`, `fig:causal-removal`, `tab:causal-transfer-matrix`, `tab:causal-condition-writability` | `experiments/08_steering/` | `multilayer_causal/run_experiment.py` with `configs/arms_sec4_*.yaml`; `src/runner.py`; `src/indicator_axes.py`. Wave log: `multilayer_causal/experiments/sec4_causal/INDEX.md` | `experiments/sec4_causal/` |
| [09](experiments/09_audits/) | Does the internal state add information beyond the game log? Is ruin higher at the same cumulative stake? Do the language findings survive other codebooks? | Appendix: `tab:added-controls`, `tab:exposure-matched`, `tab:convergent-codebook`, `tab:instrument-robustness`, moving-target sensitivity | `experiments/09_audits/` | `src/nested_baseline.py`; `src/exposure_matched.py`; `src/multi_instrument_robustness.py`; `src/moving_target_paper_metric.py` | `rebuttal_neurips_2026/nested_baseline_and_audits_e2/` |

Notes on the table:

- `slot_machine/gpt/` on the dataset holds **GPT-4.1-mini**, not GPT-4o-mini. The GPT-4o-mini
  slot-machine file is in `analysis/gpt_results_fixed_parsing/`, produced by
  `experiments/01_slot_machine/original_api_runners/gpt_experiments/src/gpt_fixed_parsing_experiment.py`.
- The paper's API investment-choice corpus is `investment_choice/bet_constraint/` (10 rounds, 3.2×
  middle option, caps $10–$70). `investment_choice/bet_constraint_cot/` (29 of 32 cells) and
  `investment_choice/initial/` are not used by the paper; their runners are in `archive/`.
- The additional controls (03's three-model run, 04, 05, 09) answer questions raised in review;
  their data sits under `rebuttal_neurips_2026/` on the dataset.

## Results by experiment

**01 — Slot machine.** Letting the model choose its bet raises bankruptcy in all six models: 0–3.1%
with a fixed $10 bet against 5–72% when the model picks the bet (LLaMA-3.1-8B 0.4% → 72.3%, Gemini
3.1% → 48.1%). Bets rise more after streaks of wins and of losses under variable betting
(Figure 2c–d). → [README](experiments/01_slot_machine/README.md)

**02 — Investment choice.** A self-set goal (`G`) roughly doubles bankruptcy, from about 19% to
about 36%, and the share of games in which the model raises its own target during play goes from
about 11–17% to about 48–50%; the direction is the same in all six models. The goal effect on
bankruptcy comes from the API runs. → [README](experiments/02_investment_choice/README.md)

**03 — Matched cap.** With the maximum bet matched between arms, the choosing arm still goes broke
more often: GPT-4o-mini variable-arm bankruptcy is about 15/16/18% at $30/$50/$70, and on three API
models ten of twelve cap × model intervals clear zero with none reversed. The fixed arm's low rate
partly reflects declining to play. → [README](experiments/03_matched_cap/README.md)

**04 — Choice ladder.** On LLaMA, bankruptcy is 2% with a forced $70 stake, 5% when the stake is
chosen once, 85% when it can be revised every round and 80% with a wider cap: the effect comes from
per-round revision, not from choice as such. → [README](experiments/04_choice_ladder/README.md)

**05 — Framing and worked example.** The fixed-vs-variable gap is carried by LLaMA (+76 pp),
Gemini (+20) and Gemma (+14); an explicit expected-value instruction narrows LLaMA to +40 and
closes Gemini and Gemma. One escalating worked example moves Gemini from 21% to 52% bankruptcy.
→ [README](experiments/05_framing_worked_example/README.md)

**06 — Mystery wheel.** A third negative-EV task whose odds are hidden; its games supply the third
column of the neural readout and the cross-task sharing tests of 07 and 08.
→ [README](experiments/06_mystery_wheel/README.md)

**07 — SAE readout.** Betting aggressiveness, loss chasing and extreme choice can be read from the
decision-time state at layer 22 with game-grouped cross-validation and a permutation null; which
indicator reads best depends on the task. On Gemma the tasks share a small, balance-linked risk
subspace, and a goal prompt makes betting aggressiveness easier to read.
→ [README](experiments/07_sae_readout/README.md)

**08 — Steering.** A direction built from the model's own high- and low-bet rounds raises and
lowers the wager on Gemma (dose ladder, z ≈ 4.4 against norm-matched random directions), and
removing it lowers betting on both models; the SAE readout direction stays within the random band.
→ [README](experiments/08_steering/README.md)

**09 — Audits.** The sparse SAE block adds little beyond game-log observables once folds are
grouped by state (ΔR² +0.0024, below the pre-set margin), while the raw hidden state adds +0.059;
at equal cumulative stake, 24 of 24 threshold cells favour the forced arm; the language-marker
contrasts are rechecked under a frozen convergent codebook and other instruments. → [README](experiments/09_audits/README.md)

## Reproducing figures and tables

The figures and tables are drawn in the companion paper repository `LLM_Addiction_NMT_KOR`, which is
private. Each experiment README names the generator script there (for example
`scripts/figures/fig02_slot_machine.py`, `scripts/figures/fig03_investment_choice_1x4.py`,
`scripts/figures/fig04_causal_battery.py`, `scripts/tables/body_tables.py`). The dataset's
`paper_neurips_2026/camera_ready/` holds those scripts (`scripts/`), one index file per figure and
table (`paper_index/`) and `PAPER_ASSET_MAP.md`. For the neural pipeline, see also
[`experiments/07_sae_readout/docs/PAPER_CANONICAL.md`](experiments/07_sae_readout/docs/PAPER_CANONICAL.md).

## Known traps

- **Round outcomes are stored in three different ways.** The open-weight exports put the win/loss
  in `decisions[i]["win"]`. The four API exports put it in
  `round_details[i]["game_result"]["result"]`, which is a **dict** and must never be `str()`d. The
  GPT-4o-mini fixed-parsing export has no per-step outcome and keeps it one level up, in the game's
  own `game_history` list. A reader that knows only one layout reports **zero** post-loss decisions
  for the others instead of failing. The correct reader is `result_from_record` /
  `build_outcome_series` in `experiments/07_sae_readout/src/run_multimodel_distortion_analysis.py`.
  After any change, check that every corpus's round-level win rate lands in [0.25, 0.35]; the task
  sets it at 0.30.
- **The prompt-module letters differ by model.** LLaMA writes `H` for the hidden-patterns module;
  the other five models write `R`. Several figures label the axis `H` and match the data on `R`.
- **The `model` field inside a file decides which model it is, not the directory name** (see the
  `slot_machine/gpt/` note above).
- **Older maps use old paths and numbering.** `docs/PAPER_CANONICAL_CODE.md` and `docs/MANIFEST.md`
  predate the camera-ready and this layout; use the Experiments table above and `PATH_MAP.md`.

## Citation

```bibtex
@inproceedings{lee2026gambling,
  title     = {Can Large Language Models Develop Gambling Addiction?},
  author    = {Lee, Seungpil and Shin, Donghyun and Lee, Yoonjung and Kim, Sundong},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
