# Can Large Language Models Develop Gambling Addiction?

Seungpil Lee, Donghyun Shin, Yoonjung Lee, Sundong Kim · Gwangju Institute of Science and Technology (GIST) · NeurIPS 2026

**[Paper](https://arxiv.org/abs/2509.22818)** · **[Project page](https://llm-addiction.pages.dev)** · **[Data](https://huggingface.co/datasets/llm-addiction-research/llm-addiction)**

<p align="center">
  <img src="site/assets/img/hero.jpg" alt="Illustration of a language model at a slot machine" width="720">
  <br><sub>Illustration from the project page. It shows no data.</sub>
</p>

This repository holds the experiment and analysis code for the paper. Six LLMs (GPT-4o-mini,
GPT-4.1-mini, Gemini-2.5-Flash, Claude-3.5-Haiku, LLaMA-3.1-8B, Gemma-2-9B) play two
negative-expected-value gambling tasks, a slot machine and an investment choice, under prompts that
either let them choose the bet size or ask them to set their own goal. We score each game with
indicators taken from clinical criteria for problem gambling (betting aggressiveness, loss chasing,
extreme bets) and with bankruptcy. On the two open-weight models we then read those indicators from
the hidden state at decision time with sparse-autoencoder (SAE) features. Finally we steer the model
by adding or removing a direction in that hidden state.

**Code is here. Data and figures are elsewhere.** Every game log, SAE feature file and steering run
is on the Hugging Face dataset. Result JSONs are gitignored in this repository (`*.json`, `*.csv`,
`*.npz` are in `.gitignore`), so a fresh clone contains the code without the results.

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

<details>
<summary>The nine findings as numbered in the camera-ready paper</summary>

| # | Finding |
|---|---|
| F1 | Variable betting amplifies bankruptcy |
| F2 | Variable betting amplifies streak-driven bet escalation |
| F3 | Goal-setting prompts reshape risk preferences |
| F4 | The goal effect generalises across models; bet-size autonomy survives a matched-cap control |
| F5 | The bet-size effect comes from per-round revisable choice; framing and instruction set its size |
| F6 | The decision-time state carries the risk indicators, and the task decides which one it carries best |
| F7 | On Gemma the tasks share a small, balance-linked risk subspace |
| F8 | Goal-setting makes betting aggressiveness easier to read |
| F9 | A behaviour-built direction moves the wager; the readout direction reports it |

</details>

## Experiments

One row per experiment. "Paper" gives the camera-ready Finding and the figure or table, including
appendix table labels (`tab:…`). Code paths are relative to the repository root. Data paths are
relative to the root of the Hugging Face dataset.

| Experiment | Question | Paper | Code folder | Entry scripts | Data on HF |
|---|---|---|---|---|---|
| Slot machine, six models | Does choosing the bet size raise bankruptcy? | F1–F2, Fig. 2; `tab:appendix-slot-comprehensive` | `paper_experiments/slot_machine_6models/` | `src/llama_gemma_experiment.py` (LLaMA, Gemma); `src/run_gpt5_experiment.py` (GPT-4.1-mini); `src/run_claude_experiment.py`; `src/run_gemini_experiment.py` | `behavioral/slot_machine/{llama,gemma}_v4_role/`; `slot_machine/{gpt,claude,gemini}/`; `analysis/gpt_results_fixed_parsing/` (GPT-4o-mini) |
| Investment choice | Does a self-set goal change risk preference? | F3–F4, Fig. 3a–c; `tab:appendix-investment-comprehensive` | API models: `paper_experiments/investment_choice_experiment/`, `paper_experiments/investment_choice_extended_cot/`. Open-weight: `exploratory_experiments/alternative_paradigms/src/investment_choice/` | API: `investment_choice_extended_cot/src/run_experiment.py` (same code as `investment_choice_experiment/src/src/`); open-weight: `run_experiment.py` in the folder named. See the note below the table | `investment_choice/bet_constraint/`; `investment_choice/bet_constraint_cot/`; `behavioral/investment_choice/v2_role_{llama,gemma}/` |
| Matched cap, GPT-4o-mini | Is the bet effect the freedom to choose or the larger range? | F4 (first run) | `paper_experiments/sm_cap_ablation/` | `src/gpt_fixed_bet_size_experiment.py`; `src/gpt_variable_max_bet_experiment.py` | `analysis/fixed_variable_comparison/` |
| Matched cap, three API models | Does the matched-cap result hold beyond one model? | F4 (second run), Fig. 3d; `fig:matched-cap-forest`, `tab:matched-cap-intervals`, `tab:matched-cap-companion` | `paper_experiments/track0_w3_replication/` | `src/run_track0_api.py`; driver `src/run_mc_ladder.sh` | `rebuttal_neurips_2026/matched_cap_mc32/` |
| Choice ladder | Which part of choosing matters: choosing once, revising each round, or a wider cap? | F5; `tab:choice-ladder` | `paper_experiments/e8_constraint_choice/` | `src/run_e8.py` | `rebuttal_neurips_2026/policy_choice_ladder_e8/` |
| Role framing × rationality instruction; worked example | Is the gap role-play or instruction following? Can one example move it? | F5; `tab:e7-per-model`, `tab:worked-example-intervals` | `paper_experiments/e7_factorial/` | `src/run_e7.py` | `rebuttal_neurips_2026/framing_rationality_factorial_e7/`; `rebuttal_neurips_2026/in_context_demo_{api,open_weight,open_weight_persona}/` |
| Game-log baseline and audits | Does the internal state add information beyond the game log? Is ruin higher at the same cumulative stake? Do the language findings survive other codebooks? | Appendix: `tab:added-controls`, `tab:exposure-matched`, `tab:convergent-codebook`, `tab:instrument-robustness`, moving-target sensitivity | `paper_experiments/e2_coding/` | `src/nested_baseline.py`; `src/exposure_matched.py`; `src/multi_instrument_robustness.py`; `src/moving_target_paper_metric.py` | `rebuttal_neurips_2026/nested_baseline_and_audits_e2/` |
| Mystery wheel | Third task for the neural analyses: spin or stop with hidden odds | F6 (Table 1 columns), F7; appendix "Rules of the mystery-wheel task" | `exploratory_experiments/alternative_paradigms/src/mystery_wheel/` | `run_experiment.py`; LLaMA runs were launched with the top-level `run_mw_vllm.py` / `run_mw_parallel.py` | `behavioral/mystery_wheel/{llama,gemma}_v2_role/` |
| SAE readout | Can the risk indicators be read from the decision-time state? | F6, Table 1 | `sae_v3_analysis/` | `src/run_groupkfold_recompute.py`; `src/run_table1_perm_null.py` | `sae_v3_analysis/results/`; `sae_features_v3/` |
| Cross-task sharing | Do the tasks share a risk subspace? | F7, Table 2 | `sae_v3_analysis/` | `src/cross_domain.py`; `src/run_rq2_aligned_hidden_transfer_sweep.py` | `sae_v3_analysis/results/` |
| Condition modulation | Does a goal prompt make the readout sharper? | F8, Table 3 | `sae_v3_analysis/` | `src/condition_analysis_v2.py`; `src/run_groupkfold_recompute.py` | `sae_v3_analysis/results/` |
| Steering and removal | Does a direction in the hidden state move the wager? | F9, Fig. 4; `tab:causal-battery-suffnec`, `fig:causal-removal`, `tab:causal-transfer-matrix`, `tab:causal-condition-writability` | `multilayer_causal/` | `run_experiment.py` with `configs/arms_sec4_*.yaml`; `src/runner.py`; `src/indicator_axes.py`. Wave log: `experiments/sec4_causal/INDEX.md` | `experiments/sec4_causal/` |
| Other additional controls | Portfolio task, persona framing, compliance directions, top-K feature removal, game-length confound | Not cited in the camera-ready | `paper_experiments/{m1_portfolio_discriminant,m2_persona_decoupling,m5_compliance_residualisation,d_distributed_effect,track_L_length_confound}/` | see each folder's README | see each folder's README |

Notes on the table:

- `slot_machine/gpt/` on the dataset holds **GPT-4.1-mini**, not GPT-4o-mini. The GPT-4o-mini
  slot-machine file is in `analysis/gpt_results_fixed_parsing/`, and it was produced by the older
  runner `legacy/gpt_experiments/src/gpt_fixed_parsing_experiment.py`.
- The open-weight investment-choice and mystery-wheel runners sit under `exploratory_experiments/`,
  but they are the ones that produced the paper's data.
- Investment choice has several runner copies. `investment_choice_experiment/src/run_investment_experiment.py`
  is the first, 10-round version. The runners whose output folders carry the names of the released
  API data (`bet_constraint`, `bet_constraint_cot`) are in `legacy/investment_choice_bet_constraint/src/`
  and `legacy/investment_choice_bet_constraint_cot/src/`. The 100-round version is
  `investment_choice_extended_cot/src/` (released as `investment_choice/extended_cot/`). The
  dataset's `MANIFEST.md` names the files behind each panel of Figure 3.
- The folders that answer questions raised after submission (matched cap on three API models,
  choice ladder, framing factorial, audits) are the "additional controls" of the paper's appendix.
  Their data sits under `rebuttal_neurips_2026/` on the dataset.

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
    allow_patterns=["slot_machine/*", "behavioral/*", "MANIFEST.md"],
)
```

The full dataset is large: `sae_features_v3/` alone holds hidden-state files of tens of GB. The
dataset's `MANIFEST.md` maps each figure and table to its files.

## Setup

- **Python 3.10 or 3.11.** The cloud jobs ran on images with Python 3.10, PyTorch 2.7–2.8 and CUDA
  12.6; the HPC environment described in `CLAUDE.md` used Python 3.11.
- **No requirements file.** The packages the code imports are:

  ```bash
  pip install numpy scipy scikit-learn statsmodels matplotlib pyyaml tqdm \
      torch "transformers>=4.44" accelerate safetensors "huggingface_hub<1.0" sae_lens \
      openai anthropic google-genai
  ```

  `vllm` is needed only for `run_mw_vllm.py`.
- **Open-weight models** (`meta-llama/Llama-3.1-8B-Instruct`, `google/gemma-2-9b-it`) are gated on
  Hugging Face. Accept their licences and log in, or set `HF_TOKEN`.
- **API keys are read from environment variables.** Never commit them; `.env` is gitignored.

| Runner | Environment variables it reads |
|---|---|
| `paper_experiments/slot_machine_6models/src/run_gpt5_experiment.py` | `GPT5_API_KEY` or `OPENAI_API_KEY`; model from `GPT5_MODEL` (default `gpt-4.1-mini`) |
| `paper_experiments/slot_machine_6models/src/run_claude_experiment.py` | `CLAUDE_API_KEY`; model from `CLAUDE_MODEL` |
| `paper_experiments/slot_machine_6models/src/run_gemini_experiment.py` | `GEMINI_API_KEY`; model from `GEMINI_MODEL` |
| `paper_experiments/sm_cap_ablation/src/*.py` | `GPT_API_KEY` or `OPENAI_API_KEY` |
| `paper_experiments/investment_choice_experiment/src/models/*_runner.py` | `OPENAI_API_KEY` or `GPT_API_KEY`; `CLAUDE_API_KEY` or `ANTHROPIC_API_KEY`; `GEMINI_API_KEY` |
| `paper_experiments/investment_choice_extended_cot/src/models/*_runner.py` | `OPENAI_API_KEY` or `GPT_API_KEY`; `ANTHROPIC_API_KEY`; `GEMINI_API_KEY` |
| `paper_experiments/track0_w3_replication/src/run_track0_api.py` (also used by `e7_factorial/src/run_e7.py` for API models) | `OPENAI_API_KEY`; `ANTHROPIC_API_KEY` or `CLAUDE_API_KEY`; `GOOGLE_API_KEY` or `GEMINI_API_KEY` |
| `paper_experiments/track0_w3_replication/src/run_mc_ladder.sh` | Sources `$HOME/.env`, then stops unless `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` and `GOOGLE_API_KEY` are all set |
| `multilayer_causal/` | `HF_TOKEN` (model download and checkpoint sync; see below) |

The three slot-machine runners, `run_investment_experiment.py` and
`investment_choice_extended_cot/src/run_all_experiments.py` also read a `.env` file next to the
script (or, for the last one, in the folder above `src/`). Variables already set in the shell take
precedence.

Things to know before running anything:

- **Most scripts will not run unmodified.** They were written against absolute paths on the
  machines the experiments ran on (`/home/v-seungplee/...`, `/data/llm_addiction/...`,
  `/home/jovyan/...`) and write their outputs there. Treat a script as a record of what was run,
  and point its inputs and outputs at your own copy of the dataset before rerunning it.
- **`multilayer_causal/` syncs checkpoints to the dataset.** When `HF_TOKEN` is set, each arm first
  tries to resume from the released checkpoint and then uploads its progress to
  `llm-addiction-research/llm-addiction`. With a read-only token the uploads fail and are logged;
  `--smoke` turns the sync off.
- **API results will not reproduce bit for bit.** The Claude-3.5-Haiku checkpoint has been withdrawn
  from its API. The stored responses on the dataset are the record.

One check runs anywhere, with no dataset and no GPU:

```bash
python3 sae_v3_analysis/tests/test_distortion_outcome_schema.py
```

## Reproducing figures and tables

The figures and tables are drawn in the companion paper repository `LLM_Addiction_NMT_KOR`, which is
private. They can be regenerated from the Hugging Face data:

- the dataset's `paper_neurips_2026/camera_ready/` holds the figure and table scripts (`scripts/`),
  one index file per figure and table (`paper_index/`) and `PAPER_ASSET_MAP.md`;
- the dataset's `MANIFEST.md` maps each paper element to the files behind it.

In this repository, [`PAPER_CANONICAL_CODE.md`](PAPER_CANONICAL_CODE.md) (figure → code → data) and
[`MANIFEST.md`](MANIFEST.md) (paper claim → file) are earlier maps. Use them with care: they use
pre-camera-ready section and figure numbers, and the steering row of `PAPER_CANONICAL_CODE.md`
points at runs the paper no longer uses (Figure 4 comes from `multilayer_causal/`). The Experiments
table above has the current numbering. For the neural pipeline, see
[`sae_v3_analysis/README.md`](sae_v3_analysis/README.md) and
[`sae_v3_analysis/docs/PAPER_CANONICAL.md`](sae_v3_analysis/docs/PAPER_CANONICAL.md).

## Repository map

Status labels: **paper** = produced a result in the camera-ready paper; **additional control** =
added after submission, cited in the appendix unless marked otherwise; **record** = plans, notes and
logs kept for the audit trail; **not cited** = code the paper does not use.

| Path | Status | What it holds |
|---|---|---|
| `paper_experiments/` | mixed | One folder per experiment; see the next table |
| `sae_v3_analysis/` | paper | SAE readout, cross-task sharing and condition modulation (F6–F8), with its own `docs/` and `results/` |
| `multilayer_causal/` | paper | Steering and removal harness (F9). Its README describes an earlier pilot; start from [`experiments/sec4_causal/README.md`](multilayer_causal/experiments/sec4_causal/README.md) for the Figure 4 battery |
| `exploratory_experiments/` | not cited, with two exceptions | Other paradigms and analyses. The investment-choice and mystery-wheel runners under `alternative_paradigms/src/` are paper |
| `legacy/` | mixed | Here "legacy" does not mean deprecated. The older API slot-machine runners produced released data. Read [`legacy/README.md`](legacy/README.md) first |
| `site/` | project page | Source of https://llm-addiction.pages.dev |
| `analysis/` | not cited | Early analysis, monitoring and model-loading test scripts for the open-weight runs |
| `scripts/` | not cited | Shell launchers for the open-weight runs (SLURM directives disabled) |
| `plans/`, `PLAN_*.md` | record | Experiment plans written before each run |
| `docs/` | record | Design specs and plans for `multilayer_causal/` |
| `rebuttal_review/` | record | Notes and tables from the review period |
| `amlt/` | record | Cluster job files for the May 2026 additional controls |
| `drafts/` | record | A draft restructuring of the discussion section |
| `investment_choice_bet_constraint_cot/` | record | One code-review PDF |
| `run_mw_vllm.py`, `run_mw_parallel.py` | paper | LLaMA mystery-wheel launchers that call the runner in `exploratory_experiments/` |
| `PAPER_CANONICAL_CODE.md`, `MANIFEST.md` | record | Earlier figure → code → data maps (pre-camera-ready numbering) |
| `CLAUDE.md`, `STRUCTURE.md`, `SLURM_GUIDE.md`, `EXPERIMENT_DESIGN_COMPARISON.md`, `ev_transparency_gambling_avoidance.md`, `token_truncation_root_cause_analysis.*` | record | Environment notes and analyses from earlier stages. `CLAUDE.md` describes the HPC cluster, and one finding it lists was withdrawn (see its banner) |

Inside `paper_experiments/`:

| Folder | Status | Paper |
|---|---|---|
| `slot_machine_6models/` | paper | F1–F2 |
| `investment_choice_experiment/`, `investment_choice_extended_cot/` | paper | F3–F4 |
| `sm_cap_ablation/` | paper | F4 |
| `track0_w3_replication/` | additional control | F4 |
| `e8_constraint_choice/`, `e7_factorial/` | additional control | F5 |
| `e2_coding/` | additional control | appendix audits |
| `m1_portfolio_discriminant/`, `m2_persona_decoupling/`, `m5_compliance_residualisation/`, `d_distributed_effect/`, `track_L_length_confound/` | additional control, not cited | — |
| `llama_sae_analysis/`, `pathway_token_analysis/` | not cited | Earlier drafts; the activation-patching claims in `llama_sae_analysis/` were withdrawn |

## Known traps

- **Round outcomes are stored in three different ways.** The open-weight exports put the win/loss
  in `decisions[i]["win"]`. The four API exports put it in
  `round_details[i]["game_result"]["result"]`, which is a **dict** and must never be `str()`d. The
  GPT-4o-mini fixed-parsing export has no per-step outcome and keeps it one level up, in the game's
  own `game_history` list. A reader that knows only one layout reports **zero** post-loss decisions
  for the others instead of failing. The correct reader is `result_from_record` /
  `build_outcome_series` in `sae_v3_analysis/src/run_multimodel_distortion_analysis.py`. After any
  change, check that every corpus's round-level win rate lands in [0.25, 0.35]; the task sets it
  at 0.30.
- **The prompt-module letters differ by model.** LLaMA writes `H` for the hidden-patterns module;
  the other five models write `R`. Several figures label the axis `H` and match the data on `R`.
- **The `model` field inside a file decides which model it is, not the directory name** (see the
  `slot_machine/gpt/` note above).
- **`PAPER_CANONICAL_CODE.md` gives the dataset id as `iamseungpil/llm-addiction-research`** in its
  Figure 2a–c row. That id is wrong; the dataset is `llm-addiction-research/llm-addiction`.

## Citation

```bibtex
@inproceedings{lee2026gambling,
  title     = {Can Large Language Models Develop Gambling Addiction?},
  author    = {Lee, Seungpil and Shin, Donghyun and Lee, Yoonjung and Kim, Sundong},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
