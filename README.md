# llm-addiction

Experiment and analysis code for the NeurIPS 2026 paper on addictive-like gambling
behaviour in large language models (submission 24231).

Two behavioural paradigms — a slot machine and a repeated investment choice — are run
across six models (LLaMA-3.1-8B, Gemma-2-9B, GPT-4o-mini, GPT-4.1-mini,
Claude-3.5-Haiku, Gemini-2.5-Flash) under prompt manipulations of autonomy and goal
framing. A sparse-autoencoder readout and a causal battery then ask whether the
behavioural indices are legible in the open-weight models' representations.

**This repository holds code. The paper is elsewhere and the data is elsewhere.**

| | Where |
|---|---|
| Paper source (LaTeX, figures, per-float index) | companion repository `LLM_Addiction_NMT_KOR` |
| Released data | HF dataset `llm-addiction-research/llm-addiction` |
| Which code produced which figure | [`PAPER_CANONICAL_CODE.md`](PAPER_CANONICAL_CODE.md) |
| Which released file backs which paper claim | [`MANIFEST.md`](MANIFEST.md) |

## Where to start

- **"Which script drew Figure N?"** → `PAPER_CANONICAL_CODE.md`. It exists because
  directory names in this repository are not a reliable status signal, and it is the
  authority when a directory name and this map disagree.
- **"Which released file backs this number?"** → `MANIFEST.md`, then
  `NEURIPS_CANONICAL_INDEX.md` in the paper repository (its §5 is the do-not-cite list).
- **"How does the neural pipeline run?"** → [`sae_v3_analysis/README.md`](sae_v3_analysis/README.md)
  and [`sae_v3_analysis/docs/`](sae_v3_analysis/docs/).
- **"What is under `legacy/`?"** → [`legacy/README.md`](legacy/README.md). Short
  version: in *this* repository `legacy/` does **not** mean deprecated, and several
  directories under it hold paper-canonical code. On the released dataset the same
  word does mean deprecated. Do not carry one meaning across to the other.

## Layout

```
paper_experiments/     first-class paper experiments: the six-model slot machine,
                       investment choice, the cap ablation, the rebuttal tracks
sae_v3_analysis/       the neural pipeline (SAE readout, GroupKFold recompute,
                       causal battery) plus its own docs/ and results/
legacy/                outside the paper_experiments/ tree; NOT deprecated by default
exploratory_experiments/, multilayer_causal/   exploratory work the paper does not cite
plans/, rebuttal_review/, docs/                planning and rebuttal records
```

## Documents: current versus historical

| Document | Status |
|---|---|
| `PAPER_CANONICAL_CODE.md` | **Current.** The figure → code → data map. |
| `MANIFEST.md` | **Current.** Paper claim → released-file map. |
| `sae_v3_analysis/docs/PAPER_CANONICAL.md`, `PAPER_MANIFEST.md`, `WORKSPACE_RUNBOOK.md` | **Current** for the neural pipeline. |
| `legacy/README.md`, `sae_v3_analysis/release_labels/` | **Current.** Deprecation conventions, and the label files staged for the released dataset. |
| `CLAUDE.md` | **Historical environment, mixed content.** Useful on architecture and conventions; its paths describe an HPC cluster that is not this machine, and one of its listed key findings was withdrawn from the paper. See the banner at the top of that file. |
| `STRUCTURE.md` | **Historical.** Carries its own archive note. |
| `PLAN_*.md`, `plans/*`, `rebuttal_review/*` | **Historical.** Planning records kept for the audit trail; they describe intent at the time of writing, not the final pipeline. |
| `EXPERIMENT_DESIGN_COMPARISON.md`, `SLURM_GUIDE.md`, `token_truncation_root_cause_analysis.md` | **Historical.** |

## Running anything

There is no single entry point, and most scripts will not run unmodified. They were
written against absolute paths on the machines the experiments ran on —
`/scratch/x3415a02/...`, `/data/llm_addiction/...`, `/home/ubuntu/...` — and expect
either an HPC scheduler or live API keys. Treat a script as a record of what was run
and repoint its inputs at the released dataset before rerunning it.

The one thing that does run anywhere, with no dataset and no GPU:

```
python3 sae_v3_analysis/tests/test_distortion_outcome_schema.py
```

## Known traps

- **Round outcomes are spelled three different ways.** Open-weight V4role exports put
  the W/L in `decisions[i]["win"]`; the four API exports put it in
  `round_details[i]["game_result"]["result"]`, which is a **dict** and must never be
  `str()`d; the GPT-4o-mini fixed-parsing export has no per-step outcome at all and
  keeps it one level up in the game's own `game_history` list. A reader that knows
  only one spelling reports **zero** post-loss decisions for the others instead of
  failing. The correct reader is `result_from_record` / `build_outcome_series` in
  `sae_v3_analysis/src/run_multimodel_distortion_analysis.py`. Sanity check after any
  change: every corpus's round-level win rate must land in [0.25, 0.35], since the
  task spec is 0.30.
- **The prompt alphabet is not uniform.** LLaMA writes `H` for hidden-patterns; the
  other five models write `R`. Several figures label the axis `H` and match the data
  on `R`.
- **`slot_machine/gpt/` on the dataset is GPT-4.1-mini**, not GPT-4o-mini. The
  canonical GPT-4o-mini file is
  `analysis/gpt_results_fixed_parsing/gpt_fixed_parsing_complete_20250919_151240.json`.
  The `model` field inside a file decides which model it is, never the directory name.
- **`PAPER_CANONICAL_CODE.md` names the dataset as `iamseungpil/llm-addiction-research`
  in its Figure 2a-c row.** That id is wrong; the dataset is
  `llm-addiction-research/llm-addiction`, as every other reference in this repository
  has it.
