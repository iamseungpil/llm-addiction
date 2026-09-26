# 05 — Role framing × rationality instruction, and worked examples

**Question.** At a matched $70 cap, is the fixed-versus-variable bankruptcy gap produced by the role
sentence in the prompt, or by the model not knowing that stopping is the best move? Can one worked
example of a game move behaviour?

**Paper result.** Finding 5; appendix tables `tab:e7-per-model` (App. G.2) and
`tab:worked-example-intervals`. The gap is carried by LLaMA (+76 pp), Gemini (+20) and Gemma
(+14); the rationality instruction narrows LLaMA to +40 and closes Gemini and Gemma. With a worked
example, Gemini's variable-arm bankruptcy goes from 21% (cautious example) to 52% (escalating
example).

## Design (from the code)

- The slot machine of [01](../01_slot_machine/README.md) at a $70 cap (`--cap`, default 70),
  fixed vs variable, `--n_games` games per cell (default 100).
- Two prompt preambles crossed: **role framing** (the role sentence of the open-weight runners)
  and a **rationality instruction** (`--rat`: the game has negative expected value and stopping at
  once maximises it).
- Worked-example arm: one completed game prepended as an example, cautious or escalating.
- Game logic and parser come unchanged from
  `experiments/03_matched_cap/track0_w3_replication/src/`, and API calls go through its
  `run_track0_api.py`.
- The design and analysis plan were fixed before data collection in
  [`PREREGISTRATION.md`](PREREGISTRATION.md).

| Argument | Values |
|---|---|
| `--model` | `gpt-4o-mini`, `gpt-4.1-mini`, `gemini-2.5-flash`, `claude-haiku-4-5-20251001`, `gemma`, `llama` |
| `--mode` | `fixed` or `variable` |
| `--preamble` | `none`, `role`, `role_nc` (role without the final compliance sentence), `demo_cautious`, `demo_escalate`, `demo_cautious_persona`, `demo_escalate_persona` |
| `--rat` | flag: add the rationality instruction |
| `--cap` | default 70 |
| `--n_games` | default 100 |
| `--output_dir` | required |

## Quick Start

```bash
python experiments/05_framing_worked_example/src/run_e7.py \
    --model gemini-2.5-flash --mode variable --preamble role --rat \
    --n_games 100 --output_dir out/e7

python experiments/05_framing_worked_example/src/run_e7.py \
    --model llama --mode fixed --preamble role --n_games 100 --gpu 0 --output_dir out/e7
```

API keys (read through `run_track0_api.py`): `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` or
`CLAUDE_API_KEY`, `GOOGLE_API_KEY` or `GEMINI_API_KEY`. Open-weight models run on the GPU given by
`--gpu`. The harness paths are resolved from the script's location, so no `PYTHONPATH` is needed.

Output: one file per cell,
`<output_dir>/e7_<model>_cap<cap>_<mode>_<preamble>_rat<0|1>_<timestamp>.json`, plus a progress
log `cell_status.txt`. The runner refuses to start if the output directory already holds a file for
the same cell (override with `--allow_existing_cell`).

## Data on Hugging Face

- `rebuttal_neurips_2026/framing_rationality_factorial_e7/` — the factorial cells
- `rebuttal_neurips_2026/in_context_demo_api/`, `in_context_demo_open_weight/`,
  `in_context_demo_open_weight_persona/` — the worked-example cells

## Figures and tables (paper repository, private)

- `paper_data/tables/appendix/code/gbsa_q3_e7_per_model.py` — `tab:e7-per-model`.
- `paper_data/tables/appendix/code/worked_example_intervals.py` — `tab:worked-example-intervals`
  (and the worked-example row of `tab:added-controls`).
