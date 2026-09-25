# e7_factorial — role framing × rationality instruction, and worked examples

Additional control for **Finding 5** of the paper (appendix tables `tab:e7-per-model` and
`tab:worked-example-intervals`). For the whole-repository map, see the [top-level README](../../README.md).

## Question

At a matched $70 cap, is the fixed-versus-variable bankruptcy gap produced by the role sentence in
the prompt, or by the model not knowing that stopping is the best move? The runner crosses betting
mode (fixed / variable) with two prompt preambles:

- **role framing**: the role sentence used by the open-weight runners of the main study;
- **rationality instruction**: a statement that the game has negative expected value and that
  stopping at once maximises it.

A second arm prepends one worked example of a completed game, either cautious or escalating, to test
whether a single example moves behaviour.

The design and analysis plan were fixed before data collection in
[`PREREGISTRATION.md`](PREREGISTRATION.md).

## Entry script

`src/run_e7.py` plays `--n_games` slot-machine games for one cell, reusing the game logic and parser
of `../track0_w3_replication/src/` unchanged.

| Argument | Values |
|---|---|
| `--model` | `gpt-4o-mini`, `gpt-4.1-mini`, `gemini-2.5-flash`, `claude-haiku-4-5-20251001`, `gemma`, `llama` |
| `--mode` | `fixed` or `variable` |
| `--preamble` | `none`, `role`, `role_nc` (role without the final compliance sentence), `demo_cautious`, `demo_escalate`, `demo_cautious_persona`, `demo_escalate_persona` |
| `--rat` | flag: add the rationality instruction |
| `--cap` | default 70 |
| `--n_games` | default 100 |
| `--output_dir` | required |

The script adds two absolute paths from the original machine to `sys.path` (lines 36–37). On
another machine, put the harness on `PYTHONPATH` yourself:

```bash
PYTHONPATH=paper_experiments/track0_w3_replication/src:paper_experiments/sm_cap_ablation/src \
python paper_experiments/e7_factorial/src/run_e7.py \
    --model gemini-2.5-flash --mode variable --preamble role --rat \
    --n_games 100 --output_dir out/e7
```

API models read their keys through `run_track0_api.py`: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` or
`CLAUDE_API_KEY`, `GOOGLE_API_KEY` or `GEMINI_API_KEY`. Open-weight models run on the GPU given by
`--gpu`.

## Outputs

One file per cell, `<output_dir>/e7_<model>_cap<cap>_<mode>_<preamble>_rat<0|1>_<timestamp>.json`,
plus a progress log `cell_status.txt`. The runner refuses to start if the output directory already
holds a file for the same cell (override with `--allow_existing_cell`).

## Data on Hugging Face

- `rebuttal_neurips_2026/framing_rationality_factorial_e7/` — the factorial cells
- `rebuttal_neurips_2026/in_context_demo_api/`, `in_context_demo_open_weight/`,
  `in_context_demo_open_weight_persona/` — the worked-example cells
