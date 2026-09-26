# e8_constraint_choice — the choice ladder

Additional control for **Finding 5** of the paper (appendix table `tab:choice-ladder`). For the
whole-repository map, see the [top-level README](../../README.md).

## Question

Letting a model choose its bet changes several things at once. Which of them raises bankruptcy?
The ladder adds one freedom at a time on the slot machine:

| Arm | What the model may do |
|---|---|
| `forced_fixed` | Nothing: every bet is the stake given by `--cap` (10, 30, 50 or 70) |
| `choose_fixed` | Pick one stake ($10/$30/$50/$70) before the first round, then bet it every round |
| `variable_cap70` | Pick any bet up to $70, again every round |
| `variable_open` | Pick any bet up to $100, every round |

On LLaMA-3.1-8B, bankruptcy is 2% (forced $70), 5% (chosen once), 85% (revised each round) and 80%
(wider cap). Gemma almost always declines to play, so its ladder sits at the floor.

The design was fixed before data collection in [`PREREGISTRATION.md`](PREREGISTRATION.md).

## Entry script

`src/run_e8.py` runs one arm for one open-weight model. It imports the game logic and parser of
`../track0_w3_replication/src/` unchanged, and the model loader and role text of
`../e7_factorial/src/run_e7.py`. Paths are resolved from the repository root, so it runs from a
checkout without extra setup.

```bash
python experiments/04_choice_ladder/src/run_e8.py \
    --model llama --arm choose_fixed --n_games 200 --gpu 0 --output_dir out/e8

python experiments/04_choice_ladder/src/run_e8.py \
    --model llama --arm forced_fixed --cap 70 --n_games 100 --gpu 0 --output_dir out/e8
```

`--model` is `llama` or `gemma`; `--persona` prepends the role sentence to every prompt.

## Outputs

One file per arm, `<output_dir>/e8_<model>_<arm>_<tag>_<timestamp>.json`, with the per-round prompt
and full response, the parsed stake choice, and a manifest of seeds and denominators.

## Data on Hugging Face

`rebuttal_neurips_2026/policy_choice_ladder_e8/`
