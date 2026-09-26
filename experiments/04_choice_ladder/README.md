# 04 — Choice ladder

**Question.** Letting a model choose its bet changes several things at once. Which of them raises
bankruptcy: choosing once, revising the bet every round, or a wider cap?

**Paper result.** Finding 5, appendix table `tab:choice-ladder` (App. G.3). On LLaMA-3.1-8B,
bankruptcy is 2% with a forced $70 stake, 5% when the stake is chosen once, 85% when it can be
revised every round and 80% with a wider cap. Gemma almost always declines to play, so its ladder
sits at the floor.

## Design (from the code)

The slot machine of [01](../01_slot_machine/README.md) ($100, win probability 0.30, payout 3.0×),
with one freedom added per arm:

| Arm | What the model may do |
|---|---|
| `forced_fixed` | Nothing: every bet is the stake given by `--cap` (10, 30, 50 or 70) |
| `choose_fixed` | Pick one stake ($10/$30/$50/$70) before the first round, then bet it every round |
| `variable_cap70` | Pick any bet up to $70, again every round |
| `variable_open` | Pick any bet up to $100, every round |

- Open-weight models only (`--model llama` or `gemma`); `--n_games` games per arm.
- No role sentence unless `--persona` is given; the paper's ladder runs carry none (disclosed).
- Game logic and parser are imported unchanged from
  `experiments/03_matched_cap/track0_w3_replication/src/`; the model loader and role text from
  `experiments/05_framing_worked_example/src/run_e7.py`.
- The design was fixed before data collection in [`PREREGISTRATION.md`](PREREGISTRATION.md).

## Quick Start

```bash
python experiments/04_choice_ladder/src/run_e8.py \
    --model llama --arm choose_fixed --n_games 200 --gpu 0 --output_dir out/e8

python experiments/04_choice_ladder/src/run_e8.py \
    --model llama --arm forced_fixed --cap 70 --n_games 100 --gpu 0 --output_dir out/e8
```

Output: one file per arm, `<output_dir>/e8_<model>_<arm>_<tag>_<timestamp>.json`, with the
per-round prompt and full response, the parsed stake choice, and a manifest of seeds and
denominators.

## Data on Hugging Face

`rebuttal_neurips_2026/policy_choice_ladder_e8/`

## Figures and tables (paper repository, private)

`tab:choice-ladder` is written directly in the appendix source (`neurips_content_en/appendix.tex`)
from the summaries in that folder; no generator script for it exists in the paper repository.
