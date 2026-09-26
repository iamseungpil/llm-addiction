# sec4_causal — the steering and removal battery (Figure 4)

This folder records the causal experiments behind **Finding 9** (Figure 4; appendix
`tab:causal-battery-suffnec`, `fig:causal-removal`, `tab:causal-transfer-matrix`,
`tab:causal-condition-writability`). For the whole-repository map, see the
[top-level README](../../../../../README.md).

## Question

Does a direction in the decision-time hidden state move the wager when it is added (sufficiency) or
projected out (necessity)? The battery compares a direction built from the model's own high-bet and
low-bet rounds, the SAE readout direction, a balance direction, and norm-matched random directions,
on Gemma-2-9B and LLaMA-3.1-8B.

## Where things are

- **[`INDEX.md`](INDEX.md)** is the wave-by-wave log: one entry per wave with its config, key
  numbers and verdict. Read it first.
- Each wave is one config in `../../configs/arms_sec4_*.yaml` (`p0`, `w2` … `w9`, `w10a`, `w10b`, `w11ic`, `w11mw`, `w13`, `w14`, `rawridge`).
- The harness is `../../run_experiment.py`, which calls `../../src/runner.py`. The directions are
  built in `../../src/indicator_axes.py`.

## Running one arm

Run from `experiments/08_steering/` (the folder that holds the `multilayer_causal` package):

```bash
python multilayer_causal/run_experiment.py \
    --arms multilayer_causal/configs/arms_sec4_p0.yaml --arm sec4_behavioural_ap3 --gpu 0 --n 200
```

Arm ids are the `id` fields under `arms:` in the config. Each arm reads its direction from
`multilayer_causal/assets/sec4/*.npz`, which is not in the repository: download it from the
dataset's `experiments/sec4_causal/assets/`, or rebuild it with `src/indicator_axes.py --dest ...`
(see `--help`).

`--smoke` runs a short check and turns off the Hugging Face sync. Without `--smoke`, a set
`HF_TOKEN` makes each arm resume from, and upload to, the dataset folder named by `MLC_HF_BASE`
(default `experiments/multilayer_causal/checkpoints`; the paper waves set it to
`experiments/sec4_causal/checkpoints`).

## Outputs

Per-arm `jsonl` checkpoints and a `_summary.json` go to `--out` (default `multilayer_causal/out/`).

## Data on Hugging Face

`experiments/sec4_causal/` — `checkpoints/<wave>/` (rollouts), `assets/` (directions),
`analysis/` (per-wave analysis JSON and plots).
