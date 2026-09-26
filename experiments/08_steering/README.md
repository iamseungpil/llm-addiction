# 08 — Steering and removal (Figure 4)

**Question.** Does a direction in the decision-time hidden state move the wager when it is added
(sufficiency) or projected out (necessity)?

**Paper result.** Finding 9, Figure 4 (`fig:causal-battery`); appendix `tab:causal-battery-suffnec`,
`fig:causal-removal`, `tab:causal-transfer-matrix`, `tab:causal-condition-writability`. A
direction built from the model's own high-bet and low-bet rounds moves betting on Gemma (dose
ladder, z ≈ 4.4 against norm-matched random directions) and removing it lowers betting on both
models; the SAE readout direction stays within the random band.

## Design (from the code)

- Directions compared on the same decision states: **behavioural** (built from the model's own
  high-bet vs low-bet rounds), the **SAE readout** direction of [07](../07_sae_readout/README.md),
  a **balance** (confound) direction, and **norm-matched random** directions
  (`multilayer_causal/src/indicator_axes.py`).
- Sufficiency: add α · (dose unit) · direction at every position of a write window (Gemma
  layers 16–21, LLaMA 14–19), α from −3 to +3; necessity: project the direction out
  (h ← h − (h·u)u) at every forward pass (`multilayer_causal/src/runner.py`, `hooks.py`).
- Each wave is one config, `multilayer_causal/configs/arms_sec4_*.yaml` (`p0`, `w2` … `w14`,
  `rawridge`); every arm is a list of single-decision trials resampled from the held-out slot
  machine, investment choice or mystery wheel states with frozen seeds.
- The wave-by-wave log with configs, key numbers and verdicts is
  [`multilayer_causal/experiments/sec4_causal/INDEX.md`](multilayer_causal/experiments/sec4_causal/INDEX.md).

## Quick Start

`multilayer_causal` is a Python package (`from multilayer_causal.src import ...`), so run it from
this folder:

```bash
cd experiments/08_steering
# one arm; --smoke runs a short check and turns off the Hugging Face sync
python multilayer_causal/run_experiment.py \
    --arms multilayer_causal/configs/arms_sec4_p0.yaml --arm sec4_behavioural_ap3 --gpu 0 --n 200
# build the directions (see --help); they are also on HF under experiments/sec4_causal/assets/
python -m multilayer_causal.src.indicator_axes --help
# unit tests (no GPU)
python -m pytest multilayer_causal/tests -q
```

With `HF_TOKEN` set, each arm resumes from and uploads to the dataset folder named by
`MLC_HF_BASE`; the paper waves set it to `experiments/sec4_causal/checkpoints`. The direction files
(`multilayer_causal/assets/sec4/*.npz`) are not in the repository; download them from the dataset's
`experiments/sec4_causal/assets/`. At the reorganisation commit the unit tests give 261 passed, 17
failed; the failures read direction assets or generated JSON files that are gitignored
(`multilayer_causal/assets/*.npz`, `configs/llama_gap_estimate.json`), and the same 17 fail at
`pre-reorg`.

## Folder contents

| Path | What it is |
|---|---|
| `multilayer_causal/` | The harness: `run_experiment.py`, `src/`, `configs/`, `amlt/` job templates, `scripts/`, `tests/`, and `results/` (per-wave analysis JSON) |
| `multilayer_causal/experiments/sec4_causal/` | README and wave log for the Figure 4 battery |
| `docs/` | Design specs and plans for the harness (records) |

## Data on Hugging Face

`experiments/sec4_causal/` — `checkpoints/<wave>/` (rollouts), `assets/` (directions), `analysis/`
(per-wave analysis JSON). The harness code tarball the cluster jobs download is
`experiments/multilayer_causal/code/multilayer_causal.tar.gz`.

## Figures and tables (paper repository, private)

- `scripts/figures/fig04_causal_battery.py` — Figure 4 and the removal figure
  (`fig04b_causal_removal.pdf`).
- `scripts/tables/body_tables.py` — `tab:causal-battery-suffnec`.
- `scripts/tables/appendix_neural_tables.py`, `scripts/figures/fig_cross_context_write.py` —
  `tab:causal-transfer-matrix`, `tab:causal-condition-writability` and the cross-task panels.
