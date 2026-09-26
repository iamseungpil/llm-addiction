# e2_coding — game-log baseline and behavioural audits

Additional controls behind the paper's appendix (`tab:added-controls`, `tab:exposure-matched`,
`tab:convergent-codebook`, `tab:instrument-robustness`, and the moving-target sensitivity
section). For the whole-repository map, see the [top-level README](../../README.md).

## Questions

- Does the decision-time internal state predict betting better than the game log alone?
- At the same cumulative stake, does the choosing arm still go broke more often than the forced
  arm?
- Do the cognitive-distortion language results survive when the author-written keyword set is
  replaced by other codebooks?
- Does the goal effect on the moving-target rate survive a stricter definition?

## Scripts (`src/`)

| Script | What it does |
|---|---|
| `nested_baseline.py` | Nested five-fold comparison: game-log observables, SAE features, raw hidden state, and their union, with folds grouped by game and by state |
| `residual_race.py`, `residual_race_folds.py` | The same three-way comparison through the paper's own readout pipeline; the second keeps per-fold R² |
| `exposure_matched.py` | Ruin in the forced and choosing arms compared at equal cumulative stake |
| `moving_target_paper_metric.py` | Reproduces the paper's moving-target rate and adds the stricter definition |
| `multi_instrument_robustness.py` | Reruns the distortion-frequency contrasts under several keyword instruments |
| `convergent_codebook.py`, `bathina_cds.py` | Instrument definitions imported by the script above (`convergent_codebook.FROZEN.py` is the frozen copy) |
| `refusal_audit.py`, `reparse_audit.py` | Audits of what models say when they decline to play, and of parser decisions |
| `axis_decoding.py` | How well the behaviour-built steering direction decodes the betting indicator |
| `build_items.py` | Builds the item set for human coding |

`site/` is a Cloudflare Worker that serves the human-coding interface for those items.

## Running

Every script has hard-coded input and output paths from the original machine
(`/home/v-seungplee/...`); most can be overridden on the command line. Two examples, with the
dataset downloaded to `data/`:

```bash
python experiments/09_audits/src/exposure_matched.py \
    --glob 'data/rebuttal_neurips_2026/matched_cap_mc32/final_*.json' --cap 70 \
    --out exposure_matched.json

python experiments/09_audits/src/moving_target_paper_metric.py \
    --root data --out moving_target_paper_metric.json
```

`nested_baseline.py` also needs the SAE features and hidden states from `sae_features_v3/`, at a
path set inside the script; see `--help` for its options (`--layer`, `--baseline`, `--arm`,
`--groupings`, `--cache`). `residual_race*.py` read `design_v2.npz` from the working directory.

## Outputs

Each script writes one JSON file (default name next to the script, or `--out`).

## Data on Hugging Face

`rebuttal_neurips_2026/nested_baseline_and_audits_e2/` holds the outputs, including
`design_v2.npz`. A copy of this code is at `rebuttal_neurips_2026/code/e2_coding/`.
