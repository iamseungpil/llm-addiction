# 09 — Appendix controls and audits

**Questions.**

- Does the decision-time internal state predict betting better than the game log alone?
- At the same cumulative stake, does the choosing arm still go broke more often than the forced
  arm?
- Do the gambling-related language results survive when the author-written keyword set is replaced
  by other codebooks?
- Does the goal effect on the moving-target rate survive a stricter definition?

**Paper result.** Appendix: `tab:added-controls` (game-log baseline: the sparse SAE block adds
ΔR² +0.044 with folds grouped by game and +0.0024 grouped by state hash, below the pre-set 0.017
margin; the raw hidden state adds +0.059 under both), `tab:exposure-matched` (24 of 24 threshold
cells favour the forced arm), `tab:convergent-codebook`, `tab:instrument-robustness`, and the
moving-target sensitivity section (App. G.6: 4.2% of API escalation events flagged as possibly
mis-extracted; the stricter after-reaching rule gives 2.24× vs 2.83×).

## Design (from the code)

| Script (`src/`) | What it does |
|---|---|
| `nested_baseline.py` | Nested five-fold comparison: game-log observables, SAE features, raw hidden state and their union, with folds grouped by game and by state hash |
| `residual_race.py`, `residual_race_folds.py` | The same three-way comparison through the paper's own readout pipeline (`experiments/07_sae_readout/src/run_perm_null_ilc.py`); the second keeps per-fold R² |
| `exposure_matched.py` | Ruin in the forced and choosing arms compared at equal cumulative stake |
| `moving_target_paper_metric.py` | Reproduces the paper's moving-target rate and adds the stricter definition |
| `multi_instrument_robustness.py` | Reruns the language-marker contrasts under several keyword instruments, with the loader of `experiments/07_sae_readout/src/run_multimodel_distortion_analysis.py` |
| `convergent_codebook.py`, `bathina_cds.py` | Instrument definitions imported by the script above (`convergent_codebook.FROZEN.py` is the frozen copy whose SHA-256 the paper prints) |
| `refusal_audit.py`, `reparse_audit.py` | Audits of what models say when they decline to play, and of parser decisions |
| `axis_decoding.py` | How well the behaviour-built steering direction decodes the betting indicator |
| `build_items.py` | Builds the item set for human coding; `site/` is a Cloudflare Worker that serves the coding interface |

## Quick Start

With the dataset downloaded to `data/` (see the [top-level README](../../README.md#getting-the-data)):

```bash
python experiments/09_audits/src/exposure_matched.py \
    --glob 'data/rebuttal_neurips_2026/matched_cap_mc32/final_*.json' --cap 70 \
    --out exposure_matched.json

python experiments/09_audits/src/moving_target_paper_metric.py \
    --root data --out moving_target_paper_metric.json

python experiments/09_audits/src/nested_baseline.py --help
```

Most scripts default to input and output paths on the original machine (`/home/v-seungplee/...`);
override them on the command line. `nested_baseline.py` also needs the SAE features and hidden
states from `sae_features_v3/` (options `--layer`, `--baseline`, `--arm`, `--groupings`,
`--cache`). `residual_race*.py` read `design_v2.npz` from the working directory. Each script writes
one JSON file.

## Data on Hugging Face

`rebuttal_neurips_2026/nested_baseline_and_audits_e2/` holds the outputs, including
`design_v2.npz`. A copy of this code is at `rebuttal_neurips_2026/code/e2_coding/`.

## Figures and tables (paper repository, private)

These appendix tables are written directly in `neurips_content_en/appendix.tex` from the JSON
outputs above; no generator script for them exists in the paper repository.
