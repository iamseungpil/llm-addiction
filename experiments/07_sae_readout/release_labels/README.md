# `release_labels/` — staged labels for the released dataset

The files under this directory are not part of the code. They are the **label files
for the HF dataset**, kept here as real files so they can be read, reviewed and
grepped in this repository, and uploaded verbatim by
`../scripts/label_legacy_deprecations.py`.

```
release_labels/
└── legacy/
    ├── README.md                              -> legacy/README.md
    ├── v17_leaky_pipeline/
    │   └── DEPRECATION_WARNING.md             -> legacy/v17_leaky_pipeline/DEPRECATION_WARNING.md
    └── pre_groupkfold_sweep/
        └── DEPRECATION_WARNING.md             -> legacy/pre_groupkfold_sweep/DEPRECATION_WARNING.md
```

The paths on the left are relative to this directory; the paths on the right are
where each file goes in the dataset repo `llm-addiction-research/llm-addiction`.

## Why

Two retired directories on the dataset, `legacy/v17_leaky_pipeline/` and
`legacy/pre_groupkfold_sweep/`, carry a do-not-cite `README.md` but not the
`DEPRECATION_WARNING.md` filename the release uses everywhere else. Only three of
those files exist in the whole release — under `sae_patching/`,
`slot_machine/gemma/` and `slot_machine/llama/` — so a reader grepping the release by
filename sees those three and concludes the two leaky-pipeline directories are clean.
They are not: both were produced by label-leaking pipelines and are listed do-not-cite
in `NEURIPS_CANONICAL_INDEX.md` §5.

## Uploading

Nothing here has been pushed. The uploader adds files only — no copies, no deletes —
and is a dry run unless `--push` is passed:

```
python3 experiments/07_sae_readout/scripts/label_legacy_deprecations.py           # show what would go
HF_TOKEN=... python3 experiments/07_sae_readout/scripts/label_legacy_deprecations.py --push
```

Pushing publishes to a public dataset, so it is left as a deliberate, separate step
for the authors rather than something a maintenance pass does on its own.
