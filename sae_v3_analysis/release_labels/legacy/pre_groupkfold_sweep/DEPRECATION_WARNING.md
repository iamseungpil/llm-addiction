# DEPRECATION WARNING — `legacy/pre_groupkfold_sweep/`

**Status: DEPRECATED. DO NOT CITE.**

These are the 42-layer peak sweeps run under **random KFold**, before the pipeline
moved to GroupKFold-by-game-id. Random KFold splits rounds, not games, so rounds from
the same game land on both sides of the split and the model reads the held-out fold
through its own training data. The resulting cells diverge from the body by up to a
factor of three, and it is these sweeps that produced the abandoned **L24 / L16
peak-layer route** — the reason the paper fixes a single representative layer, L22,
instead.

## Use instead

```
sae_v3_analysis/results/table1_groupkfold_L22.json               # Table 1
sae_v3_analysis/results/condition_modulation_groupkfold_L22.json # Table 3
sae_v3_analysis/results/rq2_aligned_hidden_transfer_*L22_r1*.json # Table 2

sae_v3_analysis/src/run_groupkfold_recompute.py   # current generator
```

Every neural number in the paper is evaluated at the fixed representative layer L22
under strict GroupKFold-by-game-id. The peak-layer slices that do appear —
`tab:neurips-selectivity-l2` at L24 (Gemma) and L16 (LLaMA) — come from the
strict-CV robustness file, not from this sweep, and the table's own rows say so.

Appendix C.1 (`tab:appendix-sweep-verification`, `tab:appendix-sweep-peak-mismatch`)
reads this directory **on purpose**, as the transparency comparison documenting the
transition. That is the only legitimate use of anything here.

## Why this file exists alongside `README.md`

The `README.md` in this directory already says do-not-cite, but the release's
machine-checkable convention is the filename `DEPRECATION_WARNING.md`. Until this
file was added, a reader grepping the release for that filename found `sae_patching/`,
`slot_machine/gemma/` and `slot_machine/llama/` and concluded — wrongly — that this
directory was clean.

Kept and labelled, never removed. See `legacy/README.md` for the convention.
