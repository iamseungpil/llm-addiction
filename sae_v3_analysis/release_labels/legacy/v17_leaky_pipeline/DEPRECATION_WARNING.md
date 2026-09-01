# DEPRECATION WARNING — `legacy/v17_leaky_pipeline/`

**Status: DEPRECATED. DO NOT CITE.**

Everything in this directory is V17 SAE-readout output produced with a **leaky RF
deconfound**: the RandomForest is fit on the full target *before* the CV split, which
leaks test-fold labels into the training residual. The effect is not marginal. It
inflates `I_LC` for several cells — LLaMA / mystery-wheel at L22 goes from **0.293**
under strict CV to **0.779** under the leaky pipeline.

## Files

| File | What it is |
|---|---|
| `paper_neural_audit.json` | V17 binary `I_LC` pipeline output |
| `v17_nonlinear_deconfound.txt` | leaky text report |
| `v17_nonlinear_deconfound_REFERENCE.txt` | second copy of the same report |
| `build_paper_neural_audit.py` | the script that produced the above |

## Use instead

The paper cites only the strict-CV GroupKFold-by-game-id pipeline:

```
sae_v3_analysis/results/table1_groupkfold_L22.json               # Table 1
sae_v3_analysis/results/condition_modulation_groupkfold_L22.json # Table 3
sae_v3_analysis/results/rq2_aligned_hidden_transfer_*L22_r1*.json # Table 2

sae_v3_analysis/src/run_groupkfold_recompute.py   # current generator
sae_v3_analysis/src/run_perm_null_ilc.py          # strict within-fold deconfound
```

Appendix C.1 of the paper documents the transition explicitly, in
`tab:appendix-sweep-verification` and `tab:appendix-sweep-peak-mismatch`. Those two
tables read the deprecated pipeline **on purpose**, as the transparency comparison
that shows what the leak did. That is the only legitimate use of anything here.

## Why this file exists alongside `README.md`

The `README.md` in this directory already says do-not-cite, but the release's
machine-checkable convention is the filename `DEPRECATION_WARNING.md`. Until this
file was added, a reader grepping the release for that filename found `sae_patching/`,
`slot_machine/gemma/` and `slot_machine/llama/` and concluded — wrongly — that this
directory was clean.

Kept and labelled, never removed. See `legacy/README.md` for the convention.
