# Track D — §4.3 distributed-effect robustness (top-K SAE feature removal)

Plan v4 Track D priority 3, robustness-only. Spec at
`/home/v-seungplee/llm-addiction/PLAN_4NODE_EXECUTION_2026_05_07.md` §5.

## Intent

Reviewer 2 attack: "The §4.3 single-feature null contradicts your §4
distributed readout claim. If the readout is real, ablating one feature
should move the behaviour."

This module forestalls that attack by demonstrating that the §4 readout
effect lives in a *distributed* subset of SAE features in the K~50 regime,
not in any single feature. We do this by paired top-K vs random-K SAE
feature removal: top-K removal collapses the §4.3 Δ_G modulation effect,
random-K removal does not.

Track D is **robustness only**. A positive Track D does NOT substitute for
Track 0 (matched-cap mechanism replication) or Track A (construct validity).
A negative Track D triggers a §6 limitations sentence (Branch 3 in
`claim_surgery_D_outcome_branches.md`), it does not invalidate §3 / §4.

## Hypothesis

`H_D_main`: At K=50 SAE features (selected by `|spearmanr(F[:, j],
y_deconfounded_iba)|` on the training fold per the §4.1 selection rule),

* targeted top-K removal collapses Δ_G of Table 3,
* uniformly random K-subset removal does not,
* the paired bootstrap of `Δ_G_random_K − Δ_G_top_K` has lower 95% CI > 0.

Pre-registered primary verdict (frozen before computation in `configs/d_config.yaml`):

| Test | Threshold |
|---|---|
| Primary: `Δ_G_random_50 − Δ_G_top_50 > 0` (paired bootstrap CI lower bound) | passes if true |
| Secondary K=10, K=100 reported as robustness curve | not gating |

## Files

```
d_distributed_effect/
├── configs/
│   └── d_config.yaml                       # K-values, paths, pre-registered primary K and α
├── src/
│   ├── topk_removal.py                     # fit_groupkfold_with_removal + selection helpers
│   ├── run_d_topk_removal.py               # CLI: per (model, K, removal_type) Δ_G
│   └── analyze_d.py                        # paired bootstrap + outcome classification
├── tests/
│   └── test_d_smoke.py                     # leakage, distributed/atomistic synthetic, real Δ_G
├── claim_surgery_D_outcome_branches.md     # 3 §4.3 prose branches keyed to outcome
└── README.md                               # this file
```

## Reuse

Track D imports the canonical sae_v3_analysis pipeline rather than
reimplementing it. The only new code is the in-fold removal step, the
random replicate driver, and the paired bootstrap.

| Component | Source of truth |
|---|---|
| `nl_deconfound_split` (RF on [bal, rn, bal², log1p(bal), bal·rn]) | `sae_v3_analysis/src/run_perm_null_ilc.py` |
| `TOP_K = 200` and `RIDGE_ALPHA = 100.0` constants | `sae_v3_analysis/src/run_perm_null_ilc.py` |
| `load_sae_and_meta` (sparse SAE features + meta loader) | `sae_v3_analysis/src/run_perm_null_ilc.py` |
| `compute_iba` (bet/balance per round) | `sae_v3_analysis/src/run_comprehensive_robustness.py` |
| `compute_loss_chasing_continuous`, `get_meta_field` | `sae_v3_analysis/src/run_groupkfold_recompute.py` |
| Original Δ_G value (Gemma SM I_BA L22 plus_G − minus_G = 0.0903) | `sae_v3_analysis/results/condition_modulation_groupkfold_L22.json` |
| GroupKFold + within-fold deconfound + Spearman top-200 + Ridge | re-implemented inline in `topk_removal.fit_groupkfold_with_removal` so the in-fold removal step can be inserted between the active filter and the top-200 selection — algebraically identical pipeline at `removal_type='none'` or `K=0` |

The fit_groupkfold_with_removal wrapper's parity with the canonical
fit_groupkfold can be verified by comparing R² at K=0 / removal_type='none'
to the values in `results/condition_modulation_groupkfold_L22.json`.

## Running

```bash
cd /home/v-seungplee/llm-addiction/paper_experiments/d_distributed_effect

# 1. Per-cell ablation runs (1 top + 50 random per K-value, both models, ~30 min total)
for M in gemma llama; do
  for K in 10 50 100; do
    python src/run_d_topk_removal.py \
      --config configs/d_config.yaml \
      --model $M --K $K \
      --paradigm-dir slot_machine
  done
done

# 2. Paired bootstrap analysis + outcome classification
python src/analyze_d.py \
  --config configs/d_config.yaml \
  --paradigm-dir slot_machine \
  --model all

# Smoke tests (no GPU required)
pytest tests/ -q
pytest tests/ -q -m "not slow"   # selection + bootstrap + classification only
```

## Outputs

Under `/scratch/x3415a02/data/llm-addiction/d_robustness/` (or
wherever `output.root` points to in `d_config.yaml`):

```
per_run/                                                                 # produced by step 1
├── gemma_slot_machine_L22_i_ba_K10_top.json
├── gemma_slot_machine_L22_i_ba_K10_random_seed_1000.json
├── gemma_slot_machine_L22_i_ba_K10_random_seed_1001.json
├── ...
├── gemma_slot_machine_L22_i_ba_K50_top.json
├── gemma_slot_machine_L22_i_ba_K100_top.json
└── (same pattern for llama)

results/                                                                 # produced by step 2
└── d_analysis_slot_machine_i_ba_L22.json                                # paired bootstrap + verdict
```

Each per-run JSON contains `plus_G.per_fold_r2`, `minus_G.per_fold_r2`,
and the resulting `delta_g`. The analysis script aggregates across the 50
random replicates, runs the 1000-iter paired fold-resampling bootstrap,
and writes the outcome branch.

## Interpretation

Read `d_analysis_slot_machine_i_ba_L22.json::models.gemma.outcome_branch`:

| Branch | Action |
|---|---|
| `D-passes` | Use Branch 1 of `claim_surgery_D_outcome_branches.md` for §4.3 |
| `D-mixed`  | Use Branch 2; report the K curve in an appendix |
| `D-fails`  | Use Branch 3; add the §6 limitations sentence about diffuse representation |

## Open issues

1. **Bootstrap unit granularity.** The paired bootstrap resamples *fold
   indices* (5 folds × paired across replicates), not individual game_ids.
   This is a coarser unit than per-game resampling but preserves the
   GroupKFold partition structure, so it is the cleanest "by game_id at
   fold granularity" available without refactoring the fold loop. If the
   primary CI ends up borderline (lower bound in [-0.005, +0.01]), upgrade
   to per-game bootstrap by saving per-row predictions inside
   `fit_groupkfold_with_removal` and resampling games.

2. **GemmaScope active-feature pool size.** With ~131k features but only
   the active-filter (`nnz > 10`) survivors, the remaining pool is on the
   order of a few thousand features. K=100 random removal therefore
   touches roughly 100 / few-thousand ≈ 2-5% of features per replicate.
   This is enough granularity for the K=50 primary to be informative, but
   if the active pool turns out smaller than expected at run time, the
   analyze step will log the active-feature count per cell — re-tune
   K_values upward if needed (cap K ≤ active_feature_count // 2).

3. **Re-using selection criterion across removal and readout.** The same
   `|spearman ρ|` rank is used both for the top-K-to-REMOVE selection and
   the top-200-to-KEEP selection inside the Ridge readout. After dropping
   the top-K, the readout selects from rank K+1..K+200 of the survivors.
   This is the natural design for an honest "what if we removed the
   most-correlated K" test, but means the post-removal Ridge sees
   strictly weaker features than the original Ridge — by construction.
   The random-K control is the necessary apples-to-apples baseline.

## References

* Elhage et al. 2022, "Toy Models of Superposition" — predicts distributed
  feature codes that survive single-feature ablation.
* Park et al. 2024, "The Linear Representation Hypothesis" — concepts as
  K > 1 directions in SAE dictionary.
* Geiger et al. 2024, "Distributed Alignment Search" — formal causal
  framework for low-rank rotation-subspace interventions.
* Plan v4 §5 — pre-registered procedure and primary K=50 verdict.
