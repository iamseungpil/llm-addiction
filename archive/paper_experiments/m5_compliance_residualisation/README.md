# M5 — Internal-state residualisation against compliance directions

Plan v4 Track A2 priority 2. Spec at
`/home/v-seungplee/llm-addiction/PLAN_4NODE_EXECUTION_2026_05_07.md` §3.

## Intent

Reviewer 2 attack: "The §4 internal-state readout encodes compliance, not
propensity." This module tests whether the §4.3 condition modulation
(Table 3 Δ_G effect on Gemma I_BA at L22, 0.063 → 0.153 = +0.090) survives
projection out of three pre-registered compliance directions:

* `d_comp` — instruction following (follow-strictly vs ignore)
* `d_agree` — sycophancy / agreeableness (SycEval Fanous 2025 style)
* `d_role` — persona adoption (Wang 2024 style role-play vs neutral)

If Δ_G survives both individual residualisation (< 30% drop per direction)
and joint three-direction projection (< 50% drop), the readout is robust
and the §4.4 claim stands. Otherwise the paper narrows the claim per the
branches in `claim_surgery_M5_outcome_branches.md`.

## Hypothesis

`H_M5_main`: Δ_G of Gemma I_BA Ridge readout at L22 survives residualisation
against the three-direction compliance battery, both individually and
jointly. Pre-registered thresholds (Plan v4 §3.2) frozen before computation:

| Test | Threshold |
|---|---|
| Individual: `(Δ_G − Δ_G') / Δ_G < 0.30` | passes if true |
| Joint: `(Δ_G − Δ_G_joint) / Δ_G < 0.50` | passes if true |
| Stability rule: if `|Δ_G| < 0.005`, switch to absolute thresholds (0.01 / 0.015) | applied to `|Δ_G − Δ_G'|` |

## Primary contrast

Gemma-2-9b SM I_BA L22 +G − Gemma-2-9b SM I_BA L22 −G, after vs before
hidden-state residualisation. LLaMA-3.1-8b SM I_BA L22 reported in parallel
as a robustness check (the headline cell is Gemma).

## Files

```
m5_compliance_residualisation/
├── configs/
│   └── m5_config.yaml                    # all parameters (models, layer, thresholds, paths)
├── prompts/                              # 50 lines each (50 pos + 50 neg per direction)
│   ├── d_comp_pos.txt    d_comp_neg.txt
│   ├── d_agree_pos.txt   d_agree_neg.txt
│   └── d_role_pos.txt    d_role_neg.txt
├── src/
│   ├── extract_compliance_directions.py  # forward-pass last-token activations → mean diff
│   ├── compute_baseline_dp.py            # Δ_G_dp baseline on decision-point sample (C1 fix)
│   ├── residualise_sae_features.py       # H' = H − HP; re-encode through SAE
│   ├── refit_table3_residualised.py      # GroupKFold Ridge readout (reuses sae_v3_analysis)
│   └── analyze_m5.py                     # Δ_G survival ratios + threshold pass/fail
├── tests/
│   └── test_m5_smoke.py                  # algebra + threshold + fixture smoke tests
├── claim_surgery_M5_outcome_branches.md  # 3 §4.4 prose branches keyed to outcome
└── README.md                             # this file
```

## Reuse

This module deliberately *reuses* the canonical sae_v3_analysis pipeline
rather than reimplementing it:

| Component | Source of truth |
|---|---|
| `fit_groupkfold` (5-fold GroupKFold, RF deconfound, top-200 Spearman, Ridge α=100) | `sae_v3_analysis/src/run_groupkfold_recompute.py` |
| `nl_deconfound_split` | `sae_v3_analysis/src/run_perm_null_ilc.py` |
| `compute_iba` (bet/balance per round) | `sae_v3_analysis/src/run_comprehensive_robustness.py` |
| `compute_loss_chasing_continuous` | `sae_v3_analysis/src/run_groupkfold_recompute.py` |
| Original Δ_G value (Gemma SM I_BA L22 plus_G − minus_G = 0.0903) | `sae_v3_analysis/results/condition_modulation_groupkfold_L22.json` |
| Hidden-state caches at L22 (Gemma + LLaMA, all paradigms) | `/home/v-seungplee/data/llm-addiction/sae_features_v3/{paradigm}/{model}/hidden_states_dp.npz` |

The new code is the 3-direction probe extraction, the orthogonal-complement
projection, and the threshold-decision logic. Everything else is imported.

## Running

```bash
cd /home/v-seungplee/llm-addiction/paper_experiments/m5_compliance_residualisation

# 1. Extract three compliance directions per model (~30 min × 2 H100s)
python src/extract_compliance_directions.py --config configs/m5_config.yaml --model all

# 2. Compute the decision-point baseline Δ_G_dp on the same 3,200-row sample
#    M5 will residualise (no direction projection — just SAE-encode H and fit).
#    This produces results/delta_g_dp_baseline.json. (Plan v4 §13 deviation
#    2026-05-07; see "Sample-space parity" section below.)
python src/compute_baseline_dp.py --config configs/m5_config.yaml --paradigm-dir slot_machine --model all

# 3. Residualise hidden states + re-encode through SAE per (mode, model)
python src/residualise_sae_features.py --config configs/m5_config.yaml --paradigm-dir slot_machine --model all

# 4. Re-fit Table 3 Ridge readout under same GroupKFold protocol
python src/refit_table3_residualised.py --config configs/m5_config.yaml --paradigm-dir slot_machine --model all

# 5. Compute survival ratios and apply pre-registered thresholds
#    (uses delta_g_dp_baseline.json as the operational Δ_G — not the
#    canonical round-level 0.0903 which lived on a different sample space)
python src/analyze_m5.py --config configs/m5_config.yaml --paradigm-dir slot_machine

# Smoke tests (no GPU required)
pytest tests/ -q
```

## Sample-space parity (Plan v4 §13 deviation 2026-05-07)

**M5 operates on decision-point hidden states (n=3,200), recomputes the
baseline Δ_G_dp on that sample, then compares with residualised Δ_G_dp'.**

The canonical Table 3 Δ_G = 0.0903 (Gemma SM I_BA L22 +G − -G) was fit on
the round-level `sae_features_L22.npz` cache (n ≈ 21,421 rows). M5's
residualisation pipeline operates on the decision-point cache
`hidden_states_dp.npz` (n = 3,200 rows). A residualised Δ_G computed on
the decision-point sample is *not* a residualised version of the
round-level canonical Δ_G — they live on different sample spaces, so
their difference is uninterpretable as a "drop".

The fix (Option B in the C1 critical-issue review): re-baseline Δ_G on
the same 3,200-row sample by SAE-encoding H **without any residualisation**
and fitting the canonical GroupKFold readout. Call this Δ_G_dp. Then the
residualised Δ_G_dp' is a true residualised version of the same baseline
and `|Δ_G_dp − Δ_G_dp'| / |Δ_G_dp|` is well-defined.

Trade-off: the M5 headline cell is no longer the same numerical Δ_G that
appears in §4.3 Table 3 of the paper. We report both Δ_G_dp (operational
baseline used for pass/fail) and the canonical round-level Δ_G (in the
JSON output's `delta_g_canonical_round_level` field) for transparency.
Plan §3.5 mitigation: the random-direction control test verifies that
the SAE-re-encode step itself does not corrupt the readout.

## Outputs

Under `/scratch/x3415a02/data/llm-addiction/m5_residualisation/` (or
wherever `output.root` points to in the YAML):

```
directions/                                                       # produced by step 1
├── direction_gemma_d_comp.npz       # contains direction (d_model,), h_pos, h_neg, mean_pos, mean_neg
├── direction_gemma_d_agree.npz
├── direction_gemma_d_role.npz
├── direction_llama_d_comp.npz       (and so on)
└── extraction_summary.json

residualised/                                                     # produced by step 2
├── gemma_slot_machine_L22_individual_d_comp.npz                  # sparse SAE features (residualised)
├── gemma_slot_machine_L22_individual_d_agree.npz
├── gemma_slot_machine_L22_individual_d_role.npz
├── gemma_slot_machine_L22_joint_3direction.npz
├── gemma_slot_machine_L22_control_random0.npz                    # baseline (should preserve Δ_G)
└── ...

results/                                                          # produced by steps 2 + 4 + 5
├── delta_g_dp_baseline.json                                      # Δ_G_dp on n=3,200 dp sample (C1 fix)
├── refit_results_slot_machine_i_ba_L22.json                      # per-mode R²_+G, R²_-G
└── m5_analysis_slot_machine_i_ba_L22.json                        # Δ_G survival + pass/fail
```

## Interpretation

Read `m5_analysis_slot_machine_i_ba_L22.json::models.gemma.outcome_branch`:

| Branch | Action |
|---|---|
| `M5-passes` | Use Branch 1 of `claim_surgery_M5_outcome_branches.md` for §4.4 |
| `M5-partial` | Use Branch 2; report which direction overlaps most |
| `M5-fails` | Use Branch 3; narrow §4.4 claim to "compliance-aligned representation" |

The random-direction controls (`control_random*`) should always pass; if
any of them fail, the projection pipeline itself is over-correcting and
the directional results are uninterpretable. Investigate before reporting.

## Open issue: SAE-feature ↔ hidden-state coordinate transformation

The §4.3 readout fits Ridge on **SAE features** F (sparse codes), but the
compliance directions live in **hidden-state space** R^d_model. Two
formulations are coherent (full discussion in
`src/residualise_sae_features.py` module docstring):

* (A, **default**) `hidden_state` mode — project H, then re-encode through
  SAE: `H' = H − HP`, `F' = SAE_encode(H')`. Persona-Vectors-2025 style.
  Cleanest semantics; cost is one SAE forward pass per (model, mode).
* (B) `sae_decoder` mode — project F directly using a feature-space
  projection derived from the SAE encoder. Algebraically exact only for
  linear SAEs; both GemmaScope (JumpReLU) and LlamaScope (ReLU) are
  nonlinear, so this is an approximation. Not implemented; flagged here
  for future cheap-fallback work.

**This is the single biggest open issue for M5.** The hidden_state-mode
default is principled and matches the Persona Vectors 2025 method, but it
adds an SAE-encode step the original §4.3 pipeline did not have. We
mitigate via the random-direction control: if a random unit vector
through the same pipeline preserves Δ_G to within < 5%, then the
re-encoding step itself is not the source of any observed Δ_G drop.

## References

* Persona Vectors 2025, arXiv:2507.21509 — activation-space residualisation method
* Fanous 2025 (SycEval) — sycophancy probe prompts
* Wang 2024 — persona-adoption prompt patterns
* Plan v4 §3 — pre-registered procedure and thresholds
