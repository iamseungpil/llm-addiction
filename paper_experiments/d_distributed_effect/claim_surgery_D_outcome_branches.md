# Track D outcome branches — §4.3 paragraph drafts

Three pre-written §4.3 prose branches keyed to the paired-bootstrap verdict
in `d_analysis_slot_machine_i_ba_L22.json::models.{model}.outcome_branch`.
The headline cell is Gemma SM I_BA at L22; LLaMA SM I_BA at L22 reported in
parallel as a robustness check.

The pre-registered primary verdict (Plan v4 §5; frozen in `configs/d_config.yaml`):
*lower bound of 95% paired-bootstrap CI on `Δ_G_random_K − Δ_G_top_K` at K=50 > 0.*
Secondary K=10 and K=100 reported as a robustness curve.

Track D is **robustness only**. A positive Track D does NOT substitute for
Track 0 (matched-cap mechanism replication) or Track A (construct validity).
It only forestalls one specific reviewer attack: that the §4.3 single-feature
null contradicts the §4 distributed readout claim.

---

## Branch 1 — D-passes (random > top at K=50, K=10, K=100)

> The §4.3 single-feature null is consistent with a *distributed* readout.
> Removing the K=50 SAE features most correlated with the deconfounded I_BA
> target collapses the §4.3 condition modulation effect (Δ_G drops from
> 0.090 to {Δ_G_top_50}), whereas removing 50 randomly chosen SAE features
> preserves it (mean Δ_G across 50 random replicates = {Δ_G_random_50}).
> The paired bootstrap of `Δ_G_random_50 − Δ_G_top_50` yields a 95%
> confidence interval of [{ci_low_50}, {ci_high_50}], excluding zero. The
> same pattern holds at K=10 and K=100 (CI lower bounds {ci_low_10},
> {ci_low_100}). We conclude that the readout effect is carried by a
> distributed but identifiable subset of SAE features in the tens-to-hundreds
> regime, not by any single feature. The §4.3 atomistic single-feature
> intervention does not falsify the §4 distributed readability claim — it
> tests a stricter (and indeed vacuous) hypothesis. This is the pattern
> predicted by superposition theory (Elhage et al. 2022) and the linear
> representation hypothesis at K > 1 (Park et al. 2024); §4 readability is
> the *collective* causal claim, and it survives.

(*Fill in the actual {Δ_G_top_K}, {Δ_G_random_K}, {ci_low/high} values from
`d_analysis_slot_machine_i_ba_L22.json` before submitting.*)

---

## Branch 2 — D-mixed (K=50 passes but K=10 or K=100 fails)

> The §4.3 condition modulation effect is concentrated in a *moderate* set
> of SAE features around K=50 but distributes beyond the top-K narrow band.
> At K=50 the top-K-removal Δ_G is {Δ_G_top_50} versus a random-K mean of
> {Δ_G_random_50} (95% paired-bootstrap CI [{ci_low_50}, {ci_high_50}],
> excluding zero — the pre-registered primary verdict). At K={K_failing}
> however, the random-vs-top paired bootstrap CI {includes/falls below}
> zero ([{ci_low_K_fail}, {ci_high_K_fail}]), indicating that {very small
> top-K subsets miss most of the signal / very large top-K subsets capture
> only a partial slice}. The picture is therefore one of an effect with a
> *characteristic granularity* near K=50 features: large enough that any
> single feature is null (consistent with §4.3) but small enough that
> moderate-K identification recovers it (the K=50 primary). This is
> consistent with Park et al. (2024), where concept directions span K > 1
> learned features but not unboundedly many. We retain the §4 distributed
> readability claim and report the curve in an appendix.

---

## Branch 3 — D-fails (K=50 paired CI includes 0 or is below 0)

> The §4.3 condition modulation effect is fully distributed across many
> SAE features: even random K=50 removal collapses Δ_G to roughly the same
> degree as targeted top-50 removal. The paired bootstrap of
> `Δ_G_random_50 − Δ_G_top_50` yields a 95% CI of [{ci_low_50},
> {ci_high_50}], which {includes zero / falls below zero}. We acknowledge
> this finding in §6 (Limitations): the §4 readout taps a representation
> diffuse across more than ~100 SAE features, so neither single-feature
> ablation nor moderate-K targeted ablation can produce a focal causal
> intervention. This is consistent with strong superposition (Elhage et al.
> 2022): the relevant code is spread across enough features that
> low-dimensional ablation of any kind is inadequate, and any proper causal
> test would require either Distributed Alignment Search-style rotation
> subspace interventions (Geiger et al. 2024) or steering along a learned
> low-rank direction. We narrow the §4 claim accordingly to: "the readout
> succeeds at the population-of-features level, but the §4.3 single-feature
> null and the Track-D distributed-removal null jointly indicate that no
> low-K SAE-feature subset is a sufficient locus of the effect." The
> behavioural §3 results stand; the mechanistic-locus interpretation
> retreats to a representation-level (not feature-level) claim.

---

## Outcome → file mapping

| Branch | `outcome_branch` value | When |
|---|---|---|
| 1 | `D-passes` | K=50 primary CI low > 0 AND K=10 + K=100 CI low > 0 |
| 2 | `D-mixed`  | K=50 primary CI low > 0 but ≥ 1 of K=10 / K=100 CI low ≤ 0 |
| 3 | `D-fails`  | K=50 primary CI low ≤ 0 |
