# M5 outcome branches — §4.4 paragraph drafts

Three pre-written §4.4 prose branches keyed to the M5 analysis verdict. The
branch chosen depends on `m5_analysis_*.json::models.gemma.outcome_branch`
(and analogously for LLaMA — we report both, but the headline cell is
Gemma SM I_BA at L22).

The three thresholds applied are pre-registered (Plan v4 §3.2): individual
relative drop < 0.30, joint relative drop < 0.50, with absolute fallback
0.01 / 0.015 when |Δ_G| < 0.005.

---

## Branch 1 — M5-passes (all three individual + joint hold)

> The §4.3 condition modulation effect — Δ_G = 0.090 (R²_+G = 0.153, R²_−G = 0.063)
> for Gemma I_BA at L22 — is robust to projection out of three pre-registered
> compliance directions. Residualising the SAE feature space against d_comp
> (instruction-following), d_agree (sycophancy), and d_role (persona adoption)
> individually preserved Δ_G within {30%} of its original magnitude in every case;
> the joint three-direction projection preserved it within {50%}. We interpret
> this as evidence that the readout reflects a propensity-like internal state
> that is at most weakly entangled with surface compliance encoding. The
> §4.4 claim that "autonomy modulates an internal-state representation
> readable from L22 SAE features" therefore stands as written; the
> compliance-confound objection is bounded.

(*Fill in the actual passing percentages from the JSON before submitting.*)

---

## Branch 2 — M5-partial (joint passes but ≥1 individual fails, OR vice versa)

> The §4.3 condition modulation effect partially overlaps the compliance
> direction battery but is not dominantly explained by it. After joint
> three-direction residualisation, Δ_G dropped from 0.090 to {Δ_G_joint},
> a {pct}% reduction — within our pre-registered 50% bound but
> non-trivial. The individual-direction projections reveal that {direction_X}
> overlap is the largest single contributor: residualising against
> {direction_X} alone reduced Δ_G to {Δ_G_X} ({pct_X}% drop), exceeding
> the 30% individual threshold. We therefore narrow the §4.4 claim to:
> "autonomy modulates an internal-state representation that is partially
> aligned with {direction_X} encoding but is not reducible to it." We do
> not claim a clean dissociation between propensity and compliance, but we
> rule out the strong reviewer attack that the readout is driven entirely
> by compliance representation. The random-direction control (residualising
> against unit Gaussian directions in d_model space) preserved Δ_G within
> {control_pct}% in every replicate, indicating the projection method itself
> is not over-correcting.

---

## Branch 3 — M5-fails (joint residualisation collapses Δ_G past 50%)

> Compliance-direction residualisation collapses the §4.3 readout effect.
> After joint projection of SAE features onto the orthogonal complement of
> span(d_comp, d_agree, d_role) at L22, Δ_G drops from 0.090 to {Δ_G_joint},
> a {pct}% reduction — exceeding our pre-registered 50% bound. We narrow
> the §4.4 claim accordingly: from "autonomy modulates an internal-state
> readability of betting aggressiveness" to "autonomy modulates a
> representation that overlaps substantially with instruction-compliance
> encoding". The readout effect is real (random-direction controls preserve
> it) and the autonomy condition continues to behaviorally raise risk-taking
> outcomes in §3, but the *interpretation* of the hidden-state signature
> shifts from a propensity-like construct toward a surface compliance /
> role-adoption trace. We retain the §4.3 result as a positive finding
> about condition modulation while explicitly disclaiming the
> propensity-construct reading. Branch 3 also implies §4.4 should not be
> cited as evidence that internal states encode "addictive disposition";
> it should be cited as evidence that internal states encode the
> condition-prompt context strongly enough that condition-conditional
> readouts of behavioural indicators succeed.
