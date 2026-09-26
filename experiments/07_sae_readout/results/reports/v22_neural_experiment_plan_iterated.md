# V22: Neural Claim Support Plan, Iterated

**Date**: 2026-04-09  
**Goal**: Keep the neural claim strong, but support it with experiments that match the actual data path, current codebase, and realistic compute constraints.

---

## 1. Fixed Fact Base

Before proposing new experiments, the current source of truth has to be explicit.

- The paper-scale RQ1/RQ3 sample sizes are round-level, not game-level.
- Those counts match the all-round SAE extractions recorded in:
  - `slot_machine/*/extraction_summary.json`
  - `investment_choice/*/extraction_summary.json`
  - `mystery_wheel/*/extraction_summary.json`
- Therefore, the main RQ1/RQ3 readout path is best treated as a **round-level SAE sparse feature readout pipeline**, not a decision-point hidden-state-only pipeline.
- Activation steering remains a separate causal probe, mainly for LLaMA slot machine.

This matters because any new robustness experiment should reuse the same representational unit as the paper claim it is meant to strengthen.

---

## 2. Literature Anchors

The plan was filtered through four method families.

### A. Probe selectivity / control-task line

- Core idea: a probe should beat nuisance-preserving controls, not just random labels.
- Why it matters here:
  - our current `R^2` results are strong, but reviewers can still ask whether the readout is exploiting leakage or easy nuisance structure.

### B. Causal abstraction / interchange line

- Core idea: if a high-level variable is real, replacing the relevant internal state in a matched context should change downstream behavior.
- Why it matters here:
  - this is the cleanest route to a stronger causal claim.

### C. Local causal intervention line

- Core idea: causal tracing, mediation-style patching, or targeted steering should outperform random or wrong-subspace interventions.
- Why it matters here:
  - we already have steering infrastructure, so this is more realistic than building an entirely new mechanism.

### D. Distributed alignment line

- Core idea: shared computation may exist as an aligned subspace even when feature-level overlap fails.
- Why it matters here:
  - this is the strongest theoretical response to “negative transfer does not prove no shared mechanism.”

---

## 3. Candidate Experiments

### Candidate 1. Group-held-out SAE Selectivity

- Intention:
  - strengthen RQ1 without changing the paper’s representational unit.
- Hypothesis:
  - the real readout should remain positive under `GroupKFold(game_id)`.
  - nuisance-matched control targets should score materially lower.
- Validation:
  - deconfounding inside each fold
  - train-fold-only feature selection
  - compare `real_r2`, `control_r2_mean`, `selectivity_gap`, `p_selectivity`
- Distinct value:
  - stronger than plain random-label or random-feature baselines.

### Candidate 2. Matched Interchange Intervention

- Intention:
  - test whether a matched internal state swap changes the next risky action.
- Hypothesis:
  - under same round / balance bin / prompt condition / previous outcome, high-risk donor state should increase next-step risk in a low-risk recipient.
- Validation:
  - same-class donor control
  - random donor control
  - wrong-layer control
  - immediate next action as the primary target
- Distinct value:
  - strongest causal abstraction style test.

### Candidate 3. Steering Specificity Expansion

- Intention:
  - make the current causal evidence less vulnerable to the “any perturbation works” critique.
- Hypothesis:
  - the bankruptcy-related direction should beat norm-matched random directions and wrong-domain directions.
- Validation:
  - monotonicity
  - permutation p-value against random directions
  - same-domain vs cross-domain sign behavior
- Distinct value:
  - closest to current infrastructure, lowest implementation risk on GPU.

### Candidate 4. Distributed Alignment Search / low-rank shared factor test

- Intention:
  - test whether cross-task failure at the feature level hides a smaller shared aligned subspace.
- Hypothesis:
  - if a shared low-rank factor exists, alignment should recover cross-task predictive structure above null.
- Validation:
  - held-out transfer after learned alignment
  - compare against raw transfer and random alignment
- Distinct value:
  - theoretically attractive, but it risks becoming a new paper inside the paper.

---

## 4. Critique Loop

### Iteration 1

- Proposed winner: matched interchange.
- Critique:
  - strongest conceptually, but current saved assets do not give a clean all-round raw hidden-state bank across both models and all paradigms.
  - implementing this first would delay the paper while leaving RQ1 still vulnerable.
- Decision:
  - not first.

### Iteration 2

- Proposed winner: hidden-state selectivity.
- Critique:
  - this mismatches the paper’s current round-level SAE readout path.
  - even a positive result would not directly validate the table the paper actually reports.
- Decision:
  - reject as the immediate next experiment.

### Iteration 3

- Proposed winner: group-held-out SAE selectivity.
- Critique:
  - directly matches the paper pipeline.
  - CPU-feasible.
  - answers a real reviewer concern.
- Decision:
  - accept as **Primary Next Experiment**.

### Iteration 4

- Secondary choice after selectivity:
  - steering specificity expansion.
- Critique:
  - this does not solve RQ1, but it strengthens the bounded causal paragraph.
  - it is already consistent with the running `v16` steering work.
- Decision:
  - accept as **Primary Causal Follow-up**.

### Iteration 5

- Long-horizon option:
  - matched interchange.
- Critique:
  - still desirable, but only after raw all-round hidden-state assets and donor-recipient matching rules are nailed down.
- Decision:
  - keep as **future high-value experiment**, not current blocker.

---

## 5. Final Plan

### Plan A. SAE Selectivity Control

- Intention:
  - support the strong claim that the neural signal is not just a nuisance readout.
- Hypothesis:
  - `real_r2 > control_r2_mean` with a clearly positive gap under `GroupKFold(game_id)`.
- Validation:
  - configs:
    - `gemma:sm:24:i_lc`
    - `llama:sm:16:i_lc`
    - `gemma:mw:24:i_ba`
    - `llama:mw:16:i_ba`
  - report:
    - `real_group_r2`
    - `standard_kfold_r2`
    - `control_r2_mean`
    - `selectivity_gap`
    - `p_selectivity`
- Success criterion:
  - at least the two primary `I_LC` configs stay positive and clearly beat controls.

### Plan B. Symmetric Metric Audit

- Intention:
  - remove avoidable asymmetry between `I_LC` and `I_BA`.
- Hypothesis:
  - after fixing metadata and `I_BA` extraction issues, the selectivity pipeline should run symmetrically on the intended configurations.
- Validation:
  - smoke on all four configs before any long run.
- Success criterion:
  - no silent metadata fallback, no prompt-condition collapse to `UNK`, no obviously broken `I_BA` extraction.

### Plan C. Steering Specificity Follow-up

- Intention:
  - keep the causal paragraph strong but bounded.
- Hypothesis:
  - same-domain steering should beat random directions more reliably than cross-domain or wrong-sign controls.
- Validation:
  - use currently running `v16` results first
  - only add new steering runs if the ongoing multilayer result is promising
- Success criterion:
  - at least one same-domain result with a clean permutation margin and interpretable dose-response.

---

## 6. What Not To Claim Yet

- A single universal mechanism across all gambling paradigms.
- A completed causal abstraction result.
- Feature-transfer failure as proof that no shared structure exists at any level.

---

## 7. Go / No-Go Rule For Autoresearch

Autoresearch is allowed only if all three conditions hold.

1. The code path matches the paper claim it is supposed to support.
2. Smoke tests show no obvious metadata or target-construction bug.
3. The first representative config produces a sensible positive-vs-control pattern.

If any of these fail, stop and fix the pipeline before running the full batch.
