# §3.2 Claim Surgery Branches (verbatim from PLAN_4NODE_EXECUTION_2026_05_07.md §9bis)

Per codex Round 9: "Do not wait for results to decide prose." Two §3.2 paragraph variants are committed in advance and selected at Day 5 by Track 0 outcome.

## W3-passes branch (pooled CI excludes 0; ≥4/6 models positive)

> "The slot-machine bankruptcy gap is *freedom-to-choose at root* rather than range expansion. We previously established this on GPT-4o by holding the maximum bet equal between fixed and variable arms across four caps ($10/$30/$50/$70); the variable arm bankrupted more at every cap above $10 while betting smaller average amounts than fixed. Replicating the matched-cap protocol on Gemma-2-9b, LLaMA-3.1-8b, GPT-4o-mini, Claude-3.5-Haiku, and Gemini-2.5-Flash reproduces the dissociation: pooled across the six models the variable-fixed bankruptcy gap at the highest matched cap is `[β_primary] [CI]`, with the same 'small bet, more rounds' signature visible in [N pos / 6] models individually. The mechanism is therefore a stable cross-model property of the slot-machine task, not a GPT-4o specificity."

## W3-mixed branch (pooled CI excludes 0 but only 1–3/6 individual models positive)

> "The slot-machine bankruptcy gap reflects a freedom-to-choose mechanism on average, but with substantial model heterogeneity. Pooled across six models, the variable-fixed bankruptcy gap at the highest matched cap is `[β_primary] [CI]`; however, only [N pos / 6] models show this dissociation individually, with [list] reproducing the GPT-4o pattern and [list] showing range-expansion-dominated behavior or null. We therefore frame the *freedom-to-choose* mechanism as an emergent average property rather than a universal cross-model signature, and we note in §6 that mechanistic decomposition of slot-machine bankruptcy is sensitive to model architecture and instruction-tuning."

## W3-fails branch (pooled CI includes 0 OR <2/6 individual models positive)

> "The matched-cap dissociation reported earlier on GPT-4o does not generalise: replicating the protocol on five additional models yields a pooled bankruptcy-gap estimate of `[β_primary] [CI]` at the highest matched cap, with only [N pos / 6] of the six models showing the original dissociation individually. We therefore narrow the §3.2 claim from "freedom-to-choose at root" to a *GPT-4o-specific mechanistic dissociation*: the cross-model variable-fixed bankruptcy gap (Finding 1) remains a robust behavioral phenomenon, but its causal decomposition into freedom vs range is not stable across architectures. Section 6 takes up the consequence — that behavior-level evidence for the phenomenon does not entail a uniform internal mechanism."

These three branches are committed in `paper_experiments/track0_w3_replication/claim_surgery_§3.2_branches.md`. Day 5 selects the branch by pre-registered decision rule and inserts it verbatim into the rebuttal.
