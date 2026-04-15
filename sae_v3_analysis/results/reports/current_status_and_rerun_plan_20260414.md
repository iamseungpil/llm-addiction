# Current Status And Rerun Plan

**Date**: 2026-04-14  
**Scope**: post-audit steering status, replay validation, and rerun order  
**Canonical references**:

- `src/exact_behavioral_replay.py`
- `src/run_aligned_factor_steering.py`
- `src/run_v16_multilayer_steering.py`
- `docs/PAPER_CANONICAL.md`
- `results/reports/rq2_causal_experiment_plan_v3.md`

---

## 1. Executive Status

The steering pipeline has been corrected at the **runtime/provenance** level,
but the **claim-bearing reruns have not yet been completed**.

As of 2026-04-14 10:13 UTC:

- active steering processes: `0`
- visible local GPUs: `1 x A100`, idle
- replay catalog validation: `passed`
- steering replay smoke validation: `passed` for both `llama` and `gemma`
- paper-safe steering numbers: **not refreshed yet**

This means the code path is now in a materially better state than the archived
V14/V16 runs, but the paper should still treat post-audit steering as
**pending rerun**, not finished evidence.

---

## 2. What Changed

### 2.1 Old state that was invalidated

Earlier steering runs were archived for two distinct reasons:

1. `results/json/archived_steering_old_prompts/`
   - prompt distribution mismatch against the original behavioral runs
2. `results/json/archived_steering_no_paired_design/`
   - alpha-dependent seeding, which broke paired dose-response comparison

Those outputs remain useful as research history, but not as canonical
post-audit causal evidence.

### 2.2 New canonical runtime path

Claim-bearing steering must now route through:

- `src/exact_behavioral_replay.py`

This bridge:

- loads empirical behavioral condition catalogs from the original raw games,
- replays the original `prompt_condition x bet_type x bet_constraint` mixtures,
- uses deterministic shuffled catalogs so partial runs are not front-loaded by
  source ordering,
- reuses the original behavioral runner logic instead of a rewritten sandbox,
- preserves original model/task prompt asymmetries where they existed.

### 2.3 Updated entry points

The following scripts now use the replay path:

- `src/run_aligned_factor_steering.py`
- `src/run_v16_multilayer_steering.py`

These are now the only steering entry points that should be used for new
paper-facing reruns.

---

## 3. What Has Been Validated

### 3.1 Catalog integrity

The replay bridge can read the expected behavioral catalogs:

| Model | SM | IC | MW |
| --- | ---: | ---: | ---: |
| LLaMA | 3200 | 1600 | 3200 |
| Gemma | 3200 | 1600 | 3200 |

Additional sanity:

- SM prompt conditions: `32`
- IC prompt conditions: `4`
- MW prompt conditions: `32`
- IC constraints recovered: `10, 30, 50, 70`
- MW constraint recovered: `30`

### 3.2 Replay smoke checks

The following validations passed:

- `validate_behavioral_replay("llama") == True`
- `validate_behavioral_replay("gemma") == True`

Interpretation:

- the new steering entry points can load the exact behavioral catalogs;
- the replay bridge is wired correctly enough for rerun launch;
- this is a smoke/provenance pass, not a scientific rerun result.

### 3.3 What is still not validated

The following are still pending:

- full Exp C rerun under exact replay
- full Exp A rerun under exact replay
- full Exp B rerun under exact replay
- full V16 rerun under exact replay
- refreshed JSON outputs and paper numbers derived from those reruns

---

## 4. Current Interpretation Rule

Until new replay-based reruns finish:

- RQ1 and most audited non-steering neural tables remain traceable through
  `results/paper_neural_audit.json` and related audited artifacts.
- Steering claims in the paper should be treated as **awaiting post-audit
  refresh** unless they are explicitly labeled archived or exploratory.
- Older V12/V14/V16 steering outputs can still be read for historical context,
  but they should not be promoted as exact behavioral replications.

---

## 5. Rerun Order

The rerun order remains:

1. **Exp C first**
   - purpose: positive control for intervention sensitivity
   - model/task: LLaMA SM at the target layer
   - decision: if this fails under exact replay, pause broader shared-axis
     interpretation
2. **Exp A and Exp B next**
   - purpose: shared-axis causal tests on LLaMA and Gemma
   - condition: run only after Exp C passes the planned gate
3. **V16 rerun after that**
   - purpose: refresh the paper-facing representative steering result using the
     corrected replay path and alpha-independent seeding

This preserves the plan logic already documented in
`results/reports/rq2_causal_experiment_plan_v3.md`, but updates the runtime
assumption from sandbox steering to exact replay steering.

---

## 6. Practical Launch Checklist

Before restarting GPU jobs:

1. confirm the run uses either `run_aligned_factor_steering.py` or
   `run_v16_multilayer_steering.py`;
2. confirm the script still routes through `src/exact_behavioral_replay.py`;
3. confirm output JSON/log metadata records the replayed behavioral conditions;
4. keep archived wrong-prompt and non-paired results separate from new outputs;
5. refresh paper numbers only from the new replay-based JSONs.

---

## 7. Relationship To Older Status Documents

- `results/session_progress_20260331.md`
  - historical execution log from the first V14 phase
  - no longer authoritative for current steering validity
- `results/reports/v19_status_and_plan.md`
  - archive-level interpretation of neural findings
  - still useful for RQ1/RQ2 framing, but not a runtime status source
- `results/reports/v23_workspace_and_rq_plan_20260410.md`
  - still useful for workspace cleanup and experiment sequencing
  - should now be read together with the exact-replay correction above

---

## 8. Immediate Next Step

The code path is ready for rerun launch, but there is no active steering job at
the moment. The next concrete action is to launch the exact-replay reruns in
the order `Exp C -> Exp A / Exp B -> V16`, then refresh the paper-facing
steering summary only after those JSONs are complete.
