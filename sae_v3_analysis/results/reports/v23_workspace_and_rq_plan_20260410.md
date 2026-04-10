# V23: Workspace Curation And Next-Experiment Plan

**Date**: 2026-04-10  
**Scope**: repository cleanup, provenance hardening, and next neural experiments  
**Goal**: make paper-critical assets easy to find, preserve reusable history,
and run only the highest-value additional experiments.

---

## 1. Problem Statement

The workspace already contains the ingredients needed to support the Korean
paper, but three issues remain:

1. paper-critical assets and historical snapshots still share the same top
   level, which makes provenance checks slower than necessary;
2. RQ1 has a strong smoke-level selectivity control path, but the full
   reviewer-facing batch is not yet closed;
3. RQ2 shows behavioral convergence and partial hidden-state shared geometry,
   but the common substrate is not yet tested in the strongest feasible way.

The plan below separates cleanup from scientific follow-up so that we do not
break runtime paths while trying to improve paper readiness.

---

## 2. Critique Loop

### Iteration 1: aggressive cleanup

**Initial idea**

- move older result files and exploratory code into `legacy/`

**Critique**

- too risky for raw runtime assets
- too many existing scripts still assume the current file layout
- moving `json/`, `logs/`, or `src/` families would create silent path breakage

**Decision**

- keep runtime paths stable
- move only presentation/history artifacts that are not used by active runners

### Iteration 2: no cleanup, only more docs

**Initial idea**

- leave all files in place and add a single new index

**Critique**

- insufficient reduction in clutter
- top-level build artifacts and stale monitor files still obscure current assets

**Decision**

- add stronger manifests and indexes
- also move safe historical clutter into `results/legacy/`

### Iteration 3: raw cross-domain steering as primary RQ2 follow-up

**Initial idea**

- extend `run_v12_crossdomain_steering.py` with more pairs and trials

**Critique**

- this keeps testing the wrong hypothesis
- current evidence already suggests that raw 1:1 direction transfer is too
  strong a criterion when readouts are task-specific

**Decision**

- do not prioritize more raw cross-domain steering

### Iteration 4: matched interchange as first new causal experiment

**Initial idea**

- jump straight to donor-recipient hidden-state interchange

**Critique**

- strongest conceptually, but current saved assets do not yet give the cleanest
  matched all-round bank across models and tasks
- implementation cost is high relative to immediate paper value

**Decision**

- keep interchange as future/high-value work, not the first blocker

### Iteration 5: aligned low-rank RQ2 follow-up

**Initial idea**

- learn a shared low-rank hidden subspace and test held-out aligned transfer

**Critique**

- needs strict controls to avoid leakage or post-hoc overfitting
- must be sequenced after workspace curation and RQ1 selectivity closure

**Decision**

- accept as the primary new RQ2 experiment

### Iteration 6: first implementation of aligned hidden transfer

**Initial idea**

- treat aligned transfer as projecting the source readout into a shared basis
  and scoring the target directly

**Critique**

- this still behaves too much like raw source-to-target transfer
- it mismatches the intended hypothesis that geometry may be shared while the
  final readout remains task-specific
- the first smoke confirmed that issue: `readout_pca` aligned transfer was
  mostly below strong random-basis baselines

**Decision**

- redefine the main RQ2 test as:
  - build the shared basis on non-target train tasks only
  - fit the target readout inside that fixed basis on target-train labels
  - compare against random and shuffled bases
- keep raw source-readout transfer only as an auxiliary negative baseline

### Iteration 7: centroid versus readout basis

**Initial idea**

- use task readout vectors as the default shared-basis carrier

**Critique**

- smoke results show `readout_pca` is too weak for the intended story
- a centroid-difference basis is better aligned with the current hypothesis of
  coarse shared risk geometry plus task-specific decision rules

**Decision**

- prefer `centroid_pca` over `readout_pca` for the next RQ2 runs
- keep `readout_pca` as a negative diagnostic, not the mainline

---

## 3. Final Workspace Plan

### Plan W1. Preserve runtime paths

**Intention**

- reduce clutter without breaking scripts

**Hypothesis**

- keeping `src/`, `results/json/`, `results/logs/`, `results/robustness/`, and
  external data roots fixed will avoid path regressions during cleanup

**Validation**

- no current runner loses its referenced input or output path
- active `v16` jobs continue writing to their existing logs

### Plan W2. Strengthen provenance manifests

**Intention**

- make paper-critical behavior, neural analyses, and HF assets discoverable in
  one pass

**Hypothesis**

- adding a stronger runbook plus updated indexes will lower ambiguity around
  what is paper-safe versus legacy

**Validation**

- each paper claim can be traced through manifest -> artifact -> script ->
  upstream data path

### Plan W3. Move only safe historical clutter

**Intention**

- clear the top-level results directory while preserving reusable history

**Hypothesis**

- moving stale build companions and old monitor traces to `results/legacy/`
  improves navigation without affecting reproducibility

**Validation**

- moved files are not referenced by active scripts
- only presentation/history artifacts are moved

---

## 4. Final Experiment Plan

### Plan E1. Full RQ1 selectivity-control batch

**Intention**

- close the main reviewer question that the neural signal might be a nuisance
  readout

**Hypothesis**

- `real_group_r2` stays positive and beats nuisance-matched controls under
  `GroupKFold(game_id)`

**Validation**

- configs:
  - `gemma:sm:24:i_lc`
  - `llama:sm:16:i_lc`
  - `gemma:mw:24:i_ba`
  - `llama:mw:16:i_ba`
- required outputs:
  - `real_group_r2`
  - `standard_kfold_r2`
  - `control_r2_mean`
  - `selectivity_gap`
  - `p_selectivity`
- control construction:
  - fold-local nuisance bins
  - fold-local game-strata reassignment
  - no test-fold information used when generating the control target

**Why this is first**

- it matches the actual paper pipeline and is cheaper than new GPU-heavy work

### Plan E2. Aligned low-rank hidden-subspace transfer

**Intention**

- test whether the missing common substrate in RQ2 is a shared hidden geometry
  rather than shared sparse features or raw local directions

**Hypothesis**

- a target readout trained inside a non-target shared basis will outperform
  random and shuffled bases on held-out target data, even if it still trails a
  full target-specific readout

**Validation**

- estimate the shared basis on non-target train data only
- compare:
  - raw source-readout transfer
  - target readout in aligned basis
  - target readout in random orthogonal basis
  - target readout in label-shuffled basis
- score with held-out AUC

**Mandatory controls**

- no target-label information in basis fitting
- target labels allowed only when fitting the target readout after the basis is
  frozen
- identical dimensionality and regularization in aligned, random, and shuffled
  basis baselines

**Current status**

- `readout_pca` smoke is not good enough for expansion
- `centroid_pca` smoke is directionally better:
  - positive over random/shuffled on some Gemma directions
  - still clearly below the full target readout
- the current paper-safe interpretation is therefore:
  - partial shared geometry is plausible
  - full shared readout reuse is not supported

### Plan E3. Shared-versus-specific readout decomposition

**Intention**

- quantify how much of each task readout lives in the shared subspace versus a
  task-specific residual

**Hypothesis**

- each task readout decomposes into:
  - a shared component aligned with common hidden geometry
  - a residual component that explains task-specific divergence

**Validation**

- report:
  - shared projection norm fraction
  - residual norm fraction
  - target performance using shared-only, residual-only, and full readout

**Why it matters**

- this turns the current verbal RQ2 interpretation into a quantitative one
- it also tells us whether RQ2 should remain a bounded auxiliary claim or earn
  more experimental expansion

### Plan E4. Aligned-factor steering

**Intention**

- test whether the aligned shared factor is intervention-relevant

**Hypothesis**

- steering with a target-mapped shared factor will be cleaner than raw
  cross-task direction steering

**Validation**

- compare:
  - same-domain steering
  - raw source-to-target steering
  - aligned-factor steering
  - random-factor steering
- require interpretable dose-response and permutation advantage

**Gate**

- run only if E2 and E3 are positive and stable

### Plan E5. Matched interchange

**Intention**

- strongest future causal-abstraction test

**Hypothesis**

- matched donor states should change the next risky action in recipients

**Validation**

- same-class donor control
- random donor control
- wrong-layer control

**Status**

- future work, not an immediate blocker

---

## 5. RQ Gap Audit

### RQ1

- needs one more step:
  - full selectivity batch on the corrected fold-local control script

### RQ2

- still needs one bounded next step:
  - confirm whether `centroid_pca` retains a stable advantage over random and
    shuffled bases on fuller splits
- if not, freeze RQ2 at:
  - behavioral convergence
  - sparse-feature transfer failure
  - partial hidden-state geometry with task-specific residuals

### RQ3

- not the main blocker right now
- existing condition modulation is already paper-usable
- more RQ3 work is appendix/robustness, not the highest-value next step

---

## 6. Execution Order

1. workspace curation and manifest hardening
2. safe legacy moves and generated-noise cleanup
3. full RQ1 selectivity controls on the corrected fold-local script
4. RQ2 `centroid_pca` smoke confirmation
5. Gemma-only fuller RQ2 run if step 4 remains positive
6. readout decomposition analysis and paper-safe interpretation update
7. aligned-factor steering only if step 5 shows a stable basis advantage

---

## 7. Stop Rules

### Stop aligned-transfer expansion if:

- aligned basis readout does not beat random and shuffled bases
- the gain appears only in smoke-scale runs and collapses on fuller splits
- the shared component is too small or too inconsistent to support a stronger
  RQ2 claim

### Stop new causal intervention work if:

- the shared factor is not stable enough to define a target-mapped intervention
- the intervention does not beat random controls

---

## 8. Active Compute Status At Planning Time

- local:
  - one A100 is occupied by the running `llama v16`
- external nodes:
  - `e8` should not be treated as currently available capacity for this
    repository from this shell session
  - node policy also marks `metacognition_e8` as owned by a different project

This means the immediate path is:

1. local RQ1 CPU-bound runs
2. local hidden-state RQ2 smokes
3. only then reconsider remote scheduling from an explicitly allowed host
