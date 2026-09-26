# V14 Audit and V15 Follow-Up Plan

Date: 2026-04-01
Project: `sae_v3_analysis`

## 1. Code Audit: consistency with existing experiment code

### What is consistent

- The active V14 runners are:
  - `src/run_v14_experiments.py`
  - `src/run_v14_parallel.py`
- Both active runners reuse the existing V12 path directly from `src/run_v12_all_steering.py`:
  - `play_game`
  - `run_condition`
  - `compute_bk_direction`
  - `build_sm_prompt`, `build_ic_prompt`, `build_mw_prompt`
  - `parse_sm_response`, `parse_ic_response`, `parse_mw_response`
- This means the active V14 jobs are using the same prompt templates, parsing rules, game dynamics, seeds-per-trial pattern, and residual-stream hook location as the established V12 implementation.
- Cross-domain V14 also keeps the same target task gameplay semantics because it still calls `run_condition(..., task=...)` from the same V12 implementation.

### What is not fully clean

1. `src/run_v14_causal_validation.py` is stale and should not be treated as the canonical runner.
   - It uses a different code path.
   - It appears to import functions such as `play_sm_game`, `play_ic_game`, and `play_mw_game` from `run_v12_all_steering.py`, but those functions are not defined there.
   - Conclusion: this file is likely obsolete or partially broken. The active runs are not using it, which is good.

2. Reproducibility of random-control seeds in active V14 is not ideal.
   - `run_v14_experiments.py` and `run_v14_parallel.py` use:
     - `seed=hash(exp_name) % (2**31)`
   - Python hash randomization can change across interpreter sessions.
   - This does not invalidate the current running jobs, but it weakens exact rerun reproducibility.

3. V14 comments/docstrings do not fully match the active configuration.
   - Some headers still mention `n=200` or larger random-count settings while the running jobs use:
     - `Exp1`: `n_games=100`, `n_random=20`
     - `Exp2a/2b/4`: `n_games=100`, `n_random=10`
     - `Exp3`: `n_games=50`, `n_random=5`

4. Intermediate checkpointing is missing.
   - JSON files are written only after each experiment finishes.
   - If a long run is interrupted, progress inside the experiment exists only in the text log.

### Audit verdict

- The currently running V14 experiments are implementation-consistent with V12 in the places that matter most:
  - prompts
  - parsers
  - game mechanics
  - hook location
  - hidden-state direction extraction
- The main risk is not prompt drift or gameplay drift.
- The main risks are:
  - exact reproducibility of random-control seeds
  - long-run fault tolerance
  - stale auxiliary script confusion

## 2. Current intermediate readout

As of the latest audit:

- Strong:
  - `Exp1` LLaMA SM
  - `Exp2a` LLaMA IC
- Weak or mixed:
  - `Exp2b` LLaMA MW
  - `Exp3` cross-domain
- Still incomplete:
  - `Exp4` Gemma MW

### Practical interpretation

- The paper can likely support a narrowed causal claim if `Exp1` and `Exp2a` finish cleanly.
- The paper should not rely on strong MW-wide or cross-domain causal generalization unless `Exp2b`, `Exp3`, or `Exp4` improve materially by completion.

## 3. Decision rule after V14 finishes

### Case A: best-case

Condition:
- `Exp1` confirmed
- `Exp2a` confirmed
- and at least one of `Exp2b`, `Exp3`, `Exp4` also confirmed

Action:
- Update the paper with a positive-but-narrow claim:
  - strongest evidence in LLaMA SM
  - additional support in LLaMA IC
  - limited extension to MW or cross-domain only where directly confirmed

### Case B: likely case

Condition:
- `Exp1` confirmed
- `Exp2a` confirmed or borderline
- `Exp2b` and `Exp3` remain weak

Action:
- Write the causal section around:
  - direction-specific evidence in LLaMA SM
  - strong within-domain causal control in LLaMA IC
  - MW and cross-domain as exploratory / partial / non-universal

### Case C: pessimistic case

Condition:
- only `Exp1` survives clearly

Action:
- Keep the mechanistic claim restricted to the single best-supported configuration.
- Treat all other steering evidence as suggestive, not confirmatory.

### NeurIPS-oriented interpretation

The practical NeurIPS question is not whether every sub-result is strong. It is whether the paper can present:

- one clearly supported mechanistic anchor case
- one or more supportive but honestly delimited extensions
- a reproducible and internally consistent analysis story

Under this standard, the default priority after V14 is:

1. fix claim scope in the paper
2. check reproducibility risks
3. run at most one focused follow-up only if it materially upgrades a target claim

## 4. V15 follow-up experiments

Principle:
- preserve prompt and game logic from V12/V14
- vary only the minimum necessary experimental factor
- choose the follow-up based on the paper claim you want to keep, not based on symmetry for its own sake

### V15-A: MW layer re-selection

Goal:
- test whether MW weakness is a layer-choice problem rather than absence of a causal signal

Design:
- model: LLaMA
- task: MW
- reuse identical V12/V14 prompt/game logic
- compare a small candidate set of layers already implicated elsewhere:
  - `L16`, `L22`, `L25`, `L30`
- keep:
  - same alpha grid
  - same BK-direction construction method
  - same random-control protocol

Success criterion:
- at least one layer shows:
  - stronger monotonicity than current V14 MW
  - BK direction outranking random controls

### V15-B: focused cross-domain rerun on a single pair

Goal:
- avoid spreading compute over three weakly supported transfer pairs

Design:
- choose one pair only after V14 ends:
  - priority order:
    1. `MW -> SM`
    2. `IC -> SM`
    3. `MW -> IC`
- increase `n_games`
- increase random controls modestly
- keep the exact same target-task prompt/game loop

Success criterion:
- cross-domain BK direction beats its random-control distribution

### V15-C: multi-layer direction for transfer only

Goal:
- test whether transfer is weak because the signal is distributed across depth

Design:
- do not change prompts, parser, or task logic
- only change the steering vector from single-layer to a small multi-layer stack for the transfer setting
- candidate:
  - `L22 + L25 + L30`

Use only if:
- within-domain remains strong but cross-domain remains weak

### V15-D: reproducibility replication for the strongest conditions

Goal:
- make the final paper less dependent on one random-control draw

Design:
- rerun only:
  - `Exp1`-style LLaMA SM
  - `Exp2a`-style LLaMA IC
- replace `hash(exp_name)` seeding with fixed deterministic seeds

Success criterion:
- same verdict under exact rerun

## 4.1 Follow-up selection method

Use the following order:

1. If the paper is already defensible with `Exp1 + Exp2a` and conservative wording:
   - do not launch a broad new suite
   - update the paper first

2. If one additional claim is worth trying to rescue:
   - choose exactly one of:
     - `V15-A` to rescue MW as a within-domain causal extension
     - `V15-B` to rescue one cross-domain causal pair
     - `V15-D` to strengthen reproducibility of the strongest cases

3. Only use `V15-C` if a prior result suggests signal distribution across depth rather than absence of signal.

4. Do not prioritize repeating Gemma SM/IC as the first follow-up.
   - They were already weak in V12 because of ceiling/floor behavior.
   - Their main value is boundary-condition interpretation, not main-text rescue.

## 5. Recommended immediate next step

Do not stop the current V14 jobs.

After V14 completes:

1. finalize verdicts for `Exp1`, `Exp2a`, `Exp2b`, `Exp3`, `Exp4`
2. if `Exp1` and `Exp2a` are both strong:
   - update the paper first
   - then run only one focused V15 experiment for MW or cross-domain
3. if `Exp2a` weakens materially by random-control completion:
   - prioritize reproducibility and layer-sensitivity checks before paper updates

## 5.1 Idle check guidance during long runs

Because the active V14 jobs share one GPU and Gemma can offload partially to CPU, apparent slowness does not automatically mean the run is stuck. Treat the run as \textbf{active, not idle} when both conditions hold:

- at least one log file receives a new progress line within the last 1--3 hours
- GPU utilization remains persistently high

Treat the run as \textbf{suspiciously idle} only when:

- no log file advances for several hours, and
- GPU utilization drops near zero, or
- the worker processes disappear without producing final JSON files

## 6. Implementation rules for any V15 code

- Reuse `run_v12_all_steering.py` for:
  - prompt construction
  - parsing
  - `play_game`
  - `run_condition`
  - `compute_bk_direction`
- Avoid introducing a new gameplay implementation.
- Use fixed explicit seeds, not Python `hash(...)`.
- Write intermediate checkpoints after:
  - BK-direction sweep
  - each random control
- Keep output JSON schema compatible with V14 report tooling where possible.
