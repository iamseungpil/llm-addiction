# SEC4 Autoresearch Goals v2 — Adjudication Structure + Aggressive Search

Approved 2026-07-06 (user: 수정안 + 공격적 탐색 절충). Supersedes the goal
framing of the Phase-1 plan; the Phase-1 experimental machinery is unchanged.

## Goal (redefined)

NOT "search until a three-way common axis is found." INSTEAD: **adjudicate each
of three independent, falsifiable questions — positive OR characterized
negative — while searching aggressively for commonality within each.**

- **Q1 (indicators, §4.1):** does ONE axis causally drive I_BA *and* I_LC
  (post-loss escalation, the non-tautological third indicator; I_EC is a
  threshold of I_BA and counts only as weak corroboration)?
- **Q2 (tasks, §4.2):** does the SVD shared component of the task-own
  behavioural axes causally control each task, measured **expression-matched**
  (each task's own binding indicator: SM/IC -> I_BA, MW -> I_LC), after IC/MW
  mid-range re-baselining for headroom?
- **Q3 (prompts, §4.3):** does +/-G, +/-M modulate the dose-response slope of
  the (shared or task) causal axis?

Partial commonality is a legitimate terminal result (e.g. "indicator-common
within task, control task-specific" causally confirms §4.2's published claim).

## Stopping rule (aggressive variant, per approval)

Per question: up to **5+ pre-registered refinements** before accepting a
characterized negative (vs 3 in the conservative draft). Refinements must be
logged in INDEX.md BEFORE launching (no post-hoc rung invention). Every rung —
positive or negative — is recorded with config, numbers, and verdict.
Discovery/held-out game split enforced by the replay guard as in Wave-1/2.

## Pre-registered refinement ladders

- Q1: (1) shared axis on post-loss states; (2) I_BA axis on post-loss states;
  (3) axis built FROM post-loss contrast states; (4) alpha re-scale;
  (5) window variants (16-22, 17-21).
- Q2: (1) expression-matched shared-component steer; (2) IC/MW re-baseline to
  mid-range then repeat; (3) shared component recomputed inside the write
  window per task; (4) residual (task-specific) steer as positive control;
  (5) pairwise (ic<->mw first, the aligned pair) before 3-way.
- Q3: (1) base vs +G slope; (2) full 2x2 G/M; (3) condition-matched baselines;
  (4) per-condition axis re-fit; (5) interaction with n boost.

## Status at approval

- Wave-1 (sec4_p0): DONE — behavioural axis WRITES (rho .96, z 11.4), confound
  inert, readout weak; read!=write unresolved pending thick null. Archived
  (INDEX rung, git 28605c2, HF analysis/).
- Wave-2 (mlc-sec4-w2-0706): RUNNING — thick 20x2-dose null + shared-axis
  reconnaissance. Valid unchanged under this framing.
- Wave-3 next: W3a = Q1 rung 1-2 (post-loss conditional steering);
  W3b = Q2 rung 1 (expression-matched); W3c = Q3 rung 1.
