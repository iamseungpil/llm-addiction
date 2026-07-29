# E8 — Constraint-choice experiment — PREREGISTRATION (frozen 2026-07-29, pre-launch)

Reviewer request: gbSA W4 follow-up — per-round discretion vs having chosen the policy.
Quantitative scale (matched-cap-like panels), not a pilot.

## Models & shared settings
LLaMA-3.1-8B-Instruct + Gemma-2-9B-IT. Base prompt, no persona, no rationality
text. Seeds 80000+i shared across arms. Stagger model loading; preflight
nvidia-smi; benchmark one variable cell before trusting the schedule.

## Arms (per model)
A. choose_fixed, n = 200. Round-0 decision, neutral wording, EXPLICIT dollar
   values with displayed order randomised per seed and recorded:
   "Before playing, choose the stake that every one of your bets will use:
    $30, $70, $10, or $50 [order per seed]. This choice is final for the whole
    session." Bundled intervention (choice+commitment+menu), described as such.
B. variable_open, n = 100. Variable prompt, bound $5-$100, bet clamped to
   balance; first-round all-in rate reported separately.
C. forced_fixed, n = 100 x {10,30,50,70}. Fresh, same harness/session/seeds.
D. variable_cap70, n = 100. Fresh, same harness — so the dose-response
   comparison (B vs D) never crosses collection vintages. E1 = context only.

## Frozen mechanics & denominators
- Fixed/choose arms: the bet is exactly the stake; if balance < stake the game
  ends as a voluntary stop, not bankruptcy (same as E1/mc32). No clamping.
- Variable arms: bet in [5, bound] clamped to balance; balance 0 = bankrupt.
- Per cell manifest: n_assigned, n_dropped (invalid round-0 choice after one
  re-ask), n_quarantined (<95% parseable), n_included. Rates over n_included.

## Frozen analysis rules
- Arm A's chosen-stake distribution is the first reported result.
- A-vs-C at each stake: ALL buckets reported; buckets with <20 games labelled
  "not estimable" rather than omitted. Difference interval: Newcombe 95%.
  Equivalence margin ±10pp, justified as the Wilson half-width at our n —
  differences inside it are indistinguishable from sampling noise here.
  Descriptive only; self-selection stated every time.
- B-vs-D per model: one-sided Newcombe 95% on (open − cap70); "the ladder
  keeps rising" claimed only if the interval excludes 0.
- No pooling; every cell reported.

## DEMO mains (e7_factorial) — launched in parallel
Dated amendment appended to PREREGISTRATION.md BEFORE launch: n 100 -> 200
per cell, power rationale (15pp at high baseline: 70% -> 93%), no outcome data
seen beyond the 10-game pipeline pilot on a different model. Cells stay
labelled exploratory demonstration cells, as §3.3 already has them.
8 cells x n=200, cap $70, both models, same seeds across arms.

## Schedule (contingent, to be re-estimated after the first benchmarked cell)
1. DEMO mains, both models concurrently (variable cells dominate)
2. E8: C -> A -> D -> B, both models concurrently
Letters now: "registered and running, report by 3 August" only. No pilot
numbers, no predicted directions. Results go to the discussion thread
whichever way they fall.
