# W0–W2: Section-Interleaved Causal Verification — Design Spec v2

**Date**: 2026-06-10 (supersedes the §4.5-subsection placement of the 06-10 E1–E3 spec;
all E1–E3 protocol elements remain in force)
**Approved**: user, 2026-06-10 — "절별 삽입" formation; IC promoted to W1; no full-game;
ultracode implementation; parallel node submission with HF sync/resume.
**North star**: does the CORRELATION the paper reads (§4.1 indicators, §4.2 shared axis,
§4.3 condition modulation) connect to CAUSATION on the same axes?

## 0. Validity rulebook (applies to every arm; pre-registered)

- **R1 two-stage**: discovery n=50 → confirmatory n=200 (held-out states offset ≥300,
  third seed set, axes re-estimated excluding eval games) before any body-text claim.
- **R2** one pre-named primary endpoint per sub-experiment; all else exploratory.
- **R3** equivalence only via TOST (margin = 25% of the −G→+G gap) or gap-recovery % + CI.
- **R4** default test = game-cluster permutation (cluster = catalog game id).
- **R5** trials enter behavioural metrics only if parse_ok; arm invalid if parse rate <0.9.
- **R6** steering specificity vs same-norm random directions; LC additionally vs WIN branch.
- **R7** anchor gate: a sub-experiment is interpretable only if natural −G/+G anchors
  separate on its primary endpoint.

## 1. Data provenance (verified 2026-06-10)

- SM catalog `behavioral/slot_machine/gemma_v4_role/final_gemma_20260227_002507.json` —
  the single file the paper's §4 extraction used (3200 games / 21421 rounds / 87
  bankruptcies match extraction_summary.json).
- `phase_a_hidden_states.npz` (SM + IC, gemma) — same extraction run as Table 1 inputs.
- Prompt builder: frozen M3″ copy, verified **100/100 char-identical** with the
  extraction's `reconstruct_sm_prompt` on the state pool.
- IC arms use `behavioral/investment_choice/v2_role_gemma/*.json` (the §4 IC corpus —
  NOT the §3 Fig-4 four-variant dirs) to match phase_a IC.
- Never used: `legacy/v17*`, 2025-12 `.pt` checkpoints, `rq2_audit_consistent_layer.json`.

## 2. W0 — axes + retro statistics (CPU)

New module `multilayer_causal/src/axes.py` builds, for ALL 42 layers
(direction (42,3584) fp32 + scales (42,) = 0.03·median natural norm, same npz format
as directions.npz):

| asset | construction (on phase_a rows, frozen SMAdapter order) |
|---|---|
| `assets/directions_iba_v2.npz` | −G variable rounds, EXCLUDING games of pool[0:300]; bet_ratio top vs bottom tercile mean diff |
| `assets/directions_ilc.npz` | rounds whose displayed previous round is a LOSS; escalated (r_t > r_{t-1}) vs de-escalated mean diff (same exclusion) |
| `assets/directions_iec.npz` | r≥0.5 vs r<0.5, contrast computed WITHIN each prompt_condition then averaged (condition-matched; full 21421 pool, same exclusion) |
| `assets/directions_readout.npz` | per-layer Ridge(α=100, standardized) weights, target=min(r,1) on −G variable bet rounds (same exclusion) — "the axis a linear reader uses" |
| `assets/directions_ic.npz` | IC phase_a: risky choice (3,4) vs safe (1,2) mean diff per layer |

Report cos-similarity matrix among all axes (incl. existing G-axis) at L18–23.

New module `multilayer_causal/src/retro_stats.py` (reads HF checkpoints + the
deterministic pool): game-cluster permutation re-tests of all headline comparisons;
TOST gap-recovery for e1c_win18; parse-excluded sensitivity for E2 arms;
I_LC^dec (= max(0,(r_t−r_disp)/r_disp) on displayed-loss states) for every arm.
Output `out/retro_stats.json` + printed markdown.

## 3. W1 — discovery arms (GPU, n=50, state_offset 0, seeds 42+997i)

Existing e1 anchors are reused for single-decision arms (same states/seeds).

**§4.1 bundle — "the read axis as a handle"**
- `w1ro_{m20,a00,p20,p40}`: steer `directions_readout` at L18–23, α∈{−2,0,+2,+4}.
  Primary: Spearman ρ(α, I_BA paper-def), cluster permutation.
- Probe arms (`probe: true`, 3 generations/trial — decision, LOSS branch, WIN branch):
  `w1lc_anchor_minus, w1lc_anchor_plus, w1lc_patch` (patch L18–23),
  `w1lc_iba_p40`, `w1lc_ilc_p20, w1lc_ilc_p40`, `w1lc_rnd` (random dir, α=+4).
  Per-trial: r_t; LOSS branch balance′=bal−bet, WIN branch balance′=bal+2·bet, history
  appended in build_prompt format; I_LC = max(0,(r^L−r_t)/r_t); specificity = paired
  (r^L−r_t)/r_t − (r^W−r_t)/r_t. All-in first bets → LC excluded, logged as probe-bankruptcy.
  Primary: I_LC-axis specificity diff vs anchor_minus (one-sided). Gate R7 on anchors.
- Selectivity: `w1s_{ilc,iec}_{p20,p40}` single-decision steering → fills the 3-axis ×
  3-indicator matrix (LC column from probe). Decision rule: diagonal ≥2× off-diagonal
  (vs control distribution) ⇒ independent handles; else common risk axis.

**§4.2 bundle — "does the shared axis transfer causally" (task: ic)**
- `w1ic_anchor`, `w1ic_smiba_{p20,p40}` (SM I_BA-axis on IC), `w1ic_icax_p40`
  (IC-native axis, positive control), `w1ic_rnd_{s0,s1}`.
  Primary: risky-choice rate (choices 3–4 fraction; final after schema check in W0).
  Interpretation matrix pre-registered: SM✓&IC✓ shared causal axis / SM✗&IC✓
  task-specific write axes / both✗ IC insensitive to single-decision axis writes.

**§4.3 bundle — localization finish**
- Bridge: `w1b_patch5` (patch layers {8,12,22,25,30} — non-contiguous, new
  `layers_list` field), `w1b_steer5` (iba_v2 +4 at those 5).
- Edge: width-2 windows `w1e_{1617,1819,2021,2223,2425}` + `w1e_1821, w1e_2023` (patch).
  Primary: +G gap-recovery % with cluster-bootstrap CI.

Total ≈ 30 arms (probe arms cost ×3 generations). One amlt job, 4×H100, run_arms.sh
sharding, HF checkpoint sync/resume identical to E1 (paths `checkpoints/{phase}/`).

## 4. W2 — confirmatory (n=200, offset ≥300, seed set 3, re-estimated axes)

Candidates fixed after W1: strongest of {iba_v2 +4, readout +4} (primary I_BA);
I_LC-axis probe arm (primary specificity diff); SM→IC transfer if positive
(primary risky-choice rate); random@operating-dose ×10. Promotion rule: confirmatory
p<0.05 + outside random 95th pct + parse ≥0.9.

## 5. Paper integration (no §4.5; in-place inserts)

- §4 intro: one shared causal-protocol paragraph.
- §4.1 close: readout-axis result + indicator-axis control (+ selectivity verdict).
- §4.2 close: rank-r failure (E2) + cross-task causal transfer verdict.
- §4.3 close: E1/E1c localization (already confirmed) + bridge/edge ("placement, not count").
- One combined causal Table 4 (body) + full tables in Appendix F.4 as M3‴.
- Replace "correlational only" statements in abstract, §4.4, F.4 intro, limitations,
  checklist (5 sites; both EN builds, then KO mirrors).

## 6. Implementation notes (ultracode)

Disjoint-file parallel build: (A) axes.py+tests, (B) retro_stats.py+tests,
(C) ic.py (pool loader on stored full_prompt + FROZEN copy of the repo's IC parser with
parity test) + tests, (D) runner probe mode + `layers_list`/`task` dispatch + arms.yaml
W1 entries + registry tests + amlt w1 template. Then adversarial verify (full pytest,
git-status isolation check, karpathy review). No agent commits; main session commits.
