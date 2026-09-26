# §4 Unified-Axis Causal Program — Design Spec

Date: 2026-07-05
Status: DESIGN (awaiting user review)
Paper: "Can LLMs Develop Gambling Addiction?" (NeurIPS 2026, frozen submission)
Track: `multilayer_causal/` (separate pre-registered causal study)

---

## 1. Thesis

One sentence runs through all of §4:

> **Decodability is broad and shared; causal control is narrow (L16–21), task-specific, and read from a different object than it is written to.**

Each §4 subsection becomes a **(correlational read, causal write)** pair, and — per the
consistency requirement — **all three causal tests steer the SAME object: the §4.1 indicator
readout, made steerable.**

| Subsection | Read (exists, correlational) | Write (new, causal) |
|---|---|---|
| §4.1 indicator readout | indicators decodable from L22 SAE features (Table 1) | steering the indicator direction at L16–21 causally moves that indicator (sufficiency + specificity + dose + sign-reversal) |
| §4.2 cross-task sharing | geometry shares weakly (LOTO AUC>chance); readout rules don't transfer (sparse R²<0) (Table 2) | shared subspace does NOT causally control; control lives in the task-specific residual (shared/residual decomposition) |
| §4.3 condition modulation | +G/+M sharpen readout legibility (Table 3) | +G/+M modulate causal **writability** (dose-response slope), not just legibility |

Relationship to the paper's own null: **Appendix M3** already steered the §4.1 Ridge readout
direction (projected through the SAE decoder) at **L22, single-layer** and got a **null** result.
Our sole methodological delta is the **writable window (L16–21, multi-layer, cumulative)** that
Stage A localized. This is the cleanest possible contrast: same construction, one changed knob,
null → write.

---

## 2. The unified object

For each task `t ∈ {sm, ic, mw}` and indicator `k ∈ {I_BA, I_LC, I_EC}`:

```
d_{t,k}  =  SAE_decoder_L  @  ridge_coef_{t,k}          # residual-stream direction (3584-d)
```

- `ridge_coef_{t,k}` = §4.1 top-200 SAE-feature Ridge readout weights for indicator k on task t
  (deconfounded against balance+round within-fold, exactly as §4.1 / M3).
- Projected through the Gemma-Scope SAE decoder (`google/gemma-scope-9b-pt-res`, cached locally)
  to a steerable hidden-space direction — the SAME construction M3 used.
- Replicated / refit across the write window **L16–21**; oriented indicator-increasing;
  norm-matched via per-layer target-task Δ-norm (reuse `paper_axes.scales_from_phase_a`, same as
  `xtask_axes.py`).
- Cached as `assets/indicator/d_{t}_{k}.npz` with provenance string (mirror `bk_single` schema).

**Why this object and not our existing axes:** our Stage B/C/D axes were `bk_single`
(μ_stop−μ_bankrupt, §4.2 endpoint geometry) and `iba_v2` (behavioral top/bottom split —
`n_top/n_bot/lo_cut/hi_cut`). Neither is the §4.1 SAE readout. To honor "use the §4.1 features
consistently," the causal thread is rebuilt on `d_{t,k}`. Existing BK-axis results survive as
**§4.2 robustness evidence** (BK is the paper's own §4.2 sharing object).

---

## 3. P0 — gate pilot (LOCAL, no node) — IMMEDIATE STEP

**Risk being gated:** M3 got null steering exactly this object at L22. If `d_{sm,BA}` at L16–21
also does not write, the entire unified-consistency program yields negatives.

**Pilot:** build `d_{sm,BA}` per §2; steer at L16–21 on SM at α ∈ {−4, 0, +4}; measure I_BA
(bet ratio). n≈50 (pilot only).

**Gate criteria:**
- PASS: |Δ I_BA(+4) − Δ I_BA(−4)| clearly exceeds the random-null band, monotone in sign
  → proceed to full 3 waves.
- FAIL (null): the readout direction is not a write direction even at L16–21. This is itself a
  finding ("write ≠ read direction"). Fall back to behavioral-split axes for the causal thread and
  **explicitly caveat** that the causal object differs from the §4.1 readout. Re-open design.

Run location: local GPU if available, else one small fresh amlt job (SM-only, no catalog).

---

## 4. Full waves (each = one fresh amlt job, only if P0 PASSES)

Recipe = proven Stage A–D path: `sing/msrresrchbasicvc`, `80G4-H100`, `render.sh` →
`amlt run *.rendered.yaml`. Isolation via `MLC_ARMS_YAML/MLC_HF_BASE/MLC_OUT`. n≥100,
multi-seed nulls (≥3 directions) to fix audit M3. Fix `manip_proj` key path (audit m1:
read `vector_log.proj`).

### Wave E4.1 — §4.1 within-task sufficiency + specificity (SM)
- Steer `d_{sm,k}` for k∈{BA,LC,EC} at L16–21, α ∈ {−4,−2,−1,0,+1,+2,+4} (dose + sign reversal).
- Measure ALL THREE behavioral indicators per arm → **3×3 indicator-specificity matrix** + per-cell
  dose-response.
- Claims tested: (a) sufficiency — steering k moves k; (b) specificity — does one axis drive all
  three (the iba_v2 "one axis" claim, now on the SAE-derived object); (c) it is a genuine lever —
  monotone dose + sign reversal.
- Contribution vs M3: null@L22 → write@L16–21.

### Wave E4.2 — §4.2 cross-task transfer + shared/residual decomposition
- 3×3 transfer on `d_{t,k}` (source task t → apply task t'); plus steer the **LOTO rank-1/2 shared
  subspace** into each task; plus the **task-specific residual** (`d` minus its projection onto the
  shared subspace).
- Robustness: re-run the BK-axis transfer (Stage B/C object) alongside.
- Claim tested: shared component is decode-only (no causal control); residual carries control.
  Prediction from SVD (shared PC1: ic+0.91/mw+0.76/sm−0.47) and Stage B/C: transfer fails/reverses.
- **OPEN DECISION (resolved at wave time):** IC(0.13 floor)/MW(0.78 ceiling) have no behavioral
  headroom. Either (i) accept SM as the only fair target and frame IC/MW as structurally
  constrained, or (ii) re-baseline IC/MW prompts to mid-range (~0.4–0.5) for a fair causal test.
  Decide before submitting E4.2.

### Wave E4.3 — §4.3 condition modulation of writability (SM × condition)
- Run the E4.1 sufficiency intervention under prompt conditions ±G, ±M.
- Measure whether the causal write effect size (dose-response slope) depends on condition.
- Claim tested: autonomy conditions change controllability, not just legibility.

### Cross-cutting
- LLaMA replication: **limited** — LlamaScope covers L25–31 only, so the L16–21 window cannot be
  reproduced on LLaMA. Report as an honest coverage limitation; do NOT claim LLaMA replicates the
  window.

---

## 5. Node / submission plan

- **No dedicated node.** `metacognition-math/NODE_POLICY.md`: all current H200×4 holders on
  `msrresrchbasicvc` are project-owned (metacognition / behavior-uncertainty / boltzmann /
  softprompt-GRPO); gambling is not an owner; policy forbids co-opting without an explicit user
  policy revision.
- **Route = fresh scheduled jobs** to `sing/msrresrchbasicvc 80G4-H100` (Stage A–D recipe,
  target confirmed reachable, H100 available). Each wave = one `amlt run`. Downside: Standard-tier
  preemption/pause (mitigate with watch scripts + HF resume).
- If the user wants a preemption-proof dedicated holder for the gambling paper, that is a separate
  policy-revision request (out of scope here).

---

## 6. Paper integration (rebuttal now; camera-ready fuller)

Paper is frozen (2026-06-03 canonical). Integration is **rebuttal response + appendix expansion**,
not body restructuring, until camera-ready.

- **Appendix M3/M4:** "L22 single-layer null resolved — the writable locus is a distributed
  L16–21 band; same SAE-decoder-projected readout writes there." State protocol delta vs M3.
- **Figure (spine overlay, gemma+llama):** read broad / write local — L16–21 recovery peak vs
  spread-out read. (asset exists: `results/spine/`)
- **Figure (Stage D locality, null-subtracted, "peak not saturate"):** single-layer profile +
  cumulative width curve + derail. (asset: `results/xtaskd/`, regenerate per audit M6)
- **Figure (§4.1 specificity matrix + dose/sign):** from E4.1.
- **Figure (§4.2 shared vs residual):** parse/coherence + SM-target behavior bars (NOT the
  broken transfer-fraction matrix — audit F1). from E4.2.
- **§4 summary:** 2–3 sentences threading read/write dissociation across 4.1–4.3.

---

## 7. Audit fixes folded in (from 2026-07-05 code review)

- F1: drop the transfer-fraction matrix for ic/mw columns (near-zero diagonal normalizer);
  report SM column + coherence matrix instead.
- M1/M2: narrate "self-axis stops" and "misalignment collapses" as SM-specific / dose≥3, not universal.
- M3: single random null → **multi-seed (≥3) nulls**; n=50 → **n≥100**.
- M6: Stage D "saturates" → "peaks at L16–21, over-widening derails"; **null-subtract** recovery.
- m1: fix `manip_proj` to read `vector_log.proj` (manipulation check currently nan everywhere).

---

## 8. Success criteria

- **P0:** clear sign-dependent write of `d_{sm,BA}` at L16–21 above null band. (GO/NO-GO)
- **E4.1:** ≥1 indicator shows monotone dose + sign reversal; specificity structure characterized.
- **E4.2:** shared-subspace steer fails to control while residual controls (on the fair target) —
  or, if it controls, the read/write dissociation thesis is falsified (reportable either way).
- **E4.3:** condition×dose interaction estimated with CI (null is an acceptable, reportable result).

---

## 9. Order of operations

1. P0 gate pilot (local) — **now**.
2. If PASS: E4.1 (SM) → E4.2 (cross-task, after headroom decision) → E4.3 (condition).
   E4.1 and E4.3 are SM-only and could share a wave.
3. Regenerate audit-fixed figures; draft rebuttal appendix text.
