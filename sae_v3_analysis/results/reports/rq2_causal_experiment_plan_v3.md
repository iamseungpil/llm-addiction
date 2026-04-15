# RQ2 Causal Experiment Plan v3.1 (Codex-Approved)

**Date**: 2026-04-13
**Status**: Plan approved after 3 Codex critique iterations. Ready for implementation.
**Node**: E8 (4×A100 80GB)

---

## Execution Order (Sequential Gating)

1. **Phase 0** (all GPUs, ~10 min): Compute shared axes + residual stream norms. Sanity check 10 games at α=±max.
2. **Phase 1** (GPU 3, ~3h): Experiment C — positive control at L25. GATE: if C fails (perm_p > 0.05) → STOP.
3. **Phase 2** (GPU 1 + GPU 2, ~12-18h parallel): Experiments A + B. Only after C passes.

GPU 0 continues V14 L22 replication independently.

---

## Common Protocol

### Split-Sample
- Discovery: games with even game_id (50%)
- Evaluation: games with odd game_id (50%)
- Axis computed on discovery only, frozen before any intervention

### Shared Axis Computation
- Per-task centroid difference: `d_t = centroid(BK_states) - centroid(VS_states)` on discovery split
- Precision weight: `w_t = n_BK·n_VS / (n_BK + n_VS)`
- Between-class scatter: `S_B = Σ_t w_t · d_t · d_tᵀ`
- Shared axis = top eigenvector of S_B
- Sign: freeze so `u · (Σ_t w_t d_t) > 0`

### Seeding (FIXED — α-independent)
```python
seed = g + TASK_SEED_OFFSET[task]
# TASK_SEED_OFFSET = {"sm": 0, "ic": 100000, "mw": 200000}
# α does NOT enter the seed → valid paired design
```

### Primary Estimand
- Binary (BK): Cochran-Armitage trend test across 7 ordered α levels
- Continuous (terminal wealth, I_BA): OLS β₁

### Multiple Comparisons
- Each experiment pre-specifies ONE confirmatory task (α=0.05, no correction)
- Remaining tasks are exploratory (Holm correction across 2 tasks)

### Phase 0 Sanity Check
- Report residual stream L2 norm per layer
- Run 10 games at α=+max, α=-max per model×layer
- If >50% parsing failures → halve α range
- Log sanity check results in structured JSON

---

## Experiment A: Shared-Axis Steering — LLaMA L25 (GPU 1, Phase 2)

**Intent**: Test if centroid_pca rank-1 shared axis is causally intervention-relevant.
**Hypothesis**: Positive-direction steering increases BK rate and I_BA across ≥2/3 tasks.
**Layer**: L25 (strongest shared axis, +0.092 mean AUC observational).
**α range**: absolute base=1.0 → [-2, -1, -0.5, 0, +0.5, +1, +2].
**Confirmatory task**: SM (BK 36.4%, well-powered).
**Exploratory tasks**: IC (BK 8.9%, continuous primary), MW (BK 75.8%, binary OK).

**Game counts**:
- Main direction: 200 games × 7 α × 3 tasks = 4,200
- Null (100 dirs): 50 games × 7 α × 3 tasks × 100 = 105,000
- Total: 109,200 games
- Estimated: ~12-14h at ~0.5s/game (LLaMA 8B on A100)

**Kill criterion**: SM Cochran-Armitage non-significant AND all 3 continuous endpoints non-significant → "shared axis at LLaMA L25 does not produce detectable change at α∈[-2,+2]. Consistent with: (a) not causally relevant at this layer, (b) α mismatch, or (c) non-manipulable feature."

---

## Experiment B: Shared-Axis Steering — Gemma L12 (GPU 2, Phase 2)

**Intent**: Same as A for Gemma at L12 (6/6 observational transfer).
**Confirmatory task**: IC (BK 10.8%, highest Gemma rate).
**Exploratory tasks**: SM (2.7%), MW (1.7%) — **BK dropped as endpoint for SM and MW** (too few events). Continuous only.

**Game counts (budget-constrained)**:
- Main direction: SM 500×7 + IC 200×7 + MW 500×7 = 8,400
- Null IC (confirmatory, 50 dirs): 50 games × 7 α × 50 = 17,500
- Null SM/MW (exploratory, descriptive only): 0 (main direction descriptive stats only)
- Total: 25,900 games
- Estimated: ~18h at ~2.5s/game (Gemma 9B on A100)

**Note**: Minimum permutation p for IC = 1/51 ≈ 0.020.

---

## Experiment C: Per-Task BK Positive Control — LLaMA L25 SM (GPU 3, Phase 1)

**Intent**: Confirm L25 is intervention-sensitive before investing in shared-axis experiments.
**Direction**: SM-specific `centroid(BK) - centroid(VS)` from discovery split.
**Confirmatory task**: SM only.

**Game counts**:
- Main: 200 × 7 = 1,400
- Null (100 dirs): 50 × 7 × 100 = 35,000
- Total: 36,400 games
- Estimated: ~3h

**Gate**: Cochran-Armitage p<0.05 → proceed to Phase 2. Else → STOP all.

---

## What's Not Included

- Cross-model axis transfer (needs alignment map, deferred)
- Off-target layer control (follow-up if A/B succeed)
- Matched interchange (V23 Plan E5, future work)

---

## Codex Critique History

- **v1**: 5 issues flagged. Sign convention, Gemma SM power, alpha justification, C redundancy, cross-model infeasible.
- **v2**: 8 issues flagged. BLOCKER: α-dependent seeding. HIGH: 20 random dirs insufficient, alpha range unjustified, C should gate A/B. MEDIUM: multiple comparisons, MW BK endpoint.
- **v3**: Approved. One fix: Experiment B null budget infeasible (385k games). Resolved by reducing to IC-only null (50 dirs) + main-direction-only for SM/MW.

---

# v4 Addendum (2026-04-13): Prompt-Distribution Fix + Paper Update Plan

**Status**: v3 plan remains valid in structure. v4 adds a critical fix discovered during execution — the steering prompts did NOT match the original behavioral experiments — and a corresponding paper update plan. This section is additive; all v3 decisions are preserved unless explicitly overridden.

## v4.1 What Went Wrong (Discovery)

During the first v3 run, Experiment B Gemma IC produced BK=0/200 at all α levels (expected ~10.8% at baseline). Investigation revealed that `run_v12_all_steering.build_{sm,ic,mw}_prompt` used simplified prompts that diverged from the original behavioral experiments in five respects:

1. **Missing ROLE_INSTRUCTION header** — the behavioral-economics framing ("You are a participant in a behavioral economics simulation...") was absent.
2. **Missing G/M prompt components** — the addiction-inducing "Set a target amount" and "Maximize the reward" instructions were absent.
3. **Wrong round limit** — SM used 20 rounds instead of the original 100.
4. **Inverted framing in IC** — the steering IC prompt explicitly stated "All risky options have negative expected value", which nudges rational cash-out behavior (the opposite of addiction).
5. **Simplified response format** — "Final Decision: Option" instead of the original "Explain your reasoning... Final Decision: <Bet $X or Stop>".

**Consequence**: The hidden-state shared axis was learned from decisions the model made under the addiction-inducing prompt distribution (original), but intervention was tested on a completely different prompt distribution (bare/rational). This is a train-test mismatch that invalidates the causal claim, not a scientific null result.

**Scope of impact**: Steering experiments only (Stage 4 in the audit). The readout/correlation pipeline (Table 1–3, Figure cross-transfer, selectivity controls) is unaffected because it uses pre-extracted hidden states from games played with the correct original prompts. Full pipeline audit confirms IC/MW SAE feature extraction uses stored `full_prompt` fields directly, and SM extraction reconstructs G/M/H/W/P correctly (only ROLE_INSTRUCTION is omitted from the reconstruction, which affects a ~100-token prefix but preserves the addiction-inducing conditions).

## v4.2 Code Fix

The following changes were applied to `sae_v3_analysis/src/run_v12_all_steering.py` and propagate automatically to `run_aligned_factor_steering.py` (which imports the prompt builders) and to any future steering script that reuses these functions.

| Item | Before | After |
|---|---|---|
| ROLE_INSTRUCTION | absent | `"You are a participant in a behavioral economics simulation..."` (exact original) |
| Prompt components | absent | `DEFAULT_PROMPT_COMBO = "GM"` (G + M, the two strongest addiction-inducing conditions) |
| SM MAX_ROUNDS | 20 | 100 (matches original) |
| IC max_rounds | 20 | 100 (matches original) |
| IC bet range | fixed $10 | variable $1–$10 (matches original) |
| IC framing | "negative expected value" | original "10% chance of 9.0x / 25% of 3.6x / 50% of 1.8x / 100% returned" |
| MW mechanics | 30% win × 3x flat | 4-zone wheel (Red 25%/2x + Blue 8%/3x + Gold 2%/8x + Black 65%/0x) |
| MW bet range | $5–$100 | $5–$balance |
| SM/MW win tracking bug | always `True` (history showed all WINs) | proper `won` flag per-round |
| Parser precedence bug | `('bet', nested_tuple)` crash on non-parseable | flat tuple, parse fallback logged |
| max_new_tokens | 150 | 512 (accommodates reasoning prose) |
| Game timeout | 120 s | 600 s (prose responses are slower) |

Codex reviewed v4 changes in two critique rounds. First round identified 4 BUG (MAX_ROUNDS, parser precedence, always-true win tracking, MW zone mechanics) and 6 WARN items. All BUGs were fixed. Two WARN items (LLaMA ROLE_INSTRUCTION asymmetry, IC G-wording micro-difference) were accepted as known deviations to maintain steering consistency across models.

## v4.2b Exact-Replay Update (2026-04-14)

The steering runtime was further tightened after a second audit. The main
change is architectural: claim-bearing steering no longer relies on a
hand-written prompt/game sandbox. Instead it now routes through
`src/exact_behavioral_replay.py`, which:

1. loads empirical condition catalogs from the raw behavioral JSONs,
2. replays the original `prompt_condition × bet_type × bet_constraint` mixtures,
   using a deterministic shuffled catalog so partial runs are not front-loaded
   by the original file ordering,
3. reuses the original behavioral runners' prompt builders, parsers, retry
   hints, skip handling, and game mechanics, and
4. preserves model-specific prompt asymmetries present in the original runs.

Concretely:

- `run_aligned_factor_steering.py` now replays exact behavioral profiles for
  SM / IC / MW rather than fixed `GM` prompts.
- `run_v16_multilayer_steering.py` now also uses exact SM behavioral replay and
  keeps alpha-independent seeds.

Interpretation rule:

- Results from this replay path are the canonical post-audit steering outputs.
- Older sandbox-only steering outputs remain archived side evidence.

## v4.3 Re-Run Protocol (E8 4×A100 Parallel)

Launched 2026-04-13 21:30 UTC with the fixed code. All four experiments run in parallel on the E8 node.

| GPU | Script | Experiment | Expected wall time |
|:---:|---|---|---|
| 0 | `run_aligned_factor_steering.py --experiment c --model llama` | Exp C: Per-task BK direction on LLaMA L25 SM (positive control / gate) | ~6 h |
| 1 | `run_v16_multilayer_steering.py --model llama --layers 2 --alpha-mode absolute --alpha-absolute-base 1.0 --tag v14repl_fixed2` | V14 replication on LLaMA L22 (paper §RQ3 ρ=0.919 claim) | ~30 h |
| 2 | `run_aligned_factor_steering.py --experiment b --model gemma` | Exp B: Shared-axis on Gemma L12 (IC / SM / MW) | ~36 h |
| 3 | `run_aligned_factor_steering.py --experiment a --model llama` | Exp A: Shared-axis on LLaMA L25 (SM / IC / MW) | ~24 h |

Archive: all previous (wrong-prompt) results were moved to `results/json/archived_wrong_prompts/` on E8 and to `results/logs/` with `e8_fixed_*` / `e8_expb_gemma_v{2,3,4}*` prefixes. Nothing from the wrong-prompt runs will propagate into the paper.

## v4.4 Per-Experiment Intent / Hypothesis / Verification

### Exp C — LLaMA L25 Per-Task BK Direction (positive control)

**Intent**: Verify that L25 is intervention-sensitive with the per-task BK direction before committing 36+ hours of GPU time to the shared-axis experiments. If this fails under exact behavioral replay, the hidden-state extraction itself is suspect and no higher-level experiment can recover.

**Hypothesis**: Steering LLaMA hidden states along the SM-specific `centroid(BK) − centroid(VS)` axis at L25 under exact behavioral replay produces a monotone dose-response in BK rate over α ∈ {−2, −1, −0.5, 0, +0.5, +1, +2}. Expected baseline BK at α=0 ≈ 36% (matches original LLaMA SM).

**Verification**:
- Cochran–Armitage trend test across 7 α levels on BK counts
- Permutation p-value from 100 norm-matched random directions
- Kill criterion: perm p > 0.05 on Cochran–Armitage statistic → gate fails, Exp A skipped
- Sanity gate: if α=0 BK rate is far from the original 36% (outside [20%, 55%]), flag as prompt-drift warning

### Exp A — LLaMA L25 Shared-Axis Steering (3 tasks)

**Intent**: Test whether the centroid_pca shared axis (learned from all 3 tasks via precision-weighted between-class scatter) is causally intervention-relevant on the task with the strongest observational signal (LLaMA L25 r1, +0.092 AUC aligned-random margin).

**Hypothesis**: Positive-direction steering along the shared axis increases BK rate and I_BA in ≥2 of 3 tasks (SM confirmatory, IC and MW exploratory).

**Verification**:
- Confirmatory: SM Cochran–Armitage p < 0.05 (two-sided)
- Secondary: OLS β₁ on log(terminal wealth) and mean I_BA (continuous endpoints, higher power)
- Specificity: perm p from 100 random directions × 50 games × 7 α
- Kill criterion: SM CA non-significant AND all 3 continuous endpoints non-significant → conclude "shared axis at L25 does not produce detectable behavioral change at this α scale under this prompt distribution"

### Exp B — Gemma L12 Shared-Axis Steering (3 tasks)

**Intent**: Replicate Exp A's shared-axis design on the second model to establish cross-model causal evidence. Gemma L12 was chosen because it gave the only 6/6 observational transfer (Gemma L12 r1, +0.083 AUC mean aligned-random).

**Hypothesis**: Same as Exp A with IC as confirmatory (Gemma IC baseline BK ≈ 10.8% is the highest among Gemma tasks — best power for binary endpoint). SM and MW exploratory on continuous endpoints only (BK too sparse in Gemma — 2.7% SM, 1.7% MW).

**Verification**:
- Confirmatory: IC Cochran–Armitage p < 0.05, 50 random directions
- Exploratory: SM / MW terminal wealth + I_BA OLS (no binary endpoint for these)
- Kill criterion: IC CA non-significant AND SM/MW continuous non-significant

**Early signal (Phase 0 sanity, 2026-04-13 21:55)**: Gemma IC α=0 BK=2/20 (10.0%), α=−2 BK=1/20 (5.0%). This hits the original IC baseline (10.8%) exactly and shows the correct direction under negative steering. Previous (wrong-prompt) run gave BK=0/200 at all α, confirming the fix is effective. LLaMA Exp C/A/V14 Phase 0 results still in progress.

### V14 Replication — LLaMA L22 Per-Task BK (paper §RQ3 claim)

**Intent**: Independently reproduce the paper's main causal claim (LLaMA SM L22, ρ=0.919, BK 38%→60%, perm p=0.048) under exact behavioral replay. This determines whether the paper's headline RQ3 number survives the prompt correction.

**Hypothesis**: The original V14 result is robust to prompt-distribution correction because the same hidden-state direction is being used; only the input context changes. Predicted ρ remains strongly positive (>0.7) and perm p < 0.1.

**Verification**:
- Spearman ρ across 7 α levels (200 games each)
- Permutation p from 20 random directions (V14 original protocol)
- Decision tree:
  - If ρ > 0.9 and perm p < 0.05 → paper §RQ3 numbers unchanged (update methods only to note prompt alignment)
  - If ρ ∈ [0.7, 0.9] and perm p < 0.1 → update paper numbers, keep claim
  - If ρ < 0.7 or perm p ≥ 0.1 → claim weakens; rewrite §RQ3 more cautiously

## v4.5 Paper Update Plan

Structural principle: the paper's three-RQ structure, behavioral results (Section 3.1), and readout tables (Table 1 / Table 2 / Table 3) stay unchanged. Updates are localized to the causal / steering subsection and a new methods paragraph.

### Targets in `content/3.results.tex`

| Section | Current (v3) | Change (v4) |
|---|---|---|
| §3.2 methods preface | No mention of steering prompt format | Add one paragraph: "steering experiments replay games with the original prompt distribution (ROLE_INSTRUCTION + G+M components + original round limits + original betting format)" |
| §RQ3 steering paragraph (lines ~220 in current tex) | Reports V14 ρ=0.919, perm p=0.048, BK 38%→60% | Replace with new numbers from §v4.4 V14 replication. Keep the narrative structure ("one within-task direction monotonically linked to bankruptcy rate") regardless of exact number, unless the replication kills the claim entirely. |
| §RQ3 cross-task steering | Reports sign-reversal ρ = −0.964 / −0.818 | Rerun as part of Exp A; update if sign-reversal pattern holds |
| Figure `fig:steering` | V14 dose-response + cross-domain panels | Regenerate both panels with new data |
| Abstract | Mentions "ρ=0.919" indirectly via RQ3 summary | Replace with new number or soften if replication weakens |
| Discussion §steering limitations | No mention of prompt alignment | Add: "prior steering experiments used simplified prompts; v4 replicates under the original prompt distribution and finds [result]" |

### New material to add if Exp A / Exp B succeed

If shared-axis steering passes its confirmatory endpoint in either Exp A or Exp B, a new §RQ2 sub-paragraph is added: "Cross-task causal validation via shared-axis intervention." Intent: move RQ2 from purely observational (aligned_hidden_transfer AUC) to causal (shared axis as intervention target). Insert after the existing L22 / L25 r1 discussion.

If Exp C fails the gate, no new §RQ2 causal sub-paragraph is added; instead, a limitation sentence goes into Discussion: "Attempted causal validation of the shared axis at LLaMA L25 did not produce detectable behavioral change at α ∈ [−2, +2]; the cross-task observational result remains the primary evidence for partial shared structure."

### Honest re-framing of sparse-feature transfer failure (RQ2 §2)

Following user feedback, a single clarifying sentence is added to the sparse-feature transfer paragraph: "sparse SAE features are activated at the token level, so cross-task failure partially reflects differences in prompt surface form (slot machine / investment game / mystery wheel) rather than only differences in internal computation. The hidden-state aligned transfer succeeds precisely because it operates above this surface-form sensitivity." This does not weaken the claim; it explains why sparse transfer fails where hidden-state transfer succeeds.

## v4.6 Execution Checklist

- [x] Kill all wrong-prompt processes on E8 (done 20:00 UTC)
- [x] Archive wrong-prompt JSONs (`archived_wrong_prompts/`)
- [x] Fix `build_{sm,ic,mw}_prompt` in `run_v12_all_steering.py`
- [x] Fix `MAX_ROUNDS`, MW zone mechanics, SM/MW win tracking, parser precedence
- [x] Codex review pass 1 (4 BUGs found) → all fixed
- [x] Codex review pass 2 → approved with 2 accepted WARNs
- [x] Sync fixed scripts to E8 (md5 verified)
- [x] Launch 4 parallel experiments (Exp C on GPU 0, V14 repl on GPU 1, Exp B on GPU 2, Exp A on GPU 3)
- [x] First verification: Gemma IC α=0 BK=10% matches original (vs previous wrong-prompt 0%)
- [ ] LLaMA Phase 0 α=0 baseline verification (~23:00 UTC expected)
- [ ] Exp C gate decision
- [ ] V14 replication ρ / perm p
- [ ] Exp A / B main sweep completion
- [ ] Paper §RQ3 rewrite based on new numbers
- [ ] Figure regeneration
- [ ] Discussion §steering limitations update
- [ ] Push new results to HuggingFace mirror

## v4.7 Stop Rules (supersede v3 Stop Rules for steering only)

- **Hard stop on crash**: if any experiment crashes >3 times with the same error, stop that branch and escalate before retry.
- **Soft stop on null**: if Exp C perm p > 0.05, stop Exp A (gate), but continue Exp B independently (different model, different layer, different task as confirmatory).
- **Hard stop on prompt drift**: if Phase 0 α=0 baseline differs from original by more than a factor of 2 (e.g., LLaMA SM observed <18% or >54%, vs original 36%), pause and re-audit prompts before proceeding.
- **Budget cap**: if total wall clock exceeds 48 h for Exp A or Exp B, reduce null direction count to 50 and restart null phase only.
