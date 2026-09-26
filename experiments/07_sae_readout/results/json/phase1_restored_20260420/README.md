# Phase 1 Restored Dose-Response (2026-04-20)

## Provenance

Seven `aligned_factor_steering` experiments (Exp A shared-axis LLaMA × 3 tasks,
Exp B shared-axis Gemma × 3 tasks, Exp C per-task LLaMA SM) had completed
Phase 1 (n = 200 games × 7 alphas) by 2026-04-19 02:49 UTC and saved
`completed_alphas` with per-game outcomes into the alpha-level checkpoints.

On 2026-04-20 the Singularity container backing the PRM A100 job was reset
during Phase 2 (null distribution). The `/scratch/llm_addiction/` tree was
wiped. Phase 2 partial data was lost; Phase 1 checkpoints had been mirrored to
the HuggingFace dataset in time and survived intact.

This folder reconstructs the Phase 1 dose-response numbers that would have
appeared inside the final `aligned_steering_*.json` files.

## Source

- 7 checkpoints at `sae_v3_analysis/results/checkpoints/` on
  `llm-addiction-research/llm-addiction` HF dataset, committed 2026-04-17 to
  2026-04-19.
- Restoration script: `sae_v3_analysis/src/restore_phase1_from_checkpoints.py`.

All Phase 1 game-level data (200 per alpha × 7 alphas × 7 experiments =
9,800 games) was generated through `exact_behavioral_replay.py`, the
canonical bridge mandated by `docs/PAPER_CANONICAL.md` for steering numbers.

## Per-experiment result (Spearman ρ on 7-point BK-rate curve)

| Experiment | Layer | ρ_BK | p | Verdict (α=0.05) |
|---|---|---|---|---|
| A LLaMA IC (shared axis → IC) | 25 | +0.472 | 0.284 | Not significant |
| A LLaMA MW (shared axis → MW) | 25 | +0.018 | 0.969 | Not significant |
| A LLaMA SM (shared axis → SM) | 25 | -0.667 | 0.102 | Not significant |
| B Gemma IC (shared axis → IC) | 12 | +0.805 | 0.029 | Significant raw; fails Holm(m=7) |
| B Gemma MW (shared axis → MW) | 12 | -0.126 | 0.788 | Not significant |
| B Gemma SM (shared axis → SM) | 12 | -0.636 | 0.124 | Not significant |
| C LLaMA SM (per-task SM direction) | 25 | +0.571 | 0.180 | Not significant |

## Contrast with pre-audit v12 numbers

The NMT manuscript currently cites `v12_n200_20260327_030745.json` which
reported ρ = 0.919, p = 0.003 for LLaMA SM L22 (per-task direction). That run
used the pre-bugfix `play_game_rich`. The exact-replay reconstruction (Exp C
above, same direction, L25 per aligned-axis config) gives ρ = 0.571 and is
not significant at the n = 200 × 7-alpha granularity.

Phase 2 permutation null is not reconstructable from the surviving data.

## Caveats

- Layer for Exp C is L25 in this run (aligned-axis wiring); the v12 claim
  was L22. Layer-matched rerun is still required before retiring the v12
  claim or replacing it in the manuscript.
- Holm correction uses m = 7 by experiment-level family. Finer-grained
  families (by task) may change the verdict on Exp B IC.
- The `null_distribution.status = not_computed_phase2_lost` field in each
  JSON flags that permutation p-values are outstanding.
