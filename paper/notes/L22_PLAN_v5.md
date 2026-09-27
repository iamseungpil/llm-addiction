# L22 Multi-Task Strengthening Plan v5 (FINAL — frozen seeds)

## v5 changes from v4
Replace symbolic `g ∈ {1000, ...} + task_seed_offset` with explicit numeric seed lists per cell.

## Frozen evaluation seed lists (no further selection allowed)

`task_seed_offset` per existing pipeline: `{"sm": 0, "ic": 100000, "mw": 200000}`.

| Cell | Eval g range | Concrete seed range |
|---|---|---|
| H1 LLaMA SM L22 n=200 | g ∈ [1000, 1199] | seeds [1000, 1199] |
| H2 LLaMA MW L22 n=100 | g ∈ [1000, 1099] | seeds [201000, 201099] |
| H3 Gemma SM L22 n=100 | g ∈ [1000, 1099] | seeds [1000, 1099] |
| H4 LLaMA SM L23 n=50 | g ∈ [1500, 1549] | seeds [1500, 1549] |
| H5 LLaMA SM L22 n=20 axis null | g ∈ [2000, 2019] | seeds [2000, 2019] |

Original discovery NPZ used g ∈ [0, ~199] (per existing 6-model behavioral run); g ≥ 1000 guarantees no overlap. H4 and H5 use disjoint offsets [1500..., 2000...] to avoid cross-cell paired-design contamination.

## Frozen RNG seeds

- Axis-null random direction generator: `numpy.random.Generator(numpy.random.PCG64(2026042700))`.
- 30 unit directions sampled isotropically (Gaussian + L2-normalize) and rescaled to `||ĥ_BK||`.
- Bootstrap RNG: `Generator(PCG64(2026042701))` for paired bootstrap (H1, H2).
- Randomization-test RNG: `Generator(PCG64(2026042702))` for sign-flip permutations.

## Frozen alpha grid (immutable)

`α ∈ [-2.0, -1.0, -0.5, 0.0, +0.5, +1.0, +2.0]` for H1/H2/H3/H4. `α ∈ [+1.0]` for H5.

## Frozen layer/sign

| Cell | Model | Layer | Direction sign |
|---|---|---|---|
| H1, H4, H5 | LLaMA-3.1-8B | L22, L23 (H4) | `mean(h_bk) - mean(h_stop)` |
| H2 | LLaMA-3.1-8B | L22 | same |
| H3 | Gemma-2-9B | L22 (drop H3 if NPZ absent) | same |

`+α` applied at the target layer increases bk-rate (matches body §4.4 dose-response and existing pipeline `compute_per_task_direction`). Inherited from body §4.4; v5 does not re-select.

## --g-offset semantics

The `--g-offset` argument is the **relative (intra-task) index**; the absolute random seed used inside the model = `g_offset + i + TASK_SEED_OFFSET[task]` for `i ∈ [0, n_games)`. Concrete absolute seeds in the table above are derived: H2's "seeds [201000, 201099]" means `g_offset = 1000` plus `TASK_SEED_OFFSET[mw] = 200000`.

## Statistical tests (verbatim from v4)

- Primary (H1 → H2 fixed-sequence at α=0.05): paired Δ_s = bk_s(α=+2) − bk_s(α=−2); one-sided randomization test (10,000 sign-flip perms), one-sided p = (#{permuted_mean ≥ observed_mean} + 1) / 10001; paired bootstrap 95% CI from 1,000 seed-resamples.
- Secondary exploratory: 7-α Spearman ρ_bk, OLS slope.
- H5 descriptive: rank of |Δbk_rate| of true ĥ_BK among 31 directions; rank-p = rank_true / 31.

## Compute and allocation (verbatim from v4)

Same 9-GPU plan: SSH to 4 nodes (ic-0424, mw-0424, adapted-ram, natural-mongoose) and use GPUs 1/2/3 of each.

## Body update mapping

| Outcome | Body action |
|---|---|
| H1 randomization-p < 0.05 + H2 p < 0.05 | §4.4: append "stronger held-out replication at n=200 (Δbk_rate(+2,−2) = X pp, randomization p = …, bootstrap 95% CI …) and second-task generalization at MW (Δ = … pp)". Phrasing "stronger held-out replication and within-model breadth", **not** "concern resolved". |
| H1 pass + H2 fail | §4.4 mentions only n=200 SM replication |
| H1 fail | §4.4 unchanged |
| H3, H4, H5 | Appendix D.6 only |

## Smoke-critic-iterate (next step)

Write `run_l22_held_out_steering.py` separately; smoke + codex critic; iterate until clean; SSH-deploy.

---

**Plan v5 is committed; proceeding to code.**
