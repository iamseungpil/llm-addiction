# Plan v5.2 — Track 0 W3 Cross-Model Matched-Cap Replication

> **Status**: Round 5 of self+codex review loop. v5.2 incorporates self-critic Round 4.
> **Supersedes**: PLAN_4NODE_EXECUTION_2026_05_07.md §1bis (v4) for Track 0 only.
> **Key v5.2 deltas** vs v5.1: random effects forced to `(1 | model)` only (§3.3, addresses identifiability with 6 levels), signature thresholds tightened to 0.5/5.0 with margin justification (§2.1, codex round 4 (b)), Holm-corrected multiple comparisons in parity check (§3.5.1, (c)), open-weight system message decision **forced and frozen** to "no system, no ROLE_INSTRUCTION" (§3.1.2 + §8, (e)), per-model matrix re-anchored to "cap-ablation legacy + per-provider 6-model panel chat layer only" with explicit caveat that this is matched-within-provider (§3.1.2, (a)).

---

## §0. Three-axis frame

Intent → Hypothesis → Verification. Each section asks one question only.

---

## §1. Intent

What is Track 0 W3 trying to *show*?

The paper §3.2 mechanism claim H3 ("the slot-machine bankruptcy gap is freedom-to-choose at root rather than range expansion") is supported by exactly one model (GPT-4o-mini, via the matched-cap experiment in Figure 3d). Track 0 W3 asks whether that mechanism — *both* the gap at the highest matched cap *and* its qualitative signature (small bets, many rounds in the variable arm) — replicates on five additional models. The answer determines whether §3.2 keeps a cross-model mechanism claim or narrows to a GPT-4o-mini-specific dissociation.

Two non-negotiable design consequences:

1. **Same task** — slot machine, not investment choice. The IC arm of cap variation is already populated for these five models in HF and is a separate Plan v4 deliverable.
2. **Same matched-cap manipulation as Figure 3d**, defined byte-for-byte by the live re-run of `legacy/gpt_fixed_bet_size_experiment.py` and `legacy/gpt_variable_max_bet_experiment.py` (now manifested in `PAPER_CANONICAL_CODE.md`). Anything new — `ROLE_INSTRUCTION`, "cautious" qualifier, max_rounds asymmetry erasure, or different choice-text — is **a different experiment**, not Track 0.

The five added models are: Gemma-2-9b-it, LLaMA-3.1-8B-Instruct, Claude-3.5-haiku, Gemini-2.5-flash, and gpt-4o (full, as the API parity twin to gpt-4o-mini). The original-model re-baseline is gpt-4o-mini, matching what HF data files actually contain (`model: "gpt-4o-mini"`); Plan v4 §1bis prose saying "GPT-4o (full)" was imprecise and is corrected here.

---

## §2. Hypothesis

### 2.1 Co-primary (decision-driving)

H_W3_gap and H_W3_signature are **co-primary** — both must pass for §3.2 to retain the cross-model mechanism claim. Codex Round 2 C2 noted that a positive bankruptcy gap alone is consistent with both "freedom-to-choose generalises" and "weak models tilt under variable mode"; we resolve that ambiguity by requiring the qualitative signature to ride along.

**H_W3_gap** — at the highest matched cap (=$70), variable betting produces strictly higher bankruptcy than fixed betting in the population of six SM-task LLMs.

```text
bankrupt ~ condition × cap + ( condition × cap | model )
β_primary  = E[bankrupt | variable, cap=70]  −  E[bankrupt | fixed, cap=70]   (logit scale)
```

Decision rule: **lower 2.5 % posterior quantile of β_primary > 0** (Bayesian credible interval, natural readout for `bambi`/`pymc` MCMC; replaces the inconsistent Wald rule from v5.0).

**H_W3_signature** — when H_W3_gap holds, the variable-arm signature also holds in the same direction across at least 4 of 6 models:

- per-model `mean(variable_bet | cap=70) / 70 < 0.5` (model bets *clearly* sub-cap on average), **and**
- per-model `mean(variable_rounds | cap=70) / mean(fixed_rounds | cap=70) > 5.0` (variable arm survives many more rounds).

Thresholds 0.5 and 5.0 are calibrated from the legacy GPT-4o-mini data: at cap=$70 the legacy variable arm bet ~$15-20 ≈ 0.21–0.28 × cap, and played ~16–19 vs ~1–2 fixed-arm rounds (≈8-19× ratio). v5.1 used 0.6 / 3.0 which would have admitted "near-cap betting with moderate rounds" as a false-positive replication; v5.2 tightens to 0.5 / 5.0, still loose by ~2× vs the legacy effect (0.5 vs legacy 0.28; 5.0 vs legacy 8-19) so genuine replication passes with margin while a non-mechanism model that just tilts under variable mode does not.

### 2.2 Falsifiability

Three failure modes, each tied to a specific decision:

| Failure | Detection | Decision |
|---|---|---|
| Statistical fail | lower 2.5 % posterior quantile of β_primary ≤ 0 | W3-gap-fails: §3.2 narrows to GPT-4o-mini dissociation |
| Mechanism fail | β_primary > 0 but signature passes in <4/6 models | W3-mech-fails: §3.2 keeps the *gap* claim (cross-model phenomenon) but explicitly *removes* the "freedom-to-choose at root" prose |
| Protocol fail (§3.5) | Track 0 v6's gpt-4o-mini cells fail parity vs the live legacy re-run | block launch — code, not model, is the bug |

The third failure is the new gate Plan v4 lacked.

### 2.3 Scope clarifications (codex C2 follow-up)

H_W3_gap is not by itself a "freedom-to-choose" claim. The mechanism story requires both the gap *and* the qualitative bet-distribution shape — that is the entire reason §3.2 prose says "betting smaller average amounts than the cap … by playing 16-19 rounds." Promoting signature to co-primary aligns the hypothesis layer with the prose layer.

---

## §3. Verification

### 3.1 Replication target — what "matched" actually means per model

#### 3.1.1 Authoritative protocol source

The live re-run of `legacy/gpt_fixed_bet_size_experiment.py` + `legacy/gpt_variable_max_bet_experiment.py` is the protocol of record. Track 0 v6 must execute the same prompt, system message, generation parameters, and parser as those scripts on the gpt-4o-mini cells. Hand-extracted bankruptcy numbers from `LLM_Addiction_NMT_KOR/generate_paper_figures.py` are an additional sanity reference (§3.5), not the only ground truth.

#### 3.1.2 Per-model parity matrix (codex C1)

"Same protocol" decomposes differently by model class:

| Dimension | gpt-4o-mini (legacy) | gpt-4o (full) | Claude-3.5-haiku | Gemini-2.5-flash | Gemma-2-9b-it (open) | LLaMA-3.1-8B-Instruct (open) |
|---|---|---|---|---|---|---|
| Prompt body | identical to legacy | identical | identical | identical | identical | identical |
| System message | OpenAI `system` role, "rational decision maker… step by step…" — verbatim | same | **omitted** (Anthropic legacy `run_claude_experiment.py` uses no system message) | **omitted** (Google legacy `run_gemini_experiment.py` uses no system message) | **omitted** (matches `paper_experiments/slot_machine_6models/src/llama_gemma_experiment.py` SM 64-cond panel — no system message, no `ROLE_INSTRUCTION` for Track 0 either; v5.2 freezes this decision) | same — **omitted**, no `ROLE_INSTRUCTION` |
| Sampling | `temperature=0.7, max_tokens=600` | same | `temperature=0.5, max_tokens=300` (legacy claude) | default Google generation_config (legacy gemini) | `temperature=0.7, max_new_tokens=1024, do_sample=True, min_new_tokens=10` (HF generate parity to llama_gemma_experiment runner) | same |
| Parser | `improved_parse_gpt_response(bet_type, current_balance)` (recovered, §5.3) | same | same | same | same | same |
| max_rounds | fixed=100, variable=50 (legacy asymmetry preserved) | same | same | same | same | same |

This matrix is the per-model parity contract. Each entry "identical" or "same" is a property the v6 code must reproduce; entries that differ ("omitted", "0.5", "default") are inherited from each provider's legacy 6-model SM panel runner so that the comparison stays "matched within each provider's own legacy convention." We do *not* try to homogenise system messages or temperatures across providers — that would itself be a protocol drift relative to the legacy 6-model panel, which is the only place these providers have prior SM data to compare against.

### 3.2 Sample sizes (staged)

#### 3.2.1 Power back-of-envelope (codex C6)

Legacy GPT-4o-mini at cap=$70: variable bankruptcy ≈ 0.17, fixed ≈ 0.005, gap ≈ 0.165. Cluster-robust two-proportion approximation with assumed model-level ICC ρ = 0.10 and 6 models gives effective per-arm n ≈ n_total / (1 + ρ × (m̄ − 1)) where m̄ = 200 — effective n ≈ 200 / (1 + 0.10 × 199) ≈ 9.7 *per model* effectively, or 6 × 9.7 ≈ 58 effective overall per arm. At p₁ = 0.165, p₂ = 0.005, n_eff = 58, two-proportion z-test power ≈ 0.99 for the population-level gap. Stage 1 n=200/cell is therefore generous for the population-level test; the limit is per-model precision (per-model 95 % CI on bankruptcy at n=200 is ±5–8 pp at 17 %), which is fine for binary forest-plot replication but not tight enough to claim "model X uniquely fails." Stage 2 (n=500) is reserved for borderline population-level CIs only.

#### 3.2.2 Cells

- Stage 1: n=200 / cell / model. 6 × 4 × 2 = 48 cells = 9 600 games. The original Figure 3d uses 50 reps/cell — this quadruples for cross-model precision.
- Stage 2 gate (only on borderline Stage 1): n=500 / cell on cap=$70 cells, triggered when lower 2.5 % posterior quantile of β_primary ∈ [−0.005, 0.01] OR posterior 95 % CI half-width > 0.025 on the logit scale.

### 3.3 Statistical model (`bambi`)

```python
import bambi as bmb
m = bmb.Model(
    "bankrupt ~ C(condition) * C(cap) + (1 | model)",   # random intercept ONLY
    family="bernoulli",
    data=long_format_df,
)
fit = m.fit(draws=2000, tune=1000, chains=4, cores=4, random_seed=42)
```

**Why random intercept only**: with 6 models the random slope on `(condition × cap)` would estimate 8 group-level interaction parameters from 6 levels — under-identified. v5.0 / v5.1 tried `(C(condition) * C(cap) | model)` which would diverge in NUTS. v5.2 forces `(1 | model)` from launch and, as the strongest alternative, allows escalation to `(C(condition) | model)` (still 2 group-level params per model, identifiable) only if the random-intercept fit's posterior predictive shows model-level mis-fit in the gap term. The trade-off is explicit: the population-level β_primary is unbiased but we cannot put credible intervals on per-model heterogeneity from this fit. Per-model heterogeneity is reported separately as cell-level descriptive statistics with binomial CIs.

Primary readout: marginal contrast `E[bankrupt | variable, cap=70] − E[bankrupt | fixed, cap=70]` from the posterior, on logit and probability scales, with 2.5/97.5 % posterior quantiles plus parametric-bootstrap cross-check.

Secondary readout (per-model): forest plot of cell-level binomial proportions and Wald CIs for variable-minus-fixed bankruptcy at each cap, *not* model-level random-effect estimates.

### 3.4 Sanity checks

- **S1** (signature, descriptive, not gating): per-model `mean(variable_bet | cap=70) / 70` and `mean(variable_rounds | cap=70) / mean(fixed_rounds | cap=70)`. Reported in the forest plot regardless of pass; gate threshold for §2.1 H_W3_signature applies only to the count of models passing 0.6/3.0.
- **S2** (bankruptcy floor): per-model `bankruptcy(fixed, cap=10) ≤ 0.05`. If violated, that model has a non-baseline parsing or generation issue and is flagged for inspection before inclusion in the bambi fit.
- **S3** (game-physics determinism, deterministic-side only): replaying 5 games per cell with the same `seed` argument to `SlotMachineGame` and a *fixed dummy `response_fn`* must yield bit-identical `is_bankrupt`, `final_balance`, `total_rounds`. This isolates the slot RNG from API/model non-determinism — codex C5 noted that S4 in v5.0 was infeasible because API responses are non-deterministic, so v5.1 reframes S3 to test only the game-physics RNG, not the model's text generation.
- **S4** (parser idempotence): running the recovered legacy parser on each game's stored response strings reproduces the per-game decisions recorded in the JSON. Catches parser regressions at audit time.

### 3.5 Protocol parity check (the new gate Plan v4 lacked)

#### 3.5.1 Two-track parity

**Track A — code parity**: run Track 0 v6's gpt-4o-mini cap-ablation cells at **n=200 / cell** (codex C7b raised parity n from 50 to 200; binomial SE on a 17 % rate at n=200 is ~2.7 pp — ±5 pp tolerance is now ~2 SE, not 1 SE), then re-run `legacy/gpt_fixed_bet_size_experiment.py` and `legacy/gpt_variable_max_bet_experiment.py` *independently* on the same prompts at the same n. Compare cell means *and* per-game decision distributions (Kolmogorov–Smirnov on rounds-played; chi-square on bankruptcy).

Pass criterion (Holm-corrected at family-wise α=0.10 across the 8 cells; cell-level α' = 0.10/(8 − k) for the k-th sorted p-value):

- per-cell |bankruptcy_v6 − bankruptcy_legacy| ≤ 5 pp **AND**
- KS distance on rounds-played per cell ≤ 0.10 **AND**
- chi-square on (bankrupt, voluntary_stop, max_rounds) outcome distribution: pooled-across-cells p > 0.10 (single test, not 8) **OR** if reported per-cell, Holm-corrected at family-wise α=0.10 (so the smallest p must clear 0.10/8 = 0.0125, second smallest 0.10/7 = 0.0143, …).

The pooled test is recommended because it has higher power and a cleaner family-wise interpretation; v5.1 specified naive p>0.10 per cell which gives FWER ≈ 1−(1−0.10)^8 = 57 % — codex round 4 (c) flagged this and v5.2 corrects.

**Track B — historical parity**: separately compare to the hand-extracted Figure 3d numbers `[0.5, 0.3, 4.7, 0.5]` / `[1.0, 14.0, 16.5, 17.0]` with the same ±5 pp tolerance. Track B is informational only; Track A is the gate.

If Track A fails, **stop**: the v6 code drifted from legacy and must be fixed before the cross-model run can launch.

#### 3.5.2 Decision tree

```text
[ run Track 0 v6 gpt-4o-mini @ n=200 ] ─┐
                                          ├─→ [ Track A parity check (§3.5.1) ]
[ re-run legacy scripts @ n=200 ] ────────┘
                                            │
                                       ┌────┴────┐
                                     PASS      FAIL
                                       │         │
                                       │     STOP — fix v6 code, rerun §3.5
                                       ▼
                          [ launch 5-model x 4cap x 2mode @ n=200 ]
                                       ▼
                         [ fit hierarchical mixed-logit (§3.3) ]
                                       ▼
                              [ check sanity (§3.4) ]
                                       ▼
                          [ readout β_primary 95 % CrI ]
                                       │
                              ┌────────┼────────┐
                              │        │        │
                          CrI > 0  CrI ≤ 0  signature fails
                              │        │        │
                          W3-passes  W3-gap-fails  W3-mech-fails
                          surgery    surgery     (§2.2)
                          (§2.2)    (§2.2)
```

---

## §4. Out of scope

- M2 / M5 / M1 / D — own modules.
- IC cap-variation — already in HF (`investment_choice/bet_constraint*`); separate Track B deliverable.
- Any prompt manipulation beyond `BASE` — matched-cap deliberately fixes prompt content. Adding `+G`/framing makes it a different track.

---

## §5. Risks and fallbacks

1. **API non-determinism** on five new providers — Stage 2 gating handles, and §3.5 KS check on rounds-played catches gross drift even within a stochastic generator.
2. **AMLT preemption** — mitigated by `.markers` resumability (added 2026-05-08).
3. **Parser drift** — codex C5 raised this. **Resolved**: legacy `improved_gpt_parsing.py` recovered from git commit `9a4ee94^` and now lives at `legacy/improved_gpt_parsing.py` (237 lines). Track 0 v6 must `from improved_gpt_parsing import improved_parse_gpt_response` directly rather than re-implementing a `parse_response` function in `game_logic.py`. The current Track 0 `parse_response` is *not* equivalence-tested against the recovered legacy and must be replaced.
4. **gpt-4o-mini deprecation** — confirm OpenAI is still serving `gpt-4o-mini` (currently it is). If deprecated mid-rebuttal, pin to a specific snapshot date in the OpenAI API call (`gpt-4o-mini-2024-07-18` is the original).
5. **Chat-template token injection** (codex C7a, new) — Gemma and LLaMA chat templates inject system/turn tokens (`<|begin_of_text|>`, `<start_of_turn>user`, etc.) that are not present in the OpenAI raw-prompt protocol. Two implications: (a) the prompt body is byte-identical to legacy, but the wrapped-input token stream differs by tens of tokens; (b) Gemma/LLaMA cannot have *exactly* the same prompt as gpt-4o-mini because the OpenAI API hides its own templating. Mitigation: run the open-weight models with `tokenizer.apply_chat_template(...)` to take the official wrapping (matches `llama_gemma_experiment.py`'s SM 64-cond runner); document the difference in the §3.5 parity report; do not try to remove the chat template tokens.
6. **Bambi NUTS divergences** with `(condition × cap | model)` random slope at 6 models — 6 levels is on the low end for hierarchical identification. If divergences > 1 % we drop the random slope and refit with `(1 | model)` only, reporting it explicitly in the deviation log.

---

## §6. Decision-rule log (frozen before Stage 1 launch)

- Primary rule: §3.6 decision tree (above).
- Co-primary: H_W3_gap (β_primary CrI > 0) AND H_W3_signature (≥4/6 models pass 0.6/3.0 thresholds).
- Surgery prose: PLAN_4NODE_EXECUTION_2026_05_07.md §9bis "W3-passes" / "W3-fails" remain authoritative; v5.1 adds a third branch "W3-mech-fails" — see §2.2.
- Pre-registration freeze: this file (Plan v5.1) is the contract. Any change post-launch goes in a §13-style deviation log.

---

## §7. Resolved open issues

1. ~~GPT-4o vs gpt-4o-mini in §3.2 prose~~ → §1 corrects to gpt-4o-mini (matches HF data `model` field).
2. ~~n=200 power not derived~~ → §3.2.1 back-of-envelope shows ~0.99 power on population-level gap, MDE adequate.
3. ~~Legacy parser recovery~~ → done (§5.3); v6 imports `improved_gpt_parsing` directly.
4. ~~Stage 2 thresholds inherited without re-derivation~~ → kept Plan v4 thresholds, but §3.2.1 power calc shows Stage 2 likely unnecessary for the population claim. Stage 2 is now contingency-only.

---

## §8. Resolved decisions (closed before launch)

1. **Parity check execution path**: legacy scripts will be run via a thin shim that (a) `sys.path.insert(0, "/home/v-seungplee/llm-addiction/legacy")` so `from improved_gpt_parsing import improved_parse_gpt_response` resolves, and (b) writes outputs to `/scratch/x3415a02/data/llm-addiction/track0_w3/parity_legacy_baseline/` instead of the legacy script's hardcoded `/home/ubuntu/...` paths. The shim file lives at `paper_experiments/track0_w3_replication/src/run_legacy_baseline.py` and is the *only* monkeypatch allowed — the legacy script's prompt + parser + system msg is untouched.
2. **Open-weight system message** (codex round 4 (e)): **decision = (a) "no system message, no ROLE_INSTRUCTION"** for Gemma and LLaMA. This matches the legacy gpt cap-ablation protocol exactly (which has no system message) at the cost of *not* matching `llama_gemma_experiment.py`'s SM 64-cond panel protocol (which prepends ROLE_INSTRUCTION). The justification: Track 0's matched-cap manipulation is the protocol of record (Figure 3d source), not the 6-model 64-cond panel. Open-weight models in Track 0 will therefore see a *different* prompt than they saw in the 64-cond panel — that's an explicit design choice, documented here, and noted in the deviation log.
3. **Track A parity API cost**: ~15 min of gpt-4o-mini at n=200/cell × 8 cells = 1 600 games. Approved.

## §9. Decisions still open for round 5+ (minor)

These are not launch-blockers but should be revisited before Stage 2:

1. Whether to escalate from `(1 | model)` to `(C(condition) | model)` based on Stage 1 posterior predictive checks. Pre-commit: only if model-level residuals show systematic mis-fit at the gap.
2. Whether the §3.5 Track B (hand-extracted reference) catches anything Track A misses. Pre-commit: keep both for cheap defense-in-depth.
3. Whether to add a bambi posterior-predictive plot per model to the appendix. Pre-commit: yes, will be a 1-page appendix figure.
