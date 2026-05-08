# Plan v3 — Track L: Length / Survival Confound Re-analysis

> **Status**: Round 3, 2026-05-08. v3 incorporates codex Round 2 cleanup (4 fixes + 4 new issues).
> **v3 deltas vs v2**:
> - **Two co-primary datasets**: IC cap-variation (matched-cap mechanism defense) + SM 6-model panel (Figure 2 / §3.1 length-confound defense). Each gets its own readout; the rebuttal cites whichever passes (or both if both pass).
> - **Multinomial readout renamed**: `RR_per_decision` (relative risk on cause-specific bankruptcy contrast), not `HR`. Cox sensitivity uses Cox HR; we keep the names distinct.
> - **HR/RR threshold table tightened**: lower 95 % CI ≤ 1.0 ⇒ L-fails REGARDLESS of point estimate (codex Round 2 critique 5).
> - **Decision tree regenerated** to match multinomial primary (old "discrete logistic + Schoenfeld" line removed).
> - **Cluster ID = `(dataset, timestamp, cap, prompt_combo, model, game_id)`** — globally unique, codex Round 2 issue (cross-source `game_id` collisions).
> - **IC timestamp policy**: 2025-11-28 IC re-run is primary; 2025-11-21 + 2025-11-25 are reproducibility checks reported in appendix only.
> - **Firth fallback**: binary cause-specific (bankruptcy vs not-bankruptcy) Firth-penalised logistic on the same overlap subset; multinomial Firth is non-standard so we don't use it.
> - **Bet-acceptance check**: builder verifies `bet ∈ (0, balance_before]` AND `decision == 'continue' / 'bet'` AND a non-empty `outcome` field — guards against parser-rejected bets that the engine still logged.
> - **`long_table_overlap` clarified for IC**: IC has rounds 1–10, so "exclude variable-only late rounds" is moot (no SM-style 11–100 overhang); the SM-side overlap restriction stays.

> **Status (v2 superseded)**: Round 2, 2026-05-08. v2 incorporated codex Round 1 critique (6 issues).
> **Origin**: codex Round 3 (paper-wide review) verdict — `§3 variable-arm length/survival
> confound is the most latent reject-risk attack`. Re-analysis on existing data, no new runs.
>
> **v2 deltas vs v1** (codex Round 1):
> 1. Bankruptcy event redefined as `(bet_amount >= balance_before) AND (result == 'L') AND (balance_after == 0)` — `balance_after == 0` alone misclassifies parser artifacts and rounding edges.
> 2. Voluntary stop is **informative** mode-dependent censoring; v2 promotes a multinomial competing-event hazard `{continue, bankrupt, voluntary_stop}` to primary, with cause-specific hazard for bankruptcy as the readout. Fine–Gray remains a Cox-side sensitivity only.
> 3. **Primary dataset switched to IC cap-variation** (`gpt4o_mini`, `gpt41_mini`, `claude_haiku`, `gemini_flash` × cap ∈ {10,30,50,70} × fixed/variable, HF `investment_choice/bet_constraint/results/`). SM 6-model panel demoted to supplementary appendix. Reason: §3.1 SM has only cap=\$10, so it cannot speak to the matched-cap §3.2 mechanism claim that Track L is defending.
> 4. Random intercept on 6 models is **demoted to sensitivity only**; primary fit is logistic with model as a 6-level fixed-effect dummy + cluster-robust SE by `(model, game_id)`. (Echoes Plan v5.2 §3.3 random-intercept-only argument: 6 levels are under-identified for hierarchical estimation.)
> 5. HR decision rule tightened: pre-registered lower 95 % CI > 1.2 AND point RR > 1.5 → "L-passes (per-decision risk)"; lower 95 % CI ∈ (1.0, 1.2) OR RR ∈ (1.0, 1.5) → "L-mixed"; lower 95 % CI ≤ 1.0 → "L-fails (exposure-cumulative only)".
> 6. New **support / separability diagnostics** in §3.4: report bankruptcy cell counts per `(model, bet_type, balance_quartile, round_block)`; primary fit restricted to overlapping balance × round support (variable-arm rounds 11–100 with no fixed-arm counterpart are excluded from the per-decision contrast and reported as a sensitivity).

---

## §0. Three-axis frame

Intent → Hypothesis → Verification.

---

## §1. Intent

What is Track L trying to *show*?

The paper §3.1 reports that variable betting raises bankruptcy from a 0–3 % range under fixed
to a 5–72 % range under variable, with LLaMA-3.1-8B at the extreme ($0.4\%\!\to\!72.3\%$).
The §3.2 prose interprets this as `freedom-to-choose at root, not range expansion`. But the
paper's own descriptive statistics also state that under variable betting the model plays
~16–19 rounds against fixed's ~1–2 rounds. A reviewer can therefore reframe the entire §3
contribution as `variable mode causes longer play, longer play mechanically multiplies the
opportunity to go bankrupt` — i.e., the bankruptcy gap is an **exposure** effect, not a
**per-decision risk** effect.

Track L isolates per-decision bankruptcy risk from cumulative-exposure risk on existing data.
The output is a relative-risk ratio (RR) for `bet_type=variable` vs `bet_type=fixed` from a
multinomial competing-event hazard, after conditioning on round number, balance, prompt
combo, and model. If RR > 1 with a CI excluding 1, the §3.2 freedom-to-choose interpretation
survives the exposure-control re-analysis. If RR ≤ 1, the §3 claim must narrow from
`variable mode is more dangerous per decision` to `variable mode allows longer play, which
compounds risk over rounds`.

This intent has three non-negotiable design consequences:

1. **Existing-data only**. No new LLM queries; no AMLT.
2. **Two co-primary datasets** (v3): (i) IC cap-variation HF cache
   `investment_choice/bet_constraint/results/` covers `gpt4o_mini`, `gpt41_mini`,
   `claude_haiku`, `gemini_flash` × cap ∈ {10,30,50,70} × fixed/variable × {BASE,G,M,GM}
   × 50 reps, `max_rounds=10` — directly mirrors Figure 3d's matched-cap structure for
   defending §3.2's mechanism claim. (ii) §3.1 6-model SM panel raw JSONs (paper-canonical
   cap=\$10 SM panel — 6 models × 64 conditions × 50 reps, `max_rounds=100`) — defends the
   length-confound attack on Figure 2 / §3.1's headline cross-model bankruptcy gap. The
   rebuttal cites whichever passes; both are reported.
3. **Per-round event-history form**. Each row is `(dataset, timestamp, model, cap,
   prompt_combo, bet_type, game_id, round, decision, bet_amount, balance_before,
   balance_after, outcome ∈ {continue, bankrupt, voluntary_stop})`. We do not collapse
   to game-level outcomes — that would re-mix length and per-decision risk back together.

Anything that requires re-querying an LLM is **out of scope for Track L** (handled by Track 0
or by Plan v4 § B/D modules).

The unit of analysis is **the round**, not the game. Existing §3 analyses were game-level;
Track L is round-level.

---

## §2. Hypothesis

### 2.1 Primary

**H_L_main** — at each round, conditional on balance, round number, prompt combo, and
model fixed-effect, the per-decision bankruptcy relative-risk under `bet_type=variable`
strictly exceeds that under `bet_type=fixed`. Estimated via cause-specific multinomial
logistic on three competing outcomes per round (`{continue, bankrupt, voluntary_stop}`).
Reported separately for IC and SM datasets.

Formal estimand (multinomial RRR, not Cox HR):

```text
P(outcome=bankrupt at round t | survived to t, x_t)
  = softmax(β'·x_t)[bankrupt]
where x_t = [bet_type, cap, log1p(balance_before), round, prompt_combo, model_dummy]

RR_per_decision = exp(β_var^{bankrupt})    # multinomial relative-risk ratio (RRR)
```

Decision rule: **lower 95 % CI of RR_per_decision > 1**. Equivalently, lower 2.5 %
posterior quantile of `β_var > 0` (Bayesian) or lower 95 % Wald CI of `β_var > 0`
(frequentist GLMM).

### 2.2 Secondary (mechanism robustness)

**H_L_balance_strata** — when bankruptcy is computed only within matched balance bins
(quartiles of `balance_before`), the within-bin variable-vs-fixed bankruptcy odds-ratio
remains > 1 in at least 4 of 6 models and the pooled within-bin OR > 1 with 95 % CI
excluding 1.

**H_L_round_strata** — restricting to rounds 1–2 (the "fixed-arm exit window") still shows
variable > fixed bankruptcy. This rules out `the variable gap is entirely concentrated in
late rounds where the fixed arm is no longer present`.

### 2.3 Falsifiability

**v3 pre-registered RR thresholds** (codex Round 1 critique 5):

| Outcome | RR point | lower 95 % CI | Decision |
|---|---|---|---|
| **L-passes** | > 1.5 | > 1.2 | §3 retains "per-decision freedom-to-choose at root"; appendix Track L RR table; 3–5 sentence body acknowledgement of length asymmetry |
| **L-mixed** | (1.0, 1.5] | (1.0, 1.2] | §3 keeps gap claim; tone interpretation down to "per-decision risk is elevated but mostly compounds with extended play"; balance-stratified appendix table |
| **L-fails** | (any) | ≤ 1.0 | §3 narrows to "variable mode extends play, compounding −10 % EV"; "freedom-to-choose at root" replaced. **Lower CI ≤ 1.0 ⇒ fail regardless of point estimate.** |
| **PH fail** | (any) | (any) | report discrete logistic time-varying-coefficient + Cox time-stratified; document |
| **Robustness fail** | > 1.5 but balance-strata or round-strata fails | — | "L-mixed" (above row) + balance-stratified appendix |

The first failure is the rebuttal-saving / rebuttal-narrowing dichotomy. The paper text
already partly acknowledges length asymmetry in §6 limitations; Track L makes that
acknowledgement quantitative.

---

## §3. Verification

### 3.1 Data sources

**Co-primary 1 (v3): IC cap-variation** in HF cache `investment_choice/bet_constraint/results/`
covers `gpt4o_mini`, `gpt41_mini`, `claude_haiku`, `gemini_flash` at cap ∈ {10,30,50,70}
× fixed/variable × {BASE, G, M, GM}. **Primary timestamp: `2025-11-28`** (latest stable
re-run). Reproducibility-only: 2025-11-21 and 2025-11-25 — appendix only. 4 models × 4 caps
× 2 modes × 4 prompts × 50 reps = 6 400 games at the primary timestamp, with `max_rounds=10`.
The data files contain per-round `decisions` lists with `balance_before`, `bet`, `choice`,
`outcome`, `balance_after`, `prompt`, `response` fields. Defends §3.2 matched-cap mechanism.

**Co-primary 2 (v3): SM 6-model panel** in `paper_experiments/slot_machine_6models/data/results/`
+ HF `iamseungpil/llm-addiction-research/llm-addiction:slot_machine/{model}/` raw JSONs (6 models × 64 conds × 50 reps × cap=$10, max_rounds=100). Defends §3.1 length-confound on Figure 2:

- `final_gemma_20251004_172426.json` (legacy gemma) + `final_gemma_20260227_002507.json`
  (corrected v4_role) — 3 200 games each
- `final_llama_20251004_021106.json` + `final_llama_20260315_062428.json` — same
- `gpt5_experiment_20250921_174509.json` (gpt-4.1-mini) — 3 200 games
- `claude_experiment_corrected_20250925.json` — 3 201 games
- `gemini_experiment_20250920_042809.json` — 3 200 games
- gpt-4o-mini SM 64-cond panel — corrected parsing archive (per appendix `5.methods.tex`
  note: `round_details` + `game_history` aligned by round index)

Each file has per-round records — the body of `results[i].history` (or
`results[i].decisions`) carries `round, bet, result, balance` already.

Secondary (for cap-variation): IC cap-ablation HF cache files
`investment_choice/bet_constraint/results/{gpt4o_mini, gpt41_mini, claude_haiku,
gemini_flash}_{10,30,50,70}_{fixed,variable}_*.json`. These include 4 caps × 2 modes per
model and so allow a cap-stratified hazard fit.

Both sources are read-only re-analysis; **no LLM is queried**.

### 3.2 Round-level table construction

For each game in each file, emit one row per round:

```text
columns: [model, task, prompt_combo, bet_type, game_id, round, balance_before,
          bet_amount, decision, balance_after, round_ended_in_bankruptcy,
          round_ended_in_voluntary_stop, total_rounds_in_game]
```

`round_ended_in_bankruptcy` = 1 iff **bet was placed AND `bet_amount >= balance_before` AND
result/outcome == loss AND `balance_after == 0`** (codex Round 1 critique: `balance_after == 0`
alone misclassifies parser artifacts, rounding edges, and edge cases where the model stops
on \$0 without betting). `voluntary_stop` is the competing-risk event — it terminates the
game without consuming the balance and is **mode-dependent informative censoring** (fixed
arms stop earlier; treating stop as ordinary right-censoring biases the hazard). Plan v2
therefore models all three terminal/non-terminal outcomes jointly.

The all-variable subset and the all-fixed subset are kept separately so we can run the
hazard fit either pooled across `bet_type` (with `bet_type` as covariate) or stratified.

### 3.3 Statistical model

**Primary (v2)**: cause-specific discrete-time hazard via multinomial logistic on the
three competing outcomes per round.

```python
# rows: (model, game_id, round, bet_type, cap, balance_before, bet, choice,
#        outcome ∈ {continue, bankrupt, voluntary_stop}, prompt_combo)
import statsmodels.api as sm
import statsmodels.formula.api as smf
# Multinomial: outcome ~ bet_type + cap + log(balance) + round + C(model) + C(prompt_combo)
m = smf.mnlogit(
    "outcome ~ C(bet_type) + C(cap) + np.log1p(balance_before) + round + C(model) + C(prompt_combo)",
    data=long_table_overlap,   # restricted to overlapping balance × round support
).fit(cov_type="cluster", cov_kwds={"groups": long_table_overlap[["model","game_id"]].astype(str).agg("_".join, axis=1)})
beta_var_bankrupt = m.params.loc["C(bet_type)[T.variable]", "bankrupt"]
RR_per_decision   = np.exp(beta_var_bankrupt)         # cause-specific bankruptcy hazard ratio
```

Model-as-fixed-effect dummy (codex Round 1 critique 4) avoids the 6-group random-intercept
identifiability problem. Cluster-robust SE on `(model, game_id)` accounts for round-within-
game dependence.

**Sensitivity-only (v2)**: random intercept on `model` via `mixedlm` for sensitivity. Report
in appendix; not a launch-gate.

**Robustness 1 (PH)**: Cox proportional hazards via `lifelines.CoxTimeVaryingFitter` with
`bet_type` as baseline covariate and `balance_before` as time-varying. Schoenfeld residual
test for PH; if rejected, fall back to time-varying-coefficient Cox or to time-stratified
discrete logistic.

**Robustness 2 (separability/support, codex Round 1 critique 6)**: identify cells with
zero bankruptcy events `(model × bet_type × balance_quartile × round_block)`; if any cell
has 0 events under fixed mode and >5 under variable mode, report Firth's penalised
logistic regression instead of standard logistic. Restrict the primary contrast to the
overlapping balance × round support (variable-arm rounds 11–100 in SM, where no fixed-arm
counterpart exists, are excluded from the per-decision RR; reported as a separate
"variable-only late-round" appendix).

**Robustness 3 (Fine–Gray competing risk)**: Cox-side sensitivity using
`lifelines.CoxPHFitter` with bankruptcy as the event of interest and voluntary stop as a
competing risk via the Fine–Gray subdistribution hazard.

### 3.4 Sanity checks + support diagnostics (block conclusions if any fails)

- **S1 (primary IC)**: marginal bankruptcy rates per `(model, cap, bet_type)` for IC reproduce
  any pre-existing IC body numbers (no headline IC bankruptcy in the paper body — IC is
  bankruptcy-rate / variance-share / goal-reset-rate panel). For supplementary SM panel:
  LLaMA SM fixed ≈ 0.4 %, variable ≈ 72 % — within ±2 pp.
- **S2**: variable-arm `total_rounds_in_game` distribution clearly right-shifted vs fixed
  (KS distance ≥ 0.5 in all models, both IC and SM). Confirms the length asymmetry the
  analysis is trying to control.
- **S3**: row-count parity — number of round-level rows per (model, bet_type) is
  consistent with the per-game `total_rounds` summed over games.
- **S4 (support / separability, codex Round 1 critique 6)**: cell-count table for
  bankruptcy events per `(model × bet_type × balance_quartile × round_block)`. Flag any
  cell with `n_bankrupt == 0` AND `n_total >= 50` (zero events at adequate sample size →
  separation risk). If >2 cells flagged, switch primary fit to Firth-penalised logistic.
- **S5 (overlapping support)**: report fraction of round-level rows where both bet_types
  are observed at matched balance × round bins. The primary RR is computed on the
  overlapping subset only.

### 3.5 Decision tree

```text
[ build round-level tables from existing JSONs (IC + SM) ]
                ▼
[ S1: reproduce per-(model, bet_type, cap) headline marginal bankruptcy ] ── fail → STOP, fix table builder
                ▼ pass
[ S2-S3: length-asymmetry KS, row-count parity ] ── fail → STOP, fix table builder
                ▼ pass
[ S4-S5: support diagnostics + overlap restriction ] ── n_cells_separated > 2 → switch to binary cause-specific Firth on bankruptcy vs not
                ▼
[ fit cause-specific multinomial logit (3 outcomes) with fixed-effect model dummy + cluster-robust SE on (dataset, timestamp, cap, prompt_combo, model, game_id) ]
                ▼
[ readout RR_per_decision + 95 % CI per dataset (IC, SM) ]
                │
        ┌───────┼───────┐
        │       │       │
   point>1.5  point∈(1, 1.5]  lower CI ≤ 1.0
   lower CI>1.2
        │       │       │
   L-passes   L-mixed   L-fails
                ▼
[ stratified robustness: balance quartile + round-block (SM rounds 1-2 subset, IC all rounds since max=10) ]
                ▼
[ Cox time-varying + Fine-Gray competing risk + mixedlm random-intercept sensitivities ]
                ▼
[ report Track L appendix + 3-5 §3 body sentences ]
```

---

## §4. Out of scope

- New LLM queries — Track 0 W3 territory
- Causal interventions — §6 already reports the three null causal protocols
- Cap-ablation cross-model — Track 0 W3
- IC re-analysis is co-primary (v3); SM 6-model panel is co-primary (v3). Both reported.
- Mystery wheel — different game; not part of §3 length-confound concern

---

## §5. Risks and fallbacks

1. **Mixed-effects identifiability with 6 models**. 6 group levels is borderline for a
   random intercept. Fallback: model as fixed-effects categorical indicator + cluster-
   robust SEs by `(model, game_id)`.
2. **Tied bankruptcies at round 1**. Many fixed-mode games end at round 1; ties dominate
   early hazard. Use Efron tie correction in Cox; for discrete-time logistic this is
   automatic.
3. **Voluntary stop competing risk**. Cox standard ignores competing risks. We report a
   sensitivity Fine–Gray subdistribution hazard if `voluntary_stop` rate differs across
   `bet_type` (it does — fixed-mode stops more often after one round). Discrete-time
   logistic is robust to this because it conditions per-round.
4. **Round-level NMI / collinearity**. `balance_before` and `round` are heavily correlated
   in some games. Use VIF check on the design matrix; orthogonalise `balance_before`
   with respect to `round` if VIF > 10.
5. **Header-row misalignment in legacy JSONs**. gpt-4o-mini SM 64-cond corrected archive
   has `round_details` + `game_history` in separate fields (per `5.methods.tex`); the
   builder must align by `round` index, not by list position. S3 row-count parity test
   catches this.
6. **`I_LC` ratio caveat already disclosed**. Track L doesn't re-analyse `I_LC` directly;
   the hazard analysis is on `bankrupt`, which is binary and not ratio-based.

---

## §6. Decision-rule log (frozen before Stage 1 launch)

- Primary readout: lower 95 % CI of RR_per_decision (or β_var on logit scale).
- Pre-registered direction: H_L_main predicts RR_per_decision > 1.
- Surgery prose branches:
  - **L-passes**: §3 retains "freedom-to-choose at root" prose, adds 3–5 sentence
    body acknowledgement of length asymmetry, points to Track L appendix RR table.
  - **L-fails**: §3 narrows to "variable mode extends play, compounding −10 % EV trap";
    "freedom-to-choose at root" replaced.
- Pre-registration: this Plan v1 file is the contract until Round-0 codex review converges.

---

## §7. Open issues for Round 1 critic

1. Discrete-time logistic vs Cox proportional hazards — which is the canonical
   per-decision hazard for round-based gambling data? (Plan v1 picks discrete-time
   logistic as primary; Cox as robustness.)
2. Random effect on 6 models — under-identified per Plan v5.2 §3.3 precedent. Same
   fallback (fixed effects on model) applies here.
3. The matched-cap §3.2 Figure 3d raw data is **not on this machine** (was on the
   original ubuntu host). Track L cannot directly re-analyse Figure 3d; the §3.1
   6-model panel is the closest available cap=\$10 SM panel data. Should the secondary
   IC cap-variation analysis be required, or is §3.1 enough?
4. Voluntary stop as competing risk — Fine–Gray vs naive Kaplan–Meier vs discrete
   logistic. Plan v1 picks discrete logistic as primary; Fine–Gray as a sensitivity
   check.
5. How aggressively should §3 body acknowledgement be — 3 sentences (minimum
   paragraph) or 5–7 sentences (a full subsection paragraph)?
