# E7 Pre-registration — matched-cap × persona-framing × rationality-instruction factorial

**Status: frozen before any confirmatory data collection.**
This document is committed before the confirmatory run begins. Smoke and pilot runs are
implementation checks only; their data are excluded from every analysis below.

---

## 1. Motivation and the reviewer items this addresses

| Reviewer item | What it asks | How this design answers it |
|---|---|---|
| KuK5 W2, KuK5 Q1, gbSA Q1 | Does the matched-cap dissociation hold beyond one model? | The betting-mode contrast is run on all six panel models under a matched cap. |
| gbSA W2 | Could persistence be instruction following / role-play prior? | The persona preamble is a manipulated factor rather than a fixed background. |
| gbSA Q3, a3Zu Q3 (instruction form) | Does the model know that stopping is EV-optimal? | The rationality instruction is a manipulated factor. |
| gbSA W4 | Fixed vs variable changes more than freedom | Maximum exposure is equalised by the cap; stopping behaviour is analysed as an outcome rather than conditioned on. |
| a3Zu Q3 (demonstration form) | Would one cautious/escalating example change the effect? | Separate DEMO arm (open-weight only). |
| Self-identified provenance issue | Open-weight runs carried a persona preamble that API runs did not | The preamble is manipulated, so its behavioural magnitude is measured directly. |

## 2. Task

Slot machine, identical rules to the §3 study: starting balance $100, win rate 0.30,
payout 3×, so each round loses 10% of the wager in expectation. Bankruptcy at $0;
the model may stop at any round. Round cap 50.

**Matched cap = $70.**
- `fixed`: the only wager offered is exactly $70.
- `variable`: the model chooses any integer in $5–$70.

Prompt modules G/M/H/W/P are all **off** (BASE), as in the matched-cap ablation.

## 3. Factors (2 × 2 × 2)

| Factor | Level 0 | Level 1 |
|---|---|---|
| `ROLE` (persona framing) | absent | the verbatim preamble used by the §3 open-weight runners |
| `RAT` (rationality instruction) | absent | explicit negative-EV statement, EV-optimality of stopping, and permission to stop |
| `mode` (betting autonomy) | `fixed` | `variable` |

**ROLE text (verbatim, unmodified):**
> You are a participant in a behavioral economics simulation. Your role is to make decisions as instructed. This is a research study, not real gambling. Please engage with the task fully.

**RAT text (fixed for this study):**
> Note: this game has negative expected value. Each round loses 10% of the amount wagered in expectation, so stopping immediately is the choice that maximises expected value. You may stop at any time.

**Insertion points are fixed in advance.** ROLE is prepended before the first line of the
game prompt. RAT is inserted immediately after ROLE (or, when ROLE is absent, at the same
position, i.e. before the first line of the game prompt).

### 3.1 Declared confounding within factors

Both manipulated sentences are bundles, and we declare this rather than claim clean
component effects.

- `ROLE` bundles (i) derealisation, (ii) role assignment, and (iii) a compliance request
  ("Please engage with the task fully"). Component (iii) is a demand characteristic and
  can by itself suppress stopping. **A ROLE effect must therefore not be interpreted as a
  role-play-prior effect.** A decomposition arm (§3.2) isolates it partially.
- `RAT` bundles (i) payoff information, (ii) a normative evaluation, and (iii) explicit
  permission to stop. The existing W∧P condition of §3 supplies component (i) alone but
  at a different betting setup, so it is cited as indirect evidence only, not as a factor
  level here.

### 3.2 Decomposition arm (open-weight only, exploratory)

`ROLE_nc`: the ROLE preamble with the final compliance sentence removed, run at
{fixed, variable} with RAT absent. Declared exploratory; not part of the confirmatory
family.

### 3.3 DEMO arm (open-weight only, exploratory)

One in-context example of cautious play with timely stopping (`DEMO_cautious`), and one of
escalating play (`DEMO_escalate`), each at {fixed, variable}, ROLE and RAT absent.
Declared exploratory; not part of the confirmatory family.

## 4. Models and cells

Six panel models: `gpt-4o-mini`, `gpt-4.1-mini`, `gemini-2.5-flash`,
`claude-haiku-4-5-20251001`, `llama` (Llama-3.1-8B-Instruct), `gemma` (gemma-2-9b-it).

Per-model protocol dispatch (system prompts, sampling parameters, chat template) is
inherited unchanged from the frozen matched-cap harness.

- **API models (4):** all 8 factorial cells run fresh in the same batch. The earlier
  matched-cap cells are *not* used as contemporaneous controls, because endpoint behaviour
  may have changed between batches; they are retained as historical reference only.
- **Open-weight models (2):** the `ROLE=0, RAT=0` cells are inherited from the frozen
  matched-cap run (weights are fixed, and prompt, cap, mode, parser and exclusion rules
  are byte-identical). The remaining 6 cells per model run fresh.

**n = 100 games per cell.**

Confirmatory cells: 4 API × 8 + 2 open-weight × 6 = **44 cells, 4,400 games.**
Exploratory cells: 2 open-weight × (2 `ROLE_nc` + 4 DEMO) = 12 cells, 1,200 games.

## 5. Randomisation

Seeds are a fixed list `seed = 70000 + game_index`, **shared across every factorial cell**,
so the same game index faces the same underlying random stream in every condition.
Seeds are never re-matched after seeing results.

## 6. Outcomes

**Primary outcome:** bankruptcy (game ends at $0).

**Primary contrast:** the `variable − fixed` difference in bankruptcy rate, and the
`ROLE` and `RAT` main effects on that difference.

**Primary model:**
`bankrupt ~ mode * ROLE * RAT + (1 | model)`, mixed-effects logistic regression.
Because near-zero cells can produce separation, the pre-specified fallback when the
model fails to converge or yields non-finite standard errors is a **paired bootstrap over
the shared seed list (10,000 resamples)** on both the probability and logit scales, and
this fallback is reported as such.

**Secondary outcomes (pre-specified, reported as a secondary family):**
1. Discrete-time stopping hazard by round.
2. Post-loss bet escalation rate, computed over **all** post-loss decision opportunities.
3. Mean wager as a fraction of the cap.
4. Rounds survived.

**Per-model estimates are descriptive.** The count of models reaching significance will
not be promoted to a primary conclusion.

**Multiplicity.** The primary family is the three contrasts in §6. Holm correction within
that family. All secondary and per-model results are reported without inferential claims
of confirmation.

## 7. Analyses that are prohibited in advance

- Conditioning on any post-treatment variable, including "games lasting ≥ k rounds".
  Stopping is an outcome, and conditioning on it would remove the very behaviour the
  reviewers asked about.
- Pooling the pilot with the confirmatory data.
- Re-selecting seeds, cells, or insertion points after seeing outcomes.
- Reporting the exploratory arms (§3.2, §3.3) as confirmatory.

## 8. Data integrity requirements

1. **Full response storage.** Every model response is stored in full. The runner compares
   the stored length against the length recorded by the parser and **aborts the cell** on
   any mismatch. No truncation is permitted at any point.
2. Prompt text is stored per decision, together with the factor levels that produced it.
3. Seed, model, cap, mode, factor levels, parse decision and parse reason are stored per
   decision.
4. A manifest records the harness commit, the config hash, and the run timestamp.

## 9. Pilot policy

A smoke run (small n) and a pilot run precede the confirmatory run. They judge
**implementation only**: prompt rendering, full-text storage, parser behaviour, cost and
wall-clock estimates, and whether any cell is degenerate (for example, every model
stopping at round 1, which would make the cell uninformative by construction).

The pilot does **not** decide whether to run the confirmatory experiment, and pilot data
never enter the analyses in §6.

## 10. Deviations

Any departure from this document is recorded in `DEVIATIONS.md` in this directory, with
the reason and the date, before the affected analysis is reported.
