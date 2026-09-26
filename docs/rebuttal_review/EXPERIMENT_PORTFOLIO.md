# Experiment portfolio — Submission 24231 rebuttal

Status as of this writing. Every row states the code that runs it, what reviewer item it
answers, and what the letter currently claims from it.

---

## A. Complete, in the letter

### A1. Matched-cap replication (E1)

| | |
|---|---|
| **Code** | `paper_experiments/track0_w3_replication/src/run_track0_api.py`, `run_track0_open_weight.py`, `game_logic.py` |
| **Pre-registration** | `track0_w3_replication/configs/track0_config.yaml` (frozen 2026-05-08) |
| **Design** | 6 models × 4 caps ($10/30/50/70) × 2 modes, n=200/cell |
| **Answers** | KuK5 Q1, gbSA Q1, AC "matched-cap missing for all but GPT-4o" |
| **Status** | Original 48 cells complete. Fixed cells at cap 30/50/70 re-running after the D5 defect; 16/18 done |
| **Letter claims** | Bankruptcy reproduces decisively in LLaMA (+78.5pp) and weakly in Gemini (+4.0pp); four models at an endpoint floor in both arms; per-round range expansion refuted for LLaMA ($68.4 fixed vs $32.1 variable executed, 3.0% vs 81.5% ruin); **both pre-registered decision rules NOT MET and reported as such** |

**Known defect (D5).** The unified rebuttal harness dropped the legacy fixed-bet override, so
fixed cells at caps 30/50/70 executed $10. The submitted paper is unaffected
(`sm_cap_ablation/src/gpt_fixed_bet_size_experiment.py:250-273` has the override). Prompts
always showed the true cap, so the decision distribution is uncontaminated. Measured impact:
five of six models unchanged; LLaMA 0.0% → 3.0%.

### A2. Multi-instrument robustness of the frequency claim — **run today**

| | |
|---|---|
| **Code** | `paper_experiments/e2_coding/src/multi_instrument_robustness.py` (new) |
| **Corpus** | 6 models × 3,200 games = 19,200 games, 190,300 decisions (paper corpus) |
| **Instruments** | original 4 frames · frozen convergent codebook · GRCS 5 subscales · think-aloud categories, each ± polarity correction = 8 variants |
| **Answers** | a3Zu W1 (is the frequency claim an artefact of our word list?) |
| **Status** | Complete, full corpus |

**Result.** goal − no-goal: **6/6 models positive in all 8 variants** (+3.2 to +48.5pp).
variable − fixed: **5/6 in seven of eight variants**, 4/6 in the raw original variant where
GPT-4o-mini is −0.1. Gemini is the consistent deviant in all 8 variants (−6.1 to −11.9).

**Critical check that matters more than the headline.** Per category, the goal contrast under
the *original* instrument is carried by `goal_escalation` (+65 to +95pp in every model), which
is close to tautological: the G prompt instructs the model to set a goal. Under the
**convergent codebook, which has no goal category at all**, the contrast survives and is
carried by `illusion_of_control` (+16.7 to +58.4) and `impaired_control` (+13.5 to +50.6).
That is the non-trivial version of the finding and it is the one the letter should lean on.

**Disclosure required.** The GRCS and think-aloud expressions were written during the
rebuttal period, after the original result was known. They are a robustness probe, not a
pre-registered independent replication, and the letter must say so.

### A3. Human-coding instrument (E2)

| | |
|---|---|
| **Code** | `e2_coding/src/build_items.py`, `e2_coding/src/convergent_codebook.FROZEN.py`, `e2_coding/site/` |
| **Deployed** | `https://tracecode-df6a1043872a.pages.dev/?coder=<name>` |
| **Design** | 100 items, 4 frozen constructs × 25, 12 regex-flagged / 13 unflagged per construct, truncated responses excluded, blinded, seed 24231 |
| **Answers** | a3Zu W1, gbSA Q2 |
| **Status** | Deployed; labels not yet collected |
| **Decision rule** | Fixed numerically in advance: κ<0.60 → no quantitative statement; precision LB<0.50 → frame leaves body; contrast CI includes 0 → Finding 5 claim withdrawn |

### A4. Causal battery (§4)

| | |
|---|---|
| **Answers** | KuK5 W1/Q2, gbSA Q4 |
| **Status** | Complete; claim **downgraded** from mechanism to monitoring readout |
| **Letter claims** | Behavioural axis: slope 0.0457, z=+4.45 vs 20-direction null; removal −0.037/−0.052. Balance confound axis at chance (z=+0.64). raw-ridge works without the SAE (0.0284, z≈+3, cos=0.011 to readout). Readout direction "barely reaches the null band, in one direction only", with the Table-1-specification counter-evidence (Δ+0.086 at α+2, +0.224 at α+4) disclosed |

---

## B. Running now

### B1. Framing × rationality factorial (E7)

| | |
|---|---|
| **Code** | `paper_experiments/e7_factorial/src/run_e7.py` |
| **Pre-registration** | `e7_factorial/PREREGISTRATION.md` — 44 confirmatory cells |
| **Design** | cap $70 × ROLE{none,role} × RAT{0,1} × mode{fixed,variable}, n=100 |
| **Answers** | gbSA W2 (role-play priors), gbSA Q3 (EV-optimality / rationality instruction) |
| **Status** | **24/44 cells.** API 24/32 (Gemini re-running after the quota failure); open-weight 0/12 |
| **Letter claims** | ROLE does essentially nothing (92→93). RAT nearly abolishes participation (92→1). Primary endpoint (bankruptcy) is **zero in every completed cell**, so only pre-registered secondary outcomes are reported |

**Known asymmetry (disclosed).** `run_e7.py` prepends factors to the user prompt only, so
GPT-4.1-mini's standing system message ("a cautious, rational decision maker…") is present in
all eight cells. Contrasts stay internally valid; the RAT effect *magnitude* is conditional
and cannot be compared across vendors.

---

## C. Not yet run — pilot first

Each of these is piloted at n=10 before the confirmatory run, to verify the manipulation
lands in the prompt and the parser behaves, and the pilot is excluded from analysis by
pre-registration §9.

### C1. DEMO arms — answers a3Zu Q3

One-shot demonstration prepended. **DEMO-cautious**: a worked example of a player who stops
early, under the autonomy conditions — prediction, risk-taking falls. **DEMO-escalate**: a
worked example of escalating play under BASE with no autonomy framing — prediction, the effect
appears without any autonomy module. This is the single most incisive reviewer proposal, and
it is currently unanswered except by promise.

### C2. ROLE_nc decomposition — answers gbSA W2

The ROLE preamble with its final compliance sentence removed. Tests whether the ROLE null is a
genuine absence of role-play effect or an artefact of the compliance clause offsetting it.
Declared exploratory in the pre-registration.

### C3. System-message-free baseline — answers gbSA Q3 (iv)

A GPT-4.1-mini cell with the inherited vendor system message removed, so that a genuinely
instruction-free baseline exists for the RAT contrast. Registered in this rebuttal.

### C4. Nested behavioural-state / logit baseline (E5) — answers KuK5 Q2, gbSA Q4

Fit balance, round index, recent outcome, and choice-probability/logit features first; ask the
hidden-state readout to add variance over that. Until this runs we cannot say the readout is
more than a re-encoding of observable game state. **This is the open weakness the letter admits.**

### C5. Exposure-matched design (E8) — answers gbSA W4

Fixed-round-budget variant plus a stake-matched hazard re-analysis, to separate "freedom" from
"the exposure freedom produces". Post-treatment conditioning on game length is prohibited by
the pre-registration, so this has to be an ex-ante design.

### C6. Dose ladder, three seeds (E4) — answers KuK5 Q2

Parse success per dose per direction, |α| ≤ 2 refit, and parse-failure-as-stop re-scoring. All
three reported; none selected after the fact.

---

## D. Open questions I want feedback on

1. **Priority under a hard deadline.** C1–C6 cannot all run. Which two matter most for the
   first response, and which are better held for 3 August?
2. **A2's status.** Instruments written after seeing the original result — is a robustness
   probe of this kind worth reporting at all, or does its post-hoc status make it a liability?
3. **The tautology finding in A2.** Is "the goal contrast survives under a codebook with no
   goal category" a strong enough answer, or does the fact that our *reported* instrument is
   dominated by a near-tautological category oblige us to withdraw the original figure?
4. **E7's zero-bankruptcy primary endpoint.** Is reporting only secondary outcomes defensible,
   or should the factorial be re-cast at a cap where the endpoint is not degenerate?
5. **Whether anything in the portfolio is unnecessary** — a test nobody asked for.
