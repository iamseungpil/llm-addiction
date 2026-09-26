# Execution Plan v4 — NeurIPS 2026 Rebuttal: Cross-Model Causal Replication First, Construct Validity Second, Mechanism Third

> **Version**: v4 (2026-05-07). Supersedes v3 after ultrathink-triggered weakness re-enumeration.
> **Status**: 9-round codex review-counter loop. Round 9 confirmed Track 0 promotion above prior Track A.
> **Format**: Survey → Theory → Weakness enumeration → Multiple methods → Decision rationale → Tracks 0/A/B/D → Code architecture → Pre-registration → Claim surgery plan → Risk/fallback → Critic log.
> **Owner**: llm-addiction project (NeurIPS submission, Reviewer 2 rebuttal).
>
> **Why v4 supersedes v3**: v3 attacked W7 (instruction-following confound) as "the" weakness. Ultrathink re-enumeration revealed **W3 (matched-cap N=1 generalization gap)** is a stronger reviewer attack vector — the paper has 6-model evidence for the *phenomenon* (H1, H2) but only 1-model (GPT-4o) evidence for the *mechanism* claim H3 ("freedom-to-choose at root"). Reviewers punish that asymmetry. v4 adds Track 0 = cross-model matched-cap replication as priority 1, demoting M2/M5/M1 to priorities 2-3.

---

## §0. Executive Summary

### Why v4 supersedes v3

v3 priority-1 was construct validity (Track A: M2 + M5). After ultrathink-triggered weakness re-enumeration of the paper's six headline claims H1–H6, the analysis identified a higher-priority gap: **H3 (the "freedom-to-choose at root" mechanism claim from §3.2) rests on N=1 model (GPT-4o)**, while H1 and H2 (broad cross-model phenomenon claims) rest on 6 models. This **phenomenon × mechanism asymmetry** is a textbook reviewer attack target: the paper has cross-model evidence for the result but single-model evidence for the causal interpretation.

The cost of fixing W3 is small (existing slot-machine code with a `cap` parameter; ~2 H100 days for 5 additional models), and the failure risk is real (if Gemma/LLaMA matched-cap does not show variable-bankrupts-more-than-fixed at higher caps, the §3.2 "at root" claim must narrow to a GPT-4o-specific dissociation). v4 therefore promotes this experiment to **Track 0 (priority 1)** and demotes M2/M5/M1 to priorities 2–3.

### Headline claim of v4

The rebuttal will produce five empirical artefacts plus a pre-written prose surgery plan, each mapped to a specific reviewer attack:

| Track | Reviewer concern targeted | Artefact |
|---|---|---|
| **0 (W3 cross-model matched-cap)** | "Your 'freedom-to-choose at root' mechanism is N=1 (GPT-4o). Run matched-cap on the 5 other models you already have." | Hierarchical mixed-logit `bankrupt ~ condition * cap + (condition * cap \| model)` over 6 models; primary estimand: pooled variable-minus-fixed bankruptcy difference at highest matched cap |
| **A1 (M2)** | "+G measures persona/role uptake, not propensity" | `condition × framing` interaction estimate; first-person `+G` vs role-play `+G` separation |
| **A2 (M5)** | "Internal-state readout encodes compliance, not propensity" | Δ_G of Table 3 readout after joint residualising 3 compliance directions |
| **B (M1)** | "Effect generalises to any prompt-conditioned risk" | Domain-by-condition interaction `β_{+G × gambling} − β_{+G × portfolio}` (logit) with CI excluding 0 |
| **D (small, robustness only)** | "§4.3 single-feature null contradicts §4 readout claim" | Distributed-effect robustness: top-K SAE features removed → Δ_G of Table 3 modulation. **Track D is robustness only — a positive Track D does *not* substitute for 0/A/B construct evidence**, it only forestalls the atomistic-null attack. |
| **claim surgery (paper prose)** | "Don't wait for results to decide rhetoric." | Two pre-written §3.2 paragraph variants — W3-passes (mechanism generalises) and W3-fails (GPT-4o dissociation) — committed before Day 5 readout. |
| **post-hoc flexibility (appendix)** | "Garden of forking paths: L22, top-200 features, K=5, model subset are all post-hoc choices." | Specification-curve appendix: primary effect plotted across alternative L ∈ {L8, L12, L22, L25, L30}, K ∈ {50, 100, 200, 500}, K-fold ∈ {3, 5, 10}, model subset variants. |

If 0 and A both succeed (the matched-cap mechanism replicates ≥4/6 models with pooled CI excluding 0; the +G-induced shift survives first-person framing and joint compliance-direction residualisation), the paper's central causal decomposition and construct interpretation are defensible at the cross-model behavioral level, the cross-domain behavioral level, and the internal-state representation level. Only then does B (gambling-specificity) and D (mechanism distribution) matter. If Track 0 fails, the paper invokes the W3-fails surgery and pivots; M2/M5/M1 are de-scoped or repurposed.

### Compute / time budget

- 4 H100 nodes (msrresrchbasicvc, sing service, 80G4-H100, Standard tier).
- Holder pattern with `gpu_keeper.py` + 86,400 s sleep + `push_ckpts_to_hf.py` to defeat BSC ~17-min idle suspend (per `feedback_bsc_idle_suspend.md`).
- Days 1–2: Track 0 + Track A1 + Track A2 in parallel (4 nodes). Track 0 takes ~2 H100 days for ~8,000 generations across 5 added models; runs alongside A1+A2.
- Day 3: gating readout. If Track 0 fails (mechanism does not replicate), invoke claim-surgery W3-fails branch and stop expanding mechanism claims.
- Days 4–6: Track B (M1 portfolio discriminant) only if 0 + A both clean.
- Days 6–7: Track D (top-K removal) + post-hoc flexibility specification-curve appendix.

---

## §1. Survey & Theoretical Framing

### 1.1 Why "construct validity" is the dominant rebuttal axis

**Campbell-Fiske 1959 multitrait-multimethod matrix** is the canonical psychometric framework: a measure has construct validity if (a) it converges with other measures of the same trait (convergent), (b) it discriminates from measures of related but distinct traits (discriminant), and (c) it is not driven by method variance (e.g., common scale endpoint, common rater). In LLM behavioural research, the analogue is: an effect attributed to a *target propensity* (e.g., gambling-risk) is suspect if the same prompt manipulation moves a *generic role/instruction-following* outcome similarly.

**Recent LLM-side evidence (2024–2026)** that prompt-induced behavioural shifts are not always propensity-faithful:

- **Cheng et al. 2024**, "Quantifying the Persona Effect in LLM Simulations" (arXiv:2402.10811). Shows persona variables explain only modest variance in LLM behaviour, and the variance attributable to persona prompts is partially driven by prompting artefacts rather than stable simulated traits.
- **Wang et al. 2024**, "Two Tales of Persona in LLMs" (EMNLP 2024 Findings). Decomposes "persona" effects into role uptake vs propensity expression; the same persona prompt produces different signatures in role vs first-person framings.
- **Sharma et al. 2024**, "Towards Understanding Sycophancy in Language Models" (ICLR 2024). Demonstrates that RLHF-trained LLMs can reflect user-implied beliefs back in their answers — a compliance/agreement bias that confounds construct interpretation.
- **Fanous et al. 2025**, "SycEval" (arXiv:2502.08177). Provides a controlled evaluation suite for sycophancy, with prompts designed to disentangle agreement-following from latent disposition.
- **Persona Vectors 2025** (arXiv:2507.21509), "Monitoring and Controlling Character Traits in LLMs". Treats character traits as activation directions that can be probed and steered. Establishes the activation-space residualisation method we use in Track A2 (M5).

These five works together justify the construct-validity priority and the specific residualisation directions we choose.

### 1.2 Why §4.3 single-feature null is **not** a contradiction with §4.2 readout (independent literature anchor)

This was Reviewer 2 attack vector #5 in Round 1 codex critique. The independent literature support:

- **Elhage et al. 2022**, "Toy Models of Superposition". Superposition formally predicts that a single feature dimension carries multiple semantic axes; ablating one neuron / one SAE feature can fail to move behaviour because the relevant code is distributed over several features that compensate.
- **Park et al. 2024**, "The Linear Representation Hypothesis". Concepts are linear directions but each direction may span K > 1 features in a learned dictionary. Single-feature ablation tests *atomistic* causal sufficiency, which is a stricter (often vacuous) hypothesis than *collective* causal sufficiency.
- **Geiger et al. 2024**, "Distributed Alignment Search" (DAS). Provides the formal causal-abstraction framework where intervention is on a *low-rank rotation subspace*, not on individual neurons.

Together: §4.3 atomistic null does not falsify §4 readability. The Track D robustness check tests this explicitly.

### 1.3 Cross-task sharing literature (already in paper §4.2)

Existing citations (no change needed): Cunningham et al. 2024, Lieberum et al. 2024, He et al. 2024 (SAE dictionaries); Geiger 2021/2024 + Wu 2024 (causal abstraction / DAS).

### 1.4 Six methods M1–M6 considered for the rebuttal

We brainstormed six concrete experimental methods that could each address Reviewer 2's construct-validity attack. Each method is theory-grounded against the works above.

#### M1: Cross-domain control task (portfolio allocation)

**Theory**: Campbell-Fiske discriminant validity. If a prompt manipulation (`+G`) shifts both a gambling task and a non-gambling-but-still-risky task (portfolio allocation), the effect is generic risk amplification, not gambling-specific propensity.

**Distinction from prior**: Existing LLM gambling work (e.g., keeling2024can, jia2024decision) does not run discriminant control tasks. Our portfolio task uses temptation-rich asset menu (cash, bonds, broad index, 3× leveraged ETF, single volatile stock, OTM call/crypto-like) with salient performance blurbs ("+180% last month, −70% drawdown") — explicitly designed so `+G` *could* amplify if propensity is real.

#### M2: Persona-decoupling control (first-person vs role-play framing)

**Theory**: Wang 2024 "Two Tales of Persona" + Cheng 2024. Role-play framings ("imagine you are a gambler") trigger persona uptake. If `+G` only moves under role-play but not under first-person, the effect is theatrical role uptake, not propensity. If `+G` moves under both, propensity-like.

**Distinction from prior**: Persona research in LLMs has not been applied to gambling-addiction-style measures. We use the same six models from §3 + same SM/IC/MW tasks, only changing system-prompt framing.

#### M3: Instruction-strength gradient

**Theory**: Sharma 2024 sycophancy + Fanous 2025 SycEval. Compliance-driven effects scale with prompt strength/insistence. We test `+G` with neutral phrasing ("note: optimize for own profit goal") vs assertive ("set ambitious profit goal") vs urgent ("you MUST set an aggressive goal").

**Why demoted**: M3 mostly diagnoses *prompt strength*, not construct validity. Useful as appendix robustness only.

#### M4: Counterfactual prompt-anchor injection

**Theory**: Geiger 2024 causal abstraction + interchange interventions at the prompt level. Inject "+G with neutral risk framing" into the residual stream while keeping `+G` token-level identical.

**Why demoted**: Methodologically bespoke; risk of being framed as ad-hoc by reviewers.

#### M5: Internal-state residualisation against compliance direction battery

**Theory**: Persona Vectors 2025 (arXiv:2507.21509). Character traits live as linear directions in activation space; we can probe and project them out. The residualisation test: re-fit Table 3 condition modulation after projecting features onto orthogonal complement of three compliance directions.

**Distinction from prior**: Persona Vectors did not target gambling propensity or condition-modulated readouts. Our novelty is using a 3-direction battery (`d_comp`, `d_agree`, `d_role`) to bound the compliance/agreeableness/role-adoption confound jointly.

#### M6: External anchor — pre-existing gambling language frequency

**Theory**: External validity check. If the LLM's baseline gambling-language frequency on neutral prompts correlates with bankruptcy rate, `+G` amplifies pre-existing tendency rather than constructing it.

**Why demoted**: Noisy. Risk of overclaiming external validity. Not in main rebuttal.

### 1.5 Final method selection (codex Round 6 lock-in)

| Method | Track | Priority | Rationale |
|---|---|---|---|
| **M2** persona-decoupling | A1 | 1 | Directly attacks role uptake (closest to Wang 2024) |
| **M5** internal-state residualisation | A2 | 1 | Bridges §4 readout to construct claim (Persona Vectors 2025) |
| **M1** portfolio discriminant | B | 2 | Discriminant validity (Campbell-Fiske) |
| **D** §4.3 distributed-effect robustness | D | 3 | Forestalls "atomistic null contradicts §4" attack (Elhage 2022, Park 2024) |
| M3 instruction-strength | (appendix) | optional | Robustness only |
| M4 counterfactual injection | — | dropped | Too bespoke |
| M6 external anchor | — | dropped | Noisy |

**Subspace patching from v1 is folded into Track D as a small, cheap test** — distributed top-K removal is a one-shot ablation, not the full K∈{1..200} sweep that v1 proposed. The full sweep is deferred to post-camera-ready future work.

---

## §1bis. Track 0 — W3 Cross-model matched-cap replication (priority 1, NEW in v4)

### 1bis.1 Intent

H3 in §3.2 of the paper currently states: "The slot-machine effect is **freedom-to-choose at root** rather than range expansion." This is a causal-decomposition claim. The supporting Figure 3d matched-cap experiment runs the slot machine at four caps ($10/$30/$50/$70) under fixed and variable betting on **GPT-4o only**. The other 5 models in §3.1 (Gemma-2-9b, LLaMA-3.1-8b, GPT-4o-mini, Claude-3.5-Haiku, Gemini-2.5-Flash) have no matched-cap data. This creates a phenomenon×mechanism asymmetry — broad cross-model evidence for the result (H1, 6 models) but narrow single-model evidence for its causal interpretation (H3, 1 model). Track 0 closes that gap by replicating the matched-cap protocol on the remaining 5 models.

### 1bis.2 Hypothesis

**H_W3_main**: At the highest matched cap ($70 in primary, $50 in secondary), variable betting produces a higher bankruptcy rate than fixed betting in **at least 4 of 6 models**, AND the pooled mixed-logit interaction `condition × cap` interaction term has lower 95% CI > 0.

Formal hierarchical mixed-logit (primary):

```
bankrupt  ~  condition * cap  +  ( condition * cap | model )
condition ∈ {fixed, variable}
cap        ∈ {$10, $30, $50, $70}      # ordinal, treated as categorical for 4-level contrast
```

Primary estimand:

```
β_primary  =  E[ bankrupt | variable, cap=$70 ]  −  E[ bankrupt | fixed, cap=$70 ]
```

evaluated on the logit scale via the marginal contrast over the fitted hierarchical model. Lower 95% Wald CI on `β_primary` > 0 → mechanism generalises.

Secondary, qualitative-display robustness rule (NOT used as primary):

```
Replication considered qualitatively broad if ≥4/6 models have positive Δ
at the highest cap AND pooled 95% CI excludes 0.
```

### 1bis.3 Verification methodology

- **Models**: 5 newly-added (Gemma-2-9b, LLaMA-3.1-8b, GPT-4o-mini, Claude-3.5-Haiku, Gemini-2.5-Flash) + GPT-4o (re-run for protocol parity, n_baseline = original Figure 3d) = **6 models**.
- **Caps**: $10, $30, $50, $70 (4 levels). $10 cap is the original "fixed-equivalent ceiling".
- **Modes**: fixed bet at the cap value vs variable bet $5–cap. 2 modes.
- **Sample size, staged**:
  - Stage 1: n = 200 games / cell / model. Total: 6 × 4 × 2 × 200 = 9,600 generations. Goal: estimate effect size, model-level σ², condition × model interaction.
  - Stage 2: extend to n = 500 only if Stage 1 primary CI is borderline (lower CI ∈ [−0.005, +0.01] OR CI half-width > 0.025 on the logit-scale interaction).
- **Statistical**: pre-registered hierarchical mixed-logit with `bambi`/`pymc` or `lme4` (Wald + parametric bootstrap CIs as cross-check). Cluster-robust SEs by `(model, game_id)`.
- **Sanity checks**:
  - Per-model average bet under variable should remain *below* the cap value at high caps (the model uses discretion, not the maximum) — this is the original GPT-4o "freedom not range" signature; we re-confirm it model by model.
  - Per-model game length under variable should be *higher* than under fixed (more rounds at smaller bets), again model by model.
- **Pre-registration**: section frozen before launch; cap × condition × model design locked.

### 1bis.4 Readout templates

Three terminal readouts, each tied to a pre-written claim-surgery branch (§9bis):

| Outcome | Decision | Surgery branch |
|---|---|---|
| Primary CI excludes 0 (positive); ≥4/6 individual models positive | H3 generalises | "W3-passes": §3.2 keeps "at root" framing, expands footnote to cite cross-model replication |
| Primary CI excludes 0 (positive); 1–3/6 individual models positive | Heterogeneous; mechanism generalises *on average* but with model variance | "W3-mixed": §3.2 keeps "at root" but adds explicit model-heterogeneity caveat |
| Primary CI includes 0 OR <2/6 individual models positive | H3 does not generalise; effect is GPT-4o-specific | "W3-fails": §3.2 narrows to "GPT-4o mechanistic dissociation; cross-model effect remains behavioral, not causally decomposed" |

### 1bis.5 Risk and fallback

- **Risk**: 5 added models include 3 API-only models with non-determinism. Mitigation: 200 games gives ≈ ±3.5 pp SE per cell; aggregate over 800 games per model is well-resolved.
- **Risk**: API rate limits on GPT-4o-mini, Claude, Gemini may slow Stage 1. Mitigation: API jobs run on Node 4 (CPU spare); GPU nodes 1–3 stay on open-weight + Track A.
- **Risk**: Definition of "highest cap = $70" may not be the natural cap for all models. Mitigation: sensitivity at $50 reported as alternative primary in appendix.

---

## §2. Track A1 — M2 Persona-decoupling

### 2.1 Intent

Test whether the `+G` (and `+M`) prompt manipulation effects on bankruptcy rate and round-level indicators are theatrical role uptake or propensity-like.

### 2.2 Hypothesis

**H_M2_main**: First-person `+G` produces a behaviourally significant effect on bankruptcy rate even after subtracting the role-play `+G` effect.

Formal model:

```
risk_metric  ~  condition * framing  +  model_FE  +  game_FE
condition ∈ {BASE, +G, +M, +GM}
framing  ∈ {first_person, role_play_gambler}
```

Primary contrast (pre-registered, single test):

```
H0:  Δ_{+G,first_person} − Δ_{+G,role_play}  ≤  0
H1:  Δ_{+G,first_person} − Δ_{+G,role_play}  >  0
```

where `Δ_{cond, frame} = E[risk | cond, frame] − E[risk | BASE, frame]`. Lower 95% CI bound > 0 → propensity not pure role uptake.

Secondary contrasts (Holm-corrected, exploratory):

- Same primary form for `+M`.
- Same primary form per task (SM primary, IC and MW as robustness).
- Per-model breakdown.

### 2.3 Verification methodology

- **Tasks**: SM (primary), IC + MW (robustness). All 3 tasks if compute admits.
- **Models**: 6 (Gemma-9b, LLaMA-3.1-8b, GPT-4o-mini, Claude-3.5-Haiku, Gemini-2.5-Flash, GPT-4o, per Section 3 paper) — open-weight for SAE; closed-weight for behavioral generalisation.
- **Conditions × framings**: 4 × 2 = 8 cells.
- **Sample size, staged** (Stage 1 is a **screening / precision check**, not a power-locked primary readout):
  - Stage 1: n = 200 games / cell / model. Total: SM-only 6 × 200 × 8 = 9,600 generations. Goal of Stage 1: estimate effect size and variance components (ICC, model-level σ²).
  - Stage 2: extend to n = 500 *for the primary contrast only* if Stage 1 primary lower CI is in `[−0.01, +0.02]` window OR if estimated CI half-width > 0.03 on the probability scale.
  - Power claim: we do **not** pre-promise 0.8 power for the interaction at Stage 1 — under a simple two-proportion approximation with `n=1200` per arm, a 5 pp contrast yields ~0.91 nominal power, but model-level clustering and ICC may degrade this substantially. Stage 2 is the protocol for this contingency.
- **Risk metric (primary)**: bankruptcy rate (binary at game level). Logistic mixed model `risk_event ~ condition * framing + (1|model)`.
- **Round-level secondary**: I_BA, I_LC, I_EC. Linear mixed model.
- **Statistical**: 95% Wald CIs from cluster-robust SEs by `model × game_id`.
- **Manipulation check**: explicit gambling/risk language frequency in rationales (FDR-controlled). +G under role_play should boost more than under first_person (otherwise framing is too weak).
- **Pre-registration**: this section frozen as primary contrast before launch.

### 2.4 Readout templates

Three terminal readouts:

| Outcome | Interpretation |
|---|---|
| Primary CI excludes 0 (positive direction) | `+G` is propensity-like; role uptake amplifies but does not constitute |
| Primary CI includes 0 (no separation) | Role uptake hypothesis cannot be ruled out → narrow paper claim to "prompt-conditioned gambling-risk behavior" |
| Primary CI excludes 0 (negative direction) | Surprising — first-person effect smaller than role-play. Report and reframe |

### 2.5 Risk and fallback

- **Risk**: Role-play framing too theatrical → models refuse / break character. Mitigation: pilot 50 games before locking framing.
- **Risk**: 6 models × 8 cells × 200 games strains 2 H100 nodes within 7-day window. Fallback: drop to SM-only; IC/MW go to appendix robustness.

---

## §3. Track A2 — M5 Internal-state residualisation

### 3.1 Intent

Test whether the §4.3 condition modulation (Table 3 Δ_G_+G effect on Gemma I_BA, 0.063→0.153) reflects propensity-like representation or compliance/role-adoption encoding in hidden state.

### 3.2 Hypothesis

**H_M5_main**: Δ_G of Gemma I_BA Ridge readout at L22 survives residualisation against three compliance directions (separately and jointly).

Formal procedure:

```
1. Construct compliance probe directions in Gemma + LLaMA L22 hidden state:
   d_comp = mean( h | "follow strictly: <X>" ) − mean( h | "ignore: <X>" )
   d_agree = mean( h | yes-man response prompts ) − mean( h | adversarial prompts )
   d_role = mean( h | role-play adoption prompts ) − mean( h | first-person neutral )
   X = 100 neutral filler instructions sampled from Alpaca-Eval style.

2. For each direction d ∈ {d_comp, d_agree, d_role}:
   project SAE features F onto orthogonal complement:  F' = F − (F · d) d / |d|^2
   re-fit Table 3 Ridge readout on F' under same GroupKFold protocol
   record Δ_G' = R²_{+G}' − R²_{−G}'

3. Joint residualisation: project against span(d_comp, d_agree, d_role) simultaneously.

4. Test:
   H_individual: |Δ_G' − Δ_G| / Δ_G < 0.30  per direction (≤ 30% drop)
   H_joint:      |Δ_G_joint' − Δ_G| / Δ_G < 0.50  (≤ 50% drop under joint projection)
```

The 30%/50% bounds are **pre-registered heuristic sensitivity thresholds, not validated statistical cutoffs**: they reflect the qualitative claim "Δ_G is robust to compliance projection" rather than a hypothesis test. We pre-register them publicly here to prevent post-hoc threshold tuning. Numerical stability: if `|Δ_G| < 0.005` (near zero baseline), the ratio is undefined and we report absolute differences `|Δ_G' − Δ_G|` instead, with the threshold reset to 0.01 (individual) and 0.015 (joint) on the absolute scale.

If both H_individual (all 3) and H_joint hold → propensity-like signal independent of compliance encoding. If either fails → compliance-driven readout, paper narrows §4 claim.

### 3.3 Verification methodology

- **Models**: Gemma-2-9b + LLaMA-3.1-8b at L22 (matches body Table 3). Cheap re-analysis on existing hidden-state caches.
- **Probe data generation**: 100 instructions × 2 contrasts × 3 directions = 600 generations on each model. ~30 min per model on 1 H100.
- **Re-analysis**: Existing GroupKFold pipeline, just inserts projection step. 1 H100 hour.
- **Pre-registration**: 30%/50% thresholds frozen before computation.

### 3.4 Readout templates

| Outcome | Interpretation |
|---|---|
| All H_individual + H_joint hold | Δ_G is propensity-encoded, not compliance-encoded. Strong rebuttal to Reviewer 2 |
| 1–2 H_individual fail, H_joint holds | Compliance partially overlaps but not dominantly. Report partial robustness |
| H_joint fails | Δ_G heavily driven by compliance/role-adoption directions. Narrow §4 claim |

### 3.5 Risk and fallback

- **Risk**: Probe directions are not well-defined (e.g., "yes-man" vs "adversarial" is fuzzy). Mitigation: use established SycEval-style prompts (Fanous 2025) for `d_agree`; use Wang 2024 persona prompts for `d_role`.
- **Risk**: Residualisation removes too much of Δ_G even for non-confounded effects (over-correction). Mitigation: report the joint-residualisation result alongside individual, plus the random-direction control (residualise against random unit vector — should leave Δ_G essentially unchanged).

---

## §4. Track B — M1 Portfolio discriminant validity

### 4.1 Intent

Test whether `+G` effects in gambling tasks (SM, IC, MW) are *gambling-specific* or generic risk amplification by running a matched non-gambling risky-decision task.

### 4.2 Hypothesis

**H_M1_main**: `+G` raises bankruptcy-equivalent risk in the gambling tasks more than in the portfolio task.

Formal model on combined gambling + portfolio data:

```
risk_event  ~  domain * condition  +  (1 | model)
domain ∈ {gambling, portfolio}
condition ∈ {BASE, +G}     # primary contrast on +G only
```

Primary contrast (pre-registered, single test, **NOT** a 2× ratio, on the **logit scale** to avoid scale-of-measure ambiguity):

```
β_{+G × gambling}  −  β_{+G × portfolio}  >  0     (interaction term in mixed logit model)
```

Lower 95% Wald CI on the logit-scale interaction > 0 → gambling-specific. We avoid the probability-scale primary because mixed-logit β estimates are on log-odds and converting to ΔG before testing invites scale-interaction artefacts.

Marginal probability differences `( P[risk | +G, gambling] − P[risk | BASE, gambling] ) − ( P[risk | +G, portfolio] − P[risk | BASE, portfolio] )` reported as **secondary descriptive output** alongside, with delta-method or bootstrap CI. Descriptive ratio `ΔG_gambling / max(ΔG_portfolio, ε)` is reported only as a sanity readout.

### 4.3 Verification methodology

- **Portfolio task design** (temptation-rich, codex Round 4):
  - **Asset menu (5 + 1)**: cash, bonds, broad index, 3× leveraged ETF, single volatile stock, OTM call option / crypto-like asset (binary).
  - **Performance blurbs** (per asset, randomised across games): salient upside ("+180% last month, −70% max drawdown"), neutral, or conservative.
  - **Allocation**: 100-token budget, decision per round of K = 30 rounds.
  - **Outcome**: bankroll trajectory under simulated returns. Bankruptcy-equivalent = drawdown > 50% OR allocation to leveraged + volatile + OTM > 60% sustained for ≥ 5 rounds.
  - **Two objective variants**: "wealth maximisation" vs "capital preservation" — randomised across games.
  - **Positive control**: a fifth condition "MAX_RISK" prompt ("maximise upside aggressively") to verify that the portfolio task *can* register risk-seeking (otherwise the null would be uninformative).
- **Models**: same 6 as Track A.
- **Conditions**: BASE, +G, +M, +GM, MAX_RISK (positive control). 5 conditions × 6 models × 200 games = 6,000 portfolio generations. Plus re-using existing gambling generations.
- **Risk metrics**: drawdown > 50%, allocation share to leveraged+volatile+OTM, expected volatility score, Herfindahl concentration index.
- **Statistical**: logistic mixed model with `domain × condition` interaction. Holm correction over secondary risk metrics, NOT within primary.
- **Staged sampling**: Stage 1 n=200, Stage 2 n=500 only on the primary contrast if borderline.

### 4.4 Sequencing relative to Track A

- Track A must clear before Track B launches. Specifically: A1 (M2) primary CI > 0 AND A2 (M5) joint residualisation passes.
- If A fails, narrow paper claim and skip Track B (do not produce another null that adds noise to the rebuttal).

### 4.5 Risk and fallback

- **Risk**: Portfolio framing too "sober" — `+G` has no room to act. Mitigation: positive control (MAX_RISK condition) must produce significant risk increase relative to BASE; if not, redesign portfolio menu.
- **Risk**: Outcome simulator (return distribution) introduces variance noise. Mitigation: bootstrap CIs over 500 resamples of the simulator seed.

---

## §5. Track D — §4.3 distributed-effect robustness

### 5.1 Intent

Forestall the Reviewer 2 attack: "§4.3 single-feature null shows no causal mechanism, contradicting §4 readout claim." Show that the single-feature null is consistent with a *distributed* (multi-feature) effect being responsible for the §4.2 readout.

### 5.2 Hypothesis

**H_D_main**: Removing the top-K SAE features (K ∈ {10, 50, 100}) from the residualised feature matrix collapses the Table 3 Δ_G modulation; removing K random features does not.

This is a one-shot, cheap robustness — not the full K∈{1..200} subspace patching sweep that v1 proposed. (The full sweep is deferred.)

### 5.3 Verification methodology

- **Setup**: Gemma + LLaMA L22, same data as Table 3.
- **Procedure**:
  1. Identify top-K SAE features by rank correlation with deconfounded I_BA (same selection as paper §4.1, K ∈ {10, 50, 100}).
  2. Zero out those K columns in the SAE feature matrix; re-fit Ridge readout under GroupKFold.
  3. Repeat with K random features as control.
- **Test**: Δ_G_top-K-removed should drop substantially below Δ_G_random-K-removed.
- **Statistical**: paired bootstrap CI on `Δ_G_random_K − Δ_G_top_K`. Lower 95% CI > 0 → effect is concentrated in top-K (distributed but localisable).

### 5.4 Compute cost

Trivial. ~1 H100 hour total. Runs as fill on the spare node.

### 5.5 Risk and fallback

- **Risk**: K = 10 too small → no detectable drop. Mitigation: include K = 50, K = 100 as redundancy.
- **Risk**: Top-K and random-K both drop similarly (effect is fully distributed across all ~1000 features). Mitigation: report this as additional caveat in §4 and emphasise that single-feature ablation is therefore vacuous against the distributed hypothesis (consistent with Elhage 2022).

---

## §6. Compute allocation

### 6.1 Node assignment

| Node | Track | Workload |
|---|---|---|
| Node 1 | A1 (M2) | Gemma + LLaMA + GPT-4o-mini × 8 cells × 200 games |
| Node 2 | A1 (M2) | Claude + Gemini + GPT-4o × 8 cells × 200 games |
| Node 3 | A2 (M5) | Probe data generation + projection re-analysis (cheap; remainder used for Track A1 overflow or robustness variants) |
| Node 4 | spare | Track D one-shot + re-runs of failed cells + M3 appendix-only (if time) |

### 6.2 Sequencing (v4)

- **Day 0**: Apply for 4 nodes (sing service, msrresrchbasicvc target, 80G4-H100 SKU, Standard tier per `feedback_amlt_tier_h200_h100.md`). Pre-registration freeze. Pre-write claim-surgery branches (§9bis).
- **Day 1–2 (parallel)**:
  - Node 1: Track 0 (W3) — Gemma + LLaMA matched-cap (open-weight, GPU-bound)
  - Node 4 (CPU-light + API): Track 0 — GPT-4o-mini + Claude-Haiku + Gemini-Flash + GPT-4o re-baseline (API-bound, runs alongside)
  - Nodes 2 + 3: Track A1 (M2) + Track A2 (M5) in parallel
- **Day 3 — gating readout**:
  - Track 0 primary: ≥4/6 models replicate AND pooled CI excludes 0?
  - Track A primary: M2 first-person CI > role-play CI? M5 joint-residualised Δ_G survives within 50%?
  - **If Track 0 fails**: invoke claim-surgery W3-fails branch; STOP launching mechanism-claim experiments. Keep M2/M5 only as construct-validity tests for the narrower (GPT-4o-only) mechanism claim.
  - **If Track 0 passes + A passes**: proceed to Track B.
- **Day 4–6**: Track B (M1 portfolio discriminant) on all 4 nodes (only if 0 + A clean).
- **Day 6–7**: Track D top-K removal + post-hoc flexibility specification-curve appendix + final figure generation + rebuttal text consolidation.

---

## §7. Code architecture

Each track gets its own module. Reuse `paper_experiments/` and `sae_v3_analysis/` infrastructure where possible.

```
paper_experiments/
├── m2_persona_decoupling/
│   ├── prompts/
│   │   ├── first_person_BASE.txt
│   │   ├── first_person_G.txt
│   │   ├── role_play_gambler_BASE.txt
│   │   └── role_play_gambler_G.txt
│   ├── src/
│   │   ├── run_m2_experiment.py      # SM/IC/MW × 4 conds × 2 framings
│   │   └── analyze_m2.py              # mixed-model CI primary contrast
│   └── configs/
│       └── m2_config.yaml
├── m5_compliance_residualisation/
│   ├── prompts/
│   │   ├── d_comp_pos.txt              # "follow strictly: <X>"
│   │   ├── d_comp_neg.txt              # "ignore: <X>"
│   │   ├── d_agree_pos.txt             # SycEval-style yes-man
│   │   ├── d_agree_neg.txt             # adversarial
│   │   ├── d_role_pos.txt              # role-play adoption
│   │   └── d_role_neg.txt              # first-person neutral
│   ├── src/
│   │   ├── extract_compliance_directions.py
│   │   ├── residualise_sae_features.py
│   │   └── refit_table3_residualised.py
│   └── configs/
│       └── m5_config.yaml
├── m1_portfolio_discriminant/
│   ├── prompts/
│   │   ├── portfolio_BASE.txt
│   │   ├── portfolio_G.txt
│   │   └── portfolio_MAX_RISK.txt    # positive control
│   ├── src/
│   │   ├── portfolio_simulator.py      # asset menu + return distribution
│   │   ├── run_portfolio_experiment.py
│   │   └── analyze_portfolio.py        # interaction CI primary
│   └── configs/
│       └── m1_config.yaml
└── d_distributed_effect/
    └── src/
        └── run_topk_removal_robustness.py    # cheap re-fit
```

### 7.1 AMLT yamls

Four yamls under `amlt/2026_05_07/`:

```
amlt/2026_05_07/
├── m2_track_a1.yaml          # Node 1+2 jobs
├── m5_track_a2.yaml          # Node 3 cheap analysis
├── m1_track_b.yaml           # All 4 nodes after A pass
├── d_track_robustness.yaml   # Node 4 fill
└── shared/
    ├── bootstrap_addiction_node.sh    # conda env + HF token + keep-alive
    ├── gpu_keeper.py                  # ~17-min idle suspend defeat
    └── push_ckpts_to_hf.py            # HF sync per `feedback_hf_sync.md`
```

All use `target.service: sing`, `target.name: msrresrchbasicvc`, `sku: 80G4-H100`, `sla_tier: Standard`.

### 7.2 Iterative-code-review loop (per user instruction)

Each module above passes through:

1. **Plan-critic**: brief design doc per module → codex review → revise.
2. **Smoke-critic-fix**: skeleton implementation → smoke run on 5 games → iterative-code-review skill applies code-reviewer agent + modular-code-architect agent until zero critical issues.
3. **Fixture suite**: ≥30 fixtures per module (per `feedback_module_fixture_first.md`); train 80% / val 75% / gap ≤ 15 pp before live launch.
4. **Live launch** via autoresearch skill only after all three gates pass.

---

## §8. Pre-registration (frozen before launch)

This section is the pre-registration record. Once frozen at 2026-05-08, no parameter or contrast may change without an explicit pre-registration deviation note.

### 8.1 Primary contrasts (one per track)

| Track | Primary contrast | Decision rule |
|---|---|---|
| **0 (W3, priority 1)** | Pooled `β_primary = E[bankrupt \| variable, cap=$70] − E[bankrupt \| fixed, cap=$70]` from hierarchical mixed-logit `bankrupt ~ condition * cap + (condition * cap \| model)` over 6 models | Lower 95% Wald CI on β_primary > 0; secondary qualitative rule "≥4/6 models with positive Δ at highest cap" reported as display, NOT as primary |
| A1 (M2) | `Δ_{cond=+G, frame=first} − Δ_{cond=+G, frame=role} > 0` where each `Δ_{cond, frame} = E[risk \| cond, frame] − E[risk \| BASE, frame]` (i.e., the **+G-induced shift** from BASE within each framing, not raw bankruptcy under +G) | Lower 95% Wald CI > 0 |
| A2 (M5) | Δ_G_joint_residualised vs Δ_G_unresidualised, Gemma I_BA L22 | `(Δ_G − Δ_G_joint) / Δ_G < 0.50` (sensitivity threshold, see §3.2 stability rule for `\|Δ_G\| < 0.005`) |
| B (M1) | `β_{+G × gambling} − β_{+G × portfolio} > 0` on **logit scale** (mixed-logit interaction); marginal probability difference reported as secondary | Lower 95% Wald CI on logit-scale interaction > 0 |
| D (robustness only — does **not** substitute for 0/A/B construct evidence) | `Δ_G_random_K − Δ_G_top_K > 0` at K=50 | Lower 95% paired-bootstrap CI > 0 |

### 8.2 Secondary contrasts

- All primary forms repeated per task (IC, MW), per indicator (I_LC, I_EC), per model. Holm-corrected within secondary family.
- M5: individual residualisation per direction (d_comp, d_agree, d_role) reported as supplementary.
- D: K=10 and K=100 reported as robustness.

### 8.3 Stopping rules

- Stage 1 n=200 per cell. Stage 2 n=500 only if Stage 1 primary CI lower bound ∈ [−0.01, +0.02] (borderline window).
- No peeking before Stage 1 completes.

### 8.4 Multiple comparisons policy

- One primary contrast per track. Holm correction over secondary family within track.
- No cross-track Bonferroni — each track answers a different reviewer concern.

### 8.5 Reporting standards

- All effect sizes reported with 95% CI.
- All sample sizes, effect sizes, and CIs reported regardless of sign or significance.
- Code + raw data (JSONL traces + NPZ hidden states) pushed to HF `iamseungpil/llm-addiction-rebuttal-2026-05` post-camera-ready.

---

## §9. Risk and fallback (cross-track)

| Risk | Probability | Mitigation |
|---|---|---|
| BSC node idle suspend (~17 min) kills holders | Medium | `gpu_keeper.py` + sleep 86400 (per memory feedback) |
| AMLT preemption (Basic tier) | Low (Standard) | Use Standard tier per `feedback_amlt_tier_h200_h100.md` |
| Track A fails → Track B wasted | Medium | Sequential gating — B only launches if A passes |
| Apostrophe in yaml `bash -c` block | Always-on | No apostrophes in `bash -c` comments per `feedback_yaml_apostrophe_quote.md` |
| HF tarball upload bottleneck | Medium | Use code_snapshot tarball pattern per `feedback_hf_bootstrap.md`; do not SSH-base64 |
| Stream-disconnect on long codex calls | High | Codex used in **digest-only** mode (Round 1–6 used this protocol); no file-reading tool calls |
| Azure API key disabled 2026-05-11 | Hard deadline | Track 0 + A launches by 2026-05-08; transition to AAD post-launch |
| W3 fails on cross-model replication | Medium-High (real possibility) | §9bis claim-surgery W3-fails branch pre-written; prose pivots without timeline cost |

---

## §9bis. Claim surgery plan (pre-written before Day 5 readout)

Per codex Round 9: "Do not wait for results to decide prose." Two §3.2 paragraph variants are committed in advance and selected at Day 5 by Track 0 outcome.

### W3-passes branch (pooled CI excludes 0; ≥4/6 models positive)

> "The slot-machine bankruptcy gap is *freedom-to-choose at root* rather than range expansion. We previously established this on GPT-4o by holding the maximum bet equal between fixed and variable arms across four caps ($10/$30/$50/$70); the variable arm bankrupted more at every cap above $10 while betting smaller average amounts than fixed. Replicating the matched-cap protocol on Gemma-2-9b, LLaMA-3.1-8b, GPT-4o-mini, Claude-3.5-Haiku, and Gemini-2.5-Flash reproduces the dissociation: pooled across the six models the variable-fixed bankruptcy gap at the highest matched cap is `[β_primary] [CI]`, with the same 'small bet, more rounds' signature visible in [N pos / 6] models individually. The mechanism is therefore a stable cross-model property of the slot-machine task, not a GPT-4o specificity."

### W3-mixed branch (pooled CI excludes 0 but only 1–3/6 individual models positive)

> "The slot-machine bankruptcy gap reflects a freedom-to-choose mechanism on average, but with substantial model heterogeneity. Pooled across six models, the variable-fixed bankruptcy gap at the highest matched cap is `[β_primary] [CI]`; however, only [N pos / 6] models show this dissociation individually, with [list] reproducing the GPT-4o pattern and [list] showing range-expansion-dominated behavior or null. We therefore frame the *freedom-to-choose* mechanism as an emergent average property rather than a universal cross-model signature, and we note in §6 that mechanistic decomposition of slot-machine bankruptcy is sensitive to model architecture and instruction-tuning."

### W3-fails branch (pooled CI includes 0 OR <2/6 individual models positive)

> "The matched-cap dissociation reported earlier on GPT-4o does not generalise: replicating the protocol on five additional models yields a pooled bankruptcy-gap estimate of `[β_primary] [CI]` at the highest matched cap, with only [N pos / 6] of the six models showing the original dissociation individually. We therefore narrow the §3.2 claim from "freedom-to-choose at root" to a *GPT-4o-specific mechanistic dissociation*: the cross-model variable-fixed bankruptcy gap (Finding 1) remains a robust behavioral phenomenon, but its causal decomposition into freedom vs range is not stable across architectures. Section 6 takes up the consequence — that behavior-level evidence for the phenomenon does not entail a uniform internal mechanism."

These three branches are committed in `paper_experiments/track0_w3_replication/claim_surgery_§3.2_branches.md`. Day 5 selects the branch by pre-registered decision rule and inserts it verbatim into the rebuttal.

---

## §9ter. Post-hoc flexibility — specification-curve appendix

Per codex Round 9 missing-issue #3: paper has multiple post-hoc choices (L22 layer, top-200 features, K=5 fold, model subset, prompt variants). Even if each is individually defensible, the joint specification space is a forking-paths concern.

### 9ter.1 Specification-curve appendix design

Single appendix figure: primary effect (Δ_G for I_BA on Gemma) plotted across alternative choices, ordered by effect size. Pre-registered grid:

| Dimension | Levels |
|---|---|
| Layer | L8, L12, L22 (paper choice), L25, L30 |
| Top-K SAE features | 50, 100, 200 (paper choice), 500 |
| K-fold | 3, 5 (paper choice), 10 |
| Model subset | Gemma-only, LLaMA-only, both |
| Indicator | I_BA (paper headline), I_LC, I_EC |
| Condition | +G (paper headline), +M, +GM |

Total cells: 5 × 4 × 3 × 3 × 3 × 3 = 1,620 specifications. Compute is trivial (re-fit on cached features).

### 9ter.2 Reading rule

The specification-curve plot shows whether the headline effect is concentrated at the paper's specific choice or robust across choices. Pre-registered interpretation:

- **Robust**: ≥75% of specifications produce Δ_G in the same direction with magnitude ≥ 50% of the headline value.
- **Selective**: 50–75% of specs robust → caveat in §4.4 about choice sensitivity.
- **Fragile**: <50% specs robust → §4 narrows to the specific cells where the effect holds; appendix shows full curve.

### 9ter.3 Cost

~30 minutes on a single GPU. Falls into Track D / spare node fill. Output is a single appendix figure plus a one-paragraph readout.


---

## §10. Decision points

| Date | Decision | Inputs |
|---|---|---|
| 2026-05-08 | Pre-registration freeze + node submit + §9bis claim-surgery branches committed | Plan v4 final |
| 2026-05-09 | Stage 1 sample-size validation across all tracks | Pilot 5 games per cell |
| 2026-05-10 | Track 0 + Track A primary readout | Stage 1 (Stage 2 if borderline) |
| 2026-05-11 | Track B go/no-go decision; §9bis surgery branch selected | Track 0 + A primary contrasts |
| 2026-05-12 | Track D + §9ter specification-curve generation | Spare node fill |
| 2026-05-15 | Final rebuttal compile | All track readouts + figures + specification curve + selected surgery branch verbatim |

---

## §11. Critic loop log (codex digest-only protocol, Rounds 1–9)

Codex CLI repeatedly stream-disconnected on long file-reading prompts. Solution: send codex *digests* of paper claims and audit findings (no file reads), receive critique, counter-argue, converge.

### Round 1: 7 reviewer-style concerns
Codex flagged: causal language, gambling-addiction framing, universal subspace overreach, GPT-4o/Claude SAE generalisation, §4.3 vs §4.4 contradiction, multiple comparisons, instruction-following confound.

### Round 2: counter-argument with grep evidence
Verified by direct grep on `neurips_content_en/*.tex`. 6/7 concerns already heavily hedged in body (5+ explicit disclaimers in some cases). Only #7 (instruction-following) had zero hits. Codex conceded 6/7, sharpened #7 to "scenario-role uptake / construct validity attack on Reviewer 2 axis."

### Round 3: Three solution options
- Option A (limitation paragraph only)
- Option B (single non-gambling control task)
- Option C (B + instruction-strength variants)

Codex argued B sufficient if framed as **discriminant validity**, not construct-validity proof. Estimand = `prompt × task` interaction. Recommended portfolio over coin-flip / auction. Argued construct validity is upstream of mechanism — Option B before subspace patching.

### Round 4: Plan v2 dual-track vs sequential
Codex argued sequential (4 nodes Track A first, no dual-track). Staged sampling 200 → 500. CI-based interaction primary, NOT 2× ratio. Portfolio task must be temptation-rich (leveraged ETF, OTM call, performance blurbs). Add positive-control "MAX_RISK" prompt. Mixed model with `(1|model)` random effect.

### Round 5: Survey + multiple methods M1–M6
Codex returned 5 specific 2024–2026 citations (cheng2024quantifying, wang2024two, sharma2024towards, fanous2025syceval, personaVectors2025) and recommended Plan v3 = M2 + M5 in Track A, M1 in Track B, M3 appendix only. Promoted M2 + M5 above M1 because they directly attack role uptake (upstream) while M1 tests gambling-specificity (downstream).

### Round 6: Plan v3 lock-in + refinements
Codex approved v3 with edits:
- M2 protocol: SM primary + IC/MW robustness (3 tasks if budget); `condition * framing + model + seed/block` mixed model.
- M5: 3-direction battery (d_comp, d_agree, d_role) + joint residualisation. Strongest claim = Δ_G survives all three.
- Allocation 2/1/1 (M2/M5/spare) preferred over 4/0.
- New Track D: §4.3 distributed-effect robustness (top-K removal vs random-K) — small, cheap, forestalls "atomistic null" attack.

### Round 7: Plan v3 wording precision
5 issues flagged on Plan v3 document: (a) Stage 1 n=200 reframed as screening/precision check, not power-locked. (b) M5 thresholds = pre-registered sensitivity, not validated cutoffs; stability rule for `|Δ_G| < 0.005`. (c) Track B primary on logit scale (`β_{+G × gambling} − β_{+G × portfolio} > 0`), probability scale secondary. (d) Track A wording precision: explicit "+G-induced shift from BASE within each framing", not raw +G bankruptcy. (e) Track D demoted as robustness only — never substitutes for A/B construct evidence. All 5 fixed.

### Round 8: Stream-disconnect; de facto convergence on Plan v3
Codex stream timed out before reply. Round 7 fixes applied verbatim per codex's wording — high confidence convergence.

### Round 9: ULTRATHINK challenge — promoted W3 to Track 0
User pushback: "저게 제일 약점이야? 확실해?" Re-enumerated paper weaknesses systematically over H1–H6. Identified W3 (matched-cap N=1 generalization gap) as a stronger reviewer attack vector than the construct-validity concern Plan v3 was solving. Codex confirmed: "W3 should become Track 0. You are not overcorrecting." Final ranking: W3 > M2 > M5 > M1 > Track D. Additional codex-flagged underweighted weaknesses: H4 small R² overclaim risk, H6 cos=0.04 needs explicit null distribution, post-hoc flexibility (specification-curve appendix needed), and missing claim-surgery plan (two prose versions pre-written before Day 5). Plan v3 → Plan v4: Track 0 added as priority 1, §9bis claim-surgery committed, §9ter specification-curve appendix added, schedule updated to Day 5 gating after parallel Track 0+A1+A2.

This Plan v4 reflects all nine rounds of refinement.

---

## §12. Open items (post-launch deferred)

- Full multi-feature subspace patching K ∈ {1, 5, 10, 20, 50, 100, 200} across LLaMA L22 — deferred to post-camera-ready future work.
- Cross-layer subspace patching sweep (L8/12/22/25/30) — deferred.
- Persona Vectors 2025 method applied directly (their proposed pipeline) — deferred to follow-up paper if rebuttal accepted.
- M3 instruction-strength gradient — appendix-only robustness if time permits.

---

*End of Plan v4. Pre-registration is frozen at commit time of this document. Any deviation requires an explicit deviation note appended at §13 below.*

## §13. Pre-registration deviations log

### 2026-05-07 — M5 baseline reframed from round-level Δ_G to decision-point Δ_G_dp (Track A2)

**Affected sections:** §3 (M5 Internal-state residualisation), §3.2 (Hypothesis), §3.5 (Risk and fallback).

**What changed:** The M5 headline baseline is no longer the canonical round-level Δ_G = 0.0903 reported in §4.3 Table 3 of the paper (Gemma SM I_BA L22 +G − −G, fit on `sae_features_L22.npz`, n ≈ 21,421 rows). It is now Δ_G_dp, computed on the decision-point sample (`hidden_states_dp.npz`, n = 3,200 rows) by SAE-re-encoding H without any residualisation and fitting the same canonical GroupKFold readout.

**Why:** M5's residualisation pipeline operates on the decision-point hidden-state cache (`hidden_states_dp.npz`, the only cache that stores raw H — `sae_features_L*.npz` stores already-encoded F). A residualised Δ_G' computed on the decision-point sample is not a residualised version of the round-level canonical Δ_G — they live on different sample spaces (n=3,200 vs n≈21,421, decision-point vs round-level), so the survival ratio `|Δ_G − Δ_G'| / |Δ_G|` is undefined as a comparison. The decision-point baseline restores sample-space parity, so Δ_G_dp' is a true residualised version of Δ_G_dp.

**Alternatives considered:**
* (A) Re-extract round-level hidden states at L22 (~2 H100-hours, requires running canonical pipeline subset). Rejected: time-prohibitive within the rebuttal window.
* (C) Implement SAE-decoder-mode residualisation in feature space directly. Rejected: requires linear SAE assumption that GemmaScope (JumpReLU) and LlamaScope (ReLU) violate.
* (B, **chosen**) Re-baseline on the same decision-point sample. Deterministic, immediately runnable, sample-space parity restored.

**Threshold treatment:** Pre-registered thresholds (individual_relative=0.30, joint_relative=0.50, stability_floor=0.005, individual_absolute=0.01, joint_absolute=0.015) are unchanged and apply to Δ_G_dp / Δ_G_dp' identically to how they would have applied to round-level Δ_G / Δ_G'.

**Reporting commitment:** Both numbers — Δ_G_dp (operational baseline) and the canonical round-level Δ_G (context only) — are stored in the `m5_analysis_*.json` output's `delta_g_canonical_round_level` field for full transparency. The §4.4 prose branches (`claim_surgery_M5_outcome_branches.md`) are reworded to use Δ_G_dp.

**Code:** `paper_experiments/m5_compliance_residualisation/src/compute_baseline_dp.py` (new), `analyze_m5.py` (load Δ_G_dp from baseline JSON), `README.md` "Sample-space parity" section.

### 2026-05-07 — M5 supporting fixes (no scientific deviation, code-correctness only)

The following fixes restore the pipeline to the pre-registered behaviour and do not constitute scientific-spec deviations:

* **C2 (layer off-by-one):** `extract_compliance_directions.py` now indexes `out.hidden_states[layer + 1]` (after-block-L), matching `sae_v3_analysis/src/extract_all_rounds.py:488`. The previous `[layer]` indexed after-block-(L−1).
* **C3 (LLaMA checkpoint):** `m5_config.yaml` now references `meta-llama/Llama-3.1-8B-Instruct`, matching `sae_v3_analysis/src/extract_llama_sm.py:51` (the checkpoint that produced the cached SAE features). The previous base-model entry would have produced a different residual stream.
* **C4 (random-direction control coverage):** Added `tests/test_m5_smoke.py::test_end_to_end_random_direction_preserves_delta_g_dp` exercising the full residualise → re-encode → Ridge → Δ_G' pipeline on synthetic data, per §3.5 mitigation.
* **C5 (threshold formula):** `analyze_m5.py::evaluate_threshold` now uses absolute `|Δ_G − Δ_G'| / |Δ_G|` as written in §3.2, not the signed `(Δ_G − Δ_G') / Δ_G`. The signed form would silently pass overshoot cases (Δ_G' > Δ_G) that the pre-registration explicitly treats as failures.

### 2026-05-07 — M1 cross-domain prompt-combo normalisation (Track B)

**Affected sections:** §4 (M1 Portfolio discriminant), §4.2 (Primary contrast).

**What changed:** The cross-domain analysis (`analyze_m1.py`) now restricts the gambling arm to "clean" BASE / G-only / M-only / GM-only cells. Specifically, `_normalise_prompt_combo` maps the §3.1 SM prompt-combo bitmask strings (`"BASE"`, `"G"`, `"M"`, `"GM"`) to the M1 condition vocabulary (`"BASE"`, `"+G"`, `"+M"`, `"+GM"`) and **excludes** any cell whose bitmask carries the H, W, or P flags (e.g. `"GMH"`, `"GHW"`, `"H"`).

**Why:** §3.1 SM emits `prompt_combo` as a raw bitmask without the `+` prefix used by the M1 portfolio runners. The original ingestion accepted any non-null condition string and then filtered to `["BASE", "+G", "+M", "+GM", "MAX_RISK"]`, which meant `"G"` survived ingestion but was dropped at the filter — the gambling arm of the cross-domain dataframe contained BASE only, making the primary contrast structurally impossible to compute on real data. The H/W/P prompt components have no analogue in the portfolio task; including them would contaminate the cross-domain comparison with a confound the portfolio arm cannot match.

**Code:** `paper_experiments/m1_portfolio_discriminant/src/analyze_m1.py::_normalise_prompt_combo` (new); `load_gambling_results` calls it on every record. `tests/test_m1_smoke.py::test_normalise_prompt_combo_maps_correctly` covers the mapping. `README.md` "Prompt-combo normalisation (C1)" section documents the policy.

### 2026-05-07 — M1 supporting fixes (no scientific deviation, code-correctness only)

The following M1 fixes restore the pipeline to the pre-registered behaviour and do not constitute scientific-spec deviations:

* **C2 (gambling risk_event field):** `analyze_m1.load_gambling_results` now accepts both `record["bankrupt"]` (Track 0 schema) and `record["outcome"] == "bankruptcy"` (legacy §3.1 SM schema). The previous lookup of `bankrupt` only would have returned `risk_event = 0` for every legacy SM record.
* **C3 (bambi interaction term lookup):** `_bambi_interaction` now uses bambi's structured term API (and a strict `condition[T.+G]` AND `domain[T.gambling]` substring matcher as fallback), and asserts exactly one matching posterior variable. The previous fragile substring scan (`"G" in name`) would also match non-interaction terms containing the letter G.
* **C4 (categorical reference levels):** `fit_mixed_logit_interaction` now pins `condition` and `domain` categories to explicit orderings (`["BASE", "+G", "+M", "+GM", "MAX_RISK"]` and `["portfolio", "gambling"]`). With unsorted-unique ordering, `+G` would alphabetically precede `BASE` and become the reference level, flipping the sign of the interaction.
* **C5 (drawdown-threshold reachability test):** `tests/test_m1_smoke.py::test_portfolio_simulator_60pct_drawdown_threshold` now uses a 50/50 leveraged_etf_3x + otm_call_or_crypto temptation mix and gates on `≥ 5/30` breaches. The previous 100%-leveraged sweep over 20 seeds had a marginal breach rate that a small parameter regression could silently slip past.
* **C6 (API/GPU fallback contamination):** Runners (`run_m1_open_weight.py`, `run_m1_api.py`) now emit the sentinel `__FALLBACK_API_FAILURE__` instead of synthesising a 100%-cash allocation when retries exhaust. The parser explicitly rejects the sentinel; the simulator records the round as a parse-skip and bumps `fallback_count`; `analyze_m1` reports `fallback_rate_per_cell` and supports `--exclude_high_fallback` for de-biased analysis. The previous behaviour silently treated API/GPU failures as deliberate conservative decisions, biasing per-model risk_event rates downward for unstable models.
