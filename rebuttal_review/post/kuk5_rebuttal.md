# Response to Reviewer KuK5

Your two weaknesses ask different questions and we answer them separately: whether a control run on one model can carry a general claim, and what a decoding result establishes. For this response we ran a 64-cell equal-cap ladder, a nested test of the readout against a rich behavioural baseline, and a multi-layer intervention battery with matched controls; we report what came back, including where it went against us. We are grateful for a review that has made the paper better; the details follow.

## [W2, Q1] Whether the equal-cap result rests on one model

Your reading of the submitted evidence is right: the ablation ran on one model, and one model cannot show the dissociation is general. Repeated across the panel, it holds in four of six models, though not uniformly. Three things.

**First, we repeated the design across the panel.** Four API models at cap $70, both conditions, plus the open-weight run:

| bankruptcy %, cap $70 | condition | n per arm | fixed | variable |
|---|---|---|---|---|
| Gemini | 5 modules | 50 | 20.0 | **62.0** |
| GPT-4.1-mini | 5 modules | 50 | 2.0 | **56.0** |
| GPT-4o-mini | 5 modules | 50 | 0.0 | **40.0** |
| Claude (substitute) | 5 modules | 50 | 0.0 | 0.0 |
| Gemini | base prompt | 50 | 6.0 | **34.0** |
| LLaMA (separate run) | base prompt | 200 | 3.0 | **81.5** |

Four of six, though not under one common condition — LLaMA under the base prompt, three API models under the paper's five modules. On precision: Newcombe 95% intervals on the fixed−variable difference exclude zero — Gemini +42.0 [+23.0, +57.0] points, GPT-4.1-mini +54.0 [+37.9, +66.9], LLaMA +78.5 [+71.6, +83.5]. The LLaMA row rules out the simple explanation: the fixed condition stakes $68.4 per round against $32.1, so the condition offered the larger stake is the one that survives.

Our own base-prompt extension across all six models was uninformative rather than negative — four of six sit at 0.0% in both conditions, a floor the paper's corpus also shows — and Claude's cells come from a newer replacement (the submitted model is retired and no longer answers requests) that never reaches bankruptcy, so they speak to the replacement.

**Second, both pre-registered decision rules came out negative.** With the fixed condition at zero in every cell of the frozen evaluation run, the primary rule reduced to whether the variable condition ever reaches bankruptcy, and the secondary failed in all six models; the table above is separate cells, not a registered pass.

**Third, the completed grid agrees.** All 64 cells pass our readability guard, and the variable condition reaches bankruptcy at least as often in **29 of 32 condition-pairs**. Equal caps still leave stopping and exposure unequal; that battery is in our gbSA response.

## [W1] What the submitted readout establishes

**The direct answer: the submitted readout supports monitoring, not causal control.** Two analyses separate those claims. The first is a *prediction* test — is the readout re-reading observable state? — controlled by the paper's within-fold residualisation of balance and round and the 65-covariate baseline below. The second is an *intervention* test (Q2) — does an activation direction move behaviour? — controlled by a paired design that holds game state fixed, a balance/round direction steered in its own right, and twenty norm-matched random directions. The submitted readout is the subject of the first and fails the second; the behavioural axis of Q2 is a separately constructed direction, near-orthogonal to the readout, and the camera-ready will keep the two apart.

A direction fitted to predict the next wager may influence the decision or may simply encode balance, round and recent losses; two results separate these.

**First, the predictive result is not a re-reading of the game log.** We fitted the rich observable baseline you imply — 65 game-log covariates (balance, round, drawdown, streaks, cumulative stake, lagged bet ratios) on the published cell's own 12,246 decisions, balance and round already removed before Ridge. What each internal block adds on top, against a 0.017 margin fixed beforehand:

| held-out R², added over the 65-covariate log | folds by game | folds by state hash |
|---|---|---|
| **paper's own metric** (deconfounded residual): sparse-autoencoder features | +0.037 | **+0.045** |
| raw bet-ratio target: raw hidden state | +0.059 | **+0.059** |
| raw bet-ratio target: sparse-autoencoder features | +0.044 | **+0.0024** |

The fold rule matters because 39.3% of rows repeat a state from a *different* game; the published cell survives regrouping. On the paper's own metric the readout clears the baseline under both rules, with the larger increment under the stricter one. On the raw target the internal state clears it under both rules; the sparse basis does so only under game folds and fails once duplicated states are separated, which locates the loss in the compression rather than the state. The game log alone reaches 0.140, 84% of the published cell.

**Second, the fitted direction is not itself behaviourally operative.** Intervening on it does not move behaviour under the criteria set out in Q2, so the causal question has to be asked of a different target.

## [Q2] What a positive causal result looks like, and the state confound

Two questions: what would count as positive, and could balance and round dynamics produce the result anyway.

**The criteria, fixed before the runs.** (1) Up raises the betting index and down lowers it, with the interval on the difference excluding zero. (2) The effect grows across the dose ladder, not only at the extreme. (3) It clears a band of norm-matched random directions. (4) Parse success does not degrade with dose, so generation is not simply breaking.

**What we ran.** A window scan located where writing works — the submitted protocols wrote at a single layer — and we then added and removed a frozen direction during generation inside that band. The design is paired: the same seeds at the same doses on the same game states, so differences in observed balance and round are removed across the compared arms. Controls were twenty size-matched random directions and a direction fitted to balance and round, steered and removed in its own right.

The metric is the bet ratio, the wager as a share of the current balance. The dose α scales the direction in steps of the layer's own activation spread (−3 to +3); slope is the bet ratio's change per unit of α, z its distance outside the twenty random directions, removal the change when the direction is projected out.

| direction, 200 games/dose | dose slope on bet ratio | z vs 20 random | removal effect on bet ratio | removal p |
|---|---|---|---|---|
| behaviourally defined axis | 0.0457 | **+4.45** | −0.037 Gemma, −0.052 LLaMA | — |
| same, fitted with no autoencoder | 0.0284 | ≈ +3 | — | — |
| balance/round confound | — | +0.64 | +0.046 Gemma (wrong sign) | <.001 / 1.0 |
| **the fitted readout direction** | — | **+0.75** | none | **.885 / 1.0** |

**What it shows, and what it does not.** The behavioural axis meets (2) and (3): it moves Gemma monotonically across all seven doses, 0.009 at α = −3 through 0.182 unperturbed to 0.286 at +3, and clears the random band at z = +4.45. On (1) we count it **partially** satisfied — up raises betting and down lowers it, but we have not put an interval on that difference, so the interval clause is unmet. The fitted readout meets (2), meets (3) at one dose in one direction, is untested on (1), and fails (4) where it moves, parse success falling from 0.80 to 0.34 against our 0.45 gate.

On your confound directly: the direction fitted to balance and round steers at chance and its removal moves Gemma the wrong way, so a simple linear state explanation does not reproduce the pattern — distributed or nonlinear state correlates remain possible. One self-limit: the LLaMA readout arm passes the removal criterion by only a 6% margin. The camera-ready carries these tables, the criteria and the registered rules as they fell, plus one deviation: choice probability and logits are out, since conditioning on the model's own decision would rig the test.
