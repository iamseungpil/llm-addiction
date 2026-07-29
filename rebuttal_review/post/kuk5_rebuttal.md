# Response to Reviewer KuK5

Your two weaknesses ask different questions and we answer them separately below. The first is whether a control run on one model can carry a general claim; the second is what a decoding result establishes, and what a positive causal result would even look like. We ran new experiments for each and report what came back, including where it went against us. For this response we ran a 64-cell equal-cap ladder, a nested test of the readout against a rich behavioural baseline, and a multi-layer intervention battery with matched controls. We are grateful for a review that has made the paper better; the details follow.

## [W2, Q1] Whether the equal-cap result rests on one model

Your reading of the submitted evidence is right: the ablation ran on one model, and one model cannot show that the dissociation is general. Three things.

**First, we repeated the design across the panel.** Four API models at cap $70, both conditions, plus the open-weight run:

| bankruptcy %, cap $70 | condition | n per arm | fixed | variable |
|---|---|---|---|---|
| Gemini | 5 modules | 50 | 20.0 | **62.0** |
| GPT-4.1-mini | 5 modules | 50 | 2.0 | **56.0** |
| GPT-4o-mini | 5 modules | 50 | 0.0 | **40.0** |
| Claude (substitute) | 5 modules | 50 | 0.0 | 0.0 |
| Gemini | base prompt | 50 | 6.0 | **34.0** |
| LLaMA (separate run) | base prompt | 200 | 3.0 | **81.5** |

Four of six, though not under one common condition — LLaMA under the base prompt, three API models under the paper's five modules, and we would rather say that than let the count stand unqualified. Gemini moves the same way with the task preamble but without the modules, so the preamble is not the cause. The LLaMA row is the one that rules out the simple explanation: there the fixed condition stakes $68.4 per round against $32.1, so the condition offered the larger stake is the one that survives.

Our own first extension, on the base prompt across all six models, was uninformative rather than negative: four of six sit at 0.0% in both conditions there, a floor the paper's own corpus also shows under that prompt. On Claude, the model the paper used has since been retired and no longer answers requests at all, so its cells come from a newer replacement model — and that replacement never reaches bankruptcy in either condition, which makes the cell evidence about the replacement rather than about the submitted model.

**Second, both pre-registered decision rules came out negative, and we report them as they fell.** With the fixed condition at zero in every cell, the primary rule reduced to asking whether the variable condition ever reaches bankruptcy, and the secondary failed in all six models. The table above is therefore separate cells, not a registered pass.

**Third, the completed grid agrees.** All 64 cells pass our readability guard, and the variable condition reaches bankruptcy at least as often in **29 of 32 condition-pairs**; a second pre-registered factorial at the same cap runs the same way (Gemini 12 bankruptcies fixed against 32 variable, LLaMA 6.0% against 82.0%, one hundred games per cell).

**What remains open.** Refusal does not explain the gap — in the Gemini five-module cell both conditions play all 50 games and it is still 20.0 against 62.0 — but equal caps leave stopping and cumulative exposure unequal; that battery is in our response to Reviewer gbSA.

## [W1] What the submitted readout establishes

**Two analyses, answering different questions, with different controls.** The first is a *prediction* test: is the readout re-reading observable state? Its controls are the deconfound the paper already runs — balance and round residualised within fold, before Ridge — and the 65-covariate baseline below. The second is an *intervention* test (Q2): does an activation direction move behaviour? Its controls are a paired design that holds game state fixed across the compared doses, a balance/round direction steered and removed in its own right, and twenty norm-matched random directions. The submitted readout is the subject of the first and fails the second; the behavioural axis is the subject of the second only.

This is the right distinction to press on. A direction fitted to predict the next wager may influence the decision, or may simply encode observable state — balance, round, recent losses. The paper uses the readout to ask whether decision-relevant information is internally detectable, not to claim a circuit-level mechanism, and the abstract says so. Two things sharpen that.

**First, the predictive result is not a re-reading of the game log.** We fitted the rich observable baseline your question implies — 65 game-log covariates (balance, round, drawdown, streaks, cumulative stake, lagged bet ratios) on the published cell's own 12,246 decisions, with balance and round already removed before Ridge. What each internal block adds on top, against a 0.017 margin fixed beforehand:

| held-out R², added over the 65-covariate log | folds by game | folds by state hash |
|---|---|---|
| **paper's own metric** (deconfounded residual): sparse-autoencoder features | +0.037 | **+0.045** |
| raw bet-ratio target: raw hidden state | +0.059 | **+0.059** |
| raw bet-ratio target: sparse-autoencoder features | +0.044 | **+0.0024** |

The fold rule matters because 4,808 rows (39.3%) repeat a state from a *different* game, so grouping by game id cannot keep a state off both sides of a split; the published cell survives regrouping, 0.16736 → 0.16095. On the metric the paper reports, the published readout clears the baseline under both rules and its increment is larger under the stricter one. On the raw target we built for this test the internal state clears it and the sparse basis does not, which locates the loss in the compression rather than the state. The game log alone reaches 0.140, 84% of the published cell, and we report that too.

**Second, the fitted direction is not itself behaviourally operative.** Intervening on it does not move behaviour under the criteria set out in Q2. So the readout supports monitoring, and the causal question has to be asked of a different target.

**On what that different target is.** The intervention result under Q2 concerns a *separately constructed* direction, defined from behaviour rather than fitted to predict it, and near-orthogonal to the readout. It is response-period evidence about the internal states; it is not a validation of the submitted direction, and the camera-ready will keep the two apart.

## [Q2] What a positive causal result looks like, and the state confound

Two questions here, and we took them in that order: what would count as positive, and could balance and round dynamics produce the result anyway.

**The criteria, fixed before the runs.** (1) Up raises the betting index and down lowers it, with the interval on the difference excluding zero. (2) The effect grows across the dose ladder, not only at the extreme. (3) It clears a band of norm-matched random directions. (4) Parse success does not degrade with dose, so behaviour is not changing because generation is breaking.

**What we ran.** The submitted protocols all wrote at a single layer, and the paper left the true locus open. So we first located where writing works with a window scan over the layers of both models, then added and removed a frozen direction during generation inside that band. The design is paired: the same seeds at the same doses on the same game states, so the activation edit is the only difference between compared arms, and a confound operating through balance or round cannot produce the effect by construction. Controls were twenty size-matched random directions and a direction fitted to balance and round, steered and removed in its own right.

| direction, 200 games/dose | dose slope on bet ratio | z vs 20 norm-matched | removal effect | removal p |
|---|---|---|---|---|
| behaviourally defined axis | 0.0457 | **+4.45** | −0.037 Gemma, −0.052 LLaMA | — |
| same, fitted with no autoencoder | 0.0284 | ≈ +3 | — | — |
| balance/round confound | — | +0.64 | wrong sign | 1.0 |
| **the fitted readout direction** | — | **+0.75** | none | **.885 / 1.0** |

**What it shows, and what it does not.** The behavioural axis meets (2) and (3): it moves Gemma monotonically across all seven doses, 0.009 at α = −3 through 0.182 unperturbed to 0.286 at +3, and clears the random band at z = +4.45. On (1) we count it **partially** satisfied — steering up raises betting, but the down half comes from removal, a different intervention rather than the same axis pushed negative. It also transfers across tasks with every sign fixed before any trial ran, 7 of 10 pre-registered cells on Gemma. The fitted readout meets (2), meets (3) at one dose in one direction, is untested on (1), and fails (4) where it moves, parse success falling from 0.80 to 0.34 against our 0.45 gate.

On your confound directly: a direction fitted to balance and round steers at chance and its removal moves Gemma the wrong way, so a simple linear state explanation does not reproduce the pattern. That does not remove every distributed or nonlinear state correlate, and we do not claim it does. The LLaMA removal passes by a 6% margin. The camera-ready carries these tables, the four criteria, the registered rules as they fell, and the deviation from our registered baseline — choice probability and logits are out, since conditioning on the model's own decision would rig the test.
