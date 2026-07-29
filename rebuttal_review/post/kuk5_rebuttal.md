# Response to Reviewer KuK5

We thank the reviewer for reading our paper so closely, and for the chance to improve it. Both weaknesses named real gaps, and both sent us back to the bench. We ran a new experiment for each and report what came back, whichever way it fell.

- **W1 — can the neural decoding support a mechanistic reading?** No, and the submitted abstract already says so. What holds is the readout as a *predictive* signal: it clears a rich behavioural baseline and a stricter fold rule. What we add is a causal result the paper did not have, and the null there is direction-specific.
- **W2 — does the equal-cap claim rest on one model?** It did. Both arms now run at the same cap across the panel, under two prompt conditions and in a second harness; under the paper's own condition set the dissociation appears in four of six.
- **Q1 — does it hold on the other five?** Four of six. Gemma is 0.0% in both arms at all four caps, Claude 0.0 in both panel arms — nulls, not absences.
- **Q2 — a positive result, and a balance/round correlate?** Four criteria below, applied to both directions. We steered the balance/round direction itself and ran the nested test your question implies: the hidden state clears it, the sparse basis does not.

## [W2, Q1] The equal-cap test across the panel

**You were right that it rested on one model.** The published ablation crosses each cap with 32 prompt conditions × 50 repetitions (n = 1,300–1,600): forced 0.0/4.7/0.4% against choosing 14.3/16.4/17.3% at $30/$50/$70, matching the Figure 3d caption. Our first extension used the base prompt only (six models × four caps × both arms, n = 200, 48 of 48 cells) and was uninformative — Claude's cells there are re-collected for the panel below, and of the other five, three never ruin in either arm. That is the condition, not drift: the paper's corpus restricted to BASE has the same floor.

LLaMA is decisive there, in the last row below. The forced arm stakes $68.4 per round, the choosing arm $32.1, and the arm offered the larger stake survives — which a larger per-round stake cannot produce. Exposure is the other half: 0.92 rounds per game against 15.17.

**The paper's five modules recover it on three more.** Separate cells: four API models, cap $70, n = 50, both arms carrying the same task preamble, which the published ablation did not, so absolute rates are not comparable across the two grids.

| bankruptcy % | forced | choosing |
|---|---|---|
| Gemini, 5 modules | 20.0 | **62.0** |
| GPT-4.1-mini | 2.0 | **56.0** |
| GPT-4o-mini | 0.0 | **40.0** |
| Claude (substitute) | 0.0 | 0.0 |
| Gemini, base prompt | 6.0 | **34.0** |
| LLaMA, base, n = 200 | 3.0 | **81.5** |

Gemini moves the same way with the preamble but *without* the five modules, so the preamble is not the cause. Four of six.

**Both pre-registered rules failed.** With the forced arm at 0 in every cell, the primary rule stopped operationalising a panel-level arm effect and collapsed into asking whether the choosing arm ever ruins. The secondary is 0 of 6: given the choice, five of six wager 22–46% of the cap, not the registered half. What we report above is therefore the separate cells, not a registered pass.

**A second pre-registered factorial repeats it at that cap**, n = 100 per cell and all 44 complete. Its two Gemini pairs run 0.0% forced against 4.0% choosing and 12.0% against 32.0%; LLaMA runs 6.0% against 82.0%. Refusal does not explain the gap in the one cell where both arms play every game, and the rest of the confound battery — action space, game length, exposure, stopping — is in our response to Reviewer gbSA, W4.

**The completed ladder agrees.** All 64 cells are now collected and all clear our 95%-readable guard. Across the four API models, four caps and both prompt conditions, the choosing arm ruins at least as often as the forced arm in **29 of 32 arm-pairs**; the three exceptions are one Claude cell and two Gemini base-prompt cells. Claude 3.5 Haiku is end-of-life (404), so the panels substitute Haiku 4.5, a different model.

## [W1, Q2] What the internal-state results show

**You are right that the submitted analysis cannot carry a mechanistic reading, and the abstract says so.** What it can carry is monitoring, and that survives the harder test your Q2 implies. On the published cell's own 12,246 decisions we built the rich observable baseline: 65 game-log covariates (drawdown, streaks, cumulative stake, lagged bet ratios), reaching R-squared 0.590 alone. Balance and round were already removed before Ridge, and the confound direction itself steers at chance. Against a 0.017 margin fixed beforehand:

| added over the 65-covariate log | folds by game | folds by state hash | resamples clearing |
|---|---|---|---|
| raw hidden state | +0.059 | **+0.059** | 99.5% |
| **SAE features (the paper's)** | +0.044 | **+0.0024** | **29.5%** |

Both rows share one target and differ only in the fold rule, which matters because 4,808 rows (39.3%) duplicate a state from a *different* game, so game-id grouping cannot keep a state off both sides of a split. The published cell survives that regrouping, 0.16736 → 0.16095. **The hidden state clears the bar under either rule; the sparse basis clears it only under the looser one.** On the deconfounded residual — the metric the published 0.167 is in — the log alone reaches 0.140, 84% of the cell, and the SAE adds +0.037 there, which does survive state grouping. So the compression, not the internal state, is where the increment is lost.

**Why the causal result came out weak: where we wrote, not whether the activations matter.** Of the three published protocols one edits the prompt rather than the model; the two that edit activations both act at layer 22 — steering adds the direction *fitted to predict* betting at the last prompt token, n = 50 per dose, and patching replaces the L22 block output at three scopes. The submitted Limitations left the locus open: not consolidated at L22 "in a form that single-layer patching can write into the model", with an "earlier-layer, distributed multi-layer pathway, or SAE-feature subspace target" as the open question. We ran that arm.

**What we ran.** *First*, a width-6 window scan tiling every layer of both models: single layers are insufficient, Gemma localises to L16–21, and on LLaMA four candidate windows were steered head-to-head, L14–19 winning. *Second*, inside that band we steer and remove frozen unit axes on the raw residual stream, at every token position on prefill and every decode step, seeds tied to the trial index so doses are paired by construction. *Third*, the same battery on the balance/round confound direction and on twenty norm-matched random directions, so edits of equal size are compared.

**What came out.** A *behaviourally defined* axis — the difference between activations on high-bet and low-bet decisions, no probe fitted — moves Gemma monotonically across all seven doses.

| Gemma, mean bet ratio | α = −3 | −2 | −1 | 0 | +1 | +2 | +3 |
|---|---|---|---|---|---|---|---|
| behavioural axis | **0.009** | 0.049 | 0.127 | 0.182 | 0.247 | 0.271 | **0.286** |

| direction, 200 games/dose | slope | z vs 20 matched | removal |
|---|---|---|---|
| behavioural axis | 0.0457 | **+4.45** | −0.037 Gemma, −0.052 LLaMA |
| same, no autoencoder | 0.0284 | ≈ +3 | — |
| balance/round confound | — | +0.64 | p = 1.0, wrong sign |
| **readout direction** | — | **+0.75** | **p = .885 / 1.0** |

The slope tracks the prompt condition (+G 0.0469 against +M 0.0218, difference 0.0237, 95% interval [+0.018, +0.030]), and survives with no autoencoder, so it is not an artefact of the sparse basis.

**And it transfers across tasks with every sign fixed beforehand.** The three tasks' behavioural axes align where the endpoint directions did not (pairwise cosine +0.43 to +0.67 against at most 0.04), and the loadings of the direction they most agree on fixed every steering cell's predicted sign *before any trial ran*. On Gemma the shared axis lowers slot-machine betting (z = −3.7) and raises investment-choice risk (z = +3.5): 7 of 10 pre-registered cells, the misses being mystery-wheel cells at that task's ceiling, and the matrix clean only on Gemma. Reading does not cross tasks; writing does. Self-limit: the LLaMA removal passes by a 6% margin.

**The direction you ask about stays null after the repair**, in the last row of that table. Under one alternative specification it does move (+0.086 at α = +2, p < 1e−4), but parse success collapses from 0.80 to 0.34 against our 0.45 gate (n = 50, exploratory), so we do not count it. **The readout therefore stays a monitoring signal, and we claim no identity between it and the behavioural axis** — this is a new result about the internal states, not a rescue of the submitted direction.

**What a positive result would have looked like (Q2).** Four criteria, fixed before the runs: (1) up raises the betting index and down lowers it, interval on the difference excluding zero; (2) the effect grows across the ladder, not only at the extreme dose; (3) it clears a band of matched random directions; (4) parse success does not degrade with dose. The fitted direction meets (2); meets (3) at one dose in one direction; is untested on (1), which we do not count in our favour; and fails (4) where it moves. The behavioural axis meets (2) and (3), removal supplying the down half of (1).

**Camera-ready.** These tables; the four criteria; the failed rules as failed; the registered-baseline deviation, since conditioning on the model's own choice probability would rig the test; and this panel replacing the single-model ablation.

If any part of this falls short, we would be glad to take it further during the discussion period.
