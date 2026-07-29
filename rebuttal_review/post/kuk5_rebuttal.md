# Response to Reviewer KuK5

Your two weaknesses ask different questions: whether the equal-cap result generalises beyond GPT-4o, and what the submitted decoding result establishes. We answer them separately, and we report the negative response-period results alongside the positive ones. We are grateful for a review that has made the paper better.

## [W2, Q1] Whether the equal-cap result rests on one model

**Three further models show the dissociation — Gemini, GPT-4.1-mini and LLaMA — though not uniformly across models and prompts.** One clarification first: the submitted ablation's model is GPT-4o-mini, and the caption's "GPT-4o" is shorthand we will correct. We repeated the comparison with the same maximum wager in both conditions:

| model | prompt | n per arm | fixed % | variable % |
|---|---|---|---|---|
| Gemini | five modules | 50 | 20.0 | **62.0** |
| GPT-4.1-mini | five modules | 50 | 2.0 | **56.0** |
| GPT-4o-mini (the ablation's model) | five modules | 50 | 0.0 | **40.0** |
| Gemini | base prompt | 50 | 6.0 | **34.0** |
| LLaMA | base prompt | 200 | 3.0 | **81.5** |

The strongest cell is LLaMA. Its fixed condition executes the larger stake, \$68.4 per played round against \$32.1, yet reaches far less bankruptcy. A wider wager range alone therefore cannot explain the contrast. Gemini moves the same way with and without the five modules, so the module text is not the cause. On precision, Newcombe 95% intervals on the variable-minus-fixed difference exclude zero in all three largest contrasts: Gemini +42.0 [+23.0, +57.0] points, GPT-4.1-mini +54.0 [+37.9, +66.9], LLaMA +78.5 [+71.6, +83.5].

What did not work, briefly. Our base-prompt run across all six models was floor-limited: four of six sit at 0.0% in both conditions. Both pre-registered panel-level rules came out negative. The primary, a posterior bound on the pooled effect, is ill-posed with the fixed arm at zero; the secondary required four of six models to wager over half the cap, and models given freedom wager 22–46% of it. The table above is therefore separate replications, not a registered panel-wide pass. Claude's cells come from a newer replacement, because the submitted model is retired and no longer answers requests; the replacement stays at that floor. The completed 64-cell grid points the same way: the variable condition reaches bankruptcy at least as often in 29 of 32 condition-pairs. What equal caps still leave open, stopping and exposure, is treated in our gbSA response.

## [W1] What the submitted readout establishes

**The submitted readout supports monitoring, not causal control.** The submitted presentation mixed two questions, and we now keep them apart.

**Does the readout add predictive information beyond the observable game state?** Yes. We built the baseline your question implies: 65 game-log covariates (balance, round, drawdown, streaks, cumulative stake, lagged bet ratios) on the published cell's 12,246 decisions. Because 39.3% of rows repeat an observable state from a *different* game, we evaluated two fold rules: grouping by game, and a stricter grouping that keeps any repeated state on one side of the split. One deviation from the registered baseline specification: choice probability and logits are excluded, since conditioning on the model's own decision would rig the test.

| target | features | ΔR², game folds | ΔR², state folds |
|---|---|---|---|
| paper's deconfounded metric | sparse autoencoder | +0.037 | **+0.045** |
| raw bet ratio | raw hidden state | +0.059 | **+0.059** |
| raw bet ratio | sparse autoencoder | +0.044 | **+0.0024** |

Against a margin of 0.017 fixed beforehand, the published readout clears the baseline under both rules on the paper's own metric, and the raw hidden state clears it on the raw target. The sparse features fail the stricter rule on the raw target; the loss sits in the compression, not the state. The game log alone reaches 0.140, 84% of the published cell, and the revision reports that too.

**Does intervening on the fitted readout direction move behaviour?** No. It fails the criteria set out in Q2 below. The causal result there concerns a different direction, computed from behaviour rather than fitted to predict it; it is not a validation of the submitted readout, and the camera-ready keeps the two apart.

## [Q2] What a positive causal result looks like, and the state confound

**What we accept as positive, fixed before the runs:** (1) pushing the direction up raises betting and pushing it down lowers it, with an interval on that difference excluding zero; (2) the effect grows along the dose ladder rather than appearing only at the extreme; (3) it clears twenty size-matched random directions; (4) parse success does not degrade, so behaviour is not changing because generation is breaking.

**What we ran, in plain terms.** We computed a direction from the difference between the model's internal activations on high-bet and low-bet decisions, froze it, and added or removed it while the model played, at seven strengths: α from −3 to +3, in units of the layer's own activation spread. Prompt, seed and game state are identical across the compared runs, so the activation edit is the only difference. The same procedure ran on twenty random directions of matched size, and on a direction fitted to balance and round. That last control is your confound, steered directly.

**What came back.** Pushing the direction up raises betting and pushing it down lowers it. Gemma's bet ratio (the wager as a share of the current balance) runs 0.009 at α = −3, 0.182 unperturbed, 0.286 at +3, monotone at every step; LLaMA runs 0.156, 0.207, 0.251. The slope, 0.0457 per unit of α, sits z = +4.45 outside the twenty random directions. Removing the direction lowers betting in both models (−0.037 Gemma, −0.052 LLaMA). Criterion (1) we count as only partially met: the interval on the up-versus-down difference has not been computed yet; it reports, together with the dose-wise parse rates for criterion (4), with the completed battery by 3 August.

**On your confound directly.** The direction fitted to balance and round steers at chance (z = +0.64), and removing it moves Gemma the wrong way (+0.046, p < .001). A simple linear state account therefore does not reproduce the pattern, though distributed or nonlinear correlates remain possible. The fitted readout direction also steers at chance (z = +0.75), and its removal changes nothing (p = .885 and 1.0). One self-limit: the LLaMA readout arm passes its removal criterion by only a 6% margin. The camera-ready carries the four criteria and these numbers as they fell.
