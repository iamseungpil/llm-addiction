# Response to Reviewer KuK5

Your two weaknesses concern different claims: whether the matched-cap result generalises beyond one model, and what the submitted decoding result establishes relative to causal control. We address them separately and report negative response-period results alongside the supportive ones. We sincerely thank the reviewer for helping us improve the paper.

## [W2, Q1] Whether the equal-cap result rests on one model

**The reviewer is correct that the submitted one-model ablation cannot establish generality.** We therefore repeated the comparison across the panel. One clarification first: the submitted ablation's model is GPT-4o-mini, and the caption's "GPT-4o" is shorthand we will correct in the camera-ready.

| model | prompt | n per arm | fixed % | variable % |
|---|---|---|---|---|
| Gemini | five modules | 50 | 20.0 | **62.0** |
| GPT-4.1-mini | five modules | 50 | 2.0 | **56.0** |
| GPT-4o-mini | five modules | 50 | 0.0 | **40.0** |
| Claude (substitute) | five modules | 50 | 0.0 | 0.0 |
| Gemini | BASE | 50 | 6.0 | **34.0** |
| LLaMA | BASE | 200 | 3.0 | **81.5** |

The rows are model-and-prompt-specific tests, not one common-condition replication: LLaMA uses BASE, the three informative API rows the paper's five modules.

The clearest result is LLaMA: the fixed condition executes the larger realised stake, \$68.4 per played round versus \$32.1, yet reaches far less bankruptcy, so a larger wager range cannot explain the difference on its own. Gemini moves the same way under BASE and under the five modules, so the module text is not the sole explanation either. Newcombe 95% intervals on the variable-minus-fixed risk difference exclude zero for the three largest contrasts: Gemini +42.0 percentage points [+23.0, +57.0], GPT-4.1-mini +54.0 [+37.9, +66.9], LLaMA +78.5 [+71.6, +83.5].

The BASE extension across all six models was largely floor-limited: four of six sit at 0.0% in both conditions. Claude's cells use a newer replacement (the submitted checkpoint is retired) that also stays at the floor, so they speak to the replacement.

Both pre-registered panel-level decision rules were negative, and we report them as such. The primary, a posterior bound on the pooled effect, is ill-posed with the fixed arm at zero; the secondary required four of six models to wager over half the cap, where models given freedom wager 22–46% of it. The table therefore represents separate model-condition estimates, not a registered panel-wide pass.

As a broader descriptive check, all 64 grid cells pass the integrity guard and the variable condition reaches bankruptcy at least as often as fixed in 29 of 32 condition-pairs, ties included. Equal caps still leave stopping and exposure unequal; that battery and the registered follow-up are in our gbSA response.

## [W1] What the submitted readout establishes

**The direct answer is that the submitted readout supports monitoring, not causal control.** The submitted presentation combined two questions, and we now keep them apart.

The first question is predictive: does the representation add information beyond the visible game log? We fitted a 65-covariate baseline (balance, round, drawdown, streaks, cumulative stake, lagged bet ratios) on the published cell's 12,246 decisions, with balance and round already residualised within fold before Ridge. One deviation from the registered baseline specification: choice probability and logits are excluded, since conditioning on the model's own decision would rig the test.

Because the same observable state recurs across games, we evaluated two split rules: by game, and a stricter state-hash rule keeping repeated states on one side of the split. Each entry below is the held-out variance added over the same baseline.

| prediction target | representation added | ΔR², game folds | ΔR², state folds |
|---|---|---|---|
| *submitted deconfounded residual* | sparse-autoencoder features | +0.037 | **+0.045** |
| *raw bet ratio* | raw hidden state | +0.059 | **+0.059** |
| *raw bet ratio* | sparse-autoencoder features | +0.044 | **+0.0024** |

Compare within prediction target — the two targets are different problems. On the submitted metric the readout clears the baseline under both rules; on the raw target the raw hidden state keeps its increment under the stricter split while the sparse representation keeps little. In plain terms, the dense state carries information beyond the game log, and the sparse compression loses much of it. The log alone reaches 0.140 on the submitted target, 84% of the published cell.

The second question is causal: does intervening on the submitted fitted direction alter behaviour? It does not satisfy the criteria below, so it remains a monitoring signal. The intervention result concerns a **separately constructed behavioural axis**, the mean activation difference between high-bet and low-bet decisions, near-orthogonal to the readout. It is not a causal validation of the readout, and the revised paper keeps the two apart.

## [Q2] What a positive causal result looks like, and the state confound

**Purpose.** A readout can predict without being behaviourally operative. If the high-bet-minus-low-bet component contributes to the decision, adding it should raise betting, removing it should lower betting, and matched controls should not reproduce the pattern.

**Positive criteria, fixed before the runs.** (1) Increasing the direction raises the betting index and decreasing it lowers the index, with the interval on that difference excluding zero. (2) The effect changes consistently across the dose ladder rather than appearing only at an extreme. (3) It exceeds a band of norm-matched random directions. (4) Parse success stays above the validity gate, so apparent change is not broken generation.

**Method.** The submitted protocols wrote at one layer; a window scan located the write bands at Gemma layers 16–21 and LLaMA layers 14–19, and we added or removed the frozen direction there during generation. The design is paired: prompt, seed, dose and game state are matched across compared runs, so the activation edit is the only difference. The same procedure ran on twenty norm-matched random directions and on a direction fitted to balance and round. That last control is your confound, steered directly.

The outcome is the bet ratio, the wager divided by current balance; the slope is its change per unit of α; z says how far that slope sits outside the twenty random directions. For Gemma the axis moves monotonically across the whole ladder:

| dose α | −3 | −2 | −1 | 0 | +1 | +2 | +3 |
|---|---|---|---|---|---|---|---|
| mean bet ratio | 0.009 | 0.049 | 0.127 | **0.182** | 0.247 | 0.271 | 0.286 |

The targeted direction and its controls compare as follows; paired removal p-values are Gemma / LLaMA where both exist.

| direction, 200 games/dose | dose slope | z vs 20 matched | removal effect | removal p |
|---|---|---|---|---|
| behavioural axis | 0.0457 | **+4.45** | −0.037 Gemma; −0.052 LLaMA | — |
| same, fitted without an autoencoder | 0.0284 | ≈ +3 | — | — |
| balance/round direction | — | +0.64 | +0.046 Gemma (wrong sign); −0.006 LLaMA | <.001 / 1.0 |
| submitted fitted-readout direction | — | +0.75 | no detectable effect | .885 / 1.0 |

**Result.** The axis meets the dose-consistency and matched-control criteria — monotone across all seven doses, slope outside the random band — and removing it lowers betting in both models. Criterion (1) we count as only partially met: the downward evidence comes mainly from removal, not from an interval contrasting positive and negative doses. That interval, with the dose-wise parse rates for criterion (4), reports by **3 August**, whichever way they fall.

The submitted fitted-readout direction remains null under valid outputs. Under one alternative specification it shows apparent movement, but parse success falls from 0.80 to 0.34, below the 0.45 validity gate, so we do not count that setting as a positive intervention.

On your confound directly: the balance/round direction steers at chance, and its removal moves Gemma the wrong way, so a simple linear state account does not reproduce the pattern. Nonlinear or distributed correlates remain possible, and we say so. One self-limit: the LLaMA readout arm passes its removal criterion by only a 6% margin.

The resulting claim is deliberately narrow: the response-period evidence identifies an *intervention-sensitive behavioural direction*, not complete causal validation of the submitted readout. The readout remains a monitoring signal; the behavioural axis is a separate result about the model's internal state.
