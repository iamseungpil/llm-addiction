# Response to Reviewer KuK5

Thank you for pressing on the two places the paper is weakest. Below we separate what the submitted readout establishes from what a new intervention establishes, and repeat the matched-cap test on more models and caps.

## [W1, Q2] What the internal evidence shows, and what a positive causal result would look like

> "leaving the internal evidence as correlation that the authors themselves can only frame as a monitoring signal rather than a cause of the behavior."

> "what would a positive result on those protocols have looked like, and does their failure leave open that the readout tracks a correlate of balance/round dynamics [...]?"

Your reading is correct and we adopt it: the fitted readout is a monitoring signal, not a demonstrated cause, and its effect is modest partly because the submitted analysis reads one sparse-autoencoder feature set from a single cell. We ran two further analyses: whether that signal carries anything beyond the visible game, and whether a separately built direction changes behaviour under intervention.

The predictive analysis uses the published readout cell, not the whole corpus: the behavioural corpus holds roughly 190,300 decisions, while hidden activations were stored for one model, prompt family and betting condition, giving 12,246 decisions in Gemma's slot-machine variable arm at layer 22. Those are different scopes, not a conflict. On those decisions we predict first from 65 observable covariates (balance, round, drawdown, streaks, cumulative stake, lagged bet ratios), then add the internal representation and measure ΔR², the extra held-out variance explained. Because identical states recur across games we use game-grouped folds and a stricter state-hash rule keeping any repeated state on one side of the split; choice probability and logits are excluded as post-decision quantities that would leak the target.

| prediction target | representation added | game folds ΔR² | state-hash folds ΔR² |
|---|---|---|---|
| submitted deconfounded target | sparse-autoencoder features | +0.037 | **+0.045** |
| raw bet-ratio target | raw hidden state | +0.059 | **+0.059** |
| raw bet-ratio target | sparse-autoencoder features | +0.044 | **+0.0024** |

Observable state explains much of the prediction, but the dense hidden state keeps its increment under the stricter split, so the readout monitors something beyond the visible log; sparse compression loses most of it.

The intervention answers your first question. A positive causal result has to show four things at once: adding the direction raises betting, removing it lowers betting, size-matched control directions fail to reproduce the pattern, and outputs stay parseable.

Every arm below edits the raw residual stream across a six-layer band, Gemma-2-9B at layers 16 to 21 and LLaMA-3.1-8B at 14 to 19, while the model replays decision states taken from the paper's own corpus under variable betting, none of them carrying the goal module, at 200 games per arm. LLaMA's band was located by scanning four candidate windows and then re-run from scratch at the winner, so the ladder below is not the run that chose the window. One unit of dose adds 3% of that layer's median residual-stream norm along a unit-length direction, and the ladder runs from α = −3 to +3. Doses share seeds and therefore replay identical game states, so a confound running through balance or round index cannot produce the difference, and the axis was built on games disjoint from those replayed here. The behavioural axis is the difference in mean activation between the top and bottom quarter of decisions ranked by betting once balance and round are residualised out; no probe is fitted to it.

| analysis | the direction, and what it tests | dose response | removing the component |
|---|---|---|---|
| behavioural axis | high-bet minus low-bet activation: does editing a behaviourally defined direction change the choice | Gemma bet ratio 0.014 at −3, 0.061 unedited, 0.256 at +3: slope 0.0457 [0.0426, 0.0486], paired +3 against −3 +0.244 [+0.224, +0.263], above all twenty size-matched random directions, the largest of which reaches 0.0141. LLaMA 0.162, 0.213, 0.274, monotone across the seven doses, paired +0.092 [+0.049, +0.136], above all five random directions run there | Gemma 0.065 → 0.028, −0.037 [−0.052, −0.022]; LLaMA 0.206 → 0.154, −0.052 [−0.081, −0.026] |
| specificity controls | the same contrast rebuilt in the raw residual stream with no autoencoder, and a direction fitted to predict balance and round | raw-stream axis 0.034 at −2 to 0.206 at +3, slope 0.0332, z = +3.21 against the same twenty-direction band. The balance/round direction is inert on Gemma, slope 0.0072 and z = +0.64 inside that band, but not on LLaMA, paired +0.103 [+0.073, +0.134] | balance/round removal *raises* Gemma's betting, +0.046 [+0.029, +0.062], and leaves LLaMA unchanged, −0.006 [−0.040, +0.028] |

Intervals are 95% bootstrap intervals on seed-matched pairs. Parse validity holds between 0.96 and 1.00 across the seven doses of Gemma's behavioural ladder and between 0.77 and 0.96 on LLaMA's; the raw-stream ladder's α = −3 cell falls to 0.57 and is excluded by the 0.80 validity threshold registered for this task.

The submitted fitted readout went through the same protocol and did not move behaviour: steering leaves Gemma inside the random band, z = +0.75, and moves LLaMA by −0.028 [−0.062, +0.005] across the same range, while removing it changes nothing in either model, −0.003 [−0.015, +0.010] and −0.019 [−0.051, +0.013]. One alternative specification appears to move behaviour, but parse success falls from 0.80 to 0.34 there, below even the loosest validity threshold used anywhere in the battery, so we do not count it.

The axis moves betting in both directions while size-matched randoms do not, so it is behaviourally operative rather than merely predictive in this setting, and it does not depend on the sparse basis. Separating it from a balance/round account takes both halves of the design, and only on Gemma does steering alone do that. On LLaMA the balance/round direction steers as well, and what separates the two there is removal: taking out the behavioural axis lowers betting, taking out the balance/round direction does nothing. What we cannot exclude is a nonlinear or distributed state correlate that neither the fitted direction nor the 65 covariates capture. So the internal evidence stands on two legs, a readout that monitors and an axis that writes. We do not claim the second validates the first, and the revision keeps them apart.

## [W2, Q1] Whether the matched-cap dissociation holds beyond one model

> "The strongest behavioral claim [...] rests on a matched-cap ablation run on a single model (GPT-4o) [...]"

> "Does the matched-cap dissociation [...] hold on any of the other five models, or only GPT-4o?"

Agreed, and we repeated the controlled test rather than leaning on the broader pattern. The submitted ablation ran on GPT-4o-mini; the paper's "GPT-4o" label is shorthand the camera-ready corrects throughout. In the new runs, choosing to play in the fixed condition commits the model to the cap-sized wager, while the variable condition picks any wager up to the same cap each round. Bankruptcy in % of games, fixed → variable, with the 95% interval on the difference, under the five-module prompt at 50 games per arm:

| model | cap \$10 | cap \$30 | cap \$50 | cap \$70 |
|---|---|---|---|---|
| GPT-4o-mini | 0 → 2 [−5.3, +10.5] | 0 → 20 [+8.7, +33.0] | 4 → 26 [+8.1, +35.9] | 0 → 40 [+25.7, +53.8] |
| GPT-4.1-mini | 0 → 12 [+2.4, +23.8] | 2 → 40 [+23.0, +51.9] | 30 → 56 [+6.6, +42.8] | 2 → 56 [+37.9, +66.9] |
| Gemini-2.5-Flash | 8 → 16 [−5.3, +21.4] | 12 → 66 [+35.8, +67.2] | 66 → 84 [+1.0, +33.8] | 20 → 62 [+23.0, +57.0] |

Nine of the twelve differences exclude zero and none is negative. At cap \$70 Gemini plays all 50 games in both arms, so refusal cannot explain its gap. The submitted Claude-3.5-Haiku checkpoint has been retired by the provider, so that row cannot be re-run; its replacement sits at 0% in both arms and is uninformative. At cap \$70 we also ran both open-weight models at 100 games per arm: LLaMA-3.1-8B 6.0% → 82.0% (+76.0 pp [+65.2, +83.1]) and Gemma-2-9B 1.0% → 15.0% (+14.0 pp [+6.8, +22.3]). LLaMA is the cleanest case for your dissociation: its fixed arm wagers more per played round, \$66.5 against \$34.1, and still ruins far less.

These cells look unlike the submitted figure's 14–17% because that ablation averaged over prompt conditions while these are per-condition cells; its fixed arms, near 0/5/1%, match ours. Across the grid the variable arm ruins at least as often as fixed in 29 of 32 pairs. The three exceptions are informative rather than contradictory: two are Gemini under the plain prompt, where the reversal is small and its interval spans zero (34% against 26% at cap \$50, 2% against 0% at cap \$10), and the third is the retired Claude checkpoint's replacement. Both prespecified panel rules were negative, so we claim the same controlled dissociation in five models and four caps, not a registered panel-wide effect.

One caveat on the fixed arms: part of what they buy is stopping early rather than betting better. At cap \$70 LLaMA re-bets after a first loss in 26% of fixed games against 100% of variable games, which is why the camera-ready renames the condition forced-maximum and reports realised stake and re-betting beside every bankruptcy number. The remaining freedom channel, a one-time stake choice against per-round discretion and against a widened bound, was run for this response and is reported in our gbSA reply.
