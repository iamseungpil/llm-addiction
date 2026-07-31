# Response to Reviewer KuK5

Thank you for pressing directly on the two weakest points. We agree that the effect of the submitted readout is small and that it cannot on its own establish a cause of behaviour. The submitted analysis used a limited top-200 sparse feature set and a Ridge readout in one model and one experimental condition, and the paper reads it as a monitoring signal rather than as a full mechanism. Following your feedback we ran two separate experiments: whether a direction that actually changes behaviour exists in the activations, and whether the predictive power of the hidden state is no more than a restatement of the observable game state.

## [W1, Q2] The readout, causal intervention and state information

> "leaving the internal evidence as correlation [...] rather than a cause of the behavior."

> "what would a positive result [...] have looked like, and does their failure leave open that the readout tracks a correlate of balance/round dynamics [...]?"

The intervention experiment found that a separate direction able to change the betting choice does exist in the activations. It is built independently of the submitted fitted readout, and the result is clearest in Gemma, while in LLaMA it separates less sharply from a direction related to game state.

We judged a causal intervention by whether adding the direction raises betting, removing the same component lowers it, and size-matched random directions fail to reproduce either. After first removing the influence of balance and round, the behavioural axis is the difference in mean activation between high and low bet-ratio decisions. The games used to build the axis are disjoint from the games used to evaluate it, and the same prompt, seed and game state are replayed at every dose. The band is layers 16 to 21 in Gemma and 14 to 19 in LLaMA, and one unit of dose is 3% of that layer's median residual norm.

| model | α = −3 | α = 0 | α = +3 | dose effect | component removal |
|---|---|---|---|---|---|
| Gemma-2-9B | 0.014 | 0.061 | 0.256 | +0.244 [+0.224, +0.263] | −0.037 [−0.052, −0.022] |
| LLaMA-3.1-8B | 0.162 | 0.213 | 0.274 | +0.092 [+0.049, +0.136] | −0.052 [−0.081, −0.026] |

The values are mean bet ratios, the wager divided by the current balance. α is how strongly the axis is added to the activations: −3 suppresses it, 0 is the unedited run, +3 strengthens it. The dose effect is the difference between α = −3 and α = +3, and removal is the change when the direction's component is projected out. Each bracket is the 95% interval on that difference, over seed-matched pairs.

Betting rose with the dose in both models and fell when the component was removed. Gemma's dose effect was larger than all 20 norm-matched random directions, and a control direction fitted to predict balance and round stayed inside the random band, so a simple linear reading of game state does not reproduce the behavioural-axis result in Gemma.

In LLaMA the steering and the removal went the same way, but the balance-and-round control direction also changed betting under steering. Removing that control direction produced no behavioural change, whereas removing the behavioural axis lowered betting. LLaMA therefore also supplies intervention evidence, but the specificity of the direction is less clear-cut than in Gemma. Building the same high-versus-low betting contrast in the raw residual stream, with no sparse autoencoder involved, also produced a positive dose-response, so the intervention effect does not depend on the sparse basis.

The submitted fitted-readout direction went through the identical steering and removal protocol and produced no behavioural change. We therefore do not use the new behavioural axis as a causal validation of the submitted readout. The submitted direction stays a monitoring signal, and the behavioural axis is reported separately as evidence that a direction able to change behaviour exists in the activations.

We then asked whether the hidden representation merely repeats the observable game state. On the 12,246 decisions of the Gemma-2-9B variable condition for which activations were stored, we first predicted the target from 65 observable covariates including balance, round, drawdown, streaks, cumulative stake and lagged bet ratios, then measured how much held-out R² increased when the internal representation was added. To keep an identical game state out of both sides of the split we used a game-grouped split and a stricter state-hash split.

| prediction target | information added | game split ΔR² | state-hash split ΔR² |
|---|---|---|---|
| submitted deconfounded target | sparse features | +0.037 | +0.045 |
| raw bet ratio | raw hidden state | +0.059 | +0.059 |
| raw bet ratio | sparse features | +0.044 | +0.0024 |

ΔR² is the increase in held-out explained variance when the internal representation is added on top of the observable-variable baseline. The game split keeps decisions from one game on one side; the state-hash split is stricter and also prevents the same observable state from appearing on both sides.

Observable game state explained much of the prediction. The dense hidden state nevertheless kept its increment on the raw bet ratio even under the stricter split, while the increment from the sparse representation fell sharply there. The effect of the submitted readout is therefore limited, but the hidden state as a whole is not simply a restatement of the visible game log.

The two experiments answer different questions: whether the hidden representation holds information beyond the observable game state, and whether changing the activations changes the choice. The submitted readout is not causal, but predictive information beyond the visible log and a separately defined direction that moves behaviour together strengthen the neural analysis. We are grateful for a question that made us separate prediction from intervention.

## [W2, Q1] The matched-cap result in further models

> "The strongest behavioral claim [...] rests on a matched-cap ablation run on a single model [...]"

> "Does the matched-cap dissociation [...] hold on any of the other five models?"

The direction of the matched-cap result is not confined to GPT-4o-mini. The submitted analysis evaluated that model on the slot machine at caps of \$10, \$30, \$50 and \$70, and the paper writes GPT-4o for it, which the camera-ready corrects. Following your point, the new experiment does not add caps but extends the same controlled comparison to other models.

Under the BASE prompt several models sit at zero bankruptcy, which leaves nothing to compare, so during the response period we evaluated the five-module prompt, where bankruptcy events are observable. Three API models ran the same four caps at 50 games per condition. The Claude-3.5-Haiku checkpoint the submitted runs used reached end of life in February 2026 and no longer resolves, so it could not be re-run, and this is how far we were able to take the comparison.

Each cell reads fixed bankruptcy → variable bankruptcy, then the difference with its 95% interval. The difference is variable minus fixed.

| model | cap \$10 | cap \$30 | cap \$50 | cap \$70 |
|---|---|---|---|---|
| GPT-4o-mini | 0 → 2; +2 [−5.3, +10.5] | 0 → 20; +20 [+8.7, +33.0] | 4 → 26; +22 [+8.1, +35.9] | 0 → 40; +40 [+25.7, +53.8] |
| GPT-4.1-mini | 0 → 12; +12 [+2.4, +23.8] | 2 → 40; +38 [+23.0, +51.9] | 30 → 56; +26 [+6.6, +42.8] | 2 → 56; +54 [+37.9, +66.9] |
| Gemini-2.5-Flash | 8 → 16; +8 [−5.3, +21.4] | 12 → 66; +54 [+35.8, +67.2] | 66 → 84; +18 [+1.0, +33.8] | 20 → 62; +42 [+23.0, +57.0] |

In GPT-4o-mini at cap \$10, 0 → 2; +2 [−5.3, +10.5] means fixed ruined in no games and variable in one of fifty. The point difference is +2 percentage points, but the interval includes zero, so at this cap no difference can be established; with fifty games per arm a single game is two points, and at the smallest cap almost nothing ruins either arm. At cap \$30 the same model reads +20 [+8.7, +33.0], where the whole interval is above zero and the difference is clear.

Nine of the twelve intervals exclude zero, and there is no cell where the variable arm ruined less than the fixed arm. At Gemini's cap \$70 both conditions played all fifty games, so a difference in participation cannot account for 20% against 62%.

Two open-weight models were also compared at cap \$70, at 100 games per condition, under the participation-framing prompt rather than the five-module prompt, so they are matched to each other and not to the table above.

| model | fixed → variable | difference | mean wager per played round |
|---|---|---|---|
| LLaMA-3.1-8B | 6.0% → 82.0% | +76.0 pp [+65.2, +83.1] | \$66.5 → \$34.1 |
| Gemma-2-9B | 1.0% → 15.0% | +14.0 pp [+6.8, +22.3] | \$66.8 → \$24.4 |

In both open-weight models the fixed condition staked roughly twice as much per played round and still went bankrupt far less often. The submitted matched-cap result therefore does not rest on GPT-4o-mini alone, and the reading that the maximum stake or the realised wager cannot by itself account for the fixed-variable difference is supported in further models.

Because these experiments used different prompt conditions and sample sizes, we do not read them as one homogeneous panel effect; they show that the direction recurs wherever bankruptcy events are observable. Part of the lower bankruptcy of the fixed condition also comes from stopping: at cap \$70 LLaMA re-bet after its first loss in 26% of fixed games against 100% of variable games, so the revision reports participation, realised wager and first-loss re-betting beside every bankruptcy figure. We are grateful for a question that let us show which models and which behavioural routes reproduce the difference, rather than generalising from one model.
