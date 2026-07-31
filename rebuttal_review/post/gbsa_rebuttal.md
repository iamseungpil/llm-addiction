# Response to Reviewer gbSA

Thank you for naming the alternative explanations precisely. We make the clinical framing behavioural and test how far role-play, wager size, task understanding, stopping and action structure account for the fixed-variable difference.

## [W1, W2] Clinical framing, role-play, and stable risk preferences

> "Title and framing feel overly anthropomorphic [...]"

> "may reflect instruction following, role-play priors, or misunderstanding the task objective [...]"

We agree that the title and some wording can be read as claiming that an LLM has a human-like mind and undergoes a genuine pathology. The camera-ready title becomes *Gambling-Like Risk-Taking in Large Language Models*, with diagnostic wording replaced by behavioural description of participation, wagering, persistence and stopping. The paper does not study whether a mental state exists; Section 2 restricts the term: "'addiction-like' is not a claim that an LLM experiences craving or withdrawal; it names a behavioural pattern". What we analyse is how the risky behaviour of one model changes with the prompt and the structure of the choice, which is a setting-dependent policy rather than a context-independent risk preference.

Role-play and task understanding do shape what these models do, as the sensitivity to a rationality instruction and to a worked example demonstrates, and which prompts amplify or suppress risky behaviour is what the paper asks. The reward-maximisation instruction and the system message are identical in both arms of every compared cell, so shared wording cannot produce the difference between them.

We also tested whether the behaviour is only repetition of the prompt. Holding prompt, seed and game state fixed, moving a betting-related activation direction changed the bet ratio in proportion to the edit in both models, and removing that component lowered betting; controls and model-specific limits are in our reply to KuK5's W1 and Q2. Part of the internal state therefore affects the immediate choice, and the fixed-variable difference is not explained by shared instructions or task misunderstanding alone.

## [W3] Uncertainty reporting

> "Most main figures lack confidence intervals or error bars [...]"

We agree. The appendix carries bootstrap intervals in a few places, for instance on the variable-minus-fixed bankruptcy gap, and some results carry a p-value, but the body figures a reader sees first show no uncertainty, and a p-value conveys neither effect size nor precision. The camera-ready adds sample sizes and 95% intervals to every primary figure and table, on the condition difference rather than on each arm, and the same rule applies to every result added here.

## [W4, Q1] Fairness of the comparison and replication in other models

> "variable betting changes not only freedom of choice but also action space [...] game length, and stopping behaviour."

> "Why is the matched-cap ablation only run on one GPT-4o-family model?"

The submitted matched-cap analysis evaluated GPT-4o-mini on the slot machine at caps of \$10, \$30, \$50 and \$70 and found the variable condition staking less per round while playing longer and going bankrupt more often; the paper writes GPT-4o for that model, which the camera-ready corrects. That was evidence against a pure maximum-stake account, but it came from one model, and stopping and action structure still varied together.

Repeating the equal-maximum comparison on further API models gave the same result: the fixed condition, where the environment sets the wager to the cap, staked more per round than the variable condition, which chooses any amount up to that cap each round, and still went bankrupt less often. Per-model figures with intervals, and the prompt condition each cell used, are in our reply to KuK5's W2 and Q1.

Stopping is a real difference: at cap \$70 LLaMA re-bet after its first loss in 26% of fixed games and 100% of variable games, so part of the fixed condition's lower bankruptcy comes from not entering or stopping early rather than from choosing amounts better. The revision keeps the names *fixed* and *variable*, defines fixed at first use as the environment setting the wager, and reports participation, realised wager and first-loss re-betting beside every bankruptcy figure.

To separate action structure we added a condition in which the model chooses the amount before play and cannot change it, against the environment-set condition, per-round choice, and a wider bound.

| condition (LLaMA-3.1-8B, cap \$70) | bankruptcy | difference from the row above | re-bets after a first loss | mean rounds |
|---|---|---|---|---|
| amount set by the environment | 2.0% | — | 18% | 1.0 |
| amount chosen once before play | 5.0% | +3.0 pp [−2.5, +7.2] | 45% | 1.9 |
| amount chosen each round | 85.0% | +80.0 pp [+70.8, +86.1] | 100% | 15.9 |
| amount chosen each round, wider bound | 80.0% | −5.0 pp [−15.6, +5.6] | 100% | 15.1 |

Each bracket is the 95% interval on the difference against the row above, so +3.0 pp [−2.5, +7.2] includes zero while +80.0 pp [+70.8, +86.1] lies entirely above it. Letting the model set its own amount once therefore behaved like the environment setting it; only choosing again each round raised bankruptcy and re-betting, and widening the range added nothing. Given the choice, LLaMA never picks \$70 in 200 games. The result fits per-round discretion, the ability to revise a decision after a loss, better than the existence of a choice or the width of the choice set.

## [Q2] Keyword lists, scoring rules, human validation

> "keyword lists, annotation rules, human validation results?"

This is a codebook analysis looking in reasoning traces for language patterns studied in gambling-cognition research. The categories and definitions come from that prior work, which supplies no keyword lexicon ready for free-form LLM responses; the exact expressions are therefore ours, written to operationalise those definitions, fixed before any statistic was computed and used unchanged for every model and condition. The complete frozen codebook, all thirty expressions across the four categories, is printed in our reply to a3Zu's first weakness.

These figures apply no temporal window: every decision in a game is scored. The revision adds, as a stricter sensitivity analysis, a version that excludes decisions where the prompt supplies the cue and restricts impaired-control expressions to decisions after a loss. A match inside a negation counts as a mention rather than an endorsement, and goal escalation is scored separately as behavioural persistence. The regex file with its hash and worked accepted and rejected examples are published with the revision.

Without independent human annotation we do not present this as a validated classifier, and it is not used to diagnose sentences or to argue that this language mediates bankruptcy; it asks whether related wording becomes more frequent where risk increases. If the content validity is judged insufficient we will limit both the expressions and the claim further, and add human annotation of a stratified sample.

## [Q3] Does the model know that stopping is EV-optimal?

> "Does the model explicitly know stopping immediately is EV-optimal? [...] an explicit rationality instruction?"

The submitted results alone cannot establish that a model knows immediate stopping is optimal and acts on it. The P component supplies the win rate and the W component the payout structure, so conditions containing both state everything needed to compute a negative expected value per bet, yet bankruptcy did not fall consistently there. Some tallies show it higher, but those conditions also carry more prompt components, so we claim only that supplying the information did not consistently reduce risky behaviour.

We then tested the conclusion directly. The prompt stated that the game has negative expected value, that each round loses 10% of the amount wagered in expectation, that stopping immediately therefore maximises expected value, and that the model may stop at any time.

| model, variable condition | bankruptcy before | with the instruction | difference | games with a wager |
|---|---|---|---|---|
| LLaMA-3.1-8B | 82.0% | 47.0% | −35.0 pp [−46.4, −22.0] | 100 → 99 |
| Gemma-2-9B | 15.0% | 0.0% | −15.0 pp [−23.3, −8.2] | 100 → 71 |

Both intervals lie entirely below zero. In Gemma participation fell as well, while LLaMA still wagered at least once in 99 of 100 games, so the instruction moderates how much is lost without removing participation. Having the information needed to compute expected value does not mean a model acts on it, and supplying the conclusion moves behaviour in the safer direction without eliminating betting.

## [Q4] Why sparse-autoencoder features and Ridge

> "why SAE top-200 + Ridge instead of logit/choice-probability controls or a simpler behavioral-state baseline?"

The autoencoder was used to inspect internal representations at the feature level rather than to maximise accuracy, and the top-200 cut bounded compute; it is not a claim that 200 is optimal. Because that readout carried a small signal in the expected direction, we did not present it as causal evidence.

Both baselines you name are now in place, with full results in our reply to KuK5's W1 and Q2. Over a behavioural-state baseline of 65 observable variables the sparse representation still added held-out variance on the paper's deconfounded target, while on the raw bet-ratio target the dense hidden state kept its increment even under an evaluation separating repeated states. Logit and choice probability come from the same decision computation as the target, so they are not an independent state control. Building the behavioural direction in the raw residual stream, with no autoencoder, also reproduced the dose-response, so the sparse basis is an inspection tool rather than a requirement.
