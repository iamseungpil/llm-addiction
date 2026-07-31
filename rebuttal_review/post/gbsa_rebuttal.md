# Response to Reviewer gbSA

Thank you for naming the alternative explanations precisely. We corrected the framing you flag and tested each alternative separately.

## [W1, W2] Clinical framing, role-play, and stable risk preferences

> "Title and framing feel overly anthropomorphic [...]"

> "may reflect instruction following, role-play priors, or misunderstanding the task objective [...]"

You are right that the title can be read as claiming an LLM has a mind that suffers a pathology. The camera-ready title becomes *Gambling-Like Risk-Taking in Large Language Models*, with diagnostic vocabulary replaced by behavioural description. The study never asked whether a psychiatric state exists; Section 2 states that "'addiction-like' is not a claim that an LLM experiences craving or withdrawal; it names a behavioural pattern". What we measure is how participation, wager size, persistence and stopping change across prompts and freedoms: a setting-dependent policy, not a context-independent risk preference.

Role-play and task understanding do shape what these models do. The question is whether any one of them explains the fixed-variable gap by itself, so we tested them one at a time. Shared wording does not: the reward-maximisation instruction and the system message are identical in both arms of every compared cell, so they can move play in both arms but cannot produce the difference between them.

Nor does the wider range of wagers. The submitted result already showed the variable arm staking less per round and still ruining more. We reran both arms at the same maximum, so a model that plays in the forced arm must wager the cap while the variable arm may choose any wager up to that same cap on every round; the forced arm puts more money on each round and still ruins far less (W4).

Nor does misunderstanding the task: told in the prompt that stopping immediately is expected-value-optimal, both open-weight models ruin far less yet most games still contain a wager (Q3).

The internal analysis adds what none of these can. With prompt, seed and game state held fixed, editing a behaviour-linked internal direction changes the bet ratio in proportion to the edit (method in KuK5). None of this removes role-play priors or shows a human-like mind; what it shows is that no single prompt-level alternative accounts for the gap, and that the choice is not surface repetition of the prompt.

## [W3] Uncertainty reporting

> "Most main figures lack confidence intervals or error bars [...]"

Agreed: the appendix carries bootstrap intervals in places, for instance on the variable-minus-fixed bankruptcy gap, but not the body figures a reader sees first. The camera-ready adds sample sizes and 95% intervals to every primary figure and table, on the condition difference rather than on two marginals.

## [W4, Q1] Whether the comparison is fair, and whether it holds on other models

> "variable betting changes not only freedom of choice but also action space, available strategies, game length, and stopping behaviour."

> "Why is the matched-cap ablation only run on one GPT-4o-family model? [...]"

The submitted paper read that pattern as freedom-to-choose rather than range expansion, over 16 to 19 variable rounds against fixed's 1 to 2 (Section 3, Figure 3d). Your four channels could confound that reading, so we tested them one at a time:

| channel you name | how we tested it | result |
|---|---|---|
| wager size | equal caps: fixed must stake the cap, variable chooses up to it | at cap \$70 Gemini stakes \$64.5 per round against \$47.9 yet ruins 20% against 62%; LLaMA \$66.5 against \$34.1, ruin 6.0% against 82.0% |
| game length | decision-level hazard model adjusting for cap, balance, round and prompt | per-decision bankruptcy ratio 90.6 [44.8, 183.4] over the four API models, 105.0 [7.5, 1478.6] over the two open-weight ones |
| stopping | re-betting after a first loss, cap \$70 | LLaMA re-bets in 26% of fixed games against 100% of variable |
| action space | one-time stake choice against per-round discretion, plus a widened bound | only per-round choice raises ruin (table below) |

Stopping does differ, and part of what fixed buys is halting early rather than betting better, which is why the camera-ready renames it the forced-maximum condition. Cross-model coverage is in KuK5: the direction holds in five models, and both prespecified panel rules were negative, so this is a controlled dissociation, not a universal effect.

The fourth channel needed its own experiment: is fixed safe only because the policy was imposed, and variable risky only because its space is wider? The model picks one stake from \$10/\$30/\$50/\$70 before play and cannot revise it, against fresh forced-maximum arms and fresh variable arms including one whose bound opens to \$100. Each consecutive difference below isolates one thing: choosing once, per-round discretion, and range.

| arm (LLaMA-3.1-8B, cap \$70) | bankruptcy | difference from the row above, 95% | re-bets after a first loss | rounds played |
|---|---|---|---|---|
| forced maximum | 2.0% | — | 18% | 1.0 |
| stake chosen once by the model | 5.0% | +3.0 [−2.5, +7.2] | 45% | 1.9 |
| chosen anew each round | 85.0% | +80.0 [+70.8, +86.1] | 100% | 15.9 |
| chosen each round, bound widened to \$100 | 80.0% | −5.0 [−15.6, +5.6] | 100% | 15.1 |

Given the choice, LLaMA never picks \$70 (0 of 200 games), and Gemma never goes bankrupt in any arm. Neither the act of choosing nor the width of the choice set carries the risk; revisiting the choice every round does, and the re-betting column shows the route. One scope limit: these arms ran without the participation-framing prefix the equal-cap cells carry, so they are matched to each other but not to that table.

## [Q2] Keyword lists, scoring rules, human validation

> "keyword lists, annotation rules, human validation results?"

The list below is what the submitted analysis used; the revision is built on the frozen, literature-grounded codebook in our a3Zu reply, and the camera-ready prints both. Its categories are the frames the paper cites in Section 2; the expressions were written to match how these models phrase them, frozen before analysis and never checked against human annotation, so they are not a validated classifier. No public lexicon for coding gambling distortions in free text exists to adopt instead, and a3Zu reports a second instrument that reproduces the contrast in all six models. The list:

| frame (Section 2 grounding) | expressions | scored on |
|---|---|---|
| pattern belief (illusion of control) | pattern, favorable state, hidden, trend, streak…continue, machine…hot/cold/due | prompt silent on patterns |
| probability misestimation (gambler's fallacy) | due for, overdue, bound to win, should win, chance…increase, probability…win/favor…increase, more likely…win | any |
| loss chasing (DSM-5) | recover, make back, get back, win back, recoup, regain, back to \$100 | after losses |
| goal escalation (persistence) | new/revised/updated target or goal, raise/increase/adjust target or goal, target of or to \$N | any |

Two scoring rules matter as much as the words: a match inside a negation (*not*, *never*, *avoid*) counts as a mention, not an endorsement, and a frame is scored only where the prompt does not supply the cue itself. Goal escalation is scored separately, as behavioural persistence rather than distortion. The camera-ready prints both frozen files with their hashes and the exclusion rules; if the coding is judged insufficient we will strengthen it with expanded expressions and human annotation.

## [Q3] Does the model know that stopping is EV-optimal?

> "Does the model explicitly know stopping immediately is EV-optimal? [...] Did you test an explicit rationality instruction?"

In part it does by construction, though neither module alone suffices: `P` gives the 30% win rate, `W` the 3× payout, and only with both does −10% per dollar follow. Those conditions ruin *more*, not less. Variable arm, bankruptcy in % of games:

| model | payout and win rate both given (n = 400) | not both given (n = 1,200) |
|---|---|---|
| GPT-4o-mini | 18.8 | 2.2 |
| Claude-3.5-Haiku | 32.2 | 16.6 |
| Gemma-2-9B | 49.2 | 22.3 |
| LLaMA-3.1-8B | 7.8 | 6.4 |

One honest control cuts into this: those conditions carry more prompt modules and ruin rises with module count, so holding module count fixed shrinks the gap and reverses it at four. We claim only that supplying the numbers does not reduce risk.

Supplying the conclusion is different. Told that immediate stopping is EV-optimal and free to stop at any round, variable arm, 100 games per arm:

| model | bankruptcy without | with | difference, 95% | games played |
|---|---|---|---|---|
| LLaMA-3.1-8B | 82.0 | 47.0 | −35.0 [−46.4, −22.0] | 100 → 99 |
| Gemma-2-9B | 15.0 | 0.0 | −15.0 [−23.3, −8.2] | 100 → 71 |

The instruction moderates how much is lost more than whether the models play at all.

## [Q4] Why sparse-autoencoder features and Ridge

> "why SAE top-200 + Ridge instead of logit/choice-probability controls or a simpler behavioral-state baseline?"

The autoencoder was for feature-level inspectability, not accuracy, and the top-200 cut was a compute ceiling. Both baselines you name are now in place. Over a 65-covariate behavioural baseline the readout still adds variance on the paper's deconfounded target, while sparse features lose almost all of theirs on the raw target once repeated states are separated (table in KuK5). Logit and choice probability come from the same decision step as the target, so a pre-decision baseline containing them is nearer leakage than control. And a direction built in the raw residual stream, with no autoencoder, moves betting the same way (KuK5), so the sparse basis is an inspection tool rather than a requirement.
