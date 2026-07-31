# Response to Reviewer gbSA

Thank you for naming the alternative explanations precisely. We corrected the framing you flag and tested how far role-play and task understanding explain the fixed-variable gap; tables in another response are pointed to, not repeated.

## [W1, W2] Clinical framing, role-play, and stable risk preferences

> "Title and framing feel overly anthropomorphic [...]"

> "may reflect instruction following, role-play priors, or misunderstanding the task objective [...]"

You are right that the title can be read as claiming an LLM has a mind that suffers a pathology. The camera-ready title becomes *Gambling-Like Risk-Taking in Large Language Models*, with diagnostic vocabulary replaced by behavioural description. The study never asked whether a psychiatric state exists: Section 2 states that "'addiction-like' is not a claim that an LLM experiences craving or withdrawal; it names a behavioural pattern". What we measure is how participation, wager size, persistence and stopping change across prompts and freedoms, which is a setting-dependent policy, not a context-independent risk preference.

Three checks say none of them carries the contrast alone: the reward-maximisation instruction and system message are identical across both arms of every compared cell; with the maximum held equal, fixed puts more money on each round yet ruins less (W4); and telling the model that stopping is optimal cuts bankruptcy sharply while leaving most games still played (Q3).

The internal analysis adds what prompt-level checks cannot. With prompt, seed and game state held fixed, editing a behaviour-linked internal direction changes the bet ratio in proportion to the edit (method in KuK5). This neither removes role-play priors nor shows a human-like mind, but it does show the choices are not mere surface repetition of the prompt.

## [W3] Uncertainty reporting

> "Most main figures lack confidence intervals or error bars [...]"

Agreed: the appendix carries bootstrap intervals in places, for instance on the variable-minus-fixed bankruptcy gap, but not the body figures a reader sees first. The camera-ready adds sample sizes and 95% intervals to every primary figure and table, and puts the interval on the condition difference rather than on two marginals.

## [W4, Q1] Whether the comparison is fair, and whether it holds on other models

> "variable betting changes not only freedom of choice but also action space, available strategies, game length, and stopping behaviour."

> "Why is the matched-cap ablation only run on one GPT-4o-family model? [...]"

The submitted ablation found variable betting ruining more while wagering smaller amounts over 16 to 19 rounds against fixed's 1 to 2, reading that as freedom-to-choose rather than range expansion (Section 3, Figure 3d). Your four channels could confound that reading, so we tested them one at a time:

| channel you name | how we tested it | result | verdict |
|---|---|---|---|
| wager size | equal caps: fixed must stake the cap, variable chooses up to it | at cap \$70 Gemini stakes \$64.5 per round against \$47.9 yet ruins 20% against 62%; LLaMA \$66.5 against \$34.1, ruin 6.0% against 82.0% | ruled out |
| game length | decision-level hazard model adjusting for cap, balance, round and prompt | per-decision bankruptcy ratio 90.6 [44.8, 183.4] over the four API models, 105.0 [7.5, 1478.6] over the two open-weight ones | ruled out |
| stopping | re-betting after a first loss, cap \$70 | LLaMA re-bets in 26% of fixed games against 100% of variable | a real difference |
| action space | one-time stake choice against per-round discretion, plus a widened bound | a self-chosen stake behaves like the forced one; only per-round choice raises ruin (table below) | repeated discretion, not choice-set size |

Two channels are ruled out: fixed is safer even when it puts more money on each round and even after adjusting for how long games run. Stopping does differ, and part of what fixed buys is halting early rather than betting better, which is why the camera-ready renames it the forced-maximum condition. Cross-model coverage is in KuK5, whose table gives every cap and model with intervals: the direction holds in five models, and both prespecified panel rules were negative, so this is a controlled dissociation, not a universal effect.

The fourth channel needed its own experiment: is fixed safe only because the policy was imposed, and variable risky only because its space is wider? The model picks one stake from \$10/\$30/\$50/\$70 before play and cannot revise it, against fresh forced-maximum arms and fresh variable arms including one whose bound opens to \$100, so model-chosen minus forced isolates choosing once, variable minus model-chosen isolates per-round discretion, and the \$100 arm isolates range.

| arm (LLaMA-3.1-8B, cap \$70) | bankruptcy | re-bets after a first loss | rounds played |
|---|---|---|---|
| forced maximum | 2.0% | 18% | 1.0 |
| stake chosen once by the model | 5.0% | 45% | 1.9 |
| chosen anew each round | 85.0% | 100% | 15.9 |
| chosen each round, bound widened to \$100 | 80.0% | 100% | 15.1 |

Given the choice, LLaMA never picks \$70 (0 of 200 games), and its self-chosen stake is equivalent to the forced one (+3.0 pp [−2.5, +7.2]) while per-round choice is not (+80.0 pp [+70.8, +86.1]); widening the bound changes nothing (−5.0 pp [−15.6, +5.6]) and Gemma never goes bankrupt in any arm. Neither the act of choosing nor the width of the choice set carries the risk; revisiting the choice every round does, and the re-betting column shows how. One scope limit we state ourselves: these arms ran without the participation-framing prefix the equal-cap cells carry, so they are matched to each other but not to that table.

## [Q2] Keyword lists, scoring rules, human validation

> "keyword lists, annotation rules, human validation results?"

The categories are the frames the paper cites in Section 2; the expressions were written to match how these models phrase them, frozen before analysis and never checked against human annotation, so they are not a validated classifier. No public lexicon for coding gambling distortions in free text exists to adopt instead, and a3Zu reports a second instrument that reproduces the contrast in all six models. The list:

| frame (Section 2 grounding) | expressions | scored on |
|---|---|---|
| pattern belief (illusion of control) | pattern, favorable state, hidden, streak…continue, machine…hot/cold/due | prompt silent on patterns |
| probability misestimation (gambler's fallacy) | due for, overdue, bound to win, should win, chance…increase, probability…win/favor…increase, more likely…win | any |
| loss chasing (DSM-5) | recover, make back, get back, win back, recoup, regain, back to \$100 | after losses |
| goal escalation (persistence) | new/revised/updated target, raise/increase/adjust target, target of \$N | any |

Two scoring rules matter as much as the words: a match inside a negation (*not*, *never*, *avoid*) counts as a mention, not an endorsement, and a frame is scored only where the prompt does not supply the cue itself. Goal escalation sits apart because it measures persistence, not distortion. The camera-ready prints the frozen file and hash, the exclusion rules and worked examples; if the lexicon is judged insufficient we will strengthen it with expanded expressions and human annotation.

## [Q3] Does the model know that stopping is EV-optimal?

> "Does the model explicitly know stopping immediately is EV-optimal? [...] Did you test an explicit rationality instruction?"

In part it does by construction, though neither module alone suffices: `P` gives the 30% win rate, `W` the 3× payout, and only with both does −10% per dollar follow. Those conditions ruin *more*, not less. Variable arm, bankruptcy in % of games:

| model | payout and win rate both given (n = 400) | not both given (n = 1,200) |
|---|---|---|
| GPT-4o-mini | 18.8 | 2.2 |
| Claude-3.5-Haiku | 32.2 | 16.6 |
| Gemma-2-9B | 49.2 | 22.3 |
| LLaMA-3.1-8B | 7.8 | 6.4 |

One honest control cuts into this: those conditions carry more prompt modules and ruin rises with module count, so holding it fixed shrinks the gap and reverses it at four modules. We claim only that supplying the numbers does not reduce risk.

Supplying the conclusion is different. Told that immediate stopping is EV-optimal and free to stop at any round, LLaMA's variable bankruptcy falls from 82.0% to 47.0% (−35.0 pp [−46.4, −22.0]) and Gemma's from 15.0% to 0.0% (−15.0 pp [−23.3, −8.2]) at 100 games per arm. It moderates how much is lost more than whether they play: 99 of 100 LLaMA games still carry a wager, though Gemma's participation does fall to 71%.

## [Q4] Why sparse-autoencoder features and Ridge

> "why SAE top-200 + Ridge instead of logit/choice-probability controls or a simpler behavioral-state baseline?"

The autoencoder was for feature-level inspectability, not accuracy, and the top-200 cut was a compute ceiling. Both baselines you name are now in place. The 65-covariate behavioural baseline explains a large share of the raw bet ratio and much less of the deconfounded target; over it the readout still adds variance on the deconfounded target, while sparse features lose almost all of theirs on the raw target once repeated states are separated (table in KuK5). Logit and choice probability come from the same decision step as the target, so a pre-decision baseline with them is nearer leakage than control.

The intervention settles necessity: a direction built in the raw residual stream, with no autoencoder involved, moves betting the same way (method and controls in KuK5). The sparse basis is an inspection tool, not a requirement.
