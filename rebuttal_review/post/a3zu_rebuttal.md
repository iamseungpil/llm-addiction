# Response to Reviewer a3Zu

Thank you for the close reading. Following your points we sharpened what the language analysis claims, re-audited the moving-target metric, and ran the demonstration experiment you proposed on the four API models.

## [W, Q2] Human validation of the categories, and the role of the human literature

> "reasoning traces are scanned for language associated with loss chasing [...] but never validated against human judgement."

> "Have you looked at whether humans playing the exact same slot-machine game [...] show similar patterns?"

Both points are fair: no independent annotator checked the flagged sentences, and we ran no human control on this game. On the second question, the human literature is design ground here rather than a comparison group. The task, the five prompt modules and the two measurement axes come from clinical gambling research cited in Section 2, and the paper asks how one model's behaviour changes when the prompt and the betting freedom change, so every primary result is a within-model contrast. Our slot machine simplifies those paradigms; we would expect the same qualitative direction in people, but we did not test it, so no model rate should be read as a human rate.

On the first, it matters what exists. We looked for a validated instrument and found none: the gambling-cognition instruments (the Gambling Related Cognitions Scale, the Gamblers' Beliefs Questionnaire and their relatives) are self-report questionnaires, not schemes for coding free text; Toneatto's typology and the think-aloud tradition define categories but are applied by trained human raters; the closest gambling-specific public resource, a DSM-5 and GRCS annotation guide, is also for human coders; and the one published distortion lexicon we found covers general, non-gambling distortions. With roughly 190,300 decisions to score, hand coding was not an option either, so we wrote a lexicon: the categories are the frames the paper cites, the expressions map how these models actually phrase them. One frozen rule set scores every model and condition with no per-condition tuning, and a match inside a negation such as *not*, *never* or *avoid* counts as a mention, not an endorsement.

Writing expressions against observed text invites a fair objection: a lexicon read off a corpus will fire on that corpus. We answer it with a second instrument. Its four constructs are the ones Goodie & Fortune's meta-analysis (2013, cited in the paper) finds converging across the validated instruments; its expressions were deduced from those published definitions alone, without consulting our responses, and frozen before any statistic. Re-scoring the full corpus reproduces the goal-versus-no-goal contrast in all six models: illusion of control +16.7 to +58.4 percentage points, impaired control +13.5 to +50.6. An instrument that never saw our text finds the same contrast, so the effect is not an artefact of how the first lexicon was built. Two limits we hold ourselves to: illusion of control can over-fire in the variable condition, where stake size genuinely is under the model's control, and the variable-minus-fixed contrast is negative in Gemini under every variant we tried.

What remains is that neither instrument was checked against human judgement, and no counting method shows this language mediates bankruptcy; it describes how choices are justified. The corpus does show the language tracking the risky conditions, with loss-chasing expressions rising under variable betting in five of six models and under the goal prompt in all six, with multiplicity control (Section 3). We read that resemblance as consistent with the paper's framing, that models trained on human-generated text can reproduce human-like reasoning patterns under conditions the gambling literature calls risky, while noting this study does not test the training data as a cause. The camera-ready calls the measure distortion-associated language throughout and prints the full expression list, scoring scopes, exclusion rules and worked examples in the appendix; if the lexicon is judged insufficient we will strengthen it further, expanding the expressions and adding human annotation.

## [Q1] Parsing error in the moving-target metric

> "What is the parsing error or ambiguity rate for the moving-target metric?"

Among goal-condition escalation events, 4.2% meet a conservative automated flag: no *goal*, *target* or *aim* within 150 characters of the matched number, and that number already printed in the prompt, so it is unlikely to be a newly stated goal. This is an automated flag, not a human-adjudicated error rate.

The finding does not run through that parser. In the open-weight runs the goal is stored by the game engine, so no text extraction happens, and there the goal conditions escalate in 24.6% and 30.6% of games while the no-goal conditions read 0.0%. The extractor's real weakness is on the other side: without a goal prompt it sometimes reads a balance or wager as a goal, inflating the no-goal baseline in the submitted figure. The camera-ready reports the engine-state and text-extracted measurements separately, and under the stricter rule (state a goal, meet it, raise it) on the API sample the contrast is 2.24 times against 2.83, smaller and in the same direction.

## [Q3] A cautious example and an escalating example

> "What happens with one example of cautious play and timely stopping? [...] Also try the opposite: one escalating-play example under BASE."

We ran both, under BASE as you specify, with the same participation-framing prefix in every arm, so the example is the only thing that differs. Each is a four-round worked session registered verbatim before launch: the cautious player wagers \$20 every round, loses twice, wins once and stops at \$120; the escalating player raises \$20 to \$40 after the first loss and also stops after a win. Both re-bet after losses and stop on a win, so the registered direction test isolates escalation of the stake, judged against a ±10-percentage-point equivalence band. Each cell is compared with its own no-demonstration baseline collected under an identical prompt stack. The four API models ran at 100 games per cell, cap \$70; the result is therefore about these four, not about open-weight models. Bankruptcy, % of games:

| model, condition | no demo | cautious | escalating | escalating − cautious, 95% | mean stake, cautious → escalating |
|---|---|---|---|---|---|
| GPT-4o-mini, variable | 0 | 4 | 8 | +4.0 [−3.0, +11.4] | \$19.7 → \$23.7 |
| GPT-4.1-mini, variable | 0 | 0 | 2 | +2.0 [−2.0, +7.0] | \$19.1 → \$24.4 |
| Gemini-2.5-Flash, fixed | 12 | 13 | 11 | −2.0 [−11.3, +7.3] | \$60.7 → \$65.1 |
| **Gemini-2.5-Flash, variable** | 32 | **21** | **52** | **+31.0 [+17.8, +42.7]** | \$24.0 → \$30.2 |

The four omitted cells sit at 0% in every arm, two of them because those models decline the forced \$70 game outright. Your hypothesis holds in the one cell with room to test it: in Gemini's variable condition the cautious example cuts bankruptcy below its own baseline, 21% against 32%, while the escalating example raises it to 52%.

Two further signals show what the bankruptcy floors hide. The last column is one of them: wherever the model chooses its own stake, the escalating example raises what it actually bets, so the manipulation transmits even in cells where ruin cannot move. The other is participation, which in three cells rises significantly further under the escalating example than the cautious one, meaning models are drawn into games they would otherwise decline.

So a single example calibrates play in both directions, which is the anchoring effect you proposed with a sharper edge than expected: a cautious example helps where there is room to be hurt, but any example legitimises playing at all, and the escalating one both draws models in and, where per-round discretion exists, more than doubles ruin. One limit we flag ourselves: the examples' win lines imply a more generous payout than the game implements, and since both arms share that text the direction contrast is unaffected.
