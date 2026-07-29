# Response to Reviewer a3Zu

Your weakness and your three questions all test whether our instruments carry the weight the framing puts on them. Below we answer each directly, separating completed analyses from the demonstration experiment that is still running. We are grateful for a review that has made the paper better.

## W. Validation of the distortion categories against human judgement

**No — the lexical rules were never validated against human judgement, and we will not present them as a validated instrument.** The constructs come from prior gambling-cognition work: Raylu & Oei's Gambling Related Cognitions Scale (GRCS) subscales, Toneatto's typology, Goodie & Fortune's review, and Smith et al.'s DSM-5/GRCS guide. The expression lists and scoring rules are our own. The submitted paper already scopes the claim: "the language analysis is not evidence that the model independently discovers those distortions, only that high-risk regimes are accompanied by loss-recovery and control-like justifications in the generated reasoning". The camera-ready will label the quantity distortion-*associated language* throughout, and goal escalation will be reported separately as a persistence measure.

What we can offer is a dependence check on the category you would most suspect. Re-scored over the full corpus (19,200 games) with the paper's expression lists, the goal contrast is dominated by `goal_escalation` (+65 to +95 points). That category is near-tautological: the goal prompt tells the model to set a goal. A codebook frozen during the response period with no goal category at all still shows the goal contrast in all six models: `illusion_of_control` +16.7 to +58.4 points, `impaired_control` +13.5 to +50.6. That is a robustness probe, not an independent replication, and it has its own limits. `illusion_of_control` over-fires in the variable condition, where stake size genuinely is under the model's control; and the variable-minus-fixed contrast is negative in Gemini under every variant we tried.

## Q1. Parsing error rate for the moving-target metric

**Where a goal exists, the extractor's error rate is 4.2%.** We count an extraction as an error under two joint conditions: no *goal*, *target* or *aim* within 150 characters of the matched number, and the number is one the prompt itself printed, meaning it cannot be a goal the model set. Of 3,638 goal-condition escalation events, 4.2% meet both conditions, and our pipeline reproduces the published figures to one decimal throughout.

Looking also turned up something more consequential: the published baseline mixes two instruments. The columns are the paper's four prompt conditions — BASE; M, reward maximisation; G, self-directed goal-setting; GM, both.

| moving-target rate, % of games | BASE | M | G | GM |
|---|---|---|---|---|
| open-weight, goal read from engine state (n = 800) | 0.0 | 0.0 | **24.6** | **30.6** |
| API, goal extracted from free text (n = 1,600) | 25.4 | 16.5 | **62.3** | **56.4** |
| Figure 3(c), the two pooled (n = 2,400) | 17.0 | 11.0 | **49.8** | **47.8** |

Where the engine holds the goal, no text parsing is involved. The no-goal conditions read 0.0% because there is no goal to raise, and the goal conditions still escalate in a quarter to a third of games. Where the extractor reads free text, the no-goal conditions return 25.4% and 16.5% for something that is not there. Pooling the two is how 11–17% came to stand as a baseline; the revision reports the within-goal rate in its place. The contrast also survives the paper's stricter definition, raising a self-set goal *after meeting it*: 2.24× against the published rule's 2.83× on the same sample, smaller but in the same direction.

## Q2. Whether humans show the same pattern in the same game

**We have not run that comparison, and the paper makes no claim at that level.** Doing it responsibly needs a separate human-subject protocol with ethics approval and human-specific outcome definitions, which does not fit a response window.

What the indicators are built for is narrower: they are **relative audit instruments**. They track how the same model's behaviour shifts when one condition changes, within one task. Their absolute levels were never calibrated to human rates and carry no weight in any finding; every result in the paper is a within-model contrast. The task and the constructs are drawn from human slot-machine research, and that is where the correspondence ends. No rate we report should be read as a human prevalence, and the camera-ready says so outright. Prior think-aloud work enters as qualitative context, not a matched control.

## Q3. Cautious and escalating demonstrations

**Not answered yet — the demonstration arms you asked for are registered and running, and we will report every pre-specified cell by 3 August, whichever way the results fall.** One arm prepends a worked example of cautious play with timely stopping; the other an escalating example. Both run in the fixed and variable modes under the plain baseline, on the two open-weight models, 200 games per cell at cap \$70. The cell size was raised from the registered 100 before launch, with a dated amendment; seeds are shared across arms and directions were fixed in advance. Your premise is correct that every submitted condition is zero-shot; this is the first demonstration test.

Two completed analyses bear on the same reading without standing in for it. Supplying the numbers expected value needs did not reduce risk. Across the paper's 32 conditions, the cells where the −10% expectation is computable from the prompt reach bankruptcy *more* often, not less, in all four models with stored text: GPT-4o-mini 18.8% against 2.2%, Claude 32.2 against 16.6, Gemma 49.2 against 22.3, LLaMA 7.8 against 6.4. Prompt richness is a confound and is controlled; holding module count fixed, the gap is +9.9 and +7.2 points at two and three modules and −2.7 at four. Stating the conclusion outright does change behaviour. Telling LLaMA that immediate stopping maximises expected value, with permission to stop at any round, cuts variable-condition bankruptcy from 81.5% (n = 200) to 3.0% (n = 100). Yet 69 of 100 games still wager, against 2 in the fixed condition. Calibration is a strong moderator, which supports your reading; the contrast survives it, which supports the paper's. An instruction says what to conclude, and only a play-through shows what play looks like. That is why the running experiment is the test.

**Scope.** These are artificial negative-expected-value games. What we claim is condition-dependent risk-taking and decodable signals, not addiction, a fixed disposition, or a mechanism; where the traces resemble the human literature's loss-chasing or control language, we report resemblance rather than a shared cause.
