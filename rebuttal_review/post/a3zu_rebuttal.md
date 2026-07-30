# Response to Reviewer a3Zu

Your weakness and three questions concern whether our measurement instruments support the weight placed on them. We address each directly below, distinguishing what the submitted analyses establish, what the response-period checks add, and what remains unresolved. We sincerely thank the reviewer for helping us improve the paper.

## W. Validation of the distortion categories against human judgement

**The reviewer is correct that literature-grounded categories do not make our lexical classifier human-validated.** We did not validate the submitted expression rules against independent human judgements, and we will not present them as a validated psychological instrument.

The constructs come from prior gambling-cognition research — Raylu & Oei's Gambling Related Cognitions Scale, Toneatto's typology, Goodie & Fortune's review, Smith et al.'s DSM-5/GRCS guide — while the expression lists and scoring rules are our own operationalisation. The submitted paper already limits the interpretation: "the language analysis is not evidence that the model independently discovers those distortions, only that high-risk regimes are accompanied by loss-recovery and control-like justifications in the generated reasoning".

We performed a second check to determine whether the reported contrast is driven by the category most directly induced by the prompt. In the submitted codebook, the goal-versus-no-goal contrast is dominated by `goal_escalation` (+65 to +95 percentage points), which is near-tautological because the goal prompt instructs the model to set a goal. We therefore rescored the complete corpus of 19,200 games using a frozen response-period codebook with the goal category removed:

| category | positive models | goal − no-goal difference | main limitation |
|---|---|---|---|
| `illusion_of_control` | 6/6 | +16.7 to +58.4 pp | may over-fire where wager size is genuinely under model control |
| `impaired_control` | 6/6 | +13.5 to +50.6 pp | author-defined lexical rule without human validation |

This result shows that explicit goal-setting language does not wholly account for the contrast. It is a response-period robustness probe, not an independent replication or a substitute for human validation. Two limits we hold ourselves to: `illusion_of_control` can over-fire in the variable condition, and the variable-minus-fixed contrast is negative in Gemini under every instrument variant we tested.

**Revision.** The camera-ready labels the measure *distortion-associated language*, reports goal escalation separately as behavioural persistence, and prints the expression lists, scoring scopes and worked examples in its appendix.

## Q1. Parsing error or ambiguity in the moving-target metric

**The audit revealed two distinct issues: 4.2% of goal-condition events meet a conservative automated error criterion, and the submitted figure pools two different measurement instruments.**

The columns below are the paper's four prompt conditions: BASE; M, reward maximisation; G, self-directed goal setting; and GM, both.

| moving-target rate, % of games | BASE | M | G | GM |
|---|---|---|---|---|
| open-weight, goal read from engine state (n = 800) | 0.0 | 0.0 | **24.6** | **30.6** |
| API, goal extracted from free text (n = 1,600) | 25.4 | 16.5 | **62.3** | **56.4** |
| Figure 3(c), the two pooled (n = 2,400) | 17.0 | 11.0 | **49.8** | **47.8** |

For the open-weight models the environment stores the goal directly, so no text parser is involved: the no-goal conditions read 0.0%, while the goal conditions still escalate in 24.6% and 30.6% of games — the strongest parsing-independent evidence for the finding.

For the API models, goals must be extracted from text, and the extractor returns 25.4% and 16.5% where no goal exists — balances and wagers mistaken for goals. Pooling those values with the engine-state rows is how 11–17% came to stand as a baseline; we will replace the pooled figure and report the two instruments separately.

Among the 3,638 goal-condition escalation events, **4.2% meet our conservative automated error criterion**: no instance of *goal*, *target*, or *aim* occurs within 150 characters of the matched number, and that number was already printed in the prompt, so it cannot be a newly stated goal. This is a conservative flagged-error rate rather than a human-adjudicated error rate. The pipeline reproduces the submitted values to one decimal across the four conditions and the corresponding appendix cells.

We also repeated the analysis using the paper's stricter behavioural definition: the model must state a goal, achieve it, and then raise it. Restricting the comparison to API runs so both sides use the same instrument:

| strict-rule escalation % | BASE | M | G | GM |
|---|---|---|---|---|
| n = 1,600 per condition | 24.6 | 14.8 | **46.1** | **42.2** |

The pooled goal-to-no-goal contrast becomes 2.24×, compared with 2.83× under the submitted rule on the same sample: smaller, but in the same direction.

## Q2. Whether humans show the same pattern in the same game

**We have not conducted a matched human experiment.** A responsible comparison would require a separate human-subject protocol, ethics approval, recruitment, and human-specific definitions of participation, escalation, and stopping. It cannot be completed within the rebuttal period.

Our indicators have a narrower purpose: they are **relative audit instruments** that measure how the same model changes when one task condition changes. Their absolute levels were not calibrated to human prevalence or behavioural rates, and every primary result in the paper is a within-model contrast.

Human slot-machine research motivates the task and the categories examined, but it is not a matched control. The camera-ready will explicitly state that no model rate should be interpreted as a human rate. Prior human think-aloud work will be used only as qualitative context for the language categories.

## Q3. Cautious and escalating demonstrations

**The requested direct experiment is registered and running, and every specified cell reports by 3 August regardless of direction.** All submitted conditions were zero-shot. One new arm prepends a worked example of cautious play with timely stopping; the other an escalating-play example. Both run in the fixed and variable conditions under the plain BASE prompt, on the two open-weight models, 200 games per cell at cap \$70 — raised from the registered 100 before launch by a dated amendment — with seeds shared across arms and directions fixed in advance. Until the runs complete, we do not treat this question as answered.

Two completed analyses provide relevant calibration evidence without replacing the requested test.

**First, making the numerical inputs for expected value available did not reduce bankruptcy.** Across the submitted 32 conditions, the conditions in which the −10% expectation is computable reach bankruptcy more often, not less, in all four models with stored text: GPT-4o-mini 18.8% versus 2.2%, Claude 32.2% versus 16.6%, Gemma 49.2% versus 22.3%, and LLaMA 7.8% versus 6.4%. Prompt richness remains a potential confound: holding module count fixed, the gaps are +9.9 and +7.2 percentage points at two and three modules and −2.7 at four. We therefore claim only that numerical availability does not itself reduce risk; it does not establish that the model performed or used the calculation.

**Second, stating the decision-theoretic conclusion directly does change behaviour.** An instruction that immediate stopping maximises expected value, while permitting stopping at any round, reduces LLaMA's variable-condition bankruptcy from 81.5% (n = 200) to 3.0% (n = 100). This supports the reviewer's interpretation that calibration is a strong moderator. It does not eliminate participation: 69 of 100 variable-condition games still contain a wager, compared with 2 of 100 fixed-condition games.

An instruction supplies the conclusion; only a worked example tests calibration by demonstration, which is what the running experiment measures.

**Scope.** These are artificial negative-expected-value games. We claim condition-dependent risk-taking and decodable signals, not addiction, a fixed disposition, or a shared psychological mechanism. Where generated traces resemble loss-chasing or control language from the human literature, we report resemblance rather than a shared cause.
