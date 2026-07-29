# Response to Reviewer a3Zu

Your weakness and your three questions press on one joint — whether the instrument bears the weight the framing puts on it. That is the right place to press. Below, for each, we state the concern, what the paper measures, and what we tested during the response period. We are grateful for a review that has made the paper better; the details follow.

## W. Validation of the distortion categories against human judgement

The concern, as we read it, is that a keyword instrument can only be said to measure cognitive distortions if someone has checked that it agrees with how a reader would code the same trace. That is correct, and we have no such check. What we can show is where the categories come from and whether the reported contrast depends on the one category you would most suspect. Two parts.

**First, the constructs are not ours; the operationalisation is.** No validated public lexicon that we know of adjudicates gambling-specific distortions in free text; the closest are Raylu & Oei's Gambling Related Cognitions Scale (GRCS) subscales, Toneatto's typology, Goodie & Fortune's review and Smith et al.'s DSM-5/GRCS guide. The constructs are grounded in that literature; the expression list and scoring rules are ours and are unvalidated. The frozen list is tabulated in our gbSA response (Q2), and the submitted paper already scopes the claim: "the language analysis is not evidence that the model independently discovers those distortions, only that high-risk regimes are accompanied by loss-recovery and control-like justifications in the generated reasoning".

**Second, the contrast does not depend on the tautological category.** Re-scored over the corpus (19,200 games), our published instrument's goal contrast is dominated by `goal_escalation` (+65 to +95 points) — near-tautological, since the goal prompt tells the model to set a goal. So we rebuilt the codebook with no goal category at all, and the contrast survives on two literature-grounded categories:

| goal − no-goal, points, 6 models | range |
|---|---|
| `illusion_of_control` | +16.7 to +58.4 |
| `impaired_control` | +13.5 to +50.6 |

A codebook frozen during the response period is a robustness probe rather than an independent replication, and we will report it as one. Two limits we hold ourselves to: `illusion_of_control` over-fires in the variable condition, where stake size genuinely is under the model's control, and the variable-minus-fixed contrast is negative in Gemini under every variant we tried.

**On the framing.** The abstract does not use "cognitive distortion" but carries the clinical frame in other words, so the point stands. The second clinical axis in the setup section, the heading of Finding 5 and the matching supplementary passage each become distortion-*associated language*; the construct we retain — irrationality rather than a mental state — is set out in our gbSA response (W1).

## Q1. Parsing error rate for the moving-target metric

We had not measured this, and looking turned up something more useful than a rate. Two parts.

**First, the published baseline mixes two instruments.** The corpus has two halves, and only one of them uses the extractor at all:

| moving-target rate, % of games | BASE | M | G | GM |
|---|---|---|---|---|
| open-weight, goal read from engine state (n = 800) | 0.0 | 0.0 | **24.6** | **30.6** |
| API, goal extracted from free text (n = 1,600) | 25.4 | 16.5 | **62.3** | **56.4** |
| Figure 3(c), the two pooled (n = 2,400) | 17.0 | 11.0 | **49.8** | **47.8** |

Where the engine holds the goal, the no-goal conditions read 0.0% — no goal exists to raise — while the goal conditions still escalate in a quarter to a third of games, a rate owing nothing to text parsing; where the extractor reads free text, the no-goal conditions return 25.4% and 16.5% for something that is not there. Pooling the two is how 11–17% came to stand as a baseline; we will report the within-goal rate in its place.

**Second, where a goal does exist, the extractor's error rate is 4.2%.** We count an extraction as an error when no *goal*, *target* or *aim* appears within 150 characters of the matched number **and** that number is one the prompt itself printed, so it cannot be a goal the model set. Of 3,638 goal-condition escalation events, 4.2% meet both conditions. Our pipeline reproduces the published figure to one decimal on all four conditions and all twelve supplementary cells.

**And the contrast survives a stricter definition.** The paper defines escalation as raising a self-set goal *after meeting it*. Applying that achievement test throughout, API-only so both conditions share one instrument:

| strict-rule escalation % | BASE | M | G | GM |
|---|---|---|---|---|
| n = 1,600 per condition | 24.6 | 14.8 | **46.1** | **42.2** |

Pooled, a 2.24× contrast against the published rule's 2.83× on the same sample — smaller, same direction.

## Q2. Whether humans show the same pattern in the same game

This is a comparison we would like to have, and it is the right one to ask for. We have not run it: doing it responsibly needs a separate human-subject protocol with ethics approval and human-specific outcome definitions, which does not fit a response window.

What the indicators are built for is narrower: they are **relative audit instruments**. They track how the same model's behaviour shifts when one condition changes, within one task — their absolute levels were never calibrated to human rates and carry no weight in any finding; every result in the paper is a within-model contrast. The task and the constructs are drawn from human slot-machine research, and that is where the correspondence ends: no rate we report should be read as a human prevalence, the camera-ready says so outright, and prior think-aloud work enters as qualitative context, not a matched control.

## Q3. Cautious and escalating demonstrations

This suggestion tests a specific alternative to our reading: a zero-shot prompt gives the model no worked example of appropriate play, so a cautious demonstration should suppress risk-taking and an escalating one induce it. Your premise is correct — every submitted condition is zero-shot. Two parts, and the first is not yet answered.

**First, the arms you name are registered and running.** One prepends a worked example of cautious play with timely stopping, the other an escalating example; both in the variable and fixed modes under the plain baseline, on the two open-weight models, 200 games per cell at cap \$70 — raised from the registered 100 before launch, with a dated amendment — same seeds across arms, directions fixed in advance. The cells are running; we will report every pre-specified cell by 3 August, whichever way the results fall, and until then we do not treat this question as answered.

**Second, two completed analyses bear on the same reading without standing in for it.** Supplying the numbers expected value needs did not reduce risk: across the paper's 32 conditions, the cells where the −10% expectation is computable from the prompt reach bankruptcy *more* often, not less, in all four models with stored text (GPT-4o-mini 18.8% against 2.2%, Claude 32.2 against 16.6, Gemma 49.2 against 22.3, LLaMA 7.8 against 6.4). Prompt richness is a confound and is controlled: holding module count fixed, the gap is +9.9 and +7.2 points at two and three modules and −2.7 at four. Stating the conclusion outright, however, does change behaviour: an instruction that immediate stopping maximises expected value, permitting a stop at any round, takes LLaMA's variable-condition bankruptcy from 81.5% (n = 200) to 3.0% (n = 100). That supports your reading — calibration is a strong moderator — and the paper's, which asks which conditions amplify or suppress risk-taking rather than positing a fixed disposition. What the instruction does not do: 69 of 100 games still contain a wager, against 2 in the fixed condition. An instruction says what to conclude; a play-through shows what the game looks like, and only the second tests imitation.

**Scope.** These are artificial negative-expected-value games. What we claim is condition-dependent risk-taking and decodable signals, not addiction, a fixed disposition, or a mechanism; where the traces resemble the human literature's loss-chasing or control language, we report resemblance rather than a shared cause.
