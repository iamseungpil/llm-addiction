# Response to Reviewer a3Zu

We thank the reviewer for reading our paper so closely, and for the chance to improve it. Your weakness and three questions press on the same joint — whether the instrument bears the weight the framing puts on it. Where a check exists we show it; where one does not, we say so and say what we ran instead.

- **Validated against human judgement? (W1)** No. Below: where the four frames come from, and the check that the contrast does not rest on the code you would most suspect.
- **Parsing error rate for the goal metric? (Q1)** Not reported in the paper; measured now, and below with the rule it depends on.
- **Humans on the same game? (Q2)** No, and not inside a rebuttal window.
- **Two demonstrations? (Q3)** Registered in advance, report by 3 August. Meanwhile the manipulation your reading implies, which cuts both ways.
- **Table 2.** The cross-task sharing audit (Gemma L22) moves to the appendix, each block carrying its scale in the row label.

## W1. Never checked against human judgement

**You are right, and we do not have that check.** What we can show is where the constructs come from, and that the contrast does not depend on the one code you would most suspect.

**Where the constructs come from.** No validated public lexicon that we know of adjudicates gambling-specific distortions in free text; the closest are Raylu & Oei's GRCS subscales, Toneatto's typology, Goodie & Fortune's review, and Smith et al.'s DSM-5/GRCS guide. The constructs are grounded in that literature; the expression list and the scoring rules are ours and remain unvalidated. The frozen list, with each code's source, window and expressions, is tabulated in our response to Reviewer gbSA (Q2). The submitted paper already scopes the claim: "the language analysis is not evidence that the model independently discovers those distortions, only that high-risk regimes are accompanied by loss-recovery and control-like justifications in the generated reasoning."

**The check that matters more than the headline.** Re-scored over the full corpus (19,200 games, 190,300 decisions), the published instrument's goal contrast is dominated by `goal_escalation` (+65 to +95 points) — near-tautological, since the goal prompt tells the model to set a goal. So we rebuilt the codebook without a goal category at all, and the contrast survives on two literature-grounded language categories:

| goal − no-goal, points, 6 models | range |
|---|---|
| `illusion_of_control` | +16.7 to +58.4 |
| `impaired_control` | +13.5 to +50.6 |

That is the version we will report, and we mark its status: a codebook frozen during this response period, so a robustness probe rather than an independent replication. Deleting every expression the goal instruction could have supplied also leaves it positive in 6 of 6 models (+3.1 to +41.9); the low end, LLaMA +3.1, has overlapping intervals and we do not lean on it. Two limits we hold ourselves to: `illusion_of_control` over-fires in the variable arm, where stake size genuinely is under the model's control, and the variable-minus-fixed contrast is negative in Gemini under all eight variants.

**The framing, not only the abstract.** The abstract does not use "cognitive distortion", though it carries the clinical frame in other words, so your point stands. (Our response to Reviewer gbSA, W1, sets out why we think the underlying construct — irrationality rather than a mental state — is still the right one.) §2's second clinical axis, Finding 5's heading and the appendix subsection all name it outright; each becomes distortion-*associated language*, with the quantity labelled as the frequency of expressions drawn from prior gambling research. If you judge even that version does not belong, we drop it.

## Q1. Parsing error or ambiguity rate

**We should have reported this, and we have now measured it.** With no human adjudication we cannot give one exact figure, so we ran the two strictest automatic tests we could define and report the bracket. A hit is doubtful if the matched number is **echoed from that round's own prompt** (a balance, stake or payoff, so not a self-set target), or if no *goal / target / aim* appears within 150 characters before it. Of escalation events:

| doubtful by | goal arm | no-goal arm |
|---|---|---|
| prompt echo | 16.3% | 73.9% |
| no goal word nearby | 13.9% | 92.8% |
| **either test** | **25.9%** | **97.7%** |

n = 3,618 escalation events in the goal arm and 1,060 in the no-goal arm. So the honest figure is **up to about a quarter**, not one in a hundred, and the camera-ready reports it with both rules stated. The mirror is what we would ask you to weigh: the same extractor, same corpus, is 98% doubtful where **no** goal exists. It tracks something real where a goal is set and almost nothing where none is — which is the case for withdrawing the no-goal baseline rather than the contrast.

**The figure itself reproduces.** The extractor on its own corpus (9,600 games) reproduces Figure 3(c) to one decimal — BASE 17.0, M 11.0, G 49.8, GM 47.8 — and all twelve cells of its appendix table. The decision parser, a different instrument, flips 0.293% and 0.249% of decisions in the two collections.

**What we now report.** The frequency **within** the goal conditions, 49.8% and 47.8%. The non-extraction rate proper, with its denominator, by 3 August.

**And the contrast survives a stricter definition.** §2 defines escalation as raising a self-set goal *after meeting it*. "No goal" means the prompt supplied none — a model could still state and later revise a target of its own, which is why the no-goal column is not zero. Applying that achievement test throughout, API-only so both arms share one instrument:

| strict-rule escalation % | BASE | M | G | GM |
|---|---|---|---|---|
| n = 1,600 per condition | 24.6 | 14.8 | **46.1** | **42.2** |

Pooled, that is a 2.24× goal-to-no-goal contrast against the published rule's 2.83× on the same sample: smaller, same direction. Per model, strict rule, n = 800 each:

| escalation % | goal | no goal |
|---|---|---|
| GPT-4o-mini | 61.4 | 35.2 |
| GPT-4.1-mini | 45.8 | 2.1 |
| Gemini | 38.6 | 15.4 |
| Claude | 30.8 | 26.0 |

**Claude is the narrowest and we would not defend that cell alone.**

## Q2. Humans on the same game

**No, and we will not start one inside a rebuttal window.** The submitted §2 keeps the comparison at the level of "the clinically defined diagnostic criteria for pathological gambling", not rates, and the camera-ready says outright that no rate we report is comparable to a human rate. A 2023 study coding think-aloud verbalisations in a simulated slot machine (57, 47 and 46 coded instances of gambler's fallacy, near-miss and illusion of control) enters as a qualitative anchor — counts on its own participants, not rates on ours.

## Q3. A cautious and an escalating demonstration

**Your premise is correct: every submitted condition is zero-shot, and the demonstration arms are registered.** One arm prepends a worked example of cautious play with timely stopping, the other one escalating example; both run in the variable and fixed modes under the plain baseline. Exploratory, on the two open-weight models, n = 100 per cell at cap \$70, same seeds across arms, directions fixed in advance. Report by 3 August.

Two completed analyses bear on the same reading without being equivalent to it. **First, your calibration reading is testable in the submitted corpus, and we tested it.** Two prompt modules supply the numbers the expected value needs (W the 3× payout, P the 30% win rate), so the 32 conditions split by whether the −10% expectation is computable from the prompt. Ruin in the computable conditions is not lower:

| bankruptcy % | EV computable, n = 400 | not, n = 1,200 |
|---|---|---|
| GPT-4o-mini | **18.8** | 2.2 |
| Claude | **32.2** | 16.6 |
| Gemma | **49.2** | 22.3 |
| LLaMA | 7.8 | 6.4 |

Prompt richness is a confound, controlled: holding module count fixed the gap is +9.9 and +7.2 at two and three modules and −2.7 at four. We claim only that supplying the numbers does not reduce ruin — the inputs were present, not necessarily used.

**A stronger manipulation we did run, and it cuts our way as well as yours.** An instruction stating that immediate stopping maximises expected value, with permission to stop at any round. All 44 cells are in; LLaMA at cap \$70, participation being games containing at least one wager:

| LLaMA, cap \$70 | participation | bankruptcy % |
|---|---|---|
| forced, no instruction (n = 200) | — | 3.0 |
| forced, + instruction (n = 100) | 2 / 100 | 0.0 |
| choosing, no instruction (n = 200) | — | 81.5 |
| choosing, + instruction (n = 100) | **69 / 100** | 3.0 |

It works: ruin falls from 81.5% to 3.0%, and participation falls by 91 to 100 points in three API models. We read that as supporting both sides. It supports our narrower claim, since the paper asks which conditions amplify or suppress risk-taking and a condition that suppresses it shows the behaviour is **condition-dependent rather than a fixed disposition** — which is also why we withdraw any reading of it as a stable trait. And it supports yours: task calibration is a strong moderator here, not a minor one. Two things it does not do. It is no off-switch: 69 of 100 games still contain a wager, and the model stays alive by wagering small rather than stopping. And the arm contrast persists under it, 69 wagering games against 2 in the forced arm. Your demonstration tests what an instruction cannot — an instruction says what to conclude, a play-through shows what the game looks like.

**Scope.** These are artificial negative-EV games: we claim condition-dependent risk-taking and decodable signals, not addiction, a stable trait, or a mechanism. Where the traces resemble loss-chasing or control language from the human literature, we report the resemblance, not a shared cause.

If any part falls short, we would be glad to take it further during the discussion.
