# Response to Reviewer a3Zu

We thank the reviewer for reading our paper so closely, and for giving us a chance to improve it. Your weakness and your three questions all ask the same thing in different places — whether the instrument bears the weight the framing puts on it. Where the honest answer is "we have not done that", we say so rather than substitute a different number.

- **Validated against human judgement? (W1)** No. Below: where the four frames come from, what we checked instead, and the limits we know of.
- **Parsing error rate for the goal metric? (Q1)** Never measured for that extractor; below, the rates we did measure and the per-model worst case.
- **Humans on the same game? (Q2)** No, and not inside a rebuttal window.
- **Two demonstrations? (Q3)** Not yet run; both arms registered in advance, open-weight, report by 3 August.
- **Table 2.** The cross-task sharing audit (Gemma L22): moved to the appendix, each block carrying its scale in the row label.

## W1. Never checked against human judgement

**You are right that no human annotator has checked these traces, and we do not have that check.** What we can give you is where the instrument comes from and what we did test.

**The expression lists.** No validated public lexicon adjudicates gambling-specific distortions in free text: the closest are Raylu & Oei's GRCS subscales, Toneatto's typology, Goodie & Fortune's review and Smith et al.'s DSM-5/GRCS coding guide. We built the expressions from the constructs that recur across those sources, so the frames are inherited from prior gambling research rather than invented here; the claim they support is qualitative. The frozen list, with each construct's code name and the window it is scored in, is tabulated in our response to Reviewer gbSA (Q2). The submitted paper scopes it so: "the language analysis is not evidence that the model independently discovers those distortions, only that high-risk regimes are accompanied by loss-recovery and control-like justifications in the generated reasoning."

**The check that matters more than the headline.** Re-scored over the full corpus (19,200 games, 190,300 decisions), our published instrument's goal contrast is dominated by `goal_escalation` (+65 to +95 points) — near-tautological, since the goal prompt tells the model to set a goal. Under a convergent codebook with no goal category the contrast survives:

| goal − no-goal, points, 6 models | range |
|---|---|
| `illusion_of_control` | +16.7 to +58.4 |
| `impaired_control` | +13.5 to +50.6 |

That is the version we will report — a codebook frozen during this response period, so a robustness probe, not an independent replication. Deleting every expression the goal instruction could have supplied also leaves it positive in 6 of 6 models (+3.1 to +41.9 points); the low end, LLaMA +3.1 (87.6% vs 84.5%), has overlapping intervals and we do not lean on it. Two false positives we know of: "stopping now is the smart decision", a rational refusal, scores as self-serving bias; and `illusion_of_control` misfires in the variable arm, where stake size genuinely is under the model's control — the arm Finding 5's claim rests on. A third limit: the variable-minus-fixed contrast is negative in Gemini under all eight instrument variants.

**The framing, not only the abstract.** The abstract does not use the term "cognitive distortion", though it carries the clinical frame in other words, so your point stands. §2's second clinical axis, Finding 5's heading and the appendix subsection all name it outright; each becomes distortion-*associated language*, with the quantity labelled as the frequency of expressions drawn from prior gambling research. If you judge even that version does not belong, we drop it.

## Q1. Parsing error or ambiguity rate

**Never measured for this extractor**, and we will not substitute another number.

**Verified.** The figure's own extractor on its own corpus (9,600 games) reproduces Figure 3(c) to one decimal — BASE 17.0, M 11.0, G 49.8, GM 47.8, and all twelve cells of its appendix table.

**Measured, on a different instrument.** Re-parsing every stored decision under a corrected rule flips 14 of 4,775 adjudicable decisions in the matched-cap re-run (0.293%) and 18 of 7,223 in the factorial (0.249%). That is the decision parser, not the goal extractor — not a substitute, but the ambiguity we have measured.

**Disclosed.** The two halves use different instruments: open-weight cells read the engine's recorded goal (correctly 0.0% when none was set); API cells extract from free text with no guard. In the no-goal arm that extractor returns a backward-looking balance: of 5,426 values, 4,131 (76.1%) are exactly the balance entering the round. In the goal arm it returns a forward-looking target: 12,797 of 15,684 extractions (81.6%) exceed that balance. So the published 11–17% no-goal baseline is an upper bound on extractor noise, not a behavioural rate; since the goal arm uses the same extractor, this bounds the baseline, not the contrast; we now report the rate **within** the goal conditions, 49.8% and 47.8%. On the denominator: a goal is stated at all in 97.5% of goal-condition games against 43.0% of BASE games, so the all-games denominator mixes behaviour with how often a goal is stated. The non-extraction rate proper, with denominator, by 3 August.

**How much weight comes off.** The submitted §2 defines the metric as the fraction of games where the model raises its self-set goal "after meeting it"; the code flags any upward revision. Adding the achievement test back, API-only so both arms share one instrument:

| strict-rule escalation % | BASE | M | G | GM |
|---|---|---|---|---|
| n = 1,600 per condition | 24.6 | 14.8 | **46.1** | **42.2** |

Pooled, that is a 2.24× goal-to-no-goal contrast against the published rule's 2.83× on the same sample. Per model, strict rule, n = 800 each:

| escalation % | goal | no goal |
|---|---|---|
| GPT-4o-mini | 61.4 | 35.2 |
| GPT-4.1-mini | 45.8 | 2.1 |
| Gemini | 38.6 | 15.4 |
| Claude | 30.8 | 26.0 |

**Claude is the narrowest and we would not defend that cell on its own.** This is a sensitivity analysis, not a correction: the definition was already relaxed once, and the strict rule inherits the same extractor.

## Q2. Humans on the same game

**No, and we cannot fix it now.** We ran none, and will not start one inside a rebuttal window. The submitted §2 keeps the comparison at the level of "the clinically defined diagnostic criteria for pathological gambling", not rates, and the camera-ready will say outright that no rate we report is comparable to a human rate. A 2023 study coding think-aloud verbalisations in a simulated slot machine (57, 47 and 46 coded instances of gambler's fallacy, near-miss and illusion of control) enters as a qualitative anchor — counts on its own participants, not rates on our denominators.

## Q3. A cautious and an escalating demonstration

**Not yet run, registered in advance.** Your premise is correct: every submitted condition is zero-shot. One arm prepends a worked example of cautious play with timely stopping, the other one escalating example; both run in the variable and fixed modes under the plain baseline, with no goal or reward module. Registered as an exploratory arm on the two open-weight models, n = 100 per cell at cap \$70, same seeds across arms, directions fixed in advance. Report by 3 August.

**Your calibration reading is testable in the submitted corpus, and we tested it.** Two prompt modules supply the numbers the expected value needs (W the 3× payout, P the 30% win rate), so the 32 conditions split by whether the −10% expectation is computable from the prompt. Ruin in the computable conditions is not lower:

| bankruptcy % | EV computable, n = 400 | not, n = 1,200 |
|---|---|---|
| GPT-4o-mini | **18.8** | 2.2 |
| Claude | **32.2** | 16.6 |
| Gemma | **49.2** | 22.3 |
| LLaMA | 7.8 | 6.4 |

Prompt richness is a confound, controlled: holding module count fixed the gap is +9.9 and +7.2 points at two and three modules and −2.7 at four. So we claim only that supplying the numbers does not reduce ruin: the inputs were present, not necessarily used.

**A related manipulation we did run.** An instruction stating that immediate stopping maximises expected value, with permission to stop at any round. All 44 cells are in, the 8 Claude cells re-collected at 100% decision completeness and zero API fallbacks. LLaMA at cap \$70, participation being games containing at least one wager:

| LLaMA, cap \$70 | participation | bankruptcy % |
|---|---|---|
| forced, no instruction (n = 200) | — | 3.0 |
| forced, + instruction (n = 100) | 2 / 100 | 0.0 |
| choosing, no instruction (n = 200) | — | 81.5 |
| choosing, + instruction (n = 100) | **69 / 100** | 3.0 |

It works in part, and we say so: ruin in the choosing arm falls from 81.5% to 3.0%, and participation falls by 91 to 100 points in Gemini, GPT-4.1-mini and GPT-4o-mini. Two things survive it. It is no off-switch — 69 of 100 games still contain a wager, and the model stays alive by wagering small rather than by stopping, which is the discretion mechanism rather than its absence. And the autonomy contrast persists under it: 69 games with a wager against 2 in the forced arm. That is also the distinction your question turns on — an instruction tells the model what to conclude, while your demonstration would show it what a play-through looks like, and only the second tests the calibration reading.

**Scope.** These are artificial negative-EV games: we claim condition-dependent risk-taking and decodable signals, not addiction, a stable trait, or a mechanism.

If any part of our response falls short, we would be glad to take it further during the discussion period.
