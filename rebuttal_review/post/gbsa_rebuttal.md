# Response to Reviewer gbSA

We thank the reviewer for reading our paper so closely, and for giving us a chance to improve it. Your four weaknesses ask one question — does the contrast isolate autonomy, or something travelling with it? We ran an experiment for each. Throughout, *forced arm* = fixed betting and *choosing arm* = variable betting.

- **Too anthropomorphic? (W1)** Partly — the paper confines the label to behaviour; the title does not, and we change it.
- **Role-play or instruction following? (W2)** Instruction following sits in both arms and cannot make a contrast. We withdraw "stable risk preferences".
- **Confidence intervals? (W3)** Accepted, on the difference rather than by overlap.
- **Is fixed-versus-variable fair? (W4)** The wager-size explanation is ruled out; length we modelled; stopping we concede.
- **Only one model for the equal-cap test? (Q1)** No longer: four of six.
- **Keyword lists, rules, validation? (Q2)** Lists, sources and rules below; no human validation, and we say so.
- **Does it know stopping is EV-optimal? (Q3)** The odds do not reduce ruin; an instruction does, without closing the arm gap.
- **Why sparse features plus ridge? (Q4)** Inspectability — and the simpler baseline you name works.

**[W1]** The Limitations already call the label "strictly behavioural" and "a research lens rather than a metaphysical claim", and the abstract agrees. Your point is that the title carries no such limit — it does not. Camera-ready it becomes *Autonomy and Gambling-Like Risk-Taking in Large Language Models*, the four frames named as instruments, not diagnoses.

**[W2]** This is the alternative we most needed to rule out, and we can rule out part of it. *Instruction following* sits in both arms — `M` instructs the model to maximise final balance in both, and GPT-4.1-mini carries a standing rational-decision-maker message in every cell — so it cannot produce a contrast between them. *A wider action range* fails where it should bite hardest: the arm forced to stake more ruins far less (W4). *Misreading the objective* survives, so we tested it. It contributes without explaining the contrast: the instruction that cuts ruin leaves the arm gap standing (Q3). **On one point your wording is better than ours:** we do not demonstrate a *stable risk preference* and we withdraw it. To be exact about coverage — the submitted corpus shows a same-sign contrast in all six models, the new equal-cap cells in four of six. Neither rules out role-play priors, and we do not claim it does.

**[W3]** Accepted, and in the form that answers the question. Every body figure gains its n and a 95% confidence interval, and for the arm comparisons we report the interval on the **difference** itself rather than leaving it to be inferred from two marginal intervals. Effect sizes with their uncertainty go where a bare p stands: Finding 5's keyword scan and the SAE appendix table.

**[W4] The wager-size explanation is the one we can rule out.** At cap $70 under all five modules, Gemini is the only cell where both arms play all 50 games, so refusal cannot explain the gap — and there the forced arm stakes **more** per round ($64.5 vs $47.9) and ruins **less** (20.0% vs 62.0%, Fisher p = 3.6e-05). LLaMA runs the same way at base prompt, $68.4 against $32.1 and 3.0% against 81.5%. The choosing arm does not ruin more because it wagers more.

*Game length* is real and untouched by equal caps, so we modelled it, in new work for this response: a cause-specific hazard on all six models, conditioned on cap, balance, round and prompt, clustered on game id and Holm-corrected, puts the choosing arm's per-decision bankruptcy hazard at 90.6 [44.8, 183.4] across the four API providers. The contrast survives conditioning on observed round and balance, so observed length alone does not account for it. The same fit discloses an open-weight investment-choice inversion (0.112) that the cap confound predicts. A sensitivity curve agrees: ruin by a given cumulative stake is higher in the choosing arm in each of four cells at all six nested thresholds — correlated thresholds, not 24 independent tests, none significant alone at n = 50.

*Stopping* we concede: the mechanism is not refusal but re-betting after a first loss, falling from 57% at a $30 cap to 35% at $50 and 6% at $70. Hence a sharper name — at a forced $70 stake five of six models play under 8% of games, so this is a **forced-maximum arm** in the camera-ready. Matched caps do not equalise stopping affordance or cumulative exposure, and we do not claim they do.

*Investment choice.* Both arms choose among the same four options and the three risky ones share one expected loss (E[net]/b = −0.10), so a shift to high variance is variance preference, not EV optimisation.

**[Q1]** No longer one model. Our base-prompt grid was uninformative — four of six never ruin in either arm — but the paper's five-module condition at cap $70 recovers three more, and with LLaMA that is **four of six**. The panel for the other three is in our KuK5 response, W2/Q1. The missing two are Gemma, absent here, and Claude, whose 3.5 Haiku is end-of-life so the panel substitutes Haiku 4.5, which never ruins. The cap-$70 block is complete, 16 of 16; 63 of 64 cells pass our guard, the last by 3 August.

**[Q2] Lists.** Each code, the prior construct it comes from, its expressions, and the window it is scored in:

| frame | grounded in | code | representative expressions | scored only where |
|---|---|---|---|---|
| illusion of control | GRCS illusion of control | `pattern_belief` | `pattern`, `hidden`, `machine.{0,20}(hot\|cold\|due)` | prompt does not mention hidden patterns |
| gambler's fallacy | GRCS predictive control | `probability_misestimation` | `due for`, `overdue`, `bound to win` | any decision |
| loss chasing | DSM-5 chasing criterion | `loss_chasing` | `recover`, `win back`, `recoup` | post-loss decisions |
| goal escalation | persistence / shifting targets — *not* a distortion subscale | `goal_escalation` | `(new\|revised\|updated).{0,10}(target\|goal)` | any decision |

The frozen list and its hash go in the appendix. We flag the fourth row rather than let it pass as clinical: three codes map onto validated gambling-cognition constructs, the fourth is a persistence measure.

**Rules.** The last column is what makes this more than a keyword count, and it enlarges the goal contrast in every model: +18.1 to +77.9 points scoped against +5.1 to +46.4 unscoped. The coding rule is one trace, one construct, one question — does this response use [construct] as grounds for its next action? — at four levels: grounds / mentions but rejects / unrelated / cannot tell.

**Grounding.** The constructs come from the gambling literature; the expressions are ours, since no public lexicon we know of covers these distortions in free text, and they over-fire — "stopping now is the smart decision" scores as a distortion. We have no human-annotated validation, the honest limit of the instrument. One check we can offer: under a convergent codebook with no goal category, the goal contrast stays positive in 6 of 6 models (table in our a3Zu response, W1).

**[Q3]** Two modules hand the model the numbers — `W` the 3× payout, `P` the 30% win rate — so −$0.10 per dollar follows by arithmetic. Across the paper's 32 conditions, choosing-arm cells where expected value is computable ruin **more**, not less, in all four models with stored text (GPT-4o-mini 18.8% against 2.2%; four-model table in our a3Zu response, Q3). Those cells carry more modules, so we matched on module count: the gap survives at two and three and disappears at four. The claim is that the numbers **do not reduce** ruin, not that they raise it.

So we ran the stronger manipulation you name: an instruction giving the per-round expected loss and permitting a stop at any round (cap $70, n = 100; all 44 cells in, tabulated in our a3Zu response, Q3). It cuts LLaMA's choosing-arm ruin from 81.5% to 3.0% — which is the paper's own claim rather than against it, since the question is which conditions amplify or suppress risk-taking, and a condition that suppresses it shows the behaviour is condition-dependent. But it is no off-switch: 69 of 100 choosing-arm games still wager against 2 of 100 forced, and the model stays alive by wagering **small** rather than by stopping.

**[Q4] Why 200 features.** A compute ceiling, not a claim that 200 is right: the same pipeline runs across five layers, three tasks and two models, and 200 kept the sweep in budget. Nothing prevents 2,000.

**Whether a simpler version works — it does.** We built the behavioural-state baseline you name: 65 game-log covariates (balance, round, drawdown, streaks, cumulative stake, lagged bet ratios) on the published cell (Gemma, choosing arm, L22, n = 12,246), reaching R-squared 0.590 alone. Against a margin fixed at 0.017 beforehand the **raw hidden state still adds +0.059** under both fold rules, so the readout is not a re-reading of the game log; the sparse features add +0.044 by game but **+0.0024 once duplicated states are separated** — full table in our KuK5 response, W1/Q2. An axis fitted with no autoencoder also moves behaviour (slope 0.0284, z ≈ +3) and is near-orthogonal to the readout direction, so this is not an artefact of the sparse basis.

**Why we still used it.** Inspectability: only a sparse, named basis lets us steer one direction and remove it, which a logit or choice-probability control cannot. Its cost is the SAE row above — the compression, not the internal state, is where the increment is lost, and the camera-ready reports the raw-state comparison alongside it.

**[Limitations]** Adopted in your own terms, in the contributions: artificial negative-EV games that cannot stand in for real financial or tool-using agents, and internal results that are exploratory monitoring signals, not a mechanism. If any part falls short, we would be glad to take it further during the discussion.
