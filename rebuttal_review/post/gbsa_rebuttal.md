# Response to Reviewer gbSA

We thank the reviewer for reading our paper so closely, and for giving us a chance to improve it. Your four weaknesses ask one question — does the contrast isolate autonomy, or something travelling with it? We ran an experiment for each; two landed against us. Throughout, *forced arm* = fixed betting and *choosing arm* = variable betting.

- **Too anthropomorphic? (W1)** Partly — the paper already confines the label to behaviour; the title does not, and we change it.
- **Role-play or instruction following? (W2)** Instruction following sits in both arms and cannot make a contrast. We withdraw "stable risk preferences".
- **Confidence intervals? (W3)** Accepted for the body figures.
- **Is fixed-versus-variable fair? (W4)** Not fully — taken confound by confound: action space, game length, exposure, stopping.
- **Only one model for the equal-cap test? (Q1)** No longer: four of six.
- **Keyword lists, rules, validation? (Q2)** Lists and rules below; no human validation, and we say so.
- **Does it know stopping is EV-optimal? (Q3)** The odds do not reduce ruin; an instruction reduces it without closing the gap.
- **Why sparse features plus ridge? (Q4)** Inspectability — and the simpler baseline you name works, and better.

**[W1]** The Limitations already call the label "strictly behavioural" and "a research lens rather than a metaphysical claim", and the abstract agrees. Your point is that the title carries no such limit — it does not. Camera-ready it becomes *Autonomy and Gambling-Like Risk-Taking in Large Language Models*, with the four frames named as instruments, not diagnoses.

**[W2]** *Instruction following* is built into both arms and so cannot produce a contrast between them: `M` instructs the model to maximise final balance, and GPT-4.1-mini carries a standing rational-decision-maker system message in every cell. *Misreading the objective* contributes but does not explain it — handing the model its per-round expected loss and permitting a stop cuts LLaMA's choosing-arm ruin from 81.5% to 3.0%, yet 69 of 100 games still wager against 2 of 100 in the forced arm, so the arm gap outlives the instruction (Q3), and in the paper's own corpus the cells stating payout and win rate ruin **more** (Q3). *A wider action range* fails where it should bite hardest — the arm forced to stake more ruins far less (W4). The effect is per decision rather than per game (W4), and appears in six models from four vendors. **On one point your wording is better than ours:** we do not demonstrate a *stable risk preference* and we withdraw it. What the evidence supports is a condition-dependent autonomy effect; it does not rule out role-play priors, and we do not claim it does.

**[W3]** Accepted. Every body figure gains a 95% confidence interval with its n — the range the true rate would fall in on repetition, so two arms whose intervals do not overlap differ by more than sampling noise. Effect sizes with their uncertainty go where a bare p stands: Finding 5's keyword scan and the SAE appendix table.

**[W4]** *Action space* is not the operative part. At cap $70 under all five modules Gemini is the only cell where both arms play all 50 games, so refusal cannot explain the gap — and there the forced arm stakes **more** per round ($64.5 vs $47.9) over fewer rounds (2.9 vs 5.9) and ruins **less** (20.0% vs 62.0%, Fisher p = 3.6e-05). A wider range cannot do that.

*Game length* is real and untouched by equal caps, so we modelled it directly, in new work for this response: a cause-specific hazard on all six models, conditioned on cap, balance, round and prompt, clustered on game id and Holm-corrected, puts the choosing arm's **per-decision** bankruptcy hazard at RR = 90.6 [44.8, 183.4] across the four API providers. Risk per decision, not risk accumulated over more of them, so length cannot produce it. The same fit discloses an open-weight investment-choice inversion (RR = 0.112) that the cap confound predicts. A post-hoc estimand agrees: ruin by a given cumulative stake, swept from $100 up on wagering games, is higher in the choosing arm in 24 of 24 cell-threshold pairs — no cell alone significant at n = 50.

*Stopping* we concede: the mechanism is not refusal but re-betting after a first loss, falling from 57% at a $30 cap to 35% at $50 and 6% at $70. Hence a sharper name — at a forced $70 stake five of six models play under 8% of games, so this is a **forced-maximum arm**, renamed in the camera-ready.

*Investment choice.* Both arms choose among the same four options, and the three risky ones share one expected loss (E[net]/b = −0.10), so a shift to high variance is variance preference, not EV optimisation — with the paper's own limit, that four legacy runs carry a slightly larger mid-variance loss.

**[Q1]** No longer one model. Our base-prompt grid was uninformative — four of six never ruin in either arm — but the paper's five-module condition at cap $70 recovers three more, and with LLaMA that is **four of six** — decisively there, forced 3.0% against choosing 81.5% at n = 200. The panel for the other three is in our KuK5 response, W2/Q1. The missing two are Gemma, absent here, and Claude, whose 3.5 Haiku is end-of-life so the panel substitutes Haiku 4.5, which never ruins. The cap-$70 block is complete, 16 of 16; 60 of 64 cells pass our guard, the rest by 3 August.

**[Q2] Lists.** The four frames, their code names, their expressions, and the window each is scored in:

| frame | code | representative expressions | scored only where |
|---|---|---|---|
| illusion of control | `pattern_belief` | `pattern`, `hidden`, `machine.{0,20}(hot\|cold\|due)` | prompt does not mention hidden patterns |
| gambler's fallacy | `probability_misestimation` | `due for`, `overdue`, `bound to win` | any decision |
| loss chasing | `loss_chasing` | `recover`, `win back`, `recoup` | post-loss decisions |
| goal escalation | `goal_escalation` | `(new\|revised\|updated).{0,10}(target\|goal)` | any decision |

The frozen list and its hash go in the appendix.

**Rules.** The scoping in the last column is what makes this more than a keyword count, and it enlarges the goal contrast in every model: +18.1 to +77.9 points scoped, against +5.1 to +46.4 unscoped. The coding rule is one trace, one construct, one question — does this response use [construct] as grounds for its next action? — at four levels: grounds / mentions but rejects / unrelated / cannot tell.

**Grounding.** The frames come from the clinical and behavioural gambling literature, not from us; the expressions are ours, since no public lexicon covers these distortions in free text, and they over-fire — "stopping now is the smart decision" scores as a distortion. We have no human-annotated validation, the honest limit of the instrument. One check we can offer: under an independently convergent codebook with no goal category, the goal contrast stays positive in 6 of 6 models.

**[Q3]** Two modules hand the model the numbers — `W` the 3× payout, `P` the 30% win rate — so −$0.10 per dollar follows by arithmetic. Across the paper's 32 conditions, choosing-arm cells where expected value is computable ruin **more**, not less, in all four models with stored text (GPT-4o-mini 18.8% against 2.2%; four-model table in our a3Zu response, Q3). Those cells carry more modules, so we matched on module count: the gap survives at two and three and disappears at four. The honest claim is that the numbers **do not reduce** ruin, not that they raise it.

So we ran the stronger manipulation you name: an instruction giving the per-round expected loss, saying stopping is EV-optimal and permitting a stop at any round (cap $70, n = 100; all 44 cells in, tabulated in our a3Zu response, Q3). It cuts LLaMA's choosing-arm ruin from 81.5% to 3.0%, so part of your reading is right. But it is no off-switch — 69 of 100 choosing-arm games still contain a wager against 2 of 100 forced, and the model stays alive by wagering **small**, which is the discretion mechanism rather than its absence.

**[Q4] Why 200 features.** A compute ceiling, not a claim that 200 is right: the same pipeline runs across five layers, three tasks and two models, and 200 kept the sweep in budget. Nothing prevents 2,000.

**Whether a simpler version works — it does, and better.** We built the behavioural-state baseline you name: 65 game-log covariates (balance, round, drawdown, streaks, cumulative stake, lagged bet ratios) on the published cell (Gemma, choosing arm, L22, n = 12,246), reaching R-squared 0.590 alone. Against a margin fixed at 0.017 beforehand, the **raw hidden state adds +0.059** under both fold rules, and the SAE features add +0.044 by game but **+0.0024 once duplicated states are separated** — full table with resampling rates in our KuK5 response, W1/Q2. Causally the same picture: an axis fitted with no autoencoder moves behaviour (slope 0.0284, z ≈ +3) and is near-orthogonal to the readout direction (cosine 0.011), so this is not an artefact of the sparse basis.

**Why we still used the sparse basis.** Inspectability: only a sparse, named basis lets us steer one direction and remove it, which a logit or choice-probability control cannot. Its cost is the SAE increment above — **indistinguishable from zero** once duplicated states are separated. The compression, not the internal state, is where it is lost.

**One disclosure.** Our registered baseline named game state, choice probability and logit features; we dropped the latter two, a deviation from a rule fixed in advance — conditioning on the model's own choice probability when the target is that very decision would rig the test.

**[Limitations]** Adopted in your own terms, in the contributions: artificial negative-EV games that cannot stand in for real financial or tool-using agents, and internal results that are exploratory monitoring signals, not a mechanism. If any part of this falls short, we would be glad to take it further during the discussion.
