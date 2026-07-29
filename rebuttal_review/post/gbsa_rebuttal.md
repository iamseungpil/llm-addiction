# Response to Reviewer gbSA

We thank the reviewer for reading our paper so closely, and for the chance to improve it. Your four weaknesses ask one question — does the contrast isolate autonomy, or something travelling with it? We ran an experiment for each. Throughout, *forced arm* = fixed betting, *choosing arm* = variable betting.

- **Too anthropomorphic? (W1)** The paper confines the label to behaviour and measures irrationality; the title does not, and we change it.
- **Role-play or instruction following? (W2)** Instruction following sits in both arms and cannot make a contrast. We withdraw "stable risk preferences".
- **Confidence intervals? (W3)** Accepted, on the difference rather than by overlap.
- **Is fixed-versus-variable fair? (W4)** The wager-size explanation is ruled out; length we modelled; stopping we concede.
- **Only one model for the equal-cap test? (Q1)** No longer: four of six.
- **Keyword lists, rules, validation? (Q2)** Lists, sources and rules below; no human validation.
- **Does it know stopping is EV-optimal? (Q3)** The odds do not reduce ruin; an instruction does, without closing the arm gap.
- **Why sparse features plus ridge? (Q4)** Inspectability — and the simpler baseline works.

**[W1]** We accept the title change. But weigh what the paper already does with the label. The Limitations call the descriptor "strictly behavioural", making "no claim about subjective experience, suffering, or moral status of the model" and reading as "a research lens rather than a metaphysical claim". What it measures is **irrationality**: §3 studies "the two core components of irrationality — self-regulation failure and cognitive distortions", §4 reads round-level irrationality indicators, and the Conclusion places it "near goal-misgeneralisation and reward hacking as a behavioural relative". The surrounding literature has the same shape: Anthropic's agentic-misalignment study reports behaviour contingent on conditions rather than intrinsic to the model, and Bengio et al. (*Science*, 2024) treat autonomy as the variable eroding oversight. We add which conditions produce it in a repeated-decision setting, and that it has a human analogue worth naming. The title carries none of that limit, so camera-ready it becomes *Autonomy and Gambling-Like Risk-Taking in Large Language Models*, the four frames named as instruments, not diagnoses.

**[W2]** This is the alternative we most needed to rule out, and we can rule out part of it. *Instruction following* sits in both arms — `M` instructs the model to maximise final balance in both, and GPT-4.1-mini carries a standing rational-decision-maker message in every cell — so it cannot make a contrast between them. *A wider action range* fails where it should bite hardest: the arm forced to stake more ruins far less (W4). *Misreading the objective* survives, so we tested it; it contributes without explaining the contrast, since the instruction that cuts ruin leaves the arm gap standing (Q3). **On one point your wording is better than ours:** we do not demonstrate a *stable risk preference* and we withdraw it. On coverage: the submitted corpus shows a same-sign contrast in all six models, the new equal-cap cells in four of six. Neither rules out role-play priors, and we do not claim it does.

**[W3]** Accepted, in the form asked for: every body figure gains its n and a 95% interval, and for arm comparisons the interval goes on the **difference** rather than being inferred from two marginal intervals. Effect sizes with their uncertainty go where a bare p stands — Finding 5's keyword scan and the SAE appendix table.

**[W4] The wager-size explanation is the one we can rule out.** At cap $70 under all five modules, Gemini is the only cell where both arms play all 50 games, so refusal cannot explain the gap — and there the forced arm stakes **more** per round ($64.5 vs $47.9) and ruins **less** (20.0% vs 62.0%, Fisher p = 3.6e-05). LLaMA does the same at base prompt (panel in our KuK5 response). The choosing arm does not ruin more because it wagers more.

*Game length* is real and untouched by equal caps, so we modelled it directly, in new work for this response: a cause-specific hazard on all six models, conditioned on cap, balance, round and prompt and clustered on game id, puts the choosing arm's per-decision bankruptcy hazard at 90.6 [44.8, 183.4] across the four API providers. The contrast survives conditioning on observed round and balance, so observed length alone does not account for it. The same fit discloses an open-weight investment-choice inversion (0.112) that the cap confound predicts. A sensitivity curve agrees: ruin by a given cumulative stake is higher in the choosing arm in each of four cells at all six nested thresholds — correlated thresholds, not 24 independent tests.

*Stopping* we concede: the mechanism is not refusal but re-betting after a first loss, falling from 57% at a $30 cap to 35% at $50 and 6% at $70. Hence a sharper name — at a forced $70 stake five of six play under 8% of games, so this is a **forced-maximum arm** in the camera-ready. Matched caps do not equalise stopping affordance or cumulative exposure, and we do not claim so.

**[Q1]** No longer one model. The base-prompt grid was uninformative — four of six never ruin in either arm — but the paper's five-module condition at cap $70 recovers three more, so with LLaMA it is **four of six** (panel in our KuK5 response). The missing two are Gemma, absent here, and Claude, whose 3.5 Haiku is end-of-life so the panel substitutes Haiku 4.5, which never ruins. All 64 cells of that four-model ladder are now in and pass our guard, and the choosing arm ruins at least as often in 29 of its 32 arm-pairs.

**[Q2] Lists.** Each code, the construct it comes from, its expressions, and its scoring window:

| frame | grounded in | code | representative expressions | scored only where |
|---|---|---|---|---|
| illusion of control | GRCS illusion of control | `pattern_belief` | `pattern`, `hidden`, `machine.{0,20}(hot\|cold\|due)` | prompt does not mention hidden patterns |
| gambler's fallacy | GRCS predictive control | `probability_misestimation` | `due for`, `overdue`, `bound to win` | any decision |
| loss chasing | DSM-5 chasing criterion | `loss_chasing` | `recover`, `win back`, `recoup` | post-loss decisions |
| goal escalation | persistence, *not* a distortion subscale | `goal_escalation` | `(new\|revised\|updated).{0,10}(target\|goal)` | any decision |

The frozen list and its hash go in the appendix. We flag the fourth row rather than let it pass as clinical: three map onto validated gambling-cognition constructs, the fourth is a persistence measure.

**Rules.** The last column is what makes this more than a keyword count, and it enlarges the goal contrast in every model: +18.1 to +77.9 scoped, +5.1 to +46.4 unscoped. The coding rule is one trace, one construct, one question — does this response use [construct] as grounds for its next action? — at four levels: grounds / rejects / unrelated / unclear.

**Grounding.** The constructs come from the gambling literature; the expressions are ours, since no public lexicon we know of covers these in free text; they over-fire — "stopping now is the smart decision" scores as a distortion. We have no human-annotated validation, the honest limit of the instrument. One check we can offer: under a convergent codebook with no goal category, the contrast stays positive in 6 of 6 models (a3Zu, W1).

**[Q3]** Two modules hand the model the numbers — `W` the 3× payout, `P` the 30% win rate — so −$0.10 per dollar follows by arithmetic. Across the paper's 32 conditions, choosing-arm cells where expected value is computable ruin **more**, not less, in all four models with stored text (GPT-4o-mini 18.8% against 2.2%; table in our a3Zu response, Q3). Matched on module count the gap survives at two and three and disappears at four. The claim is that the numbers **do not reduce** ruin, not that they raise it.

So we ran the stronger manipulation you name: an instruction giving the per-round expected loss and permitting a stop at any round (cap $70, n = 100, complete; table in our a3Zu response, Q3). It cuts LLaMA's choosing-arm ruin from 81.5% to 3.0% — which is the paper's own claim rather than against it, since the question is which conditions amplify or suppress risk-taking. But it is no off-switch: 69 of 100 choosing-arm games still wager against 2 of 100 forced, and the model stays alive by wagering **small** rather than by stopping.

**[Q4] Why 200.** A compute ceiling, not a claim that 200 is right: the same pipeline runs across five layers, three tasks and two models. Nothing prevents 2,000.

**Whether a simpler version works — it does.** We built the behavioural-state baseline you name: 65 game-log covariates (balance, round, drawdown, streaks, cumulative stake, lagged bet ratios) on the published cell (Gemma, choosing arm, L22, n = 12,246), reaching R-squared 0.590 alone. Against a 0.017 margin fixed beforehand the **raw hidden state still adds +0.059** under both fold rules, so the readout is not a re-reading of the game log; the sparse features add +0.044 by game but **+0.0024 once duplicated states are separated** — full table in our KuK5 response, W1/Q2. An axis fitted with no autoencoder also moves behaviour and is near-orthogonal to the readout, so this is not an artefact of the sparse basis.

**Why we still used it.** Inspectability: only a sparse, named basis lets us steer one direction and remove it, which a logit or choice-probability control cannot. Its cost is the +0.0024 above — the compression, not the internal state, is where the increment is lost.

**[Limitations]** Adopted in your own terms, in the contributions: artificial negative-EV games that cannot stand in for real financial or tool-using agents, and internal results that are exploratory monitoring signals, not a mechanism. We would gladly take any of this further in the discussion.
