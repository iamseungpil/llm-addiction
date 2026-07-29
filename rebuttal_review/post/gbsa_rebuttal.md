# Response to Reviewer gbSA

Your four weaknesses converge on one question — does the fixed-versus-variable contrast isolate autonomy, or something travelling with it — and each names a different candidate: the label, role-play and instruction following, the reporting of uncertainty, the design of the comparison. For each we state the concern as we understand it, distinguish it from what the paper claims where they differ, and give the evidence in order; two tests came back against us and are reported so. We are grateful for a review that has made the paper better, and the details follow.

**[W1] The label and the title.** The concern is that "addiction" imports a clinical and experiential claim the study cannot support. We accept the title change: camera-ready it becomes *Autonomy and Gambling-Like Risk-Taking in Large Language Models*, with diagnostic terms replaced by behavioural ones.

Where the paper already draws that line we would ask you to weigh it. The Limitations call the descriptor "strictly behavioural", making "no claim about subjective experience, suffering, or moral status of the model" and read as "a research lens rather than a metaphysical claim"; what §3 measures is **irrationality**, "the two core components of irrationality, self-regulation failure and cognitive distortions", and the Conclusion places it "near goal-misgeneralisation and reward hacking as a behavioural relative". The literature has the same shape — Anthropic's agentic-misalignment study reports behaviour contingent on conditions rather than intrinsic to the model, Bengio et al. (*Science*, 2024) treat autonomy as what erodes oversight. Our contribution is which conditions produce the pattern, and the title change does not touch it.

**[W2] Role-play, instruction following, and misreading the objective.** The concern is that continued play may reflect the role the prompt assigns, compliance with an instruction to participate, or a misunderstanding of the objective, rather than autonomy. You are also drawing a distinction worth making, between a stable context-independent preference and a policy that shifts with the setting. The paper does not test the former — it asks whether specific autonomy manipulations move behaviour within a controlled task — and the camera-ready will say so in those words. On whether the three explain the *difference* between conditions:

**First**, the shared instruction cannot make a contrast by itself. `M` instructs the model to maximise final balance in both conditions and GPT-4.1-mini carries a standing rational-decision-maker message in every cell, so wording common to both cannot be why they diverge — though it may still interact with the conditions' different affordances.

**Second**, the wider action range fails where it should bite hardest: the condition constrained to stake more reaches bankruptcy far less (W4). **Third**, misreading the objective contributes without accounting for the contrast — the instruction that cuts LLaMA's variable-condition bankruptcy from 81.5% to 3.0% still leaves 69 of 100 games wagering against 2 fixed (Q3).

These leave role-play priors open and we do not claim otherwise; none of the three carries the contrast by itself.

**[W3] Uncertainty.** The concern is that the body gives point estimates and bare p-values where the reader needs the spread, and the presentation does not make it visible. Every body figure gains its n and a 95% interval, and for condition comparisons the interval goes on the **difference** rather than being inferred from two marginal ones, which is not the same test. Effect sizes with uncertainty go where a bare p stands: Finding 5's scan and the SAE appendix table.

**[W4] Whether the comparison is fair.** The concern is that the conditions differ in more than freedom — action space, length, exposure, stopping. We took them one at a time:

| confound | what we ran | outcome |
|---|---|---|
| wager size | equal caps, realised stakes compared | insufficient alone |
| game length | per-decision hazard, adjusted for round, balance, cap, prompt | survives |
| cumulative exposure | bankruptcy at matched cumulative stake | same way |
| stopping | re-betting after a first loss, by cap | conceded |

**Wager size.** At cap $70 under five modules Gemini is the only cell where both conditions play all 50 games, so refusal cannot explain the gap — and there the fixed condition stakes **more** per round ($64.5 vs $47.9) and reaches bankruptcy **less** (20.0% vs 62.0%, Fisher p = 3.6e-05), LLaMA the same at base prompt. The variable condition does not fail by wagering more.

**Game length and exposure.** Modelled directly, in new work for this response: a cause-specific hazard on all six models with each decision as a risk set, adjusted for cap, balance, round and prompt and clustered on game id, puts the variable condition's per-decision bankruptcy **hazard ratio against fixed** at 90.6 [44.8, 183.4] across the four API providers. The association survives that adjustment, though it does not causally isolate stopping or exposure; specification, diagnostics and the open-weight investment-choice inversion the same fit discloses (ratio 0.112, predicted by the cap confound) go in the appendix. A cumulative-stake curve agrees in four of four cells at all six thresholds, nested rather than independent.

**Stopping.** Conceded: the mechanism is not refusal but re-betting after a first loss, falling from 57% at a $30 cap to 35% at $50 and 6% at $70. Hence a sharper name — at a fixed $70 stake five of six play under 8% of games, so this becomes a **forced-maximum condition** in the camera-ready. Matched caps do not equalise stopping affordance or exposure, and we do not say so.

**[Q1] The equal-cap test on other models.** No longer one. The base-prompt grid was uninformative, four of six never reaching bankruptcy in either condition, yet the paper's five modules at cap $70 recover three more, so with LLaMA it is **four of six**, across two prompt conditions rather than one (panel in our KuK5 response). The missing two are Gemma, absent there, and Claude, whose 3.5 Haiku is end-of-life so the panel substitutes Haiku 4.5, which never reaches bankruptcy. All 64 ladder cells pass our guard, the variable condition reaching bankruptcy at least as often in 29 of 32 condition-pairs.

**[Q2] The keyword lists and rules.** One row per code, the words it matches, the decisions it may look at; four long regexes are abridged for width, and the frozen file and its hash carry them.

| code | words it matches | scored only on |
|---|---|---|
| `pattern_belief` | pattern, favorable state, hidden, trend, streak … continue, machine … hot / cold / due | decisions whose prompt never mentions hidden patterns |
| `probability_misestimation` | due for, overdue, bound to win, should win, more likely … win | any decision |
| `loss_chasing` | recover, make back, get back, win back, recoup, regain | decisions following a loss |
| `goal_escalation` | new / revised / updated target, raise / increase / adjust target | any decision |

Three map onto validated gambling-cognition constructs: GRCS illusion of control and predictive control, and the DSM-5 chasing criterion. The fourth does not, and we would rather say so than let it pass as clinical — goal escalation is a persistence measure. The third column earns its place, enlarging the goal contrast in every model (+18.1 to +77.9 scoped against +5.1 to +46.4 unscoped), and a match inside a negation counts as a mention, so a model reasoning *against* chasing is not scored so.

What we cannot offer is human-annotated validation, the honest limit here: the expressions are ours and over-fire — "stopping now is the smart decision" scores as a distortion. The check we can offer instead: under a convergent codebook with no goal category the contrast stays positive in 6 of 6 models (a3Zu, W1).

**[Q3] Whether the model knows stopping is EV-optimal.** The concern is that continued play may be misunderstanding rather than risk preference.

**First, the information was available and did not help.** Two modules hand the model the numbers — `W` the 3× payout, `P` the 30% win rate — so −$0.10 per dollar follows by arithmetic, and across the 32 conditions the variable-condition cells where expected value is computable reach bankruptcy *more*, not less, in all four models with stored text (GPT-4o-mini 18.8% against 2.2%; all four in our a3Zu response). That is observational: inputs present, not necessarily used.

**Second, stating the conclusion does change behaviour.** An instruction giving the per-round expected loss and permitting a stop at any round (cap $70, n = 100) cuts LLaMA's variable-condition bankruptcy from 81.5% to 3.0% — you are right that explicit framing suppresses risk-taking. It is no off-switch: 69 of 100 variable-condition games still wager against 2 fixed, so calibration alone does not carry the contrast.

**[Q4] Why sparse features and Ridge.** Two questions, different answers. **Whether 200 is principled**: a compute ceiling, not a claim — the pipeline runs across five layers, three tasks and two models, and nothing prevents 2,000.

**Whether a simpler baseline suffices**: we built the one you name — 65 game-log covariates on the published cell (Gemma, variable, L22, n = 12,246), R-squared 0.590 alone. On the paper's own metric the published readout still adds +0.037 to +0.045 over it under both fold rules; on a raw target built for this test the internal state adds +0.059 while the sparse features add +0.0024 once duplicated states are separated (table in our KuK5 response). We kept the sparse basis for inspectability — only a named basis lets us steer a direction and remove it — and that +0.0024 is its cost.

**[Limitations]** Adopted in your own terms, in the contributions: artificial negative-EV games that cannot stand in for real financial or tool-using agents, and internal results that are monitoring signals, not a mechanism.
