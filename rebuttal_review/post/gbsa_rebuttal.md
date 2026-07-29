# Response to Reviewer gbSA

Your four weaknesses converge on one question: does the fixed-versus-variable contrast isolate autonomy, or something travelling with it? Each names a different candidate — the label, role-play and instruction following, the treatment of uncertainty, the comparison's design. We answer each with its own evidence; two tests came back against us and are reported so. We are grateful for a review that has made the paper better.

**[W1] The label and the title.** **We accept the title change.** Camera-ready it becomes *Autonomy and Gambling-Like Risk-Taking in Large Language Models*, with diagnostic terms replaced by behavioural ones.

The paper's Limitations already call the descriptor "strictly behavioural", making "no claim about subjective experience, suffering, or moral status of the model". What the behaviour section measures is **irrationality**: self-regulation failure and cognitive distortions. Our contribution is which conditions produce that pattern, and it does not change with the title.

**[W2] Role-play, instruction following, and misreading the objective.** **None of the three explains the difference between conditions by itself, and we do not claim they are absent.** You are right to separate a stable context-independent preference from a setting-dependent policy: the paper tests only the latter, and the camera-ready will say so.

**First**, the shared instruction cannot make a contrast on its own: the reward module tells the model to maximise its final balance in both conditions, and GPT-4.1-mini carries a standing rational-decision-maker message in every cell. Wording common to both cannot be why they diverge.

**Second**, the wider action range fails where it should bite hardest: the condition constrained to stake more reaches bankruptcy far less (W4).

**Third**, misreading the objective contributes without accounting for the contrast: the instruction that cuts LLaMA's variable-condition bankruptcy from 81.5% to 3.0% still leaves 69 of 100 games wagering, against 2 in the fixed condition (Q3). These leave role-play priors open; none of the three carries the contrast alone.

**[W3] Uncertainty.** **Agreed.** Every body figure gains its n and a 95% interval; for condition comparisons the interval goes on the **difference**, not on two marginals; and effect sizes with uncertainty replace bare p-values in Finding 5's keyword scan and the internal-state tables.

**[W4] Whether the comparison is fair.** **Equal caps answer the maximum-wager explanation; they do not equalise stopping or exposure. One candidate is refuted, one is adjusted for, and one we concede and are testing now.**

*Wager size (refuted).* At cap \$70 under five modules, Gemini is the only cell where both conditions play all 50 games, so refusal cannot explain the gap. There the fixed condition stakes **more** per round (\$64.5 vs \$47.9) and reaches bankruptcy **less** (20.0% vs 62.0%, Fisher p = 3.6e-05); LLaMA shows the same at base prompt. The variable condition does not fail by wagering more.

*Game length and exposure (adjusted for).* We fitted a cause-specific hazard model for this response, across the four API models. It compares bankruptcy risk decision by decision, so longer games do not mechanically accumulate more risk; it adjusts for cap, balance, round and prompt, with errors clustered on game id. The variable condition's per-decision bankruptcy hazard ratio against fixed is 90.6 [44.8, 183.4]. The association survives that adjustment; it does not causally isolate stopping or exposure, and the full specification will be in the revised paper.

*Stopping (conceded, and under test).* In LLaMA's fixed arm (n = 200 per cap), re-betting after a first loss falls from 57% at a \$30 cap to 35% at \$50 and 6% at \$70. At a forced \$70 stake five of six models play under 8% of games, so the camera-ready renames the fixed condition the **forced-maximum condition**. To separate choosing a policy from per-round discretion, a registered follow-up is running in the same harness, and every pre-specified cell reports by **3 August**, whichever way it falls (both open-weight models, 100–200 games per cell):

| arm | who picks the stake | when | can it change during play |
|---|---|---|---|
| forced fixed stake | environment | each played round | no |
| model-chosen fixed stake | model | before play | no |
| variable | model | every round | yes |

**[Q1] The equal-cap test on other models.** **No longer one model.** The strongest cell is LLaMA: the fixed condition stakes \$68.4 per round yet 3.0% reach bankruptcy, while the variable condition stakes \$32.1 and 81.5% do. Gemini and GPT-4.1-mini move the same way under the paper's five modules, where the ablation's own GPT-4o-mini also replicates; several base-prompt cells are floor-limited in both conditions, so we do not claim a uniform panel. The full panel, the completed 64-cell grid, and the negative pre-registered panel rules are in our KuK5 response; Claude's cell comes from a replacement model (the submitted one is retired) and stays at the floor.

**[Q2] The keyword lists and rules.** One row per code — the construct it operationalises, the words it matches, the decisions it may score. The longest expressions are abridged for width; the frozen file and its hash carry them in full.

| code | construct | words it matches | scored only on |
|---|---|---|---|
| `pattern_belief` | illusion / predictive control | pattern, favorable state, hidden, trend, streak … continue, machine … hot / cold / due | decisions whose prompt never mentions hidden patterns |
| `probability_misestimation` | probability misjudgement | due for, overdue, bound to win, should win, more likely … win | any decision |
| `loss_chasing` | chasing criterion | recover, make back, get back, win back, recoup, regain | decisions following a loss |
| `goal_escalation` | persistence (not clinical) | new / revised / updated target, raise / increase / adjust target | any decision |

What we cannot offer is human-annotated validation, the honest limit here: the expressions are ours and over-fire. The check we can offer: under a convergent codebook with no goal category the goal contrast stays positive in 6 of 6 models (a3Zu, W).

**[Q3] Whether the model knows stopping maximises expected value.** **Having the numbers did not help; being told the conclusion did — and the contrast survives even that.** Two prompt modules hand the model the payout and the win rate, so the ten-percent expected loss follows by arithmetic. Yet the cells where expected value is computable did not reach bankruptcy less in any of the four models with stored text (GPT-4o-mini 18.8% against 2.2%; the other three and the prompt-richness control are in our a3Zu response, Q3). An instruction giving the per-round expected loss and permitting a stop at any round cuts LLaMA's variable-condition bankruptcy from 81.5% to 3.0%. You are right that explicit framing suppresses risk-taking. It is no off-switch: 69 of 100 variable-condition games still wager, against 2 fixed, so calibration alone does not carry the contrast.

**[Q4] Why sparse-autoencoder features and Ridge regression.** **The sparse basis is there for inspectability, not statistical necessity.** Whether top-200 features is principled: a compute ceiling, not a claim; nothing prevents 2,000. On the simpler baseline you name: 65 game-log covariates reach 0.590 on the raw bet-ratio target and 0.140 in the paper's own metric. There the published readout still adds +0.037 to +0.045 under both fold rules; on the raw target the internal state adds +0.059 while the sparse features add +0.0024 once duplicated states are separated (table in our KuK5 response). Steering does not require the sparse basis either: a direction fitted in the raw residual stream also moves behaviour. So +0.0024 is the price of inspectability, and we say so.

**[Limitations]** Adopted in your own terms: artificial negative-expected-value games that cannot stand in for real financial or tool-using agents, and internal results that are monitoring signals, not a mechanism.
