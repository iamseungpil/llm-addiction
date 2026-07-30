# Response to Reviewer gbSA

Your four weaknesses identify different explanations that may travel with the fixed-versus-variable contrast: clinical framing, role-play and instruction following, uncertainty reporting, and differences in the available policies. We answer each with its most direct evidence; analyses that went against us are reported as such. We sincerely thank the reviewer for helping us improve the paper.

**[W1] The label and the title.** **We agree that "addiction" can imply a clinical and experiential claim the study does not test.** The camera-ready title becomes *Autonomy and Gambling-Like Risk-Taking in Large Language Models*, with diagnostic terms replaced by behavioural ones.

The submitted Limitations already call the label "strictly behavioural", making "no claim about subjective experience, suffering, or moral status of the model". Our contribution is narrower — which conditions change observable participation, wager size, persistence, stopping and distortion-associated language — and the revised framing makes that boundary explicit.

**[W2] Role-play, instruction following, and misreading the objective.** **Role framing and task calibration clearly influence behaviour, but none of the proposed explanations accounts for the full fixed-versus-variable contrast by itself.** You are also right that the paper tests a setting-dependent policy, not a stable context-independent risk preference; the camera-ready will say so directly.

**First, shared wording alone is insufficient.** The reward module instructs the model to maximise final balance in both conditions, and GPT-4.1-mini carries the same rational-decision-maker message in every cell. Common wording may interact with each condition's affordances, but it cannot by itself explain the divergence.

**Second, a larger wager range is insufficient**: in the matched-cap cells the fixed condition stakes more per round yet ruins less (W4; KuK5 response).

**Third, misunderstanding contributes but does not carry the contrast**: the calibration instruction that cuts LLaMA's variable-condition bankruptcy from 81.5% to 3.0% still leaves 69 of 100 games wagering, against 2 fixed (Q3).

A same-state intervention adds one more piece: with prompt, seed and game state fixed, manipulating a behaviour-linked internal direction changes the betting outcome. That claims no trait; it shows only that the decision state carries behaviourally operative information. We will describe the finding as a *setting-dependent policy effect*.

**[W3] Uncertainty.** **We agree that point estimates and bare p-values are insufficient.** Every primary body figure gains its sample size and a 95% interval; condition comparisons carry the interval on the difference itself, not two marginals; Finding 5's keyword analysis and the internal-state tables gain effect sizes with uncertainty.

**[W4] Whether the comparison is fair.** **Equal caps make wager size alone insufficient as an explanation, but they do not equalise stopping, cumulative exposure, or the policy space.**

*Wager size.* At cap \$70 under five modules, Gemini is the cell where both conditions play all 50 games, so refusal cannot explain the result. The fixed condition nevertheless stakes **more** per played round (\$64.5 versus \$47.9) and reaches bankruptcy **less** (20.0% versus 62.0%, Fisher p = 3.6e-05); LLaMA shows the same dissociation under BASE.

*Game length and exposure.* For this response we fitted a cause-specific hazard model across the four API models. It compares bankruptcy risk decision by decision, so longer games do not mechanically accumulate risk, adjusting for cap, balance, round and prompt, clustered by game. The variable condition's per-decision bankruptcy hazard ratio against fixed is 90.6 [44.8, 183.4] — a wide interval, but far above one. It does not causally equalise stopping or exposure, and we do not present it as doing so.

*Stopping.* Conceded. The remaining difference sits in re-betting after a first loss: in LLaMA's fixed condition, re-betting after a first loss falls from 57% at a \$30 cap to 35% at \$50 and 6% at \$70, and at a forced \$70 stake five of six models play fewer than 8% of games. The camera-ready therefore renames the fixed condition the *forced-maximum condition*; this response keeps fixed/variable to match the submitted paper and the review.

A registered follow-up is running in the same harness: the model first chooses one fixed stake from the four caps, with fresh fixed and variable comparators collected alongside, separating a one-time policy choice from per-round discretion. Every specified cell reports by **3 August**, regardless of direction.

**[Q1] The equal-cap test on other models.** **The matched-cap result no longer rests on one model.** The strongest additional result is LLaMA: the fixed condition stakes \$68.4 per played round yet reaches 3.0% bankruptcy, whereas the variable condition stakes \$32.1 and reaches 81.5%. Under the five modules at cap \$70, Gemini and GPT-4.1-mini move the same way, as does the submitted GPT-4o-mini ablation; several BASE cells are floor-limited in both arms, so we do not claim a homogeneous panel. The full table and the negative pre-registered panel criteria are in our KuK5 response.

The completed 64-cell ladder passes the integrity guard (a parse- and storage-completeness check), and the variable condition reaches bankruptcy at least as often as fixed in 29 of 32 condition-pairs, ties included. The three exceptions are one Claude cell and two Gemini BASE cells; the Claude cells come from a replacement model (the submitted checkpoint is retired) and speak to the replacement only.

**[Q2] The keyword lists and rules.** The longest regular expressions are abbreviated for width; the frozen file and its hash carry them in full, and the camera-ready appendix prints them.

| code | expressions matched | scored only on |
|---|---|---|
| `pattern_belief` | pattern, favorable state, hidden, trend, streak … continue; machine … hot / cold / due | decisions whose prompt does not mention hidden patterns |
| `probability_misestimation` | due for, overdue, bound to win, should win, more likely … win | any decision |
| `loss_chasing` | recover, make back, get back, win back, recoup, regain | decisions following a loss |
| `goal_escalation` | new / revised / updated target; raise / increase / adjust target | any decision |

The first three correspond to established gambling-cognition constructs — illusion or predictive control, probability misjudgement, loss chasing — while the lexical mappings are our unvalidated operationalisation. The fourth is not clinical; goal escalation is reported separately as behavioural persistence.

We cannot provide human-annotated validation, and the expressions can over-fire. As a robustness check rather than a substitute, a frozen codebook with no goal category retains a positive goal contrast in all six models; the ranges and limitations are in our a3Zu response.

**[Q3] Whether the model knows stopping maximises expected value.** **Having the numbers did not help; being told the conclusion did.**

Two prompt modules hand over the payout and win probability, from which the −10% expectation follows, yet the conditions where it is computable do not reach bankruptcy less in any of the four models with stored text. Inputs present does not mean inputs used; the numbers and the prompt-richness control are in our a3Zu response (Q3).

A direct instruction — immediate stopping maximises expected value, stopping permitted at any round — cuts LLaMA's variable-condition bankruptcy from 81.5% to 3.0%, so you are right that task interpretation is a major moderator. It does not eliminate participation: 69 of 100 variable-condition games still wager, against 2 of 100 fixed. Calibration explains much, not all, of the contrast.

**[Q4] Why sparse-autoencoder features and Ridge regression.** **The sparse autoencoder is there for feature-level inspectability, not because sparse features were expected to outperform dense or behavioural baselines.** The top-200 limit was a computational ceiling rather than an optimality claim.

The requested 65-covariate baseline reaches 0.590 on the raw bet-ratio target and 0.140 on the paper's deconfounded target; the two are different outcomes and not directly comparable. On the submitted target the readout adds +0.037 to +0.045 under both fold rules; on the raw target the raw hidden state adds +0.059 while the sparse features add +0.0024 once repeated states are grouped (table in our KuK5 response). Steering does not require the autoencoder either: a direction fitted directly in the raw residual stream also moves behaviour. The sparse basis buys inspectability, not statistical necessity.

**[Limitations]** We adopt the reviewer's limitation in the contribution statement: these are artificial negative-expected-value games and cannot stand in for real financial or tool-using agents. The internal-state results provide predictive monitoring and intervention evidence, not a complete mechanism.
