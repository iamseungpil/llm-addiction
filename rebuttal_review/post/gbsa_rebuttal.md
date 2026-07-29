# Response to Reviewer gbSA

Your four weaknesses converge on one question — does the fixed-versus-variable contrast isolate autonomy, or something travelling with it — and each names a different candidate: the label, role-play and instruction following, the treatment of uncertainty, the comparison's design. For each we state the concern and give the evidence in order; two tests came back against us and are reported so. We are grateful for a review that has made the paper better; the details follow.

**[W1] The label and the title.** The concern is that "addiction" imports a clinical and experiential claim the study cannot support. We accept the title change: camera-ready it becomes *Autonomy and Gambling-Like Risk-Taking in Large Language Models*, diagnostic terms replaced by behavioural ones.

The paper's Limitations already call the descriptor "strictly behavioural", making "no claim about subjective experience, suffering, or moral status of the model"; what the behaviour section measures is **irrationality** — self-regulation failure and cognitive distortions — and our contribution, which conditions produce that pattern, does not change with the title.

**[W2] Role-play, instruction following, and misreading the objective.** The concern is that continued play may reflect the assigned role, compliance with an instruction to participate, or a misread objective rather than autonomy. You are right to separate a stable context-independent preference from a setting-dependent policy: the paper tests only the latter, and the camera-ready will say so. On whether the three explain the *difference* between conditions:

**First**, the shared instruction cannot make a contrast by itself: the reward module instructs the model to maximise its final balance in both conditions, and GPT-4.1-mini carries a standing rational-decision-maker message in every cell — wording common to both cannot be why they diverge. **Second**, the wider action range fails where it should bite hardest: the condition constrained to stake more reaches bankruptcy far less (W4). **Third**, misreading the objective contributes without accounting for the contrast — the instruction that cuts LLaMA's variable-condition bankruptcy from 81.5% to 3.0% still leaves 69 of 100 games wagering against 2 fixed (Q3). These leave role-play priors open; none of the three carries the contrast alone.

**[W3] Uncertainty.** The concern is that the body gives point estimates and bare p-values where the reader needs the spread. Every body figure gains its n and a 95% interval, and for condition comparisons the interval goes on the **difference** rather than being inferred from two marginal ones. Effect sizes with uncertainty replace bare p-values in Finding 5's keyword scan and the internal-state tables.

**[W4] Whether the comparison is fair.** The concern is that the conditions differ in more than freedom — action space, length, exposure, stopping. Equal caps address the maximum-wager explanation; they do not equalise stopping or exposure, so we took each in turn.

**Wager size.** At cap $70 under five modules Gemini is the only cell where both conditions play all 50 games, so refusal cannot explain the gap — yet the fixed condition stakes **more** per round ($64.5 vs $47.9) and reaches bankruptcy **less** (20.0% vs 62.0%, Fisher p = 3.6e-05); LLaMA the same at base prompt. The variable condition does not fail by wagering more.

**Game length and exposure.** Modelled directly for this response: a cause-specific hazard across the four API models — each decision a risk set, adjusted for cap, balance, round and prompt, clustered on game id — puts the variable condition's per-decision bankruptcy **hazard ratio against fixed** at 90.6 [44.8, 183.4]. The association survives adjustment without causally isolating stopping or exposure; the full specification will be in the revised paper.

**Stopping.** Conceded: the mechanism is not refusal but re-betting after a first loss — 57% at a $30 cap, 35% at $50, 6% at $70. Hence a sharper name: at a forced $70 stake five of six models play under 8% of games, so this becomes a **forced-maximum condition** in the camera-ready. A registered follow-up is running — the model picks its own fixed stake from the four caps, with fresh forced and variable comparators in the same harness — to separate choosing a policy from per-round discretion; report by 3 August, whichever way the cells fall.

**[Q1] The equal-cap test on other models.** No longer one. The strongest cell is LLaMA: the fixed condition stakes $68.4 per round yet 3.0% reach bankruptcy; the variable condition stakes $32.1 and 81.5% do. The base grid is otherwise uninformative — four of six models never reach bankruptcy in either condition — but the paper's five modules at cap $70 recover three more, so it is **four of six** across two prompt conditions; the panel and the completed 64-cell grid are in our KuK5 response. The missing two: Gemma, absent there, and Claude, where the submitted model is retired and no longer answers requests and the replacement's cell never reaches bankruptcy, so it speaks to the replacement.

**[Q2] The keyword lists and rules.** Long regexes are abridged; the frozen file and its hash carry them.

| code | words it matches | scored only on |
|---|---|---|
| `pattern_belief` | pattern, favorable state, hidden, trend, streak … continue, machine … hot / cold / due | decisions whose prompt never mentions hidden patterns |
| `probability_misestimation` | due for, overdue, bound to win, should win, more likely … win | any decision |
| `loss_chasing` | recover, make back, get back, win back, recoup, regain | decisions following a loss |
| `goal_escalation` | new / revised / updated target, raise / increase / adjust target | any decision |

Three map onto validated gambling-cognition constructs — illusion of control, predictive control, the chasing criterion; the fourth does not, and goal escalation is a persistence measure rather than a clinical one.

What we cannot offer is human-annotated validation, the honest limit here: the expressions are ours and over-fire. The check we can offer: under a convergent codebook with no goal category the contrast stays positive in 6 of 6 models (a3Zu, W).

**[Q3] Whether the model knows stopping maximises expected value.** The concern is that continued play may be misunderstanding rather than risk preference.

**First, the information was available and did not help.** Two prompt modules hand the model the payout and the win rate, so the ten-percent expected loss follows by arithmetic — yet the cells where expected value is computable did not reach bankruptcy less in any of the four models with stored text (numbers and the prompt-richness control in our a3Zu response, Q3). Inputs present, not necessarily used.

**Second, stating the conclusion does change behaviour.** An instruction giving the per-round expected loss and permitting a stop at any round cuts LLaMA's variable-condition bankruptcy from 81.5% to 3.0% — you are right that explicit framing suppresses risk-taking. It is no off-switch: 69 of 100 variable-condition games still wager against 2 fixed, so calibration alone does not carry the contrast.

**[Q4] Why sparse-autoencoder features and Ridge regression.** **Whether 200 is principled**: a compute ceiling, not a claim; nothing prevents 2,000.

**Whether a simpler baseline suffices**: we built the one you name — 65 game-log covariates on the published cell, reaching 0.590 on the raw bet-ratio target and 0.140 in the paper's own metric. There the published readout still adds +0.037 to +0.045 under both fold rules; on a raw target built for this test the internal state adds +0.059 while the sparse features add +0.0024 once duplicated states are separated (table in our KuK5 response). We kept the sparse basis for feature-level inspectability rather than statistical necessity — steering itself does not require it — and +0.0024 is its cost.

**[Limitations]** Adopted in your own terms: artificial negative-expected-value games that cannot stand in for real financial or tool-using agents, and internal results that are monitoring signals, not a mechanism.
