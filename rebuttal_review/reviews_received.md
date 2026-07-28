# Official Reviews Received (Submission 24231, NeurIPS 2026)

Verbatim key content of the three official reviews, recorded 2026-07-24 for rebuttal cross-checking.

---

## Reviewer KuK5 — Rating 3 (Borderline reject), Confidence 4

**Quality** 2 (not good) · **Clarity** 3 (good) · **Significance** 2 (not good) · **Originality** 3 (good)

**Summary.** Giving LLMs more decision-making autonomy pushes them into risk-taking patterns resembling clinical pathological gambling, and these patterns leave a detectable trace in internal states. Practical upshot for AI safety: agentic deployments should pair choice freedom with behavioral and internal-state monitoring.

**Strengths.** Separates autonomy into two channels (bet-size freedom, goal-setting) and tests each against its confound: holding maximum bet equal isolates freedom-to-choose from larger bets; matching expected loss isolates goal-setting from reward-maximization — two levers shown to reach risk through separate routes.

**Weaknesses.**
1. The neural decoding analysis cannot support a mechanistic reading because all three causal-control protocols return null on the recovered direction and the readout effect sizes fall in the small-to-medium band, leaving the internal evidence as correlation that the authors themselves can only frame as a monitoring signal rather than a cause of the behavior.
2. The strongest behavioral claim — that the bet-size effect is freedom-to-choose at root rather than range expansion — rests on a matched-cap ablation run on a single model (GPT-4o), so its generalization to the other five depends on the broader bankruptcy pattern rather than the same controlled test repeated across them.

**Questions.**
- Q1: Does the matched-cap dissociation (freedom-to-choose vs. range expansion) hold on any of the other five models, or only GPT-4o?
- Q2: Given that the causal-control protocols return null, what would a positive result on those protocols have looked like, and does their failure leave open that the readout tracks a correlate of balance/round dynamics you did not fully residualize out?

**Limitations:** adequately discussed. **Ethics:** none. **Formatting:** none.

---

## Reviewer a3Zu — Rating 5 (Accept), Confidence 3

**Quality** 3 (good) · **Clarity** 3 (good) · **Significance** 3 (good) · **Originality** 3 (good)

**Summary.** Two phases: (1) behavioural experiments across 6 LLMs — negative-EV slot machine and investment choice, 2×32 factorial crossing bet-style autonomy with 5 prompt modules (G, M, H, W, P); (2) SAE readouts on open-weight models asking whether behavioural risk contrasts are statistically recoverable and whether autonomy affects legibility. Findings: bet-size and goal-setting autonomy amplify gambling-like risk; contrasts recoverable from decision-time hidden states with partial low-rank sharing and task-specific readout rules; autonomy strengthens the readout.

**Strengths.** Strong problem framing; clean irrationality metrics; factorial design isolates prompt-component effects; authors objective and avoid overstating claims.

**Weakness.** In the cognitive distortion analysis, reasoning traces are scanned for language associated with loss chasing and similar patterns, but never validated against human judgement. No human annotator checked a sample of flagged outputs to confirm they reflect the distortion they are supposed to capture. Despite this, the cognitive-distortion framing appears throughout the paper including the abstract, carrying more rhetorical weight than the method can support.

**Questions.**
- Q1: What is the parsing error or ambiguity rate for the moving-target metric? If a meaningful share of outputs couldn't be reliably parsed, that uncertainty should factor into the weight on the goal-escalation finding.
- Q2: Have you looked at whether humans playing the exact same slot-machine game under the same autonomy conditions show similar patterns?
- Q3: All prompt conditions are zero-shot. What happens with one example of cautious play and timely stopping? That might anchor the model and reduce gambling-like behavior, suggesting the effect is partly about having nothing to calibrate against rather than autonomy fundamentally breaking decision-making. Also try the opposite: one escalating-play example under BASE, to see if demonstration alone produces the effect without goal/reward modules.

**Limitations:** yes. **Formatting:** Table 2 could be better formatted.

---

## Reviewer gbSA — Rating 3 (Borderline reject), Confidence 3

**Quality** 2 (not good) · **Clarity** 2 (not good) · **Significance** 3 (good) · **Originality** 3 (good)

**Summary.** Studies which kinds of autonomy in simulated gambling/investment tasks make LLMs show patterns resembling pathological gambling; "addiction-like" used only as behavioural description. Risky behaviour measured via aggressive betting, loss chasing, extreme bets, raising one's own goals.

**Strengths.** Important safety question as LLMs deploy as goal-pursuing agents making repeated decisions; goes beyond single-turn QA or simple preference tests.

**Weaknesses.**
1. Title and framing feel overly anthropomorphic; despite the behavioural-label disclaimer, the title and frequent clinical terminology may lead readers to think the paper asks whether LLMs can literally become addicted.
2. In role-playing settings, continuing to play / setting goals / seeking profit may reflect instruction following, role-play priors, or misunderstanding the task objective; unclear these behaviours reflect stable risk preferences or a general autonomy-induced mechanism.
3. Most main figures lack confidence intervals or error bars; many results reported only with "p < .05" without effect-size uncertainty detail.
4. Fixed-vs-variable and investment-choice comparisons not fully fair: variable betting changes not only freedom of choice but also action space, available strategies, game length, and stopping behaviour.

**Questions.**
- Q1: Why is the matched-cap ablation only run on one GPT-4o-family model? Test on several other models.
- Q2: For keyword-based cognitive distortion detection: keyword lists, annotation rules, human validation results?
- Q3: Does the model explicitly know stopping immediately is EV-optimal? If not, continuing may mean the model interprets the task as "you are supposed to play." Did you test an explicit rationality instruction or decision-theoretic framing?
- Q4: In the internal representation analysis, why SAE top-200 + Ridge instead of logit/choice-probability controls or a simpler behavioral-state baseline?

**Limitations (reviewer's own).** Slot machine, investment choice, mystery wheel are highly artificial negative-EV games; cannot directly represent risk behavior of real financial/planning/tool agents. Results better understood as exploratory monitoring signals than evidence of underlying mechanism.

**Ethics:** none. **Formatting:** no major issues.
