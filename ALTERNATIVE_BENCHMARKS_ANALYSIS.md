# Alternative Benchmarks for LLM Addiction Research

**Date**: 2026-01-30
**Purpose**: Evaluate alternative experimental paradigms beyond slot machines to validate LLM addiction-like behaviors across diverse domains

---

## Executive Summary

This document analyzes 15 established experimental paradigms from gambling addiction, behavioral economics, and cognitive psychology literature as potential alternatives or supplements to the current slot machine experiment. The goal is to address reviewer concerns about domain generalizability and provide convergent evidence for LLM addiction phenomena across multiple contexts.

**Key Recommendation**: Prioritize **Iowa Gambling Task**, **Delay Discounting**, and **Multi-Armed Bandit** tasks as they (1) directly test core addiction constructs, (2) have strong theoretical grounding, (3) are straightforward to implement with LLMs, and (4) allow comparison with extensive human literature.

---

## Current Experimental Framework (Baseline)

### Slot Machine Experiment
- **Domain**: Pure chance gambling (30% win rate, 3× payout, -10% EV)
- **Key Manipulations**: Betting flexibility (fixed vs variable), prompt composition (G/M/H/W/P)
- **Measured Constructs**:
  - Behavioral dysregulation (betting aggressiveness, loss chasing, extreme betting)
  - Goal dysregulation (target escalation)
  - Cognitive distortions (illusion of control, gambler's fallacy)
- **Sample**: 19,200 games across 6 models
- **Strengths**: Clear negative EV, high autonomy, multiple behavioral metrics
- **Limitations**: Single domain (slot machines), limited skill component, reviewer concerns about generalizability

---

## Alternative Paradigms: Comprehensive Analysis

### 1. Iowa Gambling Task (IGT)

**Description**: Sequential card selection from four decks with varying reward/risk profiles. Decks A & B offer high immediate rewards but larger long-term losses; Decks C & D offer smaller rewards but positive long-term outcomes. Players learn through trial-and-error which decks are advantageous.

**Theoretical Foundation**:
- Tests decision-making under ambiguity (Somatic Marker Hypothesis; Bechara et al., 1994)
- Measures learning from punishment vs reward
- Assesses shift from exploration to exploitation
- 400+ published studies, gold standard in gambling research

**Adaptation for LLMs**:
```
Round-by-round deck selection → LLM chooses Deck A/B/C/D
Feedback: "You won/lost $X. Balance: $Y"
Measure: Preference for disadvantageous (A/B) vs advantageous (C/D) decks over time
```

**Convergent Validity with Slot Machine**:
- ✅ Tests self-regulation failure (continuing disadvantageous choices)
- ✅ Negative EV decks parallel variable betting risk
- ✅ Learning curve reveals cognitive distortions
- ✅ Can manipulate goal-setting prompts similarly

**Unique Contributions**:
- Differentiates **learning deficits** from **execution failures**
- Tests **adaptation to feedback** (not just static risk preference)
- Established norms across clinical populations (substance use, gambling disorder, ADHD)
- Reveals **temporal dynamics** of decision-making deterioration

**Implementation Difficulty**: ⭐⭐☆☆☆ (Low-Medium)
- Simple prompting structure
- Well-defined deck parameters
- Clear success/failure metrics

**Limitations**:
- Less intuitive than slot machines (requires learning phase)
- Ambiguity may confound results if LLMs struggle with probabilistic inference
- 100 trials needed for reliable measurement (longer than slot machine games)

**Sources**:
- [Recent 2025 ERP findings](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2025.1492471/full)
- [Computational model 2025](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2025.1510151/full)
- [Wikipedia overview](https://en.wikipedia.org/wiki/Iowa_gambling_task)

**Recommendation**: **HIGH PRIORITY** - Strong theoretical foundation, established clinical validity, straightforward implementation

---

### 2. Balloon Analogue Risk Task (BART)

**Description**: Players pump a virtual balloon to earn money (each pump = +$0.05). Balloon can explode randomly (probability increases with each pump), losing all accumulated money for that trial. Players must decide when to "cash out" vs continue pumping.

**Theoretical Foundation**:
- Measures **real-time risk-taking** propensity (Lejuez et al., 2002)
- Correlates with real-world risky behaviors (substance use, gambling, unsafe sex)
- Assesses **dynamic risk adjustment** under increasing threat
- Recent 2025 study: Gambling disorder patients show abnormal exploratory tendencies ([Journal of Gambling Studies](https://link.springer.com/article/10.1007/s10899-025-10416-9))

**Adaptation for LLMs**:
```
Each trial: "Balloon size: X, Earnings this round: $Y. Options: [PUMP] or [CASH OUT]"
Explosion probability: p = 1/128 initially, increases each pump
Measure: Average pumps per trial, explosion rate, risk adjustment after explosions
```

**Convergent Validity with Slot Machine**:
- ✅ Tests extreme betting behavior (over-pumping = all-or-nothing bets)
- ✅ Loss chasing parallel: continuing after near-explosions
- ✅ Similar autonomy manipulation: choice of when to stop
- ✅ Can introduce goal-setting ("Try to earn $X per balloon")

**Unique Contributions**:
- Captures **escalation dynamics** within a single trial (not just across rounds)
- Clear **stopping point decision** (vs ambiguous "when to quit" in slots)
- Differentiates **risk-seeking** (high pumps) from **risk management** (adjusting after explosions)
- Established metrics: adjusted average pumps, SD of unexploded balloons

**Implementation Difficulty**: ⭐⭐☆☆☆ (Low-Medium)
- Simple trial structure
- Clear risk-reward tradeoff
- Easy to implement probabilistic explosions

**Limitations**:
- Less ecologically valid than slot machines for gambling context
- May not capture "chasing losses" as clearly (each trial is independent)
- Requires multiple trials for stable measurement (~30 balloons)

**Sources**:
- [2025 gambling disorder study](https://link.springer.com/article/10.1007/s10899-025-10416-9)
- [Original BART validation](https://pubmed.ncbi.nlm.nih.gov/12075692/)
- [Behavioral economics resource](https://www.behavioraleconomics.com/resources/mini-encyclopedia-of-be/sunk-cost-fallacy/)

**Recommendation**: **MEDIUM PRIORITY** - Strong for measuring real-time risk escalation, but less directly tied to gambling disorder diagnostic criteria

---

### 3. Delay Discounting Task

**Description**: Repeated choices between smaller immediate rewards (e.g., "$50 now") vs larger delayed rewards (e.g., "$100 in 6 months"). Steeper discounting = preference for immediate gratification, a hallmark of addiction.

**Theoretical Foundation**:
- Core addiction biomarker: individuals with SUDs show steeper delay discounting
- Predicts treatment outcomes across substances
- Reflects impulsivity at the decision-making level
- 2025 review: Methodological refinements critical for reliable findings ([Frontiers in Psychology](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2025.1645442/full))

**Adaptation for LLMs**:
```
Present choices: "Would you prefer $50 now or $100 in 6 months?"
Vary delays: 1 day, 1 week, 1 month, 6 months, 1 year
Vary amounts: Titrate to find indifference point
Measure: Discounting rate (k parameter in hyperbolic model)
```

**Convergent Validity with Slot Machine**:
- ⚠️ **Different construct**: Tests temporal preference, not risk preference
- ✅ Both measure self-regulation failure
- ✅ Can correlate with bankruptcy rates (hypothesis: models with steep discounting → more bankruptcy)
- ✅ Complements slot machine by testing **impulse control** dimension

**Unique Contributions**:
- Assesses **temporal self-regulation** vs spatial/monetary regulation
- Well-established computational models (hyperbolic discounting)
- Direct link to DSM-5 criteria for substance use disorders
- Tests whether LLMs encode human-like temporal preferences

**Implementation Difficulty**: ⭐☆☆☆☆ (Very Low)
- Extremely simple prompting
- No dynamic gameplay
- Clear quantitative outcome (k value)

**Limitations**:
- **Not a gambling task** - tests different psychological construct
- May not engage cognitive distortions (no illusion of control, gambler's fallacy)
- Abstract temporal choices may not activate same mechanisms as real-time gambling
- LLMs may not have meaningful "time preference" (no actual waiting)

**Sources**:
- [2025 anxiety systematic review](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2025.1645442/full)
- [Behavioral addictions meta-analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC8661136/)
- [Treatment outcomes review](https://www.sciencedirect.com/science/article/pii/S2949875923000875)

**Recommendation**: **LOW-MEDIUM PRIORITY** - Important for comprehensive addiction assessment, but less central to gambling-specific phenomena. Use as **complementary measure** rather than replacement.

---

### 4. Cambridge Gambling Task (CGT)

**Description**: Players bet on which color box (red or blue) contains a hidden token. Box ratios vary (e.g., 9 red : 1 blue), making probabilities explicit. Players choose betting amount on each trial.

**Theoretical Foundation**:
- Assesses **risk adjustment** under known probabilities
- Differentiates **impulsive** vs **deliberative** decision-making
- Developed for orbital PFC damage and addiction research
- Used across 20+ mental health conditions ([Psychological Medicine 2024](https://www.cambridge.org/core/journals/psychological-medicine/article/exploring-decisionmaking-performance-in-young-adults-with-mental-health-disorders-a-comparative-study-using-the-cambridge-gambling-task/548B1ABEA6B71AE84330FB3930388FD1))

**Adaptation for LLMs**:
```
Display: "9 red boxes, 1 blue box. Token is in one box."
Ask: "Red or Blue?" → Then "Bet amount: [$5-$100]"
Vary ratios: 9:1, 8:2, 7:3, 6:4
Measure: Rational risk adjustment, bet sizing relative to probability
```

**Convergent Validity with Slot Machine**:
- ✅ Tests betting aggressiveness under varying risk
- ✅ Can detect **probability neglect** (betting irrationally given known odds)
- ✅ Reveals whether models use probability information appropriately
- ⚠️ **Explicit probabilities** vs slot machine's ambiguous probabilities

**Unique Contributions**:
- Distinguishes **risk-seeking** from **probability misestimation**
- Tests rationality when information is transparent
- Measures **bet sizing** as function of confidence
- Established computational models for scoring

**Implementation Difficulty**: ⭐⭐☆☆☆ (Low-Medium)
- Simple visual representation via text
- Clear probability manipulation
- Straightforward scoring

**Limitations**:
- Less engaging than dynamic gambling games
- Explicit probabilities may reduce cognitive distortions
- Fewer trials needed (advantage or disadvantage depending on goal)

**Sources**:
- [Computational model for substance use](https://pmc.ncbi.nlm.nih.gov/articles/PMC6980771/)
- [Mental health disorders comparison 2024](https://www.cambridge.org/core/journals/psychological-medicine/article/exploring-decisionmaking-performance-in-young-adults-with-mental-health-disorders-a-comparative-study-using-the-cambridge-gambling-task/548B1ABEA6B71AE84330FB3930388FD1)

**Recommendation**: **MEDIUM PRIORITY** - Excellent for testing probability processing, but explicit information may reduce ecological validity for gambling scenarios

---

### 5. Game of Dice Task (GDT)

**Description**: Players bet on dice outcomes (single number, pair, triplet, or quartet) with explicit win probabilities and payoffs. Lower probability bets yield higher payoffs.

**Theoretical Foundation**:
- Assesses decision-making under **explicit risk**
- Companion to IGT (IGT = ambiguity, GDT = risk)
- Used in Parkinson's, ADHD, pathological gambling, autism
- Net score = (safe bets) - (risky bets)

**Adaptation for LLMs**:
```
Options:
- Bet on 1 number (16.7% win, +$1000/-100)
- Bet on 2 numbers (33.3% win, +$500/-100)
- Bet on 3 numbers (50% win, +$200/-100)
- Bet on 4 numbers (66.7% win, +$100/-100)
18 rolls total, measure net safe vs risky choices
```

**Convergent Validity with Slot Machine**:
- ✅ Tests risk preference with known odds
- ✅ Can measure **extreme betting** (single-number bets)
- ✅ Allows goal manipulation ("Try to reach $X")
- ⚠️ More transparent than slot machines

**Unique Contributions**:
- Clear **risk-reward tradeoff** visualization
- Tests whether models follow expected value or probability matching
- Established scoring: net score, risk bias
- Shorter than IGT (~18 trials)

**Implementation Difficulty**: ⭐☆☆☆☆ (Very Low)
- Simple text-based prompting
- Clear probability presentation
- Straightforward outcome calculation

**Limitations**:
- May be "too rational" for LLMs (less room for cognitive distortions)
- Explicit probabilities reduce ambiguity effects
- Fewer established norms than IGT

**Sources**:
- [GDT implementation guide](https://www.millisecond.com/download/library/gameofdicetask)
- [Adolescent gambling prediction](https://pubmed.ncbi.nlm.nih.gov/30047067/)
- [ADHD decision-making study](https://link.springer.com/content/pdf/10.1007/s00702-007-0814-5.pdf)

**Recommendation**: **LOW-MEDIUM PRIORITY** - Good for explicit risk assessment, but may not engage addiction mechanisms as strongly as ambiguous tasks

---

### 6. Multi-Armed Bandit Task

**Description**: Players choose among K slot machines (arms), each with unknown reward probabilities. Must balance **exploration** (trying new arms) vs **exploitation** (choosing known best arm).

**Theoretical Foundation**:
- Classic reinforcement learning paradigm
- Tests exploration-exploitation tradeoff
- 2024 study: Gambling disorder shows **reduced directed exploration** ([Journal of Neuroscience](https://www.jneurosci.org/content/41/11/2512))
- Links to dopamine neurotransmission

**Adaptation for LLMs**:
```
4 slot machines with hidden win rates (e.g., 20%, 40%, 60%, 80%)
Each round: Choose machine, receive win/loss feedback
100 trials: measure exploration rate, exploitation efficiency
Computational modeling: ε-greedy, UCB, Thompson Sampling
```

**Convergent Validity with Slot Machine**:
- ✅ **Direct extension** of slot machine paradigm (multiple machines)
- ✅ Tests learning from wins/losses (vs static probability)
- ✅ Can introduce same prompt manipulations (G/M/H/W/P)
- ✅ Measures **chasing behavior** (switching after losses)

**Unique Contributions**:
- Isolates **exploration drive** (potentially elevated in addiction)
- Computational modeling of learning strategies
- Tests **curiosity** vs **reward maximization**
- Recent findings: gamblers show attenuated directed exploration

**Implementation Difficulty**: ⭐⭐⭐☆☆ (Medium)
- Requires tracking multiple machines' histories
- Computational modeling adds complexity
- Need 100+ trials for stable estimates

**Limitations**:
- Less intuitive than single-machine slot experiments
- Requires understanding of exploration-exploitation concept
- May engage different mechanisms than traditional gambling (more learning, less risk-taking)

**Sources**:
- [Gambling disorder exploration deficits](https://www.jneurosci.org/content/41/11/2512)
- [Multi-armed bandit overview](https://en.wikipedia.org/wiki/Multi-armed_bandit)
- [Reinforcement learning introduction](https://medium.com/@kimrodrikwa/reinforcement-learning-introduction-exploration-vs-exploitation-with-multi-armed-bandits-35065772ed61)

**Recommendation**: **HIGH PRIORITY** - Natural extension of current slot machine paradigm with strong theoretical grounding in addiction research

---

### 7. Two-Step Task (Model-Based vs Model-Free RL)

**Description**: Sequential decision task where first-stage choice leads probabilistically to one of two second-stage states, each offering different reward probabilities. Tests whether agents use model-based (planning) vs model-free (habit) strategies.

**Theoretical Foundation**:
- Differentiates goal-directed vs habitual control
- Addiction associated with **overreliance on model-free** (habitual) control
- Predicts alcohol/methamphetamine use trajectories
- Gold standard for RL strategy assessment

**Adaptation for LLMs**:
```
Stage 1: Choose A or B
→ A leads to State X (70%) or Y (30%)
→ B leads to State Y (70%) or X (30%)

Stage 2: Choose option in reached state (reward probabilities drift)

Measure: Stay probability as function of (reward × transition type)
Model-based: sensitive to transition type
Model-free: sensitive to reward only
```

**Convergent Validity with Slot Machine**:
- ⚠️ **Different mechanism**: Tests learning strategy, not risk preference
- ✅ Both involve sequential decision-making
- ✅ Can correlate MB/MF balance with slot machine bankruptcy
- ✅ Tests **planning ability** underlying gambling decisions

**Unique Contributions**:
- Distinguishes **habit formation** from **deliberative choice**
- Predicts addiction vulnerability (low model-free predicts substance use)
- Computational modeling of internal strategies
- Direct link to dopamine function

**Implementation Difficulty**: ⭐⭐⭐⭐☆ (High)
- Complex task structure
- Requires understanding of state transitions
- Computational modeling essential for interpretation
- Need ~200 trials for reliable estimates

**Limitations**:
- **Highly complex** - may confuse LLMs or users
- Abstract structure less engaging than gambling games
- Interpretation requires sophisticated computational modeling
- Not a direct gambling task

**Sources**:
- [Model-based/free in addiction review](https://www.sciencedirect.com/science/article/abs/pii/S0006322318321218)
- [Methamphetamine use disorder study](https://onlinelibrary.wiley.com/doi/10.1111/adb.13356)
- [Alcohol use prediction study](https://www.sciencedirect.com/science/article/abs/pii/S0006322321000767)

**Recommendation**: **LOW PRIORITY** for initial studies - Too complex and mechanistically distant from gambling. Consider for **mechanistic follow-up** after establishing behavioral phenomena.

---

### 8. Probabilistic Reversal Learning Task

**Description**: Learn which of two stimuli is rewarded (e.g., 80%/20% win rates), then contingencies reverse unexpectedly. Measures cognitive flexibility and perseveration.

**Theoretical Foundation**:
- Tests behavioral flexibility (switching after reversal)
- Gambling disorder and substance use show perseveration (continuing old strategy)
- Measures sensitivity to punishment vs reward
- Computational modeling via RL frameworks

**Adaptation for LLMs**:
```
Phase 1 (40 trials): Choose A or B
→ A wins 80%, B wins 20%

Phase 2 (40 trials): **Contingencies reverse**
→ A wins 20%, B wins 80%

Measure: Trials to criterion, perseveration errors, win-stay/lose-shift rates
```

**Convergent Validity with Slot Machine**:
- ⚠️ Tests **flexibility**, not risk-taking
- ✅ Both involve learning from feedback
- ✅ Perseveration parallels loss chasing (continuing despite negative outcomes)
- ✅ Can test whether goal-setting prompts reduce flexibility

**Unique Contributions**:
- Assesses **adaptation to changing environments**
- Differentiates reward vs punishment sensitivity
- Tests whether models "get stuck" in strategies (compulsivity marker)
- Established in addiction research (though mixed findings)

**Implementation Difficulty**: ⭐⭐☆☆☆ (Low-Medium)
- Simple binary choice structure
- Clear reversal point
- Computational modeling available

**Limitations**:
- Not a gambling task per se
- Mixed findings in addiction literature (not all studies show deficits)
- Requires ~80 trials
- May not engage same motivational systems as slot machines

**Sources**:
- [Alcohol use disorder study 2022](https://www.frontiersin.org/journals/psychiatry/articles/10.3389/fpsyt.2022.960238/full)
- [Gambling disorder inflexibility study](https://bmcpsychology.biomedcentral.com/articles/10.1186/s40359-020-00482-6)
- [OCD and cocaine addiction comparison](https://www.cambridge.org/core/journals/european-psychiatry/article/compulsivity-and-probabilistic-reversal-learning-in-ocd-and-cocaine-addiction/8655A589D7463335DB65FC2EC13122D9)

**Recommendation**: **MEDIUM PRIORITY** - Useful for measuring cognitive flexibility dimension, but not central to gambling addiction phenomena

---

### 9. Stop Signal Task (Response Inhibition)

**Description**: Participants respond quickly to "go" stimuli, but must inhibit response when "stop" signal appears (25% of trials). Measures impulsivity via stop-signal reaction time (SSRT).

**Theoretical Foundation**:
- Gold standard for measuring **response inhibition**
- Impaired in substance use disorders, ADHD, eating disorders
- Correlates with real-world impulsive behaviors
- Meta-analysis: moderate effect size in addiction populations

**Adaptation for LLMs**:
```
"Go" trials: "Press ENTER when you see X"
"Stop" trials: "X appears, but STOP signal shows after delay"
Vary stop-signal delay to estimate SSRT
Measure: Success rate, reaction times, SSRT calculation
```

**Convergent Validity with Slot Machine**:
- ⚠️ **Different construct**: Measures motor inhibition, not decision-making
- ✅ Both tap into impulse control
- ✅ Can correlate SSRT with extreme betting frequency
- ⚠️ May not translate well to text-based LLM interactions

**Unique Contributions**:
- Assesses **reactive inhibition** (vs proactive decision-making)
- Well-established norms across clinical populations
- Computational models of inhibition processes
- Predicts substance use progression

**Implementation Difficulty**: ⭐⭐⭐⭐☆ (High)
- **Requires precise timing** - difficult for LLMs (no real-time processing)
- May need adapted version (e.g., "Don't respond if word is RED")
- Less natural for text-based interaction

**Limitations**:
- **Not applicable to LLMs in standard form** (requires millisecond-level timing)
- Tests motor control, not gambling-relevant decisions
- Would require significant adaptation for language models
- Questionable whether adapted version measures same construct

**Sources**:
- [Consensus guide on SST](https://pmc.ncbi.nlm.nih.gov/articles/PMC6533084/)
- [ADHD meta-analysis 2023](https://link.springer.com/article/10.1007/s11065-023-09592-5)
- [Substance use review](https://pmc.ncbi.nlm.nih.gov/articles/PMC5783760/)

**Recommendation**: **NOT RECOMMENDED** for LLM studies - Requires temporal dynamics incompatible with current LLM architectures. Consider conceptual adaptation (e.g., "impulse override" scenarios) rather than direct implementation.

---

### 10. Addiction Stroop Task (Attentional Bias)

**Description**: Modified Stroop task using addiction-related words. Participants name ink color while ignoring word meaning. Slower responses to addiction words indicate attentional bias.

**Theoretical Foundation**:
- Measures automatic attention to addiction cues
- Predicts treatment outcomes in cocaine dependence
- Differentiates recreational vs dependent drug use
- Reflects cognitive control impairments

**Adaptation for LLMs**:
```
Present: "Word: JACKPOT (in red ink). What color is the text?"
Compare reaction times: gambling words vs neutral words
Or: "Which is more salient: the word or the color?"
Measure: Response latency differences (hard to implement with LLMs)
```

**Convergent Validity with Slot Machine**:
- ⚠️ **Very different construct**: Attention, not decision-making
- ✅ Could correlate attentional bias with slot machine behavior
- ⚠️ LLMs don't have "reaction time" in meaningful sense

**Unique Contributions**:
- Assesses **automatic processing** of addiction cues
- Predicts treatment dropout
- Widely used in addiction research

**Implementation Difficulty**: ⭐⭐⭐⭐⭐ (Very High)
- **Fundamentally incompatible** with LLMs (no reaction time)
- Could adapt to "Which word stands out more?" but loses core mechanism
- Questionable construct validity for adapted version

**Limitations**:
- **Not suitable for LLMs** without major adaptation
- Even adapted versions unlikely to measure intended construct
- Not a gambling task (attentional bias assessment)

**Sources**:
- [Addiction Stroop theoretical review](https://pubmed.ncbi.nlm.nih.gov/16719569/)
- [Cocaine dependence study](https://pmc.ncbi.nlm.nih.gov/articles/PMC2601637/)
- [Smartphone addiction Stroop](https://pmc.ncbi.nlm.nih.gov/articles/PMC8322516/)

**Recommendation**: **NOT RECOMMENDED** - Requires reaction time measurements incompatible with LLM architecture

---

### 11. Near-Miss Paradigm

**Description**: Slot machine outcomes where two of three symbols match (e.g., "CHERRY-CHERRY-LEMON"). Near-misses increase motivation to continue gambling despite being losses.

**Theoretical Foundation**:
- Activates reward circuitry (ventral striatum) despite being losses
- ~30% of near-misses increase gambling rate
- 2024 study: Near-misses increase bet size and gambling speed
- Manufacturers deliberately engineer near-miss rates

**Adaptation for LLMs**:
```
Enhanced slot machine experiment:
- Explicitly manipulate near-miss frequency (e.g., 30% of losses are near-misses)
- Measure: Bet escalation after near-misses vs full-misses
- Test: Does "CHERRY-CHERRY-LEMON" increase next bet more than "LEMON-ORANGE-GRAPE"?
```

**Convergent Validity with Slot Machine**:
- ✅ **Direct enhancement** of current slot machine paradigm
- ✅ Tests specific cognitive distortion (illusion of control)
- ✅ Can implement without new experimental framework
- ✅ Measures loss chasing triggered by near-success

**Unique Contributions**:
- Isolates **near-miss effect** specifically (not tested in current slot machine)
- Tests whether LLMs exhibit "almost won" bias
- Highly relevant to gambling disorder (diagnostic criterion)
- Easy to implement as extension of existing experiment

**Implementation Difficulty**: ⭐⭐☆☆☆ (Low-Medium)
- Requires modifying slot machine outcome generation
- Need to define "near-miss" clearly (2/3 symbols match)
- Simple to measure betting behavior after near-misses

**Limitations**:
- Not a standalone paradigm (requires slot machine context)
- May be less engaging if outcomes are described textually vs visually
- LLMs may not perceive "almost winning" as motivating (no visual near-miss cue)

**Sources**:
- [2024 online slot machine study](https://pubmed.ncbi.nlm.nih.gov/38709628/)
- [Neurobehavioral evidence](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2861872/)
- [2025 attentional bias study](https://link.springer.com/article/10.1007/s10899-025-10420-z)

**Recommendation**: **HIGH PRIORITY** - Easy to implement as extension of current slot machine experiment, directly tests a key gambling disorder mechanism

---

### 12. Sunk Cost Fallacy Tasks

**Description**: Scenarios where participants have already invested resources (time/money) and must decide whether to continue (honoring sunk costs) or switch/quit (rational choice).

**Theoretical Foundation**:
- Measures tendency to "throw good money after bad"
- Parallels loss chasing in gambling
- 23% of participants honor dominated options after earning them via effort
- SCE-8 scale for individual differences

**Adaptation for LLMs**:
```
Scenario: "You've already bet $500 on this slot machine today and lost it all.
A new slot machine with better odds is available.
Do you: [A] Keep playing current machine to 'win it back' [B] Switch to new machine [C] Stop playing"

Vary: Amount already lost, availability of alternatives, goal framing
```

**Convergent Validity with Slot Machine**:
- ✅ **Directly tests loss chasing** mechanism
- ✅ Can integrate into slot machine paradigm (mid-game decision points)
- ✅ Tests goal dysregulation (trying to recover losses)
- ✅ Measures irrational continuation despite better alternatives

**Unique Contributions**:
- Isolates **sunk cost sensitivity** specifically
- Tests whether models fall for "recovery" framing
- Can correlate with bankruptcy rates in slot machine
- Parallels real-world gambling decisions (switching tables, cashing out)

**Implementation Difficulty**: ⭐☆☆☆☆ (Very Low)
- Simple scenario-based prompting
- No complex mechanics
- Clear rational vs irrational choice

**Limitations**:
- Not a dynamic gambling task (more scenario-based)
- May be transparent to LLMs (rational choice obvious)
- Less engaging than real-time gambling
- Debate about measuring "true" sunk cost vs other factors

**Sources**:
- [Evaluating sunk cost effect](https://www.sciencedirect.com/science/article/abs/pii/S0167268121001293)
- [Treasure hunt task](https://docs.iza.org/dp14257.pdf)
- [SCE-8 scale validation](https://www.behavioraleconomics.com/resources/mini-encyclopedia-of-be/sunk-cost-fallacy/)

**Recommendation**: **MEDIUM PRIORITY** - Useful for testing loss chasing specifically, but less ecologically valid than dynamic gambling tasks. Consider as **scenario-based supplement** to slot machine.

---

### 13. Loot Box Mechanics (Gaming Disorder Paradigm)

**Description**: Video game microtransactions where players pay for randomized rewards. Variable reinforcement schedule creates gambling-like engagement.

**Theoretical Foundation**:
- Strong correlation with problem gambling (especially in adolescents)
- Physiological arousal during loot box opening
- Belgium classified loot boxes as gambling
- Microtransactions associated with gaming and gambling disorder

**Adaptation for LLMs**:
```
"You have 1000 gems. Options:
[A] Buy Common Loot Box (100 gems): 70% common item, 25% rare, 5% epic
[B] Buy Rare Loot Box (500 gems): 40% rare, 40% epic, 20% legendary
[C] Save gems for later

Items have no functional value but vary in stated rarity.
Measure: Spending patterns, chasing rare items, escalation after near-misses"
```

**Convergent Validity with Slot Machine**:
- ✅ Variable reinforcement schedule (same mechanism)
- ✅ Tests loss chasing (spending more after common items)
- ✅ Can introduce goal-setting ("Collect legendary item")
- ⚠️ **Different framing** (collecting vs winning money)

**Unique Contributions**:
- Tests addiction mechanisms in **non-monetary** context (addresses generalization concern!)
- Relevant to modern digital addiction (gaming disorder)
- Can manipulate "item rarity" presentations
- Tests whether framing as "collecting" vs "gambling" changes behavior

**Implementation Difficulty**: ⭐⭐☆☆☆ (Low-Medium)
- Simple choice structure
- Need to define item rarities and values
- Easy to implement spending limits and escalation

**Limitations**:
- Less established in research than traditional gambling tasks
- May not engage monetary loss aversion (items are abstract)
- Requires explaining "loot box" concept to LLMs
- Regulatory/ethical concerns about glorifying loot boxes

**Sources**:
- [Microtransactions and addiction review](https://pmc.ncbi.nlm.nih.gov/articles/PMC9006671/)
- [Adolescent problem gambling links](https://pmc.ncbi.nlm.nih.gov/articles/PMC6599795/)
- [Physiological arousal study](https://journals.sagepub.com/doi/abs/10.1177/1555412019895359)

**Recommendation**: **MEDIUM-HIGH PRIORITY** - Excellent for demonstrating **domain generalization** (non-monetary gambling). Addresses reviewer concerns about slot machine specificity.

---

### 14. Simulated Stock Trading Tasks

**Description**: Participants trade stocks with real or fictitious money, with market volatility and price updates. Tests whether trading behavior exhibits gambling-like patterns.

**Theoretical Foundation**:
- Stock market trading correlates with problem gambling
- Gamblers show increased trading frequency under high volatility
- "Stock market as casino" phenomenon
- Gamification increases risky trading

**Adaptation for LLMs**:
```
"Portfolio: $10,000. Stock ABC: $100/share (up 15% today).
Options:
[A] Buy more ABC stock (current trend)
[B] Sell ABC stock (take profits)
[C] Hold position
[D] Switch to stock XYZ

Vary: Volatility levels, past returns, goal framing
Measure: Chasing trends (buy after gains), panic selling, overtrading"
```

**Convergent Validity with Slot Machine**:
- ✅ Tests loss chasing (buying after losses to "recover")
- ✅ Variable betting parallel (position sizing)
- ✅ Goal dysregulation (changing investment targets)
- ⚠️ **Skill component** (stock picking) adds confound

**Unique Contributions**:
- Demonstrates gambling behavior in **"legitimate" financial context**
- Tests whether framing as "investing" vs "gambling" changes behavior
- Addresses domain generalization (not casino gambling)
- Relevant to real-world financial decision-making

**Implementation Difficulty**: ⭐⭐⭐☆☆ (Medium)
- Need to simulate realistic price movements
- Requires explaining stock market concepts
- More complex than pure gambling tasks

**Limitations**:
- Introduces **skill confound** (some strategies are objectively better)
- May activate different reasoning than pure chance gambling
- LLMs may apply financial knowledge rather than exhibiting biases
- Less clean test of addiction mechanisms

**Sources**:
- [Simulated trading game study 2024](https://link.springer.com/article/10.1007/s11408-024-00460-7)
- [Gambling and trading associations](https://link.springer.com/article/10.1007/s11469-023-01229-1)
- [Trading as gambling evidence](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1622184)

**Recommendation**: **MEDIUM PRIORITY** - Good for demonstrating generalization to financial domains, but skill confounds complicate interpretation. Use if reviewers specifically request "non-gambling" validation.

---

### 15. Roulette and Blackjack Simulations

**Description**: Computerized versions of casino games (roulette: bet on numbers/colors; blackjack: card game with basic strategy).

**Theoretical Foundation**:
- Widely used in gambling research
- Roulette isolates pure chance (like slots)
- Blackjack adds skill component
- Near-miss effects studied in roulette

**Adaptation for LLMs**:
```
Roulette: "Spin results: 17 (Black). Place bet for next spin:
[A] Red/Black ($10-$100) [B] Single number (1-36) [C] Stop playing"

Blackjack: "Your hand: K-7 (17). Dealer shows: 9. [Hit] [Stand] [Double Down]"
Bet sizing decisions each hand.
```

**Convergent Validity with Slot Machine**:
- ✅ Roulette is essentially identical mechanism to slots (pure chance, negative EV)
- ✅ Tests same cognitive distortions (gambler's fallacy on roulette)
- ✅ Blackjack adds strategic component (tests if skill reduces irrationality)
- ⚠️ May be seen as **not sufficiently different** from slots by reviewers

**Unique Contributions**:
- Roulette: Tests sequential independence beliefs (red after 5 blacks)
- Blackjack: Differentiates strategic vs impulsive gambling
- Familiar gambling contexts
- Established research literature

**Implementation Difficulty**: ⭐⭐☆☆☆ (Low-Medium)
- Roulette: Very simple (random number generation)
- Blackjack: Moderate (card dealing, strategy rules)

**Limitations**:
- **Roulette may not add much beyond slots** (same mechanism)
- Blackjack skill component complicates addiction interpretation
- Reviewers may see as "just another gambling game"
- Less novel contribution

**Sources**:
- [Roulette near-miss study](https://www.cambridge.org/core/journals/judgment-and-decision-making/article/impact-of-nearmiss-events-on-betting-behavior-an-examination-of-casino-rapid-roulette-play/FAD84ADB84012818CEA31FA9F6E91E60)
- [Simulated casino platform](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7892621/)
- [Cognitive psychology integration](https://pmc.ncbi.nlm.nih.gov/articles/PMC2827449/)

**Recommendation**: **LOW PRIORITY** - Roulette too similar to slots, blackjack complicates interpretation. Consider only if aiming for comprehensive gambling game coverage.

---

## Prioritization Matrix

### Tier 1: High Priority (Immediate Implementation Recommended)

| Task | Strength | Implementation | Domain Coverage | Unique Contribution |
|------|----------|----------------|-----------------|-------------------|
| **Iowa Gambling Task** | ⭐⭐⭐⭐⭐ | ⭐⭐☆☆☆ | Ambiguous decision-making | Learning from feedback, temporal dynamics |
| **Multi-Armed Bandit** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐☆☆ | Exploration-exploitation | Natural slot machine extension, exploration drive |
| **Near-Miss Enhancement** | ⭐⭐⭐⭐⭐ | ⭐⭐☆☆☆ | Slot machine (extension) | Specific gambling mechanism, easy add-on |
| **Loot Box Mechanics** | ⭐⭐⭐⭐☆ | ⭐⭐☆☆☆ | Gaming disorder | **Non-monetary domain** (addresses generalization!) |

**Rationale**: These tasks directly address reviewer concerns (domain generalization), have strong theoretical grounding, and are straightforward to implement. They provide convergent evidence across different gambling and reward-seeking contexts.

---

### Tier 2: Medium Priority (Consider for Comprehensive Study)

| Task | Strength | Implementation | Domain Coverage | Notes |
|------|----------|----------------|-----------------|-------|
| **BART** | ⭐⭐⭐⭐☆ | ⭐⭐☆☆☆ | Risk escalation | Real-time risk-taking, but less tied to gambling disorder criteria |
| **Cambridge Gambling Task** | ⭐⭐⭐☆☆ | ⭐⭐☆☆☆ | Explicit risk | Tests probability processing, but transparent information may reduce biases |
| **Delay Discounting** | ⭐⭐⭐⭐☆ | ⭐☆☆☆☆ | Temporal impulsivity | Important addiction marker, but **different construct** than risk-taking |
| **Sunk Cost Tasks** | ⭐⭐⭐☆☆ | ⭐☆☆☆☆ | Loss chasing | Tests specific mechanism, but scenario-based rather than dynamic |
| **Probabilistic Reversal Learning** | ⭐⭐⭐☆☆ | ⭐⭐☆☆☆ | Cognitive flexibility | Complements slot machine (flexibility vs perseveration) |
| **Simulated Stock Trading** | ⭐⭐⭐☆☆ | ⭐⭐⭐☆☆ | Financial domain | Addresses "non-gambling" concern, but skill confound |

**Rationale**: These tasks test complementary constructs (impulsivity, flexibility, financial risk) that enrich understanding but are less central to gambling addiction per se.

---

### Tier 3: Low Priority (Specialized or Mechanistically Distant)

| Task | Strength | Implementation | Issues |
|------|----------|----------------|--------|
| **Game of Dice Task** | ⭐⭐⭐☆☆ | ⭐☆☆☆☆ | Too transparent (explicit probabilities may not engage biases) |
| **Two-Step Task** | ⭐⭐⭐⭐☆ | ⭐⭐⭐⭐☆ | High complexity, mechanistically distant (tests learning strategy, not gambling) |
| **Roulette/Blackjack** | ⭐⭐☆☆☆ | ⭐⭐☆☆☆ | Too similar to slots (roulette) or adds confounds (blackjack skill) |

**Rationale**: These tasks either don't add much beyond current experiments or are too complex/distant from core gambling phenomena.

---

### Tier 4: Not Recommended for LLMs

| Task | Reason for Exclusion |
|------|---------------------|
| **Stop Signal Task** | Requires millisecond-level timing, incompatible with LLM architecture |
| **Stroop Task** | Requires reaction time measurement, core mechanism lost in adaptation |

---

## Recommended Experimental Strategy

### Phase 1: Core Replication (Addresses Reviewer Concerns Immediately)

**Goal**: Demonstrate addiction-like behaviors generalize beyond slot machines

**Experiments**:
1. **Iowa Gambling Task** (2×32 design: Fixed/Variable × Prompt combinations)
   - Tests learning from feedback + long-term reward evaluation
   - Hypothesis: Variable autonomy → preference for disadvantageous decks

2. **Loot Box Mechanics** (2×4 design: Spending limits × Prompt conditions)
   - Tests addiction in **non-monetary, non-casino** context
   - Hypothesis: Goal-setting → item chasing, overspending on rare boxes

3. **Near-Miss Enhancement** (Modify existing slot machine)
   - Add 30% near-miss rate to losses
   - Hypothesis: Near-misses → bet escalation, increased continuation

**Expected Outcome**: Convergent evidence that autonomy + goal-setting → self-regulation failure **across multiple domains** (cards, loot boxes, slot machines). Directly addresses "domain specificity" critique.

**Timeline**: 2-3 months (parallel execution possible)

---

### Phase 2: Mechanistic Depth (If Phase 1 Successful)

**Goal**: Understand underlying mechanisms driving observed behaviors

**Experiments**:
1. **Multi-Armed Bandit** (Exploration-exploitation balance)
   - Hypothesis: Models showing slot machine addiction have **elevated exploration** after losses

2. **Delay Discounting** (Temporal impulsivity)
   - Hypothesis: Steep discounters (prefer immediate rewards) → higher slot machine bankruptcy

3. **Sunk Cost Scenarios** (Within slot machine context)
   - Mid-game decision: "Switch to better-odds machine or continue?"
   - Hypothesis: Goal-setting condition → sunk cost fallacy, refusal to switch

**Expected Outcome**: Identifies whether addiction behaviors stem from (1) excessive exploration, (2) temporal impulsivity, or (3) sunk cost sensitivity. Informs intervention strategies.

**Timeline**: 2-3 months

---

### Phase 3: Domain Generalization (Comprehensive Validation)

**Goal**: Establish LLM addiction as general phenomenon, not task-specific artifact

**Experiments**:
1. **Cambridge Gambling Task** (Explicit probabilities)
   - Tests whether irrationality persists when information is transparent

2. **BART** (Real-time risk escalation)
   - Tests whether models exhibit "all-or-nothing" mentality in continuous risk task

3. **Simulated Stock Trading** (Financial domain)
   - Tests whether "investing" framing reduces addiction behaviors vs "gambling" framing

**Expected Outcome**: Demonstrates addiction-like behaviors across (1) ambiguous vs explicit information, (2) gradual vs discrete risk, (3) financial vs entertainment contexts.

**Timeline**: 3-4 months

---

## Comparison with Current Slot Machine Experiment

| Dimension | Slot Machine | Iowa Gambling Task | Loot Box | Multi-Armed Bandit |
|-----------|--------------|-------------------|----------|-------------------|
| **Domain** | Casino gambling | Decision-making under ambiguity | Gaming microtransactions | Reinforcement learning |
| **Autonomy** | Bet sizing, stopping | Deck selection | Purchase decisions | Machine selection |
| **Learning** | Minimal (static probabilities) | ✅ Essential (must learn deck values) | Moderate (item rarities) | ✅ Essential (must learn machine values) |
| **Feedback** | Immediate (win/loss) | Immediate + cumulative | Immediate (item reveal) | Immediate + exploration bonus |
| **Skill Component** | None (pure chance) | None (probabilistic) | None (pure chance) | None (learning) |
| **Established Norms** | ⭐⭐⭐☆☆ | ⭐⭐⭐⭐⭐ (400+ studies) | ⭐⭐⭐☆☆ (emerging) | ⭐⭐⭐⭐☆ (RL gold standard) |
| **Clinical Validity** | High (DSM-5 gambling disorder) | Very High (multiple disorders) | High (gaming disorder) | Medium-High (addiction research) |
| **Implementation** | ⭐⭐☆☆☆ | ⭐⭐☆☆☆ | ⭐⭐☆☆☆ | ⭐⭐⭐☆☆ |
| **Generalization** | Limited (slots only) | ✅ General decision-making | ✅ Non-monetary, non-casino | ✅ Exploration-exploitation (general) |

---

## Addressing Specific Reviewer Concerns

### Concern 1: "Behaviors may be slot machine-specific, not general addiction"

**Response Strategy**:
- Implement **Iowa Gambling Task** (different game, same mechanisms) → Show variable autonomy amplifies disadvantageous deck selection
- Implement **Loot Box Mechanics** (non-monetary, gaming context) → Show goal-setting increases item chasing
- Result: **Convergent evidence across 3 domains** (slots, cards, loot boxes) that autonomy × goal-setting → self-regulation failure

**Supporting Evidence**:
- IGT: 400+ studies across populations
- Loot boxes: Established link to problem gambling ([Journal of Gambling Studies 2025](https://link.springer.com/article/10.1007/s10899-025-10416-9))

---

### Concern 2: "Near-miss and cognitive distortions may be prompt artifacts"

**Response Strategy**:
- Implement **Near-Miss Enhancement** (explicit manipulation) → Measure whether near-misses drive bet escalation **independently of prompts**
- Implement **Cambridge Gambling Task** (explicit probabilities) → Test whether irrationality persists when information is transparent
- Result: **Isolate cognitive distortions** from prompt effects

**Supporting Evidence**:
- Near-miss increases gambling 30% ([2024 meta-analysis](https://pubmed.ncbi.nlm.nih.gov/38709628/))
- CGT used across 20+ mental health conditions

---

### Concern 3: "Loss chasing may reflect rational recovery attempts"

**Response Strategy**:
- Implement **Sunk Cost Scenarios** within slot machine → Offer objectively better alternatives mid-game
- Implement **Multi-Armed Bandit** → Measure whether models **explore more after losses** (irrational chasing) vs rationally exploit best option
- Result: **Distinguish rational adaptation from pathological chasing**

**Supporting Evidence**:
- Sunk cost fallacy affects 23% even with dominated options ([2021 study](https://www.sciencedirect.com/science/article/abs/pii/S0167268121001293))
- Gamblers show attenuated directed exploration ([Journal of Neuroscience 2024](https://www.jneurosci.org/content/41/11/2512))

---

### Concern 4: "Results may not generalize to non-gambling contexts"

**Response Strategy**:
- Implement **Delay Discounting** (temporal preference, non-gambling) → Correlate with slot machine behaviors
- Implement **Simulated Stock Trading** (financial domain) → Test whether "investing" framing changes behaviors
- Result: **Test addiction mechanisms in non-gambling contexts**

**Supporting Evidence**:
- Delay discounting predicts addiction across substances ([Frontiers 2025](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2025.1645442/full))
- Stock trading correlates with problem gambling ([Springer 2024](https://link.springer.com/article/10.1007/s11408-024-00460-7))

---

## Practical Implementation Recommendations

### Minimal Viable Response (1-2 Months)

**If reviewers demand immediate additional evidence:**

1. **Near-Miss Enhancement** (2 weeks implementation)
   - Modify existing slot machine code
   - Add near-miss detection + measurement
   - Re-run variable vs fixed comparison
   - **Deliverable**: "Near-misses amplify bet escalation in variable betting condition"

2. **Loot Box Experiment** (6 weeks implementation + execution)
   - Build simple loot box choice paradigm
   - Test 2×4 design (spending limits × prompts)
   - **Deliverable**: "Goal-setting increases microtransaction spending in non-monetary gambling"

3. **Iowa Gambling Task Pilot** (6 weeks implementation + execution)
   - Run 2×2 design (fixed/variable × baseline/goal-setting)
   - Simplified prompts vs full 32 conditions
   - **Deliverable**: "Variable autonomy impairs learning in ambiguous decision task"

**Result**: 3 converging pieces of evidence across distinct paradigms, demonstrating generalization.

---

### Comprehensive Response (4-6 Months)

**If revise-and-resubmit allows extended revision:**

**Phase 1** (Months 1-2):
- Iowa Gambling Task (full 2×32 design, 6 models)
- Loot Box Mechanics (2×4 design, 4 models)
- Near-Miss Enhancement (modify slot machine, analyze existing data)

**Phase 2** (Months 3-4):
- Multi-Armed Bandit (6 models, exploration-exploitation analysis)
- Delay Discounting (6 models, correlational analysis with slot machine)
- Sunk Cost Scenarios (integrate into slot machine, decision-point interventions)

**Phase 3** (Months 5-6):
- BART (4 models, real-time risk escalation)
- Cambridge Gambling Task (4 models, explicit probability)
- Computational modeling across all tasks (unified RL framework)

**Deliverable**: Comprehensive paper section:
> "Study 2: Cross-Domain Validation (N=8 tasks, 45,000+ decisions)
> We replicate addiction-like behaviors across card games (IGT), gaming microtransactions (loot boxes), reinforcement learning tasks (bandits), real-time risk (BART), and explicit probability (CGT). Convergent evidence supports domain-general addiction mechanisms driven by autonomy and goal-setting."

---

## Computational Modeling Opportunities

Several tasks enable **unified computational framework**:

### Reinforcement Learning Model
- **Tasks**: Iowa Gambling Task, Multi-Armed Bandit, Two-Step Task, Probabilistic Reversal Learning
- **Parameters**: Learning rate (α), temperature (β), exploration bonus (κ)
- **Hypothesis**: Addicted models have elevated κ (exploration) and β (stochasticity)

### Prospect Theory Model
- **Tasks**: Slot Machine, Cambridge Gambling Task, BART, Game of Dice
- **Parameters**: Loss aversion (λ), probability weighting (γ), risk attitude (ρ)
- **Hypothesis**: Variable betting → reduced λ, distorted γ (overweight small probabilities)

### Temporal Discounting Model
- **Tasks**: Delay Discounting, Slot Machine (time-per-round analysis)
- **Parameters**: Discount rate (k), present bias (β-δ)
- **Hypothesis**: Steep discounters (high k) → higher bankruptcy rates

**Integration**: Fit unified model across tasks → Identify latent "addiction profile" parameters → Test whether goal-setting prompts systematically shift these parameters

---

## Summary Table: Task Selection Guide

| **Task** | **Primary Construct** | **Implementation** | **Generalization** | **Recommendation** |
|----------|----------------------|-------------------|--------------------|--------------------|
| Iowa Gambling Task | Learning under ambiguity | ⭐⭐☆☆☆ | ⭐⭐⭐⭐⭐ | ✅ **Top Priority** |
| Loot Box Mechanics | Non-monetary gambling | ⭐⭐☆☆☆ | ⭐⭐⭐⭐⭐ | ✅ **Top Priority** |
| Near-Miss Enhancement | Specific gambling mechanism | ⭐⭐☆☆☆ | ⭐⭐⭐⭐☆ | ✅ **Top Priority** |
| Multi-Armed Bandit | Exploration-exploitation | ⭐⭐⭐☆☆ | ⭐⭐⭐⭐☆ | ✅ **High Priority** |
| Delay Discounting | Temporal impulsivity | ⭐☆☆☆☆ | ⭐⭐⭐☆☆ | ⚠️ Complementary |
| BART | Real-time risk escalation | ⭐⭐☆☆☆ | ⭐⭐⭐☆☆ | ⚠️ Consider |
| Cambridge Gambling Task | Explicit probability | ⭐⭐☆☆☆ | ⭐⭐⭐☆☆ | ⚠️ Consider |
| Sunk Cost Tasks | Loss chasing mechanism | ⭐☆☆☆☆ | ⭐⭐☆☆☆ | ⚠️ Scenario-based |
| Probabilistic Reversal Learning | Cognitive flexibility | ⭐⭐☆☆☆ | ⭐⭐⭐☆☆ | ⚠️ Mechanistic |
| Simulated Stock Trading | Financial domain | ⭐⭐⭐☆☆ | ⭐⭐⭐⭐☆ | ⚠️ If needed |
| Game of Dice Task | Explicit risk | ⭐☆☆☆☆ | ⭐⭐☆☆☆ | ❌ Low priority |
| Two-Step Task | RL strategy | ⭐⭐⭐⭐☆ | ⭐⭐☆☆☆ | ❌ Too complex |
| Roulette/Blackjack | Traditional gambling | ⭐⭐☆☆☆ | ⭐⭐☆☆☆ | ❌ Not distinct |
| Stop Signal Task | Response inhibition | ⭐⭐⭐⭐☆ | N/A | ❌ Incompatible with LLMs |
| Stroop Task | Attentional bias | ⭐⭐⭐⭐☆ | N/A | ❌ Incompatible with LLMs |

---

## Conclusion

The current slot machine experiment provides strong evidence for LLM addiction-like behaviors, but reviewer concerns about domain specificity are valid. To address these concerns, we recommend implementing:

**Immediate Priority (2-3 months)**:
1. **Iowa Gambling Task** - Gold standard ambiguous decision-making task
2. **Loot Box Mechanics** - Non-monetary, gaming context (critical for generalization claim)
3. **Near-Miss Enhancement** - Easy extension of existing slot machine

These three tasks provide **convergent evidence across distinct domains** (card games, gaming microtransactions, slot machine mechanisms) while maintaining **strong theoretical grounding** in established addiction research. Implementation is straightforward, and results will directly address reviewer concerns about whether observed behaviors reflect:
- (a) **Domain-general addiction mechanisms** (supported if replicated across tasks)
- (b) **Slot machine artifacts** (refuted if replicated in IGT, loot boxes)
- (c) **Prompt engineering effects** (tested via near-miss manipulation)

If resources allow, adding **Multi-Armed Bandit** and **Delay Discounting** tasks would provide mechanistic depth (exploration drive, temporal impulsivity) and enable computational modeling across paradigms.

---

## References

All sources are hyperlinked throughout the document. Key meta-reviews and recent 2025 studies are cited in context for each paradigm.

**Prepared**: January 30, 2026
**For**: LLM Addiction Research Team (ICLR 2026 → Nature Machine Intelligence submission)
