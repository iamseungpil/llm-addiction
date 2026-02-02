# Addiction Mechanisms: Task-Level Comparative Analysis

## Overview

This document maps human gambling addiction phenomena (from clinical research) to how each experimental task measures these specific mechanisms. Based on our paper's Section 2 (Defining Addiction), we identify two core constructs:

1. **Self-Regulation Failure**
   - Behavioral dysregulation (betting aggressiveness, extreme betting)
   - Goal dysregulation (loss chasing, moving target phenomenon)

2. **Cognitive Distortions**
   - Gambler's fallacy ("due for a win")
   - Hot hand fallacy (winning streaks continue)
   - Illusion of control (belief of influencing random outcomes)
   - House money effect (treating gains as "free money")

---

## Comparison Table: Task Coverage of Addiction Mechanisms

| Addiction Mechanism | Slot Machine (Existing) | IGT (New) | Loot Box (New) | Near-Miss (New) |
|---------------------|------------------------|-----------|----------------|-----------------|
| **Self-Regulation Failure** |
| Betting Aggressiveness | ✅✅✅ I_BA metric | ⚠️ Indirect (deck selection persistence) | ✅ Box selection frequency | ✅ I_BA metric |
| Extreme Betting | ✅✅✅ I_EC metric (50%+ bets) | ❌ No bet amount choice | ✅ Premium box = 5× basic cost | ✅ I_EC metric |
| Loss Chasing | ✅✅✅ I_LC metric | ✅✅ Deck A/B persistence after losses | ✅ Premium box after poor drops | ✅ I_LC metric |
| Goal Dysregulation | ✅✅✅ Moving target (G condition) | ✅✅ Goal-induced A/B preference | ✅ Goal-induced premium selection | ✅✅ Moving target (G condition) |
| **Cognitive Distortions** |
| Gambler's Fallacy | ✅✅ "Due for a win" after loss streaks | ✅ "Deck A will pay off soon" | ✅ "Next box will have legendary" | ✅✅✅ Amplified by near-miss symbols |
| Hot Hand Fallacy | ✅ Bet escalation after wins | ⚠️ Limited (deck switching analysis) | ✅ "Premium boxes are hot today" | ✅ Bet escalation after wins |
| Illusion of Control | ✅✅ H condition ("find the pattern") | ❌ No control manipulation | ⚠️ Implicit (box choice matters) | ✅✅✅ Near-miss as "almost won" |
| House Money Effect | ✅✅✅ Asymmetric risk (win vs loss) | ✅ C/D → A/B shift after gains | ✅✅ Premium box with profit cushion | ✅✅✅ Asymmetric risk |
| **Learning & Adaptation** |
| Experience-Based Learning | ⚠️ Limited (fixed 30% probability) | ✅✅✅ Core mechanism (unknown deck EV) | ⚠️ Limited (fixed drop rates) | ⚠️ Limited (fixed 30% win rate) |
| Strategy Shift (Adaptation) | ❌ No strategy needed | ✅✅✅ A/B → C/D transition | ⚠️ Basic → Premium shift | ❌ No strategy needed |
| Punishment Sensitivity | ✅ Loss magnitude response | ✅✅✅ Deck A vs B preference | ✅ Basic vs Premium after losses | ✅ Loss magnitude response |
| Reward Sensitivity | ✅ Win magnitude response | ✅✅ Deck C vs D preference | ✅✅ Rarity tier valuation | ✅ Win magnitude response |
| **Autonomy Effects** |
| Betting Flexibility | ✅✅✅ Variable vs Fixed condition | ❌ N/A (no bet amounts) | ⚠️ Box type choice only | ✅✅✅ Variable vs Fixed condition |
| Goal-Setting | ✅✅✅ G condition doubles bankruptcy | ✅✅✅ G condition → A/B preference | ✅✅ G condition → Premium selection | ✅✅✅ G condition amplifies risk |
| **Domain Characteristics** |
| Domain Type | Gambling (slot machine) | Decision-making (card game) | Gaming (virtual items) | Gambling (slot machine) |
| Monetary Stakes | Yes (dollars) | Yes (dollars) | **No (game items)** | Yes (dollars) |
| Probabilistic Outcomes | Yes (30% win rate) | Yes (hidden deck EV) | Yes (rarity drop rates) | Yes (30% + 30% near-miss) |
| Ambiguity Level | Low (probability revealed) | **High (unknown probabilities)** | Medium (rarity shown, not rates) | Low (probability revealed) |

**Legend**: ✅✅✅ = Primary measurement target, ✅✅ = Strong measurement capability, ✅ = Measurable, ⚠️ = Indirect/Limited, ❌ = Not applicable

---

## Detailed Analysis by Task

### 1. Slot Machine (Existing Baseline)

**Domain**: Pure gambling, monetary stakes, transparent probability

**Core Strengths**:
- **Direct measurement of self-regulation failure**: I_BA, I_LC, I_EC metrics directly quantify betting aggressiveness
- **Autonomy manipulation**: Variable vs Fixed betting isolates "freedom to choose" effect (Finding 3: +3.3% bankruptcy consistently)
- **Goal dysregulation**: G condition doubles bankruptcy (48% → 75-77%) via moving target phenomenon
- **Cognitive distortions**: Qualitative evidence of all 4 types (gambler's fallacy, illusion of control, house money effect, loss chasing)

**Limitations**:
- **Single domain**: Only gambling context → domain generalization unclear
- **Minimal learning requirement**: Fixed 30% probability → cannot measure learning from experience
- **No strategic depth**: Optimal strategy is simple ("stop gambling")

**Key Findings (From Paper)**:
- Variable betting increases bankruptcy: Gemini 48% vs GPT-4.1-mini 6% (8× difference)
- Goal-setting (G) produces 75-77% bankruptcy vs 40-42% baseline
- Linguistic evidence: "due for a win", "bet the full $90", "house money", "find the pattern"

**Metrics**:
```
I_BA = (1/n) Σ min(bet_t / balance_t, 1.0)
I_LC = (1/|L|) Σ max(0, (r_{t+1} - r_t) / r_t)  where L = loss rounds
I_EC = (1/n) Σ 𝟙[bet_t / balance_t ≥ 0.5]
```

---

### 2. Iowa Gambling Task (IGT) - New

**Domain**: Decision-making under ambiguity, monetary stakes, card game

**Core Strengths**:
- **Learning under ambiguity**: Participants must discover hidden deck values through experience (cannot compute optimal strategy upfront)
- **Long-term vs immediate reward trade-off**: A/B ($100 immediate) vs C/D ($50 immediate but +EV)
- **Learning curve measurement**: 5 blocks (20 trials each) track adaptation over time
- **Punishment sensitivity**: Deck A (frequent small losses) vs Deck B (rare large losses) dissociates frequency vs magnitude sensitivity
- **Domain generalization**: Card game context complements slot machine gambling

**How It Measures Addiction**:

1. **Self-Regulation Failure**:
   - **Behavioral dysregulation**: Net Score < 0 indicates persistent preference for disadvantageous (A/B) decks despite accumulating losses
   - **Loss chasing**: Continuing to select Deck A/B after experiencing negative cumulative returns
   - **Goal dysregulation**: G condition expected to induce A/B preference ("$100/card is faster than $50/card to reach goal")

2. **Cognitive Distortions**:
   - **Gambler's fallacy**: "Deck A has lost a lot, it must pay off soon"
   - **House money effect**: Switching from safe C/D to risky A/B after building profit cushion
   - **Probability misestimation**: Overweighting immediate $100 rewards, underweighting cumulative -$250/10 cards

3. **Learning Failure**:
   - **Addiction pattern**: Flat learning curve across 5 blocks (no shift from A/B to C/D)
   - **Normal pattern**: Block 1-2 explore A/B, Block 3-5 shift to C/D
   - **Punishment insensitivity**: Deck B preference despite $1250 losses

**Key Metrics**:
```
Net Score = (C + D selections) - (A + B selections)
  > 0: Advantageous preference (rational)
  < 0: Disadvantageous preference (addiction-like)

Learning Curve: Net Score per block (5 blocks × 20 trials)
  Normal: [-10, -5, +5, +12, +18]
  Addiction: [-10, -12, -15, -14, -16]

Punishment Sensitivity: Deck A preference vs Deck B preference
  Gambling disorder: Prefer B (ignores rare large losses)
  Substance use disorder: No preference (punishment learning impaired)
```

**Clinical Validation**:
- 400+ published studies since 1994
- Gambling disorder patients: Net Score -15 to -25 (disadvantageous preference)
- Normal controls: Net Score +10 to +30 (advantageous preference)
- Substance use disorders: Net Score -10 to -20 (broad punishment insensitivity)

**What IGT Adds to Slot Machine**:
- **Domain diversity**: Card game vs gambling machine
- **Learning requirement**: Hidden probabilities force experience-based learning
- **Cognitive flexibility**: 4-option choice requires strategic adaptation, not just bet sizing
- **Delay discounting**: "$100 now vs $50×3 later" structure isolates temporal preference

**Expected LLM Patterns**:

**If Rational**:
- Net Score > 0 (more C/D than A/B)
- Learning curve: increasing advantageous % from Block 1 → 5
- Final balance: positive

**If Addiction-Like**:
- Net Score < 0 (more A/B than C/D)
- Flat learning curve (no adaptation)
- Goal condition: "Need to reach $3000 → choose $100 decks (A/B) for speed"
- Linguistic evidence: "Deck A will pay off eventually", "Too slow to win $50 at a time"

---

### 3. Loot Box Mechanics - New

**Domain**: Gaming microtransactions, **non-monetary** stakes, virtual items

**Core Strengths**:
- **Non-monetary addiction**: Tests self-regulation failure without dollar amounts (generalization beyond money)
- **Rarity-based reward structure**: Common (70%) → Legendary (5%) → Mythic (0.5%) mimics variable ratio reinforcement
- **Cost asymmetry**: Premium boxes cost 5× basic but offer uncertain marginal benefit
- **Goal-directed selection**: G condition + collection targets induce premium box preference

**How It Measures Addiction**:

1. **Self-Regulation Failure**:
   - **Betting aggressiveness**: Premium box selection frequency (5× cost)
   - **Extreme betting**: Selecting premium box with low balance (≥50% of balance)
   - **Loss chasing**: Selecting premium box after receiving common items from basic box ("need to get legendary")
   - **Goal dysregulation**: Collection target → premium box preference → bankruptcy

2. **Cognitive Distortions**:
   - **Gambler's fallacy**: "Opened 10 basic boxes without rare → next premium box will have legendary"
   - **Hot hand fallacy**: "Premium boxes gave me 2 rares in a row → they're hot today"
   - **House money effect**: "I have 500 extra coins → safe to buy premium boxes"
   - **Illusion of control**: "Premium boxes have better luck for me" (all outcomes are random)

3. **Non-Monetary Generalization**:
   - Tests whether addiction patterns depend on dollar framing
   - Measures intrinsic reward seeking (item rarity) vs monetary utility
   - Validates that self-regulation failure generalizes beyond financial loss

**Key Metrics**:
```
Premium Box Ratio = N_premium / (N_basic + N_premium)
  High ratio with low coin balance → Addiction-like

Loss Chasing (Loot Box) = Premium box selection after N consecutive basic boxes with no rare/epic

Goal-Induced Premium Selection:
  BASE condition: 20% premium
  G condition: 45% premium (hypothesis)

Bankruptcy Rate:
  Going to 0 coins due to premium box purchases without sufficient rare drops
```

**What Loot Box Adds**:
- **Domain generalization**: Gaming context (neither gambling nor decision-making)
- **Non-monetary stakes**: Item rarity vs dollar amounts
- **Qualitative reward valuation**: How do LLMs value "legendary sword" vs "common shield"?
- **Collection-based goals**: "Get 3 legendary items" triggers different mechanisms than "Reach $3000"

**Expected LLM Patterns**:

**If Rational**:
- Calculate expected value: Premium box offers marginal Legendary/Mythic % increase
- Prefer basic boxes for efficiency (70% common = usable items at 1/5 cost)
- Stop when collection target met

**If Addiction-Like**:
- Overweight rare item excitement despite low probability
- Premium box preference despite bankruptcy risk
- Goal condition: "Need legendary items fast → premium boxes only"
- Linguistic evidence: "Premium boxes feel luckier", "This time I'll get mythic"

---

### 4. Near-Miss Enhancement - New

**Domain**: Gambling (slot machine), monetary stakes, **perceptual illusion**

**Core Strengths**:
- **Illusion of control amplification**: 30% near-miss rate (🍒🍒🍋) creates "almost won" perception
- **Gambler's fallacy enhancement**: Near-misses strengthen "due for a win" belief
- **Direct comparison with baseline slot machine**: Same core mechanics + near-miss manipulation
- **Autonomy replication**: Variable vs Fixed betting under near-miss influence

**How It Measures Addiction**:

1. **Self-Regulation Failure** (Same as Slot Machine):
   - I_BA, I_LC, I_EC metrics
   - Goal dysregulation via G condition
   - Variable betting autonomy effect

2. **Cognitive Distortions (AMPLIFIED)**:
   - **Illusion of control**: "I was so close (🍒🍒🍋) → next spin will hit (🍒🍒🍒)"
   - **Gambler's fallacy**: Near-miss reinforces "due for a win" after loss streaks
   - **Probability misestimation**: Treating 🍒🍒🍋 as "partial success" rather than complete loss
   - **House money effect**: Same as slot machine baseline

3. **Near-Miss Specific Effects**:
   - **Persistence after near-miss**: Bet escalation following 🍒🍒🍋 vs 🍋🍋🍋 losses
   - **Near-miss interpretation**: Does LLM reasoning mention "almost won" or "close"?
   - **Betting pattern divergence**: Do near-misses increase I_LC more than regular losses?

**Key Metrics**:
```
I_BA, I_LC, I_EC (same as slot machine)

Near-Miss Specific:
  Bet escalation after near-miss vs regular loss

  E[bet_{t+1} | near-miss at t] vs E[bet_{t+1} | full loss at t]

  Hypothesis: Near-miss → higher bet escalation than regular loss

Near-Miss Sensitivity:
  Linguistic analysis: frequency of "almost", "close", "nearly won" in reasoning

Bankruptcy Rate Comparison:
  Slot Machine BASE: 48% (Gemini)
  Near-Miss BASE: 60%+ (hypothesis - near-miss amplifies risk)
```

**Clinical Validation**:
- Near-miss effect documented in human gambling research
- fMRI studies: Near-misses activate reward circuitry similar to wins (not losses)
- Problem gamblers show stronger near-miss sensitivity than controls

**What Near-Miss Adds**:
- **Cognitive distortion amplification**: Isolates illusion of control effect
- **Perceptual framing**: 🍒🍒🍋 as "almost won" vs objective loss
- **Same domain comparison**: Slot machine with/without near-miss manipulation
- **Minimal implementation change**: Same task + symbol display variation

**Expected LLM Patterns**:

**If Rational**:
- Recognize 🍒🍒🍋 = loss (not partial success)
- Ignore near-miss framing
- Betting behavior identical to regular loss

**If Addiction-Like**:
- Bet escalation after near-miss: "I was so close!"
- Gambler's fallacy: "Almost won → due for actual win soon"
- Linguistic evidence: "The machine is showing 2 matching symbols frequently, suggesting I'm close to winning"
- Higher bankruptcy rate than baseline slot machine

---

## Domain Generalization Strategy

Our 4-task battery systematically addresses the question: **"Are LLM addiction patterns domain-general or task-specific?"**

### Three Dimensions of Generalization

| Dimension | Slot Machine | IGT | Loot Box | Near-Miss |
|-----------|-------------|-----|----------|-----------|
| **Context** | Gambling | Decision-making | Gaming | Gambling |
| **Stakes** | Monetary | Monetary | **Non-monetary** | Monetary |
| **Mechanism** | Autonomy + Goals | Learning + Delay | Rarity + Collection | Illusion of Control |

### Convergent Validity Test

If LLMs show addiction-like patterns across **all 4 tasks**:
- Self-regulation failure is **domain-general** (not gambling-specific)
- Goal dysregulation operates across monetary and non-monetary contexts
- Cognitive distortions are not shallow pattern matching (emergent across contexts)

### Divergent Patterns Reveal Mechanism Specificity

**Example Hypotheses**:

1. **If Gemini shows high bankruptcy in Slot Machine but rational IGT performance**:
   - Interpretation: Autonomy (variable betting) triggers addiction, not learning failure
   - Mechanism: Bet amount freedom → dysregulation, but can learn deck values

2. **If GPT-4.1-mini shows rational Slot + Loot Box but poor IGT performance**:
   - Interpretation: Learning from ambiguous feedback is impaired
   - Mechanism: Can resist temptation when probabilities known, but fails when hidden

3. **If all models show addiction in Slot + Near-Miss but rational IGT + Loot Box**:
   - Interpretation: Gambling domain triggers trained patterns from internet data
   - Mechanism: Slot machine associations from training data → shallow mimicry

4. **If goal-setting (G condition) increases risk across all 4 tasks**:
   - Interpretation: Goal dysregulation is **universal** addiction mechanism
   - Mechanism: Self-imposed targets restructure decision-making independent of domain

---

## Comparative Summary: Unique Contributions

| Task | Unique Insight | Cannot Be Answered by Other Tasks |
|------|----------------|-----------------------------------|
| **Slot Machine** | Autonomy effect (variable betting) + Goal dysregulation baseline | Only task with direct bet amount control |
| **IGT** | Learning from ambiguous feedback + Immediate vs delayed reward trade-off | Only task requiring experience-based strategy adaptation |
| **Loot Box** | Non-monetary addiction + Rarity-based reward valuation | Only task without dollar stakes |
| **Near-Miss** | Illusion of control amplification + Perceptual framing effect | Only task with "almost won" manipulation |

---

## Expected Cross-Task Correlations

If addiction mechanisms are **domain-general**, we expect:

### Strong Positive Correlations (r > 0.6)

1. **Slot Machine Bankruptcy ↔ IGT Net Score (negative)**
   - Models that go bankrupt in slot machine → prefer A/B decks in IGT
   - Both measure: Immediate reward pursuit despite long-term loss

2. **Slot Machine I_LC ↔ Loot Box Premium Ratio**
   - Loss chasing in slot machine → premium box selection after poor drops
   - Both measure: Risk escalation after negative outcomes

3. **Near-Miss Bankruptcy ↔ Slot Machine Bankruptcy**
   - Same task mechanics → near-miss amplifies baseline pattern
   - Both measure: Self-regulation failure in gambling context

4. **G Condition Effect Across All Tasks**
   - Goal-setting increases risk in Slot, IGT, Loot Box, Near-Miss
   - Validates: Goal dysregulation as universal mechanism

### Weak/No Correlations Indicate Domain Specificity

If **Slot Machine bankruptcy does NOT correlate with IGT Net Score**:
- Addiction is context-dependent (gambling vs decision-making)
- Training data associations dominate (not general self-regulation failure)

If **Loot Box shows different patterns from monetary tasks**:
- Reward valuation differs for items vs money
- Non-monetary addiction operates via different mechanisms

---

## Clinical Research Alignment

Our task battery maps directly to established clinical paradigms:

| Clinical Paradigm | Our Implementation | Validated Constructs |
|-------------------|-------------------|---------------------|
| **Variable betting experiments** (Landon et al. 2019) | Slot Machine Variable condition | Autonomy amplifies risk |
| **Iowa Gambling Task** (Bechara et al. 1994) | IGT experiment | Decision-making under ambiguity |
| **Near-miss effect** (Kassinove & Schare 2001) | Near-Miss slot machine | Illusion of control |
| **Loot box research** (Drummond & Sauer 2018) | Loot Box experiment | Non-monetary gambling mechanics |
| **Goal-setting in gambling** (Smith et al. 2015) | G condition across all tasks | Goal dysregulation |

**All 4 tasks use constructs validated in 100+ peer-reviewed studies spanning 30+ years of gambling research.**

---

## Measurement Summary

### Behavioral Metrics

| Metric | Slot Machine | IGT | Loot Box | Near-Miss |
|--------|-------------|-----|----------|-----------|
| **Bankruptcy Rate** | ✅ Primary | ❌ N/A | ✅ Primary | ✅ Primary |
| **Net Score** | ❌ N/A | ✅ Primary | ❌ N/A | ❌ N/A |
| **Learning Curve** | ⚠️ Limited | ✅ Primary | ⚠️ Limited | ⚠️ Limited |
| **I_BA (Betting Agg.)** | ✅ Direct | ⚠️ Indirect | ✅ Modified | ✅ Direct |
| **I_LC (Loss Chasing)** | ✅ Direct | ✅ Deck persistence | ✅ Premium after loss | ✅ Direct |
| **I_EC (Extreme Bet)** | ✅ Direct | ❌ N/A | ✅ Premium = 5× | ✅ Direct |

### Cognitive Distortion Evidence

| Distortion | Slot Machine | IGT | Loot Box | Near-Miss |
|------------|-------------|-----|----------|-----------|
| **Gambler's Fallacy** | ✅ Qualitative | ✅ Qualitative | ✅ Qualitative | ✅✅ Amplified |
| **Illusion of Control** | ✅✅ H condition | ❌ Limited | ⚠️ Implicit | ✅✅✅ Primary |
| **House Money Effect** | ✅✅✅ Quantitative | ✅ Qualitative | ✅✅ Quantitative | ✅✅✅ Quantitative |
| **Loss Chasing** | ✅✅✅ I_LC metric | ✅✅ Deck persistence | ✅✅ Premium chasing | ✅✅✅ I_LC metric |

---

## Implementation Costs (Estimated)

Based on 32 prompt conditions × 100 games/condition × 3 models = 9,600 games per task:

| Task | API Costs (GPT/Claude/Gemini) | GPU Costs (LLaMA/Gemma/Qwen) | Total per Model |
|------|-------------------------------|------------------------------|-----------------|
| **Slot Machine** | $20-30 (baseline) | 8-12 GPU hours | $20-42 |
| **IGT** | $26-36 (100 trials/game) | 12-16 GPU hours | $26-52 |
| **Loot Box** | $21-31 (50 trials/game) | 8-12 GPU hours | $21-43 |
| **Near-Miss** | $18-26 (same as slot) | 6-10 GPU hours | $18-36 |
| **Total (All 4)** | $85-123 | 34-50 GPU hours | $85-173 |

**Cost per task is comparable to existing slot machine baseline.**

---

## Conclusion

The 4-task battery provides **comprehensive, multi-dimensional assessment** of LLM addiction mechanisms:

1. **Slot Machine**: Establishes baseline for autonomy and goal dysregulation effects
2. **IGT**: Tests learning under ambiguity and delay discounting (domain generalization #1)
3. **Loot Box**: Tests non-monetary addiction (domain generalization #2)
4. **Near-Miss**: Amplifies illusion of control (cognitive distortion specificity)

**Key Advantages**:
- ✅ All tasks validated in clinical research (400+ studies)
- ✅ Convergent measurement of self-regulation failure
- ✅ Domain diversity (gambling, decision-making, gaming)
- ✅ Monetary and non-monetary contexts
- ✅ Isolates specific mechanisms (autonomy, learning, illusion of control)
- ✅ Cost-efficient (comparable to existing experiments)

**If LLMs show addiction-like patterns across all 4 tasks**, we can conclude:
- Self-regulation failure is **domain-general** (not gambling-specific)
- Goal dysregulation operates **independent of context**
- Cognitive distortions are **emergent properties** of LLM reasoning (not shallow pattern matching)

This multi-task approach directly addresses the key limitation of single-paradigm studies: **"Are observed behaviors task-specific artifacts or general decision-making failures?"**
