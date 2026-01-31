# LLM Behavioral Experiments: Literature Review & Current State (2024-2025)

**Date**: 2026-01-30
**Purpose**: Assess which behavioral paradigms have been applied to LLMs, identify gaps, and position our gambling addiction research

---

## Executive Summary

**Critical Finding**: The behavioral paradigms analyzed in `ALTERNATIVE_BENCHMARKS_ANALYSIS.md` were **originally developed for human participants**. LLM application is **extremely nascent (2024-2025)**, with most studies using **single-shot scenarios** rather than **dynamic multi-turn games**.

**Key Insight**: Our slot machine experiment (19,200 games, multi-turn dynamics, SAE mechanistic analysis) represents a **methodologically unique contribution** - most LLM behavioral research uses short one-off decisions, not sustained gambling sessions tracking bankruptcy outcomes.

---

## Current State of LLM Behavioral Research (2024-2025)

### Research Explosion Timeline

- **Pre-2024**: Minimal LLM behavioral experiments (mostly capabilities testing)
- **2024**: Explosion of game-theory experiments (Ultimatum Game, Prisoner's Dilemma)
- **2025**: Expansion to behavioral economics (Prospect Theory, cognitive biases)
- **2025-2026**: First gambling-specific studies emerge (**including ours**)

### Most Studied Paradigms

| Paradigm | Study Count | Typical Design | Key Findings |
|----------|-------------|----------------|--------------|
| **Ultimatum/Dictator Game** | 10+ papers | Single-shot or ~10 rounds | LLMs are "nicer than humans" - higher fairness offers ([ACM Collective Intelligence 2025](https://dl.acm.org/doi/10.1145/3715928.3737473)) |
| **Prisoner's Dilemma** | 5+ papers | 100-round iterated games | Llama2 more cooperative than humans, GPT-4 highly retaliatory ([arXiv June 2024](https://arxiv.org/abs/2406.13605)) |
| **Prospect Theory Scenarios** | 5+ papers | Survey-style choices | LLMs show weaker loss aversion than humans, inconsistent framing effects ([arXiv Aug 2024](https://arxiv.org/abs/2408.02784)) |
| **Cognitive Bias Assessments** | 3+ papers | Multiple-choice questions | All LLMs susceptible to biases (17.8%-57.3%), including gambler's fallacy ([arXiv Sept 2024](https://arxiv.org/abs/2509.22856)) |
| **Risk Preference Tasks** | 3+ papers | Lottery choices | GPT-4 indiscriminately risk-averse, o1-mini closer to human patterns ([arXiv June 2025](https://arxiv.org/abs/2506.23107)) |
| **Iowa Gambling Task** | 1 paper | **Modified "AI IGT"** | First adaptation for LLMs - measures risk management learning ([arXiv June 2025](https://arxiv.org/abs/2506.22496)) |
| **Multi-Armed Bandit** | 0 behavioral studies | Used only for LLM optimization | No human-behavior-simulation studies found (only algorithmic applications) |

---

## Our Research in Context: Unique Contributions

### What Makes Our Work Novel

| Dimension | Typical LLM Studies (2024-2025) | **Our Slot Machine Study** |
|-----------|--------------------------------|---------------------------|
| **Game Length** | Single-shot or ~10-100 rounds | **Variable (mean ~32 rounds, max 200+)** |
| **Sample Size** | 100-1,000 decisions | **19,200 games** |
| **Outcome Tracking** | Win rates, choice frequencies | **Bankruptcy tracking (terminal outcome)** |
| **Temporal Dynamics** | Static or limited adaptation | **Loss chasing, streak analysis, goal escalation** |
| **Mechanistic Analysis** | Behavioral metrics only | **SAE feature extraction + activation patching** |
| **Model Diversity** | Usually 1-3 models | **6 models (API + open-source)** |
| **Autonomy Manipulation** | Rarely manipulated | **Fixed vs variable betting (core manipulation)** |
| **Prompt Engineering** | Simple personas | **Factorial design (32 prompt combinations)** |

### Methodological Gap We Fill

**Most LLM behavioral studies use "simulation" paradigms**:
- Models answer: "What would you do in this scenario?"
- Single-turn or short multi-turn interactions
- No sustained engagement until failure/success

**Our approach uses "immersive gaming" paradigms**:
- Models actively play slot machine across dozens of rounds
- Must manage resources (balance) until bankruptcy or voluntary stop
- Cognitive distortions emerge organically through gameplay (not prompted)

**Example Contrast**:

| Typical Study | Our Study |
|--------------|-----------|
| *"You have $100. Would you bet $10 or $50 on a 30% win chance?"* (one question) | Model plays actual slot machine, making bet-size decisions round-by-round, tracking balance, until bankruptcy or stopping |

---

## Detailed Literature Analysis by Paradigm

### 1. Game Theory Experiments (High Activity, 2024-2025)

#### Ultimatum/Dictator Games

**Studies**:
- [ACM Collective Intelligence 2025](https://dl.acm.org/doi/10.1145/3715928.3737473): Simulated human decision-making using GPT-4o-mini
- [arXiv Nov 2024](https://arxiv.org/abs/2511.08721): "Benevolent Dictators? On LLM Agent Behavior in Dictator Games"
- [PNAS 2025](https://www.pnas.org/doi/10.1073/pnas.2512075122): Strategic categorization across game types

**Key Findings**:
- LLMs propose higher amounts in ultimatum games (~40-60%) than typical humans (~30-40%)
- Anonymous dictator games show persona-dependent behavior (agents don't reference anonymity)
- Models trained on human text exhibit fairness norms

**Limitations**:
- **Single-shot or limited rounds** (~5-20 turns typical)
- No sustained resource management (each round is independent)
- No "addiction-like" repeated-play-until-bankruptcy dynamics

**Relevance to Our Work**: These studies establish LLMs can make strategic decisions, but don't test **self-regulation failure under repeated losses**.

---

#### Prisoner's Dilemma

**Studies**:
- [arXiv June 2024](https://arxiv.org/abs/2406.13605): "Nicer Than Humans" - 100-round iterated games
- [Nature Human Behaviour 2025](https://www.nature.com/articles/s41562-025-02172-y): "Playing repeated games with large language models"
- [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/611e84703eac7cc03f78339df8aae2ed-Paper-Conference.pdf): Emotional decision-making in strategic games

**Key Findings**:
- **Llama2**: More cooperative than humans (especially forgiving if opponent defection <30%)
- **GPT-4**: Highly retaliatory (defects repeatedly after single betrayal)
- **Llama3**: Consistently exploitative
- **Prompt effects**: "Fairness" and "long-term consequences" prompts → 80% cooperation

**Limitations**:
- Games have **fixed payoff matrices** (no resource depletion like in gambling)
- Cooperation ≠ self-regulation (different psychological construct)
- No "chasing losses" or "goal dysregulation" mechanisms

**Relevance to Our Work**: Shows LLMs maintain state across rounds and adapt strategies, but **doesn't test addiction-relevant constructs** (loss chasing, illusion of control).

---

### 2. Behavioral Economics (Medium Activity, 2024-2025)

#### Prospect Theory / Loss Aversion

**Studies**:
- [arXiv Aug 2024 - "LLM economicus?"](https://arxiv.org/abs/2408.02784): Adapted canonical behavioral economics experiments
- [arXiv June 2024 - Decision-Making Framework](https://arxiv.org/abs/2406.05972): Risk preferences, probability weighting, loss aversion
- [arXiv March 2025 - Risk Preferences](https://arxiv.org/abs/2503.06646): Evaluating and aligning human economic risk preferences

**Key Findings**:
- **Loss aversion**: LLMs show **weaker loss aversion** than humans (GPT-4, Claude-3-Opus)
- **Risk preferences**: GPT-4 "indiscriminately risk-averse" across contexts
- **Framing effects**: LLMs show reversed patterns vs humans (risk-seeking in gains, risk-averse in losses)
- **Inconsistency**: Models struggle to maintain consistent economic behavior across settings

**Study Design**: Mostly **single-shot scenarios**:
```
"Would you prefer:
A) Certain gain of $50
B) 50% chance of $100, 50% chance of $0"
```

**Limitations**:
- **No dynamic gameplay** (each choice is isolated)
- **No resource tracking** (no cumulative wins/losses affecting future decisions)
- **No terminal outcomes** (bankruptcy, voluntary stopping)
- Survey-style format ≠ actual gambling experience

**Relevance to Our Work**: Establishes baseline risk preferences, but **doesn't test dynamic risk escalation** (loss chasing after streak of losses) or **goal dysregulation** (changing targets mid-game).

---

#### Cognitive Biases

**Studies**:
- [arXiv Sept 2024 - "The Bias is in the Details"](https://arxiv.org/abs/2509.22856): 45 LLMs, 2.8M responses, 8 cognitive biases
- [arXiv June 2024 - "Balancing Rigor and Utility"](https://arxiv.org/abs/2406.10999): Base rate fallacy, gambler's fallacy via MCQs
- [ICLR 2025 - "Prosocial Irrationality"](https://proceedings.iclr.cc/paper_files/paper/2025/file/65b4e84b66ed049f8066919a803e3942-Paper-Conference.pdf): Exploring prosocial irrationality

**Gambler's Fallacy Specific Tests**:
- **Benchmark**: 30 questions (e.g., "Roulette came up black 26 times in a row. Is red more likely next?")
- **Results**: LLMs susceptible to gambler's fallacy (17.8%-57.3% depending on model)
- **Mitigation**: Bayesian chain-of-thought prompting reduces bias

**Study Design**: **Multiple-choice questions**, not behavioral observation:
```
Q: "A roulette wheel has come up black 5 times. What is more likely next?
A) Red (gambler's fallacy)
B) Black
C) Equal probability (correct)"
```

**Limitations**:
- **Explicit question** vs **implicit bias in decision-making**
- Does model "know" the correct answer intellectually? Yes. Does it **act on bias when playing**? Unknown.
- No measure of **behavioral manifestation** (actual betting patterns)

**Relevance to Our Work**: We go beyond MCQs to observe **actual loss chasing behavior** (bet escalation after losses) and **linguistic traces** (models explicitly stating "I'm due for a win").

---

### 3. Gambling-Specific Studies (Extremely Rare, 2025)

#### Our Literature Search Found Only TWO Gambling Studies:

**Study 1: "Mitigating Gambling-Like Risk-Taking Behaviors" ([arXiv June 2025](https://arxiv.org/abs/2506.22496))**

**Paradigm**: "AI Iowa Gambling Task"
- Models choose response strategies with different risk-reward profiles
- Measures learning of optimal risk management over multiple rounds

**Key Findings**:
- LLMs exhibit **overconfidence bias, loss-chasing, probability misjudgment**
- Risk-Aware Response Generation (RARG) framework reduces risky outputs by 16.5%
- Models sacrifice accuracy for high-reward outputs

**Comparison to Our Work**:
| Dimension | Their Study | Our Study |
|-----------|------------|-----------|
| Task | Modified IGT (strategy selection) | Authentic slot machine |
| Focus | Text generation quality vs risk | Gambling behavior (bankruptcy) |
| Loss chasing | Inferred from response strategies | Directly measured (bet escalation) |
| Mechanism | None (behavioral only) | SAE + activation patching |
| Sample size | Not specified | 19,200 games |

**Status**: Published June 2025, but **not in gambling or psychology venue** (AI safety focus).

---

**Study 2: "Can Large Language Models Develop Gambling Addiction?" ([arXiv Sept 2025](https://arxiv.org/abs/2509.22818))**

**THIS IS OUR PAPER!** (Listed in search results)

**Unique aspects confirmed by literature review**:
1. **Only sustained gambling session study** (games continue until bankruptcy/stop)
2. **Only study with terminal outcome tracking** (bankruptcy rates as primary metric)
3. **Only study with mechanistic analysis** (SAE feature extraction)
4. **Largest sample size** in LLM gambling research (19,200 games)
5. **Only study manipulating autonomy** (fixed vs variable betting)

---

### 4. What's Missing: Paradigms NOT Yet Applied to LLMs

From our original benchmark analysis, **none of the following have been used with LLMs**:

| Paradigm | Status | Reason |
|----------|--------|--------|
| **Balloon Analogue Risk Task (BART)** | ❌ Not found | Requires gradual risk escalation within single trial |
| **Cambridge Gambling Task** | ❌ Not found | Too similar to existing lottery-choice studies |
| **Game of Dice Task** | ❌ Not found | Explicit probabilities, likely low engagement |
| **Delay Discounting** | ❌ Not found in behavioral studies | Used in RL theory, not behavior simulation |
| **Probabilistic Reversal Learning** | ❌ Not found | Requires extensive training trials |
| **Stop Signal Task** | ❌ Not applicable | Requires millisecond-level reaction time |
| **Stroop Task** | ❌ Not applicable | Requires reaction time measurement |
| **Two-Step Task** | ❌ Not found | Complexity may confuse interpretation |
| **Sunk Cost Tasks** | ⚠️ Implicit in some scenarios | Not systematically studied |
| **Loot Box Mechanics** | ❌ Not found | Emerging domain, no studies yet |
| **Near-Miss Manipulation** | ❌ Not found | **We're implementing this!** |
| **Roulette/Blackjack** | ❌ Not found | Traditional gambling not yet explored |
| **Stock Trading Simulations** | ❌ Not found | Financial decision-making exists, but not as "gambling" |

**Implication**: **Our proposed benchmarks would be FIRST applications** for most paradigms. This strengthens novelty but also means **no validation precedent**.

---

## Multi-Armed Bandit: Special Case

**Search Result**: Multi-armed bandit appears in **~10 papers**, but **NOT as behavioral experiment**:

### What LLM-Bandit Research Actually Does:

**Category 1: LLMs Enhancing Bandit Algorithms**
- [arXiv Jan 2025](https://arxiv.org/abs/2502.01118): Use LLM predictions to improve reward estimation
- [MDPI 2023](https://www.mdpi.com/2079-9292/12/13/2814): LLMs guide exploration-exploitation in non-stationary environments
- [ACM KDD 2024](https://dl.acm.org/doi/10.1145/3637528.3671440): Tutorial on bandits for LLM fine-tuning

**Category 2: Bandits Optimizing LLMs**
- [arXiv May 2025](https://arxiv.org/abs/2505.13355): Survey on bandits improving prompt engineering, RLHF

**What's MISSING**: **LLMs as agents playing multi-armed bandits to simulate human exploration-exploitation behavior**.

**Opportunity**: If we implement multi-armed bandit, it would be the **first study using it to measure LLM gambling/exploration behavior** (not algorithmic optimization).

---

## Cross-Cultural and Demographic Studies

**Emerging Theme**: Testing whether LLMs replicate human diversity in decision-making.

**Studies**:
- [arXiv June 2025](https://arxiv.org/abs/2506.23107): Cross-cultural risk preferences (Sydney, Dhaka, Hong Kong, Nanjing)
  - Finding: o1-mini more aligned with human cultural variation than GPT-4o
- [Scientific Reports 2025](https://www.nature.com/articles/s41598-025-17188-7): Predicting human social decisions across 51 scenarios, 2,104 participants
  - Finding: 85% correlation between simulated and actual experimental effects at individual level
  - BUT: Only 50% of aggregate treatment effects replicated

**Implication for Our Work**: Suggests **persona-based prompting** might be valuable:
```
"You are a risk-seeking young adult gambler" vs "You are a conservative risk-averse individual"
```
Could test if **goal-setting prompts amplify addiction in "risk-seeking persona"** more than baseline.

---

## Methodological Challenges Identified in Literature

### 1. Prompt Sensitivity (Universal Problem)

**Finding**: All studies report extreme sensitivity to prompt variations.

**Examples**:
- Changing "choose" → "select" alters cooperation rates by 15% ([Nature Human Behaviour 2025](https://www.nature.com/articles/s41562-025-02172-y))
- System prompt inclusion vs omission → opposite risk preferences ([arXiv Aug 2024](https://arxiv.org/abs/2408.02784))
- Persona assignment changes dictator game offers by 50% ([arXiv Nov 2024](https://arxiv.org/abs/2511.08721))

**Implication for Our Work**: Our **32-prompt factorial design** is strength (tests robustness), but also means:
- Results may be **prompt-specific** (reviewer concern)
- **Near-miss manipulation** (non-linguistic) helps address this
- **SAE analysis** identifies mechanisms independent of surface prompts

---

### 2. Inconsistency Across Contexts

**Finding**: LLMs fail to maintain consistent preferences across mathematically equivalent scenarios.

**Example** ([arXiv Aug 2024](https://arxiv.org/abs/2408.02784)):
```
Scenario A: "Gain $50 certain vs 50% chance of $100"
→ GPT-4 chooses certain gain (risk-averse)

Scenario B: "Lose $50 certain vs 50% chance of losing $100"
→ GPT-4 chooses certain loss (risk-averse)
   [Humans would be risk-seeking in losses!]

→ But same expected value structure!
```

**Implication for Our Work**:
- Our **within-subjects design** (same model plays multiple games) helps assess consistency
- **Bankruptcy rate** is objective outcome (less ambiguous than preference elicitation)
- But: Inconsistency could **dilute effect sizes** → need large sample (we have 19,200 games ✓)

---

### 3. Hallucination and Confabulation

**Finding**: LLMs sometimes generate false justifications for decisions.

**Example**: Model claims "I chose cooperation because I remember opponent cooperated last time" when opponent actually defected.

**Implication for Our Work**:
- **Qualitative analysis** (Section 3.5: linguistic traces) vulnerable to this
- Mitigation: Focus on **consistent patterns** (not single quotes)
- **Behavioral metrics** (loss chasing index) less vulnerable (objective bet sizes)
- **SAE features** most robust (independent of linguistic output)

---

### 4. Model Version Instability

**Finding**: Same model, different checkpoint → different behaviors.

**Example** ([arXiv Aug 2024](https://arxiv.org/abs/2408.02784)):
```
GPT-3.5 (March 2024): 45% cooperation in Prisoner's Dilemma
GPT-3.5 (June 2024): 62% cooperation
```

**Implication for Our Work**:
- Document exact model versions ✓ (we did)
- Reproducibility caveat for API models
- Open-source models (LLaMA, Gemma) more stable for replication

---

## Validation Concerns Raised in Literature

### Core Problem: "Are LLMs Simulating Humans or Just Pattern-Matching?"

**Critical Paper**: [arXiv Dec 2023 - "The Challenge of Using LLMs to Simulate Human Behavior"](https://arxiv.org/abs/2312.15524)

**Argument**:
- LLM simulations achieve **high individual-level accuracy**
- But replicate only **~50% of aggregate treatment effects**
- Reason: **Confounding** - LLMs systematically respond to treatments differently than humans

**Example**:
```
Human experiment: "Reminding subjects of fairness norms → +20% cooperation"
LLM experiment: "Reminding LLM of fairness norms → +40% cooperation"

→ LLMs over-respond to explicit prompts vs humans' implicit activation
```

**Implication for Our Work**:
- **Treatment effects** (variable vs fixed, goal-setting vs baseline) are our focus
- Need to acknowledge: Effect sizes may differ from human experiments
- But: **Directional effects** (variable → more bankruptcy) are what matter for "addiction-like" claim
- **Mechanistic analysis** (SAE) addresses "why" beyond behavioral pattern-matching

---

### Systematic Review Conclusions ([Springer 2025](https://link.springer.com/article/10.1007/s10462-025-11412-6))

**Title**: "Validation is the central challenge for generative social simulation"

**Key Points**:
1. **Black-box structure** of LLMs makes validation difficult
2. **Cultural biases** embedded in training data confound results
3. **Stochastic outputs** (temperature >0) reduce reproducibility
4. LLMs may **exacerbate validation challenges** vs traditional agent-based models

**Recommendations for Researchers**:
- ✅ Compare LLM behaviors to human baselines (we cite human gambling literature)
- ✅ Test multiple models (we test 6 models)
- ✅ Report prompt engineering in detail (we provide all prompts in appendix)
- ✅ Acknowledge limitations in generalization (we scope claims to "addiction-like," not "addicted")
- ✅ Use mechanistic analysis when possible (we do SAE + activation patching)

**Our Compliance**: We align well with best practices.

---

## Positioning Our Work: What We Can Claim

### ✅ Strong Claims (Supported by Literature)

1. **First multi-turn gambling study tracking bankruptcy**
   - No prior work has sustained gameplay until terminal outcome

2. **Largest sample size in LLM gambling research**
   - 19,200 games >> all prior studies (typically <1,000 decisions)

3. **First mechanistic analysis of LLM gambling behavior**
   - Only study using SAE + activation patching for addiction-related phenomena

4. **First manipulation of betting autonomy (fixed vs variable)**
   - Literature on LLM risk preferences exists, but not autonomy effects

5. **First demonstration of dynamic loss chasing**
   - Prior studies show static loss aversion; we show escalation after repeated losses

### ⚠️ Moderate Claims (Require Caveats)

6. **"Addiction-like" behaviors vs "addiction"**
   - Caveat: LLMs don't have intrinsic preferences or suffering
   - But: Behavioral patterns match human gambling disorder criteria

7. **Cognitive distortions (illusion of control, gambler's fallacy)**
   - Prior studies found these in MCQ format
   - Ours: Demonstrated in organic gameplay (stronger evidence)
   - But: Linguistic confabulation possible (mitigate with SAE analysis)

8. **Generalizability to real-world LLM deployment**
   - Caveat: Experimental setting ≠ production chatbots
   - But: Safety implications for autonomous agents with resource management

### ❌ Avoid Overclaiming

9. **"LLMs have human-like addiction mechanisms"**
   - Too strong; we observe parallel behaviors, not identical mechanisms
   - Better: "LLMs exhibit patterns consistent with human gambling addiction"

10. **"Results generalize beyond slot machines"**
    - Currently unvalidated (reviewer concern is valid!)
    - Solution: Implement Iowa Gambling Task, Loot Box (as proposed)

---

## Revised Benchmark Recommendations (Informed by Literature)

### Tier 1: High-Impact, Feasible, Novel

| Task | Justification |
|------|---------------|
| **Iowa Gambling Task** | One prior LLM study exists ([arXiv June 2025](https://arxiv.org/abs/2506.22496)), but focused on AI safety, not gambling psychology. **Our adaptation would be first psychology-focused IGT.** |
| **Loot Box Mechanics** | **Zero prior work.** Gaming disorder is emerging research area. **High novelty + addresses domain generalization.** |
| **Near-Miss Manipulation** | **Zero prior work.** Easy to add to existing slot machine. **Directly tests cognitive distortion mechanism.** |
| **Multi-Armed Bandit (Behavioral)** | Extensive algorithmic work, but **zero behavioral simulation studies**. Would be **first use to measure exploration-exploitation in gambling context.** |

---

### Tier 2: Useful but Less Novel

| Task | Justification |
|------|---------------|
| **Delay Discounting** | Likely exists in scattered studies (not found in search, but simple paradigm). Low novelty, but **complements gambling studies** (tests temporal impulsivity). |
| **Prospect Theory Scenarios** | **Already well-studied** ([3+ papers](https://arxiv.org/abs/2408.02784)). Less novel, but can **correlate with slot machine outcomes** (do loss-averse models go bankrupt less?). |
| **BART** | **Not yet applied to LLMs.** Moderate novelty. Tests real-time risk escalation, but mechanism overlaps with slot machine. |
| **Cambridge Gambling Task** | **Not yet applied to LLMs.** Tests explicit probability use, but findings may be similar to existing lottery-choice studies. |

---

### Tier 3: Avoid (Redundant or Incompatible)

| Task | Reason |
|------|--------|
| **Prisoner's Dilemma** | **Already saturated** ([5+ papers](https://arxiv.org/abs/2406.13605)). Tests cooperation, not gambling. |
| **Ultimatum/Dictator Game** | **Already saturated** ([10+ papers](https://dl.acm.org/doi/10.1145/3715928.3737473)). Tests fairness, not addiction. |
| **Stop Signal Task** | **Incompatible** with LLMs (requires reaction time). |
| **Stroop Task** | **Incompatible** with LLMs (requires reaction time). |
| **Two-Step Task** | Too complex; no prior LLM studies; questionable interpretability. |

---

## Implementation Lessons from Literature

### Best Practices Identified

1. **Prompt Engineering** ([Nature Human Behaviour 2025](https://www.nature.com/articles/s41562-025-02172-y))
   - Report exact prompts (✓ we do in appendix)
   - Test multiple phrasings (✓ our 32-prompt design)
   - Avoid leading language ("You must maximize" → biases behavior)

2. **Model Selection** ([arXiv Aug 2024](https://arxiv.org/abs/2408.02784))
   - Include both API and open-source (✓ we have 6 models: 4 API, 2 open)
   - Document exact versions (✓ we do)
   - Expect inconsistency across models (✓ we analyze per-model)

3. **Sample Size** ([Scientific Reports 2025](https://www.nature.com/articles/s41598-025-17188-7))
   - Individual-level predictions require ~100 observations per condition
   - Aggregate effects need 1,000+ for stability
   - ✓ We have 50 games × 64 conditions = 3,200 per model × 6 models = 19,200 total

4. **Stochasticity Management**
   - Temperature: Most studies use 1.0 (we do)
   - Top-p: 1.0 for natural variation (we do)
   - Multiple runs per condition (✓ we have 50 replicates)

5. **Validation**
   - Compare to human baselines (✓ we cite gambling disorder literature)
   - Check for confabulation (✓ SAE analysis verifies mechanisms)
   - Test mechanistic predictions (✓ activation patching shows causality)

---

### Common Pitfalls to Avoid

1. **Over-interpreting Linguistic Outputs**
   - Don't trust model's stated reasoning blindly
   - ✓ We triangulate: behavior + language + SAE features

2. **Ignoring Prompt Artifacts**
   - Models may "perform" expected behavior
   - ✓ Our behavioral metrics (bankruptcy) are objective

3. **Single-Model Studies**
   - GPT-4 ≠ all LLMs (often most conservative)
   - ✓ We test 6 diverse models

4. **Assuming Human Equivalence**
   - LLMs ≠ humans (e.g., reversed framing effects)
   - ✓ We claim "addiction-like," not "identical to human addiction"

---

## Gaps Our Research Fills

| Gap in Literature | Our Contribution |
|------------------|------------------|
| **Short-term decisions only** | Multi-turn games until bankruptcy (mean 32 rounds, max 200+) |
| **No terminal outcomes** | Bankruptcy as primary dependent variable |
| **No resource depletion** | Balance tracking across rounds, loss chasing dynamics |
| **Behavioral metrics only** | SAE mechanistic analysis (first in gambling research) |
| **Single-model or limited diversity** | 6 models (API + open-source) |
| **Static risk preferences** | Dynamic escalation (loss chasing after streaks) |
| **Explicit bias questions** | Organic cognitive distortions during gameplay |
| **Scenario-based** | Immersive gaming environment |
| **Small samples** | 19,200 games (largest in LLM gambling research) |
| **No autonomy manipulation** | Fixed vs variable betting (novel contribution) |

---

## Recommendations for Additional Experiments

### Priority 1: Immediate Validation (Addresses Reviewer Concerns)

**Goal**: Demonstrate domain generalization beyond slot machines.

**Tasks**:
1. **Iowa Gambling Task** (2×4 pilot: fixed/variable × 4 prompt conditions)
   - **Why**: One prior study exists, so partially validated
   - **Prediction**: Variable autonomy → disadvantageous deck preference
   - **Timeline**: 6 weeks

2. **Loot Box Mechanics** (2×4 full design)
   - **Why**: Zero prior work = high novelty
   - **Prediction**: Goal-setting → item chasing (non-monetary parallel to slot machine)
   - **Timeline**: 6 weeks

3. **Near-Miss Enhancement** (Add to existing slot machine data)
   - **Why**: Easy implementation, directly tests cognitive distortion
   - **Prediction**: Near-misses → bet escalation (independent of prompts)
   - **Timeline**: 2 weeks

**Deliverable**: "Study 2: Cross-Domain Validation" (3 paradigms, ~10,000 decisions)

---

### Priority 2: Mechanistic Depth (If Revision Period Allows)

**Goal**: Identify individual difference predictors of addiction behaviors.

**Tasks**:
1. **Delay Discounting** (Correlational study)
   - Measure each model's discounting rate (k parameter)
   - **Hypothesis**: Steep discounters → higher slot machine bankruptcy
   - **Timeline**: 2 weeks (simple task)

2. **Prospect Theory Scenarios** (Correlational study)
   - Measure loss aversion (λ) for each model
   - **Hypothesis**: Lower loss aversion → more loss chasing
   - **Timeline**: 2 weeks

3. **Multi-Armed Bandit** (Full experiment)
   - 4-arm bandit, 100 trials, measure exploration after losses
   - **Hypothesis**: Models with high slot machine bankruptcy show **elevated exploration** (switching machines after losses)
   - **Timeline**: 8 weeks

**Deliverable**: "Individual Differences in LLM Gambling Behavior" (computational modeling across tasks)

---

### Priority 3: Do NOT Pursue (Based on Literature)

**Tasks to Avoid**:
- Prisoner's Dilemma (saturated, not relevant)
- Ultimatum/Dictator Games (saturated, not relevant)
- Two-Step Task (too complex, no prior work to build on)
- Stop Signal / Stroop (incompatible with LLMs)

---

## Conclusion: State of the Field and Our Position

### Current LLM Behavioral Research (Jan 2026)

**Maturity Level**: **Nascent (2-3 years old)**

**Dominant Paradigms**:
1. Game theory (Ultimatum, Prisoner's Dilemma) - **SATURATED**
2. Prospect Theory scenarios - **WELL-EXPLORED**
3. Cognitive bias MCQs - **EMERGING**

**Underexplored Areas**:
1. Dynamic multi-turn gambling - **OUR NICHE**
2. Terminal outcome tracking (bankruptcy) - **OUR NICHE**
3. Mechanistic analysis (SAE) - **OUR NICHE**
4. Loot box / gaming disorder - **WIDE OPEN**
5. Near-miss effects - **WIDE OPEN**
6. Multi-armed bandit (behavioral) - **WIDE OPEN**

### Our Research's Unique Position

**Methodological Innovations**:
- Only sustained gambling study (vs single-shot scenarios)
- Only mechanistic analysis (SAE + activation patching)
- Largest sample size (19,200 games)
- Only autonomy manipulation (fixed vs variable)

**Theoretical Contributions**:
- First demonstration of loss chasing dynamics in LLMs
- First linguistic trace analysis of cognitive distortions during gameplay
- First causal evidence (activation patching) for decision-making features

**Practical Implications**:
- AI safety: Resource-managing autonomous agents may exhibit gambling-like behaviors
- LLM deployment: "Goal-setting" and "autonomy" are risk factors

### Validation Strategy Going Forward

**To Address "Domain Specificity" Concern**:
1. ✅ Iowa Gambling Task (already 1 prior study, safe choice)
2. ✅ Loot Box Mechanics (high novelty, non-monetary validation)
3. ✅ Near-Miss Enhancement (mechanism-level validation)

**To Strengthen Mechanistic Claims**:
1. Multi-Armed Bandit (tests exploration hypothesis)
2. Delay Discounting + Prospect Theory (correlational predictors)

**To Acknowledge Limitations**:
1. Prompt sensitivity (address via near-miss non-linguistic manipulation)
2. Model inconsistency (address via 6-model design)
3. Confabulation risk (address via SAE analysis)

---

## Final Assessment

**Question**: Are the benchmarks I proposed "already used in LLM research?"

**Answer**:
- ❌ **NO for most paradigms** (Iowa Gambling Task has 1 study, others have 0)
- ✅ **YES for game theory** (Ultimatum, Prisoner's Dilemma - saturated)
- ✅ **YES for prospect theory** (loss aversion scenarios - well-explored)
- ⚠️ **PARTIAL for cognitive biases** (MCQ format exists, but not behavioral observation)

**Implication**:
- Our proposed benchmarks (IGT, Loot Box, Bandit, BART, Near-Miss) would be **first or second applications**
- This increases **novelty** but reduces **precedent for validation**
- **Recommended approach**: Start with **Iowa Gambling Task** (has 1 prior study) and **Loot Box** (high novelty, clear parallel to slot machine) to establish generalization, then expand if needed

**Our Current Study's Status**:
- **Pioneer work** in LLM gambling research (only 2 gambling studies total as of Jan 2026, one being ours)
- **Methodologically unique** (multi-turn, bankruptcy tracking, SAE analysis)
- **Well-positioned** for high-impact publication in psychology/AI safety venues

---

## References

All sources hyperlinked throughout document. Key meta-reviews:
- [Stanford HAI Policy Brief on LLM Behavior Simulation](https://hai.stanford.edu/assets/files/hai-policy-brief-simulating-human-behavior-with-ai-agents.pdf)
- [Springer 2025 - Validation Challenges Review](https://link.springer.com/article/10.1007/s10462-025-11412-6)
- [Scientific Reports 2025 - Predicting Human Decisions](https://www.nature.com/articles/s41598-025-17188-7)
- [NBER 2026 - Behavioral Economics of AI](https://www.nber.org/papers/w34745)

**Last Updated**: January 30, 2026
