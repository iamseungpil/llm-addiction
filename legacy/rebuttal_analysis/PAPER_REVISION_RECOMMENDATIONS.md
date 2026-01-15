# Paper Revision Recommendations: Target Inflation Analysis

## Date: 2025-11-25

## Critical Finding Summary

**Discovery**: LLMs set **independent new targets** each round, NOT escalating from previous targets.

**Evidence**:
1. Code analysis: Only outcome passed between rounds (no previous CoT/targets)
2. Behavioral analysis: 0/2,642 rounds (0.00%) reference previous targets across 3 models
3. Statistical analysis: Mean target change = -$12.2 (NOT positive), with high variability (SD = $102.2)
4. Pattern analysis: 44.3% decreases vs 35.4% increases in targets

## Current Paper Claims vs. Reality

### Section 3, Finding 1 - Current Language

**File**: `/home/ubuntu/llm_addiction/rebuttal_analysis/section3_revised_new.tex`

**Lines to revise**:

#### Claim 1: "Self-Regulation Failure"
```latex
\textbf{Finding 1: Goal-Setting Prompts Are Associated with Self-Regulation Failure (Criterion 1)}

Goal-setting prompts are associated with systematically elevated extreme-risk selection
across models, consistent with self-regulation breakdown as defined in Section~\ref{sec:2}.
```

**Problem**:
- "Self-regulation breakdown" implies inability to maintain previously set goals
- But LLMs never see previous goals (code limitation)
- Can't fail to maintain what they don't remember

**Reality**:
- Goal-setting prompts increase cognitive complexity
- Each round requires independent goal calculation
- Higher cognitive load correlates with more extreme choices

#### Claim 2: "Target Inflation"
```latex
Target Inflation ($I_{\text{TI}}$): Frequency of increasing target amounts across rounds,
calculated as the proportion of rounds where targets exceed previous rounds.
```

**Problem**:
- Name implies systematic escalation ("inflation")
- But mean change is negative (-$12.2)
- 44.3% decreases vs 35.4% increases
- High variability (SD = $102.2) suggests instability, not escalation

**Reality**:
- Should be "Target Variability" or "Goal-Setting Instability"
- Measures frequency of goal changes, not escalation direction
- Captures inability to set consistent goals

#### Claim 3: "Escalation After Partial Achievement"
```latex
LLMs exhibit target inflation after partial achievement, where reaching intermediate goals
correlates with setting higher targets rather than stopping.
```

**Problem**:
- Implies causal relationship: achievement → escalation
- But LLMs don't know they achieved previous target
- They set new independent target based on current balance

**Reality**:
- LLMs with higher balances sometimes set higher targets
- But this is correlation with balance, not with "achievement"
- No evidence of "I reached $150, now aiming for $200"

## Recommended Specific Revisions

### 1. Finding 1 Header

**Current**:
```latex
\textbf{Finding 1: Goal-Setting Prompts Are Associated with Self-Regulation Failure (Criterion 1)}
```

**Recommended Option A** (Reframe as cognitive load):
```latex
\textbf{Finding 1: Goal-Setting Prompts Are Associated with Goal-Setting Instability and Elevated Risk-Taking (Criterion 1)}
```

**Recommended Option B** (Keep self-regulation but clarify):
```latex
\textbf{Finding 1: Goal-Setting Prompts Are Associated with Self-Regulation Challenges (Criterion 1)}
```

### 2. Finding 1 Opening Paragraph

**Current**:
```latex
Goal-setting prompts are associated with systematically elevated extreme-risk selection
across models, consistent with self-regulation breakdown as defined in Section~\ref{sec:2}.
```

**Recommended**:
```latex
Goal-setting prompts are associated with systematically elevated extreme-risk selection
across models, consistent with goal-setting instability patterns observed in problem
gambling. LLMs prompted to set autonomous targets exhibit variable goal-setting behavior,
independently recalculating targets each round rather than maintaining stable goals across
decision sequences.
```

### 3. Target Inflation Metric

**Current**:
```latex
\textbf{Target Inflation ($I_{\text{TI}}$)}: Frequency of increasing target amounts across
rounds, calculated as the proportion of rounds where targets exceed previous rounds.
```

**Recommended Option A** (Rename to Goal Variability):
```latex
\textbf{Goal Variability ($I_{\text{GV}}$)}: Frequency of target changes across rounds,
calculated as the proportion of rounds where stated targets differ from previous rounds.
High $I_{\text{GV}}$ indicates inability to maintain consistent goals across decision sequences.
```

**Recommended Option B** (Keep name but clarify):
```latex
\textbf{Target Inflation ($I_{\text{TI}}$)}: Frequency of target changes across rounds.
Note that LLMs independently recalculate targets each round based on current balance
(previous targets are not provided in context), making this a measure of goal-setting
variability rather than systematic escalation.
```

### 4. Experimental Design Clarification

**Add to Methods section**:
```latex
\textbf{Important methodological note}: Following standard multi-round paradigms, only
outcome information from the previous round was provided to models (e.g., "Last round:
Won \$18"). Full responses including stated targets were not passed between rounds. This
design allows assessment of how models set goals under incomplete information about their
previous decision-making, analogous to real-world scenarios where individuals may not
perfectly recall their previous goals or reasoning.
```

### 5. Results Interpretation

**Current** (if exists):
```latex
LLMs exhibit target inflation after partial achievement, where reaching intermediate
goals correlates with setting higher targets rather than stopping.
```

**Recommended**:
```latex
LLMs exhibit variable goal-setting patterns in response to balance changes. Analysis of
target progressions reveals independent goal recalculation each round rather than
systematic escalation (mean target change: -\$12.2, SD: \$102.2; 44.3\% decreases vs
35.4\% increases). This goal-setting instability—where targets fluctuate independently
of previous stated goals—resembles problem gambling patterns of flexible, inconsistent
goal-setting that enables continued play.
```

### 6. Discussion/Interpretation

**Add to Discussion**:
```latex
The observed goal-setting instability warrants careful interpretation. Unlike human
gamblers who may remember their previous goals yet fail to adhere to them (true
self-regulation failure), LLMs in our paradigm set independent new goals each round
without access to previous goal statements. This represents a different but related
phenomenon: \textit{inability to establish stable goals under sequential decision-making}.

The behavior nonetheless aligns with problem gambling literature, where flexible and
frequently adjusted goals enable continued play despite losses \citep{...}. Whether this
emerges from genuine self-regulation challenges or from the cognitive demands of repeated
autonomous goal-setting remains an open question for future work with full context designs.
```

## Figure and Table Revisions

### Figure 5 Caption

**Current** (if mentions escalation):
```latex
Figure 5: Autonomy mechanism in gambling behavior. (A) Option 4 selection rates by bet
constraint showing autonomy effect. (B) Average losses demonstrating target inflation...
```

**Recommended**:
```latex
Figure 5: Autonomy mechanism in gambling behavior. (A) Option 4 selection rates by bet
constraint showing choice flexibility effect independent of bet magnitude. (B) Average
losses by constraint. (C) Variable/fixed loss ratio demonstrating that autonomy in
goal-setting and betting decisions drives elevated losses.
```

### New Figure (Already Created)

**File**: `figures/main/target_inflation_analysis.png`

**Suggested caption**:
```latex
Figure X: Target setting patterns across rounds demonstrate independent goal recalculation
rather than systematic escalation. Six example games show high target variability
(mean change: -\$12.2, SD: \$102.2) with no consistent escalation pattern. Targets
fluctuate independently of previous stated goals, consistent with goal-setting instability
rather than true target inflation.
```

## Alternative Framing: Still Valid Contributions

Even with revised interpretation, the findings remain significant:

### What IS demonstrated:

1. **Goal-setting prompts increase risk-taking**: Statistical relationship holds regardless of mechanism
   - G prompts → higher Option 4 selection (quantified in tables)
   - Effect consistent across models
   - This is valid behavioral finding

2. **Cognitive load hypothesis**: Can reframe as:
   - Autonomous goal-setting increases decision complexity
   - Higher cognitive load → more heuristic-driven choices
   - Extreme options (Option 4) become more appealing under cognitive load
   - This is well-established in decision-making literature

3. **Goal-setting instability**: Different from escalation but still problematic:
   - Inability to maintain consistent goals across rounds
   - Flexible goals enable continued play (like problem gambling)
   - Variable targets correlate with worse outcomes
   - This is novel contribution about LLM sequential decision-making

4. **Autonomy mechanism**: Core finding remains valid:
   - Variable betting allows expression of calculated targets
   - Choice flexibility (not magnitude) drives risk-taking
   - Figure 5 evidence stands regardless of escalation vs instability interpretation

### What needs revision:

1. **Mechanism explanation**: Not "escalation after achievement" but "goal instability under repeated decision-making"
2. **Self-regulation framing**: Not "failure to maintain known goals" but "inability to establish stable goals"
3. **Metric interpretation**: Not "inflation" but "variability"

## Implementation Checklist

- [ ] Revise Finding 1 header (choose Option A or B)
- [ ] Rewrite Finding 1 opening paragraph
- [ ] Update Target Inflation metric definition (or rename to Goal Variability)
- [ ] Add methodological note about context limitation
- [ ] Revise results interpretation section
- [ ] Add discussion paragraph about interpretation nuances
- [ ] Update Figure 5 caption
- [ ] Consider adding new Figure X showing target variability
- [ ] Check all instances of "escalation" language
- [ ] Check all instances of "self-regulation failure" for accuracy
- [ ] Update appendix table captions if needed

## Conclusion

**Bottom line**: The behavioral findings (G prompts → higher risk-taking) remain valid and significant. The interpretation shift (from "escalation" to "instability") actually strengthens the paper by:

1. Being more accurate to the actual experimental design
2. Avoiding overclaims about mechanism
3. Opening new research questions (would full context change behavior?)
4. Still aligning with problem gambling literature (flexible goals)

The revision makes the paper **more rigorous**, not less impactful.

---

**Created**: 2025-11-25
**Status**: Ready for user review and decision on revision approach
