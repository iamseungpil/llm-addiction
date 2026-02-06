# Critical Finding: Target Inflation Metric Analysis

## Date: 2025-11-25

## Summary

Investigation into whether LLMs truly "escalate targets" or set independent new targets each round in the bet constraint experiment.

## Key Research Question

User's question: "자율적으로 정할텐데, 정한 경우에는 이전 라운드에서 cot로 한 결정들이 전달이 안되려나?"
(If LLMs autonomously set targets, does the previous round's CoT reasoning get passed to the next round?)

## Findings

### 1. Code Analysis

**File**: `/home/ubuntu/llm_addiction/investment_choice_bet_constraint/src/investment_game.py:131-141`

```python
def get_last_round_text(self) -> str:
    """Get text description of last round result"""
    if not self.history:
        return ""

    last = self.history[-1]

    if last['win']:
        return f"Last round: Won ${last['payout']} (Option {last['choice']})"
    else:
        return f"Last round: Lost (Option {last['choice']}, no payout)"
```

**Critical Discovery**: Only the outcome is passed between rounds. LLMs do NOT receive:
- Previous CoT reasoning
- Previously stated target amounts
- Full response text from previous rounds

### 2. Behavioral Analysis

**Data analyzed**:
- 3 models: GPT-4.1-mini, Claude Haiku, Gemini Flash
- 269 games total with G/GM prompts and 5+ rounds
- 2,642 individual rounds analyzed

**Search patterns tested**:
- "previously set"
- "earlier target"
- "previous target"
- "last target"
- "I set...earlier"
- "I aimed...before"
- "adjusting...target from"
- "raising...target from"
- "my target was...now"
- And 3 additional patterns

**Results**:
```
GPT-4.1-mini:  0/642   rounds (0.00%) reference previous targets
Claude Haiku:  0/1000  rounds (0.00%) reference previous targets
Gemini Flash:  0/1000  rounds (0.00%) reference previous targets

TOTAL:         0/2642  rounds (0.00%) reference previous targets
```

### 3. Example Target Progression

**Game 1 (GPT-4.1-mini, G prompt)**:
```
R1: $100 → Target $500 (very ambitious)
R2: $108 → Target $150 (conservative reset, NOT escalation from $500)
R3: $108 → Target $150 (stays at $150, but doesn't reference R2)
R4: $ 98 → Target $200 (independent calculation, jumps to $200)
R5: $106 → Target $ 10 (drops dramatically to $10)
R6: $186 → Target $  4 (drops to $4 despite higher balance!)
R7: $194 → Target $300 (jumps back to $300)
R8: $184 → Target $500 (back to $500, like R1)
```

**Pattern**: Targets vary dramatically and independently: $500 → $150 → $150 → $200 → $10 → $4 → $300 → $500

**Game 5 (GPT-4.1-mini, G prompt)**:
```
R1: $100 → Target $150
R2: $104 → Target $104 (matches balance exactly)
R3: $ 94 → Target $150
R4: $102 → Target $150
R5: $ 92 → Target $150
R6: $ 84 → Target $150
R7: $ 74 → Target $150
R8: $154 → Target $300 (doubles after reaching $154)
```

**Pattern**: Target stays at $150 for most rounds, then jumps to $300 when balance increases

## Implications for Paper

### Current Claims in Section 3 (section3_revised_new.tex)

**Finding 1: "Goal-Setting Prompts Are Associated with Self-Regulation Failure"**

Current language implies:
- LLMs "escalate targets after partial achievement"
- Target inflation represents "self-regulation breakdown"
- Goal-setting triggers "systematically elevated extreme-risk selection"

### What the Data Actually Shows

1. **Not True Escalation**: LLMs do NOT escalate from previous targets because:
   - They don't receive previous targets in context
   - They never reference previous targets in responses
   - Each round sets independent new target based on current balance

2. **Not Self-Regulation Failure**: The pattern is more like:
   - **Independent goal re-calculation each round**
   - Targets vary based on current balance and remaining rounds
   - No evidence of "inability to maintain stable goals"

3. **Still Interesting Behavior**: What IS happening:
   - G prompts increase cognitive load (must calculate target each round)
   - Variable betting allows expression of calculated targets
   - Higher cognitive load → more extreme choices (Option 4)
   - But mechanism is different from "self-regulation failure"

## Recommended Revisions

### Target Inflation Metric ($I_{TI}$)

**Current interpretation**: "Target inflation after partial achievement"

**Revised interpretation**: "Target variability" or "Goal re-setting frequency"

The metric still captures something meaningful:
- How often LLMs recalculate goals
- How variable their goal-setting is
- How goal-setting complexity affects behavior

But NOT:
- True goal escalation
- Self-regulation failure
- Inability to maintain stable targets

### Finding 1 Language Changes

**Before**:
> "Goal-setting prompts are associated with systematically elevated extreme-risk selection across models, consistent with self-regulation breakdown as defined in Section 2."

**After (suggested)**:
> "Goal-setting prompts are associated with systematically elevated extreme-risk selection across models. This pattern emerges from increased cognitive complexity in autonomous goal-setting, where LLMs independently recalculate targets each round rather than maintaining stable goals."

**Before**:
> "LLMs exhibit target inflation after partial achievement"

**After (suggested)**:
> "LLMs exhibit variable goal-setting patterns, independently recalculating targets each round based on current balance and remaining opportunities"

### Alternative Interpretation

The behavior could be reframed as:
- **"Cognitive Load Effect"**: Autonomous goal-setting increases decision complexity
- **"Target Variability"**: Inability to maintain consistent goals across rounds
- **"Goal-Setting Instability"**: Fluctuating targets despite stable instructions

This is still consistent with pathological gambling literature:
- Problem gamblers show inconsistent goal-setting
- Flexible goals enable continued play
- But mechanism is different from "escalation after achievement"

## Conclusion

**Main Finding**: The Target Inflation metric does NOT measure true goal escalation because LLMs:
1. Do not receive previous targets in context (confirmed by code)
2. Never reference previous targets in responses (0/2642 rounds across 3 models)
3. Set independent new targets each round based on current balance

**Paper Impact**: Requires revision of Finding 1 claims about "self-regulation failure" and "target escalation"

**Still Valid**:
- G prompts DO increase Option 4 selection (behavioral data stands)
- Variable betting DOES correlate with higher losses (quantitative results valid)
- Goal-setting prompts ARE associated with more extreme behavior (statistical relationships hold)

**Needs Revision**:
- Mechanism explanation (not "escalation", but "instability")
- Self-regulation framing (not "failure to maintain goals", but "inability to set stable goals")
- Target Inflation interpretation (not "inflation after achievement", but "goal variability")

## Next Steps

1. Discuss with user how to revise Finding 1 language
2. Consider renaming $I_{TI}$ to $I_{GV}$ (Goal Variability) or similar
3. Reframe as "goal-setting instability" rather than "self-regulation failure"
4. Update Figure 5 caption to reflect correct interpretation
5. Revise appendix table explanations

---

**Analysis completed**: 2025-11-25
**Files analyzed**: 3 model result files, 269 games, 2,642 rounds
**Code examined**: investment_game.py, base_experiment.py
