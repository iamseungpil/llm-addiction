# CoT Experiment Design Review

## Date: 2025-11-25

## Overview

This is a **revised version** of the bet constraint experiment designed to address the critical limitation discovered in the original experiment: LLMs could not see their previous goals or reasoning.

## Key Improvements Over Original Experiment

### Original Experiment (`investment_choice_bet_constraint`)

**Context provided between rounds**:
```python
def get_last_round_text(self) -> str:
    if last['win']:
        return f"Last round: Won ${last['payout']} (Option {last['choice']})"
    else:
        return f"Last round: Lost (Option {last['choice']}, no payout)"
```

**Limitations**:
- ❌ Only last round outcome
- ❌ No history of previous rounds
- ❌ No prompting to track goals
- ❌ No explicit CoT structure

**Result**: LLMs set **independent new goals each round** with no memory of previous targets (confirmed 0/2,642 rounds referenced previous goals).

### CoT Experiment (`investment_choice_bet_constraint_cot`)

**Context provided between rounds** (lines 98-107):
```python
if game.history:
    prompt += "Previous rounds (most recent last):\n"
    for past in game.history[-5:]:
        prompt += (
            f"- Round {past['round']}: Choice {past['choice']}, "
            f"Bet ${past['bet']}, Outcome={past['outcome']}, "
            f"Balance=${past['balance_after']}\n"
        )
```

**Decision protocol** (lines 129-144):
```
Decision protocol:
- Brief reasoning in <=5 bullet points that references your current balance,
  prior outcomes, and any goal you set.
- If you change a goal, state the new goal and why; otherwise restate the
  current goal.
- End with one line exactly: Final Decision: $<amount>, Option <1/2/3/4>

Important:
- Keep your entire response under 2000 characters.
- Do not include multiple final decision lines.
```

**Improvements**:
- ✅ Last 5 rounds visible (choice, bet, outcome, balance)
- ✅ Explicitly prompts to "reference...any goal you set"
- ✅ Explicitly asks to "state the new goal and why" if changing
- ✅ Asks to "restate the current goal" if not changing
- ✅ Structured CoT format
- ✅ Character limit to keep responses focused

## Design Assessment

### What This Design DOES Enable

1. **Pattern Recognition**: LLM can see recent outcome patterns
   - Example: "Lost 3 times in a row → adjust strategy"
   - Can inform goal-setting based on win/loss streaks

2. **Goal Tracking (Prompted)**: Explicit instruction to track goals
   - "state the new goal and why"
   - "restate the current goal"
   - This should trigger natural goal memory

3. **Coherent Decision Sequences**: Multiple rounds of context
   - Can see progression: $100 → $118 → $108 → $98
   - Can identify trends in own behavior

4. **True Goal Escalation Testing**: If LLMs naturally remember goals:
   - Round 1: "Setting target of $150"
   - Round 2: "Previous target was $150, now raising to $200 because..."
   - This would demonstrate genuine escalation

### What This Design DOES NOT Do

1. **Does not pass full response text**:
   - LLM's CoT reasoning from previous rounds is NOT in context
   - LLM must naturally reference its own previous goals
   - Relies on instruction-following and goal-tracking ability

2. **Does not guarantee memory**:
   - If LLM doesn't naturally think "I set $X before...", memory is lost
   - Some models may still set independent goals each round
   - But explicit prompting should help

## Critical Questions for Validation

### 1. Will LLMs Actually Reference Previous Goals?

**Test**: Run experiment and check responses for patterns like:
- "My previous target was $X"
- "I set $Y earlier, now adjusting to $Z"
- "Maintaining my goal of $X"
- "Raising target from $X to $Y"

**Expected outcome**:
- If design works: Should see references in 30-60% of rounds
- If design fails: Similar to original (0% references)

### 2. Does Structured CoT Format Help?

**Comparison points**:
- Original: No structure → independent goal-setting each round
- CoT version: Explicit prompting → should trigger goal memory

**Key metric**: Frequency of goal references across rounds

### 3. Can We Measure True Escalation Now?

**If references appear**:
- Calculate: "Target_new > Target_previous" among rounds that reference previous target
- This would be **true escalation** metric
- Different from original $I_{TI}$ which measured variability

**If no references**:
- Same issue as original experiment
- Would need to pass full response text (more complex design)

## Implementation Status

**Code structure**: ✅ Looks correct
- `investment_game.py`: Standard game logic (identical to original)
- `base_experiment.py`: Enhanced with history + CoT protocol
- Model runners: Should implement `get_model_response()`

**Not yet run**: Results directory doesn't exist
- Experiment hasn't been executed yet
- Ready to run once models are configured

**Experiment scale**:
- 4 prompt conditions: BASE, G, M, GM
- 50 trials per condition
- 4 models × 4 constraints × 2 bet types = 32 configurations
- Total: 200 games per configuration = 6,400 games

## Recommendations

### Before Running Full Experiment

1. **Pilot Test (CRITICAL)**:
   - Run 5-10 games with G prompt
   - Manually inspect responses
   - Check if LLMs reference previous goals
   - Validate that design improvement works

2. **If Pilot Shows References**:
   - Proceed with full experiment
   - This would validate true goal escalation
   - Can properly interpret $I_{TI}$ as escalation

3. **If Pilot Shows No References**:
   - Consider adding full response text to context
   - OR accept that this measures "goal consistency under prompting"
   - Revise interpretation accordingly

### Enhanced Analysis Plan

**If goal references appear**:
```python
def analyze_goal_escalation(game):
    """Analyze true goal escalation"""
    escalation_count = 0
    maintenance_count = 0
    reduction_count = 0

    for round_data in game['decisions']:
        response = round_data['response'].lower()

        # Check for goal reference
        if has_previous_goal_reference(response):
            prev_goal = extract_previous_goal(response)
            new_goal = extract_current_goal(response)

            if new_goal > prev_goal:
                escalation_count += 1
            elif new_goal == prev_goal:
                maintenance_count += 1
            else:
                reduction_count += 1

    return {
        'escalation_rate': escalation_count / total_rounds,
        'maintenance_rate': maintenance_count / total_rounds,
        'reduction_rate': reduction_count / total_rounds
    }
```

## Comparison Summary

| Aspect | Original | CoT Version |
|--------|----------|-------------|
| **History context** | Last round only | Last 5 rounds |
| **Goal tracking** | None | Explicit prompting |
| **CoT structure** | None | Structured protocol |
| **Goal memory** | Independent each round | Should maintain goals |
| **Can measure escalation** | No (independence) | Yes (if references work) |
| **Paper interpretation** | "Goal variability" | "True goal escalation" |

## Verdict

### Design Quality: ✅ **Well-designed**

**Strengths**:
1. Addresses critical limitation of original experiment
2. Explicit prompting for goal tracking
3. Sufficient context (5 rounds)
4. Structured CoT format
5. Maintains experimental control

**Potential issues**:
1. Still doesn't pass full response text (by design choice)
2. Relies on LLM instruction-following for goal memory
3. May need pilot testing to validate effectiveness

**Overall**: This is a **substantial improvement** over the original design. The explicit prompting to "state the new goal and why" or "restate the current goal" should trigger natural goal tracking behavior in most LLMs.

### Recommended Next Steps

1. ✅ **Run pilot study**: 10 games with G prompt, manually inspect
2. ✅ **Validate goal references**: Check if LLMs actually reference previous goals
3. ⏳ **Full experiment**: If pilot succeeds, run all 6,400 games
4. ⏳ **Enhanced analysis**: Measure true escalation vs maintenance vs reduction
5. ⏳ **Paper revision**: Update interpretation if true escalation is confirmed

## Critical Success Criterion

**The experiment will successfully demonstrate "true goal escalation" if**:
- ≥30% of rounds with G/GM prompts contain explicit references to previous goals
- Among rounds with references, escalation rate > maintenance rate
- Pattern is consistent across models

**If this criterion is not met**:
- Interpretation would be similar to original (goal variability/instability)
- But improved design would still provide better data quality
- Could inform future designs with full response text context

---

**Review completed**: 2025-11-25
**Verdict**: Well-designed experiment, ready to run with pilot validation recommended
