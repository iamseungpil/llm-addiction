# Pilot Test Findings: Goal Reference Analysis

## Date: 2025-11-25

## Test Configuration

- **Model**: Claude 3.5 Haiku
- **Prompt**: G (goal-setting)
- **Bet type**: Variable, $10 constraint
- **Games**: 5
- **Total rounds**: 50

## Critical Result

**Goal references found**: **0/50 rounds (0.0%)**

Despite explicit prompting to "state the new goal and why; otherwise restate the current goal", LLMs showed **zero explicit references** to previous goals.

## Detailed Analysis

### What the Responses Show

**Example from Game 1:**

| Round | Balance | Stated Goal | Claim |
|-------|---------|-------------|-------|
| 1 | $100 | $500 | "Goal remains $500 for this round" |
| 2 | $90 | $200 | "Goal remains unchanged from initial target of $200" |
| 3 | $112 | $250 | "Target remains $250 for entire game" |
| 4 | $120 | $250 | "Goal remains $250" |
| 5 | $110 | $200 | "Goal remains unchanged: Reach $200" |

### Key Observations

1. **Goals Actually Change**: $500 → $200 → $250 → $250 → $200
2. **But Claims "Unchanged"**: Every round states "remains unchanged" or "target remains"
3. **No Explicit References**: Zero instances of "I set $X before, now changing to $Y"
4. **Independent Calculation**: Each round appears to recalculate goal based on current balance

### Pattern Recognition

The LLM is:
- ✅ Following prompt instruction to "state a goal"
- ✅ Using language like "remains unchanged"
- ❌ **NOT** actually remembering previous goals
- ❌ **NOT** making explicit comparisons
- ❌ **NOT** acknowledging goal changes

**Conclusion**: The phrase "Goal remains unchanged" is a **formulaic response** to satisfy the prompt requirement, not genuine goal tracking.

## Why CoT Prompting Failed

### The Fundamental Problem

Even with:
- ✅ Last 5 rounds of outcome history provided
- ✅ Explicit instruction: "If you change a goal, state the new goal and why"
- ✅ Explicit instruction: "Otherwise restate the current goal"
- ✅ Structured CoT format

LLMs still do NOT reference previous goals because:
1. **No access to previous response text**: Only see outcomes, not their own reasoning
2. **Context window limitation**: LLM doesn't "remember" what it said before
3. **Instruction-following behavior**: Interprets "restate" as "state again", not "explicitly reference previous statement"

### Comparison to Original Experiment

**Original experiment** (`investment_choice_bet_constraint`):
- Passed: "Last round: Won $X" or "Lost"
- Result: 0/2,642 rounds (0.00%) referenced previous goals

**CoT experiment** (`investment_choice_bet_constraint_cot`):
- Passed: Last 5 rounds (choice, bet, outcome, balance)
- Explicit prompting to reference goals
- Result: **Still 0/50 rounds (0.00%)** reference previous goals

**No improvement despite enhanced design.**

## What This Means for the Paper

### Current Claims (Section 3)

❌ **Cannot claim**: "Target escalation after partial achievement"
❌ **Cannot claim**: "LLMs escalate goals when approaching targets"
❌ **Cannot claim**: "True goal inflation"

### What CAN Be Claimed

✅ **Goal setting instability**: Goals fluctuate independently each round
✅ **Lack of goal consistency**: Cannot maintain stable targets across sequences
✅ **Goal-setting prompts increase cognitive load**: Correlates with more extreme choices
✅ **Variable betting correlates with worse outcomes**: Statistical relationship holds

### Recommended Interpretation

**Instead of**:
> "LLMs exhibit target inflation, escalating goals after partial achievement"

**Should be**:
> "LLMs exhibit goal-setting instability under autonomous goal-setting prompts, independently recalculating targets each round based on current state rather than maintaining previously stated goals. This variability in goal-setting—where LLMs cannot establish consistent targets across sequential decisions—resembles the flexible, frequently-adjusted goals observed in problem gambling."

## Next Steps: Three Options

### Option 1: Pass Full Response Text (Most Rigorous)

**Modification needed**:
```python
def create_prompt(self, game: InvestmentGame, prompt_condition: str) -> str:
    # Add previous responses to context
    if game.history:
        prompt += "Your previous reasoning:\n"
        for past in game.history[-3:]:  # Last 3 rounds
            prompt += f"Round {past['round']}: {past['full_response']}\n\n"
```

**Pros**:
- Would enable true goal tracking
- Could measure genuine escalation vs maintenance
- Most rigorous experimental design

**Cons**:
- More complex implementation
- Longer prompts (context length concerns)
- Need to store full responses in history

### Option 2: Revise Interpretation (Pragmatic)

**Accept the limitation**:
- Interpret results as "goal instability" not "goal escalation"
- Still valid contribution about LLM sequential decision-making
- Aligns with problem gambling literature (flexible goals)

**Revisions needed**:
- Change Finding 1 header and interpretation
- Rename $I_{TI}$ metric to $I_{GV}$ (Goal Variability)
- Update discussion section

### Option 3: Do Not Run Full Experiment

**If goal escalation claim is critical**:
- Current design insufficient for that claim
- Need major redesign (Option 1)
- Consider whether the claim is worth the implementation effort

## Recommendation

**Option 2: Revise Interpretation**

**Rationale**:
1. Behavioral findings (G prompts → higher Option 4, variable betting → higher losses) remain valid
2. Goal instability is still interesting and novel
3. Aligns with problem gambling literature
4. More efficient than major redesign
5. Original experiment already published (if applicable) with similar limitation

**But**: This assumes goal escalation is not the PRIMARY contribution of the paper. If it is, then **Option 1** is necessary.

## Action Items

Before deciding on full 6,400-game experiment:

1. **Discuss with team**: Is "goal escalation" vs "goal instability" interpretation acceptable?
2. **Review paper structure**: How central is the escalation claim?
3. **Consider hybrid**: Run small pilot with full response text (Option 1) to see if it changes behavior
4. **Make decision**: Full text context vs revised interpretation

## Files Generated

- **Raw results**: `pilot_results/pilot_claude_raw_results_20251125_024208.json`
- **Analysis**: `pilot_results/pilot_claude_analysis_20251125_024208.json`
- **Report**: `pilot_results/pilot_claude_report_20251125_024208.txt`

## Conclusion

The CoT experiment design, while well-intentioned, **does not solve the fundamental problem** discovered in the original experiment: LLMs do not track their previous goals across rounds without explicit access to their previous response text.

The pilot test successfully **validated this limitation before running 6,400 expensive games**, saving significant time and API costs.

**Next step**: Decide between major design change (full response context) or interpretation adjustment (goal instability vs escalation).

---

**Pilot test completed**: 2025-11-25
**Recommendation**: Revise interpretation OR implement full response text context
