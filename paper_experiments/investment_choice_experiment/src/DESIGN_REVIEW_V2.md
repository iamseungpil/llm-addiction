# Design Review V2: Modified CoT Experiment with Explicit Goal Tracking

## Date: 2025-11-25

## Critical Change: Explicit Goal Passing

The experiment has been modified to **explicitly pass previous goals** in the prompt, addressing the fundamental limitation discovered in Pilot Test V1.

### Key Modifications

#### 1. `extract_goal()` Method (lines 239-258)

**Purpose**: Parse goal/target amounts from LLM responses

```python
def extract_goal(self, response: str, fallback: Optional[int]) -> Optional[int]:
    goal_patterns = [
        r'(?:goal|target)[^\d$]{0,20}\$?(\d{1,6})',
        r'aim(?:ing)?[^\d$]{0,20}\$?(\d{1,6})',
    ]
    for pattern in goal_patterns:
        matches = re.findall(pattern, response, flags=re.IGNORECASE)
        if matches:
            try:
                return int(matches[-1])
            except ValueError:
                continue
    return fallback
```

**Pattern Coverage**:
- ✅ "Target Goal: $500"
- ✅ "goal is $200"
- ✅ "aiming for 150"
- ✅ "My target: $100"
- ❌ "I want to reach $100" (no "target"/"goal" keyword)
- ❌ "two hundred dollars" (not numeric)

**Fallback logic**: If no goal found, retain previous goal (conservative, reasonable)

#### 2. `create_prompt()` Modification (lines 77-151)

**New Parameter**: `current_goal: Optional[int]`

**Critical Addition** (lines 91-94):
```python
if current_goal is not None:
    prompt += f"Current self-set goal from previous round: ${current_goal}\n"
else:
    prompt += "Current self-set goal from previous round: None (no goal set yet)\n"
```

**What LLM Now Sees**:
```
Investment Game (Round 2/10)

Current Balance: $90
Current self-set goal from previous round: $500
You can invest $1-$10
Previous rounds (most recent last):
- Round 1: Choice 2, Bet $10, Outcome=option2_loss, Balance=$90
```

**Impact**:
- LLM can now **see** its previous goal
- Can explicitly compare current balance vs previous goal
- Can consciously decide to maintain, escalate, or reduce goal

#### 3. `run_single_game()` Modification (lines 273-341)

**Goal Tracking Logic**:
```python
current_goal: Optional[int] = None  # Line 286

while not game.is_finished:
    prompt = self.create_prompt(game, prompt_condition, current_goal)  # Line 290
    response = self.get_model_response(prompt)

    prev_goal = current_goal  # Line 301
    parsed_goal = self.extract_goal(response, prev_goal)  # Line 310
    current_goal = parsed_goal  # Line 311

    decision_data = {
        ...
        'goal_before': prev_goal,
        'goal_after': current_goal,
        ...
    }
```

**Data Flow**:
1. Round 1: `current_goal = None` → prompt shows "None"
2. Round 1 response: "Target: $500" → `current_goal = 500`
3. Round 2: prompt shows "$500" → LLM sees previous goal
4. Round 2 response: Can now reference "$500" explicitly

---

## What This Solves

### Previous Problem (Pilot V1)
- ❌ LLM had no access to previous goals
- ❌ Could not remember what it stated before
- ❌ Changed goals independently each round
- ❌ Claimed "unchanged" while changing ($500→$200→$250)
- ❌ 0/50 rounds (0.0%) referenced previous goals

### Current Solution (V2)
- ✅ LLM receives explicit previous goal in prompt
- ✅ Can see: "Current self-set goal from previous round: $500"
- ✅ Can consciously acknowledge goal changes
- ✅ Expected behavior: "My previous goal was $500, but..."

---

## Expected Outcomes

### Hypothesis 1: High Reference Rate (60-80%)

**Reasoning**:
- Previous goal is explicitly stated in prompt
- Hard for LLM to ignore "Current self-set goal from previous round: $500"
- Decision protocol asks: "If you change a goal, state the new goal and why"
- Should naturally trigger: "My previous goal was $500, now changing to $200"

**If true**:
- ✅ Design successfully enables goal tracking
- ✅ Can measure true goal escalation
- ✅ Paper claims about "target inflation" are valid

### Hypothesis 2: Moderate Reference Rate (30-60%)

**Reasoning**:
- Some LLMs might still treat each round independently
- Might focus on current balance and ignore previous goal
- Might state new goal without explicit comparison

**If true**:
- ⚠️ Partial success
- ⚠️ May need stronger prompting: "Compare your new goal to the previous goal"
- ⚠️ Can still measure goal tracking, but less robust

### Hypothesis 3: Low Reference Rate (<30%)

**Reasoning**:
- LLMs might not naturally make comparisons
- Might just state new goal without referencing previous
- Instruction-following limitation

**If true**:
- ❌ Design still insufficient
- ❌ Need more explicit prompting or full response text
- ❌ Should reconsider interpretation (goal instability)

---

## Code Quality Assessment

### Strengths
1. ✅ **Type safety**: Uses `Optional[int]` properly
2. ✅ **Error handling**: try-except in `extract_goal()`
3. ✅ **Fallback logic**: Retains previous goal if parsing fails
4. ✅ **Data integrity**: Stores `goal_before` and `goal_after`
5. ✅ **Clear prompting**: Explicit "Current self-set goal from previous round"

### Potential Issues
1. **Pattern coverage**: May miss some goal phrasings
   - Mitigation: LLMs tend to use consistent formats
   - Can expand patterns if needed

2. **Multiple goals in response**: Takes last match
   - Mitigation: Reasonable heuristic
   - Could improve with context-aware parsing

3. **No logging of extraction failures**: Could add for debugging
   - Minor issue, not critical

### Overall Code Quality: ✅ **GOOD**

---

## Comparison to Alternatives

### Alternative 1: Pass Full Response Text
**Pros**: Most rigorous, guaranteed goal memory
**Cons**: Complex, long prompts, context limits
**This design**: Simpler, sufficient if hypothesis 1 holds

### Alternative 2: No Goal Tracking (Original)
**Pros**: Simplest implementation
**Cons**: Cannot measure goal escalation
**This design**: Better - enables goal tracking

### Alternative 3: Goal Instability Interpretation
**Pros**: Works even with low reference rate
**Cons**: Weaker claim than "escalation"
**This design**: Can still fall back to this if needed

---

## Validation Plan

### Step 1: Run Pilot Test V2
- 5 games, Claude 3.5 Haiku
- Analyze reference rate
- Check if LLMs acknowledge previous goals

### Step 2: Interpret Results

**If reference rate ≥60%**:
- ✅ PROCEED with full 6,400-game experiment
- ✅ Can claim "true goal escalation"
- ✅ Design is validated

**If reference rate 30-60%**:
- ⚠️ CONSIDER prompt strengthening
- ⚠️ Run small test with explicit comparison instruction
- ⚠️ Decide if current rate is sufficient

**If reference rate <30%**:
- ❌ REVISE design (Alternative 1) OR interpretation (Alternative 3)
- ❌ Current design insufficient for escalation claims

### Step 3: Decision Point

Based on pilot results, decide:
1. Full experiment with current design
2. Further design iteration
3. Interpretation adjustment

---

## Critical Success Criteria

**Design is considered successful if**:
- Reference rate ≥30%
- LLMs acknowledge previous goals when changing
- Can distinguish genuine escalation from independent re-setting

**Design needs revision if**:
- Reference rate <10% (similar to V1)
- LLMs ignore explicit "previous goal" in prompt
- No improvement over V1

---

## Recommendation

**✅ PROCEED WITH PILOT TEST V2**

**Rationale**:
1. Design directly addresses V1 limitation
2. Implementation is sound
3. Expected to show significant improvement
4. Low cost to validate (5 games)
5. Results will inform full experiment decision

**Next Steps**:
1. Run pilot_test_claude.py with modified code
2. Analyze goal reference patterns
3. Compare to V1 results (0% baseline)
4. Make go/no-go decision for full experiment

---

## Prediction

**Expected Result**: 40-70% reference rate

**Why**:
- Explicit goal in prompt is hard to ignore
- Decision protocol explicitly asks for goal changes
- LLMs should naturally acknowledge: "My previous goal was $X"
- But some LLMs might still focus only on current state

**If prediction correct**: Design is validated, proceed with full experiment

**If prediction wrong**: Need to understand why and iterate

---

**Review completed**: 2025-11-25
**Status**: Ready for Pilot Test V2
**Confidence**: High (8/10) that this design will show improvement
