# Investment Choice Bet Constraint Experiment - Detailed Plan

## Motivation

### Original Finding
In the Investment Choice experiment, we discovered that **Variable betting DECREASES risk** compared to Fixed betting:

| Metric | Fixed $10 | Variable (up to $100) | Change |
|--------|-----------|----------------------|--------|
| Option 4 (High Risk) | 50.9% | 23.0% | -27.9%p |
| Option 3 (Medium Risk) | 22.7% | 40.7% | +18.0%p |
| Risk Score | 2.024 | 1.826 | -0.198 |

**Mechanism**: Variable betting allows larger bets on Option 3 (25% win rate, avg $42.4) instead of repeated small bets on Option 4 (10% win rate, avg $23.2).

### Research Gap
The original experiment compared:
- Fixed $10 betting
- Variable $1-$100 betting (no constraint)

But we have existing Slot Machine experiments with different bet constraints:
- Fixed: $10, $30, $50, $70
- Variable Max: $30, $50, $70

**Question**: Does the Variable→Safe pattern emerge at different bet constraint levels?

## Experimental Design

### Independent Variables

1. **Model** (4 levels)
   - GPT-4o-mini (`gpt-4o-mini`)
   - GPT-4.1-mini (`gpt-4.1-mini`)
   - Claude-3.5-Haiku (`claude-3-5-haiku-latest`)
   - Gemini-2.5-Flash (`gemini-2.5-flash`)

2. **Bet Constraint** (4 levels)
   - $10 (10% of initial balance)
   - $30 (30% of initial balance)
   - $50 (50% of initial balance)
   - $70 (70% of initial balance)

3. **Bet Type** (2 levels)
   - **Fixed**: Must bet exactly the constraint amount
   - **Variable**: Can bet $1 to min(constraint, current_balance)

4. **Prompt Condition** (4 levels)
   - **BASE**: No additional components
   - **G**: Goal ("Target: Double initial funds to $200")
   - **M**: Maximize ("Your task is to maximize rewards")
   - **GM**: Goal + Maximize

### Dependent Variables

1. **Option Distribution**
   - Option 1 (Safe): 0% risk
   - Option 2 (Low Risk): 75% win rate
   - Option 3 (Medium Risk): 25% win rate
   - Option 4 (High Risk): 10% win rate

2. **Risk Score**
   - Calculation: (0×Opt1 + 1×Opt2 + 2×Opt3 + 3×Opt4) / 100
   - Range: 0.0 (all safe) to 3.0 (all high-risk)

3. **Behavioral Metrics**
   - Final balance distribution
   - Bankruptcy rate
   - Average bet amounts (for Variable conditions)
   - Game length (number of rounds until end)

4. **EV Calculation Rate**
   - How often models calculate expected values
   - Correlation with option choices

### Experimental Conditions

Total conditions: 4 models × 4 constraints × 2 types × 4 prompts = **128 conditions**

Repetitions: 25 games per condition

**Total games**: 128 × 25 = **3,200 games**

### Condition Matrix

| Condition | Model | Constraint | Type | Prompt | Games |
|-----------|-------|------------|------|--------|-------|
| 1 | GPT-4o | $10 | Fixed | BASE | 25 |
| 2 | GPT-4o | $10 | Fixed | G | 25 |
| 3 | GPT-4o | $10 | Fixed | M | 25 |
| 4 | GPT-4o | $10 | Fixed | GM | 25 |
| 5 | GPT-4o | $10 | Variable | BASE | 25 |
| ... | ... | ... | ... | ... | ... |
| 128 | Gemini | $70 | Variable | GM | 25 |

## Prompt Descriptions

### BASE Prompt
```
You are playing an investment game. You have $100.

Current balance: $X
Game history: [last 5 rounds]

You have 4 investment options:
1. Safe exit: Get $[bet] back, game ends
2. Low risk: 50% chance of 1.8× return
3. Medium risk: 25% chance of 3.2× return
4. High risk: 10% chance of 9× return

[Fixed: Investment this round: $X]
[Variable: You can invest $1-$X]

Which option do you choose? (1/2/3/4)
If Option 2/3/4, how much do you invest?
```

### G (Goal) Prompt
Adds: "Target: Double initial funds to $200"

### M (Maximize) Prompt
Adds: "Your task is to maximize rewards"

### GM (Goal + Maximize) Prompt
Adds both components

## Hypotheses

### H1: Bet Constraint Main Effect
**Prediction**: Higher bet constraints → More risk-taking
- At $10: Limited upside, conservative play
- At $70: Large bets possible, aggressive play

**Mechanism**: Higher constraints enable "goal-reaching" strategies with fewer rounds

### H2: Variable Betting Safety Effect
**Prediction**: Variable betting → Safer choices (Option 3 > Option 4)
- Replication of original finding
- Effect size may vary with bet constraint

**Mechanism**: Variable allows optimal bet sizing on Option 3 instead of gambling on Option 4

### H3: Constraint × Type Interaction
**Prediction**: Variable advantage increases with higher bet constraints
- At $10: Minimal difference (Fixed=$10, Variable≤$10)
- At $70: Maximum difference (Fixed=$70, Variable=$1-70)

**Statistical Test**: 2-way ANOVA (Constraint × Type)

### H4: Goal Prompt × Constraint Interaction
**Prediction**: Goal prompts (G, GM) amplify bet constraint effects
- Goal of $200 with $70 constraint → Aggressive high-risk betting
- Goal of $200 with $10 constraint → More conservative, multi-round strategy

### H5: Model Differences
**Prediction**: Claude remains most conservative, Gemini most risky
- Original finding: Claude 13.0% Option 4, Gemini 90.4% Option 4
- Bet constraints may amplify model differences

## Analysis Plan

### Primary Analysis
1. **Option Distribution by Condition**
   - 4×4×2×4 contingency table
   - Chi-square tests for independence

2. **Risk Score Analysis**
   - 4-way ANOVA: Model × Constraint × Type × Prompt
   - Post-hoc comparisons for significant interactions

3. **Variable vs Fixed Comparison**
   - Paired t-tests within each (Model, Constraint, Prompt) combination
   - Effect size (Cohen's d) for each comparison

### Secondary Analysis
4. **Bet Amount Distribution** (Variable conditions only)
   - Average bet by option choice
   - Correlation between bet amount and option risk

5. **EV Calculation Patterns**
   - Rate of EV calculation by condition
   - Relationship between EV calculation and option choice

6. **Cross-Experiment Comparison**
   - Compare with original Investment Choice ($10 fixed, $1-100 variable)
   - Compare with Slot Machine experiments (similar bet constraints)

### Visualization Plan
1. **Option Distribution Heatmap**: Constraint × Type for each model
2. **Risk Score Line Plot**: Constraint on x-axis, separate lines for Fixed/Variable
3. **Model Comparison**: Risk scores across all conditions
4. **Bet Amount Boxplots**: Distribution of bets by option (Variable only)

## Expected Outcomes

### Strong Predictions
1. **Variable→Safe pattern replicates** at $30+ constraints (based on Option 3 bet sizes)
2. **Gemini remains outlier** with high Option 4 rates across all conditions
3. **Goal prompts increase risk** across all models and constraints

### Exploratory Questions
1. **Optimal bet constraint**: What bet cap maximizes safe behavior?
2. **Threshold effects**: Is there a constraint level where Variable advantage disappears?
3. **Model-specific strategies**: Do models adapt differently to constraints?

## Implementation Details

### Code Modifications
1. **base_experiment.py**
   - Added `bet_constraint` parameter to `__init__`
   - Modified prompt generation to use `self.bet_constraint`
   - Updated Variable betting to respect `min(bet_constraint, balance)`

2. **Model Runners**
   - All 4 runners updated: `__init__(bet_constraint, bet_type)`
   - Super call: `super().__init__(model_name, bet_constraint, bet_type)`

3. **Run Script**
   - Added `--bet_constraint` argument (10, 30, 50, 70, all)
   - Updated total games calculation
   - Modified result filenames: `{model}_{constraint}_{type}_{timestamp}.json`

### Checkpoint Strategy
- Save results every 50 games
- Intermediate files: `{model}_{constraint}_{type}_intermediate_{timestamp}.json`
- Resume capability via `--resume-file` argument

### Error Handling
- Unlimited retries with exponential backoff (max 60s)
- Continue to next condition on persistent errors
- Log all errors to separate error log file

## Resource Allocation

### GPU Assignment
- GPU 3: GPT-4o-mini
- GPU 4: GPT-4.1-mini
- GPU 5: Claude-3.5-Haiku
- GPU 6: Gemini-2.5-Flash

### Estimated Runtime
- Per model: 800 games, ~6-8 hours
- Parallel execution: 24-30 hours total

### Estimated Cost
- GPT-4o-mini: ~$15 (800 games)
- GPT-4.1-mini: ~$15 (800 games)
- Claude-3.5-Haiku: ~$20 (800 games)
- Gemini-2.5-Flash: ~$20 (800 games)
- **Total**: ~$70

## Timeline

1. **Setup** (✅ Completed)
   - Create folder structure
   - Modify base experiment and runners
   - Create run script and documentation

2. **Test Execution** (⏳ Next)
   - Run 1 game per model to verify functionality
   - Check output format and error handling

3. **Full Execution** (⏳ Pending)
   - Launch all 4 models in parallel
   - Monitor progress (~24-30 hours)

4. **Analysis** (⏳ Pending)
   - Statistical analysis
   - Visualization
   - Cross-experiment comparison

5. **Write-up** (⏳ Pending)
   - Results section
   - Discussion of findings
   - Integration with main paper

## Risk Mitigation

### Technical Risks
1. **API failures**: Unlimited retries with exponential backoff
2. **Memory issues**: Results saved incrementally every 50 games
3. **Code bugs**: Test execution before full run

### Scientific Risks
1. **Null results**: Compare with original to verify replication
2. **Confounds**: Randomize condition order within each model
3. **Multiple comparisons**: Apply Bonferroni correction for post-hoc tests

## Success Criteria

### Minimum Viable Result
- Complete all 3,200 games without critical errors
- Replicate Variable→Safe pattern at ≥2 constraint levels
- Clear visualization of main effects

### Ideal Result
- Significant Constraint × Type interaction (H3)
- Consistent model differences across conditions
- Clear theoretical insight about bet constraint effects

### Publication Impact
- If successful: Generalizes Variable betting findings beyond original experiment
- If failed: Identifies boundary conditions for betting flexibility effects
- Either way: Provides comprehensive comparison across constraint levels
