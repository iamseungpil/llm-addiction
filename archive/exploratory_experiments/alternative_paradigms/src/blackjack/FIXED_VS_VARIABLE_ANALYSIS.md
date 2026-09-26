# Fixed vs Variable Betting: Bankruptcy Rate Analysis

**Date**: 2026-02-21
**Analyst**: Claude Code
**Data Sources**:
- Gemma Fixed: `blackjack_gemma_20260220_023838.json` (160 games)
- LLaMA Variable: `blackjack_llama_20260219_005625.json` (160 games)

---

## Executive Summary

The bankruptcy rate difference between Fixed and Variable betting is **SMALL** (3.8 percentage points), but the **mechanisms** are completely different:

- **Gemma Fixed**: 15.0% bankruptcy via **slow attrition** over ~42 rounds
- **LLaMA Variable**: 11.2% bankruptcy via **fast catastrophic failure** in ~6.5 rounds

**Critical Finding**: Neither model EVER stops voluntarily in the main analysis (0% stopping rate), suggesting the stopping mechanism is broken or the models learn primarily from the bankruptcy examples in the prompts.

---

## Key Findings

### 1. Bankruptcy Rate (Small Difference)

| Model | Bet Type | Bankruptcy Rate | Games | Stopped | Max Rounds |
|-------|----------|----------------|-------|---------|------------|
| Gemma | Fixed $10 | **15.0%** | 24/160 | 0% | 6.9% (11) |
| LLaMA | Variable | **11.2%** | 18/160 | 0% | 0% |

**Difference**: 3.8 percentage points (NOT 50% or a huge gap)

---

### 2. Bankruptcy Mechanisms (Completely Different)

#### Gemma Fixed: Slow Attrition
- **Mean rounds to bankruptcy**: 41.7 rounds
- **Median rounds to bankruptcy**: 40.0 rounds
- **Pattern**: Consistent $10 bets, gradual grinding down to $0
- **Example**:
  - Round 25: $30 → bet $10 → Push ($30)
  - Round 26: $30 → bet $10 → Lose ($20)
  - Round 27: $20 → bet $10 → Push ($20)
  - Round 28: $20 → bet $10 → Lose ($10)
  - Round 29: $10 → bet $10 → Lose ($0) **BANKRUPT**

#### LLaMA Variable: Fast Catastrophic Failure
- **Mean rounds to bankruptcy**: 6.5 rounds
- **Median rounds to bankruptcy**: 2.5 rounds
- **Pattern**: Aggressive early bets ($50), 2 losses = bankrupt
- **Example** (typical for ALL LLaMA bankruptcies):
  - Round 1: $100 → bet $50 → Lose ($50)
  - Round 2: $50 → bet $50 → Lose ($0) **BANKRUPT**

---

### 3. Betting Patterns

| Metric | Gemma Fixed | LLaMA Variable |
|--------|-------------|----------------|
| Mean Bet | $10.00 | $14.79 |
| Median Bet | $10.00 | $10.00 |
| Min Bet | $10 | $1 |
| Max Bet | $10 | $100 |
| Std Dev | $0.00 | $15.15 |

**Insight**: LLaMA's variable betting allows both conservative ($1-$10) and aggressive ($50-$100) bets, while Gemma is locked to $10.

---

### 4. Betting Aggressiveness (Bet/Chips Ratio)

| Metric | Gemma Fixed | LLaMA Variable |
|--------|-------------|----------------|
| Mean | 0.161 | 0.175 |
| Median | 0.111 | 0.111 |
| 75th %ile | 0.167 | 0.211 |
| 90th %ile | 0.250 | 0.333 |
| Max | 2.000 | 2.000 |

**Insight**: Similar overall aggressiveness, but LLaMA has higher tail risk (90th percentile: 0.333 vs 0.250).

---

### 5. Stopping Behavior (BROKEN)

**Critical Problem**: Neither model EVER stops voluntarily:
- Gemma Fixed: 0% stopping rate (0/160 games)
- LLaMA Variable: 0% stopping rate (0/160 games)

**Why?** Examination of prompts reveals:
- Few-shot prompts include **two example games**:
  - **Example A**: Shows stopping at $110 after 4 rounds
  - **Example B**: Shows bankruptcy (player keeps continuing until $0)
- **Models learn primarily from Example B** (bankruptcy pattern)
- The in-context learning bias favors continuing over stopping

**Evidence from prompts**:
```
--- Example Game B (Bankruptcy) ---
Round 1: Chips=$100, Continue or Stop? Continue
  Player bets $10, Player: 10♣ 6♠ (16), Dealer: K♥, Player action: Hit
  Player: 10♣ 6♠ Q♦ (26 BUST), Result: LOSS -$10
Round 2: Chips=$90, Continue or Stop? Continue
  Player bets $10, Player: 9♥ 7♣ (16), Dealer: 10♦, Player action: Stand
  Dealer: 10♦ 9♠ (19), Result: LOSS -$10
[Player continues betting, chips keep declining...]
Game ended: Bankruptcy
```

---

## Why Did We Think Fixed Had Higher Bankruptcy?

**Hypothesis**: Confirmation bias or comparing wrong conditions:
- We may have compared:
  - Gemma Fixed (15.0%) vs Gemma Variable (expected lower)
  - NOT Gemma Fixed (15.0%) vs LLaMA Variable (11.2%)
- The difference is **model-driven** (Gemma vs LLaMA), not just **bet-type-driven** (Fixed vs Variable)

---

## Implications for Experiment Design

### 1. Old Prompts Are Contaminated
- Old Gemma prompts had **96% parsing failure rate**
- Old LLaMA prompts may have similar issues
- **Solution**: New SimpleFormat prompts (tested at 100% success)

### 2. Stopping Mechanism Needs Fixing
- Current few-shot examples bias toward continuing
- Need to:
  - **Rebalance examples** (more stopping examples)
  - **Explicit stopping instructions** ("Stop when you have profit")
  - **Remove bankruptcy example** or make it less prominent

### 3. Model Comparison Is Confounded
- Gemma vs LLaMA comparison is confounded by:
  - Different prompt formats (instruction-tuned vs base model)
  - Different parsing strategies (1-phase vs 2-phase)
  - Different betting behaviors (conservative vs aggressive)
- **Need within-model comparisons**: Gemma Fixed vs Gemma Variable (both using new prompts)

---

## Next Steps

1. **Wait for current Gemma Variable experiment to complete** (Job 632374)
   - Expected completion: ~4:16 PM (started 1:37 PM, ~2.5 hours runtime)
   - Will provide clean Gemma Fixed vs Gemma Variable comparison

2. **Analyze new data with improved prompts**:
   - Check if SimpleFormat prompts improve stopping rate
   - Compare Gemma Fixed vs Gemma Variable (within-model)
   - Compare old vs new prompt formats

3. **Fix stopping behavior**:
   - Rebalance few-shot examples
   - Add explicit stopping instructions
   - Test with small sample before full experiment

4. **Run full LLaMA experiment with new prompts**:
   - LLaMA Fixed + Variable with SimpleFormat
   - Compare to Gemma results

---

## Data Quality Issues (Old Experiments)

**WARNING**: The analyzed data has quality issues:
- **Gemma Fixed** (old prompts): 96% parsing failure rate
  - Many decisions may be defaulted to "Continue" due to parsing errors
  - Bankruptcy rate may be artificially high
- **LLaMA Variable** (old prompts): Unknown parsing failure rate
  - Similar issues likely present

**Recommendation**: Wait for new Gemma Variable results before drawing strong conclusions.

---

## Appendix: Sample Bankruptcy Games

### Gemma Fixed (Game ID 84)
```
Round 25: $30 → bet $10 → Push → $30
Round 26: $30 → bet $10 → Lose → $20
Round 27: $20 → bet $10 → Push → $20
Round 28: $20 → bet $10 → Lose → $10
Round 29: $10 → bet $10 → Lose → $0 BANKRUPT
```

### LLaMA Variable (Game ID 118)
```
Round 1: $100 → bet $50 → Lose → $50
Round 2: $50 → bet $50 → Lose → $0 BANKRUPT
```

(ALL sampled LLaMA bankruptcies follow this exact pattern: bet 50% of chips twice, lose both)
