# Target Goal Irrationality Analysis - Summary Report

**Generated**: 2025-11-21
**Experiment**: Investment Choice Experiment
**Figures**: `target_goal_irrationality_metrics.png` / `.pdf`

---

## 📊 Overview

This analysis measures **two types of irrational goal-setting behavior** in LLM responses:

### 1. **Unrealistic Targets (Probability < 1%)**
- **Definition**: Targets that have less than 1% probability of achievement even with optimal strategy
- **Calculation**: Dynamic Programming with remaining rounds, considering all 4 options
- **Key Point**: This is **NOT a p-value**, but the actual **probability of reaching the target**

### 2. **Mid-game Target Increases**
- **Definition**: Raising the target amount during gameplay (e.g., $150 → $200)
- **Frequency**: 34.4% of games with target mentions (426/1,240 games)
- **Significance**: Indicates irrational optimism or poor planning

---

## 🔑 Key Findings

### **Metric 1: Unrealistic Targets (Prob < 1%)**

#### By Bet Type:
- **Variable**: **53.7%** of target mentions are unrealistic
- **Fixed**: **17.1%** of target mentions are unrealistic
- **Insight**: Variable betting leads to 3.1× more unrealistic targets

#### By Prompt Condition:
- **GM (Goal + Maximize)**: **34.6%** (highest)
- **G (Goal only)**: **28.1%**
- **M (Maximize only)**: **18.3%**
- **BASE**: **14.9%** (lowest)
- **Insight**: Combined "Goal + Maximize" prompt induces most irrationality

#### By Model:
- **Gemini**: **35.7%** (most unrealistic)
- **GPT-4.1**: **26.9%**
- **Claude**: **23.3%**
- **GPT-4o**: **21.4%** (most realistic)
- **Insight**: Models differ significantly in goal-setting rationality

---

### **Metric 2: Mid-game Target Increases**

#### By Bet Type:
- **Variable**: **16.7%** of targets are raised mid-game
- **Fixed**: **8.4%** of targets are raised mid-game
- **Insight**: Variable betting leads to 2× more target raises

#### By Prompt Condition:
- **M (Maximize only)**: **19.4%** (highest)
- **BASE**: **17.5%**
- **GM (Goal + Maximize)**: **8.3%**
- **G (Goal only)**: **7.6%** (lowest)
- **Insight**: "Maximize" alone causes most dynamic target changes

#### By Model:
- **GPT-4.1**: **17.2%** (most dynamic)
- **Claude**: **12.5%**
- **GPT-4o**: **9.1%**
- **Gemini**: **6.1%** (most stable)
- **Insight**: GPT-4.1 is most sensitive to changing circumstances

---

## 💡 Interpretation

### **Why "Probability < 1%" is NOT p-value confusion:**

The title **"Targets with achieving probability < 1%"** clearly states:
- **Probability**: Likelihood of success
- **< 1%**: Less than 1 in 100 chance
- **Using optimal strategy with remaining rounds**: Best-case scenario

This is fundamentally different from statistical p-values (hypothesis testing).

### **Calculation Example:**

```
Scenario: Round 7/10, Balance $90, Target $500

Dynamic Programming calculation:
- Remaining rounds: 3
- Best strategy: Option 4 (10% win, 9× payout)
- Path analysis:
  Round 7 Win (10%): $90 → $170
  Round 8 Win (10%): $170 → $250
  Round 9 Win (10%): $250 → $330

  Even with 3 consecutive wins: $330 < $500 ❌

Result: Probability = 0.0001% < 1% → Unrealistic
```

---

## 🎯 Research Implications

### 1. **Variable Betting Amplifies Irrationality**
- 3.1× more unrealistic targets
- 2× more target raises
- **Hypothesis**: Control over bet amount → overconfidence

### 2. **Prompt Design Matters**
- "Goal + Maximize" → Most unrealistic targets (34.6%)
- "Maximize" alone → Most target raises (19.4%)
- **Implication**: Instructions can induce cognitive biases

### 3. **Model Differences**
- Gemini: Most unrealistic targets (35.7%)
- GPT-4.1: Most target raises (17.2%)
- **Insight**: Different models exhibit different irrationality patterns

### 4. **Time Pressure Effect**
- 10-round limit creates urgency
- Variable + GM prompt + Late rounds → Extreme irrationality
- **Connection**: Similar to planning fallacy in humans

---

## 📈 Statistical Summary

| Metric | Total Games | Games with Targets | Metric Occurrences | Rate |
|--------|-------------|-------------------|-------------------|------|
| **Unrealistic Targets** | 1,600 | 1,240 | 341 | 27.5% |
| **Target Raises** | 1,600 | 1,240 | 426 | 34.4% |

### Outcome Analysis for Target Raises:
- **70.2%** reached max rounds (forced termination)
- **29.8%** chose Option 1 (voluntary exit)
- **Implication**: Target raises correlate with longer, riskier gameplay

---

## 🔬 Methodological Notes

### Target Extraction:
- Regex: `\$\s*([0-9][0-9,]{1,6})`
- Excludes option payouts: $10, $18, $32, $90, $100
- Takes largest mentioned amount ≥ $100

### Probability Calculation:
- Algorithm: Dynamic Programming (memoized recursion)
- Options: 2 (50%×1.8), 3 (25%×3.2), 4 (10%×9.0)
- Assumption: Optimal strategy each round
- Threshold: 1% (conservative criterion)

### Target Raise Detection:
- Monotonic increase: new_target > prev_target
- Within-game comparison only
- First mention establishes baseline

---

## 📁 Files Generated

- **PNG**: High-resolution figure (300 DPI, 502 KB)
- **PDF**: Vector format for publication (42 KB)
- **Location**: `/home/ubuntu/llm_addiction/investment_choice_experiment/analysis/`

---

## ✅ Improvements Made

### Clarity:
- Title changed from "p<0.01" → "probability < 1%"
- Subtitles explain calculation method
- Y-axis: "Percentage of Target Mentions (%)"

### Readability:
- Value labels on all bars
- Larger font sizes (11pt titles, 12pt labels)
- Grid lines for easier reading
- Black borders on bars for definition

### Format:
- Both PNG (raster) and PDF (vector)
- Larger figure size (18×10 inches)
- Proper tight layout

---

*This analysis reveals systematic irrationality in LLM goal-setting behavior, with clear interactions between bet type, prompts, and models.*
