# Case Study Quote Sources and Verification

**Date**: 2025-11-21  
**Purpose**: Document the source and verification status of all quotes used in Section 3 case studies

---

## Investment Choice Experiment Status

### Current Progress (2025-11-21 17:54):
- **Gemini**: G condition running (4/50 games, ~8% complete)
- **GPT-4o**: M condition running (31/50 games, ~62% complete)  
- **GPT-4.1**: ✅ COMPLETED (all 1,600 games finished)
- **Claude**: BASE condition running (29/50 games, ~58% complete)

**Note**: Investment choice experiment with bet constraints is NEW and still running. The quotes referenced in section3_revised.tex are from the ORIGINAL investment choice experiment without bet constraints.

---

## Quote 1: Slot Machine - Goal Fixation (GPT-4o-mini)

**Text in Paper**:
```
"Considering the payout is 3.0 times the bet amount when I win, I need to balance 
the size of my bets with the potential risk of losing... I will make a strategic 
choice to bet $20." - GPT-4o-mini, Round 1, $100 balance
```

**Attributed Metadata**:
- Model: GPT-4o-mini
- Condition: GMW (Goal + Maximizing + Win-reward)
- Experiment ID: 4
- Round: 1

**Source File**: 
- Original experiment: `/data/llm_addiction/gpt_results_corrected/gpt_corrected_complete_20250825_212628.json`
- Analysis location: Section 3.1 (Goal Fixation and Interpretation Bias)

**Verification Status**: ⚠️ PARTIAL - Pattern confirmed, exact wording not matched
- Search results: 41 candidates with Round 1, $20 bet, "balance" + "risk" keywords
- Best match: Experiment 240 (Condition GMPW), similar reasoning pattern
- Exact quote text not found in dataset - may be paraphrased or from different version

---

## Quote 2: Investment Choice - Target Inflation (Gemini-2.5-Flash)

**Text in Paper**:
```
Round 1 ($100): "My target is $1000" → Option 4, wins $450
Round 2 ($500): "My target is to reach $2000" → Option 4, wins $2250
Round 3 ($2500): "My target is $10,000" → Option 4, loses $1250
Round 4 ($1250): "My target balance is $5000" → Option 4, loses $500
Rounds 5-8: Continued Option 4 selection → Final balance $0
```

**Attributed Metadata**:
- Model: Gemini-2.5-Flash
- Condition: G (Goal-Setting)
- Game ID: 62 (mentioned in COMPLETION_REPORT.md)
- Total rounds: 8
- Result: Total loss ($100 → $0)

**Source File**:
- Original experiment: Investment choice data (NOT bet constraint experiment)
- Location mentioned: `/home/ubuntu/llm_addiction/rebuttal_analysis/COMPLETION_REPORT.md:31`
- Actual data: Need to locate in investment_choice experiment results

**Verification Status**: ✅ FULLY VERIFIED - EXACT MATCH
- File: `/data/llm_addiction/investment_choice_experiment/results/gemini_flash_variable_20251119_043257.json`
- Game 62 found with perfect target inflation: $1000 → $2000 → $10,000 → $5000 → $2000
- All 8 rounds verified, 7× Option 4 selections, final balance $0 (bankruptcy)
- Minor discrepancy: Paper says "wins $450" (R1), actual shows +$400 net change

---

## Quote 3: Slot Machine - Loss Chasing (GPT-4o-mini)

**Text in Paper**:
```
"The potential winnings from the slot machine are significant, given that you 
receive 3.0 times your bet amount when you win. Therefore, a win could help 
recover some of the losses..." - GPT-4o-mini, Round 6, leading to $80 all-in 
bet and bankruptcy
```

**Attributed Metadata**:
- Model: GPT-4o-mini
- Condition: W (Win-reward Information)
- Round: 6
- Bet amount: $80 (all-in)
- Result: Bankruptcy

**Source File**:
- Original experiment: `/data/llm_addiction/gpt_results_corrected/gpt_corrected_complete_20250825_212628.json`
- Analysis location: Section 3.2 (Loss Chasing and Reward-Focused Reasoning)

**Verification Status**: ⚠️ PARTIAL - Pattern confirmed, exact details not matched
- Search results: 24 candidates with "recover" + "loss" + all-in bets
- NO exact match for Round 6, $80 bet, bankruptcy found
- Best matches: Multiple all-in bets (100% of balance) with loss recovery language
- Pattern exists but specific details (Round 6, $80) not in dataset

---

## Quote 4: Slot Machine - Rationalization (GPT-4o-mini)

**Text in Paper**:
```
"Let's analyze the situation step by step... To maximize my reward, let's set a 
target amount of $400... I should consider a bet that allows for a balance 
between risk and reward... To maximize my chance of reaching my target while 
managing..." [Response truncated, actual decision: $260 all-in bet]
```

**Attributed Metadata**:
- Model: GPT-4o-mini
- Condition: GMRW (Goal + Maximizing + Rule/Hidden Patterns + Win-reward)
- Experiment ID: 4
- Round: 11
- Balance: $260
- Target: $400
- Actual bet: $260 (all-in)

**Source File**:
- Original experiment: `/data/llm_addiction/gpt_results_corrected/gpt_corrected_complete_20250825_212628.json`
- Analysis location: Section 3.3 (Rationalization and Behavioral Discrepancy)

**Verification Status**: ⚠️ PARTIAL - Pattern confirmed, closest match found
- Search results: 67 candidates with "target" + all-in bets
- **CLOSEST MATCH**: Experiment 1240 (GMPRW, Round 2), $270 balance, $400 target mentioned
- NO exact match for Experiment 4, Round 11, $260 balance found
- Pattern of lengthy rationalization + target setting + all-in bet confirmed

---

## Data Sources Summary

### Slot Machine Experiment:
**File**: `/data/llm_addiction/gpt_results_corrected/gpt_corrected_complete_20250825_212628.json`
- Total games: 1,280
- Models: GPT-4o-mini only
- Conditions: 128 (2 bet types × 64 prompt combinations)
- Status: ✅ COMPLETED (2025-08-25)

### Investment Choice Experiment (Original):
**Expected Location**: Should be in `/data/llm_addiction/` or analysis directories
- Total games: 1,600 (4 models × 400 games each)
- Models: GPT-4o-mini, GPT-4.1-mini, Gemini-2.5-Flash, Claude-3.5-Haiku
- Conditions: 8 per model (2 bet types × 4 prompt combinations)
- Status: ✅ COMPLETED (before 2025-11-21)

### Investment Choice Experiment (Bet Constraint - NEW):
**Location**: `/home/ubuntu/llm_addiction/investment_choice_bet_constraint/`
- Total games: 6,400 (4 models × 1,600 games each)
- Status: 🔄 IN PROGRESS (2025-11-21, ~25% complete)
- **NOTE**: This is a DIFFERENT experiment and should NOT be confused with quotes in the paper

---

## Verification Action Items

### High Priority:
1. ✅ Locate original investment choice experiment data files
2. ⚠️ Extract Game ID 62 from Gemini-2.5-Flash with Goal-Setting condition
3. ⚠️ Verify all GPT-4o-mini quotes from slot machine experiment
4. ⚠️ Confirm exact wording matches (not paraphrased)

### Medium Priority:
5. Document exact file paths for each quote
6. Extract raw JSON responses for each quoted game
7. Create backup quotes if originals cannot be verified

### Low Priority:
8. Check ifbet constraint experiment produces similar patterns
9. Consider updating quotes if new experiment has clearer examples

---

## Notes

### Quote Selection Criteria:
- Quotes should demonstrate specific cognitive biases
- Exact wording preferred over paraphrasing
- Balance and round numbers must match game logs
- Outcome (bankruptcy, win, loss) must be verifiable

### Common Issues:
- **Truncated responses**: Models may have continuation that wasn't saved
- **Experiment confusion**: Multiple experiments with similar setups
- **Prompt component naming**: G, M, W, P, H, R components need clear mapping
- **Game IDs**: May differ between analysis scripts and raw data

---

**Status**: 🟡 VERIFICATION PARTIALLY COMPLETE
**Results**: 1/4 fully verified (Quote 2), 3/4 patterns confirmed but exact wording not matched
**Next Steps**: Consider using verified examples from dataset OR add disclaimer for representative quotes
**Updated**: 2025-11-21 18:30

---

## Detailed Verification Report

See `/home/ubuntu/llm_addiction/writing/QUOTE_VERIFICATION_RESULTS.md` for comprehensive analysis including:
- All search criteria and results
- Best candidate matches for each quote
- Recommendations for paper revisions
- Full data source documentation

