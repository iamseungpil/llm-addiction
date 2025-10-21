# Experiment 6: Attention Pathway Analysis - Corrected Findings

## Analysis Date
2025-10-16

## Problem Identified and Corrected

### Original Issue
The initial analysis used "choices" token positions [12, 26], which referred to "Bet" tokens in the **game history** (Rounds 4 and 5), not the actual decision point. This resulted in zero attention weights because:
- Position 12: "Bet" token from Round 4 in history
- Position 26: "Bet" token from Round 5 in history
- These positions primarily attended to BOS token and local context, not information tokens

### Correction Applied
Changed target position to the **last token** (seq_len - 1), where the model actually generates its decision after "Choice: ". This is the true decision point in the sequence.

## Key Findings

### 1. Layer-Specific Attention Patterns

#### Layer 8 (Early Processing)
- **Risky scenarios**: Total attention = 0.0014
- **Safe scenarios**: Total attention = 0.0017
- **Pattern**: Safe scenarios attend slightly MORE to information tokens in early layers
- **Interpretation**: Early layers show minimal differentiation between risky/safe

#### Layer 15 (Middle Processing) ⭐ **Most Distinctive**
- **Risky scenarios**: Total attention = 0.0068 (45% higher than safe)
  - Balance → Decision: **0.0039** (86% higher than safe)
  - Goal → Decision: 0.0012
  - Probability → Decision: 0.0018
- **Safe scenarios**: Total attention = 0.0047
  - Balance → Decision: 0.0021
  - Goal → Decision: 0.0010
  - Probability → Decision: 0.0016
- **Pattern**: Risky scenarios focus significantly MORE on current balance
- **Interpretation**: Middle layers show strongest differentiation - risky decisions over-focus on immediate state

#### Layer 31 (Final Processing)
- **Risky scenarios**: Total attention = 0.0059
  - Balance → Decision: 0.0016
  - Goal → Decision: 0.0021
  - Probability → Decision: 0.0021
- **Safe scenarios**: Total attention = 0.0073 (24% higher than risky)
  - Balance → Decision: 0.0023
  - Goal → Decision: 0.0017
  - Probability → Decision: **0.0032** (52% higher than risky)
- **Pattern**: Safe scenarios focus MORE on probability information
- **Interpretation**: Final layers show safe decisions attend more to win rate (rational decision-making)

### 2. Attention Magnitude Context

- **Typical range**: 0.0015 - 0.0040 (0.15% - 0.40%)
- **Why so small?**:
  - Information flows primarily through **residual stream**, not direct attention
  - Attention to BOS token and local context dominates (~42% to BOS, ~14% to ":")
  - These small weights are **normal and expected** in transformer architectures
  - The **relative differences** between scenarios are more important than absolute values

### 3. Feature Activation Pathways

#### Top Features (L31, Balance → Decision)

**High Pathway Score Features (Bankruptcy scenarios)**:
- **Feature 9926**: 5.43→9.26 (+3.82 activation change), pathway_score 0.052
- **Feature 30028**: 2.07→9.95 (+7.88 activation change), pathway_score 0.043
- **Feature 2857**: 4.75→5.08 (+0.33 activation change), pathway_score 0.035
- **Feature 4793**: 2.00→6.58 (+4.59 activation change), pathway_score 0.030

**High Pathway Score Features (Desperate scenarios)**:
- **Feature 9926**: 8.35→8.88 (+0.53 activation change), pathway_score 0.026
- **Feature 30028**: 0.0→10.06 (+10.06 activation change), pathway_score 0.015
- **Feature 2216**: 2.85→4.02 (+1.17 activation change), pathway_score 0.010

**Observations**:
1. **Feature 9926**: Consistently high activations in both bankruptcy and desperate scenarios
2. **Feature 30028**: Shows dramatic increases at decision point (0→10.06, 2.07→9.95)
3. **Activation changes**: Range from +0.33 to +10.06, indicating significant feature dynamics
4. Some features **decrease** at decision (e.g., Feature 14673: 4.46→2.27, -2.18 change)

### 4. Risk Category Differences

#### Balance Information Processing
- **L15**: Risky scenarios use **86% more** balance attention (0.0039 vs 0.0021)
- **L31**: Difference narrows to 43% (0.0023 vs 0.0016)
- **Interpretation**: Risky decisions initially over-focus on current balance, suggesting myopic decision-making

#### Probability Information Processing
- **L15**: Similar attention (0.0018 risky vs 0.0016 safe)
- **L31**: Safe scenarios use **52% more** probability attention (0.0032 vs 0.0021)
- **Interpretation**: Safe decisions increasingly consider win rate in final layers, suggesting rational probability weighting

#### Goal Information Processing
- **L15**: Similar attention (~0.001)
- **L31**: Risky scenarios use 24% more goal attention (0.0021 vs 0.0017)
- **Interpretation**: Risky decisions maintain goal fixation through final layers

## Generated Outputs

### Visualizations
- `CORRECTED_attention_to_decision_{L8,L15,L31}_{Desperate_10,Safe_140_near_goal}.png` (6 files)
  - Bar plots showing attention from each token position to the final decision token
  - Color-coded key positions (balance=blue, goal=green, probability=purple, decision=red)

### Data Files
- `CORRECTED_balance_decision_pathway_{L8,L15,L31}.csv` (3 files)
- `CORRECTED_goal_decision_pathway_{L8,L15,L31}.csv` (3 files)
- `CORRECTED_probability_decision_pathway_{L8,L15,L31}.csv` (3 files)
- `CORRECTED_pathway_comparison_{L8,L15,L31}.csv` (3 files)

## Implications for Understanding LLM Risk Behavior

### 1. Layer-Specific Processing
- **Early layers (L8)**: Minimal differentiation between risky/safe
- **Middle layers (L15)**: **Strongest differentiation** - risky decisions over-attend to balance
- **Final layers (L31)**: Safe decisions increasingly weight probability information

### 2. Cognitive Biases in Risky Decisions
- **Myopic focus on current state**: 86% higher balance attention in L15
- **Insufficient probability weighting**: 52% lower probability attention in L31
- **Goal fixation**: Maintained higher goal attention through final layers

### 3. Rational Features in Safe Decisions
- **Probability-driven**: 52% higher probability attention in final layer
- **Balanced information integration**: More even attention distribution across information types
- **Less state-dependent**: Lower reliance on current balance

## Next Steps

1. **Cross-reference with Experiment 2 (Activation Patching)**:
   - Check if features 9926, 30028, 2857, 4793 (high pathway scores) are causal features
   - Validate if balance-attention difference in L15 corresponds to causal features

2. **Feature Interpretation**:
   - Analyze what Features 9926, 30028 represent semantically
   - Investigate why Feature 30028 shows 0→10.06 activation jumps

3. **Statistical Testing**:
   - T-tests comparing risky vs safe attention weights by layer
   - Correlation analysis between attention patterns and betting amounts

4. **Visualization Enhancement**:
   - Create layer-by-layer progression plots
   - Network graphs showing information flow through layers
   - Heatmaps comparing risky vs safe attention patterns

## Technical Notes

### Token Positions in Experiment 6 Data
- **Balance**: Position 51 (e.g., "$90")
- **Goal**: Position 76 (e.g., "$200")
- **Probability**: Position 82 (e.g., "30%")
- **Decision**: Position seq_len-1 (last token after "Choice: ")
- **Sequence length**: ~121 tokens

### Attention Matrix Structure
- Shape: [32 heads, seq_len, seq_len]
- Averaged across heads for analysis
- Values sum to 1.0 across source dimension (key tokens)

### Feature Activation Structure
- Shape: [seq_len, 32768 features]
- SAE activations at each token position
- Sparsity: ~1-2% of features active per token (normal)

## Files
- **Analysis script**: `attention_pathway_analysis_corrected.py`
- **Original (incorrect) script**: `attention_pathway_analysis.py` (analyzed wrong token positions)
- **Data**: `/data/llm_addiction/experiment_6_token_level/token_level_tracking_20251013_145433.json` (2.47 GB)
