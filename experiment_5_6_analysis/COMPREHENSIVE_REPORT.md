# Comprehensive Analysis Report: Experiments 5 & 6
## Context-Dependent Feature Encoding in LLaMA Gambling Behavior

**Analysis Date:** 2025-10-16
**Analyst:** Claude Code
**Data Sources:**
- Experiment 5: Multi-round mean value patching (441 features, layers 25-31)
- Experiment 6: Token-level activation tracking (10 scenarios, layers 8, 15, 31)

---

## Executive Summary

This analysis reveals a **paradoxical result**: forcing SAE features to constant mean values (either "safe" or "risky") **increases bankruptcy rates** rather than steering behavior as expected. This suggests these features encode **context-dependent information** that requires dynamic variation based on game state (balance, round number, history).

### Key Findings

1. **Harmful Intervention Effect**: 215 features (48.8%) increased bankruptcy in BOTH safe and risky patching conditions
2. **Mean Forcing Paradox**: Baseline 53.5% bankruptcy → 62.7% (safe mean) / 62.6% (risky mean)
3. **Critical Dynamic Features**: Features cannot be fixed to constant values without breaking decision-making
4. **Token Position Specificity**: Harmful features show distinct activation patterns at balance, goal, and probability tokens
5. **Layer-Specific Effects**: Layer 27 uniquely shows protective effects (-3.9%), while Layer 26 is most harmful (+13.9%)

---

## 1. Experiment 5 Analysis: Mean Value Patching Effects

### 1.1 Overall Statistics

| Metric | Baseline | Safe Mean | Bankrupt Mean |
|--------|----------|-----------|---------------|
| **Bankruptcy Rate** | 53.5% | 62.7% (+9.2%) | 62.6% (+9.1%) |
| **Final Balance** | $56.44 | $46.11 (-$10.33) | $49.15 (-$7.29) |
| **Average Rounds** | 61.2 | 61.2 (same) | 61.2 (same) |

**Interpretation:** Forcing features to either safe or risky mean values INCREASES bankruptcy by ~9%, suggesting these features encode context-dependent information that should vary dynamically during gameplay.

### 1.2 Feature Impact Classification

| Category | Count | Percentage | Definition |
|----------|-------|------------|------------|
| **Harmful (Both)** | 215 | 48.8% | Increased bankruptcy in BOTH conditions (Δ > +5%) |
| **Harmful (Safe only)** | 281 | 63.7% | Increased bankruptcy when patched to safe mean |
| **Harmful (Risky only)** | 272 | 61.7% | Increased bankruptcy when patched to risky mean |
| **Protective (Safe)** | 55 | 12.5% | Decreased bankruptcy when patched to safe mean (Δ < -5%) |
| **Protective (Risky)** | 57 | 12.9% | Decreased bankruptcy when patched to risky mean |

**Critical Insight:** The 215 "Harmful Both" features are **critical dynamic features** that require context-dependent activation. Fixing them to ANY constant value breaks decision-making.

### 1.3 Top 10 Most Harmful Features

| Rank | Feature | Layer | Safe Mean Δ Bankruptcy | Risky Mean Δ Bankruptcy | Baseline Bankruptcy |
|------|---------|-------|----------------------|------------------------|-------------------|
| 1 | L30-7254 | 30 | +50.0% | +33.3% | 16.7% |
| 2 | L29-4177 | 29 | +40.0% | +30.0% | 30.0% |
| 3 | L29-22605 | 29 | +36.7% | +43.3% | 30.0% |
| 4 | L29-25917 | 29 | +33.3% | +33.3% | 43.3% |
| 5 | L26-9019 | 26 | +33.3% | +30.0% | 36.7% |
| 6 | L28-8933 | 28 | +33.3% | +23.3% | 30.0% |
| 7 | L28-26395 | 28 | +33.3% | +20.0% | 33.3% |
| 8 | L26-9654 | 26 | +30.0% | +33.3% | 40.0% |
| 9 | L28-5698 | 28 | +30.0% | +23.3% | 30.0% |
| 10 | L28-13766 | 28 | +30.0% | +20.0% | 33.3% |

**Pattern:** Most harmful features are in layers 28-30, suggesting higher layers encode more abstract decision-making concepts that are especially sensitive to mean forcing.

### 1.4 Layer-Wise Analysis

| Layer | Mean Δ Bankruptcy (Safe) | Mean Δ Bankruptcy (Risky) | Feature Count | Interpretation |
|-------|-------------------------|--------------------------|---------------|----------------|
| 25 | +5.8% | +4.2% | 63 | Moderately harmful |
| 26 | **+13.9%** | +9.4% | 79 | **Most harmful layer** |
| 27 | **-3.9%** | +1.0% | 14 | **Only protective layer** |
| 28 | +10.9% | +7.0% | 80 | Highly harmful |
| 29 | +12.4% | +11.2% | 105 | Highly harmful |
| 30 | +9.6% | +10.6% | 89 | Highly harmful |
| 31 | +5.9% | +4.8% | 11 | Moderately harmful |

**Key Insight:** Layer 27's protective effect suggests it may encode more stable/abstract concepts that benefit from consistency, while Layers 26, 28-29 encode highly context-dependent decision factors.

---

## 2. Experiment 6 Analysis: Token-Level Activation Patterns

### 2.1 Scenario Coverage

| Risk Category | Scenarios | Balance Range |
|--------------|-----------|---------------|
| **Bankrupt Edge** | Bankruptcy_90_all_in | $90 |
| **Desperate** | Desperate_10, Very_risky_25 | $10-$25 |
| **Risky** | Risky_40 | $40 |
| **Medium** | Initial_100_first_round, Medium_60 | $60-$100 |
| **Safe** | Safe_130_one_win, Safe_140_near_goal | $130-$140 |
| **Very Safe** | Goal_achieved_200, Big_success_280 | $200-$280 |

### 2.2 Top Features by Token Position

#### Layer 31 (only overlapping layer with Exp 5)

| Token Position | Top Feature | Mean Activation | Interpretation |
|----------------|-------------|-----------------|----------------|
| **Balance** | L31-9926 | 8.80 | Tracks current bankroll state |
| **Goal** | L31-537 | 8.68 | Represents goal-related reasoning |
| **Probability** | L31-9926 | 7.75 | Risk assessment feature |
| **Choice** | L31-9926 | 9.03 | Decision-making activation |

**Pattern:** Feature L31-9926 is consistently active across all token positions, suggesting it's a **general gambling decision feature**.

#### Layer 15 (Mid-layer)

| Token Position | Top Feature | Mean Activation |
|----------------|-------------|-----------------|
| **Balance** | L15-3179 | 15.34 |
| **Goal** | L15-3179 | 14.06 |
| **Probability** | L15-3179 | 16.50 |
| **Choice** | L15-3179 | 17.77 |

**Pattern:** Feature L15-3179 dominates all positions even more strongly than L31-9926, suggesting it's a **core gambling behavior feature** at mid-layer.

#### Layer 8 (Early layer)

| Token Position | Top Feature | Mean Activation |
|----------------|-------------|-----------------|
| **Balance** | L8-2083 | 14.52 |
| **Goal** | L8-2083 | 15.44 |
| **Probability** | L8-2083 | 17.91 |
| **Choice** | L8-2083 | 15.98 |

**Pattern:** Feature L8-2083 shows similar dominance at early layer, with strongest activation at probability token (17.91), suggesting early probability encoding.

---

## 3. Cross-Analysis: Harmful Features in Token Space

### 3.1 Layer 31 Harmful Features (Only Overlapping Layer)

**Note:** Only 11 of the 215 "Harmful Both" features are in Layer 31, which is the only layer present in both Exp 5 and Exp 6.

| Feature | Δ Bankruptcy (Safe) | Dominant Position | Dominant Activation | Interpretation |
|---------|-------------------|-------------------|-------------------|----------------|
| **L31-22745** | +23.3% | Goal | 0.49 | Goal-tracking feature |
| **L31-16925** | +20.0% | Probability | 0.003 | Subtle probability encoding |
| **L31-32310** | +16.7% | Balance | 0.00 | Silent balance tracker |
| **L31-7824** | +16.7% | Balance | 0.68 | Active balance encoder |
| **L31-5891** | +16.7% | Balance | 0.19 | Balance state feature |
| **L31-4059** | +16.7% | Goal | 0.27 | Goal reasoning |
| **L31-15533** | +13.3% | Balance | 0.00 | Silent balance feature |
| **L31-9826** | +13.3% | Goal | 0.26 | Goal-oriented decision |
| **L31-21440** | +13.3% | Goal | 0.33 | Goal representation |
| **L31-14016** | +6.7% | Probability | 0.53 | Risk assessment |

### 3.2 Dominant Token Position Distribution (Harmful Features)

| Position | Count | Percentage | Interpretation |
|----------|-------|------------|----------------|
| **Balance** | 4 | 36.4% | Track current bankroll state |
| **Goal** | 4 | 36.4% | Encode goal-directed reasoning |
| **Probability** | 3 | 27.3% | Represent risk assessment |
| **Choice** | 0 | 0.0% | No choice-dominant harmful features in L31 |

**Insight:** Harmful features are evenly split between balance tracking and goal reasoning, with some risk assessment features. This suggests mean forcing disrupts the **integration of current state with goal-directed planning**.

### 3.3 Protective Features (Layer 31)

| Feature | Δ Bankruptcy (Safe) | Dominant Position | Dominant Activation | Interpretation |
|---------|-------------------|-------------------|-------------------|----------------|
| **L31-12810** | -26.7% | Goal | 0.23 | Protective goal reasoning |
| **L31-14374** | -16.7% | Probability | 0.20 | Safe risk assessment |
| **L31-14913** | -16.7% | Choice | 0.15 | Conservative choice bias |
| **L31-20466** | -16.7% | Goal | 0.30 | Risk-aware goal tracking |
| **L31-28105** | -16.7% | Goal | 0.23 | Balanced goal pursuit |
| **L31-17787** | -10.0% | Choice | 1.76 | **Strong conservative choice** |
| **L31-27952** | -10.0% | Choice | 0.32 | Safe betting preference |
| **L31-27388** | -10.0% | Choice | 1.62 | **Strong safe decision** |

**Key Difference:** Protective features show **stronger choice token activation** (4 out of 8 vs 0 out of 11 for harmful), suggesting they encode more direct "stop gambling" or "bet conservatively" signals.

---

## 4. Mechanistic Interpretation

### 4.1 Why Mean Forcing Increases Bankruptcy

**Hypothesis:** Critical features encode **context-dependent values** that must vary based on:

1. **Current balance** (low → more conservative, high → can afford risk)
2. **Round number** (early → explore, late → exploit/stop)
3. **Recent history** (winning streak → confidence, losing streak → caution)
4. **Distance to goal** (near goal → conservative, far from goal → take risks)

When forced to constant mean values, these features:
- Cannot adapt to changing game state
- Lose their ability to signal "stop now" when balance is low
- Cannot trigger risk-averse behavior when close to bankruptcy
- Break the feedback loop between state observation and decision-making

### 4.2 Layer 27 Protective Effect Hypothesis

Layer 27 is the only layer showing protective effects (-3.9% bankruptcy with safe mean). Possible explanations:

1. **Abstraction level:** L27 may encode more abstract concepts (e.g., "gambling is risky") that benefit from consistency
2. **Feature sparsity:** Only 14 features in L27 were tested, potentially sampling more stable features
3. **Architectural position:** L27 sits between highly context-dependent lower layers (25-26) and higher decision layers (28-31)

### 4.3 Token Position Insights

**Balance Token:**
- Encodes current financial state
- Harmful features activate strongly here (4/11 dominant)
- Mean forcing breaks "stop when broke" signal

**Goal Token:**
- Represents target amount reasoning
- Harmful features also dominant here (4/11)
- Mean forcing disrupts goal-distance calculations

**Probability Token:**
- Encodes risk assessment
- 3/11 harmful features dominant here
- Mean forcing prevents adaptive risk evaluation

**Choice Token:**
- Direct action selection
- Protective features activate strongly here (4/8 dominant)
- Suggests protective features encode explicit "choose conservatively" signals

---

## 5. Implications for Interpretability Research

### 5.1 Challenges Revealed

1. **Linear steering limitations:** Simply setting features to "safe" or "risky" values fails when features encode context-dependent information
2. **Causal complexity:** Features are not simple "more risk" or "less risk" dials but context-sensitive encodings
3. **Emergent dynamics:** Decision-making emerges from dynamic feature interactions, not static feature values

### 5.2 Recommendations for Future Work

1. **Context-aware interventions:** Develop patching methods that vary feature values based on game state
2. **Dynamic range testing:** Test multiple activation levels across different contexts
3. **Temporal analysis:** Track how features change across rounds in the same game
4. **Feature interaction studies:** Investigate how multiple features coordinate to produce decisions
5. **Ablation vs manipulation:** Compare complete feature ablation (zero) vs mean forcing

### 5.3 Methodological Insights

**What worked:**
- Large-scale testing (441 features) revealed systematic patterns
- Token-level tracking identified position-specific activations
- Cross-experiment analysis connected interventional and observational data

**What to improve:**
- Need overlapping layers between experiments (only L31 overlapped)
- Require more scenarios covering broader state space
- Need temporal tracking within single game trajectories

---

## 6. Statistical Summary

### 6.1 Experiment 5 Coverage

- **Features tested:** 441
- **Layers:** 25-31 (7 layers)
- **Conditions:** Baseline, Safe Mean, Bankrupt Mean
- **Trials per condition:** 30
- **Total trials:** 39,690

### 6.2 Experiment 6 Coverage

- **Scenarios:** 10
- **Layers:** 8, 15, 31 (3 layers)
- **Features per layer:** 32,768
- **Token positions tracked:** Balance, Goal, Probability, Choice

### 6.3 Cross-Analysis Results

- **Overlapping layer:** 31 only
- **Harmful features in L31:** 11 (5.1% of 215 total)
- **Protective features in L31:** 8 (14.5% of 55 total)
- **Token activation range:** 0.0 to 17.9

---

## 7. Conclusion

This analysis reveals that **context-dependent encoding** is fundamental to how LLaMA represents gambling decisions. The paradoxical finding that forcing features to constant values (whether "safe" or "risky" means) increases bankruptcy demonstrates that these features are not simple behavioral dials but **dynamic state encoders** that must vary with game context.

The 215 "Harmful Both" features represent **critical dynamic features** whose context-dependent variation is essential for adaptive decision-making. This has important implications for AI safety and interpretability: interventions that ignore context-dependency may have counterintuitive or harmful effects.

Future work should focus on developing **context-aware intervention methods** that respect the dynamic, state-dependent nature of neural representations rather than attempting to force features to static values.

---

## 8. Files Generated

### Analysis Scripts
- `analyze_exp5_feature_effects.py` - Experiment 5 comprehensive analysis
- `analyze_exp6_token_activations.py` - Experiment 6 token-level analysis
- `cross_analysis_exp5_exp6.py` - Cross-experiment feature mapping

### Data Outputs (CSV)
- `exp5_all_features.csv` - All 441 features with full metrics
- `exp5_harmful_both.csv` - 215 harmful features (both conditions)
- `exp5_harmful_safe.csv` - 281 features harmful under safe mean patching
- `exp5_harmful_risky.csv` - 272 features harmful under risky mean patching
- `exp5_protective_safe.csv` - 55 protective features
- `exp5_top20_bankruptcy_safe.csv` - Top 20 by bankruptcy rate change (safe)
- `exp5_top20_bankruptcy_risky.csv` - Top 20 by bankruptcy rate change (risky)
- `exp5_top20_balance_safe.csv` - Top 20 by balance change
- `exp5_layer_stats.csv` - Layer-wise aggregated statistics
- `exp6_L8_balance_specific_features.csv` - L8 balance-specific features
- `exp6_L8_goal_specific_features.csv` - L8 goal-specific features
- `exp6_L8_prob_specific_features.csv` - L8 probability-specific features
- `exp6_L8_choice_specific_features.csv` - L8 choice-specific features
- (Similar files for L15 and L31)
- `cross_analysis_harmful_features.csv` - Harmful features mapped to token activations
- `cross_analysis_protective_features.csv` - Protective features mapped to token activations

### Visualizations
- `heatmap_harmful_features_tokens.png` - Token position heatmap for harmful features
- `scenario_comparison_top_harmful.png` - Activation vs balance for top 4 harmful features

---

**Report Generated:** 2025-10-16
**Analysis Complete** ✅
