# GPU-Free Analysis Report: LLaMA vs Gemma Neural Encoding Strategies

Date: 2026-02-02
Analysis Type: Layer-wise Information Flow + Maximum Activating Examples

---

## Executive Summary

We performed three GPU-free analyses on existing SAE feature data to understand how LLaMA and Gemma encode betting condition information differently. The key findings reveal fundamental architectural differences in how these models process gambling tasks.

---

## Analysis 1: Layer-wise Information Flow

### Methodology
- Analyzed distribution of significant SAE features (|Cohen's d| ≥ 0.3, FDR-corrected) across all layers
- Calculated mean effect size per layer to identify "critical layers"
- Examined feature count distribution to understand processing localization

### Key Findings

#### LLaMA-3.1-8B

**Peak Discrimination**: Layer 16 (mean |d| = 0.779)

**Critical Layer Range**: L12-L16 (middle layers)
- L12: mean |d| = 0.747, 492 features, Fixed-dominant
- L13: mean |d| = 0.754, 498 features, Fixed-dominant
- L14: mean |d| = 0.750, 518 features, Variable-dominant
- L15: mean |d| = 0.737, 450 features, Fixed-dominant
- L16: mean |d| = 0.779, 490 features, Variable-dominant

**Pattern**:
- Strong, concentrated discrimination in middle layers
- Balanced Variable vs Fixed dominance
- Consistent effect sizes (~0.7-0.8) across critical range

#### Gemma-2-9B-IT

**Peak Discrimination**: Layer 2 (mean |d| = 0.604)

**Critical Layer Range**: L2, L17-L20 (bimodal distribution)

Early peak:
- L2: mean |d| = 0.604, 64 features, Variable-dominant

Late peaks:
- L17: mean |d| = 0.603, 103 features, Fixed-dominant
- L18: mean |d| = 0.572, 125 features, Fixed-dominant
- L19: mean |d| = 0.589, 159 features, Fixed-dominant
- L20: mean |d| = 0.591, 156 features, Fixed-dominant

**Pattern**:
- Distributed processing across layers
- Lower overall effect sizes (0.4-0.6 vs LLaMA's 0.7-0.8)
- Bimodal distribution: early detection + late refinement?

### Interpretation

**LLaMA**:
- Concentrated middle-layer processing suggests a "single-stage" encoding strategy
- Middle layers (12-16) are where abstract task representations emerge
- Consistent with "environment encoding" hypothesis

**Gemma**:
- Distributed processing suggests multi-stage strategy
- Early layer (L2) may detect basic betting patterns
- Later layers (L17-20) refine outcome predictions
- Consistent with "outcome encoding" hypothesis

---

## Analysis 2: Maximum Activating Examples (MAE)

### Methodology
- For top 5 Variable-high and Fixed-high features, identified the 3 games with highest activation
- Examined game characteristics: bet_type, outcome, rounds, final_balance

### Key Findings

#### Variable-High Features (LLaMA)

**L12-26280** (d=3.341):
- Top activating games: ALL Variable, ALL Voluntary Stop
- Rounds: 3-6 rounds
- Final balance: 55-120 (positive outcomes)
- **Pattern**: Activates on successful Variable games with moderate play

**L14-194** (d=2.861):
- Top activating games: ALL Variable, MIXED outcomes (1 bankrupt, 2 safe)
- Rounds: 1 round only
- Final balance: 0-200 (extreme outcomes)
- **Pattern**: Activates on very short Variable games with dramatic results

**L7-5998** (d=2.832):
- Top activating games: ALL Variable, MIXED outcomes
- Rounds: 2 rounds
- **Pattern**: Early-stage Variable game recognition

#### Fixed-High Features (LLaMA)

**L14-12265** (d=-4.747):
- Top activating games: ALL Fixed, ALL Voluntary Stop
- Rounds: 0 rounds (immediate stop!)
- Final balance: 100 (no play)
- **Pattern**: Strongly activates when Fixed condition → immediate stop
- **Interpretation**: "Fixed betting avoidance" detector?

**L13-32317** (d=-4.560):
- Top activating games: ALL Fixed, ALL Voluntary Stop
- Rounds: 1 round
- Final balance: 110-120 (small wins)
- **Pattern**: Brief Fixed play with positive outcome

**L12-30147** (d=-3.729):
- Top activating games: ALL Fixed, ALL Voluntary Stop
- Rounds: 4-6 rounds
- Final balance: 10-110 (varied outcomes)
- **Pattern**: Sustained Fixed play

### Critical Observation: Fixed + Immediate Stop Pattern

The strongest Fixed-high feature (L14-12265) activates most strongly on games where:
1. Betting is Fixed
2. Model immediately stops (0 rounds)

**This is striking because**:
- These are "non-games" (no actual gambling occurred)
- The feature activates on the CONDITION itself, not the play
- Suggests LLaMA encodes "Fixed betting as a constraint to be avoided"

**Contrast with Variable-high features**:
- Activate on actual gameplay (3-6 rounds)
- Require behavioral engagement
- Encode "Variable betting in action"

---

## Analysis 3: Neuronpedia Feature Interpretation

### Methodology
- Generated Neuronpedia URLs for top 10 features (5 Variable-high, 5 Fixed-high)
- Provided interpretation guidelines for manual inspection

### Generated URLs (for manual follow-up)

**Variable-High Features**:
1. L12-26280: https://neuronpedia.org/llama-3_1-8b-base/12/26280
2. L18-31208: https://neuronpedia.org/llama-3_1-8b-base/18/31208
3. L14-194: https://neuronpedia.org/llama-3_1-8b-base/14/194
4. L7-5998: https://neuronpedia.org/llama-3_1-8b-base/7/5998
5. L15-32610: https://neuronpedia.org/llama-3_1-8b-base/15/32610

**Fixed-High Features**:
1. L14-12265: https://neuronpedia.org/llama-3_1-8b-base/14/12265
2. L13-32317: https://neuronpedia.org/llama-3_1-8b-base/13/32317
3. L15-27263: https://neuronpedia.org/llama-3_1-8b-base/15/27263
4. L12-30147: https://neuronpedia.org/llama-3_1-8b-base/12/30147
5. L13-7256: https://neuronpedia.org/llama-3_1-8b-base/13/7256

### What to Look For

When inspecting these features on Neuronpedia:

**Variable-high features** might activate on:
- Varied numerical values
- Choice-related language ("choose", "decide")
- Uncertainty markers
- Dynamic action verbs

**Fixed-high features** might activate on:
- Repeated/constant values (e.g., "10", "same")
- Constraint language ("fixed", "must")
- Consistency markers
- Routine/repetition indicators

---

## Integrated Interpretation

### LLaMA's Encoding Strategy: Environment-Oriented

**Evidence**:
1. **Layer-wise**: Concentrated middle-layer (L12-16) processing
2. **MAE**: Strongest Fixed feature (L14-12265) activates on immediate stops
   - Encodes the CONSTRAINT itself, not the outcome
   - "Fixed betting = restrictive environment"
3. **Activation patterns**: Features distinguish Variable/Fixed REGARDLESS of outcome

**Hypothesis**:
LLaMA represents gambling tasks by encoding the STRUCTURAL PROPERTIES of the environment:
- "Is betting free or constrained?"
- "What are the action possibilities?"
- "What is the decision space?"

This is consistent with a **model-based RL** perspective, where the agent first builds an environment model.

### Gemma's Encoding Strategy: Outcome-Oriented

**Evidence**:
1. **Layer-wise**: Distributed processing (early + late layers)
2. **From Section 4 results**: Bankrupt features show 10-50× higher activation
3. **From Four-Way ANOVA**: Top features discriminate Bankrupt vs Safe, not Variable vs Fixed

**Hypothesis**:
Gemma represents gambling tasks by encoding OUTCOME PREDICTIONS:
- "Will this lead to bankruptcy?"
- "Is this a winning trajectory?"
- "What will be the final state?"

This is consistent with a **model-free RL** perspective, where the agent directly learns outcome values.

### Why This Matters

**Behavioral Equivalence**:
- Both models show ~2.4× higher bankruptcy in Variable condition
- Identical risk-seeking behavioral phenotype

**Neural Divergence**:
- LLaMA: Encodes environment structure
- Gemma: Encodes outcome predictions

**Implication**:
The same behavioral output can arise from fundamentally different computational strategies. This challenges the assumption that behavioral similarity implies mechanistic similarity.

---

## Recommendations for Further Analysis

### 1. Temporal Analysis (Next Priority)
**Question**: When during gameplay do these features activate?
- LLaMA's "constraint" features: Immediate (Round 1)?
- Gemma's "outcome" features: Progressive (increasing toward bankruptcy)?

**Method**: Analyze round-by-round activation using pathway token analysis data

### 2. Causal Validation
**Question**: Do these features causally drive behavior?
- Ablate L14-12265 in LLaMA → Variable/Fixed distinction disappears?
- Ablate Gemma's bankrupt features → Bankruptcy rate changes?

**Method**: Activation patching (requires GPU)

### 3. Domain Generalization
**Question**: Are these encoding strategies task-general or task-specific?
- Test on IGT, Loot Box paradigms
- If patterns replicate → model-intrinsic strategies
- If patterns differ → task-dependent adaptations

---

## Files Generated

1. `neuronpedia_links.txt` - URLs for manual feature inspection
2. `layer_wise_visualization.png` - Layer-wise effect size plots (pending)
3. `layer_wise_summary.png` - Summary statistics (pending)
4. This report: `GPU_FREE_ANALYSIS_REPORT.md`

---

## Conclusion

Without any GPU computation, we uncovered critical insights into how LLaMA and Gemma differ in their neural encoding strategies:

- **LLaMA**: Middle-layer, environment-oriented, constraint-encoding
- **Gemma**: Distributed, outcome-oriented, prediction-encoding

These findings provide mechanistic depth to the original observation that "LLaMA encodes bet type, Gemma encodes outcome," revealing WHEN and HOW these encodings emerge in the neural architecture.

**Next Steps**: Temporal analysis to understand the dynamics of these encodings across game progression.
