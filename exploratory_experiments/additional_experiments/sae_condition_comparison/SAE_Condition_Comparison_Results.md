# SAE Condition Comparison Analysis Results

## Overview

This document presents the results of SAE (Sparse Autoencoder) feature analysis comparing Variable vs Fixed betting conditions across two open-weight LLMs: **LLaMA-3.1-8B** and **Gemma-2-9B-IT**. The analysis identifies neural features that differentiate between betting conditions and game outcomes.

---

## Data Summary

| Metric | LLaMA-3.1-8B | Gemma-2-9B-IT |
|--------|--------------|---------------|
| Total Games | 3,200 | 3,200 |
| Variable Games | 1,600 | 1,600 |
| Fixed Games | 1,600 | 1,600 |
| Variable Bankrupt | 108 (6.8%) | 465 (29.1%) |
| Fixed Bankrupt | 42 (2.6%) | 205 (12.8%) |
| **Bankruptcy Ratio (V/F)** | **2.57x** | **2.27x** |

**Key Observation**: Both models show significantly higher bankruptcy rates under Variable betting conditions, with the ratio (Variable/Fixed) being similar (~2.3-2.6x) despite Gemma having much higher overall bankruptcy rates.

---

## Analysis 1: Variable vs Fixed (t-test with Cohen's d)

This analysis directly compares SAE feature activations between Variable and Fixed conditions using independent t-tests with FDR correction.

### Summary Statistics

| Metric | LLaMA-3.1-8B | Gemma-2-9B-IT |
|--------|--------------|---------------|
| Total Features Analyzed | 24,811 | 37,423 |
| FDR Significant (\|d\| >= 0.3) | 11,999 | 5,755 |
| Higher in Variable | 5,803 | 3,220 |
| Higher in Fixed | 6,196 | 2,535 |
| Max d (Variable) | 3.34 | 2.07 |
| Max d (Fixed) | -4.75 | -3.67 |

### Top Features Higher in Variable Condition

**LLaMA-3.1-8B:**
| Feature | Cohen's d | p_fdr |
|---------|-----------|-------|
| L12-26280 | 3.341 | 0.00 |
| L18-31208 | 3.163 | 0.00 |
| L14-194 | 2.861 | 0.00 |
| L7-5998 | 2.832 | 0.00 |
| L15-32610 | 2.751 | 0.00 |

**Gemma-2-9B-IT:**
| Feature | Cohen's d | p_fdr |
|---------|-----------|-------|
| L18-108762 | 2.069 | 0.00 |
| L19-118800 | 1.997 | 0.00 |
| L17-26394 | 1.980 | 0.00 |
| L19-23310 | 1.870 | 0.00 |
| L2-74687 | 1.790 | 0.00 |

### Top Features Higher in Fixed Condition

**LLaMA-3.1-8B:**
| Feature | Cohen's d | p_fdr |
|---------|-----------|-------|
| L14-12265 | -4.747 | 0.00 |
| L13-32317 | -4.560 | 0.00 |
| L15-27263 | -4.205 | 0.00 |
| L12-30147 | -3.729 | 0.00 |
| L13-7256 | -3.579 | 0.00 |

**Gemma-2-9B-IT:**
| Feature | Cohen's d | p_fdr |
|---------|-----------|-------|
| L4-43019 | -3.670 | 0.00 |
| L21-51952 | -2.173 | 0.00 |
| L16-19663 | -2.061 | 0.00 |
| L17-36479 | -1.794 | 0.00 |
| L20-58461 | -1.612 | 0.00 |

### Key Findings - Analysis 1

1. **Layer Distribution**: LLaMA shows peak differentiation in middle layers (L12-L15), while Gemma shows more distributed effects (L2, L4, L16-L21).

2. **Effect Sizes**: LLaMA exhibits larger effect sizes overall (max |d| = 4.75) compared to Gemma (max |d| = 3.67).

3. **Ratio Balance**: LLaMA has roughly balanced features favoring each condition (48% Variable vs 52% Fixed), while Gemma shows bias toward Variable-associated features (56% Variable vs 44% Fixed).

---

## Analysis 2: Four-Way Comparison (ANOVA with eta-squared)

This analysis compares all four conditions: Variable-Bankrupt (VB), Variable-Safe (VS), Fixed-Bankrupt (FB), and Fixed-Safe (FS) using one-way ANOVA.

### Summary Statistics

| Metric | LLaMA-3.1-8B | Gemma-2-9B-IT |
|--------|--------------|---------------|
| Total Features Analyzed | 1,015,808 | 5,505,024 |
| FDR Significant | 21,717 | 26,963 |
| Significant (eta^2 >= 0.01) | 19,253 | 24,177 |
| Max eta-squared | 0.850 | 0.918 |

### Top Features by Effect Size (eta^2)

**LLaMA-3.1-8B:**
| Feature | eta^2 | VB | VS | FB | FS |
|---------|-------|-----|-----|-----|-----|
| L14-12265 | 0.850 | 0.008 | 0.002 | 0.217 | 0.256 |
| L13-32317 | 0.840 | 0.178 | 0.160 | 0.428 | 0.454 |
| L15-27263 | 0.821 | 0.175 | 0.098 | 0.443 | 0.439 |
| L13-7256 | 0.779 | 0.676 | 0.758 | 0.906 | 1.033 |
| L12-30147 | 0.778 | 0.583 | 0.598 | 0.837 | 0.882 |

**Gemma-2-9B-IT:**
| Feature | eta^2 | VB | VS | FB | FS |
|---------|-------|-----|-----|-----|-----|
| L26-33483 | 0.918 | 20.84 | 0.37 | 19.16 | 0.28 |
| L40-108098 | 0.912 | 33.03 | 0.41 | 28.33 | 0.64 |
| L40-75697 | 0.900 | 22.26 | 0.75 | 18.70 | 0.29 |
| L38-26402 | 0.894 | 17.32 | 0.09 | 12.64 | 0.06 |
| L27-44495 | 0.893 | 8.25 | 0.12 | 7.75 | 0.09 |

### Key Findings - Analysis 2

1. **Outcome Dominance in Gemma**: Gemma's top features show **Outcome (Bankrupt vs Safe)** as the dominant factor. The activation patterns show:
   - Bankrupt conditions (VB, FB): High activation (10-30)
   - Safe conditions (VS, FS): Low activation (0.1-0.8)
   - This pattern suggests these features encode "bankruptcy-related" processing.

2. **Bet Type Dominance in LLaMA**: LLaMA's top features show **Bet Type (Variable vs Fixed)** as the dominant factor:
   - Fixed conditions (FB, FS): Higher activation (~0.2-1.0)
   - Variable conditions (VB, VS): Lower activation (~0.002-0.2)
   - This pattern suggests these features encode "betting constraint" processing.

3. **Layer Distribution**:
   - **LLaMA**: Concentrated in middle layers (L12-L15)
   - **Gemma**: Concentrated in later layers (L26-L40)

4. **Activation Scale**: Gemma features show much higher raw activation values (0-33) compared to LLaMA (0-1), reflecting different SAE training configurations.

---

## Analysis 3: Interaction Effects (bet_type x outcome)

This analysis uses 2x2 factorial ANOVA to identify features where the effect of bet type depends on outcome (and vice versa).

### Summary Statistics

| Metric | LLaMA-3.1-8B | Gemma-2-9B-IT |
|--------|--------------|---------------|
| Total Features Analyzed | 1,015,808 | 5,505,024 |
| Significant Interactions (eta >= 0.01) | 2,616 | 11,948 |
| Max Interaction eta | ~1.00 | ~1.00 |

### Top Interaction Features

**LLaMA-3.1-8B:**
| Feature | int_eta | VB | VS | FB | FS |
|---------|---------|-----|-----|-----|-----|
| (See interaction_llama_*.json for details) |

**Gemma-2-9B-IT:**
| Feature | int_eta | VB | VS | FB | FS |
|---------|---------|-----|-----|-----|-----|
| L35-38308 | 1.000 | 0.00 | 0.05 | 0.00 | 0.01 |
| L28-65755 | 1.000 | 0.00 | 0.07 | 0.00 | 0.01 |
| L33-46758 | 1.000 | 0.00 | 0.01 | 0.00 | 0.32 |
| L36-119630 | 1.000 | 0.02 | 0.03 | 0.00 | 0.00 |
| L35-73212 | 1.000 | 0.00 | 0.00 | 0.05 | 0.03 |

### Key Findings - Analysis 3

1. **Interaction Prevalence**: Gemma shows 4.6x more significant interaction features than LLaMA (11,948 vs 2,616), proportional to its larger feature space.

2. **Interaction Pattern**: The top Gemma interaction features show activation primarily in VS (Variable-Safe) or FS (Fixed-Safe) conditions, with near-zero activation in Bankrupt conditions. This suggests these features encode "safe play patterns" that differ by betting condition.

3. **Layer Distribution**: Gemma interaction features cluster in layers L25-L36, consistent with the later-layer processing observed in Analysis 2.

The presence of significant interactions indicates that the effect of bet type depends on outcome (and vice versa) for certain features, revealing complex condition-dependent processing.

---

## Cross-Model Comparison

### Similarities

1. **Consistent Behavioral Pattern**: Both models show ~2.3-2.6x higher bankruptcy under Variable betting.

2. **Significant Neural Differentiation**: Both models have thousands of features that significantly differentiate between conditions.

3. **Middle-to-Late Layer Processing**: Both models show important features in the middle-to-late layers where higher-level decision processing occurs.

### Differences

1. **Dominant Factor**:
   - LLaMA: Bet Type (Variable vs Fixed) dominates
   - Gemma: Outcome (Bankrupt vs Safe) dominates

2. **Effect Size Distribution**:
   - LLaMA: Larger per-feature effect sizes, more balanced distribution
   - Gemma: Larger maximum eta^2 values, more features overall

3. **Layer Localization**:
   - LLaMA: Middle layers (L12-L15)
   - Gemma: Later layers (L26-L40)

4. **Bankruptcy Rates**:
   - LLaMA: Low overall (2.6% Fixed, 6.8% Variable)
   - Gemma: High overall (12.8% Fixed, 29.1% Variable)

5. **Interaction Effects**:
   - LLaMA: 2,616 significant interactions
   - Gemma: 11,948 significant interactions (4.6x more)

---

## Implications for Paper Results Section 2

### Proposed Narrative

The SAE condition comparison analysis reveals that **both LLaMA and Gemma models develop distinct neural representations** that differentiate between betting conditions, but the nature of these representations differs:

1. **LLaMA** appears to encode the **betting constraint** more strongly, with features that differentiate Variable from Fixed conditions regardless of outcome. This suggests the model processes the "freedom to choose bet amounts" as a salient feature.

2. **Gemma** appears to encode the **outcome** more strongly, with features that differentiate Bankrupt from Safe games regardless of betting condition. This suggests the model processes the "game result" as the primary organizing principle.

### Connecting to Behavioral Results

These neural findings help explain the behavioral observation that Variable betting leads to higher bankruptcy:

- In **LLaMA**, distinct "Variable-betting" features may enable risk-seeking decision patterns that are absent under Fixed constraints.

- In **Gemma**, the strong bankruptcy-encoding features may represent accumulated risk signals that predict (or contribute to) eventual bankruptcy.

### Figure Recommendations

1. **Figure: Four-Way Heatmap** - Shows top features' activation patterns across all four conditions
2. **Figure: Layer Effect Size Distribution** - Shows which layers contain the most discriminative features
3. **Figure: Bet Type vs Outcome Effect Scatter** - Demonstrates which factor dominates for each feature

---

## Technical Notes

- **FDR Correction**: Benjamini-Hochberg method, alpha = 0.05
- **Effect Size Thresholds**: |Cohen's d| >= 0.3, eta^2 >= 0.01
- **SAE Models**: LlamaScope (32,768 features/layer), GemmaScope (131,072 features/layer)
- **Analysis Code**: `sae_condition_comparison/src/condition_comparison.py`

---

## Files Generated

```
results/
├── LLaMA Analysis
│   ├── condition_comparison_summary_llama_20260127_150824.json
│   ├── variable_vs_fixed_llama_20260127_150824.json
│   ├── four_way_llama_20260127_150824.json
│   └── interaction_llama_20260127_150824.json
├── Gemma Analysis
│   ├── condition_comparison_summary_gemma_20260127_203518.json
│   ├── variable_vs_fixed_gemma_20260127_203709.json
│   ├── four_way_gemma_20260127_203709.json
│   └── interaction_gemma_20260127_203518.json
├── LLaMA Figures
│   ├── fig1_four_way_heatmap.png
│   ├── fig2_layer_effect_size.png
│   ├── fig3_bet_vs_outcome_scatter.png
│   └── fig4_top_features_bar.png
└── Gemma Figures
    ├── fig1_four_way_heatmap_gemma.png
    ├── fig2_layer_effect_size_gemma.png
    ├── fig3_bet_vs_outcome_scatter_gemma.png
    └── fig4_top_features_bar_gemma.png
```
