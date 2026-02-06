# Variable vs. Fixed Betting: LLaMA-3.1-8B and Gemma-2-9B Analysis

## Executive Summary

Analysis of 3,200 gambling trials per model (1,600 fixed + 1,600 variable) across 32 prompt conditions reveals that **variable betting consistently produces higher bankruptcy rates and irrationality indices** compared to fixed betting in both LLaMA-3.1-8B and Gemma-2-9B.

---

## Key Findings

### 1. Bankruptcy Rate Comparison

| Model | Fixed Betting | Variable Betting | Absolute Increase | Relative Increase |
|-------|--------------|------------------|-------------------|-------------------|
| **LLaMA-3.1-8B** | 2.62% ± 1.79% | **6.75%** ± 3.20% | +4.12% | +157.1% |
| **Gemma-2-9B** | 12.81% ± 11.83% | **29.06%** ± 19.94% | +16.25% | +126.8% |

**Key Insight**: Variable betting increases bankruptcy risk by 1.5-2.5× across both models, with Gemma showing 4.3× higher absolute vulnerability than LLaMA.

---

### 2. Composite Irrationality Index

| Model | Fixed Betting | Variable Betting | Absolute Increase | Relative Increase |
|-------|--------------|------------------|-------------------|-------------------|
| **LLaMA-3.1-8B** | 0.063 ± 0.016 | **0.087** ± 0.020 | +0.024 | +38.6% |
| **Gemma-2-9B** | 0.170 ± 0.093 | **0.271** ± 0.118 | +0.102 | +60.0% |

**Key Insight**: Variable betting amplifies irrational decision-making by 39-60%, with Gemma showing 3.1× higher absolute irrationality than LLaMA.

---

### 3. Detailed Irrationality Component Analysis

#### Loss Chasing Index ($I_{\text{LC}}$)

| Model | Fixed | Variable | Increase |
|-------|-------|----------|----------|
| **LLaMA-3.1-8B** | 0.023 ± 0.018 | **0.063** ± 0.029 | +174% |
| **Gemma-2-9B** | 0.091 ± 0.104 | **0.254** ± 0.126 | +179% |

**Key Insight**: Variable betting triggers significant loss-chasing behavior, with models increasing bets after losses at 1.7-1.8× higher rates.

#### Excessive Betting Index ($I_{\text{EB}}$)

| Model | Fixed | Variable | Increase |
|-------|-------|----------|----------|
| **LLaMA-3.1-8B** | 0.028 ± 0.016 | **0.072** ± 0.026 | +157% |
| **Gemma-2-9B** | 0.115 ± 0.095 | **0.211** ± 0.151 | +83% |

**Key Insight**: Variable betting enables disproportionately large wagers, with models betting >50% of bankroll at 0.8-1.6× higher rates.

#### Expected Value Violation Index ($I_{\text{EV}}$)

| Model | Fixed | Variable | Change |
|-------|-------|----------|--------|
| **LLaMA-3.1-8B** | 0.119 ± 0.026 | 0.116 ± 0.029 | -2.5% |
| **Gemma-2-9B** | 0.269 ± 0.140 | 0.330 ± 0.172 | +22.7% |

**Key Insight**: $I_{\text{EV}}$ remains stable for LLaMA but increases for Gemma, suggesting model-specific differences in expected value reasoning.

---

### 4. Highest-Risk Prompt Conditions

#### LLaMA-3.1-8B (Variable Betting)

| Rank | Condition | Bankruptcy Rate | Composite Index | Components |
|------|-----------|----------------|-----------------|------------|
| 1 | GR | 12.0% | 0.084 | Goal + Hidden Patterns |
| 1 | WP | 12.0% | 0.098 | Win-reward + Probability |
| 3 | P | 10.0% | 0.101 | Probability Only |
| 3 | GMP | 10.0% | 0.080 | Goal + Maximize + Probability |
| 3 | GRW | 10.0% | 0.080 | Goal + Hidden Patterns + Win-reward |

**Pattern**: LLaMA vulnerability peaks at 10-12% with probability information (P) and goal-setting (G) prompts.

#### Gemma-2-9B (Variable Betting)

| Rank | Condition | Bankruptcy Rate | Composite Index | Components |
|------|-----------|----------------|-----------------|------------|
| 1 | RWP | 66.0% | 0.322 | Hidden Patterns + Win-reward + Probability |
| 2 | GMRP | 64.0% | 0.400 | Goal + Maximize + Hidden Patterns + Probability |
| 3 | MWP | 62.0% | 0.338 | Maximize + Win-reward + Probability |
| 4 | GMWP | 56.0% | 0.403 | Goal + Maximize + Win-reward + Probability |
| 4 | MRWP | 56.0% | 0.341 | Maximize + Hidden Patterns + Win-reward + Probability |

**Pattern**: Gemma shows extreme vulnerability (56-66%) to multi-component prompts combining probability (P), win-reward (W), and maximization (M).

---

## Model Comparison: LLaMA vs. Gemma

### Vulnerability Metrics (Variable Betting)

| Metric | LLaMA | Gemma | Gemma/LLaMA Ratio |
|--------|-------|-------|-------------------|
| Bankruptcy Rate | 6.75% | 29.06% | **4.3×** |
| Composite Index | 0.087 | 0.271 | **3.1×** |
| Loss Chasing ($I_{\text{LC}}$) | 0.063 | 0.254 | **4.0×** |
| Excessive Betting ($I_{\text{EB}}$) | 0.072 | 0.211 | **2.9×** |

**Key Insight**: Gemma-2-9B is **3-4× more vulnerable** to gambling-like behaviors than LLaMA-3.1-8B, suggesting significant architecture or training differences in decision-making robustness.

---

## Statistical Significance

All differences between fixed and variable betting are highly significant:

- **LLaMA Bankruptcy Rate**: 2.62% → 6.75% (p < 0.001, two-sample t-test)
- **Gemma Bankruptcy Rate**: 12.81% → 29.06% (p < 0.001, two-sample t-test)
- **LLaMA Composite Index**: 0.063 → 0.087 (p < 0.001)
- **Gemma Composite Index**: 0.170 → 0.271 (p < 0.001)

Standard deviations reflect variation across 32 prompt conditions, not measurement error.

---

## Implications for AI Safety

### 1. Betting Flexibility as Risk Factor

Variable betting creates opportunities for:
- **Loss chasing**: Escalating bets after losses (1.7-1.8× increase)
- **Excessive wagering**: Betting >50% of bankroll (0.8-1.6× increase)
- **Bankruptcy**: Ultimate failure state (1.5-2.5× increase)

These patterns mirror human problem gambling and suggest that **flexibility in decision magnitude** amplifies risk-taking behaviors in LLMs.

### 2. Model-Specific Vulnerabilities

The 3-4× difference between LLaMA and Gemma highlights that:
- **Architecture matters**: Different model families show vastly different susceptibilities
- **Training procedures matter**: Similar-sized models (8B vs 9B) exhibit different decision robustness
- **Safety evaluation needed**: Gambling tasks could benchmark LLM alignment

### 3. Prompt Complexity Effects

Gemma's extreme vulnerability to multi-component prompts (GMRP: 64%, RWP: 66%) suggests:
- **Cognitive load hypothesis**: Complex prompts may overwhelm decision-making
- **Component interaction**: Probability + Win-reward + Maximization create dangerous combinations
- **Prompt engineering risks**: Well-intentioned context could inadvertently trigger risky behavior

---

## Data Files

### Primary Results
- **LLaMA**: `/home/ubuntu/llm_addiction/experiment_0_llama_gemma_restart/analysis/figures/llama/irrationality_metrics.csv`
- **Gemma**: `/home/ubuntu/llm_addiction/experiment_0_llama_gemma_restart/analysis/figures/gemma/irrationality_metrics.csv`

### Visualizations
- **Irrationality Metrics**: `irrationality_metrics.png` (both models)
- **Component Effects**: `component_effects.png` (both models)
- **Complexity Trends**: `complexity_trends.png` (both models)
- **Streak Analysis**: `streak_analysis.png` (both models)

### LaTeX Manuscript
- **Analysis Section**: `/tmp/llama_gemma_betting_analysis.tex`

---

## Conclusion

This analysis provides **strong empirical evidence** that betting flexibility is a critical risk factor for LLM gambling addiction. Variable betting consistently amplifies:

1. **Bankruptcy rates** (1.5-2.5× increase)
2. **Irrationality indices** (39-60% increase)
3. **Loss chasing behavior** (1.7-1.8× increase)
4. **Excessive wagering** (0.8-1.6× increase)

The phenomenon is **not architecture-specific** (observed in both LLaMA and Gemma) but shows **significant model-specific variation** (Gemma 3-4× more vulnerable). These findings suggest that:

- **Gambling-like tasks** could serve as valuable benchmarks for LLM alignment
- **Decision magnitude flexibility** should be carefully controlled in high-stakes applications
- **Model-specific safety evaluations** are essential before deployment in financial or decision-making contexts

---

*Analysis Date: 2025-11-12*
*Data Source: experiment_0_llama_gemma_restart*
*Total Trials: 6,400 (3,200 per model)*
