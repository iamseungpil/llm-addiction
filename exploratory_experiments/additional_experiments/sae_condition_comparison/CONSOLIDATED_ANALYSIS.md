# SAE Condition Comparison: Consolidated Analysis

**Last Updated**: 2026-02-14
**Experiment Path**: `exploratory_experiments/additional_experiments/sae_condition_comparison/`
**Status**: Analysis Complete ✅

---

## Executive Summary

This experiment analyzes how **Variable vs Fixed betting conditions** affect SAE feature activations in LLaMA-3.1-8B and Gemma-2-9B models. Key findings show that betting autonomy is encoded at the neural level, with Variable conditions associated with 2.3-2.6x higher bankruptcy rates.

**Key Results:**
- **11,999 significant features** (LLaMA) differentiate Variable vs Fixed conditions
- **Layer-specific encoding**: LLaMA peaks at L12-15, Gemma at L18-21
- **Effect sizes**: Cohen's d up to 4.75 (extremely large)
- **Statistical validation**: Analysis 1 & 2 reliable, Analysis 3 requires filtering

---

## Research Background

### Motivation

| Model | Fixed Bankruptcy | Variable Bankruptcy | Ratio |
|-------|------------------|---------------------|-------|
| LLaMA | 2.6% (42/1,600) | 6.8% (108/1,600) | **2.6x** |
| Gemma | 12.8% (205/1,600) | 29.1% (465/1,600) | **2.3x** |

**Research Question**: Why does Variable betting lead to 2x higher bankruptcy rates? Can SAE features explain this difference at the neural level?

### Data Summary

- **Total Games**: 3,200 per model (6,400 total)
- **Conditions**: Variable (1,600) vs Fixed (1,600)
- **SAE Features**: LLaMA 32,768/layer (31 layers), Gemma 131,072/layer (42 layers)
- **Analysis Outputs**: ~4GB of results (JSON + visualizations)

---

## Analysis Pipeline

### Analysis 1: Variable vs Fixed (Main Effect)

**Method**: Independent t-test + Cohen's d + FDR correction
**Sample Size**: 1,600 vs 1,600 (highly powered)
**Reliability**: ✅ **HIGH** (sufficient sample, correct statistics)

#### Results Summary

| Metric | LLaMA-3.1-8B | Gemma-2-9B |
|--------|--------------|------------|
| Total Features Analyzed | 24,811 | 37,423 |
| Significant (FDR, \|d\| ≥ 0.3) | **11,999** | 5,755 |
| Higher in Variable | 5,803 (48%) | 3,220 (56%) |
| Higher in Fixed | 6,196 (52%) | 2,535 (44%) |
| Max d (Variable) | 3.34 (L12-26280) | 2.07 (L18-108762) |
| Max d (Fixed) | -4.75 (L14-12265) | -3.67 (L4-43019) |

#### Top Variable-Associated Features

**LLaMA-3.1-8B:**
- L12-26280: d = 3.34 (extremely large effect)
- L18-31208: d = 3.16
- L14-194: d = 2.86

**Gemma-2-9B:**
- L18-108762: d = 2.07
- L19-118800: d = 2.00
- L17-26394: d = 1.98

#### Top Fixed-Associated Features

**LLaMA-3.1-8B:**
- L14-12265: d = -4.75 (strongest discriminative feature)
- L13-32317: d = -4.56
- L15-27263: d = -4.21

**Gemma-2-9B:**
- L4-43019: d = -3.67
- L21-51952: d = -2.17
- L16-19663: d = -2.06

---

### Analysis 2: Four-Way ANOVA (Outcome × Condition)

**Method**: One-way ANOVA across 4 groups + eta-squared
**Groups**: Variable-Bankrupt (VB), Variable-Safe (VS), Fixed-Bankrupt (FB), Fixed-Safe (FS)
**Reliability**: ⚠️ **MEDIUM** (sample imbalance, but FDR corrected)

#### Sample Sizes

**LLaMA:**
- VB: 108 (6.8%)
- VS: 1,492 (93.2%)
- FB: 42 (2.6%) ← **smallest group**
- FS: 1,558 (97.4%)

**Gemma:**
- VB: 465 (29.1%)
- VS: 1,135 (70.9%)
- FB: 205 (12.8%)
- FS: 1,395 (87.2%)

#### Results Summary

| Metric | LLaMA | Gemma |
|--------|-------|-------|
| Features Analyzed | 1,015,808 | 5,505,024 |
| FDR Significant | 21,717 | 26,963 |
| High Effect (η² ≥ 0.01) | 19,253 | 24,177 |
| Max η² | 0.850 | 0.918 |

#### Example: L14-12265 (LLaMA, η² = 0.850)

```
Group Means:
  VB: 0.008, VS: 0.002, FB: 0.217, FS: 0.256
→ Clearly higher in Fixed conditions (especially FB/FS)
```

---

### Analysis 3: Interaction (Bet Type × Outcome)

**Method**: Two-way ANOVA approximation + interaction eta-squared
**Reliability**: ❌ **LOW** (sparse feature artifacts, requires reanalysis)

#### ⚠️ Critical Issue: Sparse Feature Artifacts

**Problem**: 92% of top features have interaction_eta ≈ 1.0 due to extreme sparsity

**Example: L1-3679**
```
Mean activation: 0.000001
Non-zero games: 4 / 3,200 (99.88% zeros)
Group means: All ≈ 0 (one group has 1e-6)
→ interaction_eta = 0.9999 is a numerical artifact, not real effect
```

**Why It's a Problem:**
- Violates ANOVA assumptions (insufficient sample)
- Numerical instability near zero
- Misleading effect sizes

**Solution**: Apply sparsity filtering before analysis
```python
# Exclude features with:
# - Activation rate < 1%
# - Mean activation < 0.001
```

**Status**: ⚠️ Reanalysis with filtering required before using results

---

## Key Findings

### 1. Layer-Specific Encoding

**LLaMA-3.1-8B:**
- Peak differentiation: **L12-L15** (middle layers)
- Variable features: Distributed across L7-L18
- Fixed features: Concentrated in L12-L15

**Gemma-2-9B:**
- Peak differentiation: **L18-L21** (upper-middle layers)
- More distributed pattern across layers
- Early layer (L4) also shows strong Fixed encoding

### 2. Effect Size Differences

- LLaMA: Larger effect sizes (max |d| = 4.75)
- Gemma: Moderate effect sizes (max |d| = 3.67)
- Both: Effect sizes indicate strong neural separation

### 3. Feature Balance

- LLaMA: ~50/50 split (Variable vs Fixed features)
- Gemma: 56% Variable, 44% Fixed
- Interpretation: Both models encode autonomy, but different distributions

---

## Statistical Issues & Solutions

### Issue 1: Sparse Feature Artifacts (CRITICAL ⚠️⚠️)

**Impact**: Analysis 3 (Interaction) results unreliable
**Cause**: Features with <1% activation create numerical artifacts
**Solution**: Filter features before analysis

```python
def filter_sparse_features(features, min_activation_rate=0.01):
    activation_rate = np.count_nonzero(features, axis=0) / len(features)
    return features[:, activation_rate >= min_activation_rate]
```

**Status**: ❌ Not yet applied, reanalysis needed

---

### Issue 2: Sample Imbalance (MEDIUM ⚠️)

**Impact**: Four-way analysis has low power for small groups
**Details**:
- LLaMA Fixed-Bankrupt: n = 42 (too small for stable inference)
- Gemma Fixed-Bankrupt: n = 205 (acceptable but not ideal)

**Mitigation**:
- Analysis 1 unaffected (1,600 vs 1,600 comparison)
- FDR correction applied
- Effect sizes reported alongside p-values

**Recommendation**: Report sample sizes explicitly in paper

---

### Issue 3: Two-Way ANOVA Approximation (LOW ⚠️)

**Impact**: Interaction effects are approximations, not exact
**Details**: Current implementation uses separate one-way ANOVAs for efficiency

**Verification Needed**:
```python
# Compare top 100 features with statsmodels exact ANOVA
import statsmodels.api as sm
from statsmodels.formula.api import ols
```

**Status**: Low priority (approximation generally accurate for main effects)

---

## Reliability Assessment

| Analysis | Reliability | Recommendation |
|----------|-------------|----------------|
| **Analysis 1** (Variable vs Fixed) | ✅ **HIGH** | Use in main paper figures |
| **Analysis 2** (Four-Way) | ⚠️ **MEDIUM** | Use with sample size disclaimer |
| **Analysis 3** (Interaction) | ❌ **LOW** | Reanalyze after filtering, supplementary only |

---

## Usage Guide

### Quick Start

```bash
# 1. Activate environment
conda activate llm-addiction

# 2. Configure paths in configs/analysis_config.yaml
# - feature_dir: Path to SAE NPZ files
# - experiment_file: Path to game results JSON

# 3. Run main analysis
python -m exploratory_experiments.additional_experiments.sae_condition_comparison.src.condition_comparison --model llama

# 4. Visualize results
python scripts/visualize_variable_fixed.py
```

### Output Files

**Results** (`results/`):
- `variable_vs_fixed_*.json` - Analysis 1 (125KB)
- `four_way_*.json` - Analysis 2 (619MB LLaMA, 3.3GB Gemma)
- `interaction_*.json` - Analysis 3 (534MB LLaMA, 2.9GB Gemma)

**Figures** (`results/figures/`):
- `variable_fixed_cohens_d_distribution_*.png`
- `variable_fixed_top_features_heatmap_*.png`
- `four_way_group_means_*.png`

---

## Additional Analyses Completed

### Prompt Component Analysis

**Goal**: Identify which prompt components (G, M, R, W, P) drive outcome differences

**Key Finding**: Goal-setting (G) component shows 3x more significant features than other components

**Output**: `results/prompt_component/` (10 JSON files, 3 figures)

---

### Prompt Complexity Analysis

**Goal**: Test if prompt complexity (0-5 components) affects feature patterns

**Output**: `results/prompt_complexity/` (complexity-level breakdowns)

---

### Individual Combo Analysis

**Goal**: Analyze all 32 prompt combinations separately

**Output**: `results/prompt_combo/` (32 combo-specific analyses)

---

## Future Work

### Immediate Priorities

1. ✅ **Apply sparse feature filtering** to Analysis 3
2. ⚠️ **Verify top 100 features** with exact statsmodels ANOVA
3. 📊 **Add bootstrap confidence intervals** for small sample groups

### Paper Integration

**Main Figures** (ready to use):
- Analysis 1: Cohen's d distribution + top features heatmap
- Analysis 2: Four-way group means (with sample size disclaimer)

**Supplementary** (after filtering):
- Analysis 3: Interaction effects
- Prompt component/complexity analyses

**Limitations Section**:
- Report sample imbalance (Fixed-Bankrupt n=42)
- Note SAE scale differences (LlamaScope vs GemmaScope)
- Explain dead feature exclusion

---

## References

**Key Files**:
- `README.md` - Basic usage guide
- `ANALYSIS_ISSUES_REPORT.md` - Detailed statistical issues (source of Issues section above)
- `EXPERIMENT_SUMMARY.md` - Full experimental details (source of Pipeline section)
- `SAE_Condition_Comparison_Results.md` - Complete numerical results (source of Results section)

**Code**:
- `src/condition_comparison.py` - Main analysis (3 analyses)
- `src/utils.py` - Statistical utilities
- `scripts/visualize_*.py` - Visualization scripts

**Data Locations**:
- SAE features: `/scratch/x3415a02/data/llm-addiction/`
- Game results: `paper_experiments/slot_machine_6models/results/`

---

## Changelog

**2026-02-14**: Consolidated 17 .md files into this single document
**2026-02-02**: Completed all 3 analyses + prompt analyses
**2026-02-01**: Identified sparse feature artifacts in Analysis 3
**2025-XX-XX**: Initial experiment design

---

## Contact

For questions about this analysis, refer to the main project documentation:
- `CLAUDE.md` - Project-wide guidance
- `STRUCTURE.md` - Repository organization
