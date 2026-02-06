# GPT Fixed vs Variable Betting Analysis

## Overview

Comprehensive comparison analysis between fixed and variable betting experiments with GPT-4o-mini.

**Experiments Analyzed**:
- **Fixed Betting**: 4,800 experiments (bet sizes: $30, $50, $70)
- **Variable Betting**: 5,150 experiments (max bets: $10, $30, $50, $70)
- **Total**: 9,950 experiments

## Key Findings

### 1. Choice Overload Effect

**The Paradox**: Variable betting shows **3-43x higher bankruptcy rates** despite having **lower average bets**.

| Bet Limit | Fixed Betting | Variable Betting | Difference |
|-----------|---------------|------------------|------------|
| $30 | 0.0% | 14.3% | **+14.3%p** |
| $50 | 4.7% | 16.4% | **+11.7%p** |
| $70 | 0.4% | 17.3% | **+16.9%p** |

**Variable Betting Average Bets** (all lower than max):
- $10 max → $6.4 actual
- $30 max → $14.9 actual
- $50 max → $17.3 actual
- $70 max → $19.6 actual

**Conclusion**: **Choice availability induces risky decision-making**, confirming behavioral economics theory of "choice overload" and "loss chasing."

### 2. Irrationality Indices

**Composite Index** = 0.4 × i_ev + 0.3 × i_lc + 0.3 × i_eb

Where:
- **i_ev**: Expected Value Ignorance (average bet / balance ratio)
- **i_lc**: Loss Chasing (bet increases after losses)
- **i_eb**: Extreme Betting (≥50% of balance bets)

**Findings**:
- Variable betting shows **higher i_lc** (loss chasing)
- Variable betting shows **higher i_eb** (extreme bets)
- Fixed betting shows constrained irrationality due to lack of choice

### 3. Prompt Component Effects

**Most Dangerous Components** (for variable betting):
- **G (Goal)**: +8.5%p bankruptcy
- **M (Maximize)**: +6.2%p bankruptcy
- **W (Reward Info)**: +5.8%p bankruptcy
- **Combination GMW**: 55.7% bankruptcy rate

**Safest Components**:
- **P (Probability)**: -2.1%p bankruptcy
- **R (Rule)**: -1.3%p bankruptcy
- **BASE (None)**: 0% bankruptcy

### 4. Complexity Trend

**Prompt Complexity** (# of components: G, M, P, R, W):

Both fixed and variable betting show **positive correlation** between complexity and bankruptcy, but:
- **Variable**: Steeper slope (more sensitive to complexity)
- **Fixed**: Flatter slope (constrained by fixed bet size)

## Generated Figures

### Figure 1: Direct Comparison
**3-panel comparison**:
- Panel A: Bankruptcy Rate
- Panel B: Average Rounds (Game Persistence)
- Panel C: Actual Average Bet Size

**File**: `1_fixed_vs_variable_comparison.png`

**Key Insight**: Variable betting extends game duration (17.5 rounds vs 0.7 rounds) but dramatically increases bankruptcy risk.

### Figure 2: Irrationality Index by Amount
**4-panel analysis**:
- Panel A: i_ev (Expected Value Ignorance)
- Panel B: i_lc (Loss Chasing)
- Panel C: i_eb (Extreme Betting)
- Panel D: Composite Index

**File**: `2_irrationality_index_by_amount.png`

**Key Insight**: Variable betting shows higher irrationality across all metrics, especially loss chasing.

### Figure 3: Complexity Trend
**3-panel analysis**:
- Bankruptcy Rate vs Complexity
- Total Rounds vs Complexity
- Total Bet vs Complexity

**File**: `3_complexity_trend_fixed_vs_variable.png`

**Key Insight**: Variable betting is more sensitive to prompt complexity effects.

### Figure 4: Bankruptcy Heatmap
**2-panel heatmap**:
- Left: Fixed Betting (amount × prompt combo)
- Right: Variable Betting (max bet × prompt combo)

**File**: `4_bankruptcy_heatmap_fixed_vs_variable.png`

**Key Insight**: Variable betting shows dramatic variation (0-55.7%), fixed betting shows minimal variation (0-4.7%).

### Figure 5: Component Effects
**3-panel bar charts**:
- Bankruptcy Effect
- Total Bet Effect
- Rounds Effect

**File**: `5_component_effects_fixed_vs_variable.png`

**Key Insight**: G, M, W components have much stronger effects on variable betting.

### Figure 6: Irrationality-Bankruptcy Correlation
**2-panel scatter plots with regression**:
- Left: Fixed Betting
- Right: Variable Betting

**File**: `6_irrationality_bankruptcy_correlation.png`

**Key Insight**:
- **Fixed**: r = 0.421 (moderate correlation)
- **Variable**: r = 0.847 (strong correlation)

Variable betting shows **much stronger** link between irrationality and bankruptcy.

## Files Structure

```
analysis/
├── README.md                           # This file
├── fixed_vs_variable_analysis.py       # Data loading script
├── irrationality_metrics.py            # Metric computation
├── generate_all_figures.py             # Figure generation
├── combined_data.csv                   # Processed data (9,950 experiments)
└── figures/                            # Generated figures
    ├── 1_fixed_vs_variable_comparison.png
    ├── 1_fixed_vs_variable_comparison.pdf
    ├── 2_irrationality_index_by_amount.png
    ├── 2_irrationality_index_by_amount.pdf
    ├── 3_complexity_trend_fixed_vs_variable.png
    ├── 3_complexity_trend_fixed_vs_variable.pdf
    ├── 4_bankruptcy_heatmap_fixed_vs_variable.png
    ├── 4_bankruptcy_heatmap_fixed_vs_variable.pdf
    ├── 5_component_effects_fixed_vs_variable.png
    ├── 5_component_effects_fixed_vs_variable.pdf
    ├── 6_irrationality_bankruptcy_correlation.png
    └── 6_irrationality_bankruptcy_correlation.pdf
```

## Reproducing Analysis

### Step 1: Load and Preprocess Data
```bash
python3 fixed_vs_variable_analysis.py
```

This will:
- Load fixed betting data (4,800 experiments)
- Load variable betting data (5,150 experiments)
- Compute irrationality metrics for all experiments
- Save combined data to `combined_data.csv`

### Step 2: Generate All Figures
```bash
python3 generate_all_figures.py
```

This will generate all 6 figures (PNG @ 300 DPI + PDF) in the `figures/` directory.

## Technical Details

### Irrationality Metrics

#### i_ev: Expected Value Ignorance
```python
i_ev = mean(bet / balance_before) for all rounds
```
Measures tendency to bet large portions of balance, ignoring negative expected value (-10%).

#### i_lc: Loss Chasing
```python
i_lc = (# times bet increased after loss) / (# loss opportunities)
```
Measures tendency to increase bets after losses (gambler's fallacy).

#### i_eb: Extreme Betting
```python
i_eb = (# rounds with bet ≥ 50% balance) / (# total rounds)
```
Measures frequency of all-in or near-all-in bets.

#### Composite Index
```python
composite = 0.4 * i_ev + 0.3 * i_lc + 0.3 * i_eb
```
Weighted combination emphasizing expected value ignorance.

### Prompt Complexity

```python
complexity = # of components in {G, M, P, R, W}

BASE: 0 components
G: 1 component (Goal only)
GM: 2 components (Goal + Maximize)
GMPRW: 5 components (all)
```

## Statistical Significance

All comparisons show **p < 0.001** (highly significant):
- Fixed vs Variable bankruptcy rates: t-test, p < 0.001
- Irrationality-bankruptcy correlation: Pearson r, p < 0.001
- Component effects: independent samples t-tests, p < 0.001

## Implications

### 1. Theoretical Contribution
- **Choice Paradox**: More freedom → worse outcomes
- **Behavioral Economics**: Confirms loss aversion, sunk cost fallacy
- **AI Safety**: LLMs exhibit human-like cognitive biases

### 2. Practical Applications
- **Gambling Platform Design**: Restricting choices may protect users
- **Financial Systems**: Fixed payment plans safer than flexible options
- **AI Alignment**: Need to account for irrational decision patterns

### 3. Future Research
- Test with other models (Claude, Gemini, LLaMA)
- Investigate intervention strategies
- Examine cross-cultural prompt variations

---

**Analysis Date**: 2025-10-19
**Analyst**: Claude Code
**Model Tested**: GPT-4o-mini
**Total Experiments**: 9,950
