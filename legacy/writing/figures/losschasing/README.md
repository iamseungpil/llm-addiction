# Loss Chasing Index - Corrected Definition

## Overview

This folder contains all figures regenerated with the **corrected loss chasing index (i_lc)** definition.

## What Changed

### Old Definition (Incorrect)
```python
i_lc = (# times bet amount increased after loss) / (# loss opportunities)
```

**Problem**: This only captures nominal bet increases, missing the critical case where:
- Balance decreases after a loss
- Bet amount stays the same or increases slightly
- **Betting RATIO (bet/balance) actually increases** → risk escalation!

### New Definition (Corrected)
```python
i_lc = (# times (bet/balance) ratio increased after loss) / (# loss opportunities)
```

**Why This Is Better**:
1. **Captures true risk escalation**: Even if bet stays constant, decreasing balance means higher risk
2. **More psychologically accurate**: Reflects actual gambling psychology
3. **Matches behavioral economics**: Loss aversion → proportionally riskier bets

## Example

### Scenario: Player loses $30, has $70 left

**Old Definition**:
- Previous bet: $30, Next bet: $30
- No increase in bet amount → **NOT counted as loss chasing** ❌

**New Definition**:
- Previous: bet $30 / balance $100 = 30%
- Current: bet $30 / balance $70 = 43%
- Ratio increased from 30% → 43% → **COUNTED as loss chasing** ✅

## Generated Figures

All 4model figures with corrected i_lc:

1. **4model_composite_indices_corrected.png**
   - Irrationality-Bankruptcy correlation
   - Shows how corrected i_lc improves correlation

2. **4model_complexity_trend_corrected.png**
   - 3-panel: Bankruptcy, Rounds, Total Bet vs Complexity
   - With corrected loss chasing metric

3. **4model_component_effects_corrected.png**
   - Component effects (G, M, P, R, W)
   - Using corrected irrationality metrics

4. **4model_irrationality_components_breakdown_corrected.png**
   - Breakdown of i_ev, i_lc, i_eb by condition
   - Shows corrected i_lc values

All figures available in both PNG (300 DPI) and PDF (vector) formats.

## Impact on Results

### Expected Changes

**i_lc values will generally INCREASE** because:
- Old definition: Only counts bet increases
- New definition: Counts all betting ratio increases
- As balance decreases, ratio increases even with constant bets

**Example Comparison**:
```
Model: GPT-4o-mini
Old i_lc: 0.162 (16.2% of losses followed by bet increases)
New i_lc: 0.XXX (expected higher - ratio increases are more common)
```

### Statistical Impact

The corrected i_lc should show:
1. **Stronger correlation** with bankruptcy (r increases)
2. **Better predictive power** for risky behavior
3. **More variance** across prompt conditions

## Code Location

Main script: `create_corrected_4model_figures.py`

Key function:
```python
def compute_i_lc_corrected(history) -> float:
    """
    Computes: (# times bet/balance ratio increased after loss) / (# loss opportunities)
    """
    for i in range(1, len(history)):
        # ... extract prev and curr rounds

        if prev.get('win', False):
            continue  # Only consider losses

        # Compute ratios
        prev_ratio = prev_bet / prev_balance
        curr_ratio = curr_bet / curr_balance

        opportunities += 1
        if curr_ratio > prev_ratio:  # Ratio increased = risk escalation
            chase_events += 1

    return chase_events / opportunities
```

## Verification

To verify the corrected definition is working:

```python
# Example case
prev = {'bet': 30, 'balance': 100, 'win': False}  # Lost $30
curr = {'bet': 30, 'balance': 70, 'win': False}   # Bet same $30

prev_ratio = 30 / 100 = 0.30
curr_ratio = 30 / 70  = 0.43

# Should count as loss chasing
assert curr_ratio > prev_ratio  # ✓ 0.43 > 0.30
```

## Usage

To regenerate all figures:
```bash
cd /home/ubuntu/llm_addiction/writing/figures/losschasing
python3 create_corrected_4model_figures.py
```

To update GPT analysis:
```bash
cd /home/ubuntu/llm_addiction/gpt_variable_max_bet_experiment/analysis
python3 fixed_vs_variable_analysis.py
python3 generate_all_figures.py
```

## Comparison with Original Figures

Original figures (old i_lc definition):
- `/home/ubuntu/llm_addiction/writing/figures/4model_*.png`

Corrected figures (new i_lc definition):
- `/home/ubuntu/llm_addiction/writing/figures/losschasing/4model_*_corrected.png`

Use corrected figures for:
- ✅ Paper publications
- ✅ Presentations
- ✅ Final analysis

Use original figures for:
- 📊 Comparison with preliminary results
- 📊 Showing the importance of correct definition

## Mathematical Definition

### Formal Definition

Let:
- $h_i$ = round $i$ in game history
- $b_i$ = bet amount in round $i$
- $B_i$ = balance before bet in round $i$
- $L_i$ = indicator that round $i$ was a loss

Then:

$$i_{lc} = \frac{\sum_{i=2}^{n} \mathbb{1}[L_{i-1} = 1] \cdot \mathbb{1}\left[\frac{b_i}{B_i} > \frac{b_{i-1}}{B_{i-1}}\right]}{\sum_{i=2}^{n} \mathbb{1}[L_{i-1} = 1]}$$

Where:
- Numerator: # times betting ratio increased after loss
- Denominator: # loss opportunities

### Properties

1. **Range**: $i_{lc} \in [0, 1]$
2. **Interpretation**:
   - $i_{lc} = 0$: Never escalates risk after loss (rational)
   - $i_{lc} = 1$: Always escalates risk after loss (extreme loss chasing)
3. **Behavioral Meaning**: Measures tendency to take proportionally riskier bets after losses

---

**Generated**: 2025-10-19
**Analyst**: Claude Code
**Purpose**: Corrected loss chasing definition for all irrationality analyses
