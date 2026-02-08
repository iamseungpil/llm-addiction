# Alternative Paradigms - Data Summary

**Generated**: 2026-02-09
**Total Games Analyzed**: 1,460
**Paradigms**: Lootbox, Blackjack, Investment Choice
**Models**: LLaMA-3.1-8B, Gemma-2-9B

---

## Quick Statistics

### Overall Performance

| Paradigm | Model | Games | Avg Rounds | Max Rounds | Bankrupt % | Vol Stop % | Refusal % |
|----------|-------|-------|------------|------------|------------|------------|-----------|
| **Lootbox** | LLaMA | 320 | 1.91 | 11 | 0.0% | 100.0% | 36.2% |
| **Lootbox** | Gemma | 200 | 1.09 | 5 | 1.5% | 98.5% | 43.0% |
| **Blackjack** | LLaMA (v1) | 320 | 1.15 | 9 | 0.3% | 99.7% | 47.8% |
| **Blackjack** | LLaMA (v2) | 300 | 1.41 | 18 | 5.3% | 94.7% | 48.3% |
| **Investment** | LLaMA | 160 | 1.55 | 9 | 31.2% | 68.8% | 0.0% |
| **Investment** | Gemma | 160 | 2.42 | 10 | 25.0% | 75.0% | 0.0% |

**Notes**:
- Refusal % = Games with 0 rounds played
- Investment has no refusals but high Round 1 stops (47-68%)
- v1/v2 = Different experimental runs (v2 has slightly better engagement)

---

## Detailed Round Distribution

### Lootbox LLaMA (320 games)
```
Rounds      Count    Percentage
0 (refusal)   116      36.2%  ████████████████████████████████████
1              75      23.4%  ███████████████████████
2-5           101      31.6%  ███████████████████████████████
6-10           26       8.1%  ████████
11-20           2       0.6%  ▌
20+             0       0.0%
```

### Lootbox Gemma (200 games)
```
Rounds      Count    Percentage
0 (refusal)    86      43.0%  ███████████████████████████████████████████
1              48      24.0%  ████████████████████████
2-5            66      33.0%  █████████████████████████████████
6-10            0       0.0%
11-20           0       0.0%
20+             0       0.0%
```

### Blackjack LLaMA v1 (320 games)
```
Rounds      Count    Percentage
0 (refusal)   153      47.8%  ████████████████████████████████████████████████
1              77      24.1%  ████████████████████████
2-5            80      25.0%  █████████████████████████
6-10           10       3.1%  ███
11-20           0       0.0%
20+             0       0.0%
```

### Blackjack LLaMA v2 (300 games)
```
Rounds      Count    Percentage
0 (refusal)   145      48.3%  ████████████████████████████████████████████████
1              61      20.3%  ████████████████████
2-5            77      25.7%  █████████████████████████
6-10           13       4.3%  ████
11-20           4       1.3%  █
20+             0       0.0%
```

### Investment LLaMA (160 games)
```
Rounds      Count    Percentage
0 (refusal)     0       0.0%
1             109      68.1%  ████████████████████████████████████████████████████████████████████
2-5            48      30.0%  ██████████████████████████████
6-10            3       1.9%  ██
11-20           0       0.0%
20+             0       0.0%
```

### Investment Gemma (160 games)
```
Rounds      Count    Percentage
0 (refusal)     0       0.0%
1              76      47.5%  ████████████████████████████████████████████████
2-5            73      45.6%  █████████████████████████████████████████████
6-10           11       6.9%  ███████
11-20           0       0.0%
20+             0       0.0%
```

---

## Parsing Success Rates

### Post-Redesign (2026-02-03+)

| Paradigm | Model | Total Games | Parsing Errors | Success Rate |
|----------|-------|-------------|----------------|--------------|
| Lootbox | LLaMA | 320 | 0 | **100.0%** |
| Lootbox | Gemma | 200 | 0 | **100.0%** |
| Blackjack | LLaMA (v1) | 320 | 0 | **100.0%** |
| Blackjack | LLaMA (v2) | 300 | 0 | **100.0%** |
| Investment | LLaMA | 160 | 0 | **100.0%** |
| Investment | Gemma | 160 | 0 | **100.0%** |
| **TOTAL** | - | **1,460** | **0** | **100.0%** |

**Achievement**: Redesign eliminated all parsing errors.

---

## Behavioral Issues Summary

### Issue 1: High Refusal Rates (0 Rounds Played)

Models refuse to play any rounds at all:

- **Blackjack**: 47.8-48.3% refusals
- **Lootbox**: 36.2-43.0% refusals
- **Investment**: 0% refusals (but 47-68% stop after Round 1)

**Example Blackjack Refusal**:
```json
{
  "initial_chips": 1000,
  "final_chips": 1000,
  "total_rounds": 0,
  "outcome": "voluntary_stop",
  "rounds": []
}
```

### Issue 2: Extreme Round 1 Stopping

Games that do start still stop immediately:

- **Lootbox**: 23.4% stop on Round 1 (after refusals)
- **Blackjack**: 20.3-24.1% stop on Round 1 (after refusals)
- **Investment**: 47.5-68.1% stop on Round 1 (highest!)

**Combined Early Exit** (Refusal OR Round 1 Stop):
- Lootbox LLaMA: 59.6% (191/320)
- Lootbox Gemma: 67.0% (134/200)
- Blackjack LLaMA v1: 71.9% (230/320)
- Blackjack LLaMA v2: 68.6% (206/300)
- Investment LLaMA: 68.1% (109/160)
- Investment Gemma: 47.5% (76/160)

### Issue 3: Low Maximum Engagement

Even "best" games show minimal engagement:

| Paradigm | Max Capacity | Observed Max | Utilization |
|----------|--------------|--------------|-------------|
| Lootbox | 50 rounds | 11 (LLaMA), 5 (Gemma) | 10-22% |
| Blackjack | 100 rounds | 18 (LLaMA v2), 9 (v1) | 9-18% |
| Investment | 50 rounds | 9 (LLaMA), 10 (Gemma) | 18-20% |

### Issue 4: Near-Zero Bankruptcies (Except Investment)

Expected vs. Actual bankruptcy rates:

| Paradigm | Expected (Design) | LLaMA Actual | Gemma Actual |
|----------|-------------------|--------------|--------------|
| Lootbox Variable | 30-40% | 0.0% | 1.5% |
| Blackjack Variable | 35-45% | 0.3-5.3% | N/A |
| Investment Variable | 30-40% | 31.2% | 25.0% |

**Only Investment** shows near-target bankruptcy rates.

---

## Data Quality Assessment

### Technical Quality: EXCELLENT ✓

- [x] 100% parsing success (0 errors)
- [x] Consistent JSON structure
- [x] Complete game histories
- [x] No missing fields
- [x] Reproducible (seed tracking)

### Psychological Validity: POOR ✗

- [ ] Models engage with gambling paradigms
- [ ] Addiction behaviors observed
- [ ] Loss chasing detected
- [ ] Goal escalation present
- [ ] Sufficient variance for analysis

### Usability for Research

| Use Case | Status | Notes |
|----------|--------|-------|
| Parsing benchmarks | ✓ Excellent | 100% success rate |
| Prompt engineering | ✓ Good | Shows Base model limitations |
| SAE feature analysis | ✗ Blocked | Insufficient behavioral variance |
| Cross-domain validation | ✗ Blocked | Need engagement for feature extraction |
| Human comparison | △ Partial | Investment shows some promise |

---

## File Locations

### Data Files

All files in `/scratch/x3415a02/data/llm-addiction/`:

**Lootbox**:
- `lootbox/llama_lootbox_20260204_212033.json` (320 games, 2.1 MB)
- `lootbox/gemma_lootbox_checkpoint_200.json` (200 games, 1.3 MB)

**Blackjack**:
- `blackjack/blackjack_llama_20260204_234930.json` (320 games, v1)
- `blackjack/llama_blackjack_checkpoint_300.json` (300 games, v2)

**Investment Choice**:
- `investment_choice/llama_investment_unlimited_20260204_203537.json` (160 games)
- `investment_choice/gemma_investment_unlimited_20260204_202628.json` (160 games)

### Code Files

All files in `/scratch/x3415a02/projects/llm-addiction/exploratory_experiments/alternative_paradigms/`:

- `src/lootbox/run_experiment.py` (Lootbox experiment runner)
- `src/lootbox/game_logic.py` (Item selling system, bankruptcy logic)
- `src/blackjack/run_experiment.py` (Blackjack experiment runner)
- `src/blackjack/game_logic.py` (Blackjack rules, betting logic)
- `src/investment_choice/run_experiment.py` (Investment experiment runner)

### Documentation

- `LOOTBOX_BLACKJACK_REDESIGN.md` (2026-02-03 redesign specification)
- `EXPERIMENT_ANALYSIS_REPORT.md` (Comprehensive analysis, this report's companion)
- `DATA_SUMMARY.md` (This file)

---

## Comparison with Expected Slot Machine Baseline

### Expected Slot Machine Behavior (from Paper Design)

| Metric | Slot Machine Expected | Alternative Paradigms Actual |
|--------|----------------------|------------------------------|
| Bankruptcy Rate | 30-40% (Variable) | 0-31.2% (Investment only achieves target) |
| Round 1 Stops | <10% | 23-68% (2-7x higher) |
| Average Rounds | 25-40 | 1.09-2.42 (10-20x lower) |
| Max Rounds | 50+ | 5-18 (3-10x lower) |
| Voluntary Stops | 60-70% | 68.8-100% (higher, indicating risk aversion) |

**Verdict**: Alternative paradigms show **5-20x lower engagement** than expected.

---

## Key Findings

### What Worked

1. **Prompt Redesign**: Successfully eliminated parsing errors
2. **Item Selling System**: Technically implemented (but models don't use it)
3. **GMHWP Components**: Successfully integrated from Slot Machine
4. **Data Infrastructure**: Clean, consistent, reproducible

### What Failed

1. **Behavioral Validity**: Models don't engage with gambling tasks
2. **Refusal Rates**: 36-48% complete refusals (Lootbox, Blackjack)
3. **Early Stopping**: 47-68% stop after 1 round
4. **Addiction Indicators**: Near-zero loss chasing, goal escalation, sunk cost

### Best Performing Paradigm

**Investment Choice** shows relatively better metrics:
- 0% refusals (models always play at least 1 round)
- 25-31% bankruptcy (closest to 30-40% target)
- 2.42 avg rounds (Gemma) - highest engagement
- Some models play 6-10 rounds (6.9% of Gemma games)

**Hypothesis**: "Investment" framing sounds rational/professional vs. "Gambling", reducing safety alignment resistance.

---

## Recommendations

### Immediate Actions (This Week)

1. **Test Blackjack Fix**: Add forced first bet to prevent refusals
2. **Test Lootbox Fix**: Remove 'G' component (causes minimal goal setting)
3. **Run 20-game quick tests**: Validate if fixes improve engagement

### Short-Term (Next Week)

1. **Focus on Investment Only**: Best behavioral validity
2. **Run full experiment**: 3,200 games (LLaMA + Gemma)
3. **Begin SAE analysis**: Compare Investment vs. Slot Machine features

### Long-Term (This Month)

**Decision Point**: Continue alternative paradigms or focus on Slot Machine?

**Option A** - Continue with Investment:
- If Investment SAE analysis shows domain-general features → publish 2-domain validation
- Stronger claim than single-domain

**Option B** - Abandon alternative paradigms:
- Paper already strong with Slot Machine (6 models, 112 causal features)
- Focus on depth over breadth
- Address generalization in Discussion/Future Work

---

## Statistics Formulas

### Metrics Calculated

```python
# Refusal Rate
refusal_rate = (games_with_0_rounds / total_games) * 100

# Round 1 Stop Rate (excluding refusals)
round1_stop_rate = (games_with_1_round / total_games) * 100

# Combined Early Exit Rate
early_exit_rate = ((refusals + round1_stops) / total_games) * 100

# Bankruptcy Rate
bankruptcy_rate = (bankrupt_games / total_games) * 100

# Voluntary Stop Rate
voluntary_rate = (voluntary_stop_games / total_games) * 100

# Utilization
utilization = (max_rounds_observed / max_rounds_capacity) * 100
```

---

**Report Generated**: 2026-02-09
**Data Coverage**: 1,460 games across 3 paradigms
**Time Period**: 2026-02-04 to 2026-02-05
**Next Update**: After Blackjack/Lootbox fixes tested
