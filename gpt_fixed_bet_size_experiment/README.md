# GPT-4o Fixed Bet Size Variation Experiment

## 📍 Quick Info

- **Location**: `/home/ubuntu/llm_addiction/gpt_fixed_bet_size_experiment/`
- **Model**: gpt-4o (NOT gpt-4o-mini)
- **Bet sizes**: $30, $50, $70
- **Total experiments**: 960 (96 conditions × 10 repetitions)
- **Estimated time**: 12-15 hours
- **Estimated cost**: ~$32 USD

## Overview

This experiment investigates how **fixed bet size** affects GPT-4o's gambling behavior by testing three bet amounts: **$30, $50, and $70** (vs. original $10 baseline).

## Quick Start

### 1. Set API Key
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 2. Run Experiment
```bash
cd /home/ubuntu/llm_addiction/gpt_fixed_bet_size_experiment/src
python gpt_fixed_bet_size_experiment.py
```

### 3. Monitor Progress
```bash
# Check log file
tail -f /home/ubuntu/llm_addiction/gpt_fixed_bet_size_experiment/logs/experiment_*.log

# Check intermediate results
ls -lh /home/ubuntu/llm_addiction/gpt_fixed_bet_size_experiment/results/
```

## Experiment Design

### Conditions
- **Bet sizes**: 3 ($30, $50, $70)
- **Prompts**: 32 combinations (BASE, G, M, P, R, W, ...)
- **Total**: 96 conditions × 10 reps = **960 experiments**

### Game Settings
```
Initial balance: $100
Win rate: 30%
Payout: 3.0x
Expected value: -10% per bet
Max rounds: 100
```

### Bet Size Rationale

| Bet | % of Initial | Risk Level | Expected Result |
|-----|--------------|------------|-----------------|
| $30 | 30% | Moderate | 15-25% bankruptcy |
| $50 | 50% | High | 35-50% bankruptcy |
| $70 | 70% | Very High | 60-80% bankruptcy |

## File Structure

```
gpt_fixed_bet_size_experiment/
├── src/
│   ├── gpt_fixed_bet_size_experiment.py    # Main runner
│   └── analyze_bet_size_effects.py          # Analysis
├── results/
│   ├── intermediate_*.json                   # Every 50 experiments
│   └── complete_*.json                       # Final results
├── logs/
│   └── experiment_*.log                      # Execution log
├── EXPERIMENT_PLAN.md                        # Detailed plan
└── README.md                                 # This file
```

## Key Research Questions

1. **Does bankruptcy rate scale with bet size?**
   - H1: $30 (15-25%) < $50 (35-50%) < $70 (60-80%)

2. **Are prompt effects consistent across bet sizes?**
   - Do "maximize reward" and "set target" behave the same at all bet levels?

3. **Is there an interaction effect?**
   - Does "maximize reward" + $70 create extreme risk-taking?

4. **How does gpt-4o differ from gpt-4o-mini?**
   - Baseline comparison with original $10 experiment

## Cost Estimate

```
Model: gpt-4o
Total API calls: ~14,400 (960 games × 15 rounds avg)
Cost per call: ~$0.0015
Total cost: ~$21.60 + buffer = ~$32 USD
```

**⚠️ Warning**: gpt-4o costs 15-20x more than gpt-4o-mini

## Comparison with Original

| Feature | Original | This Experiment |
|---------|----------|----------------|
| Model | gpt-4o-mini | **gpt-4o** |
| Bet sizes | $10 | **$30, $50, $70** |
| Bet types | Fixed + Variable | **Fixed only** |
| Conditions | 64 | **96** |
| Repetitions | 50 | **10** |
| Total experiments | 3,200 | **960** |
| Time | 20-30h | **12-15h** |
| Cost | $5-7 | **~$32** |

## Status

- [x] Experiment planned
- [x] Folder structure created
- [ ] Code implemented
- [ ] Test runs completed (3 trials)
- [ ] Full experiment executed (960 runs)
- [ ] Results analyzed

## References

- **Original experiment**: `/home/ubuntu/llm_addiction/gpt_experiments/src/gpt_corrected_multiround_experiment.py`
- **Baseline results**: `/data/llm_addiction/gpt_results_corrected/gpt_corrected_complete_20250825_212628.json`
- **Detailed plan**: `EXPERIMENT_PLAN.md` (this directory)

## Notes

- All prompts are in **English** (matching LLaMA experiment)
- Uses **last-token parsing** (corrected method)
- Saves intermediate results every **50 experiments** to prevent data loss
- Automatic retry logic with exponential backoff for API errors
