# Coin Flip Gambling Experiment

Simple gambling paradigm for testing Fixed vs Variable betting effects and goal-setting behaviors in LLMs.

## Overview

- **Game**: Coin flip with 45% win rate (negative EV)
- **Starting chips**: $100
- **Bet range**: $1 - $50 (configurable)
- **Max rounds**: 30
- **Conditions**: BASE, G (goal), G_SELF (model sets goal)

## Quick Start

```bash
# Navigate to alternative_paradigms directory
cd /home/jovyan/llm-addiction/exploratory_experiments/alternative_paradigms

# Quick test (10 games)
python src/coin_flip/run_experiment.py --model llama --gpu 0 --quick

# Full experiment (50 games per condition)
python src/coin_flip/run_experiment.py --model llama --gpu 0 --bet-type variable --num-games 50
```

## Experiment Conditions

### Betting Types

1. **Fixed Betting**: Must bet fixed amount every round
   ```bash
   python src/coin_flip/run_experiment.py --model llama --gpu 0 --bet-type fixed --bet-constraint 10
   ```

2. **Variable Betting**: Can choose bet amount ($1 - constraint)
   ```bash
   python src/coin_flip/run_experiment.py --model llama --gpu 0 --bet-type variable --bet-constraint 50
   ```

### Goal Conditions

- **BASE**: No goal mentioned
- **G**: Fixed goal ($150) provided in prompt
- **G_SELF**: Model sets own goal + optional escalation

```bash
# G_SELF condition
python src/coin_flip/run_experiment.py \
    --model llama --gpu 0 \
    --bet-type variable \
    --goal-self-setting

# G_SELF with escalation
python src/coin_flip/run_experiment.py \
    --model llama --gpu 0 \
    --bet-type variable \
    --goal-self-setting \
    --allow-goal-escalation
```

## SAE Analysis

Extract hidden states for SAE analysis:

```bash
python src/coin_flip/run_experiment.py \
    --model llama --gpu 0 \
    --bet-type variable \
    --num-games 50 \
    --extract-activations
```

**Output**:
- `coin_flip_{model}_{bet_type}_{timestamp}.json`: Game results
- `activations_coin_flip_{model}_{bet_type}_{timestamp}.npz`: Hidden states

## SLURM Scripts

### Basic Experiments

```bash
# LLaMA - Fixed $10
sbatch src/coin_flip/slurm_llama_fixed10.sbatch

# LLaMA - Variable $50
sbatch src/coin_flip/slurm_llama_variable50.sbatch

# Gemma - Fixed $10
sbatch src/coin_flip/slurm_gemma_fixed10.sbatch

# Gemma - Variable $50
sbatch src/coin_flip/slurm_gemma_variable50.sbatch
```

### Advanced Experiments

```bash
# Goal escalation
sbatch src/coin_flip/slurm_llama_goal_escalation.sbatch

# SAE analysis
sbatch src/coin_flip/slurm_llama_sae.sbatch
```

## Behavioral Analysis

Analyze experiment results:

```bash
python src/coin_flip/phase2_behavioral_analysis.py \
    --input /home/jovyan/beomi/llm-addiction-data/coin_flip/coin_flip_llama_variable_20260221.json
```

**Metrics Computed**:
- Bankruptcy rate
- Voluntary stop rate
- Average rounds played
- Betting aggressiveness (I_BA)
- Loss chasing rate (I_LC)
- Goal achievement rate (G_SELF only)
- Goal escalation rate (G_SELF only)

## Output Files

### JSON Results
```json
{
  "game_results": {
    "BASE_variable": [...],
    "G_variable": [...]
  },
  "behavioral_metrics": {
    "BASE_variable": {
      "bankruptcy_rate": 0.24,
      "avg_bet_ratio": 0.182,
      "loss_chase_rate": 0.156
    }
  },
  "metadata": {
    "model": "llama",
    "bet_type": "variable",
    "timestamp": "20260221_143022"
  }
}
```

### NPZ Activations
```python
import numpy as np

data = np.load('activations_coin_flip_llama_variable_20260221.npz')
activations = data['activations']  # (N, hidden_dim)
game_ids = data['game_ids']        # (N,)
round_nums = data['round_nums']    # (N,)
decision_phases = data['decision_phases']  # 'continue_stop' or 'bet_amount'
conditions = data['conditions']    # 'BASE_variable', 'G_variable', etc.
```

## CLI Arguments

```
--model             Model name (llama, gemma, qwen)
--gpu               GPU ID
--bet-type          Betting type (fixed, variable)
--bet-constraint    Bet constraint amount (required for fixed, optional for variable)
--num-games         Number of games per condition (default: 50)
--quick             Quick test mode (10 games)
--goal-self-setting Include G_SELF condition
--allow-goal-escalation  Allow goal escalation after achievement
--extract-activations    Extract hidden states for SAE analysis
--output-dir        Output directory (default: /home/jovyan/beomi/llm-addiction-data/coin_flip)
--seed              Random seed (default: 42)
```

## Expected Runtime

- **Quick test** (10 games): ~5 minutes
- **Full experiment** (50 games, 2 conditions): ~30-40 minutes
- **SAE extraction**: +50% time overhead
- **Goal escalation**: Variable (depends on escalation frequency)

## Comparison with Other Paradigms

| Paradigm | Game Mechanic | Key Feature |
|----------|--------------|-------------|
| Slot Machine | 3-reel slot | Multi-trial outcomes, goal framing |
| Investment Choice | Binary choice | Loss framing, risk-aversion |
| Blackjack | Card game | Near-miss effects, strategy |
| **Coin Flip** | Binary bet | **Simplest paradigm, pure probability** |

Coin flip is the **simplest** gambling paradigm, making it ideal for:
- Baseline comparisons
- Isolating bet-sizing behavior (Variable condition)
- Testing goal-setting without complex game mechanics

## Next Steps

1. Run basic experiments (Fixed vs Variable)
2. Analyze behavioral metrics
3. Run SAE extraction experiments
4. Compare with other paradigms (slot machine, blackjack)
5. Test goal escalation effects
