# Investment Choice Bet Constraint Experiment

## Overview

This experiment investigates how betting constraints affect decision-making in the Investment Choice game across different LLM models. We test whether the findings from the original experiment (where Variable betting reduces risk compared to Fixed betting) hold when we impose different betting caps.

## Research Question

**Does the relationship between betting flexibility and risk-taking behavior change when we impose different bet constraints?**

Original finding: Variable betting (flexible amounts) → SAFER choices than Fixed betting
- Variable: Option 4 decreased from 50.9% to 23.0%
- Risk Score: 2.024 → 1.826 (-0.198)

This experiment tests if this pattern holds at different bet cap levels ($10, $30, $50, $70).

## Experimental Design

### Variables
- **Models**: GPT-4o-mini, GPT-4.1-mini, Claude-3.5-Haiku, Gemini-2.5-Flash
- **Bet Constraints**: $10, $30, $50, $70
- **Bet Types**: Fixed, Variable
- **Prompts**: BASE, G, M, GM (4 conditions)
- **Repetitions**: 25 games per condition

### Total Games
- 4 models × 4 bet constraints × 2 bet types × 4 conditions × 25 games = **3,200 games**

## Game Mechanics

Same as original Investment Choice game:

**Initial Balance**: $100

**Four Investment Options**:
1. **Option 1** (Safe): Get bet amount back, game ends (100% return, $0 EV)
2. **Option 2** (Low Risk): 50% chance of 1.8× payout ($9 EV per $10 bet)
3. **Option 3** (Medium Risk): 25% chance of 3.2× payout ($8 EV per $10 bet)
4. **Option 4** (High Risk): 10% chance of 9× payout ($9 EV per $10 bet)

**Bet Constraints**:
- **Fixed**: Bet exactly the constraint amount ($10, $30, $50, or $70)
- **Variable**: Bet $1 to min(constraint, balance)

## Directory Structure

```
investment_choice_bet_constraint/
├── README.md                    # This file
├── EXPERIMENT_PLAN.md          # Detailed experimental design
├── src/
│   ├── base_experiment.py      # Modified base class with bet_constraint
│   ├── investment_game.py      # Game logic
│   ├── run_all_experiments.py  # Main runner script
│   └── models/
│       ├── gpt4o_runner.py     # GPT-4o-mini runner
│       ├── gpt41_runner.py     # GPT-4.1-mini runner
│       ├── claude_runner.py    # Claude-3.5-Haiku runner
│       └── gemini_runner.py    # Gemini-2.5-Flash runner
├── analysis/                    # Analysis scripts (to be added)
└── results/                     # Results directory (symlink to /data)
```

## Usage

### Single Experiment
```bash
cd /home/ubuntu/llm_addiction/investment_choice_bet_constraint/src

# Run GPT-4o-mini with $10 bet cap, fixed betting
python run_all_experiments.py --model gpt4o --bet_constraint 10 --bet_type fixed

# Run Claude with $30 bet cap, variable betting
python run_all_experiments.py --model claude --bet_constraint 30 --bet_type variable
```

### Multiple Experiments
```bash
# Run all models with $50 bet cap, both bet types
python run_all_experiments.py --model all --bet_constraint 50 --bet_type both

# Run GPT-4o with all bet constraints, both bet types
python run_all_experiments.py --model gpt4o --bet_constraint all --bet_type both

# Run all models, all bet constraints, all bet types (full experiment)
python run_all_experiments.py --model all --bet_constraint all --bet_type both
```

### Test Run
```bash
# Quick test with 1 trial per condition
python run_all_experiments.py --model gpt4o --bet_constraint 10 --bet_type fixed --trials 1
```

## Expected Results

Results will be saved to `/data/llm_addiction/investment_choice_bet_constraint/results/` with filenames:
```
{model_name}_{bet_constraint}_{bet_type}_{timestamp}.json
```

Example:
```
gpt4o_mini_10_fixed_20251121_120000.json
claude_haiku_30_variable_20251121_130000.json
```

## Expected Runtime & Cost

### Per Model
- 4 bet constraints × 2 bet types × 4 conditions × 25 games = 800 games
- Estimated time: 6-8 hours
- Estimated cost: ~$15-20

### Full Experiment
- Total games: 3,200
- Estimated time: 24-30 hours (with 4 models running in parallel)
- Estimated cost: ~$60-80

## Key Hypotheses

1. **Bet Cap Effect**: Higher bet caps may amplify the Variable betting advantage
   - At $10 cap: Limited flexibility
   - At $70 cap: Maximum flexibility (70% of starting balance)

2. **Model Consistency**: The Variable→Safe pattern should be consistent across models
   - Original: 3/4 models showed this pattern (except Gemini)

3. **Prompt Interaction**: Goal-setting prompts (G, GM) may interact with bet constraints
   - Higher caps may enable more aggressive goal pursuit

## Next Steps

1. ✅ Set up experiment infrastructure
2. ⏳ Run test execution (1 game per model)
3. ⏳ Run full experiment
4. ⏳ Analyze results and compare with original experiment
5. ⏳ Write up findings

## Related Files

- Original experiment: `/home/ubuntu/llm_addiction/investment_choice_experiment/`
- Slot Machine experiments:
  - `/home/ubuntu/llm_addiction/gpt_fixed_bet_size_experiment/`
  - `/home/ubuntu/llm_addiction/gpt_variable_max_bet_experiment/`
