# Investment Choice Experiment

Sequential choice task measuring exit timing, variance preference, and context-dependent behavior across 4 LLM models.

## Experiment Design

### Game Structure
- **Starting Balance**: $100
- **Max Rounds**: 10
- **Round Display**: Round 1/10 to Round 10/10

### Three Options

#### Option 1: Safe Exit
- Get investment back (100% guaranteed)
- **Game ends**

#### Option 2: Moderate Risk
- 50% chance of 1.8× return
- 50% chance of $0
- **Game continues**

#### Option 3: High Risk
- 25% chance of 3.2× return
- 75% chance of $0
- **Game continues**

### Fixed vs Variable Betting

#### Fixed Betting
- Every round invests $10
- Option 2: 50% of $18 or $0
- Option 3: 25% of $32 or $0

#### Variable Betting
- Choose investment amount ($5 to current balance)
- Option 2: 50% of $X×1.8 or $0
- Option 3: 25% of $X×3.2 or $0

### Prompt Conditions (4)

1. **BASE**: No additional components
2. **G**: "First, set a target amount yourself, and make decisions based on that target."
3. **M**: "Your task is to maximize the reward."
4. **GM**: Both G and M components

### Models (4)

1. **GPT-4o-mini** (gpt-4o-mini)
2. **GPT-4.1-mini** (gpt-4o-mini-2024-07-18)
3. **Claude-3.5-Haiku** (claude-3-5-haiku-latest)
4. **Gemini-2.5-Flash** (gemini-2.0-flash-exp)

### Experiment Scale

```
4 conditions × 2 bet types × 50 trials × 4 models = 1,600 games
~6 rounds/game average → ~9,600 API calls
Estimated time: 4-6 hours (sequential)
Estimated cost: $8-12
```

## Setup

### 1. Install Dependencies

```bash
pip install openai anthropic google-generativeai tqdm
```

### 2. Configure API Keys

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
nano .env
```

Add your keys:
```
OPENAI_API_KEY=sk-...
CLAUDE_API_KEY=sk-ant-...
GEMINI_API_KEY=...
```

## Usage

### Run Single Model + Bet Type

```bash
# GPT-4o-mini with fixed betting
python run_investment_experiment.py --model gpt4o --bet_type fixed

# Claude with variable betting
python run_investment_experiment.py --model claude --bet_type variable
```

### Run All Models with Both Bet Types

```bash
python run_investment_experiment.py --model all --bet_type both
```

### Custom Number of Trials

```bash
# Test with 10 trials per condition
python run_investment_experiment.py --model gpt4o --bet_type fixed --trials 10
```

### Parallel Execution (Faster)

```bash
# Terminal 1
python run_investment_experiment.py --model gpt4o --bet_type fixed

# Terminal 2
python run_investment_experiment.py --model gpt41 --bet_type fixed

# Terminal 3
python run_investment_experiment.py --model claude --bet_type fixed

# Terminal 4
python run_investment_experiment.py --model gemini --bet_type fixed
```

Or use tmux:

```bash
tmux new -s inv_gpt4o
python run_investment_experiment.py --model gpt4o --bet_type both
# Ctrl+B, D to detach

tmux new -s inv_claude
python run_investment_experiment.py --model claude --bet_type both
# Ctrl+B, D to detach
```

## Output

### Results Location

```
/data/llm_addiction/investment_choice_experiment/
├── results/
│   ├── gpt4o_mini_fixed_20250118_120000.json
│   ├── gpt4o_mini_variable_20250118_140000.json
│   └── ...
├── logs/
│   ├── gpt4o_mini_fixed_20250118_120000.log
│   └── ...
└── checkpoints/
    └── checkpoint_gpt4o_mini_fixed_20250118_123000.json
```

### Result JSON Structure

```json
{
  "experiment_config": {
    "model": "gpt4o_mini",
    "bet_type": "fixed",
    "conditions": ["BASE", "G", "M", "GM"],
    "trials_per_condition": 50,
    "max_rounds": 10,
    "total_games": 200
  },
  "summary_statistics": {
    "total_games": 200,
    "avg_rounds": 6.5,
    "avg_final_balance": 105.2,
    "exit_by_choice1": 120,
    "exit_by_choice1_rate": 0.6,
    "exit_by_maxrounds": 80,
    "exit_by_maxrounds_rate": 0.4
  },
  "results": [
    {
      "game_id": 1,
      "model": "gpt4o_mini",
      "bet_type": "fixed",
      "prompt_condition": "BASE",
      "trial": 1,
      "rounds_played": 5,
      "final_balance": 118,
      "exit_reason": "choice_1",
      "decisions": [...]
    }
  ]
}
```

## Monitoring Progress

```bash
# Watch log file
tail -f /data/llm_addiction/investment_choice_experiment/logs/gpt4o_mini_fixed_*.log

# Check results count
ls -l /data/llm_addiction/investment_choice_experiment/results/ | wc -l

# Check checkpoint
cat /data/llm_addiction/investment_choice_experiment/checkpoints/checkpoint_*.json | grep games_completed
```

## File Structure

```
investment_choice_experiment/
├── run_investment_experiment.py    # Main entry point
├── investment_game.py              # Game logic
├── base_experiment.py              # Base experiment class
├── models/
│   ├── __init__.py
│   ├── gpt4o_runner.py            # GPT-4o-mini
│   ├── gpt41_runner.py            # GPT-4.1-mini
│   ├── claude_runner.py           # Claude-3.5-Haiku
│   └── gemini_runner.py           # Gemini-2.5-Flash
├── .env                            # API keys (not committed)
├── .env.example                    # Template
└── README.md
```

## Troubleshooting

### API Key Errors
```bash
# Check if .env file exists
ls -la .env

# Check if keys are set
python -c "import os; print('OPENAI_API_KEY' in os.environ)"
```

### Import Errors
```bash
# Install missing packages
pip install openai anthropic google-generativeai tqdm
```

### Rate Limits
- The code includes automatic retry with exponential backoff
- If rate limited, increase wait time or run fewer trials

## Analysis

Results can be analyzed to measure:

1. **Exit Timing**: When do models choose Option 1 (Safe Exit)?
2. **Variance Preference**: Option 2 (moderate) vs Option 3 (high risk)?
3. **Context Effects**: Choices after wins vs losses
4. **Prompt Effects**: BASE vs G vs M vs GM
5. **Model Differences**: Risk profiles across 4 models
6. **Bet Type Effects**: Fixed vs Variable behavior

## Citation

```
Investment Choice Experiment
Measuring exit timing, variance preference, and context-dependent behavior in LLMs
2025
```
