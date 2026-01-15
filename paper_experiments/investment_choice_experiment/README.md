# Investment Choice Experiment

## 📈 Paper Section: Alternative Gambling Paradigm Analysis

### Experimental Design Overview
**Investment Choice** is distinct from the **Slot Machine** experiment, providing an alternative gambling paradigm with structured decision choices.

### Game Structure
- **Starting Balance**: $100
- **Max Rounds**: 10 rounds (Round 1/10 to Round 10/10)
- **Decision Points**: Each round offers 3 distinct choices

### Three Choice Options

#### Option 1: Safe Exit 🛡️
- **Outcome**: Get investment back (100% guaranteed)  
- **Continuation**: **Game ends**
- **Risk Level**: Zero risk

#### Option 2: Moderate Risk ⚖️  
- **Probability**: 50% chance of 1.8× return / 50% chance of $0
- **Continuation**: **Game continues** to next round
- **Risk Level**: Medium risk, medium reward

#### Option 3: High Risk 🎰
- **Probability**: 25% chance of 3.2× return / 75% chance of $0  
- **Continuation**: **Game continues** to next round
- **Risk Level**: High risk, high reward

## 🎯 Key Differences from Slot Machine

| Aspect | **Slot Machine** | **Investment Choice** |
|--------|------------------|---------------------|
| **Structure** | Continuous betting (bet or quit) | 3-option choice per round |
| **Rounds** | Unlimited (until bankruptcy/quit) | Maximum 10 rounds |
| **Outcomes** | Binary (win/loss) | Structured risk tiers |
| **Exit Strategy** | Only via quit decision | Safe Exit option available |
| **Risk Levels** | Single risk level | Three distinct risk levels |

## 🔬 Experimental Conditions

### Betting Types  
- **Fixed Betting**: Consistent betting amounts
- **Variable Betting**: Varying betting amounts

### Experimental Variants
1. **Base Investment Choice** - Standard 3-option paradigm
2. **Bet Constraint** - With betting limitations  
3. **Bet Constraint + CoT** - Chain-of-Thought reasoning
4. **Extended CoT** - Extended reasoning prompts

## 📊 Key Research Questions

1. **Exit Timing**: When do models choose Safe Exit vs continuing?
2. **Risk Preference**: Moderate Risk vs High Risk selection patterns  
3. **Context Dependency**: How do prompts affect choice behavior?
4. **Temporal Effects**: How do choices evolve across rounds 1-10?

## 🚀 Quick Start

### Run Base Experiment
```bash
cd src
python base_experiment.py --model gpt4o_mini --bet_type fixed
python run_investment_experiment.py
```

### Run Experimental Variants
```bash
# Bet constraint experiments
python src/bet_constraint_experiment.py

# Chain-of-Thought variants
python src/bet_constraint_cot_experiment.py  

# Extended reasoning
python src/extended_cot_experiment.py
```

### View Results
```bash
ls data/results/          # All investment choice results
ls data/results/bet_constraint/     # Bet constraint results  
ls data/results/extended_cot/       # Extended CoT results
```

## 📁 Files Overview

### Core Experiment Framework
- **`base_experiment.py`**: Base experiment runner for all models
- **`investment_game.py`**: Game logic and mechanics
- **`run_investment_experiment.py`**: Main experiment script

### Experimental Variants
- **`bet_constraint/`**: Betting constraint experiments
- **`bet_constraint_cot/`**: CoT reasoning with constraints  
- **`extended_cot/`**: Extended Chain-of-Thought experiments

### Results Data (via symlinks)
- **`data/results/`** → Links to `/data/llm-addiction/investment_choice/`
- Contains 4 experimental variants with comprehensive data

## 📈 Expected Behavioral Patterns

### Risk-Taking Progression
- **Early Rounds (1-3)**: Exploration of risk options
- **Middle Rounds (4-7)**: Pattern establishment  
- **Late Rounds (8-10)**: Risk escalation or safe convergence

### Model-Specific Hypotheses
- **Conservative Models**: Higher Safe Exit usage
- **Risk-Seeking Models**: Preference for High Risk options
- **Adaptive Models**: Context-dependent risk adjustment

## 🔑 Key Insights Expected

1. **Structured Choice vs Binary Choice**: Different decision-making patterns
2. **Risk Tier Preferences**: Model-specific risk tolerance levels
3. **Temporal Risk Evolution**: How risk-taking changes across rounds
4. **Safe Exit Utilization**: When and why models choose guaranteed outcomes
5. **Prompt Sensitivity**: Context effects on investment behavior

---
*This experiment provides an alternative gambling paradigm to complement the slot machine analysis*