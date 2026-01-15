# Slot Machine 6-Models Experiment

## 📊 Paper Section 3: "Can LLM Develop Gambling Addiction?"

### Experimental Design
- **Models**: GPT-4o-mini, GPT-4.1-mini, Gemini-2.5-Flash, Claude-3.5-Haiku, LLaMA-3.1-8B, Gemma-2-9B
- **Design**: 2×32 factorial design
  - **Betting Style**: Fixed ($10) vs Variable ($5-$100)  
  - **Prompt Composition**: 32 variations (BASE + 5 components)
- **Total**: 64 conditions × 50 reps = 3,200 games per model
- **Game Settings**: 30% win rate, 3× payout, -10% expected value

### Key Paper Results
| Model | Fixed Bankruptcy | Variable Bankruptcy | Irrationality Index |
|-------|------------------|---------------------|-------------------|
| GPT-4o-mini | 0.00% | **21.31%** | 0.172 ± 0.005 |
| GPT-4.1-mini | 0.00% | **6.31%** | 0.077 ± 0.002 |
| Gemini-2.5-Flash | 3.12% | **48.06%** | 0.265 ± 0.005 |
| Claude-3.5-Haiku | 0.00% | **20.50%** | 0.186 ± 0.003 |
| LLaMA-3.1-8B | 0.11% | **7.14%** | 0.125 ± 0.015 |
| Gemma-2-9B | 12.81% | **29.06%** | 0.271 ± 0.118 |

## 🚀 Quick Start

### Run Experiments
```bash
# API-based models
python src/run_gpt5_experiment.py        # GPT-4o-mini + GPT-4.1-mini
python src/run_claude_experiment.py      # Claude-3.5-Haiku
python src/run_gemini_experiment.py      # Gemini-2.5-Flash

# Open-weight models  
python src/llama_gemma_experiment.py     # LLaMA-3.1-8B + Gemma-2-9B
```

### View Results
```bash
ls data/results/          # All 6 model results
ls data/results/gpt/      # GPT results specifically
ls data/results/claude/   # Claude results
```

## 📁 Files Overview

### Core Experiment Scripts
- **`run_gpt5_experiment.py`**: GPT-4o-mini & GPT-4.1-mini experiments
- **`run_claude_experiment.py`**: Claude-3.5-Haiku experiments  
- **`run_gemini_experiment.py`**: Gemini-2.5-Flash experiments
- **`llama_gemma_experiment.py`**: LLaMA-3.1-8B & Gemma-2-9B experiments

### Results Data (via symlinks)
- **`data/results/`** → Links to `/data/llm-addiction/slot_machine/`
- Contains all 6 models' complete experimental results

## 🔬 Methodology

**Prompt Components:**
- **G**: Goal-Setting  
- **M**: Maximizing Rewards
- **H**: Hinting at Hidden Patterns
- **W**: Win-reward Information  
- **P**: Probability Information

**Experimental Procedure:**
1. Initial capital: $100
2. Each round: Bet or Quit decision
3. Win rate: 30%, Payout: 3×, Expected value: -10%
4. Balance and game history provided after first round

## 📈 Key Findings

1. **Variable betting dramatically increases bankruptcy risk** across all models
2. **Model-specific vulnerability**: Gemini-2.5-Flash most vulnerable (48.06%), GPT-4.1-mini most resilient (6.31%)  
3. **Fixed betting provides protection** with near-zero bankruptcy rates
4. **Strong correlation** between irrationality index and bankruptcy rate

---
*This experiment demonstrates addiction-like behaviors in LLMs under gambling scenarios*