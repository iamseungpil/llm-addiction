# Paper Experiments - ICLR 2026 Submission

> Archive note: this directory reflects an older paper-specific organization
> and should not be treated as the canonical reproduction path for the current
> manuscript. For the current paper, use `sae_v3_analysis/README.md`,
> `sae_v3_analysis/docs/PAPER_CANONICAL.md`, and
> `sae_v3_analysis/docs/PAPER_MANIFEST.md`.

## Corresponding Paper: "Can Large Language Models Develop Gambling Addiction?"

This folder contains **only** the experimental code used to generate the results presented in our ICLR 2026 paper submission.

> **Note**: 추가 실험들은 `additional_experiments/` 폴더에 있습니다.

## Experiment Structure

```
paper_experiments/
├── slot_machine_6models/        # Section 3a: 6-model slot machine experiments
├── investment_choice_experiment/ # Section 3b: Investment choice alternative paradigm
├── llama_sae_analysis/          # Section 4: LLaMA SAE & activation patching
└── pathway_token_analysis/      # Section 5: Temporal & linguistic analysis
```

## 📊 **Experiment 1a: Slot Machine 6-Models** (Section 3a)
**Paper Table**: Multi-model comprehensive gambling behavior analysis  
**Models**: GPT-4o-mini, GPT-4.1-mini, Gemini-2.5-Flash, Claude-3.5-Haiku, LLaMA-3.1-8B, Gemma-2-9B  
**Design**: 2×32 factorial (Fixed/Variable betting × 32 prompt variations)  
**Total Trials**: 19,200 games (3,200 per model)  

### Key Results
- Variable betting → higher bankruptcy rates (6.31%-48.06%)
- Fixed betting → near-zero bankruptcy rates
- Gemini-2.5-Flash: highest bankruptcy rate (48.06%)
- GPT-4.1-mini: lowest bankruptcy rate (6.31%)

---

## 📈 **Experiment 1b: Investment Choice** (Section 3b)
**Paper Analysis**: Alternative gambling paradigm with structured choices  
**Structure**: 3-option choice (Safe Exit / Moderate Risk / High Risk)  
**Rounds**: Maximum 10 rounds per experiment  
**Variants**: 4 experimental conditions (base, bet constraint, CoT, extended CoT)

### Key Results
- Structured choice behavior differs from continuous betting
- Safe Exit utilization varies by model and context
- Risk tier preferences reveal model-specific tolerance levels
- Temporal progression shows evolving risk-taking patterns

---

## 🔬 **Experiment 2: LLaMA SAE Analysis** (Section 4)
**Paper Figures**: Feature separation, causal patching comparison, layer distribution  
**Model**: LLaMA-3.1-8B  
**Analysis**: 6,400 games → SAE features (Layers 25-31)  
**Discovery**: 83,684 differential features → 361 safe + 80 risky causal features  

### Key Results
- 68.1% of features distinguish bankrupt vs safe decisions
- Safe features: +29.6% stopping rate, -14.2% bankruptcy rate
- Risky features: -7.8% stopping rate, +11.7% bankruptcy rate

---

## 📈 **Experiment 3: Pathway Token Analysis** (Section 5)  
**Paper Content**: Temporal and linguistic dimensions  
**Analysis**: Token-level pathway tracking, word-feature correlations  
**Phases**: 5-phase pipeline (activations → correlations → validation → words → prompts)  

### Key Results
- Temporal progression patterns of risk-taking
- Word-level linguistic correlates of gambling behavior
- Feature pathway evolution across decision time points

---

## 🚀 Quick Start

### Run Individual Experiments
```bash
# Experiment 1a: 6-model slot machine
cd slot_machine_6models
python src/run_gpt5_experiment.py        # GPT models
python src/run_claude_experiment.py      # Claude  
python src/run_gemini_experiment.py      # Gemini
python src/llama_gemma_experiment.py     # LLaMA/Gemma

# Experiment 1b: Investment choice
cd investment_choice_experiment
python src/run_investment_experiment.py  # Base experiment
python src/bet_constraint_experiment.py  # Bet constraint variant

# Experiment 2: LLaMA SAE analysis
cd llama_sae_analysis  
python src/phase1_feature_extraction.py
python src/phase4_causal_pilot_v2.py

# Experiment 3: Pathway analysis
cd pathway_token_analysis
bash scripts/launch_all_phases_sequential.sh
```

### Access Results Data
All experimental results are accessible via symbolic links in each `data/results/` directory.

---

## 📁 File Organization

Each experiment folder contains:
- **`src/`**: Core Python scripts
- **`configs/`**: Configuration files  
- **`scripts/`**: Launch scripts
- **`data/results/`**: Symbolic links to actual data
- **`docs/`**: Documentation (when available)

## 🔗 Data Sources
- **Raw results**: `/mnt/c/Users/oollccddss/git/data/llm-addiction/`
- **Legacy code**: `/mnt/c/Users/oollccddss/git/llm-addiction/legacy/`
- **Paper source**: `/mnt/c/Users/oollccddss/git/llm-addiction/asset/ICLR_2026_Addictions_on_LLM (3).pdf`

---
*This clean organization contains only the code necessary to reproduce the paper's experimental results*
