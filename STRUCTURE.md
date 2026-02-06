# Repository Structure Guide

This document provides a clear overview of the repository organization, distinguishing between **paper experiments** (used in the main publication) and **exploratory experiments** (additional research).

## Quick Reference

```
llm-addiction/
├── paper_experiments/          # 📄 PAPER: Core experiments in publication
├── exploratory_experiments/    # 🔬 EXPLORATORY: Additional research
└── legacy/                     # 🗄️ LEGACY: Archived experiments
```

---

## 📄 Paper Experiments (`paper_experiments/`)

**These experiments appear in the paper: "Can Large Language Models Develop Gambling Addiction?"**

### 1. `slot_machine_6models/`
**Paper Section**: 3.1 (Findings 1-5)
**Sample Size**: 19,200 games (6 models × 64 conditions × 50 replications)
**Purpose**: Main behavioral study examining addiction-like behaviors across diverse models

**Key Findings**:
- Finding 1: Variable betting dramatically amplifies bankruptcy rates
- Finding 2: Variable betting amplifies streak chasing behavior
- Finding 3: Goal-setting prompts reshape risk preferences
- Finding 4: Independent effect of betting flexibility confirmed
- Finding 5: Linguistic traces reveal cognitive distortions

**Models**: GPT-4o-mini, GPT-4.1-mini, Gemini-2.5-Flash, Claude-3.5-Haiku, LLaMA-3.1-8B, Gemma-2-9B

**Design**: 2×32 factorial (Betting Style × Prompt Composition)

---

### 2. `investment_choice_experiment/`
**Paper Section**: 3.1 (Ablation Study)
**Sample Size**: 6,400 games (4 models × 8 conditions × 200 replications)
**Purpose**: Isolate effects of autonomy dimensions (betting flexibility + goal autonomy)

**Key Analyses**:
- Goal escalation after achievement (56-59% vs 21-22% baseline)
- Risk preference shifts (Option 4 selection: 15% → 41%)
- Bet constraint effects (consistent +3.3% bankruptcy)

**Models**: GPT-4o-mini, GPT-4.1-mini, Gemini-2.5-Flash, Claude-3.5-Haiku (API models only)

**Design**: 2×4 factorial (Betting Style × Prompt Condition: BASE, G, M, GM)

---

### 3. `llama_sae_analysis/`
**Paper Section**: 3.2 (Neural Mechanisms)
**Sample Size**: 6,400 games, 1,015,808 features → 112 causal features
**Purpose**: Identify neural features causally controlling gambling behavior

**Key Findings**:
- 112 causally-verified features (~1% of candidates)
- Bidirectional causal influence (safe vs risky features)
- Anatomical segregation (layer-wise specialization)
- Semantic interpretability (goal-pursuit vs stopping words)

**Methodology**:
- Phase 1: SAE feature extraction (31 layers × 32,768 features)
- Phase 2-3: Statistical analysis (FDR correction)
- Phase 4: Activation patching for causal validation

**Model**: LLaMA-3.1-8B only

---

### 4. `pathway_token_analysis/`
**Paper Section**: 5 (Methods - token-level analysis)
**Purpose**: Temporal/linguistic patterns in decision-making sequences

**Analysis**: Token-level activation patterns during gambling decisions

---

## 🔬 Exploratory Experiments (`exploratory_experiments/`)

**These experiments are NOT in the paper but support methodological development and validation.**

### 1. `steering_vector_analysis/`
**Purpose**: CAA-based steering vector extraction for directional control

**Pipeline**: 5 phases (data prep → vector extraction → validation → visualization → analysis)

**Key Features**:
- Multi-GPU support (4-GPU launcher)
- Alternative intervention method to SAE patching

**Usage**:
```bash
bash steering_vector_analysis/scripts/launch_full_analysis_4gpu.sh
python steering_vector_analysis/src/extract_steering_vectors.py --config configs/experiment_config.yaml
```

---

### 2. `gemma_sae_experiment/`
**Purpose**: Validate SAE methodology on Gemma-2-9B architecture

**Pipeline**: 6 phases (Phase 0 optional: domain-specific SAE training)

**Key Features**:
- Uses GemmaScope SAEs from HuggingFace
- Optional "SAE Boost" (Phase 0) for domain adaptation

**Usage**:
```bash
python gemma_sae_experiment/run_pipeline.py --gpu 0 --phases all --use-boost
```

---

### 3. `lr_classification_experiment/`
**Purpose**: Predict bankruptcy from hidden state representations

**Analysis Options**:
- **Option A**: Start point (initialization)
- **Option B**: End point (final decision) - **CORE**
- **Option C**: All rounds (trajectory)

**Baselines**: Chance, TF-IDF, Metadata-Only, Random Projection

**Usage**:
```bash
python lr_classification_experiment/run_experiment.py --model gemma --option B --gpu 0
python lr_classification_experiment/run_experiment.py --model all --option B --gpu 0 --baselines-only
python lr_classification_experiment/run_experiment.py --model gemma --option B --gpu 0 --quick
```

---

### 4. `alternative_paradigms/`
**Purpose**: Domain generalization validation across 3 gambling tasks

**Tasks**:

#### 4.1 Iowa Gambling Task (IGT)
```bash
python alternative_paradigms/src/igt/run_experiment.py --model llama --gpu 0 --quick
```

#### 4.2 Loot Box Mechanics
```bash
python alternative_paradigms/src/lootbox/run_experiment.py --model gemma --gpu 0 --quick
```

#### 4.3 Near-Miss Slot Machine
```bash
python alternative_paradigms/src/nearmiss/run_experiment.py --model qwen --gpu 0 --bet-type variable --quick
```

**Goal**: Test if "autonomy → risk escalation" generalizes beyond standard slot machines

---

### 5. `additional_experiments/`
**Purpose**: Post-submission extensions and follow-up analyses

**Key**: `sae_condition_comparison/` - Variable vs Fixed neural differences

**CRITICAL ISSUE**: Sparse features (<1% activation) cause interaction artifacts
**Solution**: Apply minimum activation thresholds (see `ANALYSIS_ISSUES_REPORT.md`)

**Usage**:
```bash
python -m additional_experiments.sae_condition_comparison.src.condition_comparison --model llama
```

---

## 🗄️ Legacy (`legacy/`)

Archived experiments from iterative development process. Not maintained.

---

## Data Locations

**Code Repository**: `/mnt/c/Users/oollccddss/git/llm-addiction/` (WSL2)
**Data Repository**: `/mnt/c/Users/oollccddss/git/data/llm-addiction/`

```
data/llm-addiction/
├── slot_machine/
│   ├── gemma/final_gemma_20251004_172426.json
│   ├── llama/final_llama_20251004_021106.json
│   ├── claude/claude_experiment_20250920_003210.json
│   └── gemini/gemini_experiment_20250922_003406.json
└── [other experiment outputs]
```

---

## Environment

**Conda environment**: `llama_sae_env`

```bash
conda activate llama_sae_env
```

**Key dependencies**:
- `sae_lens` 6.5.1 (SAE analysis)
- `transformers` (model loading)
- `torch` (GPU computation, bf16)
- `openai`, `anthropic`, `google-generativeai` (API models)

**No centralized package manager** - all dependencies managed through conda.

---

## Common Patterns

### Architecture
All experiments follow **phase-based pipeline** pattern:
```
Config (YAML) → Phase Scripts (Python) → Results (JSON/NPZ/JSONL)
```

### Reproducibility
- `set_random_seed(42)` before experiments
- `clear_gpu_memory()` between phases
- Models loaded in **bf16** (not float16 or quantized)

### Resumability
- Checkpoint-based resumption for GPU interruptions
- Phase scripts check for existing outputs before re-running

---

## Quick Start

### Run Paper Experiments

```bash
# Slot machine (API models)
python paper_experiments/slot_machine_6models/src/run_gpt5_experiment.py
python paper_experiments/slot_machine_6models/src/run_claude_experiment.py
python paper_experiments/slot_machine_6models/src/run_gemini_experiment.py

# Slot machine (local models)
python paper_experiments/slot_machine_6models/src/llama_gemma_experiment.py

# SAE analysis
python paper_experiments/llama_sae_analysis/src/phase1_feature_extraction.py
python paper_experiments/llama_sae_analysis/src/phase4_causal_pilot_v2.py

# Pathway token analysis
bash paper_experiments/pathway_token_analysis/scripts/launch_all_phases_sequential.sh
```

### Run Exploratory Experiments

See individual experiment READMEs in `exploratory_experiments/` subdirectories.

---

## Documentation

- **CLAUDE.md**: Primary guidance for Claude Code (most comprehensive)
- **STRUCTURE.md** (this file): Repository organization
- **AGENTS.md**: **OUTDATED** - disregard in favor of CLAUDE.md
- **exploratory_experiments/README.md**: Detailed exploratory experiments guide
- Individual experiment folders: Task-specific READMEs

---

## Paper Mapping

| Paper Section | Experiment Directory | Sample Size |
|--------------|---------------------|-------------|
| 3.1 (Findings 1-5) | `paper_experiments/slot_machine_6models/` | 19,200 games |
| 3.1 (Ablation) | `paper_experiments/investment_choice_experiment/` | 6,400 games |
| 3.2 (Neural Mechanisms) | `paper_experiments/llama_sae_analysis/` | 112 features |
| 5 (Token Analysis) | `paper_experiments/pathway_token_analysis/` | - |

**Total paper sample size**: 25,600 games across 6 models

---

## Git Workflow

**Main branch**: `master` (use this for PRs)
**Current branch**: `sae`

**Status** (snapshot at conversation start):
```
?? additional_experiments/sae_condition_comparison/[various files]
```

---

## GPU Requirements

- LLaMA-3.1-8B: ~19GB VRAM (bf16)
- Gemma-2-9B: ~22GB VRAM (bf16)
- Multi-GPU: Use `CUDA_VISIBLE_DEVICES` or `--gpu` flag
- CPU-only: `additional_experiments/sae_condition_comparison/` (statistical tests)

---

## Key Research Findings (for context)

1. **Autonomy effect**: Variable betting → +3.3% bankruptcy vs Fixed
2. **Neural mechanisms**: 112 causal features (~1% of candidates)
3. **Layer segregation**: Risky features (L24), Safe features (L4-L19)
4. **Goal dysregulation**: 56-59% escalation in goal-setting conditions
5. **Cognitive distortions**: Illusion of control, gambler's fallacy, loss chasing, house money effect

---

## Notes

- Documentation is **bilingual** (English/Korean) - both acceptable
- `.gitignore` excludes experiment outputs
- No formal test suite - validation within pipelines
- Legacy folder contains archived iterative development
- Project actively developed for **ICLR 2026 submission**
