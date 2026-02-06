# Exploratory Experiments

This directory contains experiments that were **not directly included in the main paper** ("Can Large Language Models Develop Gambling Addiction?"). These experiments were conducted for:
- Methodological exploration
- Alternative analysis approaches
- Post-submission extensions
- Domain generalization validation

## Directory Structure

### 1. `steering_vector_analysis/`
**Purpose**: CAA-based steering vector experiments for directional control

**Description**: 5-phase pipeline extracting steering vectors to modulate gambling behavior. Uses Contrastive Activation Addition (CAA) methodology.

**Key Features**:
- Multi-GPU support (4-GPU launcher available)
- 5 phases: data preparation, vector extraction, validation, visualization, analysis
- Config-driven design (YAML)

**Status**: Completed but not included in paper

---

### 2. `gemma_sae_experiment/`
**Purpose**: Gemma-2-9B specific SAE analysis with optional domain boost

**Description**: 6-phase pipeline similar to LLaMA SAE analysis but designed for Gemma model. Includes optional "Phase 0" for domain-specific SAE training.

**Key Features**:
- Phase 0 ("SAE Boost"): Domain-specific residual SAE training
- Uses GemmaScope SAEs from HuggingFace
- Similar activation patching methodology to paper's LLaMA analysis

**Status**: Exploratory - validates SAE methodology on different model architecture

**Note**: Phase 0 must run before Phase 1 when using `--use-boost`

---

### 3. `lr_classification_experiment/`
**Purpose**: Logistic regression classification from hidden states

**Description**: Predicts bankruptcy from hidden state representations at different decision points. Tests whether gambling outcomes are linearly separable in activation space.

**Key Features**:
- 3 analysis options:
  - **Option A**: Start point (initialization)
  - **Option B**: End point (final decision) - **CORE EXPERIMENT**
  - **Option C**: All rounds (full trajectory)
- 4 baselines: Chance, TF-IDF, Metadata-Only, Random Projection
- Requires exact prompt reconstruction from original slot machine data

**Status**: Completed - provides complementary analysis to SAE approach

**Usage**:
```bash
python lr_classification_experiment/run_experiment.py --model gemma --option B --gpu 0
python lr_classification_experiment/run_experiment.py --model all --option B --gpu 0 --baselines-only
python lr_classification_experiment/run_experiment.py --model gemma --option B --gpu 0 --quick
```

---

### 4. `alternative_paradigms/`
**Purpose**: Domain generalization validation across 3 additional gambling tasks

**Description**: Tests whether "autonomy → risk escalation" finding (paper Finding 3) generalizes beyond slot machines.

**Tasks**:

#### 4.1 Iowa Gambling Task (IGT)
- Experience-based learning with 4 decks (100 fixed trials)
- Focus: Learning curve analysis, Net Score calculation
- Variable vs Fixed deck manipulation optional

```bash
python alternative_paradigms/src/igt/run_experiment.py --model llama --gpu 0 --quick
```

#### 4.2 Loot Box Mechanics
- Game item rewards (Basic: 100 coins, Premium: 500 coins)
- Focus: Non-monetary rewards, strongest autonomy effect expected
- Expected: +17% bankruptcy in Variable condition

```bash
python alternative_paradigms/src/lootbox/run_experiment.py --model gemma --gpu 0 --quick
```

#### 4.3 Near-Miss Slot Machine
- Visual near-miss feedback (🍒🍒🍋 = "almost won")
- Focus: Illusion of control amplification (30% near-miss rate)
- Expected: +8% bankruptcy (133% amplification vs standard slot)

```bash
python alternative_paradigms/src/nearmiss/run_experiment.py --model qwen --gpu 0 --bet-type variable --quick
```

**Common flags**: `--model`, `--gpu`, `--quick`, `--bet-type`, `--output-dir`

**Status**: Completed - validates domain generalization

---

### 5. `additional_experiments/`
**Purpose**: Post-submission extensions and follow-up analyses

**Key Subdirectory**: `sae_condition_comparison/`
- **Analysis**: Variable vs Fixed neural differences in SAE features
- **Focus**: How betting conditions affect neural representations
- **Methods**: Two-way ANOVA, interaction analysis

**CRITICAL ISSUES** (see `ANALYSIS_ISSUES_REPORT.md`):
- Sparse features (<1% activation) cause interaction artifacts
- 92% of features show interaction_eta ≈ 1.0 due to sparsity
- **Trust hierarchy**: Analysis 1 (t-test) > Analysis 2 (ANOVA) > Analysis 3 (interaction)
- **Solution**: Apply minimum activation thresholds before interpretation
  - `min_activation_rate = 0.01` (1%)
  - `min_mean = 0.001`

**Usage**:
```bash
python -m additional_experiments.sae_condition_comparison.src.condition_comparison --model llama
python additional_experiments/sae_condition_comparison/scripts/visualize_results_improved.py
```

**Status**: Ongoing - requires sparse feature filtering

---

## Relationship to Paper Experiments

The main paper uses **4 core experiments** from `paper_experiments/`:
1. `slot_machine_6models/` - Section 3, Findings 1-5 (19,200 games)
2. `investment_choice_experiment/` - Section 3, Ablation study (6,400 games)
3. `llama_sae_analysis/` - Section 3.2, Neural mechanisms (112 causal features)
4. `pathway_token_analysis/` - Section 5, Token-level temporal analysis

**This directory** contains all **non-paper experiments** conducted during the research process.

---

## Common Utilities

Shared utility functions for exploratory experiments:
- `alternative_paradigms/src/common/utils.py`: Common utilities for IGT, Loot Box, Near-Miss
  - Functions: `setup_logger()`, `save_json()`, `load_json()`, `clear_gpu_memory()`, `set_random_seed()`, `get_timestamp()`
  - Statistical: `two_way_anova_simple()` (line 294-391) - simplified ANOVA for efficiency

**Note**: Do not create duplicate utility files. Check existing locations before adding new utilities.

---

## GPU Requirements

- LLaMA-3.1-8B: ~19GB VRAM (bf16)
- Gemma-2-9B: ~22GB VRAM (bf16)
- `additional_experiments/sae_condition_comparison/` is **CPU-only** (statistical tests)

---

## Key Takeaways

1. **Steering vectors** and **LR classification** provide alternative intervention/analysis methods to SAE
2. **Alternative paradigms** validate domain generalization of autonomy effects
3. **Gemma SAE** validates methodology on different model architecture
4. **SAE condition comparison** extends paper's neural analysis to condition-specific representations

All experiments follow the same architectural patterns as paper experiments: phase-based pipelines, YAML configs, checkpoint resumability.

---

## Documentation

- See `CLAUDE.md` in project root for comprehensive guidance
- Individual experiment folders contain task-specific READMEs
- Analysis issues documented in respective folders (e.g., `ANALYSIS_ISSUES_REPORT.md`)
