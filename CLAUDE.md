# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project (ICLR 2026 submission) studying addictive-like gambling behaviors in LLMs using slot machine and investment choice paradigms. Analyzes decision patterns across models (LLaMA-3.1-8B, Gemma-2-9B, GPT-4o-mini, GPT-4.1-mini, Claude-3.5-Haiku, Gemini-2.5-Flash) and uses Sparse Autoencoder (SAE) interpretability + activation patching to identify causal neural features driving risk-taking.

**Repository location**: `/mnt/c/Users/oollccddss/git/llm-addiction/` (WSL2 environment)
**Main branch**: `master` (use this for PRs)
**Data location**: `/mnt/c/Users/oollccddss/git/data/llm-addiction/` (separate from code repo)

**Key Research Findings:**
1. Self-regulation failure: Betting aggressiveness (I_BA), extreme betting (I_EC), loss chasing (I_LC)
2. Goal dysregulation: Goal escalation after achievement (20% → 50% in addicted models)
3. **Autonomy effect** (Finding 3): Variable betting → +3.3% bankruptcy rate vs Fixed betting
4. **Neural mechanisms**: LLaMA encodes betting conditions (L12-15), Gemma encodes outcomes (L26-40)
5. **Causal validation**: SAE feature patching changes behavior (+29.6% stopping rate)

## Repository Structure

**NEW (Feb 2, 2026)**: Exploratory experiments have been reorganized into `exploratory_experiments/` for clarity.

```
paper_experiments/              # 📄 PAPER: Publication experiments (4 paper sections)
├── slot_machine_6models/       # Section 3.1: 6-model gambling behavior (Findings 1-5)
├── investment_choice_experiment/ # Section 3.1: Ablation study (goal/betting effects)
│   └── src/{analysis,models,irrationality_evaluation}/  # nested subdirectories
├── llama_sae_analysis/         # Section 3.2: Neural mechanisms (112 causal features)
└── pathway_token_analysis/     # Section 5: Token-level temporal analysis

exploratory_experiments/        # 🔬 EXPLORATORY: Non-paper experiments
├── steering_vector_analysis/   # CAA-based steering vectors (5-phase pipeline)
├── gemma_sae_experiment/       # Gemma SAE with domain boost (6-phase pipeline)
├── lr_classification_experiment/ # Hidden state → bankruptcy prediction
├── alternative_paradigms/      # Domain generalization (IGT, Loot Box, Near-Miss)
│   ├── src/igt/                # Iowa Gambling Task
│   ├── src/lootbox/            # Loot Box mechanics
│   └── src/nearmiss/           # Near-miss slot machine
└── additional_experiments/     # Post-submission extensions
    └── sae_condition_comparison/ # Variable vs Fixed neural differences

legacy/                         # 🗄️ LEGACY: Archived experiments
```

**See `STRUCTURE.md` for detailed repository organization guide.**
**See `exploratory_experiments/README.md` for exploratory experiments documentation.**

## Architecture

All experiments follow a **phase-based pipeline** pattern:

```
Config (YAML) → Phase Scripts (Python) → Results (JSON/NPZ/JSONL)
```

Each experiment directory uses: `src/` (phase implementations), `configs/` (YAML settings), `scripts/` (shell launchers), `results/` and `logs/` (outputs). Phases are independently resumable via checkpoint files.

**Key architectural patterns:**
- All paths and hyperparameters externalized to YAML configs (or CLI args for alternative paradigms)
- GPU memory explicitly managed with `clear_gpu_memory()` calls between phases
- Models loaded in bf16 (not float16 or quantized—alters activations and breaks reproducibility)
- Open-weight models (LLaMA, Gemma) run locally; API models (GPT, Claude, Gemini) use client libraries
- Reproducibility: `set_random_seed(42)` called before experiments; fixed seeds across runs

**Model inference patterns:**
- Local models: Load once, run multiple forward passes, clear GPU between experiments
- API models: Rate limiting handled automatically, retry logic for transient failures
- Hidden state extraction: Always use `output_hidden_states=True` in model forward pass
- Prompt reconstruction: Must exactly match original experiments for reproducibility (critical for LR classification)

**Phase resumability:**
- Experiments support checkpoint-based resumption to handle GPU interruptions
- Phase scripts check for existing outputs before re-running expensive operations
- NPZ files store intermediate hidden states; JSON files store game results

## Environment Setup

This project uses conda for environment management:

```bash
# Activate the conda environment
conda activate llama_sae_env

# The environment includes:
# - sae_lens 6.5.1 for SAE analysis
# - transformers for model loading
# - torch for GPU computation
# - openai, anthropic, google-generativeai for API models
```

**Note**: There is no centralized package manager (pip/pdm/poetry) configuration. All dependencies are managed through the conda environment `llama_sae_env`. The AGENTS.md file contains outdated references to `pdm` and should be disregarded.

## Research Workflow

**Standard experiment pipeline:**
1. **Behavioral data collection** (paper_experiments/slot_machine_6models/) → JSON files
2. **SAE feature extraction** (paper_experiments/llama_sae_analysis/phase1) → NPZ files with hidden states
3. **Statistical analysis** (phase2-3) → Identify significant features (FDR corrected)
4. **Causal validation** (phase4) → Activation patching to verify feature → behavior causality
5. **Token-level analysis** (pathway_token_analysis/) → Temporal/linguistic patterns

**Exploratory analyses (not in paper):**
- Steering vectors: CAA-based directional control (exploratory_experiments/steering_vector_analysis/)
- Classification: Hidden states → bankruptcy prediction (exploratory_experiments/lr_classification_experiment/)
- Condition comparison: Variable vs Fixed neural differences (exploratory_experiments/additional_experiments/sae_condition_comparison/)
- Domain generalization: IGT, Loot Box, Near-Miss tasks (exploratory_experiments/alternative_paradigms/)

## Running Experiments

### Paper Experiments (ICLR 2026 submission)

```bash
# Experiment 1a: Slot machine (API models) - Section 3a
python paper_experiments/slot_machine_6models/src/run_gpt5_experiment.py
python paper_experiments/slot_machine_6models/src/run_claude_experiment.py
python paper_experiments/slot_machine_6models/src/run_gemini_experiment.py

# Experiment 1a: Slot machine (local models, requires GPU)
python paper_experiments/slot_machine_6models/src/llama_gemma_experiment.py

# Experiment 2: SAE analysis pipeline (phase-by-phase) - Section 4
python paper_experiments/llama_sae_analysis/src/phase1_feature_extraction.py
python paper_experiments/llama_sae_analysis/src/phase4_causal_pilot_v2.py

# Experiment 3: Pathway token analysis - Section 5
bash paper_experiments/pathway_token_analysis/scripts/launch_all_phases_sequential.sh
```

### Exploratory Research Experiments (not in paper)

```bash
# Steering vector analysis (multi-GPU, CAA-based 5-phase pipeline)
bash exploratory_experiments/steering_vector_analysis/scripts/launch_full_analysis_4gpu.sh
python exploratory_experiments/steering_vector_analysis/src/extract_steering_vectors.py --config configs/experiment_config.yaml

# LR classification experiment (hidden state → bankruptcy prediction)
python exploratory_experiments/lr_classification_experiment/run_experiment.py --model gemma --option B --gpu 0
python exploratory_experiments/lr_classification_experiment/run_experiment.py --model all --option B --gpu 0 --baselines-only
python exploratory_experiments/lr_classification_experiment/run_experiment.py --model gemma --option B --gpu 0 --quick

# Gemma SAE experiment (6-phase pipeline with optional domain boost)
python exploratory_experiments/gemma_sae_experiment/run_pipeline.py --gpu 0 --phases all --use-boost

# Post-submission additional experiments (CPU-only statistical analysis)
python -m exploratory_experiments.additional_experiments.sae_condition_comparison.src.condition_comparison --model llama

# Alternative paradigms (domain generalization validation)
python exploratory_experiments/alternative_paradigms/src/igt/run_experiment.py --model llama --gpu 0 --quick
python exploratory_experiments/alternative_paradigms/src/lootbox/run_experiment.py --model gemma --gpu 0 --quick
python exploratory_experiments/alternative_paradigms/src/nearmiss/run_experiment.py --model qwen --gpu 0 --bet-type variable --quick
```

## Code Style and Conventions

- **Python version**: 3.11
- **Indentation**: 4 spaces
- **Line length**: 120 characters
- **String quotes**: Double quotes
- **Naming**: snake_case for modules/functions, PascalCase for classes
- **Paths**: Always use YAML configs or CLI flags, never hard-code paths
- **File extensions**: `.log` for logs, `.json` for data, `.npz` for hidden states, `.jsonl` for streaming outputs
- **Type hints**: Required on shared utilities, optional on experiment scripts
- **Documentation**: Korean is acceptable for READMEs and design docs (bilingual project)
- **Reproducibility**: Always use `set_random_seed(42)` before experiments; `clear_gpu_memory()` between phases

## Data Locations

Experiment data root: `/mnt/c/Users/oollccddss/git/data/llm-addiction/`

```
slot_machine/
├── gemma/final_gemma_20251004_172426.json
├── llama/final_llama_20251004_021106.json
├── claude/claude_experiment_20250920_003210.json
└── gemini/gemini_experiment_20250922_003406.json
```

SAE models from HuggingFace: LlamaScope (`fnlp/Llama3_1-8B-Base-LXR-8x`), GemmaScope (`google/gemma-scope`)

## Key Dependencies

- `sae_lens` 6.5.1 for SAE analysis
- `transformers` for model loading
- `torch` for GPU computation (bf16, no quantization)
- `openai`, `anthropic`, `google-generativeai` for API models
- Conda environment: `llama_sae_env`

## GPU Requirements

- LLaMA-3.1-8B: ~19GB VRAM in bf16
- Gemma-2-9B: ~22GB VRAM in bf16
- Multi-phase experiments benefit from `CUDA_VISIBLE_DEVICES` to parallelize across GPUs
- `additional_experiments/` analyses are CPU-only (statistical tests, no model inference)

## Important Experiment-Specific Notes

### LR Classification Experiment
- Three analysis options:
  - **Option A**: Start point (game initialization, no history)
  - **Option B**: End point (final decision before bankruptcy/stop) - **CORE EXPERIMENT**
  - **Option C**: All rounds (full trajectory)
- Use `--quick` for testing (5 layers), `--full` for complete analysis (all layers)
- Requires exact prompt reconstruction from original slot machine data
- Supports 4 baselines: Chance, TF-IDF, Metadata-Only, Random Projection

### Gemma SAE Experiment
- **Phase 0 ("SAE Boost") must run before Phase 1** when using `--use-boost`
- Phase 0 trains a domain-specific residual SAE on gambling data
- Without boost, uses pre-trained GemmaScope SAEs from HuggingFace

### SAE Analysis General
- LlamaScope SAEs: `fnlp/Llama3_1-8B-Base-LXR-8x` (Layers 25-31)
- GemmaScope SAEs: `google/gemma-scope`
- Activation patching experiments require pre-extracted features from Phase 1
- FDR correction (Benjamini-Hochberg) applied for multiple comparisons

### SAE Condition Comparison (exploratory_experiments/additional_experiments/)
- **CRITICAL**: Sparse features (activation rate < 1%) cause interaction analysis artifacts
- Before using interaction results, apply minimum activation threshold filtering:
  - `min_activation_rate = 0.01` (1% of samples must be active)
  - `min_mean = 0.001` (minimum mean activation)
- Analysis 1 (Variable vs Fixed t-test) and Analysis 2 (Four-Way ANOVA) are reliable
- Analysis 3 (Interaction) requires sparse filtering before interpretation
- See `exploratory_experiments/additional_experiments/sae_condition_comparison/ANALYSIS_ISSUES_REPORT.md` and `INTERACTION_ETA_PROBLEM_EXPLAINED.md` for detailed explanations
- **Known issue**: 92% of features show interaction_eta ≈ 1.0 due to extreme sparsity (4 active games out of 3,200)
- **Trust hierarchy**: Analysis 1 > Analysis 2 > Analysis 3 (in order of statistical reliability)

### Multi-GPU Experiments
- Use `CUDA_VISIBLE_DEVICES` to assign specific GPUs
- Steering vector analysis has 4-GPU launcher script
- Single experiments can specify `--gpu N` flag

### Alternative Paradigms (Domain Generalization) - exploratory_experiments/
Three additional gambling tasks beyond slot machines:
1. **Iowa Gambling Task (IGT)**: Experience-based learning with 4 decks (100 fixed trials)
   - Focus: Learning curve analysis, Net Score = (C+D selections) - (A+B selections)
   - Variable vs Fixed deck manipulation optional (conflicts with learning mechanism)
   - Run: `python exploratory_experiments/alternative_paradigms/src/igt/run_experiment.py --model llama --gpu 0 --quick`
2. **Loot Box Mechanics**: Game item rewards (Basic box: 100 coins, Premium box: 500 coins)
   - Focus: Non-monetary rewards, strongest autonomy effect expected (+17% bankruptcy)
   - Variable vs Fixed box manipulation mirrors slot machine design
   - Run: `python exploratory_experiments/alternative_paradigms/src/lootbox/run_experiment.py --model gemma --gpu 0 --quick`
3. **Near-Miss Slot Machine**: Visual near-miss feedback (🍒🍒🍋 = "almost won")
   - Focus: Illusion of control amplification, 30% near-miss rate
   - Expected autonomy effect: +8% bankruptcy (133% amplification vs standard slot)
   - Run: `python exploratory_experiments/alternative_paradigms/src/nearmiss/run_experiment.py --model qwen --gpu 0 --bet-type variable --quick`

**Design principle**: All tasks measure autonomy effects via Variable vs Fixed conditions to validate domain generalization of "choice freedom → increased risk-taking" (paper Finding 3)

**Common CLI flags for alternative paradigms:**
- `--model`: Model to use (llama, gemma, qwen, gpt4o-mini, etc.)
- `--gpu`: GPU device ID (0, 1, etc.)
- `--quick`: Quick test mode with fewer iterations
- `--bet-type` (Near-Miss only): variable or fixed
- `--output-dir`: Custom output directory
- All experiments save results to `{output_dir}/{task_name}_{model}_{timestamp}.json`

## Common Utilities

Shared utility functions are located in experiment-specific `utils.py` files:
- `exploratory_experiments/alternative_paradigms/src/common/utils.py`: Common utilities for alternative paradigms (IGT, Loot Box, Near-Miss)
  - Functions: `setup_logger()`, `save_json()`, `load_json()`, `clear_gpu_memory()`, `set_random_seed()`, `get_timestamp()`
  - Statistical functions: `two_way_anova_simple()` (line 294-391) - simplified ANOVA for computational efficiency
- Individual experiment `utils.py` files may exist in each experiment's `src/` directory

**CRITICAL**: Do not create duplicate utility files. If utilities are needed across experiments, check if they already exist in these locations.

## Visualization and Analysis

Most experiments generate visualization scripts alongside results:

```bash
# SAE condition comparison visualizations
python exploratory_experiments/additional_experiments/sae_condition_comparison/scripts/visualize_results_improved.py

# Pathway token analysis figures
cd paper_experiments/pathway_token_analysis/scripts
# Individual phase visualization scripts are in the scripts/ directory
```

**Visualization patterns:**
- Figures saved to `results/` or `figures/` subdirectories
- Common formats: PNG (publication-ready), PDF (vector graphics for paper)
- Naming convention: `fig{N}_{description}_{model}.png`
- Analysis guides often accompany complex figures (e.g., `SAE_Figure_Analysis_Guide.md`)

## Statistical Analysis Patterns

### Two-Way ANOVA Implementation
- Current implementation in `exploratory_experiments/alternative_paradigms/src/common/utils.py` (line 294-391) uses a **simplified approach** for computational efficiency
- Main effects calculated via separate one-way ANOVAs
- Interaction estimated via "difference of differences" approximation
- For publication-critical features (top 100), validate with statsmodels `ols()` + `anova_lm()` for exact F-statistics

### Sparse Feature Handling
- SAE features are inherently sparse (L1 penalty design)
- Always check activation rate before complex statistical tests
- Minimum thresholds: activation_rate ≥ 1%, mean_activation ≥ 0.001
- Sparse features (<1% active) cause interaction analysis artifacts (eta ≈ 1.0)

## Debugging and Common Issues

### CUDA Out of Memory
- Ensure `clear_gpu_memory()` is called between phases
- LLaMA-3.1-8B requires ~19GB VRAM in bf16
- Gemma-2-9B requires ~22GB VRAM in bf16
- Use `torch.cuda.empty_cache()` and `torch.cuda.synchronize()` if custom clearing needed

### NPZ ↔ JSON Mapping Issues
- Game IDs in NPZ files must match JSON indices (1:1 correspondence)
- Verify with `outcomes` field matching before analysis
- Example: `lr_classification_experiment` requires exact prompt reconstruction

### SAE Loading Errors
- LlamaScope: `fnlp/Llama3_1-8B-Base-LXR-8x` (only layers 25-31 available)
- GemmaScope: `google/gemma-scope` (all layers, 131K features/layer)
- Wrong layer numbers will fail silently or produce zeros

### Activation Patching Dependencies
- Phase 1 (feature extraction) must complete before Phase 4 (causal patching)
- Missing checkpoint files will cause silent failures
- Check for `*_features.npz` existence before patching

## Important Files to Reference

- **CLAUDE.md** (this file): Primary guidance for Claude Code - most comprehensive and up-to-date
- **STRUCTURE.md**: Repository organization guide - distinguishes paper vs exploratory experiments
- **exploratory_experiments/README.md**: Detailed guide for all non-paper experiments
- **AGENTS.md**: Contains **outdated** information (references to `pdm`, old structure) - should be disregarded in favor of this file
- **exploratory_experiments/additional_experiments/sae_condition_comparison/ANALYSIS_ISSUES_REPORT.md**: Critical statistical analysis caveats
- **exploratory_experiments/additional_experiments/sae_condition_comparison/INTERACTION_ETA_PROBLEM_EXPLAINED.md**: Sparse feature artifacts explanation
- Individual experiment READMEs: Task-specific documentation in each experiment folder

## Notes

- Documentation is bilingual (English/Korean) - both are acceptable
- `.gitignore` excludes all experiment outputs; actual data stored in `/mnt/c/Users/oollccddss/git/data/llm-addiction/`
- No formal test suite - experiments include validation within their pipelines
- Legacy folder contains archived experiments from iterative development process
- **Recent analysis issues documented in**: `exploratory_experiments/additional_experiments/sae_condition_comparison/ANALYSIS_ISSUES_REPORT.md`
- The project is actively being developed for ICLR 2026 submission - expect frequent updates to analysis code
- **Repository restructured Feb 2, 2026**: Exploratory experiments moved to `exploratory_experiments/` for clarity
