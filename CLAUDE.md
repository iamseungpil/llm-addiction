# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project (ICLR 2026 submission) studying addictive-like gambling behaviors in LLMs using slot machine and investment choice paradigms. Analyzes decision patterns across models (LLaMA-3.1-8B, Gemma-2-9B, GPT-4o-mini, GPT-4.1-mini, Claude-3.5-Haiku, Gemini-2.5-Flash) and uses Sparse Autoencoder (SAE) interpretability + activation patching to identify causal neural features driving risk-taking.

## Repository Structure

```
paper_experiments/              # Publication-ready experiments (4 paper sections)
├── slot_machine_6models/       # Section 3a: 6-model gambling behavior comparison
├── investment_choice_experiment/ # Section 3b: Structured 3-option choice paradigm
│   └── src/{analysis,models,irrationality_evaluation}/  # nested subdirectories
├── llama_sae_analysis/         # Section 4: SAE feature extraction + causal patching
└── pathway_token_analysis/     # Section 5: Temporal/linguistic token-level analysis

steering_vector_analysis/       # CAA-based steering vector experiments (5-phase pipeline)
gemma_sae_experiment/           # Gemma-specific SAE with domain boost (6-phase pipeline)
lr_classification_experiment/   # Hidden state → logistic regression classification
additional_experiments/         # Post-submission extensions (e.g., sae_condition_comparison)

legacy/                         # Archived experiments and analysis code
```

## Architecture

All experiments follow a **phase-based pipeline** pattern:

```
Config (YAML) → Phase Scripts (Python) → Results (JSON/NPZ/JSONL)
```

Each experiment directory uses: `src/` (phase implementations), `configs/` (YAML settings), `scripts/` (shell launchers), `results/` and `logs/` (outputs). Phases are independently resumable via checkpoint files.

**Key architectural patterns:**
- All paths and hyperparameters externalized to YAML configs
- GPU memory explicitly managed with `clear_gpu_memory()` calls between phases
- Models loaded in bf16 (not float16 or quantized—alters activations and breaks reproducibility)
- Open-weight models (LLaMA, Gemma) run locally; API models (GPT, Claude, Gemini) use client libraries

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

**Note**: There is no centralized package manager (pdm/pip) configuration. Dependencies are managed per-experiment basis through the conda environment.

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

### Additional Research Experiments

```bash
# Steering vector analysis (multi-GPU, CAA-based 5-phase pipeline)
bash steering_vector_analysis/scripts/launch_full_analysis_4gpu.sh
python steering_vector_analysis/src/extract_steering_vectors.py --config configs/experiment_config.yaml

# LR classification experiment (hidden state → bankruptcy prediction)
python lr_classification_experiment/run_experiment.py --model gemma --option B --gpu 0
python lr_classification_experiment/run_experiment.py --model all --option B --gpu 0 --baselines-only
python lr_classification_experiment/run_experiment.py --model gemma --option B --gpu 0 --quick

# Gemma SAE experiment (6-phase pipeline with optional domain boost)
python gemma_sae_experiment/run_pipeline.py --gpu 0 --phases all --use-boost

# Post-submission additional experiments (CPU-only statistical analysis)
python -m additional_experiments.sae_condition_comparison.src.condition_comparison --model llama
```

## Code Style and Conventions

- **Python version**: 3.11
- **Indentation**: 4 spaces
- **Line length**: 120 characters
- **String quotes**: Double quotes
- **Naming**: snake_case for modules/functions, PascalCase for classes
- **Paths**: Always use YAML configs or CLI flags, never hard-code paths
- **File extensions**: `.log` for logs, `.json` for data, `.npz` for hidden states
- **Type hints**: Required on shared utilities, optional on experiment scripts
- **Documentation**: Korean is acceptable for READMEs and design docs (bilingual project)

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

### Multi-GPU Experiments
- Use `CUDA_VISIBLE_DEVICES` to assign specific GPUs
- Steering vector analysis has 4-GPU launcher script
- Single experiments can specify `--gpu N` flag

## Notes

- Documentation is bilingual (English/Korean) - both are acceptable
- `.gitignore` excludes all experiment outputs; actual data stored in `/mnt/c/Users/oollccddss/git/data/llm-addiction/`
- No formal test suite - experiments include validation within their pipelines
- Legacy folder contains archived experiments from iterative development process
