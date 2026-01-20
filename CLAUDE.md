# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project studying addictive-like gambling behaviors in LLMs using slot machine and investment choice paradigms. Analyzes decision patterns across models (LLaMA, Gemma, GPT-4o-mini, Claude, Gemini) and uses Sparse Autoencoder (SAE) interpretability techniques to identify causal neural features.

## Repository Structure

```
paper_experiments/           # Main experiments for ICLR 2026 submission
├── slot_machine_6models/    # Multi-model gambling experiments
├── investment_choice_experiment/ # Alternative paradigm with structured choices
├── llama_sae_analysis/      # SAE feature extraction and causal patching
└── pathway_token_analysis/  # Temporal and linguistic analysis

steering_vector_analysis/    # CAA-based steering vector experiments
├── src/                     # 5-phase pipeline implementation
├── configs/                 # YAML experiment configurations
└── scripts/                 # Launch scripts

legacy/                      # Archived experiments and analysis code
```

## Build and Development Commands

```bash
# Install dependencies (from Language-Model-SAEs directory)
pdm install

# Lint
pdm run ruff check src tests
pdm run ruff check --fix src tests  # auto-fix

# Format
pdm run ruff format

# Run tests
pdm run pytest tests/unit
pdm run pytest tests/unit -k "test_name"  # single test
```

## Running Experiments

```bash
# Slot machine experiments
python paper_experiments/slot_machine_6models/src/run_gpt5_experiment.py
python paper_experiments/slot_machine_6models/src/llama_gemma_experiment.py

# SAE analysis pipeline
python paper_experiments/llama_sae_analysis/src/phase1_feature_extraction.py
python paper_experiments/llama_sae_analysis/src/phase4_causal_pilot_v2.py

# Steering vector analysis
bash steering_vector_analysis/scripts/launch_full_analysis_4gpu.sh
python steering_vector_analysis/src/extract_steering_vectors.py --config configs/experiment_config.yaml
```

## Code Style

- Python 3.11, 4-space indents, 120-character lines, double quotes
- snake_case for modules/functions, PascalCase for classes
- Use configuration files or CLI flags over hard-coded paths
- Logs: `.log` suffix; Data: `.json` or `.csv`
- Type hints on shared utilities

## Data Locations

Experiment results typically stored in `/data/llm_addiction/` on the research server:
- LLaMA/Gemma raw data: `experiment_0_*_corrected/`
- GPT results: `gpt_results_fixed_parsing/`
- SAE models: HuggingFace (LlamaScope: `fnlp/Llama3_1-8B-Base-LXR-8x`, GemmaScope: `google/gemma-scope`)

## Key Dependencies

- `sae_lens` 6.5.1 for SAE analysis
- `transformers` for model loading
- `torch` for GPU computation
- Conda environment: `llama_sae_env`
