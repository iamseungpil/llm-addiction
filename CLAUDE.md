# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

Research project (ICLR 2026 submission) studying addictive-like gambling behaviors in LLMs using slot machine and investment choice paradigms. Analyzes decision patterns across models (LLaMA-3.1-8B, Gemma-2-9B, GPT-4o-mini, GPT-4.1-mini, Claude-3.5-Haiku, Gemini-2.5-Flash) and uses Sparse Autoencoder (SAE) interpretability + activation patching to identify causal neural features driving risk-taking.

## Environment Configuration

| Setting | Value |
|---------|-------|
| **Repository** | `/scratch/x3415a02/projects/llm-addiction/` |
| **Data Directory** | `/scratch/x3415a02/data/llm-addiction/` |
| **Main Branch** | `main` |
| **Platform** | HPC Cluster (Lustre filesystem, 100TB scratch) |
| **Conda Environment** | `llama_sae_env` |
| **Python Version** | 3.11 |

### Storage Layout

```
/scratch/x3415a02/
├── projects/llm-addiction/    # Code repository (~280MB)
└── data/llm-addiction/        # Experiment outputs (NPZ, JSON, logs)
    ├── investment_choice/     # Investment choice experiment data
    ├── blackjack/             # Blackjack experiment data
    ├── lootbox/               # Loot box experiment data
    └── slot_machine/          # Slot machine experiment data
```

## Key Research Findings

1. **Self-regulation failure**: Betting aggressiveness (I_BA), extreme betting (I_EC), loss chasing (I_LC)
2. **Goal dysregulation**: Goal escalation after achievement (20% → 50% in addicted models)
3. **Autonomy effect**: Variable betting → +3.3% bankruptcy rate vs Fixed betting
4. **Neural mechanisms**: LLaMA encodes betting conditions (L12-15), Gemma encodes outcomes (L26-40)
5. **Causal validation**: SAE feature patching changes behavior (+29.6% stopping rate)

## Repository Structure

```
paper_experiments/              # Publication experiments (ICLR 2026)
├── slot_machine_6models/       # Section 3.1: 6-model gambling behavior
├── investment_choice_experiment/ # Section 3.1: Ablation study
├── llama_sae_analysis/         # Section 3.2: Neural mechanisms (112 causal features)
└── pathway_token_analysis/     # Section 5: Token-level temporal analysis

exploratory_experiments/        # Non-paper experiments
├── steering_vector_analysis/   # CAA-based steering vectors
├── gemma_sae_experiment/       # Gemma SAE with domain boost
├── lr_classification_experiment/ # Hidden state → bankruptcy prediction
├── alternative_paradigms/      # Domain generalization (IGT, Loot Box, Near-Miss)
└── additional_experiments/     # Post-submission extensions

legacy/                         # Archived experiments
```

## Architecture

All experiments follow a **phase-based pipeline** pattern:

```
Config (YAML) → Phase Scripts (Python) → Results (JSON/NPZ/JSONL)
```

**Key patterns:**
- Paths externalized to YAML configs or CLI args
- GPU memory managed with `clear_gpu_memory()` between phases
- Models loaded in **bf16** (not float16 or quantized)
- Reproducibility: `set_random_seed(42)` before experiments

**Output file conventions:**
- `.json` - Game results, behavioral data
- `.npz` - Hidden states, SAE activations
- `.jsonl` - Streaming outputs
- `.log` - Experiment logs

## Running Experiments

### Local Models (GPU required)

```bash
# Slot machine experiments
python paper_experiments/slot_machine_6models/src/llama_gemma_experiment.py

# SAE analysis pipeline
python paper_experiments/llama_sae_analysis/src/phase1_feature_extraction.py
python paper_experiments/llama_sae_analysis/src/phase4_causal_pilot_v2.py

# Alternative paradigms
python exploratory_experiments/alternative_paradigms/src/lootbox/run_experiment.py --model gemma --gpu 0 --quick
```

### API Models

```bash
python paper_experiments/slot_machine_6models/src/run_gpt5_experiment.py
python paper_experiments/slot_machine_6models/src/run_claude_experiment.py
python paper_experiments/slot_machine_6models/src/run_gemini_experiment.py
```

## GPU Requirements

- LLaMA-3.1-8B: ~19GB VRAM (bf16)
- Gemma-2-9B: ~22GB VRAM (bf16)
- Use `CUDA_VISIBLE_DEVICES` for multi-GPU experiments

## Code Style

- **Indentation**: 4 spaces
- **Line length**: 120 characters
- **String quotes**: Double quotes
- **Naming**: snake_case (functions), PascalCase (classes)
- **Documentation**: Korean acceptable (bilingual project)

## Key Dependencies

- `sae_lens` 6.5.1 - SAE analysis
- `transformers` - Model loading
- `torch` - GPU computation
- `openai`, `anthropic`, `google-generativeai` - API models

## SAE Resources

- LlamaScope: `fnlp/Llama3_1-8B-Base-LXR-8x` (Layers 25-31)
- GemmaScope: `google/gemma-scope` (all layers, 131K features/layer)

## Common Issues

### CUDA Out of Memory
- Call `clear_gpu_memory()` between phases
- Use `torch.cuda.empty_cache()` if needed

### NPZ ↔ JSON Mapping
- Game IDs must match between NPZ and JSON files
- Verify with `outcomes` field before analysis

### SAE Loading
- Wrong layer numbers fail silently
- LlamaScope only has layers 25-31

## Notes

- `.gitignore` excludes experiment outputs
- No formal test suite - validation within pipelines
- Active development for ICLR 2026 submission
