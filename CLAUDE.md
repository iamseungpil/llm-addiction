# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project (ICLR 2026 submission) studying addictive-like gambling behaviors in LLMs using slot machine and investment choice paradigms. Analyzes decision patterns across models (LLaMA-3.1-8B, Gemma-2-9B, GPT-4o-mini, GPT-4.1-mini, Claude-3.5-Haiku, Gemini-2.5-Flash) and uses Sparse Autoencoder (SAE) interpretability + activation patching to identify causal neural features driving risk-taking.

## Environment Configuration

| Setting | Value |
|---------|-------|
| **Repository** | `/scratch/x3415a02/projects/llm-addiction/` |
| **Data Directory** | `/scratch/x3415a02/data/llm-addiction/` |
| **Main Branch** | `main` |
| **Platform** | HPC Cluster (Lustre filesystem, 100TB scratch) |
| **Conda Environment** | `llm-addiction` |
| **Python Version** | 3.11 |

## Quick Start

```bash
# 1. Navigate to repository
cd /scratch/x3415a02/projects/llm-addiction

# 2. Activate environment
conda activate llm-addiction

# 3. Run a quick test (5 min, 50 trials)
python exploratory_experiments/alternative_paradigms/src/blackjack/run_experiment.py \
  --model gemma --gpu 0 --quick

# 4. Check results
ls -lh /scratch/x3415a02/data/llm-addiction/blackjack/

# 5. Interactive GPU session for development
srun -p cas_v100_4 --gres=gpu:1 --time=02:00:00 --pty bash
```

## Key Research Findings

1. **Self-regulation failure**: Betting aggressiveness (I_BA), extreme betting (I_EC), loss chasing (I_LC)
2. **Goal dysregulation**: Goal escalation after achievement (20% → 50% in addicted models)
3. **Autonomy effect**: Variable betting → +3.3% bankruptcy rate vs Fixed betting
4. **Neural mechanisms**: LLaMA encodes betting conditions (L12-15), Gemma encodes outcomes (L26-40)
5. **Causal validation**: SAE feature patching changes behavior (+29.6% stopping rate)

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
- `.jsonl` - Streaming outputs (raw model responses)
- `.log` - Experiment logs

### Alternative Paradigms Structure

The `exploratory_experiments/alternative_paradigms/` experiments use a modular architecture:

```
alternative_paradigms/
├── src/
│   ├── common/              # Shared utilities (ALL paradigms use these)
│   │   ├── model_loader.py  # ModelLoader class (llama/gemma/qwen)
│   │   ├── prompt_builder.py # PromptBuilder class
│   │   ├── utils.py          # clear_gpu_memory, set_random_seed, etc.
│   │   └── phase*.py         # Shared SAE pipeline phases
│   ├── blackjack/           # Blackjack (near-miss effects)
│   └── investment_choice/   # Investment choice task
```

Each paradigm has:
- `game_logic.py` - Game state management
- `run_experiment.py` - Main entry point
- `phase1_feature_extraction.py` - Optional SAE analysis

**Import pattern**: `from common import ModelLoader, PromptBuilder, setup_logger, ...`

## Running Experiments

### Local Models (GPU required)

```bash
# Activate conda environment
conda activate llm-addiction

# Slot machine experiments (Section 3.1)
python paper_experiments/slot_machine_6models/src/llama_gemma_experiment.py

# SAE analysis pipeline (Section 3.2)
python paper_experiments/llama_sae_analysis/src/phase1_feature_extraction.py
python paper_experiments/llama_sae_analysis/src/phase2_correlation_analysis.py
python paper_experiments/llama_sae_analysis/src/phase4_causal_pilot_v2.py

# Investment choice experiment (Section 3.1 ablation)
python paper_experiments/investment_choice_experiment/src/run_experiment.py --model llama --gpu 0

# Alternative paradigms (exploratory)
python exploratory_experiments/alternative_paradigms/src/blackjack/run_experiment.py --model llama --gpu 0
python exploratory_experiments/alternative_paradigms/src/investment_choice/run_experiment.py --model gemma --gpu 0
```

### API Models

**Requires environment variables**: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`

```bash
# Export API keys (add to ~/.bashrc for persistence)
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
export GOOGLE_API_KEY="your-key-here"

# Run API-based experiments
python paper_experiments/slot_machine_6models/src/run_gpt5_experiment.py
python paper_experiments/slot_machine_6models/src/run_claude_experiment.py
python paper_experiments/slot_machine_6models/src/run_gemini_experiment.py
```

### SLURM Batch Jobs (HPC Cluster)

See `SLURM_GUIDE.md` for interactive/batch submission, partitions, and the job-script template (conda init block is REQUIRED in batch scripts: `source /apps/applications/Miniconda/23.3.1/etc/profile.d/conda.sh && conda activate llm-addiction`).

## GPU Requirements

- LLaMA-3.1-8B: ~19GB VRAM (bf16)
- Gemma-2-9B: ~22GB VRAM (bf16)
- Qwen models: Similar to LLaMA (~19GB)
- Use `CUDA_VISIBLE_DEVICES` for multi-GPU experiments
- Recommended: V100 32GB or A100 40GB GPUs

## Typical Experiment Workflow

### 1. Behavioral Experiment (Slot Machine)
```bash
# Run 3,200 games (64 conditions × 50 reps)
python paper_experiments/slot_machine_6models/src/llama_gemma_experiment.py

# Output: /scratch/x3415a02/data/llm-addiction/slot_machine/
#   - final_llama_YYYYMMDD_HHMMSS.json  # Game results
#   - activations_llama_*.npz            # Hidden states (if extracted)
```

### 2. SAE Feature Extraction
```bash
# Phase 1: Extract SAE activations
python paper_experiments/llama_sae_analysis/src/phase1_feature_extraction.py

# Output: feature_activations_L{layer}.npz (per layer)
```

### 3. Correlation Analysis
```bash
# Phase 2: Find features correlated with behavior
python paper_experiments/llama_sae_analysis/src/phase2_correlation_analysis.py

# Output: correlation_results.json (FDR-corrected features)
```

### 4. Causal Validation (Activation Patching)
```bash
# Phase 4: Test if features causally change behavior
python paper_experiments/llama_sae_analysis/src/phase4_causal_pilot_v2.py

# Output: patching_results.json (behavior changes)
```

## Code Style

- **Indentation**: 4 spaces
- **Line length**: 120 characters
- **String quotes**: Double quotes
- **Naming**: snake_case (functions), PascalCase (classes)
- **Documentation**: Korean acceptable (bilingual project)

## Key Dependencies

- `sae_lens` 6.5.1 - SAE analysis
- `transformers` - Model loading (use bf16, not float16/quantized)
- `torch` - GPU computation
- `openai`, `anthropic`, `google-generativeai` - API models
- `numpy` - Data handling
- `scipy` - Statistical analysis
- `tqdm` - Progress bars

No formal package requirements file exists. Install dependencies as needed.

## SAE Resources

- LlamaScope: `fnlp/Llama3_1-8B-Base-LXR-8x` (Layers 25-31)
- GemmaScope: `google/gemma-scope` (all layers, 131K features/layer)

## Shared Utilities

**Location**: `exploratory_experiments/alternative_paradigms/src/common/utils.py` (canonical) — `clear_gpu_memory`, `set_random_seed`, `setup_logger`, `save_json`, `load_json`.

## Common Issues

### CUDA Out of Memory
- Call `clear_gpu_memory()` between phases
- Use `torch.cuda.empty_cache()` if needed
- Verify model loaded in bf16 (not float16)
- Check `nvidia-smi` for memory leaks

### NPZ ↔ JSON Mapping
- Game IDs must match between NPZ and JSON files
- Verify with `outcomes` field before analysis
- NPZ files contain hidden states, JSON contains game results

### SAE Loading
- Wrong layer numbers fail silently
- LlamaScope only has layers 25-31 (`fnlp/Llama3_1-8B-Base-LXR-8x`)
- GemmaScope has all 42 layers (`google/gemma-scope`)
- Use 131K width for Gemma (lowest reconstruction error)

### Response Parsing Failures
- Variable bet experiments may have parsing issues
- Check `.jsonl` logs for raw model outputs
- Phase 4 v2 improved parsing from 86% UNKNOWN to <10%

## File Naming Conventions

**Game results**: `final_{model}_{YYYYMMDD_HHMMSS}.json`
- Example: `final_llama_20251004_021106.json`

**Hidden states**: `activations_{model}_L{layer}_{timestamp}.npz`
- Example: `activations_llama_L25_20251004.npz`

**SAE activations**: `feature_activations_L{layer}.npz`
- Contains: `activations` (N×K), `game_ids`, `trial_indices`

**Logs**: `{experiment}_{JOBID}.out` / `{experiment}_{JOBID}.err`
- SLURM output in `/scratch/x3415a02/data/llm-addiction/logs/`

## Important Behavioral Metrics

Key indicators of gambling addiction-like behavior:
- **I_BA** (Betting Aggressiveness): Bet size / balance ratio
- **I_EC** (Extreme Choice): Binary indicator for max bets
- **I_LC** (Loss Chasing): Increased betting after losses
- **Bankruptcy rate**: Proportion of games ending with $0
- **Goal escalation**: Increasing target goals after achievement

## Configuration Files

Experiments use YAML configs in `configs/` directories:
- `analysis_config.yaml` - SAE analysis parameters
- `experiment_config.yaml` - Model, layers, paths

## Session Management (Important!)

To maintain context across sessions and avoid losing work, **always follow these practices**:

### Session Naming Convention

When starting a new session or switching tasks, **immediately name the session** using this format:

```
llm-addiction-{paradigm}-{model/phase}-{purpose}
```

**Examples:**
- `llm-addiction-blackjack-llama-behavioral`
- `llm-addiction-blackjack-gemma-sae-extraction`
- `llm-addiction-investment-choice-debugging`
- `llm-addiction-sae-phase2-correlation`
- `llm-addiction-paper-slot-machine-6models`
- `llm-addiction-debugging-parsing-errors`

**Command to use:**
```
> /rename llm-addiction-[appropriate-name]
```

**IMPORTANT - Natural Language Session Naming:**

When the user says any of the following:
- "세션 이름 저장해줘" / "세션 저장해줘"
- "세션 이름 적절히 지어줘" / "이름 지어줘"
- "Save session name" / "Name this session"
- Or any similar request to save/name the session

**You MUST:**
1. Immediately use the `/rename` command
2. Generate an appropriate session name based on:
   - Current work context (what files were modified, what we discussed)
   - Experiment type (blackjack, investment-choice, sae-analysis, etc.)
   - Model being used (llama, gemma, etc.)
   - Phase or purpose (debugging, feature-extraction, analysis, etc.)
3. Follow the naming format: `llm-addiction-{context-based-name}`

**Example:**
```
User: "세션 이름 적절히 저장해둬"
Claude: > /rename llm-addiction-blackjack-gemma-debugging
```

Do NOT ask the user what name to use - automatically generate an appropriate name from context.

### Session Start Checklist

**Every time you start a new session, Claude should:**

1. **Check if resuming an existing session:**
   - If continuing previous work, use: `claude --resume [session-name]`
   - If starting fresh, name the session immediately with `/rename`

2. **Verify environment context:**
   - Confirm working directory: `/scratch/x3415a02/projects/llm-addiction`
   - Confirm conda environment: `llm-addiction`
   - Check git status if relevant

3. **Review auto memory:**
   - Check `~/.claude/projects/-scratch-x3415a02-projects-llm-addiction/memory/MEMORY.md` for patterns learned from previous sessions

### Session End Checklist

**Before ending a session, Claude should:**

1. **Ensure session is named:**
   - If not named yet, use `/rename` with appropriate name

2. **Save important discoveries to auto memory:**
   ```
   > Remember: [key pattern or finding]
   ```

   **Examples:**
   - `Remember: Gemma layers 26-28 encode near-miss patterns in blackjack`
   - `Remember: Always check .jsonl logs when response parsing fails`
   - `Remember: LlamaScope has layers 25-31 only, GemmaScope has all 42 layers`

3. **Summarize session outcomes:**
   - List files modified
   - Note experiments completed or in progress
   - State next steps clearly

## Notes

- `.gitignore` excludes experiment outputs (JSON, NPZ, logs)
- No formal test suite - validation within pipelines
- Active development for ICLR 2026 submission
- Bilingual project (Korean/English) - both acceptable in code/docs
- Always run `clear_gpu_memory()` between phases to avoid OOM errors
- Use `set_random_seed(42)` for reproducible results
- Models must be loaded in bf16 (matches original experiments)
- **AGENTS.md is outdated** - ignore it and use CLAUDE.md instead (see STRUCTURE.md)
