# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project (ICLR 2026 submission) studying addictive-like gambling behaviors in LLMs using slot machine and investment choice paradigms. Analyzes decision patterns across models (LLaMA-3.1-8B, Gemma-2-9B, GPT-4o-mini, GPT-4.1-mini, Claude-3.5-Haiku, Gemini-2.5-Flash) and uses Sparse Autoencoder (SAE) interpretability + activation patching to identify causal neural features driving risk-taking.

## Environment Configuration

| Setting | Value |
|---------|-------|
| **Repository** | `/home/jovyan/llm-addiction/` |
| **Data Directory** | `/home/jovyan/beomi/llm-addiction-data/` |
| **Main Branch** | `main` |
| **Development Branch** | `openhpc` (OpenHPC environment adaptation) |
| **Platform** | OpenHPC (Kubernetes/JupyterHub, GPU directly allocated) |
| **Python Version** | 3.13.11 (Anaconda) |
| **PyTorch** | 2.8.0+cu128 |

### GPU Hardware

| GPU | Model | VRAM | Status |
|-----|-------|------|--------|
| GPU 0 | NVIDIA A100-SXM4-40GB | 39.5GB | Available |
| GPU 1 | NVIDIA A100-SXM4-40GB | 39.5GB | Available |

**System**: 100 CPU cores, 1TB RAM, CUDA 12.9 (driver) / 12.8 (PyTorch)

### Storage Layout

```
/home/jovyan/
├── llm-addiction/                    # Code repository (this repo)
└── beomi/llm-addiction-data/         # Experiment outputs (NPZ, JSON, logs)
    ├── investment_choice/            # Investment choice experiment data
    ├── blackjack/                    # Blackjack experiment data
    ├── slot_machine/                 # Slot machine experiment data
    └── logs/                         # Experiment logs
```

### Key Differences from SLURM Environment

- **No SLURM**: GPU is directly allocated, no `sbatch`/`srun`/`squeue` needed
- **No conda init**: Environment is pre-activated in JupyterHub
- **Direct execution**: Run `python script.py` directly (no job submission)
- **A100 40GB × 2**: Upgraded from V100 32GB, can run larger models or parallel experiments

## Quick Start

```bash
# 1. Navigate to repository
cd /home/jovyan/llm-addiction

# 2. Run a quick test (5 min, 50 trials) - GPU already available
python exploratory_experiments/alternative_paradigms/src/blackjack/run_experiment.py \
  --model gemma --gpu 0 --quick

# 3. Check results
ls -lh /home/jovyan/beomi/llm-addiction-data/blackjack/

# 4. Use GPU 1 for parallel experiments
CUDA_VISIBLE_DEVICES=1 python another_experiment.py --gpu 0
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
├── alternative_paradigms/      # Domain generalization (IGT, Near-Miss)
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

### Running Long Experiments (OpenHPC)

GPU is directly allocated - run experiments directly without job submission:

```bash
# Run in foreground
python scripts/run_experiment.py --gpu 0

# Run in background with nohup
nohup python scripts/run_experiment.py --gpu 0 > /home/jovyan/beomi/llm-addiction-data/logs/experiment.log 2>&1 &

# Run on GPU 1 in parallel
CUDA_VISIBLE_DEVICES=1 nohup python another_experiment.py --gpu 0 > /home/jovyan/beomi/llm-addiction-data/logs/experiment2.log 2>&1 &

# Monitor logs
tail -f /home/jovyan/beomi/llm-addiction-data/logs/experiment.log
```

**Note**: Shell scripts in `scripts/` have SLURM headers commented out with `[SLURM-DISABLED]` tags. They can still be run with `bash script.sh`.

## GPU Requirements

- LLaMA-3.1-8B: ~19GB VRAM (bf16) - fits on single A100
- Gemma-2-9B: ~22GB VRAM (bf16) - fits on single A100
- Qwen models: Similar to LLaMA (~19GB)
- **Available**: 2× A100 40GB - can run two models simultaneously
- Use `CUDA_VISIBLE_DEVICES=0` or `CUDA_VISIBLE_DEVICES=1` to select GPU

## Typical Experiment Workflow

### 1. Behavioral Experiment (Slot Machine)
```bash
# Run 3,200 games (64 conditions × 50 reps)
python paper_experiments/slot_machine_6models/src/llama_gemma_experiment.py

# Output: /home/jovyan/beomi/llm-addiction-data/slot_machine/
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

All experiments use common utility functions:

```python
from common.utils import clear_gpu_memory, set_random_seed, setup_logger, save_json, load_json

# GPU memory management (call between phases)
clear_gpu_memory()

# Reproducibility (call before experiments)
set_random_seed(42)

# Logging
logger = setup_logger(__name__)
```

**Location**: `exploratory_experiments/alternative_paradigms/src/common/utils.py` (canonical)

## Debugging & Monitoring

```bash
# Check GPU usage
nvidia-smi

# Watch GPU in real-time
watch -n 1 nvidia-smi

# Check running processes
ps aux | grep python

# Monitor experiment logs
tail -f /home/jovyan/beomi/llm-addiction-data/logs/*.log

# Check background jobs
jobs -l
```

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

## Analysis Scripts

Post-experiment analysis typically involves:
1. Loading JSON results from `/home/jovyan/beomi/llm-addiction-data/`
2. Computing behavioral metrics (bankruptcy rate, bet patterns, loss chasing)
3. Statistical tests (FDR correction, Cohen's d, effect sizes)
4. Visualization (matplotlib/seaborn)

Analysis scripts are embedded in phase files (e.g., `phase2_correlation_analysis.py`).

## File Naming Conventions

**Game results**: `final_{model}_{YYYYMMDD_HHMMSS}.json`
- Example: `final_llama_20251004_021106.json`

**Hidden states**: `activations_{model}_L{layer}_{timestamp}.npz`
- Example: `activations_llama_L25_20251004.npz`

**SAE activations**: `feature_activations_L{layer}.npz`
- Contains: `activations` (N×K), `game_ids`, `trial_indices`

**Logs**: `{experiment}.log`
- Experiment logs in `/home/jovyan/beomi/llm-addiction-data/logs/`

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

**Key config fields**:
```yaml
data:
  experiment_file: /path/to/game_results.json
  output_dir: /path/to/output/

models:
  llama:
    name: meta-llama/Llama-3.1-8B
    layers: [25, 26, 27, 28, 29, 30, 31]  # LlamaScope layers

correlation:
  fdr_alpha: 0.05                   # FDR threshold
  min_cohens_d: 0.3                 # Effect size threshold
```

## Running Experiments (OpenHPC)

In the OpenHPC environment, run experiments directly:

```bash
# Navigate to repository
cd /home/jovyan/llm-addiction

# Run experiment on GPU 0
python your_experiment.py --gpu 0

# Run experiment on GPU 1 (parallel)
CUDA_VISIBLE_DEVICES=1 python your_experiment.py --gpu 0

# Background execution with logging
nohup python your_experiment.py --gpu 0 \
  > /home/jovyan/beomi/llm-addiction-data/logs/experiment_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

**Note**: Legacy SLURM shell scripts (`scripts/*.sh`) have been updated with `[SLURM-DISABLED]` comments. The actual Python commands in them still work when run with `bash script.sh`.

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
   - Confirm working directory: `/home/jovyan/llm-addiction`
   - Confirm GPU availability: `nvidia-smi`
   - Check git status if relevant

3. **Review auto memory:**
   - Check `~/.claude/projects/-home-jovyan-llm-addiction/memory/MEMORY.md` for patterns learned from previous sessions

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

### Auto Memory Usage

Claude should **proactively save** to auto memory when discovering:
- Recurring error patterns and their solutions
- OpenHPC environment-specific behaviors (GPU memory, Kubernetes quirks)
- Model-specific patterns (layer encodings, parsing issues)
- File path patterns and data locations
- Successful debugging approaches

**DO NOT save to memory:**
- Session-specific temporary state
- Information already in CLAUDE.md
- Incomplete or unverified findings

### Resuming Previous Sessions

**To resume a session:**

```bash
# Command line
claude --resume llm-addiction-blackjack-experiment

# Or interactively
claude --resume  # Opens session picker

# Continue most recent
claude --continue
```

**In session picker (keyboard shortcuts):**
- `↑`/`↓`: Navigate sessions
- `P`: Preview session content
- `R`: Rename session
- `/`: Search/filter sessions
- `Enter`: Open selected session

### Context Management

**To keep sessions focused:**
- Use `/clear` when switching to unrelated investigation
- Use subagents for exploratory research (keeps main context clean)
- Create separate sessions for distinct experiments/debugging tasks

### Recovery from Lost Sessions

If you forgot to name a session:
1. Use `claude --resume` to open session picker
2. Use `P` (preview) to find the right session by content
3. Use `R` (rename) to give it a proper name
4. Resume that session

## Notes

- `.gitignore` excludes experiment outputs (JSON, NPZ, logs)
- No formal test suite - validation within pipelines
- Active development for ICLR 2026 submission
- Bilingual project (Korean/English) - both acceptable in code/docs
- Always run `clear_gpu_memory()` between phases to avoid OOM errors
- Use `set_random_seed(42)` for reproducible results
- Models must be loaded in bf16 (matches original experiments)
- **AGENTS.md is outdated** - ignore it and use CLAUDE.md instead (see STRUCTURE.md)
