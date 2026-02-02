# Repository Reorganization Summary

**Date**: February 2, 2026
**Purpose**: Separate paper experiments from exploratory research for clarity

---

## Changes Made

### 1. New Directory Structure

Created `exploratory_experiments/` directory to contain all non-paper experiments:

```
llm-addiction/
├── paper_experiments/              # ✅ Experiments used in paper
│   ├── slot_machine_6models/
│   ├── investment_choice_experiment/
│   ├── llama_sae_analysis/
│   └── pathway_token_analysis/
│
├── exploratory_experiments/        # 🆕 NEW: Exploratory research
│   ├── steering_vector_analysis/
│   ├── gemma_sae_experiment/
│   ├── lr_classification_experiment/
│   ├── alternative_paradigms/
│   └── additional_experiments/
│
└── legacy/                         # Archived experiments
```

---

## Moved Directories

The following directories were moved from root to `exploratory_experiments/`:

1. ✅ `steering_vector_analysis/` → `exploratory_experiments/steering_vector_analysis/`
2. ✅ `gemma_sae_experiment/` → `exploratory_experiments/gemma_sae_experiment/`
3. ✅ `lr_classification_experiment/` → `exploratory_experiments/lr_classification_experiment/`
4. ✅ `alternative_paradigms/` → `exploratory_experiments/alternative_paradigms/`
5. ✅ `additional_experiments/` → `exploratory_experiments/additional_experiments/`

---

## Paper Experiments (Unchanged)

These directories remain at the root under `paper_experiments/`:

### Paper Section 3.1 - Behavioral Experiments
- **`slot_machine_6models/`** (19,200 games, 6 models)
  - Findings 1-5: Autonomy effects, betting flexibility, goal-setting
- **`investment_choice_experiment/`** (6,400 games, 4 models)
  - Ablation study: Goal escalation, risk preference shifts

### Paper Section 3.2 - Neural Mechanisms
- **`llama_sae_analysis/`** (112 causal features)
  - SAE feature extraction + activation patching

### Paper Section 5 - Token Analysis
- **`pathway_token_analysis/`**
  - Temporal/linguistic patterns

**Total paper sample size**: 25,600 games

---

## Exploratory Experiments (Moved)

### 1. `steering_vector_analysis/`
- **Purpose**: CAA-based steering vector extraction
- **Status**: Completed but not in paper
- **Usage**: `bash exploratory_experiments/steering_vector_analysis/scripts/launch_full_analysis_4gpu.sh`

### 2. `gemma_sae_experiment/`
- **Purpose**: Validate SAE methodology on Gemma-2-9B
- **Status**: Exploratory - different architecture validation
- **Usage**: `python exploratory_experiments/gemma_sae_experiment/run_pipeline.py --gpu 0 --phases all`

### 3. `lr_classification_experiment/`
- **Purpose**: Logistic regression from hidden states
- **Status**: Completed - complementary to SAE analysis
- **Usage**: `python exploratory_experiments/lr_classification_experiment/run_experiment.py --model gemma --option B --gpu 0`

### 4. `alternative_paradigms/`
- **Purpose**: Domain generalization (IGT, Loot Box, Near-Miss)
- **Status**: Completed - validates autonomy effects
- **Usage**:
  - IGT: `python exploratory_experiments/alternative_paradigms/src/igt/run_experiment.py --model llama --gpu 0 --quick`
  - Loot Box: `python exploratory_experiments/alternative_paradigms/src/lootbox/run_experiment.py --model gemma --gpu 0 --quick`
  - Near-Miss: `python exploratory_experiments/alternative_paradigms/src/nearmiss/run_experiment.py --model qwen --gpu 0 --quick`

### 5. `additional_experiments/`
- **Purpose**: Post-submission extensions
- **Key**: `sae_condition_comparison/` - Variable vs Fixed neural differences
- **Status**: Ongoing - requires sparse feature filtering
- **Usage**: `python -m exploratory_experiments.additional_experiments.sae_condition_comparison.src.condition_comparison --model llama`

---

## Documentation Updates

### New Files Created
1. **`STRUCTURE.md`** - Comprehensive repository organization guide
2. **`exploratory_experiments/README.md`** - Detailed exploratory experiments documentation
3. **`REORGANIZATION_SUMMARY.md`** (this file) - Change summary

### Updated Files
1. **`CLAUDE.md`** - Updated all paths to reflect new structure
   - Repository structure section updated
   - All command examples updated with new paths
   - Added references to new documentation files

---

## Breaking Changes

⚠️ **Path Changes**: All import paths and command paths for exploratory experiments have changed.

### Before (Old Paths)
```bash
python alternative_paradigms/src/igt/run_experiment.py --model llama --gpu 0
python lr_classification_experiment/run_experiment.py --model gemma --option B --gpu 0
python -m additional_experiments.sae_condition_comparison.src.condition_comparison --model llama
```

### After (New Paths)
```bash
python exploratory_experiments/alternative_paradigms/src/igt/run_experiment.py --model llama --gpu 0
python exploratory_experiments/lr_classification_experiment/run_experiment.py --model gemma --option B --gpu 0
python -m exploratory_experiments.additional_experiments.sae_condition_comparison.src.condition_comparison --model llama
```

---

## What Stayed the Same

✅ **Paper experiments**: No changes to `paper_experiments/` directory structure
✅ **Data location**: Still at `/mnt/c/Users/oollccddss/git/data/llm-addiction/`
✅ **Legacy folder**: Unchanged
✅ **Conda environment**: Still `llama_sae_env`
✅ **Git configuration**: Main branch still `master`

---

## Why This Change?

### Problems Before
- ❌ Unclear which experiments were in the paper
- ❌ Difficult to navigate between paper and exploratory work
- ❌ Documentation scattered across multiple locations

### Benefits After
- ✅ Clear separation: Paper vs Exploratory
- ✅ Easier to understand project structure
- ✅ Single source of truth for exploratory experiments (`exploratory_experiments/README.md`)
- ✅ Better organization for future researchers

---

## Quick Reference

### Running Paper Experiments
```bash
# Slot machine (API models)
python paper_experiments/slot_machine_6models/src/run_gpt5_experiment.py

# Slot machine (local models)
python paper_experiments/slot_machine_6models/src/llama_gemma_experiment.py

# SAE analysis
python paper_experiments/llama_sae_analysis/src/phase1_feature_extraction.py

# Token analysis
bash paper_experiments/pathway_token_analysis/scripts/launch_all_phases_sequential.sh
```

### Running Exploratory Experiments
See `exploratory_experiments/README.md` for detailed instructions.

---

## Migration Checklist

If you have local scripts or notebooks that reference the old paths:

- [ ] Update all import paths to include `exploratory_experiments/` prefix
- [ ] Update all command-line invocations with new paths
- [ ] Update any custom scripts that reference moved directories
- [ ] Re-read `CLAUDE.md` for updated guidance
- [ ] Review `STRUCTURE.md` for repository organization

---

## Paper Mapping Reference

| Paper Section | Experiment | Sample Size | Location |
|--------------|------------|-------------|----------|
| 3.1 (Findings 1-5) | Slot Machine | 19,200 games | `paper_experiments/slot_machine_6models/` |
| 3.1 (Ablation) | Investment Choice | 6,400 games | `paper_experiments/investment_choice_experiment/` |
| 3.2 (Neural) | LLaMA SAE | 112 features | `paper_experiments/llama_sae_analysis/` |
| 5 (Token) | Pathway Analysis | - | `paper_experiments/pathway_token_analysis/` |

**Not in paper**: Everything in `exploratory_experiments/`

---

## Contact

If you encounter any issues with the new structure or have questions about which experiment is which, please refer to:
1. `STRUCTURE.md` - Repository organization
2. `CLAUDE.md` - Comprehensive guidance
3. `exploratory_experiments/README.md` - Exploratory experiments details

---

**Reorganization completed**: February 2, 2026
**Paper submission target**: ICLR 2026
