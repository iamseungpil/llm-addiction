# Steering Vector Analysis for LLM Gambling Behavior

## 📊 Latest Experiment Results
**Date**: 2024-12-21  
**Status**: Complete  
**Models**: LLaMA-3.1-8B, Gemma-2-9B

## 🗂️ Project Structure
```
steering_vector_analysis/
├── src/                    # Core implementation
│   ├── extract_steering_vectors.py      # Phase 1: Extract steering vectors
│   ├── phase2_direct_steering.py        # Phase 2: Direct steering experiments  
│   ├── phase3_sae_interpretation.py     # Phase 3: SAE feature analysis
│   ├── phase4_head_component_analysis.py # Phase 4: Attention head analysis
│   ├── phase5_activation_patching.py    # Phase 5: Activation patching
│   └── utils.py                         # Utility functions
├── configs/                # Experiment configurations
│   ├── experiment_config.yaml
│   ├── experiment_config_direct_steering.yaml
│   └── experiment_config_full_layers.yaml
├── scripts/                # Launch scripts
│   ├── launch_full_analysis.sh
│   ├── launch_full_analysis_4gpu.sh
│   └── launch_pipeline.sh
├── data/                   # Data access (symlinks)
│   ├── results/           -> steering vector experiment results
│   ├── llama_data/        -> LLaMA 3,200 games data
│   └── gemma_data/        -> Gemma 3,200 games data
├── docs/                   # Documentation
│   ├── EXPERIMENT_DESCRIPTION.md       # Detailed methodology
│   └── CODE_REVIEW_20251219.md         # Code review
└── results/                # Output directory for new runs
```

## 🚀 Quick Start

### 1. Run Complete Analysis Pipeline
```bash
cd steering_vector_analysis
bash scripts/launch_full_analysis_4gpu.sh
```

### 2. Run Individual Phases
```bash
# Phase 1: Extract steering vectors
python src/extract_steering_vectors.py --config configs/experiment_config.yaml

# Phase 3: SAE interpretation  
python src/phase3_sae_interpretation.py --config configs/experiment_config.yaml

# Phase 4: Head analysis
python src/phase4_head_component_analysis.py --config configs/experiment_config.yaml
```

## 📈 Key Results Summary

### Steering Vector Magnitudes (LLaMA)
- **Layer 0**: |v| = 0.02 (minimal signal)
- **Layer 10**: |v| = 0.47 (moderate signal) 
- **Layer 20**: |v| = 1.03 (strong signal)
- **Layer 30**: |v| = 3.04 (maximum signal)

### SAE Feature Discovery
- **LLaMA**: 640 candidate features across 32 layers
- **Gemma**: 840 candidate features across 42 layers
- **Top features** show strong risk/safety direction alignment

### Data Sources
- **LLaMA**: 150 bankrupt vs 150 safe samples (4.7% bankruptcy rate)
- **Gemma**: 670 bankrupt vs 2,530 safe samples (20.9% bankruptcy rate)

## 📝 Methodology Overview

**Phase 1**: Contrastive Activation Addition (CAA) to extract steering vectors  
**Phase 2**: Direct steering experiments with various magnitudes  
**Phase 3**: SAE feature projection and interpretation  
**Phase 4**: Attention head component analysis  
**Phase 5**: Multi-feature activation patching

## 🔗 Related Files
- Original experiment code: `legacy/steering_vector_experiment/`
- Full results: `data/results/` (via symlink)
- Source data: `data/llama_data/` and `data/gemma_data/` (via symlinks)

---
*This is a clean, self-contained version focusing on the latest steering vector analysis (2024-12-21)*