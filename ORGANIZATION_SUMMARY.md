# LLM Addiction Research Project Organization Summary

## Archive Process Completed

**Date**: September 29, 2025
**Total Files Processed**: 4,290 files (1,979 home + 2,311 data)
**Total Files Archived**: 1,872 files (1,181 home + 691 data)
**Essential Files Preserved**: 2,421 files (801 home + 1,620 data)

## Archive Structure Created

### `/home/ubuntu/llm_addiction/ARCHIVE_NON_ESSENTIAL/`
- **Files Archived**: 1,181 files
- **Content**: Analysis scripts, visualization generators, debug tools, intermediate results, logs

### `/data/llm_addiction/ARCHIVE_NON_ESSENTIAL/`
- **Files Archived**: 691 files
- **Directories Moved**: `logs/`, `archive/`, `gpt_results_corrected/`, `gpt_results/`

## Essential Files Preserved (53 Core Files + Supporting Files)

### 🔬 **Core Experiment Files**
1. `/home/ubuntu/llm_addiction/gpt_experiments/src/gpt_fixed_parsing_experiment.py`
2. `/home/ubuntu/llm_addiction/improved_gpt_parsing.py`
3. `/home/ubuntu/llm_addiction/gpt5_experiment/run_gpt5_experiment.py`
4. `/home/ubuntu/llm_addiction/causal_feature_discovery/src/experiment_1_multiround.py`
5. `/home/ubuntu/llm_addiction/causal_feature_discovery/src/extract_statistically_valid_features_all_layers.py`
6. `/home/ubuntu/llm_addiction/causal_feature_discovery/src/experiment_2_final_correct.py`
7. `/home/ubuntu/llm_addiction/causal_feature_discovery/src/llama_scope_working.py`

### 📊 **Essential Analysis Scripts**
8. `/home/ubuntu/llm_addiction/analysis/exp2_causal_feature_group_analysis.py`
9. `/home/ubuntu/llm_addiction/analyze_all_models_comprehensive.py`
10. `/home/ubuntu/llm_addiction/create_corrected_3200_comprehensive_analysis.py`
11. `/home/ubuntu/llm_addiction/analyze_streak_patterns.py`
12. `/home/ubuntu/llm_addiction/find_best_separated_features.py`
13. `/home/ubuntu/llm_addiction/comprehensive_feature_count_analysis.py`

### 📈 **Essential Figure Generation Scripts**
14. `/home/ubuntu/llm_addiction/writing/table_figure/create_composite_indices.py`
15. `/home/ubuntu/llm_addiction/writing/table_figure/create_component_effects_by_bettype.py`
16. `/home/ubuntu/llm_addiction/writing/table_figure/create_4model_complexity_trend_average.py`
17. `/home/ubuntu/llm_addiction/writing/table_figure/create_4model_streak_analysis_1x2.py`
18. `/home/ubuntu/llm_addiction/create_4x3_component_effects_by_model.py`
19. `/home/ubuntu/llm_addiction/create_4x4_complexity_trend_by_model.py`
20. `/home/ubuntu/llm_addiction/create_4x2_streak_analysis_CORRECTED.py`
21. `/home/ubuntu/llm_addiction/create_MAXIMUM_separation_image1.py`
22. `/home/ubuntu/llm_addiction/writing/table_figure/create_causal_patching_effects_summary.py`
23. `/home/ubuntu/llm_addiction/create_comprehensive_6_panel_patching_figure.py`

### 📝 **Essential Paper Files**
24. `/home/ubuntu/llm_addiction/writing/3_1_can_llm_be_addicted_final.tex`
25. `/home/ubuntu/llm_addiction/writing/3_2_llama_feature_analysis_final.tex`
26. `/home/ubuntu/llm_addiction/writing/1_introduction.tex`
27. `/home/ubuntu/llm_addiction/writing/2_preliminaries.tex`
28. `/home/ubuntu/llm_addiction/writing/5_conclusion.tex`
29. `/home/ubuntu/llm_addiction/writing/3_slotmachine_final.tex`
30. `/home/ubuntu/llm_addiction/writing/appendix3_detailed_results_by_model.tex`
31. `/home/ubuntu/llm_addiction/writing/3_2_1_llama_sae_feature_discovery.tex`

### 📊 **Essential Data Files**
32. `/data/llm_addiction/gpt_results_fixed_parsing/gpt_fixed_parsing_complete_20250919_151240.json`
33. `/data/llm_addiction/gpt5_experiment/gpt5_experiment_20250921_174509.json`
34. `/data/llm_addiction/claude_experiment/claude_experiment_corrected_20250925.json`
35. `/data/llm_addiction/gemini_experiment/gemini_experiment_20250920_042809.json`
36. `/data/llm_addiction/results/exp1_multiround_intermediate_20250819_140040.json`
37. `/data/llm_addiction/results/exp1_missing_complete_20250820_090040.json`
38. `/data/llm_addiction/results/multilayer_features_20250911_171655.npz`
39. `/home/ubuntu/llm_addiction/all_models_comprehensive_stats.json`

### 🖼️ **Essential Figures**
40. `/home/ubuntu/llm_addiction/writing/figures/4model_composite_indices.pdf`
41. `/home/ubuntu/llm_addiction/writing/figures/component_effects_by_bettype.pdf`
42. `/home/ubuntu/llm_addiction/writing/figures/4model_complexity_trend_average.pdf`
43. `/home/ubuntu/llm_addiction/writing/figures/streak_analysis_1x2.pdf`
44. `/home/ubuntu/llm_addiction/writing/figures/SAE_feature_separation_main.pdf`
45. `/home/ubuntu/llm_addiction/writing/figures/causal_patching_average_effects.pdf`
46. `/home/ubuntu/llm_addiction/writing/figures/causal_features_layer_distribution_corrected.pdf`
47. `/home/ubuntu/llm_addiction/writing/figures/feature_patching.pdf`

### 📁 **Essential Directories (Complete)**
48. `/home/ubuntu/llm_addiction/claude_experiment/` (entire directory)
49. `/home/ubuntu/llm_addiction/gemini_experiment/` (entire directory)
50. `/home/ubuntu/llm_addiction/causal_feature_discovery/Language-Model-SAEs/` (entire directory)
51. `/home/ubuntu/llm_addiction/writing/figures/appendix/` (entire directory)

### 🔍 **Essential Analysis Results**
52. `/home/ubuntu/llm_addiction/analysis/exp2_causal_feature_group_summary.csv`
53. `/data/llm_addiction/results/exp2_response_log_*_20250918*.json` (48 files)

## Current Project Structure (Clean)

```
/home/ubuntu/llm_addiction/
├── ARCHIVE_NON_ESSENTIAL/           # 🗂️ All non-essential files
├── CLAUDE.md                        # 📋 Project instructions
├── all_models_comprehensive_stats.json  # 📊 Essential stats
├── analysis/                        # 📈 Essential analysis scripts
├── causal_feature_discovery/        # 🧪 Core experiment framework
│   ├── src/                        # Core experiment scripts
│   └── Language-Model-SAEs/        # Essential SAE framework
├── claude_experiment/              # 🤖 Claude experiment (complete)
├── gemini_experiment/              # 🧠 Gemini experiment (complete)
├── gpt5_experiment/                # 🚀 GPT-5 experiment (complete)
├── gpt_experiments/                # 🔧 GPT experiment tools
├── writing/                        # 📝 Paper files & figures
├── figures/                        # 📊 Essential figures
└── [Essential analysis scripts]    # 🔬 Core analysis tools

/data/llm_addiction/
├── ARCHIVE_NON_ESSENTIAL/          # 🗂️ Archived data files
├── gpt_results_fixed_parsing/      # ✅ Fixed GPT results
├── gpt5_experiment/               # 🚀 GPT-5 experiment data
├── claude_experiment/             # 🤖 Claude experiment data
├── gemini_experiment/             # 🧠 Gemini experiment data
└── results/                       # 📊 LLaMA experiment results
```

## Benefits Achieved

### ✅ **Organization**
- **Reduced file count by 43.6%** (from 4,290 to 2,421 active files)
- Clear separation between essential research files and development artifacts
- Maintained complete reproducibility for both papers

### ✅ **Preserved Functionality**
- All 53 essential files verified and accessible
- Both research papers can be fully reproduced
- All experiments can be re-run with preserved scripts and data

### ✅ **Improved Navigation**
- Clean project structure with logical groupings
- Easy identification of core vs. supporting files
- Archived development artifacts while preserving access

### ✅ **Research Integrity**
- Zero data loss - everything moved to archives, not deleted
- Complete experimental reproducibility maintained
- All essential figures and tables preserved

## Recovery Process

If any archived file is needed:
```bash
# Files can be easily retrieved from archives
cp /home/ubuntu/llm_addiction/ARCHIVE_NON_ESSENTIAL/[filename] /home/ubuntu/llm_addiction/
cp /data/llm_addiction/ARCHIVE_NON_ESSENTIAL/[filename] /data/llm_addiction/
```

## Archive Contents

### Most Common Archived File Types:
- **Analysis variations**: 150+ visualization scripts
- **Debug tools**: 40+ debugging and testing scripts
- **Intermediate results**: 100+ temporary JSON/CSV files
- **Development logs**: 50+ log files from development
- **Legacy experiments**: Old experiment versions and archives

This organization provides a clean, focused research environment while preserving complete access to all development history and alternative analysis approaches.