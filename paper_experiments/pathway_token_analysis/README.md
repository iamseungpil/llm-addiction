# Pathway Token Analysis

## 📈 Paper Section 5: "Temporal and Linguistic Dimensions of Risk-Taking Mechanisms"

### Experimental Design
- **Focus**: Token-level temporal progression and linguistic correlates
- **Analysis**: Multi-phase pathway tracking of decision-making process
- **Scope**: Feature pathway evolution, word-feature correlations, prompt-response analysis
- **Data**: LLaMA-3.1-8B gambling experiments with temporal decomposition

### Key Paper Results

#### Temporal Pathway Evolution
- **Phase-wise progression**: Features evolve systematically across decision timepoints  
- **Critical transitions**: Identifiable moments where safe→risky transitions occur
- **Layer-wise dynamics**: Different layers contribute at different temporal stages

#### Linguistic Correlates
- **Word-feature associations**: Specific vocabulary patterns predict risk-taking
- **Prompt sensitivity**: Different prompt components activate different feature pathways
- **Response patterns**: Language generation reflects underlying risk representation

## 🚀 Quick Start

### Run Complete Pipeline
```bash
# Run all 5 phases sequentially
bash scripts/launch_all_phases_sequential.sh
```

### Run Individual Phases
```bash
# Phase 1: Extract activations for 6 conditions, 50 trials each
python src/phase1_6conditions_50trials.py
python src/phase1_FDR_265features.py

# Phase 2: Compute feature correlations  
python src/phase2_compute_correlations.py
python src/phase2_patching_correlations.py

# Phase 3: Causal validation
python src/phase3_causal_validation.py  
python src/phase3_patching_causal_validation.py

# Phase 4: Word-feature analysis
python src/phase4_word_analysis.py
python src/phase4_word_feature_correlation.py

# Phase 5: Prompt-response analysis  
python src/phase5_prompt_response_words.py
python src/phase5_prompt_feature_correlation.py
```

## 📁 Files Overview

### Phase 1: Activation Extraction
- **`phase1_6conditions_50trials.py`**: Extract activations for 6 core conditions
- **`phase1_FDR_265features.py`**: FDR-corrected feature selection (265 features)
- **`phase1_extract_activations.py`**: General activation extraction utilities

### Phase 2: Correlation Analysis  
- **`phase2_compute_correlations.py`**: Feature-behavior correlation computation
- **`phase2_patching_correlations.py`**: Correlation-based patching validation

### Phase 3: Causal Validation
- **`phase3_causal_validation.py`**: Validate causal effects of pathway features
- **`phase3_patching_causal_validation.py`**: Patching-based causal testing

### Phase 4: Word-Level Analysis
- **`phase4_word_analysis.py`**: Analyze word-level patterns in responses
- **`phase4_word_feature_correlation.py`**: Correlate words with feature activations

### Phase 5: Prompt-Response Analysis
- **`phase5_prompt_response_words.py`**: Analyze prompt→response word patterns  
- **`phase5_prompt_feature_correlation.py`**: Correlate prompts with feature pathways

### Results Data (via symlinks)
- **`data/results/`** → Links to `/data/llm-addiction/analysis/pathway_token_analysis/`
- Contains 5-phase analysis results, compressed data, word correlations

## 🔬 Methodology Details

### Pathway Tracking Protocol
1. **Temporal Decomposition**: Break down decision-making into discrete time steps
2. **Token-Level Analysis**: Analyze feature activations at each token generation
3. **Correlation Mapping**: Map features to behavioral outcomes across time
4. **Causal Validation**: Verify temporal causality via targeted interventions

### Feature Selection (265 Features)
- **FDR Correction**: False Discovery Rate control for multiple comparisons
- **Effect Size Filtering**: Cohen's d > threshold for meaningful differences  
- **Layer Coverage**: Representative features across all 31 layers
- **Behavioral Relevance**: Features with demonstrated gambling behavior correlation

### Word-Feature Correlation Analysis
- **Vocabulary Extraction**: Extract key words from gambling responses
- **Feature Mapping**: Map word usage to feature activation patterns
- **Predictive Analysis**: Identify linguistic predictors of risk-taking
- **Prompt Sensitivity**: Analyze how prompts influence word-feature relationships

## 📊 Key Analysis Phases

### Phase 1-3: Core Pathway Identification
- Identify 265 key features with FDR correction
- Establish feature→behavior correlations  
- Validate causal relationships through patching

### Phase 4-5: Linguistic Integration  
- Map features to language patterns
- Analyze temporal word evolution
- Connect prompts to response generation pathways

## 🔑 Key Insights

1. **Temporal Progression**: Risk-taking features evolve systematically across decision time
2. **Linguistic Signatures**: Specific vocabulary patterns predict and reflect risk behavior  
3. **Prompt Sensitivity**: Different prompts activate distinct feature pathways
4. **Multi-scale Integration**: Token-level analysis reveals decision-making mechanisms
5. **Predictive Power**: Pathway analysis enables early risk prediction

---
*This experiment reveals the temporal and linguistic structure of risk-taking mechanisms in LLMs*