# LLaMA SAE Analysis

## 🔬 Paper Section 4: "Mechanistic Causes of Risk-Taking Behavior in LLMs"

### Experimental Design
- **Model**: LLaMA-3.1-8B
- **Experiments**: 6,400 slot machine games  
- **Analysis**: Sparse Autoencoder (SAE) + Activation Patching
- **Layers**: L25-L31 (7 layers, 32,768 features per layer)
- **Method**: Population mean activation patching for causal validation

### Key Paper Results

#### Finding 1: Differential Features Discovery
- **Total features analyzed**: 122,880 (4,096 × 30 layers)
- **Significant features**: 83,684 (68.1%) pass statistical criteria
- **Criteria**: p < 0.01, FDR correction, |Cohen's d| > 0.3
- **Layer distribution**: Early (2,195/layer) → Middle (3,361/layer) → Late (3,478/layer)

#### Finding 2: Causal Feature Effects  
From 3,365 differential features → **361 safe features** + **80 risky features**

| Feature Type | Safe Context Effect | Risky Context Effect |
|--------------|-------------------|-------------------|
| **Safe Features (361)** | +29.6% stopping rate | +28.4% stopping, -14.2% bankruptcy |
| **Risky Features (80)** | -6.4% stopping rate | -7.8% stopping, +11.7% bankruptcy |

## 🚀 Quick Start

### Run SAE Analysis Pipeline
```bash
# Phase 1: Feature extraction
python src/phase1_feature_extraction.py

# Phase 2: Correlation analysis  
python src/phase2_correlation_analysis.py

# Phase 3: Semantic analysis
python src/phase3_semantic_analysis.py

# Phase 4: Causal validation (main results)
python src/phase4_causal_pilot_v2.py

# Phase 5: Multi-feature steering
python src/phase5_multifeature_steering.py
```

### Run Original Patching Experiments
```bash
python experiment_2_L1_31_top300.py
```

## 📁 Files Overview

### Core Analysis Pipeline
- **`phase1_feature_extraction.py`**: Extract SAE features from gambling games
- **`phase2_correlation_analysis.py`**: Correlate features with behavioral outcomes  
- **`phase3_semantic_analysis.py`**: Semantic interpretation of top features
- **`phase4_causal_pilot_v2.py`**: **Main causal validation via activation patching**
- **`phase5_multifeature_steering.py`**: Multi-feature steering experiments

### Original Patching Code
- **`experiment_2_L1_31_top300.py`**: Original L1-31 top 300 features patching

### Results Data (via symlinks)  
- **`data/results/`** → Links to `/data/llm-addiction/sae_patching/`
- Contains SAE analysis results, patching outcomes, feature correlations

## 🔬 Methodology Details

### SAE Feature Analysis
1. **Data Collection**: 6,400 LLaMA games (211 bankruptcy, 6,189 safe)
2. **Feature Extraction**: 32,768 SAE features per layer (L25-L31)  
3. **Statistical Testing**: Multiple comparison correction (FDR)
4. **Effect Size**: Cohen's d for group separation

### Activation Patching Protocol
1. **Population Mean Patching**: Apply average activations from one group to another
2. **Causal Validation**: Measure behavioral changes (stopping rate, bankruptcy)
3. **Bidirectional Testing**: Safe→Risky and Risky→Safe contexts
4. **Significance Testing**: 30 independent trials per condition

## 📊 Paper Figures Generated

- **Figure: Feature Patching Methodology** - Shows SAE decomposition process
- **Figure: SAE Feature Separation** - Activation distributions (Cohen's d > 1.2)
- **Figure: Causal Patching Comparison** - Safe vs risky features effects
- **Figure: Causal Features Layer Distribution** - Distribution across layers

## 🔑 Key Insights

1. **Neural-level risk patterns**: 68.1% of features distinguish risky vs safe decisions
2. **Causal control demonstrated**: Features directly influence gambling behavior  
3. **Protective dominance**: Safe features (361) outnumber risky features (80)
4. **Bidirectional effects**: Features work consistently across contexts
5. **Intervention potential**: Targeted feature control could prevent harmful behaviors

---
*This experiment establishes the mechanistic basis of addiction-like behaviors in LLMs*