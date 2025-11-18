# Experiment 1: Gradient-based Pathway Tracking (L1-31)

## Overview

**Feature-centric gradient pathway tracking** for 2,787 causal features using Anthropic's Attribution Graphs method (2025).

## Method

### Backward Jacobian (Gradient-based)

```python
# For each target feature (e.g., L31-10692):
target_activation = SAE_L31[10692]

# Compute gradients
∂(target_activation) / ∂(all_upstream_features) = Jacobian

# Find strong contributors
upstream_features = features where |gradient| > 0.1
```

**Key difference from old correlation method**:
- ✅ **Gradient = Causal contribution** (not just correlation)
- ✅ **Feature-specific pathways** (not game-level correlation)
- ✅ **Continuous L1→L31 tracking** (backward pass)

## Data

### Input: 2,787 Causal Features
```
/home/ubuntu/llm_addiction/analysis/CORRECT_consistent_safe_features.csv    (640)
/home/ubuntu/llm_addiction/analysis/CORRECT_consistent_risky_features.csv (2,147)
```

### Method
- Anthropic Attribution Graphs (2025)
- Backward Jacobian for each feature
- Safe/Risky prompt comparison

## Usage

### Run Tracker

```bash
cd /home/ubuntu/llm_addiction/experiment_1_layer_pathway_L1_31
./launch.sh 4  # GPU 4
```

**Expected time**: ~1.5 hours

**Monitor**:
```bash
tmux attach -t pathway_gradient_gpu4
```

## Output

### Results File
```
results/gradient_pathways_YYYYMMDD_HHMMSS_final.json
```

### Structure
```json
{
  "feature": "L31-10692",
  "type": "risky",
  "layer": 31,
  "feature_id": 10692,
  "safe_pathway": [
    {
      "source": "L9-456",
      "target": "L31-10692",
      "gradient": 0.91,
      "source_layer": 9,
      "prompt": "safe"
    },
    ...
  ],
  "risky_pathway": [...],
  "safe_upstream_count": 15,
  "risky_upstream_count": 23
}
```

## Analysis Scripts

### 1. Multi-hop Path Discovery
```bash
python src/multihop_path_analyzer.py
```

Finds 3-layer paths: L_i → L_j → L_k

### 2. Middle Layer Analysis
```bash
python src/analyze_middle_layers.py
```

Special analysis for L9-L17 risky features.

### 3. Visualization
```bash
python src/visualize_pathways.py
```

Generates:
- Pathway network graph
- Layer contribution heatmap
- Sankey diagram (Early → Middle → Late)

## Key Research Questions

1. **중간 layer risky features (L9-L17)의 역할은?**
   - 초기 layer (L1-L8) 신호를 증폭?
   - 후반 layer (L25-L31)로 전달하는 bridge?

2. **Safe vs Risky pathway 차이는?**
   - Safe: Early layers (L1-L4) → Late layers (L25-L29)?
   - Risky: Middle layers (L9-L17) dominant?

3. **Prompt에 따라 pathway가 바뀌는가?**
   - Safe prompt에서는 safe features 활성화?
   - Risky prompt에서는 risky features 활성화?

## Expected Findings

### Safe Feature Pathway
```
L1-1292 (early detection)
  ↓ gradient=0.72
L24-1111 (accumulation)
  ↓ gradient=0.82
L29-3494 (final decision)
```

### Risky Feature Pathway (Middle-layer dominant!)
```
L3-111 (initial signal)
  ↓ gradient=0.81
L9-5678 (AMPLIFIER)
  ↓ gradient=0.91
L17-9999 (INTEGRATOR)
  ↓ gradient=0.75
L30-2222 (final output)
```

## Comparison with Old Method

### ❌ Old (Archived): Correlation-based
```python
# Game-level correlation (WRONG!)
r = correlation(L8_across_games, L31_across_games)
# Problem: Correlation ≠ Causation
```

### ✅ New (Current): Gradient-based
```python
# Feature-level gradient (CORRECT!)
grad = ∂(L31-10692) / ∂(L8_features)
# Gradient = Causal contribution
```

## Resources

- **Anthropic Paper**: Circuit Tracing (2025)
- **Method**: Attribution Graphs with Backward Jacobian
- **Tools**: PyTorch autograd, LlamaScope SAEs

## Next Steps

After pathway tracking completes:
1. Run multi-hop analysis
2. Analyze middle-layer role
3. Compare with word associations (Experiment 3)
4. Create visualizations
5. Write paper section

---

**Created**: 2025-10-22
**Method**: Gradient-based (Anthropic 2025)
**Features**: 2,787 (640 safe + 2,147 risky)
