# Experiment 6 Extended: L1-31 Token-Level Tracking (OPTIMIZED)

## Overview

Extends Experiment 6's token-level analysis to **ALL 31 layers** with memory-efficient storage.

**Original Experiment 6**: Analyzed layers 8, 15, 31 only (3 layers)
**This Extension**: Analyzes all layers 1-31

## Critical Optimization

### Problem with Naive Approach
- **JSON conversion**: `np.array(...).tolist()` converts numpy → Python lists
- **Memory explosion**: 10.24GB peak (all layers + conversion overhead)
- **File size explosion**: 5.12GB numpy → 7-8GB JSON
- **Single-GPU bottleneck**: OOM on most hardware

### Optimized NPZ + Incremental Saving
**Key improvements**:
1. **NPZ compressed format** instead of JSON (2-3× smaller, no conversion overhead)
2. **Layer-wise saving**: Process and save one layer at a time (165MB peak vs 10GB)
3. **Separate metadata**: Lightweight JSON metadata + heavy NPZ data files
4. **On-demand SAE loading**: Load SAEs only when needed

**Results**:
- **Memory peak**: 165MB (one layer) instead of 10GB (all layers)
- **File size**: 2-3GB instead of 7-8GB
- **No OOM issues**: Runs on any GPU with 40GB+ memory

## What This Tracks

For each of 10 scenarios:
- **Attention weights**: Which tokens attend to which (n_heads × seq_len × seq_len)
- **SAE Features**: Feature activations at each token position (seq_len × 32768)

## Data Size Options

| Option | Save Features | Save Attention | File Size | Use Case |
|--------|---------------|----------------|-----------|----------|
| **Full** | ✅ | ✅ | ~2-3 GB | Complete analysis |
| **Attention Only** | ❌ | ✅ | ~300 MB | Lightweight pathway analysis |

## Usage

### Option 1: Full Data (Features + Attention)

```bash
cd /home/ubuntu/llm_addiction/experiment_6_L1_31_token_tracking
mkdir -p logs
chmod +x launch_full.sh
./launch_full.sh
```

**Output**: `/data/llm_addiction/experiment_6_L1_31_token_tracking/`
- `[scenario_name]/layer_1.npz` (features + attention for layer 1)
- `[scenario_name]/layer_2.npz` (features + attention for layer 2)
- ...
- `[scenario_name]/layer_31.npz` (features + attention for layer 31)
- `metadata_L1_31_[timestamp].json` (lightweight metadata)

**Total size**: ~2-3GB
**Time**: ~5-6 hours

### Option 2: Attention Only (Lightweight)

```bash
cd /home/ubuntu/llm_addiction/experiment_6_L1_31_token_tracking
mkdir -p logs
chmod +x launch_attention_only.sh
./launch_attention_only.sh
```

**Output**: Same structure but NPZ files contain only attention (no features)

**Total size**: ~300MB
**Time**: ~3-4 hours

### Option 3: Custom Layers

```bash
# Example: Only layers 20-31
python3 experiment_6_L1_31_optimized.py \
    --gpu 0 \
    --layer-start 20 \
    --layer-end 31
```

## Output Format

### Directory Structure
```
/data/llm_addiction/experiment_6_L1_31_token_tracking/
  Bankruptcy_90_all_in/
    layer_1.npz
    layer_2.npz
    ...
    layer_31.npz
  Desperate_10/
    layer_1.npz
    ...
  metadata_L1_31_20251017_120000.json
```

### Metadata JSON (Lightweight)
```json
{
  "timestamp": "20251017_120000",
  "n_scenarios": 10,
  "layer_range": [1, 31],
  "save_features": true,
  "save_attention": true,
  "scenarios": [
    {
      "scenario": "Bankruptcy_90_all_in",
      "tokens": ["<s>", "Slot", " Machine", ...],
      "token_ids": [1, 34834, 13257, ...],
      "seq_len": 121,
      "key_positions": {
        "balance": [51, 53],
        "goal": [76, 78],
        "probability": [82],
        "choices": [115, 117]
      }
    }
  ]
}
```

### NPZ Files (Heavy Data)
Each `layer_[N].npz` contains:
```python
import numpy as np

data = np.load('Bankruptcy_90_all_in/layer_15.npz')

# SAE features (if --no-features not used)
features = data['features']  # shape: (seq_len, 32768)

# Attention weights (if --no-attention not used)
attention = data['attention']  # shape: (n_heads, seq_len, seq_len)
```

## 10 Scenarios Tracked

1. **Bankruptcy_90_all_in**: Balance $90, 2 consecutive losses, high risk
2. **Desperate_10**: Balance $10, 3 consecutive losses, desperate situation
3. **Safe_130_one_win**: Balance $130, 1 win, safe position
4. **Risky_40**: Balance $40, big loss streak
5. **Safe_140_near_goal**: Balance $140, near $200 goal
6. **Initial_100_first_round**: Starting position
7. **Medium_60**: Balance $60, mixed history
8. **Goal_achieved_200**: Reached $200 goal
9. **Very_risky_25**: Balance $25, 2 consecutive losses
10. **Big_success_280**: Balance $280, big wins

## Analysis Capabilities

After collection, you can analyze:

### 1. Layer-wise Attention Evolution
```python
import numpy as np

# Load attention across layers
attentions = {}
for layer in range(1, 32):
    data = np.load(f'Bankruptcy_90_all_in/layer_{layer}.npz')
    attentions[layer] = data['attention']

# Analyze: How does attention to balance tokens change L1→L31?
balance_pos = 51
for layer in [1, 10, 20, 31]:
    # Average attention to balance token across all heads
    attn_to_balance = attentions[layer][:, -1, balance_pos].mean()
    print(f"Layer {layer}: {attn_to_balance:.4f}")
```

### 2. Feature Activation Pathways
```python
# Track specific feature across layers
feature_id = 14826  # risk_taking feature

for layer in range(1, 32):
    data = np.load(f'Bankruptcy_90_all_in/layer_{layer}.npz')
    features = data['features']

    # Feature activation at final token position
    activation = features[-1, feature_id]
    print(f"Layer {layer}: {activation:.4f}")
```

### 3. Cross-Reference with Experiment 3
- Features with "balance" words → attention to balance tokens?
- Features with "stop" words → attention to decision tokens?

### 4. Attention Pathway Analysis
```python
# Balance → Decision attention strength by layer
decision_pos = 115  # "Bet" token position
balance_pos = 51

for layer in range(1, 32):
    data = np.load(f'Bankruptcy_90_all_in/layer_{layer}.npz')
    attention = data['attention']

    # Attention from decision token to balance token
    attn_strength = attention[:, decision_pos, balance_pos].mean()
    print(f"Layer {layer}: Balance→Decision = {attn_strength:.4f}")
```

## Expected Results

### Attention Patterns
- **Early layers (L1-L10)**: Local attention (adjacent tokens)
- **Middle layers (L11-L20)**: Task-relevant attention (balance, goal, probability)
- **Late layers (L21-L31)**: Decision-focused attention (choice tokens)

### Feature Activations
- **L1-L10**: Basic linguistic features
- **L11-L20**: Task features (balance tracking, goal monitoring)
- **L21-L31**: Decision features (risk assessment, stopping criteria)

## Resource Requirements

### Full Mode
- **GPU Memory**: ~40GB (LLaMA 8B + SAE)
- **Disk Space**: 2-3GB
- **Time**: 5-6 hours

### Attention Only Mode
- **GPU Memory**: ~40GB (LLaMA 8B, no SAE)
- **Disk Space**: 300MB
- **Time**: 3-4 hours

## Comparison with Original Exp 6

| Aspect | Original | Extended (Optimized) |
|--------|----------|----------------------|
| Layers | 3 (L8, L15, L31) | 31 (L1-L31) |
| File format | JSON | NPZ compressed |
| File size | 2.41 GB | 2-3 GB |
| Memory peak | 10 GB | 165 MB |
| Coverage | Sample layers | Complete network |
| Analysis | Spot check | Layer evolution |

## Memory Optimization Details

### Original Problem
```python
# BAD: All layers in memory + JSON conversion
all_layers = {}
for layer in range(1, 32):
    all_layers[f'L{layer}'] = {
        'features': features.tolist(),  # 4.58GB → 6.87GB
        'attention': attention.tolist()  # 554MB → 831MB
    }
# Peak: 10.24GB in memory!
json.dump(all_layers, f)  # File: 7-8GB
```

### Optimized Solution
```python
# GOOD: Process one layer at a time
for layer in range(1, 32):
    layer_data = {
        'features': features_array,  # numpy array (no conversion)
        'attention': attention_array
    }

    # Save immediately as NPZ
    np.savez_compressed(f'layer_{layer}.npz', **layer_data)

    # Free memory
    del layer_data
    torch.cuda.empty_cache()
    # Peak: Only 165MB (one layer)!
```

## Troubleshooting

**Out of Memory**:
- Use `--no-features` to save only attention
- Already optimized with layer-wise saving
- Should not OOM with incremental approach

**Too Slow**:
- Use `--no-features` for 40% faster extraction
- Already optimized with on-demand SAE loading

**Large Files**:
- Use `--no-features` for 7-8× smaller files
- NPZ compression already applied
- Features take ~90% of file size

**Missing NPZ files**:
- Check `/data/llm_addiction/experiment_6_L1_31_token_tracking/[scenario_name]/`
- Each scenario should have 31 NPZ files (one per layer)

## Next Steps After Collection

1. **Attention Pathway Analysis**:
   - Create heatmaps showing attention evolution L1→L31
   - Identify critical layers for each token type

2. **Feature Activation Analysis**:
   - Find which layers have strongest risky/safe differentiation
   - Track feature activation trajectories

3. **Cross-Reference**:
   - Compare with Exp 3 word associations
   - Validate attention patterns with feature meanings

4. **Visualization**:
   - Layer-by-layer attention flow diagrams
   - Feature activation timelines
   - Integrated feature-word-token analysis
