# Experiment 3 Extended: L1-31 Feature-Word Analysis

## Overview

This extends Experiment 3's feature-word co-occurrence analysis to **ALL 87,012 significant features** from L1-31 extraction.

## What This Does

**Original Experiment 3**: Analyzed 441 causal features to find which words appear more frequently when each feature is highly activated.

**This Extension**: Does the same analysis for ALL 87,012 features across layers 1-31.

### Method
1. Load all 87,012 significant features from L1-31 extraction
2. For each feature:
   - Extract SAE activations from 6,400 Exp1 responses
   - Split responses into high/low activation groups (by median)
   - Count word frequencies in each group
   - Find words with significant frequency differences

3. Output: Which words are associated with high activation of each feature

## Data Sources

- **Features**: 87,012 significant features from `/data/llm_addiction/experiment_1_L1_31_extraction/`
- **Responses**: 6,400 LLaMA responses from Exp1 (with response text)

## Files

- `experiment_3_L1_31_extended.py` - Main analysis script
- `launch_test.sh` - Test with Layer 25 only (~3,478 features)
- `launch_full.sh` - Run full analysis on 4 GPUs (all 87,012 features)

## Usage

### Test Run (Layer 25 only)

```bash
cd /home/ubuntu/llm_addiction/experiment_3_L1_31_word_analysis
./launch_test.sh
```

This will:
- Analyze ~3,478 features from Layer 25
- Take ~5-10 hours on 1 GPU
- Save checkpoints every 500 features
- Output: `/data/llm_addiction/experiment_3_L1_31_word_analysis/final_L25_25_*.json`

### Full Run (All L1-31)

```bash
cd /home/ubuntu/llm_addiction/experiment_3_L1_31_word_analysis
./launch_full.sh
```

This will launch 4 parallel jobs:
- **GPU 0**: Layers 1-8   (~15,000 features, ~20 hours)
- **GPU 1**: Layers 9-16  (~20,000 features, ~30 hours)
- **GPU 2**: Layers 17-24 (~25,000 features, ~35 hours)
- **GPU 3**: Layers 25-31 (~27,000 features, ~40 hours)

**Total time**: ~40 hours (parallel)

Monitor with:
```bash
tmux attach -t exp3_L1_8    # GPU 0
tmux attach -t exp3_L9_16   # GPU 1
tmux attach -t exp3_L17_24  # GPU 2
tmux attach -t exp3_L25_31  # GPU 3
```

## Output Format

Each result file contains:

```json
{
  "timestamp": "20251017_120000",
  "gpu": 0,
  "layers": "1-8",
  "n_features": 15234,
  "results": [
    {
      "feature": "L5-1234",
      "layer": 5,
      "feature_id": 1234,
      "type": "safe",
      "cohens_d": -1.234,
      "n_responses": 6234,
      "median_activation": 2.45,
      "high_activation_words": [
        {"word": "balance", "high_freq": 0.12, "low_freq": 0.08, "diff": 0.04},
        {"word": "stop", "high_freq": 0.10, "low_freq": 0.06, "diff": 0.04}
      ],
      "low_activation_words": [...]
    }
  ]
}
```

## Expected Results

After completion, you'll have:

1. **Feature-Word associations** for all 87,012 features
2. **Cross-reference capability**:
   - Compare with Experiment 6 (token attention)
   - Identify which features attend to "balance" tokens AND mention "balance" words
   - Find features that mention "win", "lose", "stop" etc.

3. **Layer-wise word patterns**:
   - Early layers (L1-L10): What words are processed?
   - Middle layers (L11-L20): How do patterns evolve?
   - Late layers (L21-L31): Final decision-making vocabulary?

## Estimated Resource Usage

- **GPU Memory**: ~40GB per GPU (LLaMA 8B + SAE)
- **Disk Space**: ~50-100GB total results
- **Time**: 30-40 hours on 4 GPUs (parallel)
- **CPU RAM**: ~50GB for loading 6,400 responses

## Troubleshooting

**Out of Memory**:
- Reduce batch size (already using 1 response at a time)
- Use smaller layer ranges

**Too Slow**:
- Already parallelized across 4 GPUs
- Can split further if more GPUs available

**Missing Data**:
- Ensure `/data/llm_addiction/results/exp1_multiround_intermediate_20250819_140040.json` exists
- Ensure L1-31 extraction file exists

## Next Steps After Completion

1. **TF-IDF Analysis**: Run TF-IDF enhancement (like original Exp3)
2. **Cross-Reference**: Compare with Exp6 token attention
3. **Visualization**: Create word clouds per layer
4. **Statistical Testing**: Identify most discriminative words across all layers
