# L1-31 GLOBAL FDR Feature Extraction

## Purpose
Extract statistically valid SAE features from ALL 31 layers (L1-L31) using **GLOBAL FDR correction**.

## Why This Is CORRECT

### Wrong Method (Archived)
- Applied FDR correction **separately** for each layer
- Result: 20,630 features (inflated by multiple testing)
- Each layer's correction ignores the other 30 layers

### Correct Method (This Script)
- Flatten ALL p-values from ALL 31 layers
- Apply FDR correction **ONCE** across all ~1M tests (31 × 32,768)
- Result: Much smaller set of truly significant features

## Statistical Method

1. **Cohen's d**: Effect size calculation
2. **Welch's t-test**: No equal variance assumption (`equal_var=False`)
3. **Global FDR correction**: Benjamini-Hochberg on all layers simultaneously
4. **Selection criteria**:
   - |Cohen's d| ≥ 0.3 (medium effect)
   - p-value < 0.001 (very strict)
   - FDR α = 0.05 (5% false discovery rate)

## Files

- `extract_L1_31_GLOBAL_FDR.py`: Main extraction script
- `launch_L1_31_GLOBAL_FDR.sh`: Launch script with logging
- `logs/`: Execution logs

## Usage

```bash
# Default GPU 5
./launch_L1_31_GLOBAL_FDR.sh

# Custom GPU
./launch_L1_31_GLOBAL_FDR.sh 6
```

## Expected Runtime

- **Data loading**: ~5 minutes (14GB + 433MB)
- **SAE loading**: ~10 minutes (31 SAEs)
- **Feature extraction**: ~3-4 hours (6,400 samples × 31 layers)
- **Statistical analysis**: ~30 minutes (global FDR)
- **Total**: ~4-5 hours

## Output Files

- `/data/llm_addiction/results/L1_31_GLOBAL_FDR_features_YYYYMMDD_HHMMSS.npz`
- `/data/llm_addiction/results/L1_31_GLOBAL_FDR_analysis_YYYYMMDD_HHMMSS.json`

## Monitoring

```bash
# Attach to tmux session
tmux attach -t L1_31_global_fdr

# Check log
tail -f logs/L1_31_GLOBAL_FDR_gpu5_*.log

# Check progress (feature count)
grep "Selected features" logs/L1_31_GLOBAL_FDR_gpu5_*.log
```

## Comparison

| Method | Features Found | FDR Method |
|--------|---------------|-----------|
| Wrong (archived) | 20,630 | Per-layer |
| L25-31 reference | ~3,365 | Global |
| **This (L1-31)** | **TBD** | **Global** |

Expected: ~5,000-8,000 features (more layers than L25-31, but global FDR is strict)
