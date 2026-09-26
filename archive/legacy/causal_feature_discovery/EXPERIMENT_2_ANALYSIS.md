# Experiment 2: Multi-Layer Population Mean Patching Analysis

## Executive Summary

**NEW EXPERIMENT 2 DESIGN COMPLETED**

Based on analysis of the existing population mean patching experiment, I have designed and implemented a new Experiment 2 that meets all specified requirements:

### Key Improvements Over Original Design

1. **Feature Scale**: Uses 1,340 high-effect features (|Cohen's d| > 0.8) from new multilayer data vs 356 from old data
2. **Multi-Layer Support**: Tests all layers 25-31 vs only layers 25, 30
3. **3-Condition Design**: safe_mean, risky_mean, baseline (no patching) vs 3-scale interpolation
4. **Statistical Power**: 50 trials per condition vs 30 trials
5. **Memory Efficiency**: On-demand SAE loading vs keeping all SAEs in memory

## Analysis of Existing Population Mean Patching

### 🔵 STRENGTHS IDENTIFIED

1. **Proven Methodology**: The interpolation approach using safe_mean ↔ bankrupt_mean is mathematically sound and has demonstrated effectiveness (142 causal features found in past experiments)

2. **Robust Statistical Analysis**: Uses Spearman correlation to detect monotonic relationships, with appropriate p-value thresholds and effect size requirements

3. **Dual Prompt Testing**: Tests both risky ($20 balance) and safe ($140 balance) conditions to validate feature effects across contexts

4. **Effective Feature Patching**: Forward hook implementation successfully modifies SAE feature values without breaking generation quality

### 🟡 AREAS FOR IMPROVEMENT

1. **Limited Layer Coverage**: Only tested layers 25 and 30, missing intermediate layers 26-29 and final layer 31

2. **Baseline Comparison**: No true baseline (no patching) condition for control comparison

3. **Memory Management**: Loads all SAEs simultaneously, limiting scalability

## New Feature Data Validation

### ✅ FEATURE QUALITY CONFIRMED

**Total Features**: 3,365 across layers 25-31
- Layer 25: 441 features
- Layer 26: 529 features  
- Layer 27: 451 features
- Layer 28: 541 features
- Layer 29: 559 features
- Layer 30: 540 features
- Layer 31: 304 features

**Statistical Significance**: 
- All features: p < 0.001 (100%)
- High-effect features (|Cohen's d| > 0.8): 1,340 features (39.8%)
- Average |Cohen's d|: 0.73

**Selection Criteria Applied**: |Cohen's d| > 0.8 reduces feature set from 3,365 to 1,340, focusing on features with large effect sizes while maintaining statistical rigor.

## New Experiment 2 Design

### Experimental Structure

**3-Condition Design Per Feature**:
1. **safe_mean**: Patch feature to safe population mean
2. **risky_mean**: Patch feature to bankrupt population mean  
3. **baseline**: No patching (natural feature value)

**Dual Prompt Testing**:
- **Risky prompt**: $20 balance, recent losses, variable betting
- **Safe prompt**: $140 balance, mixed results, fixed betting

**Statistical Power**: 50 trials × 3 conditions × 2 prompts = 300 trials per feature

### Metrics Measured

**Primary Metrics**:
- **avg_bet**: Average betting amount
- **stop_rate**: Frequency of voluntary stopping
- **bankruptcy_risk**: 1 - stop_rate

**Causality Criteria** (Stricter than original):
- Correlation threshold: |r| > 0.7 (vs 0.5)
- P-value threshold: p < 0.05
- Effect size thresholds: bet_effect > $8, stop_effect > 0.15

### Technical Implementation

**Memory Optimization**:
```python
# On-demand SAE loading
def load_sae(self, layer: int):
    if layer not in self.sae_cache:
        self.sae_cache[layer] = LlamaScopeDirect(layer=layer, device="cuda")
    return self.sae_cache[layer]

def unload_sae(self, layer: int):
    if layer in self.sae_cache:
        del self.sae_cache[layer]
        torch.cuda.empty_cache()
```

**Multi-Layer Support**:
- Supports layers 25-31 through LlamaScopeDirect
- Sequential layer processing to minimize GPU memory usage
- Automatic SAE cleanup when switching layers

## Runtime Analysis

### Performance Estimates

**Total Trials**: 1,340 features × 300 trials = 402,000 trials
**GPU Parallelization**: 
- GPU 4: Features 0-669 (670 features)
- GPU 5: Features 670-1339 (670 features)

**Runtime Per GPU**:
- Trials per GPU: 201,000
- Time per trial: ~0.7 seconds (including patching overhead)
- **Estimated completion**: ~39 hours per GPU

### Monitoring and Management

**Intermediate Saves**: Every 25 features tested
**Progress Tracking**: 
```bash
# Monitor active processes
tmux attach -t exp2_gpu4
tmux attach -t exp2_gpu5

# Check progress logs
tail -f exp2_gpu4_*.log
tail -f exp2_gpu5_*.log
```

## Potential Issues and Mitigations

### 🔴 CRITICAL CONSIDERATIONS

1. **Runtime Scale**: 39 hours per GPU is substantial. Consider reducing to 30 trials if needed.

2. **Multiple Testing**: Testing 1,340 features requires Bonferroni correction (p < 0.05/1340 ≈ 3.7e-5) for family-wise error control.

3. **GPU Memory**: Multi-layer SAEs require careful memory management. On-demand loading mitigates this.

### 🟡 MONITORING POINTS

1. **Causality Rate**: Expect ~20-40% causality rate based on past experiments. If much lower, may need to adjust criteria.

2. **Parse Failures**: Monitor invalid trial rates. High rates may indicate generation quality issues.

3. **Layer Performance**: Track whether certain layers show higher causality rates.

## Implementation Files

### Core Implementation
- `experiment_2_multilayer_population_mean.py`: Main experiment code
- `run_experiment_2_parallel.sh`: GPU parallelization script
- `validate_experiment_2_design.py`: Design validation on subset

### Usage Commands

**Validation Run** (15 minutes on GPU 6):
```bash
cd /home/ubuntu/llm_addiction/causal_feature_discovery/src
python validate_experiment_2_design.py
```

**Full Experiment** (39 hours per GPU):
```bash
cd /home/ubuntu/llm_addiction/causal_feature_discovery/src
./run_experiment_2_parallel.sh
```

**Individual GPU Run**:
```bash
# GPU 4 only
python experiment_2_multilayer_population_mean.py --gpu 4 --start_idx 0 --end_idx 670 --process_id gpu4

# GPU 5 only  
python experiment_2_multilayer_population_mean.py --gpu 5 --start_idx 670 --end_idx 1340 --process_id gpu5
```

## Expected Results

Based on past experiments and the new high-effect feature selection:

**Projected Outcomes**:
- **Causal features (betting)**: ~200-400 features (15-30%)
- **Causal features (stopping)**: ~200-400 features (15-30%)
- **Total unique causal features**: ~300-600 features (22-45%)

**Quality Improvements**:
- Higher effect sizes due to |Cohen's d| > 0.8 filtering
- Better layer coverage with 25-31 testing
- More rigorous causality criteria reducing false positives

## Recommendations

### Before Running Full Experiment

1. **Run Validation**: Execute `validate_experiment_2_design.py` to confirm design works
2. **Check GPU Memory**: Ensure GPUs 4 and 5 have sufficient memory for LLaMA + SAE
3. **Verify Environment**: Confirm `llama_sae_env` conda environment is activated

### During Execution

1. **Monitor Progress**: Check tmux sessions and logs regularly
2. **GPU Health**: Run `nvidia-smi` periodically to check memory usage
3. **Intermediate Results**: Review intermediate saves to track causality rates

### Post-Experiment

1. **Result Analysis**: Combine GPU 4 and GPU 5 results for comprehensive analysis  
2. **Multiple Testing Correction**: Apply appropriate p-value corrections
3. **Cross-Validation**: Consider validating top causal features with additional trials

---

**Status**: ✅ Ready for execution
**Next Steps**: Run validation, then full experiment
**Expected Completion**: 2-3 days for full experiment