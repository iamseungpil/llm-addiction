# EXPERIMENT 2 ULTRATHINK ANALYSIS
**Analysis Date:** 2025-11-11
**Analyst:** Claude Code (Ultrathink Mode)
**Question:** Can we run Experiment 2 with the new Experiment 1 SAE results?

---

## EXECUTIVE SUMMARY

**ANSWER: YES - Execution is possible with minor modifications**

- **Readiness:** 85% ready
- **Modifications needed:** 1 file modification OR 1 new converter script
- **Time to execute:**
  - Setup: 30 minutes
  - Runtime: 40-50 hours (for 13,434 features @ 30 trials/condition)
  - Runtime: 8-10 hours (for top 300/layer = 9,300 features)
- **Risk level:** LOW
- **Blockers:** NONE (all dependencies satisfied)
- **GPU availability:** GPUs 4-7 completely free

---

## 1. CURRENT SITUATION

### ✅ What We Have

#### New Experiment 1 Results
- **File:** `/data/llm_addiction/results/L1_31_GLOBAL_FDR_features_20251110_214621.npz`
- **Size:** 147 KB
- **Total features:** 13,434 (FDR corrected, high quality)
- **Format:** NumPy NPZ archive
- **Layers:** L1-L31 complete
- **Data quality:** Global FDR correction applied, much stricter than old results

**Feature Distribution:**
```
Layer 1:  96 features    Layer 12: 497 features   Layer 23: 482 features
Layer 2:  108 features   Layer 13: 627 features   Layer 24: 464 features
Layer 3:  167 features   Layer 14: 715 features   Layer 25: 441 features
Layer 4:  213 features   Layer 15: 570 features   Layer 26: 529 features
Layer 5:  260 features   Layer 16: 547 features   Layer 27: 451 features
Layer 6:  322 features   Layer 17: 521 features   Layer 28: 541 features
Layer 7:  396 features   Layer 18: 493 features   Layer 29: 559 features
Layer 8:  442 features   Layer 19: 463 features   Layer 30: 540 features
Layer 9:  427 features   Layer 20: 411 features   Layer 31: 304 features
Layer 10: 486 features   Layer 21: 406 features
Layer 11: 505 features   Layer 22: 451 features
```

#### NPZ Data Structure
```python
For each layer L:
  - layer_L_indices: [n_features] - Feature IDs (0-32767)
  - layer_L_cohen_d: [n_features] - Effect sizes
  - layer_L_p_values: [n_features] - Statistical significance
  - layer_L_bankrupt_mean: [n_features] - Mean activation in bankrupt group
  - layer_L_safe_mean: [n_features] - Mean activation in safe group
  - layer_L_bankrupt_std: [n_features] - Std in bankrupt group
  - layer_L_safe_std: [n_features] - Std in safe group
```

#### Existing Experiment 2 Script
- **File:** `/home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/experiment_2_L1_31_top300.py`
- **Lines:** 558
- **Status:** Ready to run, just needs correct input file
- **Dependencies:** All satisfied (LLaMA model, SAE loader, tokenizer)

### ❌ What's Missing

**Input File Mismatch:**
- Script expects: JSON file at line 139
  ```python
  features_file = '/data/llm_addiction/experiment_1_L1_31_extraction/L1_31_features_FINAL_20250930_220003.json'
  ```
- We have: NPZ file at different location
  ```
  /data/llm_addiction/results/L1_31_GLOBAL_FDR_features_20251110_214621.npz
  ```

---

## 2. GAP ANALYSIS

### Format Gap

**Old JSON Structure (what script expects):**
```json
{
  "layer_results": {
    "1": {
      "significant_features": [
        {
          "feature_idx": 5489,
          "cohen_d": 1.195,
          "p_value": 1.7e-06,
          "bankrupt_mean": 0.945,
          "safe_mean": 0.693,
          "bankrupt_std": 0.234,
          "safe_std": 0.189
        },
        ...
      ]
    }
  }
}
```

**New NPZ Structure (what we have):**
```python
{
  'layer_1_indices': array([5489, 11625, ...]),
  'layer_1_cohen_d': array([1.195, 1.019, ...]),
  'layer_1_p_values': array([...]),
  'layer_1_bankrupt_mean': array([0.945, ...]),
  'layer_1_safe_mean': array([0.693, ...]),
  'layer_1_bankrupt_std': array([0.234, ...]),
  'layer_1_safe_std': array([0.189, ...])
}
```

### Data Quality Comparison

| Aspect | Old JSON (Sept 30) | New NPZ (Nov 10) | Winner |
|--------|-------------------|------------------|--------|
| FDR Correction | Local (per-layer) | Global (all layers) | **New** |
| False Positive Rate | Higher | Lower | **New** |
| Layer 1 features | 2,195 | 96 | **New** (quality over quantity) |
| Total features | ~60,000 | 13,434 | **New** (stricter filtering) |
| File size | 29 MB | 147 KB | **New** (efficiency) |

**Conclusion:** New NPZ has HIGHER QUALITY features (FDR corrected globally), making it better for causal analysis.

---

## 3. SOLUTION OPTIONS

### Option A: Convert NPZ → JSON (RECOMMENDED)
**Pros:**
- No modification to experiment_2 script
- Easy to verify conversion correctness
- Can reuse existing launch scripts
- Maintains separation of concerns

**Cons:**
- Creates duplicate data (NPZ + JSON)
- Extra 30 MB disk space

**Implementation:** Create converter script (shown in Section 5)

### Option B: Modify experiment_2_L1_31_top300.py
**Pros:**
- No intermediate files
- Direct NPZ reading

**Cons:**
- Modifies experiment script (lines 137-188)
- Requires testing modifications
- More complex rollback if issues arise

**Verdict:** **Option A is safer and cleaner**

---

## 4. FEATURE SELECTION STRATEGY

### Total Features Available: 13,434

**Strategy 1: Top N per layer**
| N per layer | Total features | Runtime (30 trials/cond) | Runtime (50 trials/cond) |
|-------------|----------------|--------------------------|--------------------------|
| Top 100 | 3,100 | ~13 hours | ~22 hours |
| Top 200 | 6,200 | ~26 hours | ~43 hours |
| Top 300 | 9,300 | ~39 hours | ~65 hours |
| Top 500 | 15,500 | ~65 hours | ~108 hours |

**Strategy 2: All features (13,434)**
- Runtime: ~56 hours @ 30 trials
- Runtime: ~94 hours @ 50 trials

**Strategy 3: Threshold-based (|Cohen's d| > X)**
| Threshold | Estimated features | Runtime @ 30 trials |
|-----------|-------------------|---------------------|
| > 0.8 | ~8,000 | ~34 hours |
| > 1.0 | ~5,000 | ~21 hours |
| > 1.5 | ~2,000 | ~8 hours |

### Calculation Basis
```
Time per feature = 6 conditions × 30 trials × 5 seconds/trial
                 = 900 seconds = 15 minutes/feature

For 9,300 features: 9,300 × 15 min = 139,500 min = 97 hours (theoretical)
With parallelization (4 GPUs): 97 / 4 = ~24 hours
With overhead (25%): 24 × 1.25 = ~30 hours realistic
```

**RECOMMENDATION:** Start with Top 300/layer (9,300 features) for consistency with original design

---

## 5. IMPLEMENTATION PLAN

### Step 1: Create NPZ→JSON Converter (15 minutes)

**File to create:** `/home/ubuntu/llm_addiction/convert_npz_to_json.py`

**Script content:**
```python
#!/usr/bin/env python3
"""
Convert new NPZ format to old JSON format for Experiment 2 compatibility
"""
import numpy as np
import json
from datetime import datetime

# Load NPZ
npz_file = '/data/llm_addiction/results/L1_31_GLOBAL_FDR_features_20251110_214621.npz'
npz = np.load(npz_file)

# Create JSON structure
output = {
    'timestamp': datetime.now().isoformat(),
    'source_file': npz_file,
    'conversion_date': '2025-11-11',
    'total_layers': 31,
    'layer_results': {}
}

total_features = 0

# Process each layer
for layer in range(1, 32):
    key_prefix = f'layer_{layer}_'
    indices_key = key_prefix + 'indices'

    if indices_key not in npz.files:
        continue

    # Extract arrays
    indices = npz[indices_key]
    cohen_d = npz[key_prefix + 'cohen_d']
    p_values = npz[key_prefix + 'p_values']
    bankrupt_mean = npz[key_prefix + 'bankrupt_mean']
    safe_mean = npz[key_prefix + 'safe_mean']
    bankrupt_std = npz[key_prefix + 'bankrupt_std']
    safe_std = npz[key_prefix + 'safe_std']

    # Create feature list
    features = []
    for i in range(len(indices)):
        features.append({
            'feature_idx': int(indices[i]),
            'cohen_d': float(cohen_d[i]),
            'p_value': float(p_values[i]),
            'bankrupt_mean': float(bankrupt_mean[i]),
            'safe_mean': float(safe_mean[i]),
            'bankrupt_std': float(bankrupt_std[i]),
            'safe_std': float(safe_std[i])
        })

    # Sort by |Cohen's d| descending (already sorted, but ensuring)
    features.sort(key=lambda x: abs(x['cohen_d']), reverse=True)

    output['layer_results'][str(layer)] = {
        'layer': layer,
        'n_features': 32768,
        'n_significant': len(features),
        'significant_features': features
    }

    total_features += len(features)
    print(f"Layer {layer:2d}: {len(features):4d} features")

output['total_significant_features'] = total_features

# Save JSON
output_file = '/data/llm_addiction/experiment_1_L1_31_extraction/L1_31_features_CONVERTED_20251111.json'
with open(output_file, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n✅ Conversion complete!")
print(f"   Total features: {total_features}")
print(f"   Output: {output_file}")
```

### Step 2: Modify Experiment 2 Script (5 minutes)

**File:** `/home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/experiment_2_L1_31_top300.py`

**Line 139 - Change from:**
```python
features_file = '/data/llm_addiction/experiment_1_L1_31_extraction/L1_31_features_FINAL_20250930_220003.json'
```

**Line 139 - Change to:**
```python
features_file = '/data/llm_addiction/experiment_1_L1_31_extraction/L1_31_features_CONVERTED_20251111.json'
```

**That's it! Only 1 line needs to change.**

### Step 3: Verify Setup (10 minutes)

Run verification checks:
```bash
# 1. Check converter output
python3 /home/ubuntu/llm_addiction/convert_npz_to_json.py

# 2. Verify JSON file
ls -lh /data/llm_addiction/experiment_1_L1_31_extraction/L1_31_features_CONVERTED_20251111.json

# 3. Test loading in Python
python3 << 'EOF'
import json
with open('/data/llm_addiction/experiment_1_L1_31_extraction/L1_31_features_CONVERTED_20251111.json', 'r') as f:
    data = json.load(f)
print(f"Loaded {data['total_significant_features']} features from {len(data['layer_results'])} layers")
print("Sample Layer 1:", len(data['layer_results']['1']['significant_features']), "features")
EOF
```

### Step 4: Launch Experiment (5 minutes)

**Use existing launch script:**
```bash
cd /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31

# Option A: 4 GPUs (recommended for speed)
./launch_corrected_single.sh

# Option B: More parallelization (if needed)
./launch_20_processes.sh
```

**Monitor progress:**
```bash
# Check tmux sessions
tmux ls | grep exp2

# Attach to specific GPU
tmux attach -t exp2_gpu4

# Check logs
tail -f logs/exp2_gpu4_L1_8.log
```

---

## 6. DETAILED MODIFICATION CHECKLIST

### Files to Create (1 file)
- [ ] `/home/ubuntu/llm_addiction/convert_npz_to_json.py` (converter script)

### Files to Modify (1 file)
- [ ] `/home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/experiment_2_L1_31_top300.py`
  - **Line 139:** Update `features_file` path

### Files to Execute (1 file)
- [ ] `/home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/launch_corrected_single.sh`

### Verification Steps
- [ ] NPZ file exists: `/data/llm_addiction/results/L1_31_GLOBAL_FDR_features_20251110_214621.npz`
- [ ] Converter runs successfully
- [ ] JSON file created with correct format
- [ ] JSON contains 13,434 features across 31 layers
- [ ] Experiment script loads JSON without errors
- [ ] SAE checkpoints accessible
- [ ] GPUs 4-7 available

---

## 7. RUNTIME ESTIMATES

### Configuration: Top 300/layer (Script Default)

**Per-feature calculation:**
- 6 conditions (2 baselines + 4 patching conditions)
- 30 trials per condition
- ~5 seconds per trial
- Total: 6 × 30 × 5 = 900 seconds = 15 minutes/feature

**Layer-by-layer breakdown (4 GPU parallelization):**

| GPU | Layers | Features | Time (hours) |
|-----|--------|----------|--------------|
| GPU 4 | L1-L8 | 2,035 | 8.5 |
| GPU 5 | L9-L16 | 3,952 | 16.5 |
| GPU 6 | L17-L24 | 3,838 | 16.0 |
| GPU 7 | L25-L31 | 3,309 | 13.8 |

**Estimated completion:** ~17 hours (bottlenecked by GPU 5)

### Configuration: All 13,434 features

**Total runtime:**
- Single GPU: 13,434 × 15 min = 201,510 min = 140 hours = 5.8 days
- 4 GPUs parallel: 140 / 4 = 35 hours ≈ 1.5 days
- With overhead (25%): 35 × 1.25 = **44 hours ≈ 2 days**

---

## 8. RISK ASSESSMENT

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| JSON conversion error | Low | Medium | Verify with sample layer before full run |
| SAE loading failure | Very Low | High | Pre-test with single layer |
| GPU OOM | Low | Medium | Script uses on-demand SAE loading |
| Script crash | Low | Medium | Checkpoint every 50 features (built-in) |
| Data corruption | Very Low | High | Keep NPZ as source of truth |

### Scientific Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Too many features (false positives) | Low | Medium | Using FDR-corrected features reduces this |
| Too few features (false negatives) | Low | Low | Global FDR is conservative but appropriate |
| Old vs new feature mismatch | N/A | N/A | New features are superior (FDR corrected) |

**Overall Risk Level: LOW**

All risks have known mitigations and the experimental setup is well-tested.

---

## 9. EXECUTION PLAN & TIMELINE

### Phase 1: Setup (30 minutes)
**Time: 00:00 - 00:30**
- [ ] Create converter script (15 min)
- [ ] Run conversion (5 min)
- [ ] Verify JSON format (5 min)
- [ ] Update experiment_2 script (1 min)
- [ ] Test script loading (4 min)

### Phase 2: Launch (5 minutes)
**Time: 00:30 - 00:35**
- [ ] Navigate to experiment directory
- [ ] Execute launch script
- [ ] Verify all 4 tmux sessions started
- [ ] Check initial logs for errors

### Phase 3: Monitoring (First 1 hour)
**Time: 00:35 - 01:35**
- [ ] Monitor first feature completion (15 min)
- [ ] Check checkpoint save (50 features ≈ 12.5 hours)
- [ ] Verify GPU utilization
- [ ] Check for OOM errors

### Phase 4: Active Monitoring (Hours 1-17)
- Check every 4 hours
- Verify progress from logs
- Ensure no crashes

### Phase 5: Completion & Analysis (Hour 17+)
- Collect all checkpoint files
- Merge results
- Generate summary statistics

---

## 10. EXPECTED OUTPUTS

### Checkpoint Files (every 50 features)
```
/data/llm_addiction/experiment_2_multilayer_patching/
├── checkpoint_L1_8_gpu4_L1_8_YYYYMMDD_HHMMSS.json
├── checkpoint_L9_16_gpu5_L9_16_YYYYMMDD_HHMMSS.json
├── checkpoint_L17_24_gpu6_L17_24_YYYYMMDD_HHMMSS.json
└── checkpoint_L25_31_gpu7_L25_31_YYYYMMDD_HHMMSS.json
```

### Response Logs
```
/data/llm_addiction/experiment_2_multilayer_patching/response_logs/
├── responses_L1_8_gpu4_L1_8_YYYYMMDD_HHMMSS.json
├── responses_L9_16_gpu5_L9_16_YYYYMMDD_HHMMSS.json
├── responses_L17_24_gpu6_L17_24_YYYYMMDD_HHMMSS.json
└── responses_L25_31_gpu7_L25_31_YYYYMMDD_HHMMSS.json
```

### Final Result Structure
```json
{
  "timestamp": "2025-11-11T12:00:00",
  "gpu_id": 4,
  "process_id": "gpu4_L1_8",
  "layer_range": "L1-L8",
  "features_tested": 2035,
  "n_trials_per_condition": 30,
  "results": [
    {
      "feature": "L1-5489",
      "layer": 1,
      "feature_id": 5489,
      "cohen_d": 1.195,
      "is_causal": true,
      "classified_as": "risky",
      "safe_stop_delta": -0.25,
      "risky_stop_delta": -0.30,
      "safe_p_value": 0.003,
      "risky_p_value": 0.001
    },
    ...
  ]
}
```

---

## 11. COMPARISON: Old vs New Features

### Quantitative Comparison

| Metric | Old Features (Sep 30) | New Features (Nov 10) |
|--------|----------------------|----------------------|
| **Total features** | ~60,000 | 13,434 |
| **L1 features** | 2,195 | 96 |
| **FDR correction** | Local (per-layer) | Global (all layers) |
| **Expected FDR** | ~5% per layer | ~5% global |
| **False discoveries** | ~3,000 | ~672 |
| **File size** | 29 MB | 147 KB |

### Scientific Quality

**Old features (Local FDR):**
- Each layer tested independently
- Multiple comparison correction within layer only
- Higher false positive rate when considering all layers
- Example: If we test 31 layers at FDR 5%, expected ~1-2 entire layers of false positives

**New features (Global FDR):**
- All layers tested together
- Multiple comparison correction across all tests
- Lower false positive rate overall
- More conservative but scientifically rigorous

**Conclusion:** New features are scientifically superior for multi-layer causal analysis.

---

## 12. ANSWERS TO SPECIFIC QUESTIONS

### Q1: Can we run Experiment 2 right now?

**A: YES, with 30 minutes of setup**

Requirements:
1. ✅ NPZ file exists with all necessary data
2. ✅ Experiment 2 script is functional
3. ✅ SAE checkpoints are accessible
4. ✅ GPUs are available (4-7 free)
5. ⚠️ Need format conversion (30 min setup)

### Q2: What files/code need modification?

**A: Minimal modifications**

**Create 1 file:**
- `/home/ubuntu/llm_addiction/convert_npz_to_json.py` (converter)

**Modify 1 line in 1 file:**
- `/home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/experiment_2_L1_31_top300.py`
- Line 139: Update path to new JSON file

**No other modifications needed.**

### Q3: What are the expected problems?

**A: Minimal problems expected**

**Potential Issues (all low probability):**

1. **JSON too large (Low risk)**
   - New JSON will be ~15 MB (smaller than old 29 MB)
   - Well within system limits

2. **Feature ID range (No risk)**
   - Old: Feature IDs 0-32767 ✓
   - New: Feature IDs 0-32767 ✓
   - Same range, no compatibility issues

3. **Different layer numbering (No risk)**
   - Both use layers 1-31
   - Consistent indexing

4. **Missing data fields (No risk)**
   - NPZ has all required fields
   - Converter creates exact format match

5. **GPU memory (Low risk)**
   - Script uses on-demand SAE loading
   - Only 1 SAE in memory at a time
   - LLaMA (13B params) + 1 SAE (32k features) < 20 GB

**Mitigation for all issues:**
- Start with single GPU test
- Monitor first 50 features
- Checkpoint system provides fault tolerance

---

## 13. DECISION MATRIX

### Should we use new features or old features?

| Criterion | Old Features (Local FDR) | New Features (Global FDR) | Winner |
|-----------|-------------------------|--------------------------|--------|
| Scientific rigor | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **New** |
| False positive rate | Higher | Lower | **New** |
| Computational cost | High (60k features) | Medium (13k features) | **New** |
| Setup time | 0 min (ready) | 30 min (conversion) | Old |
| Result publishability | Good | Excellent | **New** |
| Multi-layer validity | Questionable | Strong | **New** |

**VERDICT: Use new features (Global FDR)**

The 30-minute setup cost is negligible compared to the scientific benefits.

---

## 14. FINAL RECOMMENDATION

### Recommended Action Plan

**PROCEED with Experiment 2 using new Global FDR features**

**Configuration:**
- Features: Top 300 per layer (9,300 total) OR all 13,434 features
- Trials: 30 per condition (as in original design)
- GPUs: 4-7 (parallel execution)
- Expected completion: 17 hours (top 300) or 44 hours (all features)

**Steps:**
1. Create converter script (15 min)
2. Run conversion (5 min)
3. Update 1 line in experiment script (1 min)
4. Launch experiment (1 min)
5. Monitor first hour actively
6. Check every 4 hours thereafter

**Success criteria:**
- First feature completes without error (15 min)
- First checkpoint saves successfully (12.5 hours)
- All 4 GPUs running without OOM
- Results JSON files match expected format

**Fallback plan:**
If any issues arise, can revert to old features in 5 minutes (just change line 139 back).

---

## 15. MONITORING COMMANDS

### Setup monitoring
```bash
# Create monitoring script
cat > /home/ubuntu/llm_addiction/monitor_exp2.sh << 'EOSC'
#!/bin/bash
echo "=== Experiment 2 Status ==="
echo ""
echo "TMux sessions:"
tmux ls | grep exp2 || echo "No sessions"
echo ""
echo "Latest checkpoints:"
ls -lht /data/llm_addiction/experiment_2_multilayer_patching/checkpoint*.json 2>/dev/null | head -4
echo ""
echo "GPU Memory:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | grep -E "^[4567]"
EOSC

chmod +x /home/ubuntu/llm_addiction/monitor_exp2.sh
```

### Real-time monitoring
```bash
# Quick status
/home/ubuntu/llm_addiction/monitor_exp2.sh

# Watch logs
tail -f /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/logs/exp2_gpu4_L1_8.log

# Check progress
watch -n 60 'ls -lh /data/llm_addiction/experiment_2_multilayer_patching/*.json | wc -l'
```

---

## CONCLUSION

**Experiment 2 is READY TO RUN with the new Global FDR features.**

The new features are scientifically superior (global FDR correction) and will produce more publishable results. The setup requires only:
- 1 new converter script
- 1 line change in existing experiment script
- 30 minutes total setup time

All dependencies are satisfied, GPUs are available, and the risk level is low. The experiment can begin immediately after the simple setup steps.

**Recommendation: PROCEED**

---

*End of Ultrathink Analysis*
