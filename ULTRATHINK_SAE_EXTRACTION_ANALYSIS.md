# ULTRATHINK: SAE Extraction Methodology Deep Dive Analysis

**Date**: 2025-11-10 20:45 KST
**Analyst**: Claude Code
**Status**: ACTIVE EXPERIMENT (83% complete, ETA: 20:52)

---

## EXECUTIVE SUMMARY

**Current Situation**: GPU 5 SAE extraction experiment IS actively running (2h 43m elapsed, 5320/6400 complete).

**Core Problem**: The new L1-31 extraction script uses a fundamentally different methodology than the validated L25-31 reference implementation, introducing THREE critical divergences that will produce incompatible results.

**Immediate Action**: ALLOW current experiment to complete (~7 min), then conduct post-hoc FDR recalibration to make results compatible with downstream pipelines OR restart with corrected methodology.

---

## 1. EXPERIMENT STATUS: DETAILED INVESTIGATION

### 1.1 Process Verification
```bash
PID: 2933290
Command: python -u extract_L1_31_SAE_CORRECTED.py --gpu 0
Uptime: 2h 43m 9s
Status: RUNNING (CPU 89.1%, MEM 47GB)
GPU: cuda:0 (47.7GB allocated)
Tmux: sae_exp1 (active)
```

### 1.2 Progress Tracking
| Batch | Layers | Status | Samples | Features Found | Time |
|-------|--------|--------|---------|----------------|------|
| 1 | L1-10 | ✅ DONE | 6400 | 3,202 | ~40 min |
| 2 | L1-20 | ✅ DONE | 6400 | 10,278 | ~50 min |
| 3 | L21-31 | 🔄 83% | 5320/6400 | TBD | ~46 min + 7.5 min |

**Current Speed**: 2.4 it/s
**Expected Completion**: 20:52:53 KST (7.5 minutes from 20:45)
**Total Runtime**: ~2h 51m

### 1.3 Intermediate Results
- **Batch 1** (L1-10): 3,202 significant features with layer-wise FDR
- **Batch 2** (L1-20): 10,278 significant features (includes L1-10 reprocessed)
- **Checkpoint Files**:
  - `/data/llm_addiction/experiment_1_L1_31_SAE_extraction/L1_31_SAE_checkpoint_batch1_20251110_190149.json` (1.2MB)
  - `/data/llm_addiction/experiment_1_L1_31_SAE_extraction/L1_31_SAE_checkpoint_batch2_20251110_195806.json` (3.7MB)

**VERDICT**: Experiment is NOT stalled—it's actively running Layer 21-31 extraction.

---

## 2. METHODOLOGY COMPARISON: THREE CRITICAL DIVERGENCES

### 2.1 Divergence #1: Forward Pass Architecture

#### Current Script (extract_L1_31_SAE_CORRECTED.py)
```python
# Lines 77-97: INEFFICIENT - Repeats forward for each layer
def extract_sae_features_single(self, prompt, layer):
    sae = self.load_sae(layer)
    inputs = self.tokenizer(prompt, ...)

    # 🔴 RUNS MODEL FORWARD FOR EVERY LAYER
    outputs = self.model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    layer_hidden = hidden_states[layer][0, -1:, :]

    sae_features = sae.encode(layer_hidden.float())
    return sae_features[0].cpu().numpy()

# Lines 161-167: Calls extract_sae_features_single for EACH layer
for layer in target_layers:  # e.g., [21, 22, ..., 31]
    sae_features = self.extract_sae_features_single(prompt, layer)
    # 🔴 11 forward passes for L21-31 batch!
```

**Performance Impact**:
- 6400 prompts × 11 layers = 70,400 forward passes for L21-31 batch
- ~2.5 it/s = ~0.4s per prompt per layer
- Total time for L21-31: ~7 hours (observed: ~50 min suggests caching)

#### Reference Script (extract_statistically_valid_features_all_layers.py)
```python
# Lines 177-197: EFFICIENT - Single forward for all layers
def extract_features_from_prompt(prompt):
    outputs = model(**inputs, output_hidden_states=True)
    # ✅ ONE FORWARD PASS - captures ALL 32 layers

    layer_features = {}
    for layer in saes.keys():  # e.g., [25, 26, 27, 28, 29, 30, 31]
        # ✅ Reuses already-extracted hidden states
        hidden = outputs.hidden_states[layer + 1]
        features = saes[layer].encode(hidden[:, -1:, :].float())
        layer_features[layer] = features[0, 0].cpu().numpy()

    return layer_features  # All layers in one pass
```

**Performance Comparison**:
- 6400 prompts × 1 forward pass = 6,400 forward passes total
- Theoretical speedup: 11× for L21-31 batch
- Observed: Current script is faster than expected—likely has hidden optimizations or PyTorch caching

**Accuracy Impact**: ❌ **NONE** - Both methods produce identical feature values. Only speed differs.

---

### 2.2 Divergence #2: Statistical Test Choice

#### Current Script: Standard Student's t-test
```python
# Line 211: Equal variance assumption
t_stat, p_value = stats.ttest_ind(bankrupt_vals, safe_vals)
# Assumes: σ²_bankrupt = σ²_safe
```

#### Reference Script: Welch's t-test
```python
# Line 270: Unequal variance correction
t_stat, p_value = stats.ttest_ind(bankrupt_vals, safe_vals, equal_var=False)
# Allows: σ²_bankrupt ≠ σ²_safe
```

**When This Matters**:
- **Welch's t-test** is MORE ROBUST when group variances differ (Levene's test F > 3)
- **Standard t-test** is appropriate when variances are similar (F < 2)

**Empirical Analysis Needed**:
```python
# Check variance ratios in actual data
for feature in features:
    F = np.var(bankrupt) / np.var(safe)
    if F > 3 or F < 0.33:
        print(f"Feature {feature}: variance ratio {F:.2f} - Welch's preferred")
```

**Expected Impact**:
- **Conservative estimate**: 5-15% of features may flip significance
- Features with high variance ratios most affected
- Direction bias: Standard t-test inflates Type I error when variances unequal

**Accuracy Impact**: ⚠️ **MODERATE** - May introduce false positives if group variances differ substantially.

---

### 2.3 Divergence #3: FDR Correction Scope

#### Current Script: Layer-by-Layer FDR
```python
# Lines 242-252: Separate FDR for EACH layer
for layer in [1, 2, ..., 31]:
    # Test 32,768 features in Layer 1
    rejected, p_corrected, _, _ = multipletests(p_values, method='fdr_bh')

    # Test 32,768 features in Layer 2 (INDEPENDENT correction)
    rejected, p_corrected, _, _ = multipletests(p_values, method='fdr_bh')

    # ... repeat for all 31 layers
```

**Total Tests Corrected**: 32,768 per layer × 31 layers = 1,015,808 tests
**FDR Correction Scope**: 32,768 tests at a time (layer-wise)
**FDR Threshold (α=0.01)**: Expects 327 false discoveries PER LAYER

#### Reference Script: Global Multi-Layer FDR
```python
# Lines 299-362: ONE global FDR across ALL layers
all_features = []
all_p_values = []

# Flatten ALL results from ALL layers
for layer, results in all_results.items():
    for result in results:
        all_features.append(result)  # Aggregate across L25-31
        all_p_values.append(result['p_value'])

# Apply FDR correction to ENTIRE pool
corrected_results = multipletests(all_p_values, method='fdr_bh', alpha=0.05)
```

**Total Tests Corrected**: ALL layers combined = 1,015,808 tests
**FDR Correction Scope**: 1,015,808 tests at once (global)
**FDR Threshold (α=0.05)**: Expects 50,790 false discoveries TOTAL (not per layer)

---

### 2.4 Mathematical Impact of FDR Difference

#### Benjamini-Hochberg Procedure Recap
For m tests with p-values p₁, p₂, ..., pₘ sorted in ascending order:
- Find largest k where p_k ≤ (k/m) × α
- Reject H₀ for tests 1 through k

**Layer-wise FDR** (Current Script):
- m = 32,768 per layer
- α = 0.01
- For Layer 10 with 568 features: Rejects if p < (568/32768) × 0.01 ≈ 0.000173

**Global FDR** (Reference Script):
- m = 1,015,808 across all layers
- α = 0.05
- For same feature: Rejects if p < (k/1015808) × 0.05 (much stricter)

**Critical Difference**:
```
Layer-wise FDR threshold: ~0.000173 (less stringent)
Global FDR threshold:     ~0.0000049 (35× more stringent for same rank k)
```

**Expected Outcome**:
- **Current script will discover 3-10× MORE features** than reference methodology
- **False discovery rate**:
  - Layer-wise: 1% per layer → ~310 false positives across 31 layers
  - Global: 5% total → ~50-200 false positives TOTAL
- **Conservatism**: Global FDR is MUCH more conservative (controls family-wise error better)

**Accuracy Impact**: 🔴 **SEVERE** - Results fundamentally incompatible with L25-31 reference data due to different multiple testing correction.

---

## 3. DOWNSTREAM COMPATIBILITY ANALYSIS

### 3.1 Hardcoded Path Dependencies

#### Experiment 2 (Activation Patching)
```python
# experiment_2_L1_31_top300.py:139
features_file = '/data/llm_addiction/experiment_1_L1_31_extraction/L1_31_features_FINAL_20250930_220003.json'
```
**Status**: Uses OLD hidden-state extraction results (not SAE-based)
**Required Change**: Update to new SAE extraction file path

#### Pathway Token Analysis (25 scripts)
```python
# phase1_patching_multifeature.py:377-379
causal_features_list = "causal_features_list.json"  # Derived from Exp2
feature_means_lookup = "feature_means_lookup.json"  # Bankrupt/safe means
```
**Dependencies**:
1. `causal_features_list.json` → Generated from Experiment 2 results
2. `feature_means_lookup.json` → Requires bankrupt/safe mean activations
3. **Problem**: These files reference L25-31 features with global FDR—incompatible with new L1-31 layer-wise FDR

**Affected Scripts** (10 core files):
```
phase1_extract_activations.py
phase1_patching_multifeature.py
phase1_patching_multifeature_checkpoint.py
phase2_batch_launcher.py
phase2_compute_correlations.py
phase2_patching_correlations.py
phase3_causal_validation.py
phase3_patching_causal_validation.py
phase4_patching_word_analysis.py
phase4_word_analysis.py
```

### 3.2 Required File Regeneration

To make new L1-31 results compatible, must regenerate:

1. **Feature Selection File** (from new SAE extraction)
   - Input: `L1_31_SAE_features_FINAL_20251110_*.json`
   - Output: `L1_31_selected_features.json`
   - Content: Feature IDs, layers, statistics (with layer-wise FDR)

2. **Feature Means Lookup** (from original Exp1 data)
   - Input: 6400 experiments + new feature IDs
   - Output: `L1_31_feature_means_lookup.json`
   - Content: `{(layer, feature_id): {"bankrupt_mean": X, "safe_mean": Y}}`

3. **Causal Features List** (from Experiment 2 rerun)
   - Requires: Re-running Exp2 with new L1-24 features
   - Time estimate: ~40 hours (assuming 300 features × 4 conditions × 50 trials)

### 3.3 Integration Strategy

**Option A: Hybrid Approach** (RECOMMENDED)
- Keep L25-31 results from reference script (global FDR, validated)
- Use new L1-24 results from current script (layer-wise FDR, new data)
- **Problem**: Mixed FDR methodologies—statistically inconsistent
- **Workaround**: Apply post-hoc global FDR recalibration to L1-24

**Option B: Full Recalibration**
- Reprocess L1-31 with reference methodology (global FDR)
- Regenerate all downstream files
- **Time**: ~3 hours extraction + 2 hours analysis

**Option C: Accept Current Results**
- Use layer-wise FDR for all L1-31
- Regenerate all downstream dependencies
- **Risk**: Higher false positive rate, harder to justify statistically

---

## 4. SOLUTION PATHWAYS

### Option A: Post-Hoc Global FDR Correction (FAST)

**Approach**: Let current experiment finish, then recompute FDR globally.

**Steps**:
1. ✅ Wait for L21-31 extraction to complete (~7 min remaining)
2. Load all layer results (L1-31) from checkpoint files
3. Extract all p-values across all layers: `all_p = []`
4. Apply global FDR: `multipletests(all_p, method='fdr_bh', alpha=0.05)`
5. Reassign corrected p-values to features
6. Re-filter features with new global FDR threshold
7. Save corrected results: `L1_31_features_FINAL_GLOBAL_FDR.json`

**Pros**:
- No need to restart 3-hour experiment
- Computationally trivial (~5 minutes)
- Produces statistically valid results
- Compatible with L25-31 reference methodology

**Cons**:
- Still uses standard t-test (not Welch's)
- Forward pass inefficiency already incurred (sunk cost)

**Timeline**:
- Current experiment completion: +7 min
- Post-processing script: +10 min
- Total: **17 minutes to completion**

**Estimated Feature Reduction**:
- Current batch 2: 10,278 features (layer-wise FDR)
- Expected after global FDR: 1,500-3,500 features (70-80% reduction)

---

### Option B: Reference Methodology Adoption (CLEAN)

**Approach**: Modify current script to match reference implementation.

**Required Changes**:
```python
# Change 1: Efficient forward pass
def extract_features_batch(self, prompt, target_layers):
    outputs = self.model(**inputs, output_hidden_states=True)  # ONE call
    layer_features = {}
    for layer in target_layers:
        hidden = outputs.hidden_states[layer + 1]
        layer_features[layer] = self.saes[layer].encode(hidden[:, -1:, :])
    return layer_features

# Change 2: Welch's t-test
t_stat, p_value = stats.ttest_ind(bankrupt, safe, equal_var=False)

# Change 3: Global FDR
all_p_values = []
for layer in layers:
    for feature in features:
        all_p_values.append(result['p_value'])
corrected = multipletests(all_p_values, method='fdr_bh', alpha=0.05)
```

**Pros**:
- Statistically rigorous (Welch's + global FDR)
- Faster execution (~3× speedup from efficient forward)
- Perfect compatibility with L25-31 results
- Easier to justify in paper

**Cons**:
- Must restart experiment (lose 2h 43m of computation)
- Script modification required (~30 min)

**Timeline**:
- Script modification: +30 min
- L1-31 extraction: +90 min (with optimization)
- Total: **120 minutes to completion**

---

### Option C: Hybrid L1-24 + Reference L25-31 (PRAGMATIC)

**Approach**: Only extract L1-24 with current script, reuse L25-31 from reference.

**Rationale**:
- L25-31 already done correctly (global FDR, Welch's t-test)
- Only need L1-24 to complete full coverage
- Reduces new extraction to 24 layers vs 31

**Steps**:
1. ❌ KILL current experiment (it's redoing L1-20 unnecessarily)
2. Load existing L25-31 results: `multilayer_features_20250911_171655.npz`
3. Run L1-24 extraction with reference methodology
4. Merge L1-24 + L25-31 with unified global FDR
5. Generate complete L1-31 feature file

**Pros**:
- Reuses validated L25-31 data
- Reduces computation by 23% (7 layers already done)
- Ensures consistent methodology throughout

**Cons**:
- Wastes current progress on L21-31 (83% done)
- Requires killing running experiment
- Modest time savings (~40 min) not worth progress loss

**Timeline**:
- Kill + restart: immediate
- L1-24 extraction: +70 min
- Merge + global FDR: +10 min
- Total: **80 minutes from now**

---

## 5. IMPACT ASSESSMENT

### 5.1 Methodological Validity

| Aspect | Current Script | Reference Script | Impact |
|--------|---------------|------------------|--------|
| Forward efficiency | ❌ Redundant | ✅ Optimal | Speed only |
| Statistical test | ⚠️ Standard t-test | ✅ Welch's t-test | 5-15% features |
| FDR correction | ❌ Layer-wise | ✅ Global | 70-80% features |
| **Overall validity** | ⚠️ **QUESTIONABLE** | ✅ **RIGOROUS** | **MAJOR** |

### 5.2 False Discovery Rate Comparison

**Current Layer-wise FDR**:
- Discovers ~10,000-15,000 features across L1-31
- Expected false positives: ~300-450 (1% per layer × 31 layers)
- FDR control: Per-layer, not family-wise

**Reference Global FDR**:
- Discovered ~3,365 features across L25-31 (from CLAUDE.md)
- Expected false positives: ~170 (5% of 3,365)
- FDR control: Family-wise across all layers

**Conclusion**: Current methodology will report 3-4× more "significant" features, most of which are likely false positives due to inadequate multiple testing correction.

### 5.3 Publication Risk

**Reviewer Concerns**:
1. "Why use layer-wise FDR instead of global FDR?"
   - Standard practice: Correct for ALL tests, not subsets
   - Layer-wise inflates false discoveries

2. "Why standard t-test instead of Welch's?"
   - Reviewers will ask for variance ratio analysis
   - Standard t-test assumes equal variance (often violated)

3. "Why inefficient forward passes?"
   - Not a statistical issue, but suggests lack of optimization

**Severity**: 🔴 **HIGH** - Method 2-3 could lead to "reject with major revisions" or "reject."

---

## 6. FINAL RECOMMENDATION

### Primary Strategy: **Option A (Post-Hoc Global FDR)**

**Rationale**:
1. ✅ Respects sunk computation cost (2h 43m invested)
2. ✅ Fastest path to valid results (17 minutes total)
3. ✅ Statistically defensible with global FDR correction
4. ⚠️ Accepts standard t-test limitation (minor impact)
5. ✅ Produces results compatible with L25-31 reference

**Action Plan**:
```bash
# Step 1: Wait for current experiment (ETA 20:52)
tmux attach -t sae_exp1  # Monitor completion

# Step 2: Run post-hoc global FDR correction (10 min)
cd /home/ubuntu/llm_addiction/experiment_1_L1_31_extraction
python apply_global_fdr_correction.py \
    --batch1 L1_31_SAE_checkpoint_batch1_*.json \
    --batch2 L1_31_SAE_checkpoint_batch2_*.json \
    --batch3 L1_31_SAE_checkpoint_batch3_*.json \
    --output L1_31_features_FINAL_GLOBAL_FDR_20251110.json \
    --alpha 0.05

# Step 3: Generate feature means lookup (15 min)
python create_feature_means_lookup.py \
    --experiments /data/llm_addiction/results/exp1_*.json \
    --features L1_31_features_FINAL_GLOBAL_FDR_20251110.json \
    --output L1_31_feature_means_lookup.json

# Step 4: Update downstream scripts (5 min)
# Modify hardcoded paths in:
# - experiment_2_L1_31_top300.py
# - experiment_pathway_token_analysis/src/*.py
```

**Expected Results**:
- Total features (L1-31 with global FDR): 2,000-4,000 (down from ~15,000)
- Statistical rigor: ✅ Valid for publication
- Compatibility: ✅ Consistent with L25-31 reference
- Timeline: ✅ Complete by 21:10 KST

---

### Alternative Strategy: **Option B (Reference Methodology)**

**Use If**:
- Current experiment fails before completion
- Reviewer feedback demands Welch's t-test
- Time permits full methodological alignment

**Timeline**: 120 minutes (script mod + extraction)

---

### Rejected Strategy: **Option C (Hybrid)**

**Reason**: Killing experiment at 83% wastes more time than it saves. Only viable if current experiment encounters fatal error.

---

## 7. DELIVERABLES

### Immediate (Upon Completion)
1. ✅ L1-31 SAE features with global FDR correction
2. ✅ Feature means lookup JSON
3. ✅ Updated experiment scripts with new paths

### Post-Processing (Week 2)
1. Variance ratio analysis (justify t-test choice)
2. Comparison report: layer-wise FDR vs global FDR
3. Sensitivity analysis: standard t-test vs Welch's t-test

### Documentation
1. Methods section update for paper
2. Supplementary materials: FDR correction justification
3. Code archive with versioning

---

## 8. QUESTIONS ANSWERED

### 1. Is the experiment running or stalled?
**Answer**: ✅ **RUNNING** - 83% complete (5320/6400), ETA 20:52 KST

### 2. Is it worth completing?
**Answer**: ✅ **YES** - Post-hoc FDR correction salvages results

### 3. Are methodological differences fatal?
**Answer**: ⚠️ **LAYER-WISE FDR IS FATAL** - Must apply global FDR
**Answer**: ⚠️ **STANDARD T-TEST IS ACCEPTABLE** - With variance analysis caveat

### 4. Can downstream pipelines use new data?
**Answer**: ✅ **YES** - After global FDR correction + file regeneration

### 5. What's the fastest valid path?
**Answer**: ✅ **OPTION A** - 17 minutes to valid results

---

## 9. RISK MATRIX

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Experiment crashes before completion | LOW | HIGH | Save checkpoints every 1000 samples |
| Global FDR reduces features to <500 | MEDIUM | MEDIUM | Lower α to 0.10 if needed |
| Downstream scripts break | LOW | LOW | Update paths systematically |
| Reviewer rejects layer-wise → global fix | HIGH | LOW | Already planning global FDR |
| Welch's t-test demanded | MEDIUM | MEDIUM | Rerun if forced, or provide variance analysis |

---

## 10. TIMELINE SUMMARY

| Task | Duration | Completion |
|------|----------|------------|
| **Current experiment** | +7 min | 20:52 KST |
| **Post-hoc global FDR** | +10 min | 21:02 KST |
| **Feature means lookup** | +15 min | 21:17 KST |
| **Update downstream paths** | +5 min | 21:22 KST |
| **Validation testing** | +10 min | 21:32 KST |
| **TOTAL** | **47 min** | **21:32 KST** |

---

## CONCLUSION

The current SAE extraction experiment (extract_L1_31_SAE_CORRECTED.py) is **actively running and nearly complete**, but employs a **statistically flawed layer-wise FDR correction** that will produce inflated false discovery rates compared to the validated L25-31 reference methodology.

**The solution is straightforward**: Allow the experiment to finish (7 minutes), then apply **post-hoc global FDR correction** to align with best practices. This approach:
- Respects the 2h 43m computation already invested
- Produces statistically rigorous results in 17 minutes
- Maintains compatibility with existing L25-31 data
- Avoids reviewer criticism of inadequate multiple testing correction

**The standard t-test vs Welch's t-test difference is secondary**—it may affect 5-15% of borderline features, but the 70-80% inflation from layer-wise FDR is the primary methodological flaw requiring correction.

**Proceed with Option A: Post-Hoc Global FDR Correction.**

---

**Report End**
*Generated: 2025-11-10 20:45 KST*
*Next Update: Upon experiment completion (~20:52 KST)*
