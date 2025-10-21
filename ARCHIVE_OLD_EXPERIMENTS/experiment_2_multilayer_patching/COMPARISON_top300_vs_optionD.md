# Multilayer Patching: Top 300/Layer vs Option D Comparison

## Executive Summary

**Top 300/Layer (Recommended)**: 9,300 features, 8.1 days with 4 GPUs (20 trials)
**Option D (|d| > 1.0)**: 17,699 features, 15.4 days with 4 GPUs (20 trials)

Top 300/Layer provides **47.5% reduction in features** while maintaining **balanced layer representation** and focusing on the **highest-impact features per layer**.

---

## Detailed Comparison

### 1. Feature Count

| Metric | Top 300/Layer | Option D (|d| > 1.0) |
|--------|---------------|----------------------|
| Total Features | 9,300 | 17,699 |
| Selection Method | Top 300 by \|d\| per layer | All features with \|d\| > 1.0 |
| Reduction vs Option D | 47.5% fewer | Baseline |

### 2. Time Estimates (4 GPUs Parallel)

| Trials | Top 300/Layer | Option D |
|--------|---------------|----------|
| 10 trials | 4.0 days | 7.7 days |
| **20 trials** | **8.1 days** | **15.4 days** |
| 50 trials | 20.2 days | 38.5 days |

**Recommendation**: 20 trials provides good statistical power while completing in ~1 week.

### 3. Layer Distribution

#### Top 300/Layer (Balanced)
- **L1-L10**: 3,000 features (32.3%)
- **L11-L20**: 3,000 features (32.3%)
- **L21-L31**: 3,300 features (35.5%)

#### Option D (Late-layer Heavy)
- **L1-L10**: 1,295 features (7.3%)
- **L11-L20**: 4,115 features (23.2%)
- **L21-L31**: 12,289 features (69.4%)

**Key Insight**: Top 300/Layer ensures comprehensive coverage across all computational stages (early, mid, late), while Option D heavily biases toward late layers.

### 4. Effect Size Thresholds by Layer

| Layer Range | Top 300/Layer Min \|d\| | Option D Min \|d\| |
|-------------|------------------------|-------------------|
| L1-L10 | 0.659 - 0.793 | 1.0 (uniform) |
| L11-L20 | 0.859 - 1.182 | 1.0 (uniform) |
| L21-L31 | 1.169 - 1.197 | 1.0 (uniform) |

**Key Insight**: Top 300/Layer adapts thresholds to each layer's feature distribution:
- Early layers: Lower threshold (0.659-0.793) to capture important early processing
- Mid layers: Moderate threshold (0.859-1.182)
- Late layers: Higher threshold (1.169-1.197) due to dense high-effect features

### 5. Cohen's d Range Coverage

#### Top 300/Layer
- **Min across all layers**: 0.659 (L3)
- **Max across all layers**: 1.625 (L25)
- **Average min cutoff**: ~0.95
- **Coverage**: Medium-to-very-large effects

#### Option D
- **Min**: 1.0 (uniform cutoff)
- **Max**: 1.625 (L25)
- **Coverage**: Large-to-very-large effects only

---

## Pros and Cons

### Top 300/Layer

**Pros:**
✅ Balanced representation across all 31 layers
✅ 47.5% faster than Option D (8 days vs 15 days)
✅ Adaptive thresholds per layer's feature distribution
✅ Captures important early-layer features (L1-L10)
✅ Focuses on top effects per computational stage
✅ More manageable dataset for analysis

**Cons:**
⚠️ May miss some late-layer features with 1.0 < |d| < 1.17
⚠️ Arbitrary cutoff at 300 (but justified by computational constraints)

### Option D (|d| > 1.0)

**Pros:**
✅ Uniform effect size threshold (interpretable)
✅ Captures all "very large" effects across layers
✅ More comprehensive late-layer coverage

**Cons:**
❌ 90% longer runtime (15 days vs 8 days)
❌ Heavy bias toward late layers (69% in L21-L31)
❌ Under-represents early processing (7% in L1-L10)
❌ May include redundant late-layer features

---

## Recommendation: **Top 300/Layer with 20 Trials**

### Rationale:

1. **Scientific Coverage**: Ensures representative sampling across all computational stages (early attention, mid transformations, late decision-making)

2. **Efficiency**: Completes in ~8 days vs ~15 days (47% faster)

3. **Effect Size Quality**: All selected features have medium-to-very-large effects within their layer context

4. **Layer-Adaptive**: Respects each layer's unique feature distribution:
   - Early layers naturally have fewer ultra-strong effects
   - Late layers show more concentrated decision signals
   - Top 300/layer captures the best of each stage

5. **Practical Analysis**: 9,300 features is more manageable for:
   - Interpretation and annotation
   - Cross-layer pathway analysis
   - Publication figure generation

### Implementation Plan:

```python
# Feature selection
for layer in range(1, 32):
    layer_features = load_layer_features(layer)
    sorted_by_cohen_d = sort(layer_features, key=abs(cohen_d), reverse=True)
    selected = sorted_by_cohen_d[:300]  # Top 300

# Patching experiment
n_trials = 20  # Balance between power and speed
scales = [safe_mean, baseline, risky_mean]
prompts = [safe_prompt, risky_prompt]

# Total: 9,300 × 3 × 2 × 20 = 1,116,000 runs
# With 4 GPUs: 8.1 days
```

---

## Alternative: Hybrid Approach (If Resources Permit)

**Hybrid Strategy**: Top 300/layer + All |d| > 1.5

This would add ~1,200 ultra-strong features from late layers that might fall outside top 300, while maintaining balanced early/mid coverage.

- **Total features**: ~10,000
- **Time estimate**: 9.2 days with 4 GPUs (20 trials)
- **Benefit**: Captures both breadth (top 300/layer) and depth (all ultra-strong)

---

## Decision Matrix

| Priority | Best Choice |
|----------|-------------|
| **Fastest completion** | Top 300/Layer, 10 trials (4 days) |
| **Balanced coverage + reasonable time** | **Top 300/Layer, 20 trials (8 days)** ⭐ |
| **Statistical power** | Top 300/Layer, 50 trials (20 days) |
| **Maximum comprehensiveness** | Option D, 20 trials (15 days) |

**⭐ Recommended**: Top 300/Layer with 20 trials (8.1 days)
