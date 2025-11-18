# Implementation Plan Analysis: Exp1 & Exp3

**Date**: 2025-10-22
**Objective**: Compare two approaches for feature pathway discovery (Exp1) and feature-word analysis (Exp3)

---

## Executive Summary

After ultra-think analysis, I recommend:
- **Exp1**: **Approach 2** (Patch-based) - More practical, reuses Exp2 infrastructure
- **Exp3**: **Hybrid** - Simple patch comparison for words, no complex activation extraction needed

---

## Approach 1: Intervention Chaining (Novel Method)

### Concept
Layer-by-layer causal tracking without gradients:
```
L9 feature patch → Measure L10 features → Find strongest changes
L10 feature patch → Measure L11 features → Find strongest changes
...
Build chains: L9-456 → L10-789 → L11-123 → ... → L26-1069
```

### Pros
✅ **No gradient problem**: Only forward passes + SAE encode
✅ **True local causality**: Each transition is intervention-based
✅ **Novel methodology**: Publishable, orthogonal to Anthropic
✅ **SAE no_grad is fine**: We're not computing gradients

### Cons
❌ **Computational cost**: 6-8 hours minimum (even with optimization)
❌ **Complex implementation**: New code from scratch
❌ **Chaining complexity**: Need to track 29 layer transitions
❌ **Validation difficulty**: How to verify chains are correct?

### Critical Issues Discovered

#### Issue 1: Search Space Explosion
```
Even restricting to 2,787 causal features:
- ~93 features per layer
- Layer transition pairs: 93 × 93 = 8,630
- 29 transitions: 250,070 pairs
- × 2 passes (baseline + patched) = 500,140 forward passes
- @ 0.5 sec = 69.5 hours
```

**Mitigation**: Correlation pre-screening (~2 hours) → Test only high-correlation pairs
**New estimate**: ~8-10 hours

#### Issue 2: Chain Validation
How do we know L9→L10→L11...→L26 is THE pathway vs one of many?
- Multiple parallel pathways exist
- Need threshold for "strong enough" connection
- Risk of false positives in long chains

#### Issue 3: Implementation Complexity
Requires:
- Multi-layer feature extraction infrastructure
- Patching at intermediate layers
- Chain tracking algorithm
- Statistical validation at each step

---

## Approach 2: Patch-Based with Activation Caching (Standard)

### Concept
Leverage Exp2's existing infrastructure, but save activations:

```python
For each feature:
    1. Run Exp2 patching (6 conditions)
    2. During forward pass, SAVE intermediate activations
    3. Post-hoc analysis:
       - Exp1: Compare activations across conditions
       - Exp3: Compare words across conditions
```

### Pros
✅ **Reuses Exp2 code**: `/home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/`
✅ **Industry standard**: OpenAI/Anthropic use this approach
✅ **Simpler implementation**: Modify existing pipeline
✅ **Dual purpose**: One run serves both Exp1 & Exp3
✅ **Already validated**: Exp2 proved this works

### Cons
❌ **Storage**: Need to save activations (but manageable)
❌ **Re-run Exp2**: Can't fully reuse existing response logs
❌ **Less novel**: Standard patching methodology

### Implementation Details

#### Modify Exp2 Pipeline

**Current Exp2 structure** (`experiment_2_L1_31_top300.py`):
```python
def test_single_feature_patching(feature):
    # 1. Extract safe/risky baseline activations
    safe_act = extract_activation(safe_prompt, feature)
    risky_act = extract_activation(risky_prompt, feature)

    # 2. Patch and generate
    for condition in 6_conditions:
        response = generate_with_patching(...)
        # ❌ Only saves response text
```

**Modified for activation saving**:
```python
def test_single_feature_patching_with_cache(feature):
    # 1. Same as before
    safe_act = extract_activation(safe_prompt, feature)
    risky_act = extract_activation(risky_prompt, feature)

    # 2. Patch and generate WITH ACTIVATION CACHING
    for condition in 6_conditions:
        response, all_layer_activations = generate_with_activation_caching(...)

        save_result({
            'feature': feature,
            'condition': condition,
            'response': response,
            'activations': {  # ✅ NEW!
                'L1': all_layer_activations[1],  # (32768,)
                'L2': all_layer_activations[2],
                ...
                'L30': all_layer_activations[30]
            }
        })
```

#### Storage Calculation
```
Per trial:
- 30 layers × 32,768 features × 4 bytes (float32) = 3.9 MB
- 2,787 features × 6 conditions × 30 trials = 502,260 trials
- Total: 502,260 × 3.9 MB = 1.96 TB

❌ TOO LARGE!

Solution: Store only causal features (2,787 instead of 32,768)
- 30 layers × 2,787 features × 4 bytes = 335 KB per trial
- 502,260 trials × 335 KB = 168 GB
✅ MANAGEABLE!
```

#### New Activation Extraction Function
```python
def generate_with_activation_caching(
    prompt,
    patch_layer,
    patch_feature_id,
    patch_value,
    causal_features_list  # Only extract these!
):
    """
    Generate response while caching SAE activations

    Args:
        causal_features_list: [(layer, feature_id), ...]
            e.g., [(9, 456), (26, 1069), ...]

    Returns:
        response: str
        activations: dict {layer: {feature_id: value}}
    """
    activations = {}

    with torch.no_grad():
        # Forward pass with hooks
        outputs = model(
            input_ids,
            output_hidden_states=True
        )

        # Extract activations for causal features only
        for layer, feature_id in causal_features_list:
            hidden = outputs.hidden_states[layer]

            # Patch if this is the target layer
            if layer == patch_layer:
                sae_features = sae_encode(hidden)
                sae_features[0, patch_feature_id] = patch_value
                hidden = sae_decode(sae_features)

            # Extract causal features
            sae_features = sae_encode(hidden)
            if layer not in activations:
                activations[layer] = {}
            activations[layer][feature_id] = sae_features[0, feature_id].item()

    return response, activations
```

### Exp1 Analysis: Compare Activations

```python
def analyze_feature_pathways(feature_results):
    """
    Given: 6 conditions with full activations
    Find: Which upstream features contribute to this feature

    Logic:
    - safe_baseline vs safe_with_risky_patch
    - If upstream feature X also changes → potential pathway!
    """

    baseline_acts = feature_results['safe_baseline']['activations']
    patched_acts = feature_results['safe_with_risky_patch']['activations']

    pathways = []
    for layer in range(1, target_layer):
        for upstream_feature_id in causal_features[layer]:
            baseline_val = baseline_acts[layer][upstream_feature_id]
            patched_val = patched_acts[layer][upstream_feature_id]

            effect = patched_val - baseline_val
            if abs(effect) > threshold:
                pathways.append({
                    'upstream': f'L{layer}-{upstream_feature_id}',
                    'downstream': target_feature,
                    'effect': effect
                })

    return pathways
```

### Exp3 Analysis: Compare Words (NO RE-RUN NEEDED!)

**Key insight**: We already have response texts in Exp2 logs!

```python
def analyze_feature_words(exp2_response_logs):
    """
    Use EXISTING Exp2 data, no re-run needed!
    """

    for feature in causal_features:
        # Load from existing logs
        baseline_responses = load_responses(feature, 'safe_baseline')
        patched_responses = load_responses(feature, 'safe_with_risky_patch')

        # Extract words (FIXED regex!)
        baseline_words = extract_words_with_numbers(baseline_responses)
        patched_words = extract_words_with_numbers(patched_responses)

        # Statistical comparison
        added_words = words_significantly_increased(baseline, patched)
        removed_words = words_significantly_decreased(baseline, patched)
```

**Critical fix for word extraction**:
```python
def extract_words_with_numbers(text):
    # OLD (wrong): r'\b[a-zA-Z]+\b'
    # NEW (correct): Include numbers and dollar signs!

    words = re.findall(r'\$?\d+|\b[a-zA-Z]+\b', text.lower())
    # Captures: "bet", "amount", "$10", "5", "100"

    return [w for w in words if w not in stopwords and len(w) > 1]
```

---

## Comparison Matrix

| Aspect | Intervention Chaining | Patch-based |
|--------|----------------------|-------------|
| **Exp1 Time** | 8-10 hours | 12-15 hours (re-run Exp2) |
| **Exp3 Time** | N/A (doesn't help Exp3) | 0 hours (use existing data!) |
| **Code Complexity** | High (new from scratch) | Low (modify Exp2) |
| **Storage** | Minimal | 168 GB (manageable) |
| **Causality** | Local layer-by-layer | Condition comparison |
| **Novelty** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Risk** | Medium (validation) | Low (proven) |
| **Reusability** | Low | High (serves both Exp1 & Exp3) |

---

## RECOMMENDATION

### For Exp1: **Approach 2 (Patch-based)** ✅

**Reasons**:
1. **Serves dual purpose**: One run → both Exp1 & Exp3 data
2. **Proven methodology**: Industry standard (OpenAI/Anthropic)
3. **Lower risk**: Modifies existing working code
4. **Efficient**: 12-15 hours total vs 8-10 hours (Approach 1) + separate Exp3

**Implementation**: Modify Exp2 to save causal feature activations

### For Exp3: **No re-run needed!** ✅

**Reasons**:
1. **Exp2 response logs already have text**
2. **Only need to fix word extraction regex**
3. **Can run TODAY in 30 minutes**

**Action**: Fix `extract_words()` and run analysis now

---

## Implementation Plan

### Phase 1: Exp3 (Immediate - 1 hour)

**Files to modify**:
- `/home/ubuntu/llm_addiction/experiment_3_feature_word_patching/src/causal_word_patching_analyzer.py`

**Changes**:
```python
# Line 95: Fix regex
def extract_words(self, text):
    # Include numbers and dollar signs
    words = re.findall(r'\$?\d+|\b[a-zA-Z]+\b', text.lower())

    # Enhanced stopwords
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'choose', 'choice', 'current', 'round',
        # Keep gambling-specific words!
        # 'bet', 'win', 'loss', 'balance', 'amount' → DON'T REMOVE!
    }

    return [w for w in words if w not in stopwords and len(w) > 1]
```

**Run**:
```bash
cd /home/ubuntu/llm_addiction/experiment_3_feature_word_patching
./launch_causal_word_analysis.sh
```

**Expected**: Completes in ~30 minutes, produces feature-word associations

---

### Phase 2: Exp1 (GPU run - 12-15 hours)

**Create new file**: `/home/ubuntu/llm_addiction/experiment_1_patching_with_activations/`

**Base code**: Copy from Exp2, modify for activation caching

**Key components**:

1. **Load causal features list**:
```python
safe_df = pd.read_csv('/home/ubuntu/llm_addiction/analysis/CORRECT_consistent_safe_features.csv')
risky_df = pd.read_csv('/home/ubuntu/llm_addiction/analysis/CORRECT_consistent_risky_features.csv')

causal_features = []
for _, row in safe_df.iterrows():
    layer, feature_id = parse_feature(row['feature'])  # 'L9-456' → (9, 456)
    causal_features.append((layer, feature_id))
```

2. **Modified generation function**:
```python
def generate_with_activation_caching(self, ...):
    # Extract only causal features (2,787 not 32,768!)
    # Save to disk incrementally (don't hold all in memory)
```

3. **Incremental saving**:
```python
# Save every 50 features to avoid memory issues
output_file = f'activations_cache_{timestamp}_batch_{batch_id}.npz'
np.savez_compressed(output_file, **activation_dict)
```

**Launch**:
```bash
cd /home/ubuntu/llm_addiction/experiment_1_patching_with_activations
./launch.sh  # Uses GPU 4, 12-15 hour run
```

**Expected output**:
```
/data/llm_addiction/experiment_1_activation_cache/
├── activations_batch_0.npz    (50 features)
├── activations_batch_1.npz    (50 features)
...
├── activations_batch_55.npz   (37 features)
Total: 168 GB
```

---

### Phase 3: Analysis (Post-processing)

**Pathway analysis**:
```python
def find_pathways(activation_cache):
    """
    For each target feature:
      Compare safe_baseline vs safe_with_risky_patch
      Find upstream features that also changed
    """

    for target_feature in causal_features:
        target_data = activation_cache[target_feature]

        baseline = target_data['safe_baseline']['activations']
        patched = target_data['safe_with_risky_patch']['activations']

        # Find changed upstream features
        for upstream_feature in earlier_features:
            upstream_baseline = baseline[upstream_feature['layer']][upstream_feature['id']]
            upstream_patched = patched[upstream_feature['layer']][upstream_feature['id']]

            effect = upstream_patched - upstream_baseline
            if abs(effect) > 0.1:  # Threshold
                pathways.append({
                    'source': upstream_feature,
                    'target': target_feature,
                    'effect': effect
                })
```

---

## Discussion Points

### Question 1: Storage - 168 GB acceptable?
- **Location**: `/data/llm_addiction/experiment_1_activation_cache/`
- **Format**: `.npz` compressed (can reduce to ~100 GB)
- **Alternative**: Save only top-K features per layer (further reduce)

### Question 2: Should we pre-screen with correlation?
- **Pro**: Reduces compute by 50%
- **Con**: Might miss non-linear pathways
- **Recommendation**: Run full version first, optimize if too slow

### Question 3: Exp3 - Run now or wait for Exp1?
- **Run now!**: Independent, takes 30 min, high value
- **Wait**: No benefit, just delays results

### Question 4: Number of trials per condition?
- **Exp2 used**: 30 trials per condition
- **For activation caching**: Same 30 trials sufficient
- **Storage**: Already accounted for in 168 GB estimate

---

## Timeline

| Phase | Duration | Dependencies | Output |
|-------|----------|--------------|--------|
| **Exp3 fix & run** | 1 hour | None | Feature-word associations |
| **Exp1 code** | 4 hours | Exp2 codebase | Modified patching pipeline |
| **Exp1 execution** | 12-15 hours | GPU availability | Activation cache (168 GB) |
| **Exp1 analysis** | 2 hours | Activation cache | Pathway results |
| **Total** | ~20 hours | | Both experiments complete |

---

## Risk Assessment

### Low Risk ✅
- Exp3 implementation (simple regex fix)
- Storage capacity (168 GB manageable)
- Exp2 code reuse (proven working)

### Medium Risk ⚠️
- Exp1 execution time (12-15 hours estimate)
- Memory management (need incremental saving)
- Pathway validation (statistical thresholds)

### Mitigations
- Test on 10 features first (1 hour test run)
- Monitor memory usage continuously
- Implement checkpointing (resume if crashes)

---

## Conclusion

**Approach 2 (Patch-based) is recommended** because:
1. Reuses proven Exp2 infrastructure
2. Serves both Exp1 and Exp3 (Exp3 already done via existing logs!)
3. Lower implementation risk
4. Industry-standard methodology

**Immediate action**: Fix Exp3 and run (1 hour)
**Next step**: Modify Exp2 for activation caching (4 hours coding + 12-15 hours execution)

**Total time to completion**: ~20 hours
**Total value**: Feature pathways + Feature-word associations
