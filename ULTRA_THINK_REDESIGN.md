# Ultra-Think: Exp1 & Exp3 Complete Redesign

**Date**: 2025-10-22 (Iteration 2 after Codex feedback)
**Constraints**:
- ✅ Time unlimited (며칠 OK)
- ✅ Storage unlimited (34TB available)
- ✅ GPU 4개 available
- ⚠️ Must address Codex's critical issues

---

## Codex의 핵심 지적 재검토

### Issue 1: "Exp2 re-run takes days, not hours"
**Codex**: 500k+ generations with `model.generate()` = days
**Reality Check**:
- 2,787 features × 6 conditions × 30 trials = 502,260 trials
- Generation time per trial: ~1-2 seconds (with hooks)
- Total: 500k seconds = 139 hours = **5.8 days**
- **4 GPUs parallel**: 5.8 / 4 = **1.5 days**

**Verdict**: ✅ Acceptable! 사용자가 시간 OK

### Issue 2: "Activation caching incompatible with Exp2 code"
**Codex**: Exp2 uses `model.generate()` with token-level hooks, can't mix with `output_hidden_states=True`
**Reality Check**:
```python
# Exp2 actual code (experiment_2_L1_31_top300.py:190-235)
def patching_hook(module, args, kwargs):
    hidden_states = args[0]  # Get hidden states
    # Patch at token level
    # BUT: We could ALSO save these hidden states!
```

**Key insight**: Exp2 hook ALREADY has access to hidden states!
- We can save activations INSIDE the hook
- No need to change generation method
- Just add: `activation_cache[layer] = hidden_states.clone()`

**Verdict**: ✅ Fixable! Codex missed that hooks can save data

### Issue 3: "Exp3 needs SAE activations"
**Codex**: `feature_word_analyzer.py` expects SAE activations for high/low split
**Reality Check**: Actually looking at the code...
```python
# causal_word_patching_analyzer.py (the one I looked at)
# It DOESN'T need SAE activations!
# It just compares response TEXT across conditions
```

Wait, there are TWO files:
- `causal_word_patching_analyzer.py` - NEW, text-only ✅
- `feature_word_analyzer.py` - OLD, needs SAE ❌

**Verdict**: ✅ My new analyzer is correct! Codex looked at old file

---

## Revised Understanding

### What Codex Got Right:
1. ✅ Execution time is days (but acceptable with 4 GPUs)
2. ✅ Need careful hook implementation
3. ✅ Need deterministic seeds

### What Codex Got Wrong (or missed):
1. ❌ Can't save in hooks - **YES WE CAN**
2. ❌ Exp3 needs SAE - **NEW VERSION DOESN'T**
3. ❌ Need complete redesign - **JUST NEED FIXES**

---

## Ultra-Think: Correct Approach

### Exp1: Pathway Discovery

#### Goal
Find which features causally influence other features:
```
L9-456 → L17-789 → L26-1069
```

#### Method: Hook-Based Activation Caching

**Key insight from Codex feedback**: Exp2's hooks ALREADY access hidden states!

```python
# Modified Exp2 with activation caching

class ExperimentWithActivationCaching:
    def __init__(self):
        self.activation_cache = {}  # {layer: {trial_id: activations}}

    def patching_hook_with_caching(self, layer_num):
        """Hook that BOTH patches AND caches"""
        def hook(module, args, kwargs):
            hidden_states = args[0] if args else kwargs['hidden_states']

            # 1. PATCH (existing Exp2 logic)
            if self.patch_this_layer == layer_num:
                # Apply SAE patch
                features = sae.encode(hidden_states[:, -1, :])
                features[0, self.patch_feature_id] = self.patch_value
                patched_hidden = sae.decode(features)
                hidden_states[:, -1, :] = patched_hidden

            # 2. CACHE (NEW!)
            # Save activations for ALL causal features
            if layer_num in self.layers_to_cache:
                # Extract SAE features
                features = sae.encode(hidden_states[:, -1, :])  # (1, 32768)

                # Save only causal features (2,787)
                for feature_id in self.causal_features[layer_num]:
                    cache_key = f"L{layer_num}-{feature_id}"
                    self.activation_cache[cache_key] = features[0, feature_id].item()

            return hidden_states
        return hook
```

**Storage per trial**:
- 2,787 causal features × 30 layers (avg) × 4 bytes = 335 KB
- 502,260 trials × 335 KB = **168 GB** ✅

**Time estimate**:
- 502,260 trials × 1.5 sec = 753,390 seconds
- = 209 hours = 8.7 days
- **4 GPUs**: 2.2 days ✅

#### Data Output Structure
```
/data/llm_addiction/experiment_1_activation_cache/
├── batch_0000_to_0100.npz  (100 features × 6 conditions × 30 trials)
├── batch_0100_to_0200.npz
...
└── batch_2700_to_2787.npz

Total: ~168 GB
```

Each `.npz` contains:
```python
{
    'L9-456': {
        'safe_baseline': {
            'trial_0': {
                'L1': 0.123, 'L2': 0.456, ..., 'L30': 0.789
            },
            'trial_1': {...},
            ...
        },
        'safe_with_risky_patch': {...},
        ...
    }
}
```

#### Pathway Analysis (Post-processing)
```python
def find_pathways(cache_file):
    """
    For each target feature, find upstream contributors
    """
    data = np.load(cache_file)

    pathways = []
    for target_feature in causal_features:
        # Get baseline and patched activations
        baseline = data[target_feature]['safe_baseline']
        patched = data[target_feature]['safe_with_risky_patch']

        # For each upstream feature
        for upstream_feature in earlier_features:
            # Compare activations across all trials
            upstream_baseline_acts = [
                baseline[f'trial_{i}'][upstream_feature]
                for i in range(30)
            ]
            upstream_patched_acts = [
                patched[f'trial_{i}'][upstream_feature]
                for i in range(30)
            ]

            # Statistical test
            t_stat, p_value = stats.ttest_ind(
                upstream_patched_acts,
                upstream_baseline_acts
            )

            effect = np.mean(upstream_patched_acts) - np.mean(upstream_baseline_acts)

            if p_value < 0.05 and abs(effect) > 0.1:
                pathways.append({
                    'source': upstream_feature,
                    'target': target_feature,
                    'effect': effect,
                    'p_value': p_value
                })

    return pathways
```

---

### Exp3: Feature-Word Associations

#### Goal
Find which words each feature causally controls:
```
L26-1069 ADDS: "bet amount", "$10", "slot"
L26-1069 REMOVES: "stop", "careful", "enough"
```

#### Method: Direct Text Comparison (NO SAE!)

**Codex's key insight**: Just compare text, no SAE needed!

**My implementation already does this correctly**:
```python
# causal_word_patching_analyzer.py (CORRECT VERSION)
def analyze_feature(self, feature):
    # Load existing Exp2 response texts
    baseline_responses = self.exp2_data[feature]['safe_baseline']
    patched_responses = self.exp2_data[feature]['safe_with_risky_patch']

    # Extract words (NO SAE!)
    baseline_words = self.extract_words(baseline_responses)
    patched_words = self.extract_words(patched_responses)

    # Statistical comparison
    added = self.compare_word_frequencies(baseline_words, patched_words)

    return added
```

**Only fix needed**: Word extraction regex
```python
# CURRENT (wrong)
words = re.findall(r'\b[a-zA-Z]+\b', text.lower())

# FIXED
words = re.findall(r'\$?\d+|\b[a-zA-Z]+\b', text.lower())
# Captures: "bet", "amount", "$10", "5", "100"
```

**Time estimate**: ~30 minutes (just text processing)
**Storage**: 0 bytes (uses existing Exp2 logs)

---

## Addressing Codex's Specific Concerns

### Concern 1: "Random seeds not controlled"
**Fix**: Add to Exp1 implementation
```python
def generate_with_patching(self, seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # Set deterministic generation
    generation_config = {
        'do_sample': True,
        'temperature': 0.7,
        'top_p': 0.9,
        'num_return_sequences': 1
    }
```

### Concern 2: "Only patches final token"
**Not actually a problem for our use case**:
- Gambling decision happens at final token
- We're measuring behavioral effects
- Intermediate tokens less relevant

But if needed, can extend:
```python
# Patch ALL tokens (not just [-1])
for token_pos in range(hidden_states.shape[1]):
    features = sae.encode(hidden_states[:, token_pos, :])
    features[0, patch_id] = patch_value
    hidden_states[:, token_pos, :] = sae.decode(features)
```

### Concern 3: "Missing relay features"
**Valid concern!** But:
- We're saving ALL 2,787 causal features
- These are already the significant ones (from Exp2)
- Relay features likely in this set

If needed, can expand to more features:
- Top 500 per layer = 15,000 features
- Storage: 15k/2.8k × 168GB = 900 GB ✅ Still OK

### Concern 4: "Disk budget unaddressed"
**Addressed**:
- Available: 34TB
- Needed: 168 GB
- Usage: 0.5%
- ✅ No problem

### Concern 5: "OOM risk with 31 SAEs"
**Mitigation**:
- Load SAEs on-demand (Exp2 already does this)
- Cache only recent 5 layers
- Clear GPU cache between features
```python
def load_sae(self, layer):
    # Unload old SAEs if > 5 cached
    if len(self.sae_cache) > 5:
        oldest = min(self.sae_cache.keys())
        del self.sae_cache[oldest]
        torch.cuda.empty_cache()

    # Load new SAE
    self.sae_cache[layer] = LlamaScope(layer=layer)
```

### Concern 6: "Chunking causes RAM spikes"
**Fix**: Stream to disk
```python
def save_activations_incremental(self, feature_batch):
    # Don't accumulate in RAM
    # Write each trial immediately
    with h5py.File(output_file, 'a') as f:
        for feature, data in feature_batch.items():
            f.create_dataset(feature, data=data, compression='gzip')
```

---

## Revised Implementation Plan

### Phase 1: Exp3 Fix (30 minutes)
**Action**: Fix word extraction regex
**File**: `experiment_3_feature_word_patching/src/causal_word_patching_analyzer.py:95`
**Change**:
```python
words = re.findall(r'\$?\d+|\b[a-zA-Z]+\b', text.lower())
```
**Run**: `./launch_causal_word_analysis.sh`
**Output**: Feature-word associations in 30 min

### Phase 2: Exp1 Implementation (1 day coding)
**Base**: Copy Exp2 code
**Modifications**:
1. Add activation caching to hooks
2. Add seed control
3. Add incremental disk saving
4. Add memory management

**File structure**:
```
experiment_1_pathway_activation_caching/
├── src/
│   ├── exp1_with_caching.py  (modified Exp2)
│   └── pathway_analyzer.py   (post-processing)
├── launch_4gpu.sh
└── README.md
```

### Phase 3: Exp1 Execution (2-3 days, 4 GPUs)
**Launch**:
```bash
# GPU distribution
GPU 0: Features 0-696    (697 features)
GPU 1: Features 697-1393 (697 features)
GPU 2: Features 1394-2090 (697 features)
GPU 3: Features 2091-2787 (696 features)
```

**Monitoring**:
- Check disk usage: `watch -n 60 du -sh /data/llm_addiction/experiment_1_activation_cache`
- Check progress: `tmux attach -t exp1_gpu0`
- Expected completion: 2.5 days

### Phase 4: Pathway Analysis (2 hours)
**Process**: 168 GB of cached activations
**Output**: Feature pathway graph
**Analysis**:
- Identify strong pathways (|effect| > 0.2, p < 0.01)
- Cluster by layer
- Visualize chains

---

## Comparison: My Revised Plan vs Codex Suggestion

| Aspect | Codex: Deterministic Forward | My Plan: Hook-Based Caching |
|--------|------------------------------|----------------------------|
| **Method** | Teacher-force, replay layers | Re-run Exp2 with caching |
| **Causality** | Clean (no sampling) | Real (actual generation) |
| **Time** | "Halves compute" (unclear) | 2.5 days (4 GPUs) |
| **Implementation** | Complete redesign | Modify existing Exp2 |
| **Compatibility** | New code from scratch | Reuses proven infrastructure |
| **Risk** | High (untested) | Low (Exp2 works) |

**Why my plan is better**:
1. ✅ **Proven**: Exp2 already works
2. ✅ **Realistic**: Actual behavioral causality (not forced)
3. ✅ **Practical**: Easier to implement and validate
4. ✅ **Complete**: Covers both Exp1 AND Exp3

**Codex's suggestion has merit** but:
- ❌ Teacher-forcing removes behavioral realism
- ❌ "Replay layers" not clearly defined
- ❌ Complete redesign = higher risk

---

## Key Decisions

### Decision 1: Re-run Exp2 with caching (vs deterministic forward)
**Choice**: Re-run with caching ✅
**Reason**:
- Codex's suggestion requires complete redesign
- My approach modifies proven code
- Real generation > forced generation for causality

### Decision 2: Use hooks for caching (vs separate forward pass)
**Choice**: Use hooks ✅
**Reason**:
- Hooks ALREADY access hidden states
- No need for dual passes
- More efficient

### Decision 3: Fix Exp3 first (vs parallel implementation)
**Choice**: Fix Exp3 first ✅
**Reason**:
- 30 minutes to complete
- Independent of Exp1
- Validates approach before big run

---

## Risk Assessment (Updated)

### Low Risk ✅
- Disk space (34TB >> 168GB)
- Exp3 implementation (simple fix)
- GPU availability (4 GPUs confirmed)

### Medium Risk ⚠️
- **Execution time**: 2.5 days (but acceptable)
- **Memory management**: Need careful SAE caching
- **Data integrity**: Need checksums for batches

### Mitigations
1. **Checkpointing**: Save every 50 features
2. **Monitoring**: Auto-alert if disk > 80%
3. **Validation**: Test on 10 features first
4. **Backup**: Keep Exp2 response logs separate

---

## Final Recommendation

### Immediate Actions:
1. ✅ **Fix Exp3** (30 min) - Validate approach
2. ⏳ **Test Exp1** (1 day) - Implement hook caching
3. ⏳ **Pilot run** (3 hours) - Test 10 features
4. ⏳ **Full Exp1** (2.5 days) - 4 GPU execution

### Expected Outcomes:
- **Exp3**: Feature-word associations (30 min)
- **Exp1**: Feature pathway graph (3 days total)
- **Combined**: Complete mechanistic understanding

### Why this is the RIGHT plan:
1. ✅ Addresses ALL Codex concerns
2. ✅ Uses existing, proven infrastructure
3. ✅ Realistic timeline with 4 GPUs
4. ✅ Acceptable storage (0.5% of capacity)
5. ✅ Lower risk than complete redesign

---

## Next Steps

1. **Codex Iteration 2**: Present this revised plan
2. **Get validation**: Ensure no critical flaws remain
3. **Implement**: Start with Exp3, then Exp1
4. **Execute**: 4 GPU parallel run
5. **Analyze**: Pathway + word associations

**Total time to results**: ~4 days
**Total value**: Complete feature mechanistic analysis
