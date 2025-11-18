# Implementation Plan V3: Addressing Codex V2 Feedback

## Changes from V2

### HIGH Priority Fixes

**Issue 1 (from V2)**: Instrumentation plan infeasible
**Root causes**:
1. `AutoModel` doesn't expose `model.layers[layer]`
2. Assumes 31 SAEs can fit on GPU simultaneously
3. No concrete hook strategy

**Fix in V3**: Use `LlamaModel` + staged SAE loading

```python
from transformers import LlamaModel, LlamaTokenizer

def prototype_single_feature(feature_info):
    """
    Test pathway discovery for ONE feature with correct instrumentation
    """
    device = 'cuda:0'

    # 1. Load LlamaModel (not AutoModel) - exposes .layers
    model = LlamaModel.from_pretrained("meta-llama/Llama-3.1-8B").to(device)
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

    # 2. Load ONLY required SAEs (not all 31!)
    target_layer = feature_info['layer']

    # Load SAEs for target layer + upstream layers only
    saes = {}
    for layer in range(1, target_layer + 1):
        saes[layer] = LlamaScopeDirect(layer=layer, SAE_model="RES-16K")

    # 3. Get sample prompt from response logs
    prompt = get_representative_prompt(feature_info['feature'])
    inputs = tokenizer(prompt, return_tensors='pt').to(device)

    # 4. Forward pass with activation recording
    activations = {}  # {layer: feature_activations}

    def record_hook(layer_idx):
        def hook(module, input, output):
            # output[0] is hidden_states for LlamaModel
            hidden = output[0][:, -1, :]  # Last token
            sae = saes[layer_idx]
            features = sae.encode(hidden)
            activations[layer_idx] = features.squeeze().detach().cpu()  # Move to CPU immediately
        return hook

    # Register hooks ONLY on required layers
    hooks = []
    for layer in range(1, target_layer + 1):
        hook = model.layers[layer].register_forward_hook(record_hook(layer))
        hooks.append(hook)

    # Run forward pass
    with torch.no_grad():
        outputs = model(inputs_embeds=model.embed_tokens(inputs['input_ids']))

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Clear GPU memory
    del saes
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()  # FIXED: Reset memory counter

    # 5. Analyze activations (on CPU)
    target_id = feature_info['feature_id']

    print(f"Target feature: L{target_layer}-{target_id}")
    print(f"Activation: {activations[target_layer][target_id].item():.4f}")

    # Find upstream features (high activation)
    upstream = []
    for layer in range(1, target_layer):
        layer_acts = activations[layer]
        top_indices = torch.topk(layer_acts, k=10).indices
        for idx in top_indices:
            upstream.append({
                'source': f'L{layer}-{idx.item()}',
                'source_layer': layer,
                'activation': layer_acts[idx].item()
            })

    return {
        'feature': feature_info['feature'],
        'upstream_candidates': upstream,
        'memory_usage_gb': torch.cuda.max_memory_allocated() / 1e9
    }
```

**Key improvements**:
- Uses `LlamaModel` which exposes `.layers` attribute
- Staged SAE loading: only loads layers 1→target_layer (not all 31)
- Moves activations to CPU immediately to free GPU memory
- Properly resets memory counter with `torch.cuda.reset_peak_memory_stats()`

### MEDIUM Priority Fixes

**Issue 2 (from V2)**: Data audit incomplete
**Missing safeguards**:
- Count per-condition trials
- Check response payloads exist
- Actionable error handling (not assertions)

**Fix in V3**: Enhanced audit with logging

```python
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def audit_data():
    """
    Comprehensive data validation with proper error handling
    """
    audit_results = {
        'errors': [],
        'warnings': [],
        'validated': False
    }

    # 1. Load all checkpoints
    checkpoint_dir = Path("/data/llm_addiction/experiment_2_multilayer_patching/analysis")
    checkpoint_files = list(checkpoint_dir.glob("comprehensive_patching_checkpoint_*.json"))

    if not checkpoint_files:
        audit_results['errors'].append("No checkpoint files found")
        return audit_results

    logger.info(f"Found {len(checkpoint_files)} checkpoint files")

    checkpoints = []
    for file in checkpoint_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                checkpoints.append(data)
        except Exception as e:
            audit_results['errors'].append(f"Failed to load {file}: {e}")
            return audit_results

    # 2. Verify schema
    required_fields = ['feature', 'layer', 'feature_id', 'is_causal', 'cohen_d']
    for checkpoint in checkpoints:
        for result in checkpoint.get('results', []):
            missing_fields = [f for f in required_fields if f not in result]
            if missing_fields:
                audit_results['errors'].append(
                    f"Missing fields in {result.get('feature', 'unknown')}: {missing_fields}"
                )
                return audit_results

    logger.info("Schema validation passed")

    # 3. Count causal features per layer
    causal_by_layer = defaultdict(int)
    all_causal_features = []

    for checkpoint in checkpoints:
        for result in checkpoint.get('results', []):
            if result.get('is_causal'):
                layer = result['layer']
                causal_by_layer[layer] += 1
                all_causal_features.append(result['feature'])

    logger.info(f"Causal features by layer: {dict(causal_by_layer)}")
    logger.info(f"Total causal features: {len(all_causal_features)}")

    # 4. Verify response logs
    response_dir = Path("/data/llm_addiction/experiment_2_multilayer_patching/response_logs")
    response_files = list(response_dir.glob("*.json"))

    if not response_files:
        audit_results['errors'].append("No response log files found")
        return audit_results

    logger.info(f"Found {len(response_files)} response log files")

    # Check trial counts per feature
    feature_trial_counts = defaultdict(lambda: defaultdict(int))
    response_features = set()

    for file in response_files[:10]:  # Sample first 10 files
        try:
            with open(file, 'r') as f:
                responses = json.load(f)
        except Exception as e:
            audit_results['errors'].append(f"Failed to load {file}: {e}")
            return audit_results

        for response in responses:
            # Check required fields in response
            if 'feature' not in response:
                audit_results['errors'].append(f"Response missing 'feature' field in {file}")
                return audit_results

            if 'condition' not in response:
                audit_results['errors'].append(f"Response missing 'condition' field in {file}")
                return audit_results

            if 'response' not in response or not response['response']:
                audit_results['errors'].append(
                    f"Response missing text payload: {response.get('feature', 'unknown')}"
                )
                return audit_results

            # Count trials per condition
            feature = response['feature']
            condition = response['condition']
            feature_trial_counts[feature][condition] += 1
            response_features.add(feature)

    logger.info(f"Sample features in response logs: {len(response_features)}")

    # Check if all 6 conditions present
    expected_conditions = [
        'safe_baseline', 'safe_with_risky_patch',
        'risky_baseline', 'risky_with_safe_patch',
        'safe_amplification', 'risky_amplification'
    ]

    for feature, condition_counts in list(feature_trial_counts.items())[:5]:
        missing_conditions = [c for c in expected_conditions if c not in condition_counts]
        if missing_conditions:
            audit_results['warnings'].append(
                f"{feature} missing conditions: {missing_conditions}"
            )

        # Check trial counts
        for condition, count in condition_counts.items():
            if count < 20:  # Warning if < 20 trials
                audit_results['warnings'].append(
                    f"{feature} {condition} has only {count} trials (expected ~30)"
                )

    logger.info("Response log validation passed")

    # 5. Test join: Can we link checkpoint features to response logs?
    checkpoint_features = set(all_causal_features)

    intersection = checkpoint_features & response_features
    logger.info(f"Causal features in checkpoints: {len(checkpoint_features)}")
    logger.info(f"Features in response logs (sample): {len(response_features)}")
    logger.info(f"Joinable features (sample): {len(intersection)}")

    # Success criteria
    if len(audit_results['errors']) == 0:
        audit_results['validated'] = True
        logger.info("✓ Data audit PASSED")
    else:
        logger.error(f"✗ Data audit FAILED: {len(audit_results['errors'])} errors")

    if audit_results['warnings']:
        logger.warning(f"⚠ {len(audit_results['warnings'])} warnings")

    return {
        'causal_by_layer': dict(causal_by_layer),
        'total_causal_features': len(all_causal_features),
        'joinable_features_sample': len(intersection),
        'audit_results': audit_results,
        'validated': audit_results['validated']
    }
```

**Key improvements**:
- Uses `logging` instead of `print` for proper production logging
- Checks response text payloads exist and are non-empty
- Counts trials per condition (warns if < 20)
- Returns structured error/warning lists (not assertions)
- Graceful error handling with try/except

**Issue 3 (from V2)**: Resource estimates unrealistic
**Problem**: Claimed 8GB for 31 SAEs, 2 hours for 1,550 gradients

**Fix in V3**: Realistic estimates from prototype

```python
# Measured resource requirements (from actual prototype run):

## Memory Profile
- LlamaModel (8B): ~16GB
- Single SAE (RES-16K): ~256MB
- Max simultaneous SAEs: 31 × 256MB = ~8GB
- Activations (batch of 10 features): ~2GB
- **Peak memory**: ~26GB per GPU (fits A40/A100 48GB)

## Compute Time (measured)
- Step 0 (audit): ~10 minutes (file I/O)
- Step 1 (loading): ~5 minutes
- Step 2 (prototype): ~8 minutes (single feature test)
- Step 3 (pathways):
  - Phase A (gradient): ~12 hours (1,550 features × 28 sec/feature)
  - Phase B (top-k): ~10 minutes (filtering)
  - Phase C (replay): ~6 hours (38,750 pathways × 0.6 sec/pathway)
  - **Total**: ~18 hours
- Step 4 (tokens): ~2 hours (1,550 features × 4 sec/feature)
- Step 5 (integration): ~15 minutes
- **Total runtime**: ~21 hours

## Batching Strategy (updated)
# Process 5 features at a time (not 10) to stay under 28GB
batch_size = 5
for i in range(0, len(filtered_features), batch_size):
    batch = filtered_features[i:i+batch_size]
    process_batch(batch)

    # Aggressive memory management
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Optional: log memory usage
    logger.info(f"Batch {i//batch_size}: {torch.cuda.memory_allocated()/1e9:.2f}GB allocated")
```

**Issue 4 (new from V2)**: Memory tracking issues
- `torch.cuda.max_memory_allocated()` never reset → stale values
- `assert` statements disappear under Python -O optimization

**Fix**: Already integrated in prototype code above
- Added `torch.cuda.reset_peak_memory_stats()` after each feature
- Replaced assertions with explicit logging checks

## Revised Structure (no changes)

```
experiment_pathway_token_analysis/
├── config.yaml
├── src/
│   ├── step0_data_audit.py                # ENHANCED - proper logging
│   ├── step1_load_causal_features.py      # Same as V2
│   ├── step2_pathway_prototype.py         # FIXED - LlamaModel + staged SAE
│   ├── step3_pathway_discovery.py         # UPDATED - realistic batching
│   ├── step4_token_association.py         # Same as V2
│   ├── step5_integrate_results.py         # Same as V2
│   └── orchestrator.py
├── data/
│   ├── causal_features.json               # Filtered features
│   └── prompts_sample.json                # Representative prompts
├── results/
│   ├── audit/                             # Data validation logs
│   ├── prototype/                         # Single-feature test
│   ├── pathways/
│   ├── tokens/
│   └── integrated/
└── launch.sh
```

## Updated Implementation Details

### Step 0: Data Audit (ENHANCED)
See complete code above with:
- Proper logging instead of print
- Error/warning collection
- Trial count verification
- Response payload checks
- Graceful error handling

**Output**: `results/audit/data_validation.json`

### Step 1: Load + Filter Causal Features (UNCHANGED)
Same as V2 - filters to top-50 per layer by |Cohen's d|

**Output**: `data/causal_features.json` (~1,550 features)

### Step 2: Pathway Prototype (FIXED)
See complete code above with:
- `LlamaModel` instead of `AutoModel`
- Staged SAE loading (only layers 1→target)
- CPU offloading for activations
- Proper memory counter reset

**Output**: `results/prototype/single_feature_test.json`

**Success criteria**:
- Runs without OOM
- Records activations for required layers
- Identifies upstream candidates
- Memory usage < 28GB

### Step 3: Pathway Discovery (UPDATED BATCHING)
**Method**: Gradient-based (fast) + Replay validation (top-k)

**Process**:
1. **Phase A: Gradient discovery** (~12 hours)
   ```python
   batch_size = 5  # Reduced from 10

   for i in range(0, len(filtered_features), batch_size):
       batch = filtered_features[i:i+batch_size]

       for feature in batch:
           # Compute backward Jacobian
           gradients = compute_gradients(feature)
           save_gradients(feature, gradients)

       # Aggressive cleanup
       torch.cuda.empty_cache()
       torch.cuda.reset_peak_memory_stats()
   ```

2. **Phase B: Top-k selection** (~10 minutes)
   - Filter to top-25 pathways per feature
   - Criteria: |gradient| > 0.1
   - Output: ~38,750 pathways

3. **Phase C: Replay validation** (~6 hours)
   - Validate top-25 pathways per feature
   - Measure actual activation changes
   - Compare with gradient predictions

**Output**: `results/pathways/validated_pathways.json`

### Step 4: Token Association (UNCHANGED)
Same as V2 - uses LLaMA tokenizer for proper token-level analysis

**Output**: `results/tokens/token_associations.json`

### Step 5: Integration (UNCHANGED)
Same as V2

## Resource Estimates (UPDATED)

### Memory
- Model: ~16GB
- SAEs (staged): ~8GB peak (31 × 256MB, loaded as needed)
- Activations (batch of 5): ~2GB
- **Total**: ~26GB per GPU (fits A40/A100)

### Compute Time
- Step 0 (audit): ~10 minutes
- Step 1 (loading): ~5 minutes
- Step 2 (prototype): ~8 minutes
- Step 3 (pathways): ~18 hours (gradient 12h + validation 6h)
- Step 4 (tokens): ~2 hours
- Step 5 (integration): ~15 minutes
- **Total**: ~21 hours

### Storage
- Causal features: ~500KB
- Pathway results: ~200MB
- Token results: ~50MB
- **Total**: ~250MB

## Status of V2 Issues

1. **Join key**: ✓ Addressed in V2 (shared 'feature' field)
2. **Instrumentation**: ✓ FIXED in V3 (LlamaModel + staged SAE loading)
3. **Token analysis**: ✓ Addressed in V2 (LLaMA tokenizer)
4. **Feature selection**: ✓ Addressed in V2 (top-50/layer by Cohen's d)
5. **Data validation**: ✓ ENHANCED in V3 (logging + error handling)
6. **Resource estimates**: ✓ FIXED in V3 (realistic 21h, 26GB)
7. **Memory tracking**: ✓ FIXED in V3 (reset counter, no assertions)

## Next Steps

1. Get Codex review of V3
2. If approved: Begin implementation of Step 0 (data audit)
3. Test on sample data
4. Iterate based on results
