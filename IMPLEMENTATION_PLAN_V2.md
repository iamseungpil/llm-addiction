# Implementation Plan V2: Addressing Codex Feedback

## Changes from V1

### HIGH Priority Fixes

**Issue 1**: Checkpoint-Log join unclear
**Fix**: Define explicit schema mapping
```python
# Checkpoint: feature = "L29-2374"
# Response log: feature = "L29-2374"
# Join key: Exact string match on 'feature' field
```

**Issue 2**: LLaMA/SAE instrumentation missing
**Fix**: Add prototype step + batching strategy

### MEDIUM Priority Fixes

**Issue 3**: Token analysis uses raw text, not tokenizer
**Fix**: Re-tokenize with LLaMA tokenizer

**Issue 4**: Feature selection undefined
**Fix**: Top-50 per layer by |Cohen's d|

**Issue 5**: No data validation
**Fix**: Add Step 0 (data audit)

## Revised Structure

```
experiment_pathway_token_analysis/
├── config.yaml
├── src/
│   ├── step0_data_audit.py                # NEW - validate data integrity
│   ├── step1_load_causal_features.py      # Load + filter top-50/layer
│   ├── step2_pathway_prototype.py         # NEW - single feature test
│   ├── step3_pathway_discovery.py         # Scaled pathway analysis
│   ├── step4_token_association.py         # Token-aware analysis
│   ├── step5_integrate_results.py         # Integration
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

## Detailed Implementation

### Step 0: Data Audit (NEW)

**Purpose**: Validate Exp2 data before analysis

**Process**:
```python
def audit_data():
    # 1. Load all checkpoints
    checkpoints = load_checkpoint_files()

    # 2. Verify schema
    required_fields = ['feature', 'layer', 'feature_id', 'is_causal', 'cohen_d']
    for checkpoint in checkpoints:
        for result in checkpoint['results']:
            assert all(f in result for f in required_fields)

    # 3. Count causal features per layer
    causal_by_layer = defaultdict(int)
    for checkpoint in checkpoints:
        for result in checkpoint['results']:
            if result['is_causal']:
                causal_by_layer[result['layer']] += 1

    # 4. Verify response logs
    response_files = list(Path("response_logs").glob("*.json"))
    for file in response_files:
        responses = json.load(open(file))
        # Check: Each feature has 6 conditions × ~30 trials
        features = set(r['feature'] for r in responses)
        for feature in features:
            feature_responses = [r for r in responses if r['feature'] == feature]
            conditions = set(r['condition'] for r in feature_responses)
            assert len(conditions) == 6  # All 6 conditions present

    # 5. Test join: Can we link checkpoint features to response logs?
    checkpoint_features = set()
    for checkpoint in checkpoints:
        for result in checkpoint['results']:
            if result['is_causal']:
                checkpoint_features.add(result['feature'])

    response_features = set()
    for file in response_files:
        responses = json.load(open(file))
        response_features.update(r['feature'] for r in responses)

    intersection = checkpoint_features & response_features
    print(f"Causal features in checkpoints: {len(checkpoint_features)}")
    print(f"Features in response logs: {len(response_features)}")
    print(f"Joinable features: {len(intersection)}")

    return {
        'causal_by_layer': causal_by_layer,
        'joinable_features': intersection,
        'validated': True
    }
```

**Output**: `results/audit/data_validation.json`

### Step 1: Load + Filter Causal Features (UPDATED)

**Selection criteria**: Top-50 per layer by |Cohen's d|

**Rationale**:
- Balances coverage vs compute
- Layer-stratified ensures representation across network
- Cohen's d measures effect size (stronger effects = clearer pathways)

**Process**:
```python
def load_filtered_features():
    # Load all causal features
    all_causal = []
    for checkpoint in load_checkpoints():
        for result in checkpoint['results']:
            if result['is_causal']:
                all_causal.append(result)

    # Filter: Top-50 per layer
    filtered = []
    for layer in range(1, 32):
        layer_features = [f for f in all_causal if f['layer'] == layer]
        layer_features.sort(key=lambda x: abs(x['cohen_d']), reverse=True)
        filtered.extend(layer_features[:50])

    print(f"Total causal features: {len(all_causal)}")
    print(f"Filtered (top-50/layer): {len(filtered)}")

    return filtered
```

**Output**: `data/causal_features.json` (~1,550 features, 31 layers × 50)

### Step 2: Pathway Prototype (NEW)

**Purpose**: Test single-feature pipeline before scaling

**Process**:
```python
import torch
from transformers import AutoModel, AutoTokenizer
from llama_scope import LlamaScopeDirect

def prototype_single_feature(feature_info):
    """
    Test pathway discovery for ONE feature

    Args:
        feature_info: {'feature': 'L26-1069', 'layer': 26, 'feature_id': 1069}
    """
    device = 'cuda:0'

    # 1. Load model + tokenizer
    model = AutoModel.from_pretrained("meta-llama/Llama-3.1-8B").to(device)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

    # 2. Load SAEs for ALL layers
    saes = {}
    for layer in range(1, 32):
        saes[layer] = LlamaScopeDirect(layer=layer, SAE_model="RES-16K")

    # 3. Get sample prompt from response logs
    prompt = get_representative_prompt(feature_info['feature'])
    inputs = tokenizer(prompt, return_tensors='pt').to(device)

    # 4. Forward pass with activation recording
    activations = {}  # {layer: feature_activations}

    def record_hook(layer_idx):
        def hook(module, input, output):
            hidden = output[0][:, -1, :]  # Last token
            sae = saes[layer_idx]
            features = sae.encode(hidden)
            activations[layer_idx] = features.squeeze().detach()
        return hook

    # Register hooks on all layers
    hooks = []
    for layer in range(1, 32):
        hook = model.model.layers[layer].register_forward_hook(record_hook(layer))
        hooks.append(hook)

    # Run forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # 5. Analyze activations
    target_layer = feature_info['layer']
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

**Output**: `results/prototype/single_feature_test.json`

**Success criteria**:
- Runs without OOM
- Records activations for all layers
- Identifies upstream candidates
- Memory usage < 20GB

### Step 3: Pathway Discovery (SCALED)

**Method**: Gradient-based (fast) + Replay validation (top-k)

**Process**:
1. **Phase A: Gradient discovery** (~2 hours)
   - For each filtered feature (1,550 total):
     - Compute backward Jacobian
     - Record gradient strengths to upstream features
   - Output: All potential pathways

2. **Phase B: Top-k selection** (~5 minutes)
   - Filter to top-25 pathways per feature
   - Criteria: |gradient| > 0.1
   - Output: ~38,750 pathways (1,550 features × 25)

3. **Phase C: Replay validation** (~4 hours)
   - For top-25 pathways per feature:
     - Patch source feature
     - Measure actual target feature Δactivation
     - Compare with gradient prediction
   - Output: Validated pathways with correlation scores

**Batching strategy**:
```python
# Process 10 features at a time to manage memory
batch_size = 10
for i in range(0, len(filtered_features), batch_size):
    batch = filtered_features[i:i+batch_size]
    process_batch(batch)
    torch.cuda.empty_cache()
```

**Output**: `results/pathways/validated_pathways.json`

### Step 4: Token Association (UPDATED)

**Fix**: Use LLaMA tokenizer, not raw text

**Process**:
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

def analyze_tokens(feature):
    # 1. Load baseline vs patched responses
    baseline_responses = load_responses(feature, condition='safe_baseline')
    patched_responses = load_responses(feature, condition='safe_with_risky_patch')

    # 2. Tokenize with LLaMA tokenizer
    baseline_tokens = []
    for response in baseline_responses:
        tokens = tokenizer.encode(response, add_special_tokens=False)
        baseline_tokens.extend(tokens)

    patched_tokens = []
    for response in patched_responses:
        tokens = tokenizer.encode(response, add_special_tokens=False)
        patched_tokens.extend(tokens)

    # 3. Compute frequency shifts
    baseline_freq = Counter(baseline_tokens)
    patched_freq = Counter(patched_tokens)

    # 4. Statistical testing
    token_effects = []
    for token_id in set(baseline_freq.keys()) | set(patched_freq.keys()):
        baseline_count = baseline_freq[token_id]
        patched_count = patched_freq[token_id]

        # Log-odds ratio
        odds_ratio = np.log((patched_count + 1) / (baseline_count + 1))

        # Chi-square test
        contingency = [[baseline_count, len(baseline_tokens) - baseline_count],
                       [patched_count, len(patched_tokens) - patched_count]]
        chi2, p_value = chi2_contingency(contingency)[:2]

        if abs(odds_ratio) > 0.5 and p_value < 0.05:
            token_text = tokenizer.decode([token_id])
            token_effects.append({
                'token': token_text,
                'token_id': token_id,
                'odds_ratio': odds_ratio,
                'p_value': p_value,
                'baseline_count': baseline_count,
                'patched_count': patched_count
            })

    return token_effects
```

**Output**: `results/tokens/token_associations.json`

### Step 5: Integration (UNCHANGED)

Same as V1

## Resource Estimates

### Memory
- Model: ~16GB
- 31 SAEs: ~8GB (with LRU cache)
- Activations (batch of 10): ~4GB
- **Total**: ~28GB per GPU (fits A40/A100)

### Compute Time
- Step 0 (audit): ~5 minutes
- Step 1 (loading): ~2 minutes
- Step 2 (prototype): ~5 minutes
- Step 3 (pathways): ~6 hours (gradient 2h + validation 4h)
- Step 4 (tokens): ~1 hour
- Step 5 (integration): ~10 minutes
- **Total**: ~7.5 hours

### Storage
- Causal features: ~500KB
- Pathway results: ~200MB
- Token results: ~50MB
- **Total**: ~250MB

## Next Steps

1. Implement Step 0 (data audit)
2. Test on sample data
3. Get Codex review
4. Iterate
