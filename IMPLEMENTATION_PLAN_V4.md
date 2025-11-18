# Implementation Plan V4: Final Fixes

## Changes from V3

### HIGH Priority Fix 1: Correct Instrumentation

**Issue (from V3)**: Hook assumes output is tuple, but `LlamaDecoderLayer` returns tensor

**Fix in V4**: Use `output_hidden_states=True` (NO hooks needed!)

Based on working code from `experiment_1_layer_pathway_L1_31/src/gradient_pathway_tracker.py`:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

def prototype_single_feature(feature_info):
    """
    Test pathway discovery for ONE feature - CORRECTED instrumentation
    """
    device = 'cuda:0'

    # 1. Load AutoModelForCausalLM (NOT LlamaModel!)
    model_name = 'fnlp/Llama3_1-8B-Base-LXR-8x'
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 2. Load ONLY required SAEs (staged loading)
    target_layer = feature_info['layer']

    saes = {}
    for layer in range(1, target_layer + 1):
        saes[layer] = LlamaScopeDirect(layer=layer, SAE_model="RES-16K")

    # 3. Get sample prompt from response logs
    prompt = get_representative_prompt(feature_info['feature'])
    inputs = tokenizer(prompt, return_tensors='pt').to(device)

    # 4. Forward pass WITHOUT hooks - use output_hidden_states
    activations = {}

    with torch.no_grad():
        outputs = model(
            inputs.input_ids,
            output_hidden_states=True,  # KEY: Returns ALL layer hidden states
            use_cache=False
        )

        # 5. Extract SAE features from hidden states
        for layer in range(1, target_layer + 1):
            # Access hidden state directly from outputs tuple
            hidden = outputs.hidden_states[layer][:, -1, :]  # Last token

            sae = saes[layer]
            features = sae.encode(hidden)
            activations[layer] = features.squeeze().detach().cpu()  # Move to CPU

    # Clear GPU memory
    del saes, outputs
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # 6. Analyze activations (on CPU)
    target_id = feature_info['feature_id']

    print(f"Target feature: L{target_layer}-{target_id}")
    print(f"Activation: {activations[target_layer][target_id].item():.4f}\"")

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

**Key changes**:
- Uses `AutoModelForCausalLM` (proven to work in existing Exp1)
- Uses `output_hidden_states=True` instead of forward hooks
- Accesses `outputs.hidden_states[layer]` directly (tensor, not tuple)
- NO hook registration needed!
- Simpler, more reliable

### HIGH Priority Fix 2: Complete Data Audit

**Issue (from V3)**: Only checks first 10 files and 5 features

**Fix in V4**: Check ALL files with progress tracking

```python
import logging
from collections import defaultdict
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def audit_data():
    """
    COMPLETE data validation - checks ALL files
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
    for file in tqdm(checkpoint_files, desc="Loading checkpoints"):
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                checkpoints.append(data)
        except Exception as e:
            audit_results['errors'].append(f"Failed to load {file}: {e}")
            return audit_results

    # 2. Verify schema for ALL results
    required_fields = ['feature', 'layer', 'feature_id', 'is_causal', 'cohen_d']

    for checkpoint in checkpoints:
        for result in checkpoint.get('results', []):
            missing_fields = [f for f in required_fields if f not in result]
            if missing_fields:
                audit_results['errors'].append(
                    f"Missing fields in {result.get('feature', 'unknown')}: {missing_fields}"
                )
                return audit_results

    logger.info("✓ Schema validation passed (all checkpoints)")

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

    # 4. Verify response logs - CHECK ALL FILES
    response_dir = Path("/data/llm_addiction/experiment_2_multilayer_patching/response_logs")
    response_files = list(response_dir.glob("*.json"))

    if not response_files:
        audit_results['errors'].append("No response log files found")
        return audit_results

    logger.info(f"Found {len(response_files)} response log files")

    # Check trial counts per feature - PROCESS ALL FILES
    feature_trial_counts = defaultdict(lambda: defaultdict(int))
    response_features = set()
    total_trials = 0

    for file in tqdm(response_files, desc="Validating response logs"):
        try:
            with open(file, 'r') as f:
                responses = json.load(f)
        except Exception as e:
            audit_results['errors'].append(f"Failed to load {file}: {e}")
            return audit_results

        for response in responses:
            total_trials += 1

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

    logger.info(f"✓ Total trials validated: {total_trials:,}")
    logger.info(f"✓ Total features in response logs: {len(response_features)}")

    # Check if all 6 conditions present for ALL features
    expected_conditions = [
        'safe_baseline', 'safe_with_risky_patch',
        'risky_baseline', 'risky_with_safe_patch',
        'safe_amplification', 'risky_amplification'
    ]

    missing_condition_count = 0
    low_trial_count = 0

    for feature, condition_counts in feature_trial_counts.items():
        missing_conditions = [c for c in expected_conditions if c not in condition_counts]
        if missing_conditions:
            missing_condition_count += 1
            if missing_condition_count <= 10:  # Only log first 10
                audit_results['warnings'].append(
                    f"{feature} missing conditions: {missing_conditions}"
                )

        # Check trial counts
        for condition, count in condition_counts.items():
            if count < 20:  # Warning if < 20 trials
                low_trial_count += 1
                if low_trial_count <= 10:  # Only log first 10
                    audit_results['warnings'].append(
                        f"{feature} {condition} has only {count} trials (expected ~30)"
                    )

    if missing_condition_count > 0:
        logger.warning(f"⚠ {missing_condition_count} features missing some conditions")
    if low_trial_count > 0:
        logger.warning(f"⚠ {low_trial_count} feature-conditions with <20 trials")

    logger.info("✓ Response log validation passed")

    # 5. Test join: Can we link checkpoint features to response logs?
    checkpoint_features = set(all_causal_features)
    intersection = checkpoint_features & response_features

    logger.info(f"Causal features in checkpoints: {len(checkpoint_features)}")
    logger.info(f"Features in response logs: {len(response_features)}")
    logger.info(f"Joinable features: {len(intersection)}")

    join_rate = len(intersection) / len(checkpoint_features) * 100 if checkpoint_features else 0
    logger.info(f"Join rate: {join_rate:.1f}%")

    if join_rate < 90:
        audit_results['warnings'].append(
            f"Low join rate: {join_rate:.1f}% (expected >90%)"
        )

    # Success criteria
    if len(audit_results['errors']) == 0:
        audit_results['validated'] = True
        logger.info("✅ Data audit PASSED - ALL files validated")
    else:
        logger.error(f"❌ Data audit FAILED: {len(audit_results['errors'])} errors")

    if audit_results['warnings']:
        logger.warning(f"⚠️  {len(audit_results['warnings'])} warnings (non-blocking)")

    return {
        'causal_by_layer': dict(causal_by_layer),
        'total_causal_features': len(all_causal_features),
        'total_response_features': len(response_features),
        'joinable_features': len(intersection),
        'join_rate_percent': join_rate,
        'total_trials': total_trials,
        'audit_results': audit_results,
        'validated': audit_results['validated']
    }
```

**Key changes**:
- Uses `tqdm` for progress tracking
- Processes ALL checkpoint files (not just 10)
- Validates ALL response log files (not just 10)
- Checks ALL features (not just 5)
- Reports comprehensive statistics (total trials, join rate)
- Limits warning spam (only logs first 10 of each type)

## Revised Structure (no changes)

```
experiment_pathway_token_analysis/
├── config.yaml
├── src/
│   ├── step0_data_audit.py                # FIXED - validates ALL files
│   ├── step1_load_causal_features.py      # Same as V3
│   ├── step2_pathway_prototype.py         # FIXED - uses output_hidden_states
│   ├── step3_pathway_discovery.py         # Same as V3
│   ├── step4_token_association.py         # Same as V3
│   ├── step5_integrate_results.py         # Same as V3
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

### Step 0: Data Audit (FIXED - ALL FILES)
See complete code above - now validates ALL checkpoints and response logs with progress tracking

**Output**: `results/audit/data_validation.json`

**Improvements**:
- Processes all 358 checkpoint files
- Validates all 199 response log files
- Reports total trial count (~1.8M)
- Calculates join rate between checkpoints and logs

### Step 1: Load + Filter Causal Features (UNCHANGED)
Same as V3 - filters to top-50 per layer by |Cohen's d|

**Output**: `data/causal_features.json` (~1,550 features)

### Step 2: Pathway Prototype (FIXED - NO HOOKS)
See complete code above - uses `output_hidden_states=True` instead of hooks

**Output**: `results/prototype/single_feature_test.json`

**Success criteria**:
- Runs without OOM
- Records activations for required layers
- Identifies upstream candidates
- Memory usage < 28GB

### Step 3: Pathway Discovery (UNCHANGED)
Same as V3 - gradient + replay validation with batch size 5

**Output**: `results/pathways/validated_pathways.json`

### Step 4: Token Association (UNCHANGED)
Same as V3 - uses LLaMA tokenizer

**Output**: `results/tokens/token_associations.json`

### Step 5: Integration (UNCHANGED)
Same as V3

## Resource Estimates (UNCHANGED)

### Memory
- Model: ~16GB
- SAEs (staged): ~8GB peak
- Activations (batch of 5): ~2GB
- **Total**: ~26GB per GPU (fits A40/A100)

### Compute Time
- Step 0 (audit): ~15 minutes (now checks ALL files)
- Step 1 (loading): ~5 minutes
- Step 2 (prototype): ~8 minutes
- Step 3 (pathways): ~18 hours
- Step 4 (tokens): ~2 hours
- Step 5 (integration): ~15 minutes
- **Total**: ~21.5 hours

### Storage
- Causal features: ~500KB
- Pathway results: ~200MB
- Token results: ~50MB
- **Total**: ~250MB

## Status of V3 Issues

1. **Hook instrumentation**: ✅ FIXED in V4 (uses output_hidden_states, no hooks)
2. **Data audit sampling**: ✅ FIXED in V4 (checks ALL files with progress tracking)

## Status of ALL Historical Issues

1. Join key (V1): ✅ Resolved in V2
2. Instrumentation (V1): ✅ FIXED in V4 (output_hidden_states method)
3. Token analysis (V1): ✅ Resolved in V2
4. Feature selection (V1): ✅ Resolved in V2
5. Data validation (V1): ✅ FIXED in V4 (complete audit)
6. Resource estimates (V2): ✅ Resolved in V3
7. Memory tracking (V2): ✅ Resolved in V3

## Next Steps

1. Get Codex review of V4
2. If approved: Begin implementation
3. Run Step 0 (data audit) on actual data
4. Run Step 2 (prototype) to validate instrumentation
5. If prototype succeeds → implement full pipeline
