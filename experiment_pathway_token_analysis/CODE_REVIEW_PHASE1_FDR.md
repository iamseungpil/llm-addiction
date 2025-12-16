# Code Review: Phase 1 FDR-265 Features

**Review Date**: 2025-12-08
**Reviewer**: Claude Code (Senior Code Reviewer)
**Files Reviewed**:
- `/home/ubuntu/llm_addiction/experiment_pathway_token_analysis/src/phase1_FDR_265features.py`
- `/home/ubuntu/llm_addiction/experiment_pathway_token_analysis/launch_phase1_FDR_265.sh`

---

## Executive Summary

**Overall Assessment**: CRITICAL ISSUES FOUND - Code requires fixes before deployment

**Critical Issues**: 3
**Warnings**: 4
**Suggestions**: 3

The Phase 1 script has several critical bugs that will cause runtime failures:
1. Hidden states indexing mismatch (off-by-one error)
2. Incorrect prompt parsing logic that extracts wrong text
3. Missing bet amount extraction for Phase 1 outputs

---

## CRITICAL ISSUES (Must Fix)

### 1. Hidden States Index Off-By-One Error (Lines 550-552)

**Severity**: CRITICAL - Will extract wrong layer activations

**Issue**: The code has conflicting comments and uses incorrect indexing:

```python
# Line 550-552 (WRONG)
for layer in layers_to_extract:
    # hidden_states[layer] is output of layer-1, so we need layer+1
    # But for SAE trained on residual stream after layer L, we use index L
    hidden_states_dict[layer] = full_outputs.hidden_states[layer]
```

**Problem**:
- `full_outputs.hidden_states` is a tuple where index 0 = input embeddings, index 1 = layer 0 output, etc.
- To get output of layer L, you need index L+1, NOT index L
- Current code will extract layer L-1's output when trying to patch layer L

**Evidence from LLaMA architecture**:
- `hidden_states[0]` = embedding layer output (before any transformer blocks)
- `hidden_states[13]` = output of layer 12 (block index 12)
- `hidden_states[14]` = output of layer 13 (block index 13) ← THIS is what SAE L13 was trained on

**Correct Fix**:
```python
for layer in layers_to_extract:
    # hidden_states[0] = embeddings, hidden_states[L+1] = output of layer L
    hidden_states_dict[layer] = full_outputs.hidden_states[layer + 1]
```

**Impact**: Without this fix, ALL feature extractions will be from the wrong layer, invalidating the entire experiment.

---

### 2. Incorrect Prompt Text Extraction (Lines 532-533, 583-584)

**Severity**: CRITICAL - Response parsing will fail

**Issue**: The code tries to strip the prompt using string length, but tokenization doesn't preserve character-level alignment:

```python
# Lines 532-533 (WRONG)
response = self.tokenizer.decode(full_sequence, skip_special_tokens=True)
response = response[len(prompt):].strip()  # This is WRONG
```

**Problem**:
- `len(prompt)` gives CHARACTER count (e.g., 350 characters)
- But after tokenization → detokenization, the text may have different length due to:
  - Whitespace normalization
  - Special character handling
  - Tokenizer artifacts

**Example of Failure**:
```python
prompt = "Slot Machine Game\n\nGame history:\nRound 1..."  # 350 chars
tokenized = tokenizer(prompt)  # 89 tokens
decoded = tokenizer.decode(tokenized)  # May be 352 chars (extra spaces)
response = decoded[350:]  # WRONG: Cuts into the prompt text
```

**Correct Fix**:
```python
# Use token-level slicing instead
generated_token_ids = full_sequence[prompt_len:].tolist()
response = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
```

**Impact**: Responses will include partial prompt text or cut off the actual response, breaking bet amount extraction.

---

### 3. Missing Bet Amount Extraction for Phase 1 (Lines 727-739)

**Severity**: CRITICAL - Data will be incomplete

**Issue**: Phase 1 records don't include `bet_amount`, `is_stop`, or `action` fields that are needed for downstream analysis.

**Current Output**:
```python
record = {
    'target_feature': feature_name,
    # ...
    'response': response,
    'generated_token_ids': token_ids,
    'generated_tokens': tokens,
    'all_features': all_activations
    # MISSING: bet_amount, is_stop, action
}
```

**Problem**:
- Phase 2+ may need to filter by bet behavior
- FDR analysis in other scripts expects `is_stop` field
- Inconsistent with patching experiment output format

**Correct Fix**: Add parsing logic
```python
# Parse response for action
is_stop = self.parse_is_stop(response)
bet_amount = self.parse_bet_amount(response) if not is_stop else 0

record = {
    # ... existing fields ...
    'action': 'stop' if is_stop else 'bet',
    'bet_amount': bet_amount,
    'is_stop': is_stop,
    'is_valid': True  # Mark invalid if parsing fails
}
```

**Impact**: Downstream analysis scripts will fail or require data reformatting.

---

## WARNINGS (Should Fix)

### 4. FDR Effect Size Filter Too Strict (Line 278)

**Severity**: WARNING - May lose valid features

**Issue**: The minimum effect size of 5% (`max_effect >= 0.05`) is applied AFTER FDR correction:

```python
# Line 278 (may be too strict)
if max_effect >= 0.05:  # Minimum 5% change in stop rate
    fdr_features.append(...)
```

**Problem**:
- FDR correction already controls false discovery rate
- Adding effect size filter defeats the purpose of FDR
- A feature with p=0.001 but effect=0.04 would be excluded despite high statistical confidence

**Recommendation**: Either:
1. Remove the effect size filter (FDR is sufficient)
2. Apply effect size filter BEFORE FDR (as a pre-filter)
3. Document the scientific rationale for double filtering

---

### 5. Chi-Square Test May Fail with Low Counts (Lines 189-218)

**Severity**: WARNING - Statistical test assumptions violated

**Issue**: Chi-square test requires expected frequencies ≥5 in each cell, but code doesn't check:

```python
# Line 198-202 (no count validation)
contingency = [[baseline_stop, baseline_cont], [patched_stop, patched_cont]]
try:
    _, p_safe, _, _ = stats.chi2_contingency(contingency)
except:
    p_safe = 1.0
```

**Problem**:
- If baseline_stop = 1, baseline_cont = 199, the chi-square approximation is invalid
- Should use Fisher's exact test for low counts

**Recommendation**:
```python
# Check if any expected frequency < 5
total = baseline_stop + baseline_cont + patched_stop + patched_cont
if min(baseline_stop, baseline_cont, patched_stop, patched_cont) < 5:
    # Use Fisher's exact test
    from scipy.stats import fisher_exact
    odds_ratio, p_safe = fisher_exact([[baseline_stop, baseline_cont],
                                        [patched_stop, patched_cont]])
else:
    _, p_safe, _, _ = stats.chi2_contingency(contingency)
```

---

### 6. Checkpoint Resume May Skip Features (Lines 610-632, 677)

**Severity**: WARNING - Data loss risk

**Issue**: Checkpoint uses `(feature_name, patch_condition, prompt_type, trial)` as key, but if FDR features change between runs, old checkpoints become invalid:

```python
# Line 616-627
completed = set()
for line in checkpoint_file:
    record = json.loads(line)
    key = (record['target_feature'], ...)
    completed.add(key)
```

**Problem**:
1. If you rerun with different FDR alpha → different feature list
2. Old checkpoint has features not in new FDR list → wasted computation
3. New FDR list has features not in checkpoint → correctly handled

**Recommendation**: Add FDR cache validation:
```python
def load_checkpoint(self, checkpoint_file, current_fdr_features):
    completed = set()
    current_features = {f['feature_name'] for f in current_fdr_features}

    for line in checkpoint_file:
        record = json.loads(line)
        # Only count if feature is in current FDR list
        if record['target_feature'] in current_features:
            completed.add((record['target_feature'], ...))

    return completed
```

---

### 7. No Memory Cleanup Between Trials (Lines 723-751)

**Severity**: WARNING - GPU memory leak

**Issue**: The experiment loop processes 50 trials per condition without explicit memory cleanup:

```python
# Line 716-751 (no cleanup between trials)
for trial in range(self.config.n_trials):
    response, all_activations, token_ids, tokens = self.generate_with_patching(...)
    # ... save record ...
    # NO torch.cuda.empty_cache() here
```

**Problem**:
- Each trial allocates tensors for generation
- Even with `del inputs` in generate functions, PyTorch may cache
- Over 50 trials × 265 features = 13,250 generations without cleanup

**Recommendation**:
```python
for trial in range(self.config.n_trials):
    try:
        response, all_activations, token_ids, tokens = self.generate_with_patching(...)
        # ... save record ...
    finally:
        # Cleanup every N trials
        if trial % 10 == 9:
            torch.cuda.empty_cache()
            gc.collect()
```

---

## SUGGESTIONS (Consider Improving)

### 8. Hardcoded Feature Extraction Logic (Lines 459-463)

**Severity**: SUGGESTION - Reduce code duplication

**Issue**: Feature extraction iterates through ALL 265 features for each layer:

```python
# Line 459-463 (inefficient)
for feat in self.all_features:  # 265 iterations per layer
    if feat['layer'] == layer:
        feat_id = feat['feature_id']
        feat_name = f"L{layer}-{feat_id}"
        all_activations[feat_name] = float(final_acts[feat_id].item())
```

**Recommendation**: Pre-build layer→features mapping:
```python
# In __init__
self.features_by_layer = defaultdict(list)
for feat in self.all_features:
    self.features_by_layer[feat['layer']].append(feat)

# In extract_all_features
for layer, hidden_states in hidden_states_dict.items():
    sae = self.load_sae(layer)
    feature_acts = sae.encode(hidden_states.float())
    final_acts = feature_acts[0, -1, :]

    for feat in self.features_by_layer[layer]:  # Much smaller iteration
        all_activations[f"L{layer}-{feat['feature_id']}"] = float(final_acts[feat['feature_id']].item())
```

---

### 9. Launch Script Assumes Even Distribution (Lines 126-131)

**Severity**: SUGGESTION - Improve GPU load balancing

**Issue**: The script divides features equally, but FDR count is unknown:

```bash
# Line 126-127
ESTIMATED_FDR_FEATURES=130
FEATURES_PER_GPU=$(( (ESTIMATED_FDR_FEATURES + NUM_GPUS - 1) / NUM_GPUS ))
```

**Recommendation**:
1. Run FDR analysis once first to get exact count
2. Distribute based on actual count
3. Or use dynamic work stealing (checkpoint-based)

---

### 10. Missing Validation for Feature Means (Lines 383-390)

**Severity**: SUGGESTION - Add data validation

**Issue**: The code warns about missing means but continues:

```python
# Line 383-390
if missing:
    LOGGER.warning(f"{len(missing)} features missing mean values: {missing[:5]}...")
    # Continues anyway!
```

**Recommendation**: Either fail fast or provide fallback:
```python
if missing:
    LOGGER.error(f"CRITICAL: {len(missing)} features missing mean values")
    LOGGER.error(f"Missing features: {missing}")
    raise ValueError("Cannot proceed without feature means")
```

---

## Detailed Analysis

### FDR Analysis Correctness

The Benjamini-Hochberg implementation (lines 235-293) is **mostly correct** but has one issue:

**Lines 249-250**: Using minimum p-value approach
```python
p_values_min = [min(ps, pr) for ps, pr in zip(p_values_safe, p_values_risky)]
```

**Analysis**:
- This is a conservative approach (uses most significant p-value)
- Correct for "significant in either context" logic
- However, should document that this is NOT the standard Bonferroni-Holm method
- Consider using Simes method for more power

**Recommendation**: Add comment explaining the choice:
```python
# Use minimum p-value approach: feature passes if significant in EITHER context
# This is conservative (uses best evidence) but may miss features with
# moderate effects in both contexts
p_values_min = [min(ps, pr) for ps, pr in zip(p_values_safe, p_values_risky)]
```

---

### Effect Computation Logic

**Lines 175-186**: Stop rate calculation is CORRECT

```python
for (patch_cond, prompt_type), trials in conds.items():
    valid_trials = [t for t in trials if t.get('is_valid', True)]
    if valid_trials:
        stop_count = sum(1 for t in valid_trials if t.get('is_stop', False))
        stop_rates[(patch_cond, prompt_type)] = stop_count / len(valid_trials)
```

**Analysis**:
- Correctly filters invalid trials
- Correctly computes proportion
- Handles missing fields with default values

**Recommendation**: Add zero-division check for edge cases:
```python
if valid_trials and len(valid_trials) > 0:  # Redundant but explicit
    ...
else:
    stop_rates[(patch_cond, prompt_type)] = 0.0
```

---

### Patching Logic Review

**Lines 491-514**: Patching hook is CORRECT

```python
def patching_hook(module, input, output):
    # Patch only last token position
    last_token = hidden_states[:, -1:, :].float()
    features = sae.encode(last_token)
    features[0, 0, target_feature_id] = float(patch_value)
    patched_hidden = sae.decode(features)
    hidden_states[:, -1:, :] = patched_hidden.to(original_dtype)
```

**Analysis**:
- ✅ Correctly patches last token only (causal position)
- ✅ Correctly encodes → patches → decodes
- ✅ Preserves dtype
- ✅ Handles tuple vs single return

**No issues found in patching logic**

---

### File I/O and Checkpoint Logic

**Lines 610-632**: Checkpoint loading is FUNCTIONAL but could be more robust

**Issues**:
- No version checking (what if record format changes?)
- No corruption handling (partial JSON lines)
- No duplicate detection

**Recommendation**: Add validation:
```python
def load_checkpoint(self, checkpoint_file: Path) -> Set[Tuple]:
    completed = set()
    corrupted_lines = 0

    for line_num, line in enumerate(f, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            # Validate required fields
            required = ['target_feature', 'patch_condition', 'prompt_type', 'trial']
            if all(k in record for k in required):
                key = tuple(record[k] for k in required)
                if key in completed:
                    LOGGER.warning(f"Duplicate trial found at line {line_num}")
                completed.add(key)
            else:
                LOGGER.warning(f"Line {line_num}: Missing required fields")
        except json.JSONDecodeError:
            corrupted_lines += 1

    if corrupted_lines > 0:
        LOGGER.warning(f"Skipped {corrupted_lines} corrupted lines")

    return completed
```

---

## Testing Recommendations

Before deploying this code, test the following:

### 1. Unit Tests Needed

```python
# Test hidden states indexing
def test_hidden_states_extraction():
    """Verify we extract from correct layer"""
    model = load_model()
    sae_l13 = load_sae(13)

    # Forward pass with known input
    outputs = model(input_ids, output_hidden_states=True)

    # Extract layer 13 output
    hidden_l13 = outputs.hidden_states[14]  # Index 14 for layer 13

    # Encode and verify
    features = sae_l13.encode(hidden_l13)
    # ... verify features make sense

# Test prompt extraction
def test_response_parsing():
    """Verify response doesn't include prompt text"""
    prompt = SAFE_PROMPT
    response_with_prompt = prompt + "1\nBet $10"

    # Current (wrong) method
    wrong = response_with_prompt[len(prompt):]

    # Correct method
    token_ids = tokenizer(response_with_prompt)['input_ids']
    prompt_len = len(tokenizer(prompt)['input_ids'])
    correct = tokenizer.decode(token_ids[prompt_len:])

    assert "Choice:" not in correct
    assert "1" in correct
```

### 2. Integration Tests

```bash
# Test on 1 feature, 2 trials
python phase1_FDR_265features.py \
    --gpu-id 0 \
    --n-trials 2 \
    --limit 1 \
    --output-dir /tmp/test_phase1

# Verify output format
python -c "
import json
with open('/tmp/test_phase1/phase1_FDR_265_gpu0.jsonl') as f:
    for line in f:
        record = json.loads(line)
        assert 'all_features' in record
        assert len(record['all_features']) == 265
        assert 'generated_tokens' in record
        print('✓ Output format correct')
"
```

---

## Summary of Required Fixes

**Before running this code, you MUST fix**:

1. **Line 552**: Change to `hidden_states_dict[layer] = full_outputs.hidden_states[layer + 1]`
2. **Lines 532-533, 583-584**: Use token-level slicing for response extraction
3. **Lines 727-739**: Add bet amount parsing and action fields to output

**Strongly recommended**:

4. Use Fisher's exact test for low-count chi-square (lines 189-218)
5. Add memory cleanup in trial loop (line 751)
6. Add checkpoint validation (line 677)

**Optional improvements**:

7. Pre-build layer→features mapping for efficiency
8. Add data validation for missing feature means
9. Document FDR method choice

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Wrong layer features extracted | CRITICAL | Fix hidden states indexing |
| Response parsing includes prompt | CRITICAL | Use token-level slicing |
| Missing behavioral data | HIGH | Add bet parsing |
| GPU memory leak | MEDIUM | Add periodic cleanup |
| Invalid chi-square test | MEDIUM | Use Fisher's exact for low counts |
| Checkpoint incompatibility | LOW | Add FDR cache validation |

---

## Approval Status

❌ **NOT APPROVED FOR PRODUCTION**

This code requires fixes to Critical Issues #1-3 before it can be safely deployed. The current version will produce incorrect results due to layer indexing errors and response parsing bugs.

After fixes are applied, request a follow-up review before running the full experiment.

---

**Reviewer**: Claude Opus 4.5 (Code Review Specialist)
**Date**: 2025-12-08
**Next Review**: After critical fixes are implemented
