# Critical Fixes Required for Phase 1 FDR Script

## Overview

The Phase 1 script has **3 CRITICAL BUGS** that will cause incorrect results. This document provides exact code fixes.

---

## Fix 1: Hidden States Indexing (CRITICAL)

### Location
Lines 550-552 and 600-601 in `phase1_FDR_265features.py`

### Problem
```python
# WRONG - extracts layer L-1 instead of layer L
hidden_states_dict[layer] = full_outputs.hidden_states[layer]
```

### Root Cause
`full_outputs.hidden_states` is a tuple where:
- `hidden_states[0]` = embedding layer output (before transformer blocks)
- `hidden_states[1]` = output of layer 0 (block 0)
- `hidden_states[L+1]` = output of layer L (block L)

SAE for layer L was trained on the output AFTER block L, which is at index L+1.

### Fix
Replace line 552 with:
```python
hidden_states_dict[layer] = full_outputs.hidden_states[layer + 1]
```

Replace line 601 with:
```python
hidden_states_dict[layer] = full_outputs.hidden_states[layer + 1]
```

### Verification
Add this debug code to verify:
```python
# After line 552, add:
LOGGER.debug(f"Layer {layer}: extracted from hidden_states[{layer + 1}]")
```

---

## Fix 2: Response Text Extraction (CRITICAL)

### Location
Lines 532-533 and 583-584 in `phase1_FDR_265features.py`

### Problem
```python
# WRONG - character-level slicing doesn't align with tokens
response = self.tokenizer.decode(full_sequence, skip_special_tokens=True)
response = response[len(prompt):].strip()
```

### Root Cause
After tokenization → generation → detokenization, the character count may change due to:
- Whitespace normalization (spaces → newlines)
- Special character handling
- Tokenizer artifacts

This causes the response to either:
1. Include partial prompt text (if decoded prompt is shorter)
2. Cut off actual response (if decoded prompt is longer)

### Fix

**For `generate_with_patching` (lines 530-537)**:
```python
# REPLACE lines 531-533 with:
with torch.no_grad():
    outputs = self.model.generate(
        **inputs,
        max_new_tokens=self.config.max_new_tokens,
        do_sample=self.config.do_sample,
        return_dict_in_generate=True
    )

    # Extract generated tokens (correct way)
    full_sequence = outputs.sequences[0]
    generated_token_ids = full_sequence[prompt_len:].tolist()
    response = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)

    # Extract token strings for Phase 4
    generated_tokens = [self.tokenizer.decode([tid]) for tid in generated_token_ids]

    # REMOVE line 536-537 (now redundant)
```

**For `generate_without_patching` (lines 582-588)**:
```python
# REPLACE lines 582-588 with:
with torch.no_grad():
    outputs = self.model.generate(
        **inputs,
        max_new_tokens=self.config.max_new_tokens,
        do_sample=self.config.do_sample,
        return_dict_in_generate=True
    )

    full_sequence = outputs.sequences[0]
    generated_token_ids = full_sequence[prompt_len:].tolist()
    response = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    generated_tokens = [self.tokenizer.decode([tid]) for tid in generated_token_ids]

    # REMOVE lines 586-588 (now redundant)
```

---

## Fix 3: Add Bet Amount Parsing (CRITICAL)

### Location
After line 725 in `phase1_FDR_265features.py`

### Problem
Output records don't include `action`, `bet_amount`, or `is_stop` fields that downstream analysis expects.

### Fix

**Step 1**: Add parsing helper methods to the class (after line 328):
```python
def parse_is_stop(self, response: str) -> bool:
    """Parse if response is 'stop' decision"""
    response_lower = response.lower().strip()

    # Check for stop indicators
    stop_indicators = ['stop', '2)', 'choice: 2', 'option 2']
    for indicator in stop_indicators:
        if indicator in response_lower[:50]:  # Check first 50 chars
            return True

    return False

def parse_bet_amount(self, response: str) -> int:
    """Parse bet amount from response"""
    import re

    # Look for bet amount in first 100 chars
    text = response[:100]

    # Pattern 1: "Bet $10" or "bet $25"
    match = re.search(r'bet\s+\$(\d+)', text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # Pattern 2: "$10" or "$25" (fallback)
    match = re.search(r'\$(\d+)', text)
    if match:
        amount = int(match.group(1))
        if 5 <= amount <= 100:  # Valid bet range
            return amount

    # Pattern 3: "Choice: 1" with context
    if 'choice: 1' in text.lower() or '1)' in text[:20]:
        # Check prompt type to determine fixed vs variable
        if 'Bet $10' in prompt:  # Fixed betting
            return 10
        else:  # Variable betting - default to $10 if unclear
            return 10

    return 0  # Failed to parse
```

**Step 2**: Update record creation (replace lines 727-739):
```python
try:
    response, all_activations, token_ids, tokens = self.generate_with_patching(
        prompt, target_layer, target_feature_id, patch_value
    )

    # Parse decision
    is_stop = self.parse_is_stop(response)
    bet_amount = 0 if is_stop else self.parse_bet_amount(response)
    is_valid = True

    # Validate parsing
    if not is_stop and bet_amount == 0:
        LOGGER.warning(f"Failed to parse bet amount: {response[:100]}")
        is_valid = False

    record = {
        'target_feature': feature_name,
        'target_layer': target_layer,
        'target_feature_id': target_feature_id,
        'patch_condition': patch_cond,
        'patch_value': patch_value,
        'prompt_type': prompt_type,
        'trial': trial,
        'response': response,
        'action': 'stop' if is_stop else 'bet',
        'bet_amount': bet_amount,
        'is_stop': is_stop,
        'is_valid': is_valid,
        'generated_token_ids': token_ids,
        'generated_tokens': tokens,
        'all_features': all_activations
    }

    f.write(json.dumps(record, ensure_ascii=False) + '\n')
    f.flush()

    completed_trials.add(trial_key)

except Exception as e:
    import traceback
    LOGGER.error(f"Error on {feature_name} trial {trial}: {e}")
    LOGGER.error(traceback.format_exc())
```

---

## Additional Recommended Fixes

### Fix 4: Add Memory Cleanup (Recommended)

After line 751, add:
```python
                        pbar.update(1)

                        # Cleanup every 10 trials
                        if trial % 10 == 9:
                            torch.cuda.empty_cache()
                            gc.collect()

            pbar.close()
```

### Fix 5: Improve Chi-Square Tests (Recommended)

Replace lines 189-218 with:
```python
# Chi-square test for safe prompt
safe_baseline = conds.get(('baseline', 'safe'), [])
safe_patched = conds.get(('safe_patch', 'safe'), [])
p_safe = 1.0
if safe_baseline and safe_patched:
    baseline_stop = sum(1 for t in safe_baseline if t.get('is_stop', False))
    baseline_cont = len(safe_baseline) - baseline_stop
    patched_stop = sum(1 for t in safe_patched if t.get('is_stop', False))
    patched_cont = len(safe_patched) - patched_stop

    if (baseline_stop + patched_stop) > 0 and (baseline_cont + patched_cont) > 0:
        # Check if chi-square assumptions are met
        expected_min = min(
            (baseline_stop + patched_stop) * len(safe_baseline) / (len(safe_baseline) + len(safe_patched)),
            (baseline_cont + patched_cont) * len(safe_baseline) / (len(safe_baseline) + len(safe_patched))
        )

        if expected_min >= 5:
            # Use chi-square
            contingency = [[baseline_stop, baseline_cont], [patched_stop, patched_cont]]
            try:
                _, p_safe, _, _ = stats.chi2_contingency(contingency)
            except:
                p_safe = 1.0
        else:
            # Use Fisher's exact test for low counts
            from scipy.stats import fisher_exact
            try:
                _, p_safe = fisher_exact([[baseline_stop, baseline_cont],
                                          [patched_stop, patched_cont]])
            except:
                p_safe = 1.0

# Same for risky prompt (repeat logic for risky_baseline, risky_patched)
```

---

## Testing After Fixes

### Quick Test (1 feature, 2 trials)
```bash
cd /home/ubuntu/llm_addiction/experiment_pathway_token_analysis

CUDA_VISIBLE_DEVICES=0 python src/phase1_FDR_265features.py \
    --gpu-id 0 \
    --n-trials 2 \
    --limit 1 \
    --output-dir /tmp/test_phase1_fixed \
    --patching-dir /data/llm_addiction/patching_265_FDR_20251208
```

### Verify Output Format
```python
import json

# Check output
with open('/tmp/test_phase1_fixed/phase1_FDR_265_gpu0.jsonl') as f:
    for i, line in enumerate(f, 1):
        record = json.loads(line)

        # Verify required fields
        assert 'all_features' in record, f"Line {i}: Missing all_features"
        assert 'generated_tokens' in record, f"Line {i}: Missing generated_tokens"
        assert 'action' in record, f"Line {i}: Missing action"
        assert 'bet_amount' in record, f"Line {i}: Missing bet_amount"
        assert 'is_stop' in record, f"Line {i}: Missing is_stop"

        # Verify feature count
        assert len(record['all_features']) == 265, f"Line {i}: Expected 265 features, got {len(record['all_features'])}"

        # Verify response doesn't include prompt
        assert 'Slot Machine Game' not in record['response'], f"Line {i}: Response includes prompt text"

        print(f"✓ Line {i}: All checks passed")
        print(f"  Action: {record['action']}")
        print(f"  Bet: ${record['bet_amount']}")
        print(f"  Features extracted: {len(record['all_features'])}")
        print(f"  Tokens: {len(record['generated_tokens'])}")
```

### Verify Layer Extraction
```python
# Check that features are reasonable
record = json.loads(open('/tmp/test_phase1_fixed/phase1_FDR_265_gpu0.jsonl').readline())

# Features should have non-zero activations for multiple layers
non_zero_count = sum(1 for v in record['all_features'].values() if v > 0.01)
print(f"Non-zero features: {non_zero_count}/265")
assert non_zero_count > 10, "Too few active features - possible indexing error"
```

---

## Summary

**Required fixes before deployment**:
1. ✅ Fix hidden states indexing (add +1)
2. ✅ Fix response extraction (use token slicing)
3. ✅ Add bet amount parsing

**Recommended fixes**:
4. Add memory cleanup
5. Improve chi-square tests

**Estimated time to fix**: 30-45 minutes

**Risk if not fixed**:
- Wrong layer activations → Invalid experiment
- Malformed responses → Parsing failures
- Missing data fields → Analysis failures

---

**Apply these fixes, run the test, then deploy to full GPUs.**
