# Pathway Analysis Status Report
**Date**: 2025-11-25 06:15 UTC
**Status**: Ready to Execute

---

## Critical Finding: Data Format Mismatch

### Experiment 2 L1-31 Response Logs
**Location**: `/data/llm_addiction/experiment_2_multilayer_patching/response_logs/`
**Size**: 2.8GB (187 files)
**Format**:
```json
{
  "feature": "L10-11151",
  "condition": "safe_baseline",
  "trial": 0,
  "response": "1) Bet $10...",
  "parsed": {"action": "stop", "bet": 0, "valid": true}
}
```

**❌ MISSING**: `all_features` field (SAE feature activations for all features)

### Pathway Analysis Requirements
**Phase 2-5 dependencies**:
- Phase 2 (Feature correlations): Needs `all_features` to compute correlations
- Phase 3 (Causal validation): Needs `all_features` for regression analysis
- Phase 4 (Word-feature associations): Needs `all_features` for token correlations
- Phase 5 (Prompt-feature correlations): Needs `all_features` for prompt analysis

**❌ BLOCKER**: Cannot run Phases 2-5 on existing Experiment 2 logs without `all_features`

---

## Solution: Run New Pathway Analysis Phase 1

### Why Phase 1 is Needed
Phase 1 patching experiments are designed to:
1. Patch each feature to safe_mean/risky_mean values
2. **Save ALL feature activations** (`all_features`) for each trial
3. Generate proper format for downstream analysis

### What's Ready
✅ **Feature coverage**: 100% (2,510/2,510 reparsed features have means)
✅ **Feature means**: Available in `L1_31_features_CONVERTED_20251111.json`
✅ **Reparsed causal features**: 2,510 features identified
✅ **Phase 1 code**: `/home/ubuntu/llm_addiction/experiment_pathway_token_analysis/src/phase1_patching_multifeature_checkpoint.py`

### Improvements Needed
🔧 **Token ID Saving**: Modify Phase 1 to save generated token IDs for actual BPE token analysis (user requested)

Current Phase 4 uses regex:
```python
tokens = re.findall(r'\$?\d+|\b[a-zA-Z]+\b', response.lower())
```

Should use actual model tokens:
```python
generated_token_ids = full_sequence[prompt_len:].tolist()
generated_tokens = [tokenizer.decode([tid]) for tid in generated_token_ids]
```

---

## Execution Plan

### Phase 1: Patching with Multi-Feature Extraction
**Scope**: 2,510 reparsed causal features
**Conditions**: 4 (safe/risky prompt × safe/risky patch)
**Trials per condition**: 30
**Total trials**: 2,510 × 4 × 30 = 301,200 trials

**Estimated Runtime**: 7-10 hours on 4 GPUs (GPU 4-7)
- ~75-80 trials/hour per GPU
- ~630 features per GPU (2,510 ÷ 4 = 627.5)

**Output Format**:
```json
{
  "target_feature": "L1-29",
  "target_layer": 1,
  "target_feature_id": 29,
  "patch_condition": "safe_mean",
  "patch_value": 0.123,
  "prompt_type": "safe",
  "trial": 0,
  "response": "1) Bet $10",
  "generated_token_ids": [16, 8, 13, 295, 400, 220, 605, 198],
  "generated_tokens": ["1", ")", " Bet", " $", "10", "\n"],
  "all_features": {
    "L1-29": 0.456,
    "L1-182": 0.789,
    ...  // All 2,510 features
  }
}
```

### Phase 2: Feature-Feature Correlations
**Input**: Phase 1 output (JSONL)
**Process**: Compute correlation matrices between features
**Output**: Correlation pairs with significance tests

### Phase 3: Causal Direction Validation
**Input**: Phase 2 correlations + Phase 1 data
**Process**: Regression-based directionality analysis
**Output**: Causal direction classifications (A→B, B→A, bidirectional)

### Phase 4: Token-Feature Associations (IMPROVED)
**Input**: Phase 1 output with token IDs
**Process**: Correlate actual BPE tokens with feature activations
**Output**: Token-feature association matrix with statistical significance

### Phase 5: Prompt-Feature Correlations
**Input**: Phase 1 output
**Process**: Analyze how prompt types affect feature activations
**Output**: Prompt condition effects on features

---

## Resource Requirements

### GPUs
- Phase 1: **4 GPUs (GPU 4-7)** - parallel execution
- Phases 2-5: **CPU only** - statistical analysis

### Storage
- Phase 1 output: ~10-15GB (301,200 trials × ~40KB each)
- Phase 2-5 outputs: ~500MB total

### Time
- Phase 1: 7-10 hours (GPU)
- Phase 2: 1-2 hours (CPU)
- Phase 3: 2-3 hours (CPU)
- Phase 4: 1-2 hours (CPU)
- Phase 5: 1 hour (CPU)
**Total**: ~12-18 hours

---

## Next Actions

### 1. Modify Phase 1 for Token ID Saving
Update `phase1_patching_multifeature_checkpoint.py`:
- Add `generated_token_ids` field
- Add `generated_tokens` field
- Ensure compatibility with existing checkpoint system

### 2. Launch Phase 1 on GPUs 4-7
```bash
cd /home/ubuntu/llm_addiction/experiment_pathway_token_analysis
bash launch_phase1_full_gpu4567_checkpoint.sh
```

### 3. Monitor Progress
Check checkpoints saved every 50 trials per GPU

### 4. Execute Phases 2-5 Sequentially
After Phase 1 completes, run Phases 2-5 in sequence

---

## Comparison with Experiment 2 L1-31

| Aspect | Experiment 2 L1-31 | Pathway Phase 1 |
|--------|-------------------|-----------------|
| Features tested | 13,434 | 2,510 (reparsed causal) |
| Trials per feature | 50 × 3 conditions = 150 | 30 × 4 conditions = 120 |
| Saves `all_features`? | ❌ NO | ✅ YES |
| Saves token IDs? | ❌ NO | ✅ YES (new) |
| Purpose | Test individual features | Multi-feature analysis |
| Downstream analysis | Limited (bet amounts only) | Full (Phases 2-5) |

---

## Recommendation

**Proceed with Phase 1-5 pathway analysis** as a NEW experiment (not direct analysis of existing Experiment 2 logs) because:

1. ✅ We have 100% feature coverage (means exist)
2. ✅ We have 2,510 reparsed causal features ready
3. ✅ Pathway analysis needs `all_features` which Experiment 2 doesn't have
4. ✅ User requested token-based improvements (requires saving token IDs)
5. ✅ This enables complete Phase 2-5 analysis pipeline

**Timeline**: Start Phase 1 tonight → complete by tomorrow morning → run Phases 2-5 → full results in 1-2 days

---

**Status**: ✅ **READY TO EXECUTE**
**Blocker**: None (all prerequisites met)
**Action Required**: User approval to launch Phase 1
