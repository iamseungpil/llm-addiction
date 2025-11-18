# Implementation Plan V1: Pathway & Token Analysis from Exp2 Data

## Ultra-Think Analysis

### What We Have
1. **Exp2 Checkpoints** (358 files):
   - Causal features identified (`is_causal` flag)
   - Feature metadata: layer, feature_id, Cohen's d
   - Statistical results: p-values, deltas

2. **Exp2 Response Logs** (199 files, ~1.8M trials):
   - 6 conditions per feature × ~30 trials = ~180 trials/feature
   - Full response text
   - Parsed decisions (action, bet amount, etc.)

### What We Need to Build

**RQ1: Feature Pathways**
- Input: Causal features from checkpoints
- Process: Replay prompts through SAE, record activations at ALL layers
- Output: L9-456 → L17-789 → L26-1069 pathways

**RQ2: Token Associations**
- Input: Response logs (text)
- Process: Compare word frequencies (baseline vs patched)
- Output: L26-1069 → "bet", "amount", "$10" mappings

### Critical Insight from User

> "원래 실험 2의 기록과 feature를 다시 llama sae로 분석해서..."

**This means**:
1. Use EXISTING Exp2 data (no re-running needed!)
2. Re-analyze with LLaMA SAE to extract:
   - Feature activations across layers (for pathways)
   - Token associations (from response text)

### Design Decision

**Option A**: Modify existing `experiment_1_layer_pathway_L1_31` + `experiment_3_feature_word_patching`
- Pro: Reuse existing code
- Con: Messy, two separate experiments

**Option B**: Create unified analysis pipeline
- Pro: Clean, integrated
- Con: More upfront work

**Recommendation**: Start with Option B (clean design)

## Proposed Structure

```
experiment_pathway_token_analysis/
├── config.yaml
├── src/
│   ├── step1_load_causal_features.py      # Load from Exp2 checkpoints
│   ├── step2_pathway_discovery.py         # Gradient + Replay
│   ├── step3_token_association.py         # Word frequency analysis
│   ├── step4_integrate_results.py         # Combine pathways + tokens
│   └── orchestrator.py                    # Main controller
├── data/
│   └── causal_features.json               # Extracted from checkpoints
├── results/
│   ├── pathways/
│   ├── tokens/
│   └── integrated/
└── launch.sh
```

## Implementation Steps

### Step 1: Load Causal Features (NEW)
**Input**: Exp2 checkpoint files
**Process**:
```python
causal_features = []
for checkpoint in checkpoint_files:
    for result in checkpoint['results']:
        if result['is_causal']:
            causal_features.append({
                'feature': result['feature'],  # e.g., "L29-2374"
                'layer': result['layer'],
                'feature_id': result['feature_id'],
                'cohen_d': result['cohen_d']
            })
```
**Output**: `data/causal_features.json`

### Step 2: Pathway Discovery (ADAPTED)
**Input**: Causal features + Exp2 response logs
**Process**:
1. For each causal feature:
   - Load representative prompts from response logs
   - Run through LLaMA with SAEs at ALL layers
   - Record feature activations
2. Compute gradients (backward Jacobian)
3. Validate top-k pathways via replay

**Output**: `results/pathways/feature_pathways.json`

### Step 3: Token Association (ADAPTED)
**Input**: Exp2 response logs
**Process**:
1. For each causal feature:
   - Load responses: baseline vs patched
   - Extract words/tokens
   - Compute frequency differences
   - Statistical testing (chi-square, log-odds)

**Output**: `results/tokens/token_associations.json`

### Step 4: Integration (NEW)
**Process**: Combine pathways + token data
**Output**:
```json
{
  "pathways": [
    {
      "source": "L9-456",
      "target": "L26-1069",
      "gradient": 0.91,
      "validated": true,
      "correlation": 0.87
    }
  ],
  "tokens": {
    "L26-1069": {
      "added_words": ["bet", "amount"],
      "log_odds": [3.9, 2.1]
    }
  },
  "integrated": [
    {
      "pathway": "L9-456 → L17-789 → L26-1069",
      "gradient_strength": [0.91, 0.75],
      "output_tokens": ["bet amount", "slot result"],
      "interpretation": "Early risk signal → Middle amplifier → Betting language"
    }
  ]
}
```

## Key Questions for Codex Review

1. **Data reuse strategy**:
   - Should we extract prompts from Exp2 response logs?
   - Or use fixed representative prompts?

2. **Pathway discovery method**:
   - Pure gradient (fast, existing code)?
   - Gradient + Replay (slower, more accurate)?
   - Or gradient first, then validate top-k via replay?

3. **Token analysis scope**:
   - Word frequency only (fast, existing)?
   - + Token logits (slower, requires re-generation)?

4. **Integration approach**:
   - Simple JSON merging?
   - Statistical correlation between pathway strength and token effects?
   - Visualization-focused?

5. **Causal feature selection**:
   - All causal features (~2,787)?
   - Top-k by |Cohen's d|?
   - Layer-stratified sampling?

## Next Steps

1. Get Codex review on this plan
2. Refine based on feedback
3. Implement step1 (feature loading)
4. Get Codex review on step1 implementation
5. Iterate until complete
