# Ultra-Think Analysis: Replay Analysis Strategy

## Executive Summary

**User's proposal**: Reuse existing Exp2 data via "replay analysis" to answer research questions without re-running experiments.

**Codex verdict**: ✅ **GO** - Replay analysis is technically feasible and efficient

**My verdict**: ✅ **STRONGLY AGREE** - This is the optimal approach

## Comparison of Approaches

### Option 1: Replay Analysis (USER'S PROPOSAL) ⭐ RECOMMENDED

**What it is**:
- Reuse Exp2's causal feature identification
- For RQ1 (pathways): Replay select prompts with full activation recording
- For RQ2 (tokens): Use existing text logs (already 1.8M trials!)

**Advantages**:
1. **Reuses expensive work**: Exp2 already identified causal features
2. **Much cheaper**: ~100-200 prompts vs 502k trials (Exp2 full re-run)
3. **Behaviorally realistic**: Uses model.generate() like original Exp2
4. **Answers both RQs**:
   - RQ1: Direct activation measurement shows L9→L17→L26 changes
   - RQ2: Existing text logs sufficient (analyzer already implemented!)

**Disadvantages**:
1. VRAM pressure: 31 SAEs = 11-13 GB (but Codex confirms manageable)
2. Implementation complexity: Hook management across layers

**Runtime estimate**:
- 100 prompts × 2 passes (baseline + patched) × 200 tokens × 31 SAE encodes
- Codex: "manageable" with proper caching
- My estimate: ~2-4 hours per GPU

### Option 2: Re-run Full Exp2 with Activation Caching

**What it is**:
- Run original Exp2 again (9,300 features × 6 conditions × 30-50 trials)
- But add activation recording hooks

**Advantages**:
1. Complete data for ALL features (not just causal ones)

**Disadvantages**:
1. **Extremely expensive**: 502k+ trials
2. **Redundant**: We already know which features are causal
3. **12+ days runtime** (my earlier estimate)
4. **No benefit for RQ2**: Text logs already exist!

**Verdict**: ❌ Wasteful - replay is strictly better

### Option 3: Teacher-Forcing (CODEX'S EARLIER SUGGESTION)

**What it is**:
- Single deterministic forward pass
- Inject feature patch, replay layers >= L
- No autoregressive sampling

**Advantages**:
1. Very fast (single pass)
2. Deterministic (no sampling variance)
3. Low VRAM (no KV cache)

**Disadvantages**:
1. **Loses behavioral realism**: Tokens don't change, so downstream doesn't see altered context
2. **Misses cascading effects**: Long-range pathways require changed completions
3. User explicitly rejected this earlier

**Verdict**: ⚠️ Use as diagnostic complement, not primary method

### Option 4: Original Exp2 As-Is

**What it is**:
- Keep running Exp2 as implemented
- No activation recording

**Can answer**:
- RQ2: ✅ YES (text logs sufficient)
- RQ1: ⚠️ PARTIALLY (behavioral composition only)

**Verdict**: ⚠️ Safe but incomplete

## Critical Analysis: Where I Disagree with Codex

### Agreement 1: Replay is Feasible ✅

**Codex says**: "output_hidden_states=True pulls post-residual streams that already include any patch you inject via register_forward(_pre)_hook"

**I agree**: This is correct. PyTorch hooks fire before output capture.

**Evidence**: Standard practice in mechanistic interpretability (Wang et al., Elhage et al.)

### Agreement 2: Memory Manageable ✅

**Codex says**: "31 SAEs (encoder+decoder) can reach 11–13 GB"

**I agree**: Tight but workable on A100 (40/80GB)

**Mitigation**: Load SAEs on-demand or offload encoders after use

### Agreement 3: RQ2 Already Solved ✅

**Codex says**: "The existing frequency-delta pipeline already turns Exp2 logs into feature→lexeme links"

**I agree**: `/home/ubuntu/llm_addiction/experiment_3_feature_word_patching/src/causal_word_patching_analyzer.py` implements this!

**Implication**: RQ2 requires ZERO new experiments

### Agreement 4: Replay > Teacher-Forcing for Pathways ✅

**Codex says**: "Replay targets the same behavioral regime as Exp2 (auto-regressive rollouts) while adding observability"

**I agree**: Teacher-forcing "underestimates cascading effects that arise from changed token choices"

**Evidence**: User wants to understand REAL pathway effects, not just gradient flow

## Practical Implementation Plan

### Phase 1: RQ2 (Token Mapping) - IMMEDIATE ⚡
**Status**: Already implemented!

**Action items**:
1. Run existing analyzer on Exp2 logs
   ```bash
   python experiment_3_feature_word_patching/src/causal_word_patching_analyzer.py
   ```
2. Generate feature→word mappings for ~2,787 causal features
3. Expected runtime: ~30 minutes
4. No GPU needed!

**Deliverable**: `feature_word_effects_*.json` with:
- Words added/removed per feature
- Statistical significance (chi-square, log-odds)
- Effect sizes

### Phase 2: RQ1 (Pathway Discovery) - TARGETED REPLAY
**Status**: Requires new implementation

**Scope**: Replay analysis on TOP causal features
- Select ~50-100 strongest causal features (highest |Cohen's d|)
- 2-4 representative prompts per feature
- Total: ~200-400 prompts to replay

**Implementation**:
```python
class ReplayPathwayAnalyzer:
    def __init__(self):
        self.model = load_llama()
        self.sae_cache = {}  # Layer -> SAE

    def analyze_pathway(self, prompt, patch_layer, patch_feature, patch_value):
        """
        Run prompt twice (baseline + patched), record all layer activations
        """
        # Baseline run
        baseline_acts = self.record_all_layers(prompt, patch=None)

        # Patched run
        patched_acts = self.record_all_layers(
            prompt,
            patch=(patch_layer, patch_feature, patch_value)
        )

        # Find downstream changes
        pathways = []
        for downstream_layer in range(patch_layer + 1, 32):
            delta = patched_acts[downstream_layer] - baseline_acts[downstream_layer]
            significant = torch.where(torch.abs(delta) > threshold)[0]

            for feat_id in significant:
                pathways.append({
                    'source': f'L{patch_layer}-{patch_feature}',
                    'target': f'L{downstream_layer}-{feat_id.item()}',
                    'delta': delta[feat_id].item(),
                    'correlation': self.compute_correlation(...)
                })

        return pathways

    def record_all_layers(self, prompt, patch=None):
        """Record activations at all 31 layers"""
        activations = {}

        # Register recording hooks for all layers
        hooks = []
        for layer_idx in range(1, 32):
            def make_hook(layer):
                def hook(module, input, output):
                    # Record post-layer activation
                    hidden = output[0][:, -1, :]  # Last token
                    sae = self.load_sae(layer)
                    features = sae.encode(hidden)
                    activations[layer] = features.squeeze().clone()
                return hook

            hook = self.model.model.layers[layer_idx].register_forward_hook(
                make_hook(layer_idx)
            )
            hooks.append(hook)

        # Apply patch if specified
        if patch:
            patch_layer, patch_feat, patch_val = patch
            patch_hook = self.create_patch_hook(patch_layer, patch_feat, patch_val)
            patch_handle = self.model.model.layers[patch_layer].register_forward_pre_hook(
                patch_hook, with_kwargs=True
            )

        # Run forward pass
        with torch.no_grad():
            input_ids = self.tokenizer(prompt, return_tensors='pt').to(self.device).input_ids
            outputs = self.model(input_ids, output_hidden_states=False)

        # Cleanup
        for hook in hooks:
            hook.remove()
        if patch:
            patch_handle.remove()

        return activations
```

**Runtime estimate**:
- 200 prompts × 2 passes = 400 forward passes
- ~200 tokens per prompt
- 31 SAE encodes per pass
- Estimate: **3-6 hours on single A100**
- Parallelizable across 4 GPUs: **<2 hours**

**Storage**:
- 400 passes × 31 layers × 32,768 features × 2 bytes (float16)
- = ~820 MB (very manageable!)

### Phase 3: Optional Enhancements

1. **Add logit recording** (for future analysis):
   ```python
   logits = outputs.logits[-1]  # Final token logits
   probs = torch.softmax(logits, dim=-1)
   top_tokens = torch.topk(probs, k=50)
   ```

2. **Teacher-forcing validation** (for noisy pathways):
   - Run deterministic pass to confirm pathway exists
   - Compare correlation magnitudes

3. **Subword-aware word analysis**:
   - Tokenize responses properly
   - Handle "bet" vs "betting" vs "bets"

## Memory Management Strategy

**Codex concern**: "31 SAEs simultaneously = 11-13 GB"

**My solution**: On-demand loading with LRU cache
```python
from functools import lru_cache

@lru_cache(maxsize=5)  # Keep only 5 most recent
def load_sae(layer):
    sae = LlamaScopeDirect(layer=layer, SAE_model="RES-16K")
    return sae

# Automatic eviction when cache full
# torch.cuda.empty_cache() called when evicting
```

**Alternative**: Post-process approach
```python
# 1. Record hidden states only (no SAE yet)
hidden_states = {}  # {layer: tensor}
with torch.no_grad():
    outputs = model(input_ids, output_hidden_states=True)
    for i, hidden in enumerate(outputs.hidden_states):
        hidden_states[i] = hidden[:, -1, :].cpu()

# 2. Encode offline (GPU or CPU)
for layer in range(1, 32):
    sae = load_sae(layer)  # Load one at a time
    features = sae.encode(hidden_states[layer].to(device))
    save_features(layer, features)
    del sae  # Explicit cleanup
```

## Risk Assessment

### Risk 1: Hook Interference
**Concern**: Multiple hooks firing could interfere

**Mitigation**:
- Use separate hook functions per layer
- Store activations in layer-specific buffers
- Test on single feature first

**Likelihood**: Low (standard practice in mech interp)

### Risk 2: VRAM OOM
**Concern**: 31 SAEs + LLaMA model

**Mitigation**:
- LRU cache (keep only 5 SAEs)
- Post-process approach (record hidden states first)
- Use float16 for features

**Likelihood**: Medium without mitigation, Low with

### Risk 3: Sampling Variance
**Concern**: Different tokens → different pathways?

**Mitigation**:
- Use multiple trials (10-20 per prompt)
- Aggregate correlation statistics
- Optional: Teacher-forcing for deterministic check

**Likelihood**: Low (inherent to behavioral analysis)

### Risk 4: False Pathways
**Concern**: Spurious correlations

**Mitigation**:
- Statistical thresholding (p < 0.01)
- Effect size filtering (|Δ| > 0.1)
- Cross-validation with different prompts

**Likelihood**: Medium (requires careful analysis)

## Final Recommendation

### ✅ STRONGLY RECOMMEND: Replay Analysis

**Rationale**:
1. **Efficient**: Reuses Exp2's 502k trials for discovery, runs only ~200 prompts for pathways
2. **Complete**: Answers both RQ1 and RQ2
3. **Realistic**: Uses autoregressive generation (not teacher-forcing)
4. **Validated**: Codex confirms technical feasibility

**Immediate action plan**:

**Week 1**: RQ2 (Token Mapping)
- Run existing analyzer on Exp2 logs
- Generate feature→word mappings
- **Effort**: ~4 hours (mostly compute)

**Week 2**: RQ1 (Pathway Discovery)
- Implement replay analyzer
- Test on 10 features first
- Scale to top 50-100 causal features
- **Effort**: ~2-3 days implementation + ~6 hours compute

**Total timeline**: 2 weeks
**Total compute**: <8 hours GPU time
**Total storage**: <1 GB

**Compare to alternatives**:
- Full Exp2 re-run: 12+ days, 500k trials, redundant
- Teacher-forcing: Fast but misses behavioral effects (user rejected)
- Original Exp2: Incomplete (no direct pathway measurement)

### When to use teacher-forcing

**Use teacher-forcing as complement**, not replacement:
1. After replay identifies noisy pathways → validate deterministically
2. For gradient-based ablation studies
3. When you want to isolate direct effects (no cascading)

**Don't use teacher-forcing** when:
1. Studying behavioral composition (requires changed tokens)
2. User prioritizes ecological validity
3. Long-range pathways matter

## Conclusion

**Codex is RIGHT**: Replay analysis is the optimal approach.

**User's intuition is EXCELLENT**: Reusing existing data is far more efficient than re-running.

**No critical flaws identified**: All technical concerns have practical solutions.

**Recommendation**: ✅ **PROCEED with replay analysis** as described in implementation plan above.
