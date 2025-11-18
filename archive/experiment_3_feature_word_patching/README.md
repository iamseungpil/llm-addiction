# Experiment 3: Causal Word Patching Analysis

## Overview

**Purpose**: Discover which words each feature CAUSALLY controls by comparing responses before and after feature patching.

**Key Insight**: We don't need activation extraction! Exp2 patching data already has all 6 conditions (baseline + patches), so we can directly compare word frequencies.

**Method**: Before/After Patching Comparison (True Causal Analysis)

## Why This Method is Superior

### ❌ Original Plan (Correlation-based)
- Extract activations → Split high/low → Compare word frequencies
- **Problem**: Correlation, not causation
- **Time**: 5-10 hours for activation extraction
- **Coverage**: Required complex activation caching

### ✅ New Method (Causal Patching)
- **Direct comparison**: baseline vs patched responses
- **True causation**: Feature was experimentally manipulated!
- **Time**: ~30 minutes (no model loading needed)
- **Coverage**: 100% of 2,787 causal features

## Data

### Input: Exp2 Response Logs
```
/data/llm_addiction/experiment_2_multilayer_patching/response_logs/
```

**Coverage**:
- 2,787/2,787 causal features (100%)
- All 30 layers (L1-L30) + bonus L31
- ~1.8M trials total
- 6 conditions per feature:
  - safe_baseline
  - safe_with_safe_patch
  - safe_with_risky_patch
  - risky_baseline
  - risky_with_safe_patch
  - risky_with_risky_patch

### Output
```
/home/ubuntu/llm_addiction/experiment_3_feature_word_patching/results/
└── causal_word_effects_{timestamp}.json
```

## Method

### Four Causal Comparisons per Feature

#### 1. Risky Feature Effect (What words does it ADD?)
```python
baseline = safe_baseline_responses
patched = safe_with_risky_patch_responses

added_words = words_in(patched) - words_in(baseline)
# These are words the RISKY FEATURE causally adds!
```

**Example Finding**:
- Risky feature adds: "bet amount", "$5", "slot result"
- Concrete betting language

#### 2. Safe Feature Effect (What words does it REMOVE?)
```python
baseline = risky_baseline_responses
patched = risky_with_safe_patch_responses

removed_words = words_in(baseline) - words_in(patched)
# These are words the SAFE FEATURE causally removes!
```

**Example Finding**:
- Safe feature removes: "bet amount", "slot result"
- Suppresses risky language

#### 3. Feature Amplification
```python
# Safe prompt + safe feature
safe_baseline vs safe_with_safe_patch
→ Words amplified by safe feature in safe context
```

#### 4. Feature Suppression
```python
# Risky prompt + risky feature
risky_baseline vs risky_with_risky_patch
→ Words amplified by risky feature in risky context
```

### Statistical Testing

**Effect Size**: Log-odds ratio
```python
log_odds = log((patched_freq + 0.001) / (baseline_freq + 0.001))
```

**Significance**: Chi-square test
```python
chi2, p_value = chi2_contingency([[baseline_count, ...], [patched_count, ...]])
```

**Threshold**:
- |log_odds| > 0.5
- p < 0.05

## Usage

### Run Analysis

```bash
cd /home/ubuntu/llm_addiction/experiment_3_feature_word_patching
./launch_causal_word_analysis.sh
```

**Monitor**:
```bash
tmux attach -t causal_word_analysis
```

**Expected completion**: ~30 minutes

### Output Structure

```json
{
  "feature": "L26-1069",
  "risky_feature_adds": [
    {
      "word": "amount",
      "baseline_freq": 0.001,
      "patched_freq": 0.05,
      "log_odds_ratio": 3.9,
      "p_value": 0.001,
      "baseline_count": 2,
      "patched_count": 150
    },
    ...
  ],
  "safe_feature_removes": [...],
  "safe_amplification_adds": [...],
  "risky_amplification_adds": [...]
}
```

## Key Research Questions

### 1. What language do risky features control?
- **Hypothesis**: Concrete betting language ("amount", "bet", specific numbers)
- **Test**: risky_feature_adds analysis

### 2. What language do safe features suppress?
- **Hypothesis**: Remove gambling-specific words, add cautious language
- **Test**: safe_feature_removes analysis

### 3. Are effects consistent across layers?
- **Early layers (L1-L10)**: Basic word associations?
- **Middle layers (L11-L20)**: Semantic combinations?
- **Late layers (L21-L30)**: Decision-level language?

### 4. Do features show interpretable patterns?
- **Risky features**: "bet", "win", "gamble", "$X" (amounts)
- **Safe features**: "stop", "careful", "think", "enough"

## Advantages Over Original Design

| Aspect | Original (High/Low Split) | New (Causal Patching) |
|--------|--------------------------|---------------------|
| **Causation** | ❌ Correlation only | ✅ True experimental manipulation |
| **Time** | 5-10 hours | ~30 minutes |
| **Complexity** | High (activation extraction, caching, splitting) | Low (direct comparison) |
| **Interpretation** | "Words associated with high activation" | "Words feature CAUSES to appear/disappear" |
| **Statistical Power** | Medium (observational split) | High (randomized patching) |
| **Coverage** | 100% (after extraction) | 100% (immediate) |

## Comparison with Experiment 1

### Experiment 1: Gradient Pathways
- **What**: Which features connect to which layers?
- **Method**: Backward Jacobian (gradient-based)
- **Output**: Feature → Feature connections
- **Interpretation**: "L9-456 contributes to L31-10692"

### Experiment 3: Causal Words
- **What**: Which words does each feature control?
- **Method**: Before/After patching comparison
- **Output**: Feature → Word associations
- **Interpretation**: "L26-1069 adds 'bet amount' language"

### Combined Insight
```
Gradient Pathways: L9-456 → L17-789 → L26-1069
Causal Words: L26-1069 adds "bet amount", "slot result"

→ We can trace: Early feature → Middle amplifier → Final betting language!
```

## Implementation Details

### Word Extraction
```python
def extract_words(text):
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    # Remove stopwords (the, a, an, etc.)
    # Keep words > 2 characters
    return filtered_words
```

### Frequency Comparison
```python
baseline_freq = baseline_count / total_baseline_words
patched_freq = patched_count / total_patched_words

log_odds = log((patched_freq + 0.001) / (baseline_freq + 0.001))
```

### Statistical Testing
```python
contingency = [[baseline_count, total_baseline - baseline_count],
               [patched_count, total_patched - patched_count]]
chi2, p_value = chi2_contingency(contingency)
```

## Expected Findings

### Safe Features (L25-L29 dominant)
**Words REMOVED** when patched onto risky prompt:
- "amount", "bet", "$X" (specific amounts)
- Gambling-specific vocabulary
- Risk-taking language

**Words ADDED**:
- "stop", "enough", "careful"
- Cautious decision language

### Risky Features (L9-L17 dominant)
**Words ADDED** when patched onto safe prompt:
- "bet amount", "slot result"
- Concrete betting mechanics
- Specific dollar amounts

**Words REMOVED**:
- Conservative language
- Cautious phrasing

### Middle-Layer Risky Features (L9-L17)
**Hypothesis**: Act as semantic amplifiers
- Take early signals
- Add gambling-specific vocabulary
- Bridge to decision language in late layers

## Files

```
experiment_3_feature_word_patching/
├── src/
│   ├── causal_word_patching_analyzer.py  # Main analyzer (NEW!)
│   ├── step1_extract_activations.py      # OBSOLETE (delete)
│   └── feature_word_analyzer.py          # OBSOLETE (delete)
├── launch_causal_word_analysis.sh        # Launch script (NEW!)
├── launch_extract_activations.sh         # OBSOLETE (delete)
├── results/                               # Output directory
├── logs/                                  # Execution logs
└── README.md                              # This file
```

## Next Steps

1. ✅ Run causal word analysis (~30 min)
2. ⏳ Analyze layer-specific patterns
3. ⏳ Combine with gradient pathway results
4. ⏳ Identify interpretable feature→word mappings
5. ⏳ Write paper section: "Causal Feature-Word Associations"

## Related Work

- **Anthropic Attribution Graphs (2025)**: Feature-centric gradient tracking
- **Intervention-based Discovery**: Causal manipulation > correlation
- **SAE Interpretability**: Connecting features to human-understandable concepts

---

**Created**: 2025-10-22
**Method**: Causal before/after patching comparison
**Coverage**: 2,787 features (100%)
**Data**: Experiment 2 response logs (~1.8M trials)
**Runtime**: ~30 minutes (no GPU needed!)
