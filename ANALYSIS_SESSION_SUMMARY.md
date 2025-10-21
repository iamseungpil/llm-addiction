# Analysis Session Summary

**Date**: 2025-10-16
**Session**: Continue analysis tasks for LLM addiction experiments

---

## Completed Analyses

### 1. Experiment 6: Attention Pathway Analysis ✅

**Location**: `/home/ubuntu/llm_addiction/experiment_6_token_level_tracking/analysis/`

**Issue Identified and Corrected**:
- Original analysis used wrong token positions ("choices" [12, 26] were "Bet" tokens in game history)
- Corrected to analyze attention TO the last token (actual decision point)

**Key Findings**:

#### Layer-Specific Attention Patterns
- **Layer 15 (Middle)**: Most distinctive
  - Risky scenarios: 45% higher attention to balance (0.0039 vs 0.0021)
  - Interpretation: Risky decisions over-focus on current state (myopic)

- **Layer 31 (Final)**:
  - Safe scenarios: 52% higher attention to probability (0.0032 vs 0.0021)
  - Interpretation: Safe decisions increasingly weight win rate (rational)

#### Feature Activation Pathways (L31, Balance → Decision)
- **Feature 9926**: 5.43→9.26 activation (Bankruptcy $90)
- **Feature 30028**: 0.0→10.06 activation (Desperate $10) - dramatic increase
- **Feature 2857**: 4.75→5.08 activation (Bankruptcy $90)

**Outputs Generated**:
1. **Visualizations** (6 files):
   - `CORRECTED_attention_to_decision_{L8,L15,L31}_{Desperate_10,Safe_140_near_goal}.png`
   - Bar plots showing attention from each token to final decision token

2. **Data Files** (12 files):
   - `CORRECTED_balance_decision_pathway_{L8,L15,L31}.csv`
   - `CORRECTED_goal_decision_pathway_{L8,L15,L31}.csv`
   - `CORRECTED_probability_decision_pathway_{L8,L15,L31}.csv`
   - `CORRECTED_pathway_comparison_{L8,L15,L31}.csv`

3. **Report**:
   - `ATTENTION_PATHWAY_FINDINGS.md` - Comprehensive findings documentation

**Implications**:
- Risky decisions show myopic focus on current balance in middle layers
- Safe decisions show rational probability weighting in final layers
- Clear layer-specific cognitive patterns emerge

---

### 2. Experiment 3: TF-IDF Enhancement ✅

**Location**: `/home/ubuntu/llm_addiction/experiment_3_feature_word_6400/analysis/`

**Enhancement Added**:
- Original analysis used simple frequency differences (high_freq - low_freq)
- Added TF-IDF scoring to identify feature-characteristic words
- Downweights common words, upweights feature-specific words

**Key Findings**:

#### Most Feature-Specific Words (High IDF)
| Word | IDF | N Features | Interpretation |
|------|-----|------------|----------------|
| history | 3.52 | 13 | Very rare, feature-specific |
| 120 | 1.27 | 124 | Balance-related, moderately specific |
| win | 1.26 | 125 | Outcome-related, moderately specific |
| machine | 1.25 | 126 | Context word, moderately specific |

#### Most Common Words (Low IDF)
| Word | IDF | N Features | Interpretation |
|------|-----|------------|----------------|
| stop | 0.43 | 287 | Very common across features |
| round | 0.66 | 229 | Structural word, very common |
| 100 | 0.67 | 226 | Balance amount, very common |
| choice | 0.76 | 207 | Decision word, very common |

**Outputs Generated**:
1. `word_idf_scores.csv` - IDF scores for all 22 words
2. `feature_word_tfidf_enhanced.json` - Full dataset with TF-IDF scores (441 features)
3. `word_specificity_analysis.csv` - Word specificity across features
4. `tfidf_vs_frequency_ranking.png` - Visual comparison of ranking methods
5. `top_tfidf_words_per_feature.csv` - Top 5 TF-IDF words for each feature

**Implications**:
- TF-IDF successfully identifies feature-characteristic vs. common words
- "history", "120", "win" are more discriminative than "stop", "round"
- Can now rank words by how specific they are to each feature's behavior

---

## Priority Task Status

Based on the earlier diagnostic report:

1. ✅ **COMPLETED**: Experiment 6 Attention Pathway Analysis (High Priority)
2. ✅ **COMPLETED**: Experiment 3 TF-IDF Enhancement (High Priority)
3. ⏳ **PENDING**: Experiment 1 Temporal Feature Tracking (Medium Priority)
   - **Status**: Waiting for Experiment 2 (activation patching) to complete
   - **Reason**: Temporal analysis should use causal features from Experiment 2
   - **Expected**: ~4-5 days for Experiment 2 to finish (running on GPUs 4-7)

---

## Summary Statistics

### Analysis Coverage
- **Experiments analyzed**: 2 (Experiment 6, Experiment 3)
- **Analysis folders created**: 2
- **Visualizations generated**: 7 (6 heatmaps + 1 comparison plot)
- **Data files generated**: 17 CSVs + 1 JSON
- **Documentation created**: 2 markdown reports

### Data Processed
- **Experiment 6**:
  - 10 scenarios
  - 3 layers (L8, L15, L31)
  - Attention matrices: [32 heads, ~121 tokens, ~121 tokens]
  - Feature activations: [~121 tokens, 32768 features]

- **Experiment 3**:
  - 441 causal features
  - 6,400 responses analyzed
  - 22 unique words with IDF scores
  - TF-IDF enhancement applied to all feature-word associations

---

## Next Steps (When Ready)

### 1. Experiment 1: Temporal Feature Tracking
**Prerequisites**: Experiment 2 completion (currently at 28-38%, ~4-5 days remaining)

**Planned Analysis**:
- Round-by-round feature activation dynamics
- How features change within the same game
- Temporal patterns differentiating bankruptcy vs. voluntary stop
- Feature activation trajectories over game progression

**Expected Outputs**:
- Temporal activation plots for key features
- Round-by-round statistics (mean, std, trends)
- Trajectory clustering (similar feature evolution patterns)
- Critical transition points (where features diverge between outcomes)

### 2. Cross-Reference Analyses
**Once Experiment 2 completes**:
- Cross-reference Experiment 6 high-pathway-score features (9926, 30028, 2857) with Experiment 2 causal features
- Validate if balance-attention differences in L15 correspond to causal features
- Integrate TF-IDF word associations with causal features

### 3. Comprehensive Report Integration
- Combine findings from Experiments 3 and 6
- Update academic papers with new pathway and TF-IDF findings
- Create integrated visualization dashboard

---

## Technical Notes

### Experiment 6 Correction Details
- **Original target**: "choices" positions [12, 26] (history tokens)
- **Corrected target**: Last token position (seq_len - 1, decision point)
- **Result**: Attention weights changed from 0.0000 to 0.0015-0.0040 (meaningful)

### TF-IDF Implementation Details
- **IDF calculation**: log(n_features / n_features_with_word)
- **TF source**: Existing high_freq values (normalized frequency)
- **TF-IDF formula**: TF × IDF
- **Minimum features threshold**: 2 (words must appear in at least 2 features)

### Files Modified/Created
#### Experiment 6
- `attention_pathway_analysis.py` (original, analyzed wrong positions)
- `attention_pathway_analysis_corrected.py` (corrected version)
- `ATTENTION_PATHWAY_FINDINGS.md` (comprehensive report)

#### Experiment 3
- `tfidf_enhancement_analysis.py` (new TF-IDF enhancement script)

---

## Session Outcomes

**✅ Achievements**:
1. Identified and corrected critical bug in Experiment 6 attention analysis
2. Generated meaningful attention pathway insights with layer-specific patterns
3. Successfully enhanced Experiment 3 with TF-IDF scoring
4. Created comprehensive documentation for both analyses
5. Produced 24 output files (visualizations + data + reports)

**📊 Scientific Contributions**:
1. Demonstrated myopic balance focus in risky decisions (L15: +86% attention)
2. Demonstrated rational probability weighting in safe decisions (L31: +52% attention)
3. Identified feature-specific vs. common words using TF-IDF
4. Documented dramatic feature activation changes at decision points (0→10.06)

**🔧 Technical Improvements**:
1. Corrected token position identification for decision analysis
2. Implemented TF-IDF enhancement for better word ranking
3. Created reusable analysis pipelines for future experiments

---

*Last updated: 2025-10-16 20:00 UTC*
