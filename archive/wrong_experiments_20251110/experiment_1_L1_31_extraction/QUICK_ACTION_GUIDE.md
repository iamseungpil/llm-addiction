# Quick Action Guide: SAE Extraction Completion

**Current Time**: 20:45 KST
**Experiment Status**: 83% complete (5320/6400), ETA: **20:52 KST** (7 minutes)

---

## When Experiment Completes (20:52)

### Step 1: Verify Completion (1 min)
```bash
tmux attach -t sae_exp1
# Check for "Extraction complete" message

# Verify final checkpoint exists
ls -lh /data/llm_addiction/experiment_1_L1_31_SAE_extraction/*batch3*.json
```

### Step 2: Apply Global FDR Correction (10 min)
```bash
cd /home/ubuntu/llm_addiction/experiment_1_L1_31_extraction

python apply_global_fdr_correction.py \
    --checkpoints \
        /data/llm_addiction/experiment_1_L1_31_SAE_extraction/L1_31_SAE_checkpoint_batch1_*.json \
        /data/llm_addiction/experiment_1_L1_31_SAE_extraction/L1_31_SAE_checkpoint_batch2_*.json \
        /data/llm_addiction/experiment_1_L1_31_SAE_extraction/L1_31_SAE_checkpoint_batch3_*.json \
    --output /data/llm_addiction/experiment_1_L1_31_SAE_extraction/L1_31_features_FINAL_GLOBAL_FDR_20251110.json \
    --alpha 0.05 \
    --method fdr_bh
```

**Expected Output**:
- Original features (layer-wise FDR): ~10,000-15,000
- Selected features (global FDR): ~2,000-4,000
- 70-80% reduction (normal for proper FDR correction)

### Step 3: Generate Feature Means Lookup (15 min)
```bash
# Create the lookup script first (if not exists)
python /home/ubuntu/llm_addiction/experiment_pathway_token_analysis/create_feature_means_lookup.py \
    --features /data/llm_addiction/experiment_1_L1_31_SAE_extraction/L1_31_features_FINAL_GLOBAL_FDR_20251110.json \
    --experiments \
        /data/llm_addiction/results/exp1_multiround_intermediate_20250819_140040.json \
        /data/llm_addiction/results/exp1_missing_complete_20250820_090040.json \
    --output /data/llm_addiction/experiment_1_L1_31_SAE_extraction/L1_31_feature_means_lookup.json
```

### Step 4: Update Downstream Scripts (5 min)
```bash
# Update hardcoded paths in:

# 1. Experiment 2
sed -i 's|L1_31_features_FINAL_20250930_220003.json|L1_31_features_FINAL_GLOBAL_FDR_20251110.json|g' \
    /home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/experiment_2_L1_31_top300.py

# 2. Pathway analysis (if needed)
# Manually check and update these scripts:
#   - experiment_pathway_token_analysis/src/phase1_patching_multifeature.py
#   - experiment_pathway_token_analysis/src/phase2_*.py
#   - experiment_pathway_token_analysis/src/phase3_*.py
```

---

## Expected Timeline

| Time | Task | Status |
|------|------|--------|
| 20:52 | Experiment completes | ⏳ Waiting |
| 20:53 | Verify checkpoints | ⏳ |
| 21:02 | Global FDR applied | ⏳ |
| 21:17 | Feature means ready | ⏳ |
| 21:22 | Scripts updated | ⏳ |
| **21:30** | **READY FOR EXP2** | 🎯 |

---

## Key Files

### Input (From Current Experiment)
- `L1_31_SAE_checkpoint_batch1_20251110_190149.json` (1.2MB) ✅
- `L1_31_SAE_checkpoint_batch2_20251110_195806.json` (3.7MB) ✅
- `L1_31_SAE_checkpoint_batch3_20251110_XXXXXX.json` (TBD) ⏳

### Output (After Correction)
- `L1_31_features_FINAL_GLOBAL_FDR_20251110.json` - Corrected features
- `L1_31_feature_means_lookup.json` - Bankrupt/safe means

---

## Troubleshooting

### If Batch 3 checkpoint is missing:
```bash
# Check tmux for errors
tmux attach -t sae_exp1

# Look for "Saved checkpoint" message
# File should be in /data/llm_addiction/experiment_1_L1_31_SAE_extraction/
```

### If Global FDR script fails:
```bash
# Check Python packages
pip install scipy statsmodels numpy

# Run with verbose output
python apply_global_fdr_correction.py --help
```

### If Feature count is unexpectedly low (<500):
```bash
# Try more lenient alpha
python apply_global_fdr_correction.py \
    --alpha 0.10 \  # Increase from 0.05
    ... (same other args)
```

---

## Next Phase: Experiment 2 Integration

Once global FDR correction is complete:

1. **Verify compatibility** with L25-31 reference data
2. **Select top 300 features** for Experiment 2 patching
3. **Run activation patching** with new L1-31 features
4. **Update pathway analysis** with extended layer coverage

---

**For detailed analysis, see**: `ULTRATHINK_SAE_EXTRACTION_ANALYSIS.md`
