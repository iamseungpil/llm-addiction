#!/bin/bash

# Phase 1: Patching + Multi-Feature Extraction (REPARSED 2510 features)
# WITH CHECKPOINT SUPPORT + TOKEN ID SAVING
# Distributed across GPU 4, 5, 6, 7

OUTPUT_DIR="/data/llm_addiction/experiment_pathway_token_analysis/results/phase1_patching_REPARSED"
LOG_DIR="/data/llm_addiction/experiment_pathway_token_analysis/logs"

# Input files (REPARSED versions - CORRECTED 2025-11-25)
# feature_means_lookup_REPARSED_FULL.json: 2,510 features (100% coverage)
# Converted from L1_31_features_CONVERTED_20251111.json
CAUSAL_FEATURES="/home/ubuntu/llm_addiction/experiment_pathway_token_analysis/causal_features_list_REPARSED.json"
FEATURE_MEANS="/home/ubuntu/llm_addiction/experiment_pathway_token_analysis/feature_means_lookup_REPARSED_FULL.json"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

echo "=== Phase 1 (REPARSED 2510 features WITH CHECKPOINT + TOKEN IDs): Patching Launch ==="
echo "Distributed across GPU 4, 5, 6, 7"
echo "NEW: Saves generated_token_ids and generated_tokens for Phase 4 analysis"
echo "Can resume from interruptions - safe to restart!"
echo ""

# Total features: 2510
# Distribution (balanced):
# GPU 4: features 0-627 (628 features)
# GPU 5: features 628-1254 (627 features)
# GPU 6: features 1255-1881 (627 features)
# GPU 7: features 1882-2509 (628 features)

# GPU 4: Features 0-627
CUDA_VISIBLE_DEVICES=4 nohup python3 src/phase1_patching_multifeature_checkpoint.py \
    --gpu-id 4 \
    --n-trials 30 \
    --top-n 2510 \
    --offset 0 \
    --limit 628 \
    --output-dir "$OUTPUT_DIR" \
    --causal-features "$CAUSAL_FEATURES" \
    --feature-means "$FEATURE_MEANS" \
    > "${LOG_DIR}/phase1_REPARSED_gpu4.log" 2>&1 &

PID_GPU4=$!
echo "Launched GPU 4 (Features 0-627): PID $PID_GPU4"

# GPU 5: Features 628-1254
CUDA_VISIBLE_DEVICES=5 nohup python3 src/phase1_patching_multifeature_checkpoint.py \
    --gpu-id 5 \
    --n-trials 30 \
    --top-n 2510 \
    --offset 628 \
    --limit 627 \
    --output-dir "$OUTPUT_DIR" \
    --causal-features "$CAUSAL_FEATURES" \
    --feature-means "$FEATURE_MEANS" \
    > "${LOG_DIR}/phase1_REPARSED_gpu5.log" 2>&1 &

PID_GPU5=$!
echo "Launched GPU 5 (Features 628-1254): PID $PID_GPU5"

# GPU 6: Features 1255-1881
CUDA_VISIBLE_DEVICES=6 nohup python3 src/phase1_patching_multifeature_checkpoint.py \
    --gpu-id 6 \
    --n-trials 30 \
    --top-n 2510 \
    --offset 1255 \
    --limit 627 \
    --output-dir "$OUTPUT_DIR" \
    --causal-features "$CAUSAL_FEATURES" \
    --feature-means "$FEATURE_MEANS" \
    > "${LOG_DIR}/phase1_REPARSED_gpu6.log" 2>&1 &

PID_GPU6=$!
echo "Launched GPU 6 (Features 1255-1881): PID $PID_GPU6"

# GPU 7: Features 1882-2509
CUDA_VISIBLE_DEVICES=7 nohup python3 src/phase1_patching_multifeature_checkpoint.py \
    --gpu-id 7 \
    --n-trials 30 \
    --top-n 2510 \
    --offset 1882 \
    --limit 628 \
    --output-dir "$OUTPUT_DIR" \
    --causal-features "$CAUSAL_FEATURES" \
    --feature-means "$FEATURE_MEANS" \
    > "${LOG_DIR}/phase1_REPARSED_gpu7.log" 2>&1 &

PID_GPU7=$!
echo "Launched GPU 7 (Features 1882-2509): PID $PID_GPU7"

echo ""
echo "📋 Summary:"
echo "  Total features: 2,510 (REPARSED)"
echo "  Features per GPU: 627-628"
echo "  Trials per condition: 30"
echo "  Conditions per feature: 4"
echo "  Total trials per GPU: ~75,240"
echo "  Total trials overall: 301,200"
echo ""
echo "⏱️  Estimated time per GPU: 7-10 hours"
echo ""
echo "✅ CHECKPOINT ENABLED:"
echo "   - Progress saved in real-time to JSONL"
echo "   - Safe to kill and restart"
echo "   - Will resume from last completed trial"
echo ""
echo "🆕 TOKEN ID SAVING:"
echo "   - generated_token_ids: actual BPE token IDs"
echo "   - generated_tokens: decoded token strings"
echo "   - For improved Phase 4 word-feature analysis"
echo ""
echo "Monitor progress:"
echo "  tail -f ${LOG_DIR}/phase1_REPARSED_gpu4.log"
echo "  tail -f ${LOG_DIR}/phase1_REPARSED_gpu5.log"
echo "  tail -f ${LOG_DIR}/phase1_REPARSED_gpu6.log"
echo "  tail -f ${LOG_DIR}/phase1_REPARSED_gpu7.log"
echo ""
echo "Check running processes:"
echo "  ps aux | grep phase1_patching_multifeature_checkpoint"
echo ""
echo "PIDs:"
echo "  GPU 4: $PID_GPU4"
echo "  GPU 5: $PID_GPU5"
echo "  GPU 6: $PID_GPU6"
echo "  GPU 7: $PID_GPU7"
echo ""
echo "✅ Phase 1 launched successfully!"
