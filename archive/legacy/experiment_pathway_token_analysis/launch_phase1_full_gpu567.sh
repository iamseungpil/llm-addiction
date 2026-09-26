#!/bin/bash

# Phase 1 (Full 2787 features): Patching + Multi-Feature Extraction
# GPU allocation: 5, 6, 7

OUTPUT_DIR="/data/llm_addiction/experiment_pathway_token_analysis/results/phase1_patching_full"
LOG_DIR="/data/llm_addiction/experiment_pathway_token_analysis/logs"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

echo "=== Phase 1 (FULL 2787 features): Patching Launch ==="
echo "Using GPUs 5, 6, 7"
echo ""

# GPU 5: Features 0-928 (929 features)
CUDA_VISIBLE_DEVICES=5 nohup python3 src/phase1_patching_multifeature.py \
    --gpu-id 5 \
    --n-trials 30 \
    --top-n 2787 \
    --offset 0 \
    --limit 929 \
    > "${LOG_DIR}/phase1_full_gpu5.log" 2>&1 &
PID5=$!
echo "GPU 5 started: Features 0-928 (PID: $PID5)"

# GPU 6: Features 929-1857 (929 features)
CUDA_VISIBLE_DEVICES=6 nohup python3 src/phase1_patching_multifeature.py \
    --gpu-id 6 \
    --n-trials 30 \
    --top-n 2787 \
    --offset 929 \
    --limit 929 \
    > "${LOG_DIR}/phase1_full_gpu6.log" 2>&1 &
PID6=$!
echo "GPU 6 started: Features 929-1857 (PID: $PID6)"

# GPU 7: Features 1858-2786 (929 features)
CUDA_VISIBLE_DEVICES=7 nohup python3 src/phase1_patching_multifeature.py \
    --gpu-id 7 \
    --n-trials 30 \
    --top-n 2787 \
    --offset 1858 \
    --limit 929 \
    > "${LOG_DIR}/phase1_full_gpu7.log" 2>&1 &
PID7=$!
echo "GPU 7 started: Features 1858-2786 (PID: $PID7)"

echo ""
echo "All Phase 1 jobs launched!"
echo "Logs:"
echo "  GPU 5: ${LOG_DIR}/phase1_full_gpu5.log"
echo "  GPU 6: ${LOG_DIR}/phase1_full_gpu6.log"
echo "  GPU 7: ${LOG_DIR}/phase1_full_gpu7.log"
echo ""
echo "Results will be in: $OUTPUT_DIR"
echo ""
echo "Monitor progress:"
echo "  tail -f ${LOG_DIR}/phase1_full_gpu5.log"
echo "  tail -f ${LOG_DIR}/phase1_full_gpu6.log"
echo "  tail -f ${LOG_DIR}/phase1_full_gpu7.log"
