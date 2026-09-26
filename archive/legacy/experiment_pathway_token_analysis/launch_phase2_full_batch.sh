#!/bin/bash

# Phase 2: Feature-Feature Correlation Analysis (Full 2787 features)
# Batch execution across GPU 4-7 using Phase 1 patching data

OUTPUT_DIR="/data/llm_addiction/experiment_pathway_token_analysis/results/phase2_correlations"
LOG_DIR="/data/llm_addiction/experiment_pathway_token_analysis/logs"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

echo "=== Phase 2 (FULL 2787 features): Feature Correlation Batch Launch ==="
echo "Distributed across GPU 4, 5, 6, 7"
echo "Total jobs: 2,787 features × 4 conditions = 11,148 analyses"
echo ""

# GPU 4: Features 0-696 (697 features)
echo "Launching GPU 4 (Features 0-696)..."
CUDA_VISIBLE_DEVICES=4 nohup python3 src/phase2_batch_launcher.py \
    --gpu-id 4 \
    --offset 0 \
    --limit 697 \
    > "${LOG_DIR}/phase2_batch_gpu4.log" 2>&1 &

PID_GPU4=$!
echo "  GPU 4 started: PID $PID_GPU4"

# GPU 5: Features 697-1393 (697 features)
echo "Launching GPU 5 (Features 697-1393)..."
CUDA_VISIBLE_DEVICES=5 nohup python3 src/phase2_batch_launcher.py \
    --gpu-id 5 \
    --offset 697 \
    --limit 697 \
    > "${LOG_DIR}/phase2_batch_gpu5.log" 2>&1 &

PID_GPU5=$!
echo "  GPU 5 started: PID $PID_GPU5"

# GPU 6: Features 1394-2090 (697 features)
echo "Launching GPU 6 (Features 1394-2090)..."
CUDA_VISIBLE_DEVICES=6 nohup python3 src/phase2_batch_launcher.py \
    --gpu-id 6 \
    --offset 1394 \
    --limit 697 \
    > "${LOG_DIR}/phase2_batch_gpu6.log" 2>&1 &

PID_GPU6=$!
echo "  GPU 6 started: PID $PID_GPU6"

# GPU 7: Features 2091-2786 (696 features)
echo "Launching GPU 7 (Features 2091-2786)..."
CUDA_VISIBLE_DEVICES=7 nohup python3 src/phase2_batch_launcher.py \
    --gpu-id 7 \
    --offset 2091 \
    --limit 696 \
    > "${LOG_DIR}/phase2_batch_gpu7.log" 2>&1 &

PID_GPU7=$!
echo "  GPU 7 started: PID $PID_GPU7"

echo ""
echo "📋 Summary:"
echo "  Total features: 2,787"
echo "  Features per GPU: ~697"
echo "  Conditions per feature: 4"
echo "  Total jobs per GPU: ~2,788"
echo "  Total jobs overall: 11,148"
echo ""
echo "⏱️  Estimated time per job: ~30 seconds"
echo "  Estimated time per GPU: ~23 hours"
echo ""
echo "Monitor progress:"
echo "  tail -f ${LOG_DIR}/phase2_batch_gpu4.log"
echo "  tail -f ${LOG_DIR}/phase2_batch_gpu5.log"
echo "  tail -f ${LOG_DIR}/phase2_batch_gpu6.log"
echo "  tail -f ${LOG_DIR}/phase2_batch_gpu7.log"
echo ""
echo "Check running processes:"
echo "  ps aux | grep phase2_batch_launcher"
echo ""
echo "PIDs:"
echo "  GPU 4: $PID_GPU4"
echo "  GPU 5: $PID_GPU5"
echo "  GPU 6: $PID_GPU6"
echo "  GPU 7: $PID_GPU7"
echo ""
echo "Results will be saved to:"
echo "  $OUTPUT_DIR/correlations_gpu{4,5,6,7}_*.jsonl"
