#!/bin/bash

# Phase 1: Patching + Multi-Feature Extraction (Full 2787 features)
# WITH CHECKPOINT SUPPORT - Can resume from interruptions
# Distributed across GPU 4, 5, 6, 7

OUTPUT_DIR="/data/llm_addiction/experiment_pathway_token_analysis/results/phase1_patching_full"
LOG_DIR="/data/llm_addiction/experiment_pathway_token_analysis/logs"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

echo "=== Phase 1 (FULL 2787 features WITH CHECKPOINT): Patching Launch ==="
echo "Distributed across GPU 4, 5, 6, 7"
echo "Can resume from interruptions - safe to restart!"
echo ""

# Total features: 2787
# Distribution:
# GPU 4: features 0-696 (697 features)
# GPU 5: features 697-1393 (697 features)
# GPU 6: features 1394-2090 (697 features)
# GPU 7: features 2091-2786 (696 features)

# GPU 4: Features 0-696
CUDA_VISIBLE_DEVICES=4 nohup python3 src/phase1_patching_multifeature_checkpoint.py \
    --gpu-id 4 \
    --n-trials 30 \
    --top-n 2787 \
    --offset 0 \
    --limit 697 \
    > "${LOG_DIR}/phase1_full_gpu4.log" 2>&1 &

PID_GPU4=$!
echo "Launched GPU 4 (Features 0-696): PID $PID_GPU4"

# GPU 5: Features 697-1393
CUDA_VISIBLE_DEVICES=5 nohup python3 src/phase1_patching_multifeature_checkpoint.py \
    --gpu-id 5 \
    --n-trials 30 \
    --top-n 2787 \
    --offset 697 \
    --limit 697 \
    > "${LOG_DIR}/phase1_full_gpu5.log" 2>&1 &

PID_GPU5=$!
echo "Launched GPU 5 (Features 697-1393): PID $PID_GPU5"

# GPU 6: Features 1394-2090
CUDA_VISIBLE_DEVICES=6 nohup python3 src/phase1_patching_multifeature_checkpoint.py \
    --gpu-id 6 \
    --n-trials 30 \
    --top-n 2787 \
    --offset 1394 \
    --limit 697 \
    > "${LOG_DIR}/phase1_full_gpu6.log" 2>&1 &

PID_GPU6=$!
echo "Launched GPU 6 (Features 1394-2090): PID $PID_GPU6"

# GPU 7: Features 2091-2786
CUDA_VISIBLE_DEVICES=7 nohup python3 src/phase1_patching_multifeature_checkpoint.py \
    --gpu-id 7 \
    --n-trials 30 \
    --top-n 2787 \
    --offset 2091 \
    --limit 696 \
    > "${LOG_DIR}/phase1_full_gpu7.log" 2>&1 &

PID_GPU7=$!
echo "Launched GPU 7 (Features 2091-2786): PID $PID_GPU7"

echo ""
echo "📋 Summary:"
echo "  Total features: 2,787"
echo "  Features per GPU: ~697"
echo "  Trials per condition: 30"
echo "  Conditions per feature: 4"
echo "  Total trials per GPU: ~83,640"
echo "  Total trials overall: 334,560"
echo ""
echo "⏱️  Estimated time per GPU: ~87 hours (3.6 days)"
echo ""
echo "✅ CHECKPOINT ENABLED:"
echo "   - Progress saved in real-time to JSONL"
echo "   - Safe to kill and restart"
echo "   - Will resume from last completed trial"
echo ""
echo "Monitor progress:"
echo "  tail -f ${LOG_DIR}/phase1_full_gpu4.log"
echo "  tail -f ${LOG_DIR}/phase1_full_gpu5.log"
echo "  tail -f ${LOG_DIR}/phase1_full_gpu6.log"
echo "  tail -f ${LOG_DIR}/phase1_full_gpu7.log"
echo ""
echo "Check running processes:"
echo "  ps aux | grep phase1_patching_multifeature_checkpoint"
echo ""
echo "PIDs:"
echo "  GPU 4: $PID_GPU4"
echo "  GPU 5: $PID_GPU5"
echo "  GPU 6: $PID_GPU6"
echo "  GPU 7: $PID_GPU7"
