#!/bin/bash
# Launch Phase 1 (6 conditions × 50 trials) on GPUs 4-7
# Total: 265 features × 6 conditions × 50 trials = 79,500 records
# Distribution: ~66 features per GPU

set -e

SCRIPT_DIR="/home/ubuntu/llm_addiction/experiment_pathway_token_analysis/src"
OUTPUT_DIR="/data/llm_addiction/experiment_pathway_token_analysis/results/phase1_6cond_50trials"

mkdir -p $OUTPUT_DIR

echo "=== Phase 1: 6 Conditions × 50 Trials ==="
echo "Total records: 265 × 6 × 50 = 79,500"
echo "Output: $OUTPUT_DIR"
echo ""

# GPU 4: features 0-65 (66 features)
echo "Starting GPU 4 (features 0-65)..."
CUDA_VISIBLE_DEVICES=4 python3 $SCRIPT_DIR/phase1_6conditions_50trials.py \
    --gpu-id 4 \
    --offset 0 \
    --limit 66 \
    --n-trials 50 \
    --output-dir $OUTPUT_DIR \
    > $OUTPUT_DIR/log_gpu4.txt 2>&1 &
PID4=$!
echo "GPU 4 started (PID: $PID4)"

# GPU 5: features 66-131 (66 features)
echo "Starting GPU 5 (features 66-131)..."
CUDA_VISIBLE_DEVICES=5 python3 $SCRIPT_DIR/phase1_6conditions_50trials.py \
    --gpu-id 5 \
    --offset 66 \
    --limit 66 \
    --n-trials 50 \
    --output-dir $OUTPUT_DIR \
    > $OUTPUT_DIR/log_gpu5.txt 2>&1 &
PID5=$!
echo "GPU 5 started (PID: $PID5)"

# GPU 6: features 132-197 (66 features)
echo "Starting GPU 6 (features 132-197)..."
CUDA_VISIBLE_DEVICES=6 python3 $SCRIPT_DIR/phase1_6conditions_50trials.py \
    --gpu-id 6 \
    --offset 132 \
    --limit 66 \
    --n-trials 50 \
    --output-dir $OUTPUT_DIR \
    > $OUTPUT_DIR/log_gpu6.txt 2>&1 &
PID6=$!
echo "GPU 6 started (PID: $PID6)"

# GPU 7: features 198-264 (67 features)
echo "Starting GPU 7 (features 198-264)..."
CUDA_VISIBLE_DEVICES=7 python3 $SCRIPT_DIR/phase1_6conditions_50trials.py \
    --gpu-id 7 \
    --offset 198 \
    --limit 67 \
    --n-trials 50 \
    --output-dir $OUTPUT_DIR \
    > $OUTPUT_DIR/log_gpu7.txt 2>&1 &
PID7=$!
echo "GPU 7 started (PID: $PID7)"

echo ""
echo "All GPUs started!"
echo "PIDs: GPU4=$PID4, GPU5=$PID5, GPU6=$PID6, GPU7=$PID7"
echo ""
echo "Monitor progress:"
echo "  tail -f $OUTPUT_DIR/log_gpu4.txt"
echo "  tail -f $OUTPUT_DIR/log_gpu5.txt"
echo "  tail -f $OUTPUT_DIR/log_gpu6.txt"
echo "  tail -f $OUTPUT_DIR/log_gpu7.txt"
echo ""
echo "Check completion:"
echo "  wc -l $OUTPUT_DIR/phase1_6cond_gpu*.jsonl"
echo "  Expected: 66×6×50=19800 + 66×6×50=19800 + 66×6×50=19800 + 67×6×50=20100 = 79,500 total"

# Wait for all
wait $PID4 $PID5 $PID6 $PID7

echo ""
echo "=== All GPUs completed ==="
echo "Combining results..."

# Combine all results
cat $OUTPUT_DIR/phase1_6cond_gpu*.jsonl > $OUTPUT_DIR/phase1_6cond_combined.jsonl
TOTAL=$(wc -l < $OUTPUT_DIR/phase1_6cond_combined.jsonl)
echo "Total records: $TOTAL (expected: 79,500)"
