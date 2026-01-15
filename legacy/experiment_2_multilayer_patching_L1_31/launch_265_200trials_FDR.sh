#!/bin/bash
# Launch Patching Experiment: 265 features × 6 conditions × 200 trials
# For FDR validation with correct parsing
# 16 processes (4 per GPU 0-3) for faster completion
#
# Total: 318,000 trials
# 265 features / 16 processes = ~16-17 features per process
#
# Output: /data/llm_addiction/patching_265_FDR_20251208/
# (Separated from previous experiments)

set -e

SCRIPT="/home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/patching_265_200trials.py"
OUTPUT_DIR="/data/llm_addiction/patching_265_FDR_20251208"

mkdir -p $OUTPUT_DIR

echo "========================================"
echo "Patching: 265 × 6 × 200 = 318,000 trials"
echo "16 processes (4 per GPU 0-3)"
echo "Output: $OUTPUT_DIR"
echo "========================================"
echo ""

# Features per process: 265 / 16 ≈ 17 (distribute evenly)
# GPU 0: features 0-66 (4 procs × ~17 each)
# GPU 1: features 67-133
# GPU 2: features 134-200
# GPU 3: features 201-264

# GPU 0: 4 processes
echo "Starting GPU 0 processes..."
CUDA_VISIBLE_DEVICES=0 python3 $SCRIPT --gpu-id 0 --offset 0 --limit 17 --n-trials 200 --output-dir $OUTPUT_DIR > $OUTPUT_DIR/log_p0.txt 2>&1 &
CUDA_VISIBLE_DEVICES=0 python3 $SCRIPT --gpu-id 0 --offset 17 --limit 17 --n-trials 200 --output-dir $OUTPUT_DIR > $OUTPUT_DIR/log_p1.txt 2>&1 &
CUDA_VISIBLE_DEVICES=0 python3 $SCRIPT --gpu-id 0 --offset 34 --limit 17 --n-trials 200 --output-dir $OUTPUT_DIR > $OUTPUT_DIR/log_p2.txt 2>&1 &
CUDA_VISIBLE_DEVICES=0 python3 $SCRIPT --gpu-id 0 --offset 51 --limit 16 --n-trials 200 --output-dir $OUTPUT_DIR > $OUTPUT_DIR/log_p3.txt 2>&1 &
echo "GPU 0: 4 processes started (features 0-66)"

sleep 5

# GPU 1: 4 processes
echo "Starting GPU 1 processes..."
CUDA_VISIBLE_DEVICES=1 python3 $SCRIPT --gpu-id 1 --offset 67 --limit 17 --n-trials 200 --output-dir $OUTPUT_DIR > $OUTPUT_DIR/log_p4.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 python3 $SCRIPT --gpu-id 1 --offset 84 --limit 17 --n-trials 200 --output-dir $OUTPUT_DIR > $OUTPUT_DIR/log_p5.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 python3 $SCRIPT --gpu-id 1 --offset 101 --limit 17 --n-trials 200 --output-dir $OUTPUT_DIR > $OUTPUT_DIR/log_p6.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 python3 $SCRIPT --gpu-id 1 --offset 118 --limit 16 --n-trials 200 --output-dir $OUTPUT_DIR > $OUTPUT_DIR/log_p7.txt 2>&1 &
echo "GPU 1: 4 processes started (features 67-133)"

sleep 5

# GPU 2: 4 processes
echo "Starting GPU 2 processes..."
CUDA_VISIBLE_DEVICES=2 python3 $SCRIPT --gpu-id 2 --offset 134 --limit 17 --n-trials 200 --output-dir $OUTPUT_DIR > $OUTPUT_DIR/log_p8.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 python3 $SCRIPT --gpu-id 2 --offset 151 --limit 17 --n-trials 200 --output-dir $OUTPUT_DIR > $OUTPUT_DIR/log_p9.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 python3 $SCRIPT --gpu-id 2 --offset 168 --limit 17 --n-trials 200 --output-dir $OUTPUT_DIR > $OUTPUT_DIR/log_p10.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 python3 $SCRIPT --gpu-id 2 --offset 185 --limit 16 --n-trials 200 --output-dir $OUTPUT_DIR > $OUTPUT_DIR/log_p11.txt 2>&1 &
echo "GPU 2: 4 processes started (features 134-200)"

sleep 5

# GPU 3: 4 processes
echo "Starting GPU 3 processes..."
CUDA_VISIBLE_DEVICES=3 python3 $SCRIPT --gpu-id 3 --offset 201 --limit 16 --n-trials 200 --output-dir $OUTPUT_DIR > $OUTPUT_DIR/log_p12.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 python3 $SCRIPT --gpu-id 3 --offset 217 --limit 16 --n-trials 200 --output-dir $OUTPUT_DIR > $OUTPUT_DIR/log_p13.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 python3 $SCRIPT --gpu-id 3 --offset 233 --limit 16 --n-trials 200 --output-dir $OUTPUT_DIR > $OUTPUT_DIR/log_p14.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 python3 $SCRIPT --gpu-id 3 --offset 249 --limit 16 --n-trials 200 --output-dir $OUTPUT_DIR > $OUTPUT_DIR/log_p15.txt 2>&1 &
echo "GPU 3: 4 processes started (features 201-264)"

echo ""
echo "========================================"
echo "All 16 processes started!"
echo "========================================"
echo ""
echo "Monitor:"
echo "  watch -n 30 'wc -l $OUTPUT_DIR/patching_265_200_gpu*.jsonl 2>/dev/null || echo \"Waiting for output files...\"'"
echo ""
echo "Logs:"
echo "  tail -f $OUTPUT_DIR/log_p0.txt"
echo ""
echo "Check GPU usage:"
echo "  nvidia-smi"
