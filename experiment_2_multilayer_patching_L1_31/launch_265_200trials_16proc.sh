#!/bin/bash
# Launch Patching Experiment: 265 features × 6 conditions × 200 trials
# 16 processes (4 per GPU) for faster completion
#
# Total: 318,000 trials
# 265 features / 16 processes = ~16-17 features per process
#
# Distribution:
#   GPU 0: proc 0-3   (features 0-16, 17-33, 34-50, 51-67)
#   GPU 1: proc 4-7   (features 68-84, 85-101, 102-118, 119-135)
#   GPU 2: proc 8-11  (features 136-152, 153-169, 170-186, 187-203)
#   GPU 3: proc 12-15 (features 204-220, 221-237, 238-251, 252-264)

set -e

SCRIPT="/home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/patching_265_200trials.py"
OUTPUT_DIR="/data/llm_addiction/experiment_2_multilayer_patching/patching_265_200trials"

mkdir -p $OUTPUT_DIR

echo "========================================"
echo "Patching: 265 × 6 × 200 = 318,000 trials"
echo "16 processes (4 per GPU)"
echo "========================================"
echo ""

# Features per process: 265 / 16 ≈ 17 (first 9 get 17, rest get 16)
# Actually simpler: divide into roughly equal chunks

# GPU 0: 4 processes
echo "Starting GPU 0 processes..."
CUDA_VISIBLE_DEVICES=0 python3 $SCRIPT --gpu-id 0 --offset 0 --limit 17 --n-trials 200 --output-dir $OUTPUT_DIR > $OUTPUT_DIR/log_p0.txt 2>&1 &
CUDA_VISIBLE_DEVICES=0 python3 $SCRIPT --gpu-id 0 --offset 17 --limit 17 --n-trials 200 --output-dir $OUTPUT_DIR > $OUTPUT_DIR/log_p1.txt 2>&1 &
CUDA_VISIBLE_DEVICES=0 python3 $SCRIPT --gpu-id 0 --offset 34 --limit 17 --n-trials 200 --output-dir $OUTPUT_DIR > $OUTPUT_DIR/log_p2.txt 2>&1 &
CUDA_VISIBLE_DEVICES=0 python3 $SCRIPT --gpu-id 0 --offset 51 --limit 16 --n-trials 200 --output-dir $OUTPUT_DIR > $OUTPUT_DIR/log_p3.txt 2>&1 &
echo "GPU 0: 4 processes started"

sleep 5

# GPU 1: 4 processes
echo "Starting GPU 1 processes..."
CUDA_VISIBLE_DEVICES=1 python3 $SCRIPT --gpu-id 1 --offset 67 --limit 17 --n-trials 200 --output-dir $OUTPUT_DIR > $OUTPUT_DIR/log_p4.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 python3 $SCRIPT --gpu-id 1 --offset 84 --limit 17 --n-trials 200 --output-dir $OUTPUT_DIR > $OUTPUT_DIR/log_p5.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 python3 $SCRIPT --gpu-id 1 --offset 101 --limit 17 --n-trials 200 --output-dir $OUTPUT_DIR > $OUTPUT_DIR/log_p6.txt 2>&1 &
CUDA_VISIBLE_DEVICES=1 python3 $SCRIPT --gpu-id 1 --offset 118 --limit 16 --n-trials 200 --output-dir $OUTPUT_DIR > $OUTPUT_DIR/log_p7.txt 2>&1 &
echo "GPU 1: 4 processes started"

sleep 5

# GPU 2: 4 processes
echo "Starting GPU 2 processes..."
CUDA_VISIBLE_DEVICES=2 python3 $SCRIPT --gpu-id 2 --offset 134 --limit 17 --n-trials 200 --output-dir $OUTPUT_DIR > $OUTPUT_DIR/log_p8.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 python3 $SCRIPT --gpu-id 2 --offset 151 --limit 17 --n-trials 200 --output-dir $OUTPUT_DIR > $OUTPUT_DIR/log_p9.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 python3 $SCRIPT --gpu-id 2 --offset 168 --limit 17 --n-trials 200 --output-dir $OUTPUT_DIR > $OUTPUT_DIR/log_p10.txt 2>&1 &
CUDA_VISIBLE_DEVICES=2 python3 $SCRIPT --gpu-id 2 --offset 185 --limit 16 --n-trials 200 --output-dir $OUTPUT_DIR > $OUTPUT_DIR/log_p11.txt 2>&1 &
echo "GPU 2: 4 processes started"

sleep 5

# GPU 3: 4 processes
echo "Starting GPU 3 processes..."
CUDA_VISIBLE_DEVICES=3 python3 $SCRIPT --gpu-id 3 --offset 201 --limit 17 --n-trials 200 --output-dir $OUTPUT_DIR > $OUTPUT_DIR/log_p12.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 python3 $SCRIPT --gpu-id 3 --offset 218 --limit 17 --n-trials 200 --output-dir $OUTPUT_DIR > $OUTPUT_DIR/log_p13.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 python3 $SCRIPT --gpu-id 3 --offset 235 --limit 15 --n-trials 200 --output-dir $OUTPUT_DIR > $OUTPUT_DIR/log_p14.txt 2>&1 &
CUDA_VISIBLE_DEVICES=3 python3 $SCRIPT --gpu-id 3 --offset 250 --limit 15 --n-trials 200 --output-dir $OUTPUT_DIR > $OUTPUT_DIR/log_p15.txt 2>&1 &
echo "GPU 3: 4 processes started"

echo ""
echo "========================================"
echo "All 16 processes started!"
echo "========================================"
echo ""
echo "Monitor:"
echo "  watch -n 30 'wc -l $OUTPUT_DIR/patching_265_200_gpu*.jsonl'"
echo ""
echo "Logs:"
echo "  tail -f $OUTPUT_DIR/log_p0.txt"
