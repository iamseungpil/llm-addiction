#!/bin/bash
# Launch Patching Experiment: 265 features × 6 conditions × 200 trials
# Total: 318,000 trials distributed across GPU 0-3
#
# Distribution:
#   GPU 0: features 0-66   (67 features) = 80,400 trials
#   GPU 1: features 67-132 (66 features) = 79,200 trials
#   GPU 2: features 133-198 (66 features) = 79,200 trials
#   GPU 3: features 199-264 (66 features) = 79,200 trials

set -e

SCRIPT="/home/ubuntu/llm_addiction/experiment_2_multilayer_patching_L1_31/patching_265_200trials.py"
OUTPUT_DIR="/data/llm_addiction/experiment_2_multilayer_patching/patching_265_200trials"

mkdir -p $OUTPUT_DIR

echo "========================================"
echo "Patching Experiment: 265 × 6 × 200 = 318,000 trials"
echo "Output: $OUTPUT_DIR"
echo "========================================"
echo ""

# GPU 0: features 0-66 (67 features)
echo "Starting GPU 0 (features 0-66)..."
CUDA_VISIBLE_DEVICES=0 python3 $SCRIPT \
    --gpu-id 0 \
    --offset 0 \
    --limit 67 \
    --n-trials 200 \
    --output-dir $OUTPUT_DIR \
    > $OUTPUT_DIR/log_gpu0.txt 2>&1 &
PID0=$!
echo "GPU 0 started (PID: $PID0)"

# GPU 1: features 67-132 (66 features)
echo "Starting GPU 1 (features 67-132)..."
CUDA_VISIBLE_DEVICES=1 python3 $SCRIPT \
    --gpu-id 1 \
    --offset 67 \
    --limit 66 \
    --n-trials 200 \
    --output-dir $OUTPUT_DIR \
    > $OUTPUT_DIR/log_gpu1.txt 2>&1 &
PID1=$!
echo "GPU 1 started (PID: $PID1)"

# GPU 2: features 133-198 (66 features)
echo "Starting GPU 2 (features 133-198)..."
CUDA_VISIBLE_DEVICES=2 python3 $SCRIPT \
    --gpu-id 2 \
    --offset 133 \
    --limit 66 \
    --n-trials 200 \
    --output-dir $OUTPUT_DIR \
    > $OUTPUT_DIR/log_gpu2.txt 2>&1 &
PID2=$!
echo "GPU 2 started (PID: $PID2)"

# GPU 3: features 199-264 (66 features)
echo "Starting GPU 3 (features 199-264)..."
CUDA_VISIBLE_DEVICES=3 python3 $SCRIPT \
    --gpu-id 3 \
    --offset 199 \
    --limit 66 \
    --n-trials 200 \
    --output-dir $OUTPUT_DIR \
    > $OUTPUT_DIR/log_gpu3.txt 2>&1 &
PID3=$!
echo "GPU 3 started (PID: $PID3)"

echo ""
echo "========================================"
echo "All GPUs started!"
echo "PIDs: GPU0=$PID0, GPU1=$PID1, GPU2=$PID2, GPU3=$PID3"
echo "========================================"
echo ""
echo "Monitor progress:"
echo "  tail -f $OUTPUT_DIR/log_gpu0.txt"
echo "  tail -f $OUTPUT_DIR/log_gpu1.txt"
echo "  tail -f $OUTPUT_DIR/log_gpu2.txt"
echo "  tail -f $OUTPUT_DIR/log_gpu3.txt"
echo ""
echo "Check completion:"
echo "  wc -l $OUTPUT_DIR/patching_265_200_gpu*.jsonl"
echo "  Expected: 80,400 + 79,200 + 79,200 + 79,200 = 318,000 total"
echo ""
echo "Monitor all GPUs:"
echo "  watch -n 60 'wc -l $OUTPUT_DIR/patching_265_200_gpu*.jsonl'"
