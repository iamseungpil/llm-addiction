#!/bin/bash
# Launch Phase 1 patching with 265 verified causal features
# 265 features / 4 GPUs = 67 features per GPU (rounded up)

CAUSAL_FILE="/home/ubuntu/llm_addiction/experiment_pathway_token_analysis/causal_features_265_FINAL.json"
MEANS_FILE="/home/ubuntu/llm_addiction/experiment_pathway_token_analysis/feature_means_265_FINAL.json"
OUTPUT_DIR="/data/llm_addiction/experiment_pathway_token_analysis/results/phase1_265features"
N_TRIALS=30
FEATURES_PER_GPU=67

echo "=== Phase 1: Patching with 265 Verified Causal Features ==="
echo "Causal features: $CAUSAL_FILE"
echo "Feature means: $MEANS_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Trials per condition: $N_TRIALS"
echo "Total trials: 265 features × 4 conditions × $N_TRIALS trials = $((265 * 4 * N_TRIALS))"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# GPU 4: features 0-66 (67 features)
echo "Starting GPU 4: features 0-66..."
CUDA_VISIBLE_DEVICES=4 nohup python /home/ubuntu/llm_addiction/experiment_pathway_token_analysis/src/phase1_patching_multifeature_checkpoint.py \
    --gpu-id 4 \
    --n-trials $N_TRIALS \
    --top-n 265 \
    --offset 0 \
    --limit $FEATURES_PER_GPU \
    --causal-features "$CAUSAL_FILE" \
    --feature-means "$MEANS_FILE" \
    --output-dir "$OUTPUT_DIR" \
    > "$OUTPUT_DIR/gpu4.log" 2>&1 &
echo "  PID: $!"

sleep 2

# GPU 5: features 67-133 (67 features)
echo "Starting GPU 5: features 67-133..."
CUDA_VISIBLE_DEVICES=5 nohup python /home/ubuntu/llm_addiction/experiment_pathway_token_analysis/src/phase1_patching_multifeature_checkpoint.py \
    --gpu-id 5 \
    --n-trials $N_TRIALS \
    --top-n 265 \
    --offset $FEATURES_PER_GPU \
    --limit $FEATURES_PER_GPU \
    --causal-features "$CAUSAL_FILE" \
    --feature-means "$MEANS_FILE" \
    --output-dir "$OUTPUT_DIR" \
    > "$OUTPUT_DIR/gpu5.log" 2>&1 &
echo "  PID: $!"

sleep 2

# GPU 6: features 134-200 (67 features)
echo "Starting GPU 6: features 134-200..."
CUDA_VISIBLE_DEVICES=6 nohup python /home/ubuntu/llm_addiction/experiment_pathway_token_analysis/src/phase1_patching_multifeature_checkpoint.py \
    --gpu-id 6 \
    --n-trials $N_TRIALS \
    --top-n 265 \
    --offset $((FEATURES_PER_GPU * 2)) \
    --limit $FEATURES_PER_GPU \
    --causal-features "$CAUSAL_FILE" \
    --feature-means "$MEANS_FILE" \
    --output-dir "$OUTPUT_DIR" \
    > "$OUTPUT_DIR/gpu6.log" 2>&1 &
echo "  PID: $!"

sleep 2

# GPU 7: features 201-264 (64 features - remaining)
echo "Starting GPU 7: features 201-264..."
CUDA_VISIBLE_DEVICES=7 nohup python /home/ubuntu/llm_addiction/experiment_pathway_token_analysis/src/phase1_patching_multifeature_checkpoint.py \
    --gpu-id 7 \
    --n-trials $N_TRIALS \
    --top-n 265 \
    --offset $((FEATURES_PER_GPU * 3)) \
    --limit 100 \
    --causal-features "$CAUSAL_FILE" \
    --feature-means "$MEANS_FILE" \
    --output-dir "$OUTPUT_DIR" \
    > "$OUTPUT_DIR/gpu7.log" 2>&1 &
echo "  PID: $!"

echo ""
echo "=== All 4 GPUs started ==="
echo "Monitor progress:"
echo "  tail -f $OUTPUT_DIR/gpu4.log"
echo "  tail -f $OUTPUT_DIR/gpu5.log"
echo "  tail -f $OUTPUT_DIR/gpu6.log"
echo "  tail -f $OUTPUT_DIR/gpu7.log"
echo ""
echo "Check running processes:"
echo "  ps aux | grep phase1_patching"
