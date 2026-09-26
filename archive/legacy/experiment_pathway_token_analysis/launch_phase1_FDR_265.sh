#!/bin/bash
#
# Launch Phase 1 FDR-265 Features Experiment
#
# This script launches Phase 1 after the main patching experiment completes.
# It distributes FDR-confirmed features across multiple GPUs.
#
# Prerequisites:
#   1. Main patching experiment (265 x 6 x 200) must be COMPLETE
#   2. Results must exist in: /data/llm_addiction/patching_265_FDR_20251208/
#
# Usage:
#   ./launch_phase1_FDR_265.sh           # Use default GPUs (4,5,6,7)
#   ./launch_phase1_FDR_265.sh 0,1,2,3   # Specify GPUs
#
# Output:
#   /data/llm_addiction/experiment_pathway_token_analysis/results/phase1_FDR_265/
#
# Author: LLM Addiction Research Project
# Date: 2025-12-08
#

set -e

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE1_SCRIPT="${SCRIPT_DIR}/src/phase1_FDR_265features.py"

# Default GPUs (can be overridden by command line argument)
DEFAULT_GPUS="4,5,6,7"
GPUS="${1:-$DEFAULT_GPUS}"

# Experiment parameters
N_TRIALS=50
FDR_ALPHA=0.05

# Paths
PATCHING_DIR="/data/llm_addiction/patching_265_FDR_20251208"
OUTPUT_DIR="/data/llm_addiction/experiment_pathway_token_analysis/results/phase1_FDR_265"
CAUSAL_FEATURES="/home/ubuntu/llm_addiction/experiment_pathway_token_analysis/causal_features_265_FINAL.json"
FEATURE_MEANS="/home/ubuntu/llm_addiction/experiment_pathway_token_analysis/feature_means_265_FINAL.json"

# Conda environment
CONDA_ENV="llama_sae_env"

# =============================================================================
# Validation
# =============================================================================

echo "=================================================================="
echo "Phase 1: FDR-Confirmed Feature Patching with Full Extraction"
echo "=================================================================="
echo ""

# Check if patching results exist
if [ ! -d "$PATCHING_DIR" ]; then
    echo "ERROR: Patching results directory not found: $PATCHING_DIR"
    echo "Please wait for the main patching experiment to complete."
    exit 1
fi

# Count patching result files
PATCHING_FILES=$(ls -1 ${PATCHING_DIR}/patching_265_200_gpu*.jsonl 2>/dev/null | wc -l)
if [ "$PATCHING_FILES" -eq 0 ]; then
    echo "ERROR: No patching result files found in $PATCHING_DIR"
    exit 1
fi
echo "Found $PATCHING_FILES patching result files"

# Check total records (quick count)
TOTAL_RECORDS=$(cat ${PATCHING_DIR}/patching_265_200_gpu*.jsonl 2>/dev/null | wc -l)
EXPECTED_MIN=$((265 * 6 * 100))  # At least half complete
if [ "$TOTAL_RECORDS" -lt "$EXPECTED_MIN" ]; then
    echo "WARNING: Only $TOTAL_RECORDS records found (expected ~318,000)"
    echo "Patching experiment may not be complete. Continue anyway? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        exit 1
    fi
else
    echo "Total patching records: $TOTAL_RECORDS"
fi

# Check script exists
if [ ! -f "$PHASE1_SCRIPT" ]; then
    echo "ERROR: Phase 1 script not found: $PHASE1_SCRIPT"
    exit 1
fi

# Check causal features file
if [ ! -f "$CAUSAL_FEATURES" ]; then
    echo "ERROR: Causal features file not found: $CAUSAL_FEATURES"
    exit 1
fi

# Check feature means file
if [ ! -f "$FEATURE_MEANS" ]; then
    echo "ERROR: Feature means file not found: $FEATURE_MEANS"
    exit 1
fi

echo ""
echo "Configuration:"
echo "  GPUs: $GPUS"
echo "  Trials per condition: $N_TRIALS"
echo "  FDR alpha: $FDR_ALPHA"
echo "  Output: $OUTPUT_DIR"
echo ""

# =============================================================================
# Calculate GPU Distribution
# =============================================================================

# Parse GPUs into array
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "Using $NUM_GPUS GPUs: ${GPU_ARRAY[*]}"

# We don't know exact FDR feature count until computed
# Estimate ~130 features (50% of 265) for distribution
# Actual distribution will be handled by checkpoint/resume
ESTIMATED_FDR_FEATURES=130
FEATURES_PER_GPU=$(( (ESTIMATED_FDR_FEATURES + NUM_GPUS - 1) / NUM_GPUS ))

echo "Estimated FDR features: ~$ESTIMATED_FDR_FEATURES"
echo "Features per GPU: ~$FEATURES_PER_GPU"
echo ""

# =============================================================================
# Create Output Directory
# =============================================================================

mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# =============================================================================
# Launch Processes
# =============================================================================

echo "Launching Phase 1 processes..."
echo ""

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

for i in "${!GPU_ARRAY[@]}"; do
    GPU_ID="${GPU_ARRAY[$i]}"
    OFFSET=$((i * FEATURES_PER_GPU))

    # Use limit only for non-last GPU to ensure all features are covered
    if [ $i -eq $((NUM_GPUS - 1)) ]; then
        LIMIT_ARG=""
    else
        LIMIT_ARG="--limit $FEATURES_PER_GPU"
    fi

    SESSION_NAME="phase1_fdr_gpu${GPU_ID}"
    LOG_FILE="${OUTPUT_DIR}/log_gpu${GPU_ID}.txt"

    echo "Starting GPU $GPU_ID (offset=$OFFSET, limit=${FEATURES_PER_GPU:-unlimited})..."

    # Create tmux session with the experiment
    tmux new-session -d -s "$SESSION_NAME" \
        "source ~/miniconda3/etc/profile.d/conda.sh && \
         conda activate $CONDA_ENV && \
         CUDA_VISIBLE_DEVICES=$GPU_ID python $PHASE1_SCRIPT \
           --gpu-id $GPU_ID \
           --n-trials $N_TRIALS \
           --offset $OFFSET \
           $LIMIT_ARG \
           --output-dir $OUTPUT_DIR \
           --patching-dir $PATCHING_DIR \
           --causal-features $CAUSAL_FEATURES \
           --feature-means $FEATURE_MEANS \
           --fdr-alpha $FDR_ALPHA \
           2>&1 | tee $LOG_FILE"

    echo "  Session: $SESSION_NAME"
    echo "  Log: $LOG_FILE"
done

echo ""
echo "=================================================================="
echo "All processes launched!"
echo "=================================================================="
echo ""
echo "Monitor progress:"
echo "  tmux ls                           # List sessions"
echo "  tmux attach -t phase1_fdr_gpu4    # Attach to GPU 4 session"
echo "  tail -f ${OUTPUT_DIR}/log_gpu*.txt  # Follow all logs"
echo ""
echo "Check output:"
echo "  ls -la ${OUTPUT_DIR}/"
echo "  wc -l ${OUTPUT_DIR}/phase1_FDR_265_gpu*.jsonl"
echo ""
echo "Expected output per trial:"
echo "  - response: Generated text"
echo "  - all_features: ALL 265 features' activations"
echo "  - generated_tokens: Token list for Phase 4"
echo ""
