#!/bin/bash
# =============================================================================
# Full Pipeline: Run All Phases for Steering Vector Experiment
# =============================================================================
#
# Usage:
#   ./launch_full_pipeline.sh llama 0    # Run all phases for LLaMA on GPU 0
#   ./launch_full_pipeline.sh gemma 1    # Run all phases for Gemma on GPU 1
#
# This script runs the complete steering vector experiment:
#   1. Extract steering vectors from behavioral data
#   2. Run steering experiment with different strengths
#   3. Analyze steering vectors with SAEs
#
# Total runtime: ~4-10 hours depending on model and sample size.
# =============================================================================

set -e  # Exit on error

# Parse arguments
MODEL=${1:-llama}
GPU=${2:-0}
MAX_SAMPLES=${3:-500}
N_TRIALS=${4:-50}

# Validate model
if [[ "$MODEL" != "llama" && "$MODEL" != "gemma" ]]; then
    echo "Error: Model must be 'llama' or 'gemma'"
    echo "Usage: $0 <model> <gpu> [max_samples] [n_trials]"
    exit 1
fi

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="/data/llm_addiction/steering_vector_experiment"

echo "========================================"
echo "FULL STEERING VECTOR PIPELINE"
echo "========================================"
echo "Model: ${MODEL}"
echo "GPU: ${GPU}"
echo "Max samples: ${MAX_SAMPLES}"
echo "Trials per condition: ${N_TRIALS}"
echo "========================================"
echo ""

# Record start time
START_TIME=$(date +%s)

# =============================================================================
# Phase 1: Extract Steering Vectors
# =============================================================================
echo "========================================"
echo "PHASE 1: Extracting Steering Vectors"
echo "========================================"

"${SCRIPT_DIR}/launch_extraction.sh" "$MODEL" "$GPU" "$MAX_SAMPLES"

# Find the most recent vectors file
VECTORS_FILE=$(ls -t "${OUTPUT_DIR}"/steering_vectors_${MODEL}_*.npz 2>/dev/null | head -1)

if [[ -z "$VECTORS_FILE" ]]; then
    echo "Error: No steering vectors file found after extraction"
    exit 1
fi

echo ""
echo "Vectors file: ${VECTORS_FILE}"
echo ""

# =============================================================================
# Phase 2: Run Steering Experiment
# =============================================================================
echo "========================================"
echo "PHASE 2: Running Steering Experiment"
echo "========================================"

"${SCRIPT_DIR}/launch_steering.sh" "$MODEL" "$GPU" "$VECTORS_FILE" "$N_TRIALS"

# =============================================================================
# Phase 3: SAE Analysis
# =============================================================================
echo "========================================"
echo "PHASE 3: Analyzing with SAE"
echo "========================================"

"${SCRIPT_DIR}/launch_sae_analysis.sh" "$MODEL" "$GPU" "$VECTORS_FILE"

# =============================================================================
# Summary
# =============================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "========================================"
echo "PIPELINE COMPLETE"
echo "========================================"
echo "Model: ${MODEL}"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "Output files in ${OUTPUT_DIR}/:"
echo "  - steering_vectors_${MODEL}_*.npz"
echo "  - steering_results_${MODEL}_*.json"
echo "  - sae_analysis_${MODEL}_*.json"
echo "========================================"
