#!/bin/bash
# =============================================================================
# Phase 2: Steering Experiment Launch Script
# =============================================================================
#
# Usage:
#   ./launch_steering.sh llama 0 steering_vectors_llama_20251216.npz
#   ./launch_steering.sh gemma 1 steering_vectors_gemma_20251216.npz
#
# This script applies steering vectors during generation to test
# their causal effect on betting behavior.
# Typical runtime: 2-6 hours depending on number of trials.
# =============================================================================

set -e  # Exit on error

# Parse arguments
MODEL=${1:-llama}
GPU=${2:-0}
VECTORS=${3:-}
N_TRIALS=${4:-50}

# Validate arguments
if [[ "$MODEL" != "llama" && "$MODEL" != "gemma" ]]; then
    echo "Error: Model must be 'llama' or 'gemma'"
    echo "Usage: $0 <model> <gpu> <vectors_file> [n_trials]"
    exit 1
fi

if [[ -z "$VECTORS" ]]; then
    echo "Error: Must specify vectors file"
    echo "Usage: $0 <model> <gpu> <vectors_file> [n_trials]"
    exit 1
fi

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_FILE="${SCRIPT_DIR}/configs/experiment_config.yaml"
OUTPUT_DIR="/data/llm_addiction/steering_vector_experiment"

# Resolve vectors path
if [[ ! "$VECTORS" = /* ]]; then
    VECTORS="${OUTPUT_DIR}/${VECTORS}"
fi

# Check if vectors file exists
if [[ ! -f "$VECTORS" ]]; then
    echo "Error: Vectors file not found: $VECTORS"
    echo ""
    echo "Available vector files:"
    ls -la "${OUTPUT_DIR}"/steering_vectors_*.npz 2>/dev/null || echo "No vector files found"
    exit 1
fi

echo "========================================"
echo "Phase 2: Steering Experiment"
echo "========================================"
echo "Model: ${MODEL}"
echo "GPU: ${GPU}"
echo "Vectors: ${VECTORS}"
echo "Trials per condition: ${N_TRIALS}"
echo "========================================"

# Run steering experiment
echo ""
echo "Starting steering experiment..."
echo ""

conda run -n llama_sae_env python "${SRC_DIR}/run_steering_experiment.py" \
    --model "$MODEL" \
    --gpu "$GPU" \
    --vectors "$VECTORS" \
    --config "$CONFIG_FILE" \
    --n-trials "$N_TRIALS"

echo ""
echo "========================================"
echo "Steering experiment complete!"
echo "Check results in ${OUTPUT_DIR}/"
echo "========================================"
