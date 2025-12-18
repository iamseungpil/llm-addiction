#!/bin/bash
# =============================================================================
# Phase 1: Steering Vector Extraction Launch Script
# =============================================================================
#
# Usage:
#   ./launch_extraction.sh llama 0    # Extract LLaMA vectors on GPU 0
#   ./launch_extraction.sh gemma 1    # Extract Gemma vectors on GPU 1
#
# This script extracts steering vectors from behavioral experiment data.
# Typical runtime: 1-3 hours depending on sample size.
# =============================================================================

set -e  # Exit on error

# Parse arguments
MODEL=${1:-llama}
GPU=${2:-0}
MAX_SAMPLES=${3:-500}

# Validate model
if [[ "$MODEL" != "llama" && "$MODEL" != "gemma" ]]; then
    echo "Error: Model must be 'llama' or 'gemma'"
    echo "Usage: $0 <model> <gpu> [max_samples]"
    exit 1
fi

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
CONFIG_FILE="${SCRIPT_DIR}/configs/experiment_config.yaml"

# Activate conda environment
echo "========================================"
echo "Phase 1: Steering Vector Extraction"
echo "========================================"
echo "Model: ${MODEL}"
echo "GPU: ${GPU}"
echo "Max samples: ${MAX_SAMPLES}"
echo "Config: ${CONFIG_FILE}"
echo "========================================"

# Check if conda environment exists
if ! conda env list | grep -q "llama_sae_env"; then
    echo "Error: conda environment 'llama_sae_env' not found"
    echo "Please create it first or use a different environment"
    exit 1
fi

# Run extraction
echo ""
echo "Starting extraction..."
echo ""

conda run -n llama_sae_env python "${SRC_DIR}/extract_steering_vectors.py" \
    --model "$MODEL" \
    --gpu "$GPU" \
    --config "$CONFIG_FILE" \
    --max-samples "$MAX_SAMPLES"

echo ""
echo "========================================"
echo "Extraction complete!"
echo "Check output in /data/llm_addiction/steering_vector_experiment/"
echo "========================================"
