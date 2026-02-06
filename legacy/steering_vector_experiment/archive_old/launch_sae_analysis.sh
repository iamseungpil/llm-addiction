#!/bin/bash
# =============================================================================
# Phase 3: SAE Analysis Launch Script
# =============================================================================
#
# Usage:
#   ./launch_sae_analysis.sh llama 0 steering_vectors_llama_20251216.npz
#   ./launch_sae_analysis.sh gemma 1 steering_vectors_gemma_20251216.npz
#
# This script analyzes steering vectors using SAEs to identify
# which interpretable features contribute most to behavioral differences.
# Typical runtime: 30-60 minutes.
# =============================================================================

set -e  # Exit on error

# Parse arguments
MODEL=${1:-llama}
GPU=${2:-0}
VECTORS=${3:-}
TOP_K=${4:-50}

# Validate arguments
if [[ "$MODEL" != "llama" && "$MODEL" != "gemma" ]]; then
    echo "Error: Model must be 'llama' or 'gemma'"
    echo "Usage: $0 <model> <gpu> <vectors_file> [top_k]"
    exit 1
fi

if [[ -z "$VECTORS" ]]; then
    echo "Error: Must specify vectors file"
    echo "Usage: $0 <model> <gpu> <vectors_file> [top_k]"
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
echo "Phase 3: SAE Analysis"
echo "========================================"
echo "Model: ${MODEL}"
echo "GPU: ${GPU}"
echo "Vectors: ${VECTORS}"
echo "Top K features: ${TOP_K}"
echo "========================================"

# Run SAE analysis
echo ""
echo "Starting SAE analysis..."
echo ""

conda run -n llama_sae_env python "${SRC_DIR}/analyze_steering_with_sae.py" \
    --model "$MODEL" \
    --gpu "$GPU" \
    --vectors "$VECTORS" \
    --config "$CONFIG_FILE" \
    --top-k "$TOP_K"

echo ""
echo "========================================"
echo "SAE analysis complete!"
echo "Check results in ${OUTPUT_DIR}/"
echo "========================================"
