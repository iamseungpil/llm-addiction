#!/bin/bash

# Phase 1: Feature Extraction
# Usage: bash scripts/run_phase1.sh [model] [gpu_id]
#   model: gemma or llama
#   gpu_id: GPU device ID (default: 0)

set -e

# Parse arguments
MODEL=${1:-gemma}
GPU_ID=${2:-0}

echo "======================================"
echo "Phase 1: Feature Extraction"
echo "======================================"
echo "Model: $MODEL"
echo "GPU: $GPU_ID"
echo "======================================"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( dirname "$SCRIPT_DIR" )"

# Activate conda environment
echo "Activating conda environment: llama_sae_env"
eval "$(conda shell.bash hook)"
conda activate llama_sae_env

# Change to project directory
cd "$PROJECT_DIR"

# Run Phase 1
echo ""
echo "Starting Phase 1..."
echo ""

python src/phase1_feature_extraction.py \
    --model "$MODEL" \
    --gpu "$GPU_ID" \
    --config configs/experiment_config.yaml

echo ""
echo "======================================"
echo "Phase 1 complete!"
echo "======================================"
