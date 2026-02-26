#!/bin/bash

# Phase 2: Correlation Analysis
# Usage: bash scripts/run_phase2.sh [model]
#   model: gemma or llama

set -e

# Parse arguments
MODEL=${1:-gemma}

echo "======================================"
echo "Phase 2: Correlation Analysis"
echo "======================================"
echo "Model: $MODEL"
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

# Run Phase 2
echo ""
echo "Starting Phase 2..."
echo ""

python src/phase2_correlation_analysis.py \
    --model "$MODEL" \
    --config configs/experiment_config.yaml

echo ""
echo "======================================"
echo "Phase 2 complete!"
echo "======================================"
